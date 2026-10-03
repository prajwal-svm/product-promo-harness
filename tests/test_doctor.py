import json
import os
import subprocess

import httpx
import pytest
import respx
from typer.testing import CliRunner

from promo import doctor
from promo.cli import app
from promo.providers import gemini
from promo.run import new_run

runner = CliRunner()
SUPERTONIC_HINT = "pip install supertonic or uv tool install supertonic"
# What `hyperframes doctor --json` printed on this Mac: `.ok` is false only because two optional extras are missing.
HF_PAYLOAD = {
    "ok": False,
    "checks": [
        {"name": "Version", "ok": True, "detail": "0.8.98 (latest)"},
        {"name": "Chrome", "ok": True, "detail": "cache: chrome-headless-shell"},
        {"name": "TTS (Kokoro)", "ok": False, "detail": "Not installed (optional — local voice fallback)",
         "hint": "pip install kokoro-onnx soundfile"},
        {"name": "BGM (MusicGen)", "ok": False, "detail": "Not installed (optional — local music fallback)"},
    ],
    "_meta": {"version": "0.8.98", "updateAvailable": False},
}


def _hf(**failing):
    """The checks `hyperframes doctor --json` 0.8.98 prints on this Mac, each ok unless named in `failing` (its detail)."""
    names = ("Version", "Node.js", "CPU", "Memory", "Disk", "Frames cache", "Archive extractor", "Settings lock",
             "Environment", "whisper-cpp", "TTS (Kokoro)", "BGM (MusicGen)", "onnxruntime-node", "@google/genai",
             "FFmpeg", "FFprobe", "Chrome", "Docker", "Docker running")
    checks = [{"name": n, "ok": n not in failing, "detail": failing.get(n, "fine")} for n in names]
    return {"ok": not failing, "checks": checks, "_meta": {"version": "0.8.98", "updateAvailable": False}}


def _fake_key_applies(node) -> bool:
    """Live tests must read the real keys from .env, so they never get the fake ones."""
    return node.get_closest_marker("live") is None


@pytest.fixture(autouse=True)
def keys(request, monkeypatch):
    if _fake_key_applies(request.node):
        for name in ("GEMINI_API_KEY", "ELEVENLABS_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.setenv(name, f"fake-{name.lower()}")


def test_fake_key_skips_live_tests():
    class Node:
        def __init__(self, marker):
            self.marker = marker

        def get_closest_marker(self, name):
            return self.marker if name == "live" else None

    assert _fake_key_applies(Node(None)) is True
    assert _fake_key_applies(Node(object())) is False


def test_keys_outside_a_run_are_warnings():
    checks = {c.name: c for c in doctor.check_keys({"GEMINI_API_KEY": "x"})}
    assert checks["GEMINI_API_KEY"].ok
    assert not checks["ELEVENLABS_API_KEY"].ok and checks["ELEVENLABS_API_KEY"].level == "warning"
    assert not checks["OPENROUTER_API_KEY"].ok and checks["OPENROUTER_API_KEY"].level == "warning"
    assert all(c.level == "warning" for c in doctor.check_keys({}))


def test_keys_inside_a_run_fail_only_when_the_run_needs_them():
    needs = doctor.KeyNeeds(gemini=["the voice (Gemini TTS)"], elevenlabs=[])
    checks = {c.name: c for c in doctor.check_keys({}, needs)}
    assert checks["GEMINI_API_KEY"].level == "error" and "the voice (Gemini TTS)" in checks["GEMINI_API_KEY"].detail
    assert checks["ELEVENLABS_API_KEY"].level == "warning"
    assert "this run does not need it" in checks["ELEVENLABS_API_KEY"].detail
    keyless = doctor.check_keys({}, doctor.KeyNeeds())
    assert [c.level for c in keyless] == ["warning", "warning", "warning"]


def test_keys_show_presence_never_the_value():
    env = {"GEMINI_API_KEY": "sk-secret-value", "ELEVENLABS_API_KEY": "   "}
    checks = doctor.check_keys(env)
    assert "sk-secret-value" not in " ".join(c.model_dump_json() for c in checks)
    assert [c.ok for c in checks] == [True, False, False]       # a blank key is a missing key


def test_node_version():
    assert doctor.check_node("v26.7.0").ok
    assert not doctor.check_node("v20.11.1").ok
    assert doctor.check_node("v22.0.0").ok and not doctor.check_node("not a version").ok


def test_ffmpeg_filters():
    text = "\n".join(f" .. {f}  A->A  x" for f in doctor.FFMPEG_FILTERS if f != "alimiter")
    checks = doctor.check_ffmpeg(text)
    missing = [c for c in checks if not c.ok]
    assert [c.name for c in missing] == ["ffmpeg filter alimiter"]


def test_ffmpeg_without_text_filters_is_not_a_failure():
    """drawtext, subtitles and ass are banned for the harness, so their absence is expected; their presence is fine too."""
    base = "\n".join(f" TSC {f}  A->A  x" for f in doctor.FFMPEG_FILTERS)
    without = doctor.check_ffmpeg(base)
    assert all(c.ok for c in without)
    assert [c.detail for c in without if c.name == "ffmpeg text filters"] == [
        "no drawtext/subtitles/ass, as expected: the harness never uses them"]
    withal = doctor.check_ffmpeg(base + "\n T.. drawtext  V->V  Draw text.\n T.. subtitles  V->V  Render subtitles.")
    assert all(c.ok for c in withal)
    assert "drawtext, subtitles present but never used" in [c.detail for c in withal if c.name == "ffmpeg text filters"][0]
    assert not any(f in c.name for c in without for f in doctor.TEXT_FILTERS if c.name != "ffmpeg text filters")


def test_disk_levels():
    assert doctor.check_disk(40 * 10**9).ok
    warn = doctor.check_disk(12 * 10**9)
    assert warn.ok is False and warn.level == "warning"
    assert doctor.check_disk(5 * 10**9).level == "error"


def test_disk_thresholds_are_20_and_8_gb():
    assert doctor.check_disk(20 * 10**9).ok
    assert doctor.check_disk(int(19.9 * 10**9)).level == "warning"
    assert doctor.check_disk(8 * 10**9).level == "warning"
    assert doctor.check_disk(int(7.9 * 10**9)).level == "error"


def test_python_module_check():
    assert doctor.check_module("json").ok
    bad = doctor.check_module("promo_no_such_module")
    assert not bad.ok and bad.name == "python promo_no_such_module" and "uv sync" in bad.detail


def test_supertonic_is_optional_unless_the_run_selects_it():
    assert doctor.check_supertonic("supertonic 1.1.2", required=False).ok
    optional = doctor.check_supertonic(None, required=False)
    assert not optional.ok and optional.level == "warning" and SUPERTONIC_HINT in optional.detail
    needed = doctor.check_supertonic(None, required=True)
    assert not needed.ok and needed.level == "error" and SUPERTONIC_HINT in needed.detail
    assert "shot list" in needed.detail


def test_supertonic_checks_without_the_binary(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    [warn] = doctor._supertonic_checks(required=False)
    assert warn.level == "warning" and not warn.ok and SUPERTONIC_HINT in warn.detail
    [err] = doctor._supertonic_checks(required=True)
    assert err.level == "error"


def test_supertonic_checks_print_the_version_and_look_for_the_model(fake_supertonic, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    version, model = doctor._supertonic_checks(required=False)
    assert version.ok and version.detail == "supertonic 9.9.9"
    assert not model.ok and model.level == "info" and "downloads the model" in model.detail     # nothing is a failure
    (tmp_path / ".cache" / "supertonic2").mkdir(parents=True)
    _, model = doctor._supertonic_checks(required=False)
    assert model.ok and model.detail == str(tmp_path / ".cache" / "supertonic2")


def test_supertonic_that_will_not_run_is_reported(tmp_path, monkeypatch):
    bad = tmp_path / "bin" / "supertonic"
    bad.parent.mkdir()
    bad.write_text("#!/bin/sh\necho boom >&2\nexit 3\n")
    bad.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bad.parent}{os.pathsep}{os.environ['PATH']}")
    [check] = doctor._supertonic_checks(required=True)
    assert not check.ok and check.level == "error" and "supertonic version" in check.detail


def _run_with_voice(tmp_path, monkeypatch, **voice):
    run = new_run(tmp_path, "d", "saas-kinetic", "16:9", None, "en")
    data = json.loads(run.shotlist.read_text())
    data["voice"] = voice
    run.shotlist.write_text(json.dumps(data))
    monkeypatch.chdir(run.root)
    return run


def test_run_voice_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert doctor._run_voice_provider() is None                          # not in a run folder
    run = _run_with_voice(tmp_path, monkeypatch, provider="supertonic", name="F2")
    assert doctor._run_voice_provider() == "supertonic"
    run.shotlist.write_text(json.dumps({**json.loads(run.shotlist.read_text()), "voice": {"name": "Kore"}}))
    assert doctor._run_voice_provider() == "gemini"                      # the pack's default provider
    run.shotlist.write_text("{ not json")
    assert doctor._run_voice_provider() is None                          # unreadable: another command reports it


def test_supertonic_is_required_when_the_run_selects_it(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(doctor, "_hf_doctor", lambda: {"ok": True})
    monkeypatch.setattr(doctor, "_supertonic_checks", lambda required: seen.append(required) or [])
    _run_with_voice(tmp_path, monkeypatch, provider="supertonic")
    doctor.run_checks()
    monkeypatch.chdir(tmp_path.parent)
    doctor.run_checks()
    assert seen == [True, False]


def _voice_led(run):
    """The scaffold's music-led shot list made voice-led: no bars, shots in seconds, events on shot edges."""
    data = json.loads(run.shotlist.read_text())
    data["timing_mode"] = "voice"
    for ch in data["chapters"]:
        ch.pop("bars", None)
        for shot in ch["shots"]:
            shot["seconds"] = 3.0
            shot["events"] = [{**e, "at": {"shot": shot["id"], "edge": "start", "offset": 0.2 + i}}
                              for i, e in enumerate(shot["events"])]
    run.shotlist.write_text(json.dumps(data))
    return run


def test_a_supertonic_film_with_synth_sounds_and_no_music_needs_no_key(tmp_path, monkeypatch):
    _voice_led(_run_with_voice(tmp_path, monkeypatch, provider="supertonic", name="F2"))
    assert doctor._run_key_needs() == doctor.KeyNeeds()


def test_a_gemini_music_film_needs_both_keys(tmp_path, monkeypatch):
    run = _run_with_voice(tmp_path, monkeypatch, provider="gemini", name="Kore")
    needs = doctor._run_key_needs()
    assert needs.gemini == ["the voice (Gemini TTS)"]
    assert needs.elevenlabs == ["the music bed (`promo music`, ElevenLabs)"]
    (run.music_dir / "bed.wav").write_bytes(b"the user's own track")     # not made by `promo music`: no key needed
    assert doctor._run_key_needs().elevenlabs == []
    (run.music_dir / "bed.src.mp3").write_bytes(b"what promo music downloaded")
    assert doctor._run_key_needs().elevenlabs == ["the music bed (`promo music`, ElevenLabs)"]


def test_a_lyria_music_bed_needs_the_gemini_key_not_elevenlabs(tmp_path, monkeypatch):
    """The music provider decides the key. Read with getattr: the shot list's music.provider field is new (Task 11c)."""
    from types import SimpleNamespace

    from promo.packs import load_pack
    from promo.shotlist import load_shotlist

    run = _run_with_voice(tmp_path, monkeypatch, provider="supertonic")
    sl = load_shotlist(run.shotlist)
    pack = load_pack("saas-kinetic")

    def with_music(provider):
        music = SimpleNamespace(**{**sl.music.model_dump(), "provider": provider})
        return SimpleNamespace(timing_mode=sl.timing_mode, music=music, all_shots=sl.all_shots)

    lyria = doctor.run_key_needs(run, with_music("lyria"), pack, "supertonic")
    assert lyria.gemini == ["the music bed (`promo music`, Lyria)"] and lyria.elevenlabs == []
    eleven = doctor.run_key_needs(run, with_music("elevenlabs"), pack, "supertonic")
    assert eleven.elevenlabs == ["the music bed (`promo music`, ElevenLabs)"] and eleven.gemini == []
    assert doctor.run_key_needs(run, with_music(None), pack, "supertonic") == eleven     # no provider: ElevenLabs
    both = doctor.run_key_needs(run, with_music("lyria"), pack, "gemini")
    assert both.gemini == ["the voice (Gemini TTS)", "the music bed (`promo music`, Lyria)"]


def test_generated_pack_sounds_need_the_elevenlabs_key(tmp_path, monkeypatch):
    from promo.packs import load_pack
    from promo.shotlist import load_shotlist

    run = _voice_led(_run_with_voice(tmp_path, monkeypatch, provider="supertonic"))
    pack = load_pack("saas-kinetic")
    sl = load_shotlist(run.shotlist)
    kinds = sorted({e.sfx or e.kind for s in sl.all_shots() for e in s.events})
    gen = pack.model_copy(update={"sfx": {**pack.sfx, kinds[0]: {"gen": "a soft paper swish", "duration": 0.4}}})
    assert doctor.run_key_needs(run, sl, gen, "supertonic").elevenlabs == [f"generated sound effects ({kinds[0]})"]
    assert doctor.run_key_needs(run, sl, pack, "supertonic").elevenlabs == []


def _tools_ok(monkeypatch):
    """Every tool check passes, so only the keys decide the exit code."""
    monkeypatch.setattr(doctor, "load_env", lambda: None)              # never the real harness .env
    for name in ("GEMINI_API_KEY", "ELEVENLABS_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(doctor, "_node_checks", lambda: [])
    monkeypatch.setattr(doctor, "_ffmpeg_checks", lambda: [])
    monkeypatch.setattr(doctor, "check_module", lambda name: doctor.Check(name=f"python {name}", ok=True))
    monkeypatch.setattr(doctor, "_supertonic_checks", lambda required: [])
    monkeypatch.setattr(doctor, "_asr_checks", lambda: [])
    monkeypatch.setattr(doctor, "_hf_doctor", lambda: {"ok": True})
    monkeypatch.setattr(doctor, "check_disk", lambda free: doctor.Check(name="disk", ok=True))


def test_cli_doctor_passes_a_keyless_supertonic_film_and_outside_a_run(tmp_path, monkeypatch):
    _tools_ok(monkeypatch)
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(app, ["doctor"])
    assert res.exit_code == 0, res.output                                # outside a run: warnings only
    assert "WARN  GEMINI_API_KEY" in res.output
    _voice_led(_run_with_voice(tmp_path, monkeypatch, provider="supertonic", name="F2"))
    res = runner.invoke(app, ["doctor"])
    assert res.exit_code == 0, res.output
    assert "0 errors" in res.output and "this run does not need it" in res.output


def test_cli_doctor_fails_a_gemini_film_without_its_key(tmp_path, monkeypatch):
    _tools_ok(monkeypatch)
    _run_with_voice(tmp_path, monkeypatch, provider="gemini", name="Kore")
    res = runner.invoke(app, ["doctor"])
    assert res.exit_code == 1
    assert "FAIL  GEMINI_API_KEY" in res.output and "the voice (Gemini TTS)" in res.output
    assert "FAIL  ELEVENLABS_API_KEY" in res.output and "the music bed" in res.output


def test_asr_model_needs_weights_in_the_cache(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    missing = doctor.check_asr_model("whisper", "org/model", snap)         # config only: a partial download
    assert not missing.ok and missing.level == "warning" and "org/model" in missing.detail
    (snap / "weights.safetensors").write_bytes(b"x")
    assert doctor.check_asr_model("whisper", "org/model", snap).ok
    assert doctor.check_asr_model("parakeet", "org/model", None).name == "asr parakeet model"
    assert not doctor.check_asr_model("parakeet", "org/model", None).ok


def test_hf_snapshot_of_an_unknown_model_is_none():
    assert doctor._hf_snapshot("promo-tests/no-such-model") is None


def test_hyperframes_gates_on_the_payload():
    assert doctor.check_hyperframes({"ok": True, "_meta": {"version": "0.8.98"}}).ok
    ok = doctor.check_hyperframes(HF_PAYLOAD)                  # `.ok` is false, but only for optional extras
    assert ok.ok and "TTS (Kokoro)" in ok.detail and "0.8.98" in ok.detail
    broken = json.loads(json.dumps(HF_PAYLOAD))
    broken["checks"][1] = {"name": "Chrome", "ok": False, "detail": "not found"}
    bad = doctor.check_hyperframes(broken)
    assert not bad.ok and bad.level == "error" and "Chrome: not found" in bad.detail and "Kokoro" not in bad.detail
    assert not doctor.check_hyperframes({"ok": False}).ok
    assert not doctor.check_hyperframes({"error": "needs Node.js (npx not found)"}).ok
    pinned = doctor.check_hyperframes({"ok": True, "_meta": {"version": "0.9.0"}})
    assert not pinned.ok and "pins 0.8.98" in pinned.detail


def test_hyperframes_version_check_does_not_fail_a_pinned_install():
    payload = {"ok": False, "_meta": {"version": "0.8.98"},
               "checks": [{"name": "Version", "ok": False, "detail": "0.8.98 (latest: 0.9.1)"}]}
    assert doctor.check_hyperframes(payload).ok


def test_hyperframes_on_a_mac_without_docker_is_ok():
    payload = _hf(**{"Docker": "Docker not found", "Docker running": "Docker is not running",
                     "TTS (Kokoro)": "Not installed (optional — local voice fallback)",
                     "BGM (MusicGen)": "Not installed (optional — local music fallback)"})
    assert payload["ok"] is False                       # what makes the payload's own verdict useless here
    check = doctor.check_hyperframes(payload)
    assert check.ok and check.level == "error"
    assert "not needed here: TTS (Kokoro), BGM (MusicGen), Docker, Docker running" in check.detail
    assert doctor.check_hyperframes(_hf()).ok


@pytest.mark.parametrize("name", doctor.HF_BLOCKING)
def test_hyperframes_fails_on_every_check_that_blocks_a_render(name):
    check = doctor.check_hyperframes(_hf(**{name: "broken", "Docker": "Docker not found"}))
    assert not check.ok and check.level == "error" and f"{name}: broken" in check.detail and "Docker" not in check.detail


def test_hyperframes_ignores_what_a_blocking_checks_detail_says():
    """"optional" in the detail of Chrome does not make it optional: only the check's name decides."""
    check = doctor.check_hyperframes(_hf(Chrome="not found (optional?)"))
    assert not check.ok and check.level == "error" and "Chrome: not found (optional?)" in check.detail


def test_hyperframes_warns_about_a_check_it_does_not_know():
    payload = _hf()
    payload["checks"].append({"name": "GPU sandbox", "ok": False, "detail": "no sandbox"})
    payload["ok"] = False
    check = doctor.check_hyperframes(payload)
    assert not check.ok and check.level == "warning" and "GPU sandbox: no sandbox" in check.detail
    payload["checks"].append({"name": "Chrome", "ok": False, "detail": "gone"})
    assert doctor.check_hyperframes(payload).level == "error"            # a blocking failure still wins


def test_hyperframes_not_ok_with_no_failed_check_is_a_failure():
    check = doctor.check_hyperframes({"ok": False, "checks": [{"name": "Chrome", "ok": True}]})
    assert not check.ok and check.level == "error"


def _proc(stdout="", returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")


def test_hf_doctor_reads_json_after_npx_noise(monkeypatch):
    monkeypatch.setattr(doctor, "run", lambda cmd, **kw: _proc("npm notice something\n" + json.dumps(HF_PAYLOAD)))
    assert doctor._hf_doctor()["_meta"]["version"] == "0.8.98"
    monkeypatch.setattr(doctor, "run", lambda cmd, **kw: _proc("nothing useful"))
    assert "error" in doctor._hf_doctor()
    monkeypatch.setattr(doctor, "run", lambda cmd, **kw: _proc("{ truncated"))
    assert "error" in doctor._hf_doctor()


def test_hf_doctor_without_node_is_an_error_not_a_crash(monkeypatch):
    def no_npx(cmd, **kw):
        raise FileNotFoundError(2, "No such file or directory", "npx")

    monkeypatch.setattr(doctor, "run", no_npx)
    check = doctor.check_hyperframes(doctor._hf_doctor())
    assert not check.ok and "Node.js" in check.detail


def test_run_checks_offline(monkeypatch):
    monkeypatch.setattr(doctor, "_hf_doctor", lambda: {"ok": True})
    names = [c.name for c in doctor.run_checks(live=False)]
    assert "hyperframes" in names and "disk" in names and "node" in names
    assert not any(n.endswith(" ping") for n in names)           # no network without --live


def test_run_checks_covers_every_tool_the_task_names(monkeypatch):
    monkeypatch.setattr(doctor, "_hf_doctor", lambda: {"ok": True})
    names = {c.name for c in doctor.run_checks()}
    for wanted in ("GEMINI_API_KEY", "ELEVENLABS_API_KEY", "OPENROUTER_API_KEY", "node", "npx", "ffmpeg filter scdet",
                   "ffprobe", "python librosa", "python num2words", "supertonic", "asr parakeet model",
                   "asr whisper model", "hyperframes", "disk"):
        assert wanted in names


def test_a_tool_that_will_not_run_is_a_failed_check_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert doctor._ffmpeg_checks()[0].name == "ffmpeg" and not doctor._ffmpeg_checks()[0].ok
    node, npx = doctor._node_checks()
    assert not node.ok and not npx.ok


@respx.mock
def test_live_ping_calls_each_key_validation_endpoint_with_its_own_header():
    gem = respx.get(f"https://generativelanguage.googleapis.com/v1beta/models/{gemini.DEFAULT_MODEL}").mock(
        return_value=httpx.Response(200, json={"name": "models/x"}))
    eleven = respx.get("https://api.elevenlabs.io/v1/user").mock(return_value=httpx.Response(200, json={}))
    router = respx.get("https://openrouter.ai/api/v1/key").mock(return_value=httpx.Response(200, json={}))
    checks = doctor._ping(True)
    assert [(c.name, c.ok) for c in checks] == [("gemini ping", True), ("elevenlabs ping", True), ("openrouter ping", True)]
    assert gem.calls.last.request.headers["x-goog-api-key"] == "fake-gemini_api_key"
    assert eleven.calls.last.request.headers["xi-api-key"] == "fake-elevenlabs_api_key"
    assert router.calls.last.request.headers["authorization"] == "Bearer fake-openrouter_api_key"
    assert gem.call_count == eleven.call_count == router.call_count == 1


@respx.mock
def test_live_ping_skips_keys_that_are_not_set_and_is_off_without_live(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    eleven = respx.get("https://api.elevenlabs.io/v1/user").mock(return_value=httpx.Response(200, json={}))
    assert [c.name for c in doctor._ping(True)] == ["elevenlabs ping"]
    assert doctor._ping(False) == [] and eleven.call_count == 1


@respx.mock
def test_live_ping_reports_a_rejected_key_in_one_line_and_never_echoes_it(monkeypatch):
    respx.get(f"https://generativelanguage.googleapis.com/v1beta/models/{gemini.DEFAULT_MODEL}").mock(
        return_value=httpx.Response(400, json={"error": {"message": "API key not valid: fake-gemini_api_key"}}))
    respx.get("https://api.elevenlabs.io/v1/user").mock(
        return_value=httpx.Response(401, json={"detail": {"message": "Invalid API key"}}))
    respx.get("https://openrouter.ai/api/v1/key").mock(
        return_value=httpx.Response(401, json={"error": {"message": "No auth credentials found"}}))
    gem, eleven, router = doctor._ping(True)
    assert not gem.ok and gem.level == "error" and "fake-gemini_api_key" not in gem.detail and "API key not valid" in gem.detail
    assert not eleven.ok and eleven.level == "error" and eleven.detail == "Invalid API key"
    assert not router.ok and router.level == "warning"         # OpenRouter is phase 4: a warning, as when it is missing


@respx.mock
def test_live_ping_with_a_non_ascii_key_says_so_without_showing_it(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "clé-ключ-secret")
    route = respx.get(f"https://generativelanguage.googleapis.com/v1beta/models/{gemini.DEFAULT_MODEL}").mock(
        return_value=httpx.Response(200, json={}))
    check = doctor._ping_one("gemini")
    assert not check.ok and check.level == "error" and check.detail == "key contains non-ASCII characters"
    assert "secret" not in check.model_dump_json() and route.call_count == 0


SECRET = "key-with{}SECRET-tail"


@respx.mock
@pytest.mark.parametrize("bad", ["\n", "\r", "\r\n", "\t", " ", "\x1b", "\x7f", "\u2028", "\xa0"])
def test_a_key_with_whitespace_or_a_control_character_is_refused_before_any_request(monkeypatch, bad):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET.format(bad))
    route = respx.get(f"https://generativelanguage.googleapis.com/v1beta/models/{gemini.DEFAULT_MODEL}").mock(
        return_value=httpx.Response(200, json={}))
    check = doctor._ping_one("gemini")
    assert not check.ok and check.level == "error" and check.detail == "key contains whitespace or control characters"
    assert route.call_count == 0
    dumped = check.model_dump_json()
    assert "SECRET" not in dumped and "key-with" not in dumped


@respx.mock
@pytest.mark.parametrize("bad", ["\n", "\r"])
def test_cli_never_shows_a_key_with_a_newline_or_a_carriage_return(monkeypatch, bad):
    monkeypatch.setattr(doctor, "_hf_doctor", lambda: {"ok": True})
    monkeypatch.setenv("GEMINI_API_KEY", SECRET.format(bad))
    respx.get("https://api.elevenlabs.io/v1/user").mock(return_value=httpx.Response(200, json={}))
    respx.get("https://openrouter.ai/api/v1/key").mock(return_value=httpx.Response(200, json={}))
    res = runner.invoke(app, ["doctor", "--live"])
    assert "gemini ping" in res.output and "key contains whitespace or control characters" in res.output
    assert "SECRET" not in res.output and "key-with" not in res.output and "tail" not in res.output
    assert "elevenlabs ping" in res.output and "key accepted" in res.output        # the other keys are still tested


@pytest.mark.parametrize("secret", ["abc'def\\ghi\"jkl", "plain-secret-value", "wide-ключ", "q\"uote"])
def test_redact_removes_every_form_an_exception_message_can_hold_a_key_in(secret):
    forms = [secret, repr(secret), ascii(secret), repr(secret.encode()), json.dumps(secret), str(secret.encode()),
             f"Illegal header value {secret.encode()!r}", secret.encode("unicode_escape").decode()]
    for text in forms:
        redacted = doctor._redact(f"network error: {text} (x)", secret)
        assert secret not in redacted and repr(secret)[1:-1] not in redacted and "***" in redacted
    assert doctor._redact("nothing here", secret) == "nothing here" and doctor._redact("text", "") == "text"


@pytest.mark.parametrize("bad", ["\n", "\r"])
def test_redact_covers_the_message_httpx_really_gives_for_a_header_it_cannot_send(bad):
    """The leak this guards against: h11 refuses the header and httpx's message holds the escaped bytes of the key."""
    import http.server
    import threading

    from promo.providers.http import ProviderError, request

    class Quiet(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Quiet)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    key = SECRET.format(bad)
    try:
        with pytest.raises(ProviderError) as raised:
            request("gemini", "GET", f"http://127.0.0.1:{server.server_port}/", headers={"x-goog-api-key": key},
                    retries=0, timeout=5)
    finally:
        server.shutdown()
    assert "SECRET" in raised.value.detail                       # the message does carry it
    assert "SECRET" not in doctor._redact(raised.value.detail, key)


def test_a_providers_error_that_carries_the_key_is_redacted_in_the_check(monkeypatch):
    from promo.providers.http import ProviderError

    key = "abc'def\\ghi\"jkl-SECRET"
    monkeypatch.setenv("GEMINI_API_KEY", key)

    def leak(*args, **kwargs):
        raise ProviderError("gemini", 0, f"network error: Illegal header value {key.encode()!r}")

    monkeypatch.setattr("promo.providers.http.request", leak)
    check = doctor._ping_one("gemini")
    assert not check.ok and "SECRET" not in check.detail and "***" in check.detail


@pytest.mark.parametrize("error", [RuntimeError("boom with key-SECRET"), httpx.LocalProtocolError("Illegal header value b'key-SECRET'"),
                                   UnicodeEncodeError("ascii", "key-SECRET", 0, 1, "ordinal not in range"), KeyError("key-SECRET"),
                                   ValueError("key-SECRET")])
def test_any_other_exception_is_reported_by_its_type_only(monkeypatch, error):
    monkeypatch.setenv("GEMINI_API_KEY", "key-SECRET")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr("promo.providers.http.request", fail)
    check = doctor._ping_one("gemini")
    assert not check.ok and check.detail == f"the request failed ({type(error).__name__})" and "SECRET" not in check.detail


@respx.mock
def test_live_ping_network_failure_is_a_failed_check(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    respx.get("https://api.elevenlabs.io/v1/user").mock(side_effect=httpx.ConnectError("no route"))
    [check] = doctor._ping(True)
    assert not check.ok and "network error" in check.detail


@respx.mock
def test_cli_live_flag_pings_and_plain_doctor_does_not(monkeypatch):
    monkeypatch.setattr(doctor, "_hf_doctor", lambda: {"ok": True})
    route = respx.get("https://api.elevenlabs.io/v1/user").mock(return_value=httpx.Response(200, json={}))
    respx.get("https://openrouter.ai/api/v1/key").mock(return_value=httpx.Response(200, json={}))
    respx.get(f"https://generativelanguage.googleapis.com/v1beta/models/{gemini.DEFAULT_MODEL}").mock(
        return_value=httpx.Response(200, json={}))
    assert "ping" not in runner.invoke(app, ["doctor"]).output and route.call_count == 0
    res = runner.invoke(app, ["doctor", "--live"])
    assert "elevenlabs ping" in res.output and "key accepted" in res.output and route.call_count == 1


def test_cli_never_prints_a_key(monkeypatch):
    monkeypatch.setattr(doctor, "_hf_doctor", lambda: {"ok": True})
    res = runner.invoke(app, ["doctor"])
    assert "fake-gemini_api_key" not in res.output and "fake-elevenlabs_api_key" not in res.output
    assert "GEMINI_API_KEY" in res.output and "set" in res.output


def test_cli_exit_code_and_marks(monkeypatch):
    checks = [doctor.Check(name="a", ok=True, detail="fine"),
              doctor.Check(name="b", ok=False, level="warning", detail="meh"),
              doctor.Check(name="c", ok=False, level="info", detail="note")]
    monkeypatch.setattr(doctor, "run_checks", lambda live=False: checks)
    res = runner.invoke(app, ["doctor"])
    assert res.exit_code == 0 and "WARN  b" in res.output and "info  c" in res.output
    assert "0 errors, 1 warnings" in res.output
    checks.append(doctor.Check(name="d", ok=False, detail="broken"))
    res = runner.invoke(app, ["doctor"])
    assert res.exit_code == 1 and "FAIL  d" in res.output and "1 errors, 1 warnings" in res.output


@pytest.mark.live
@pytest.mark.parametrize("provider", ["gemini", "elevenlabs", "openrouter"])
def test_live_ping_with_the_real_key(provider):
    """Each provider on its own: a setup with only some of the keys still tests the ones it has."""
    doctor.load_env()
    check = doctor._ping_one(provider)
    if check is None:
        pytest.skip(f"no real {provider} key in .env")
    assert check.ok, check.detail
