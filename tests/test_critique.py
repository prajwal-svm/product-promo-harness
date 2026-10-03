import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest
import numpy as np
import soundfile as sf
from audio_helpers import SR, mix, say, voice, with_audio
from typer.testing import CliRunner

from promo import critique, hyperframes
from promo.audio_analysis import loudness
from promo.cli import app
from promo.models import LineTake
from promo.packs import load_pack
from promo.proc import sha256_file
from promo.run import RunPaths, new_run
from promo.sfx import render_stem
from promo.shotlist import Shotlist
from promo.timing import Measured, resolve, write_timing

FIX = Path(__file__).parent / "fixtures" / "shotlist_music.json"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "promo" / "oleafly-30s"
runner = CliRunner()


def _fake_key_applies(node) -> bool:
    """Live tests must read the real key from .env, so they never get the fake one."""
    return node.get_closest_marker("live") is None


@pytest.fixture(autouse=True)
def keys(request, monkeypatch):
    if _fake_key_applies(request.node):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")
        monkeypatch.setenv("GEMINI_API_KEY", "gm-test")


def test_fake_key_skips_live_tests():
    class Node:
        def __init__(self, marker):
            self.marker = marker

        def get_closest_marker(self, name):
            return self.marker if name == "live" else None

    assert _fake_key_applies(Node(None)) is True
    assert _fake_key_applies(Node(object())) is False


@pytest.fixture
def fast(monkeypatch):
    """Skip cutting the sheets, strips and spectrogram, which the checks do not read; a test that wants the evidence
    files leaves this out."""
    monkeypatch.setattr(critique, "frames_at", lambda *a, **k: [])
    for name in ("strip", "phone_sheet", "master_sheet", "spectrogram"):
        monkeypatch.setattr(critique, name, lambda *a, **k: None)
    monkeypatch.setattr(critique, "fastest_action_t", lambda video: 0.0)


def _film(tmp_path, *, gain_db=0.0, data_edit=None, sfx=True, audio=True):
    """A run with a 320x180 shot list, a take for every line, the timing and SFX stem made from them, and a master of the
    film's length: a quiet bed, each take at its planned start and the stem, all at `gain_db`. The last value is a
    recogniser that hears each line's own words in the window it is given."""
    run = new_run(tmp_path, "d", "saas-kinetic", "16:9", None, "en")
    data = json.loads(FIX.read_text())
    data["canvas"] = {"aspect": "16:9", "width": 320, "height": 180, "fps": 30}
    if data_edit:
        data_edit(data)
    run.shotlist.write_text(json.dumps(data))
    pack = load_pack("saas-kinetic")
    shotlist = Shotlist.model_validate(data)
    takes, voices = {}, {}
    for k, line in enumerate(shotlist.all_lines()):
        words = say(line.text)
        voices[line.id] = voice(len(words) * 0.3, 10 + k)
        sf.write(run.vo_dir / f"{line.id}.wav", voices[line.id], SR, subtype="FLOAT")
        takes[line.id] = LineTake(id=line.id, file=f"audio/vo/{line.id}.wav", duration_s=len(words) * 0.3, words=words,
                                  verified=line.verify)
    t = resolve(shotlist, Measured(takes=takes), pack.limits())
    write_timing(t, run.timing)
    parts = [(voices[l.id], l.start_s, 0.5) for l in t.lines]
    if sfx:
        stem, _, _ = render_stem(t, pack, run.sfx_dir / "track.wav")
        parts.append((sf.read(stem, dtype="float32")[0], 0.0, 0.25))
    run.renders.mkdir(parents=True, exist_ok=True)
    wav = mix(tmp_path / "mix.wav", parts, gain_db=gain_db) if audio else None
    with_audio(wav, run.renders / "master.mp4")
    text = {l.id: l.text for l in t.lines}
    return run, t, pack, lambda path, lang: say(text[Path(path).stem])


def _round(tmp_path, **kw):
    run, t, pack, words = _film(tmp_path, **{k: kw.pop(k) for k in ("gain_db", "data_edit", "sfx", "audio") if k in kw})
    rd = critique.critique(run, t, pack, transcribe=kw.pop("transcribe", words), hf=kw.pop("hf", False), **kw)
    return run, rd, json.loads((rd / "summary.json").read_text())


def _check(summary, name):
    return next(c for c in summary["checks"] if c["name"] == name)


def test_critique_round_writes_evidence_summary_and_prompt(tmp_path):
    run = new_run(tmp_path, "d", "saas-kinetic", "16:9", None, "en")
    data = json.loads(FIX.read_text())
    data["canvas"] = {"aspect": "16:9", "width": 320, "height": 180, "fps": 30}
    run.shotlist.write_text(json.dumps(data))
    pack = load_pack("saas-kinetic")
    t = resolve(Shotlist.model_validate(data), Measured(), pack.limits())
    write_timing(t, run.timing)
    render_stem(t, pack, run.sfx_dir / "track.wav")
    master = run.renders / "master.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=8.3",
                    "-f", "lavfi", "-i", "sine=duration=8.3:sample_rate=48000", "-af", "volume=6dB",
                    "-pix_fmt", "yuv420p", "-shortest", str(master)], check=True)
    words = [w for l in t.lines for w in l.words]
    rd = critique.critique(run, t, pack, transcribe=lambda p, lang: words, hf=False)
    assert rd == run.critic / "1"
    summary = json.loads((rd / "summary.json").read_text())
    names = {c["name"] for c in summary["checks"]}
    assert {"tech", "loudness", "sync_sfx", "sync_vo", "vo_wer", "claims", "determinism"} <= names
    det = next(c for c in summary["checks"] if c["name"] == "determinism")
    assert det["ok"] is False and "not run" in det["detail"]
    prompt = (rd / "PROMPT.md").read_text()
    assert "harsh motion director" in prompt
    assert "The capsule cursor is the only pointer." in prompt
    assert "idle-sine-loop" not in prompt.split("## Banned")[1].split("##")[0]  # re-allowed by the pack
    for f in ("contact-00.png", "phone.png", "master.png", "strip-fastest.png", "strip-seam-c02-s1.png",
              "spectrogram.png", "sync.json", "tech.json", "loudness.json"):
        assert (rd / "evidence" / f).exists(), f
    assert critique.next_round(run) == 2


def test_a_round_leaves_only_the_evidence_files_and_no_staging_directory(tmp_path):
    run, rd, _ = _round(tmp_path)
    assert sorted(p.name for p in rd.iterdir()) == ["PROMPT.md", "evidence", "summary.json"]
    assert [p.name for p in run.critic.iterdir()] == ["1"]
    assert not list((rd / "evidence").glob("*/"))    # the frames the sheets are cut from are not kept


def test_music_mode_samples_a_frame_per_beat_and_voice_mode_two_per_second(tmp_path, monkeypatch, fast):
    asked = []
    monkeypatch.setattr(critique, "frames_at", lambda video, times, out_dir, *a, **k: asked.append(list(times)) or [])
    run, t, pack, words = _film(tmp_path / "m")
    critique.critique(run, t, pack, transcribe=words, hf=False)
    assert asked[-1] == [b for b in t.beats if b <= t.duration_s - 0.02] and asked[-1][:3] == [0.3, 0.8, 1.3]

    def voice_mode(data):
        data["timing_mode"] = "voice"
        data["music"]["bpm"] = 100
        for ch in data["chapters"]:
            ch.pop("bars", None)
            for line in ch["vo"]:
                line.pop("at", None)
            for shot in ch["shots"]:
                shot.pop("beats", None)
                for e in shot["events"]:
                    e["at"] = {"shot": shot["id"], "edge": "start", "offset": 0.4}
                shot["seconds"] = 3.0

    run, t, pack, words = _film(tmp_path / "v", data_edit=voice_mode)
    assert t.timing_mode == "voice" and t.beats and t.beats[1] - t.beats[0] == pytest.approx(0.6)
    critique.critique(run, t, pack, transcribe=words, hf=False)
    assert asked[-1] == [i * 0.5 for i in range(math.ceil(t.duration_s / 0.5))]


def test_a_contact_sheet_holds_36_frames_and_the_next_sheet_takes_the_rest(tmp_path):
    from PIL import Image

    def long_chapter(data):
        data["chapters"][0]["bars"] = 8          # 8 + 2 bars at 120 bpm: 41 beats

    _, rd, _ = _round(tmp_path, data_edit=long_chapter)
    sheets = sorted(p.name for p in (rd / "evidence").glob("contact-*.png"))
    assert sheets == ["contact-00.png", "contact-01.png"]
    assert Image.open(rd / "evidence" / "contact-00.png").size == (6 * (480 + 4), 6 * (270 + 4))


def test_a_stem_that_is_missing_fails_sync_sfx_but_the_voice_checks_run_on_their_own(tmp_path, fast):
    _, _, summary = _round(tmp_path, sfx=False)
    assert _check(summary, "sync_sfx")["ok"] is False and "promo sfx render" in _check(summary, "sync_sfx")["detail"]
    assert _check(summary, "sync_vo")["ok"] is True and _check(summary, "vo_wer")["ok"] is True


def test_unreadable_placements_fail_sync_sfx_with_a_pointer(tmp_path, fast):
    run, t, pack, words = _film(tmp_path)
    (run.sfx_dir / "placements.json").write_text("{not json")
    rd = critique.critique(run, t, pack, transcribe=words, hf=False)
    c = _check(json.loads((rd / "summary.json").read_text()), "sync_sfx")
    assert c["ok"] is False and "placements.json cannot be read" in c["detail"]
    (run.sfx_dir / "placements.json").unlink()
    c = _check(json.loads((critique.critique(run, t, pack, transcribe=words, hf=False)
                           / "summary.json").read_text()), "sync_sfx")
    assert c["ok"] is False and "no audio/sfx/placements.json" in c["detail"]


def test_a_stale_stem_is_named_in_the_sfx_check(tmp_path, fast):
    run, t, pack, words = _film(tmp_path)
    moved = t.model_copy(update={"events": [e.model_copy(update={"t": e.t + 0.2}) if e.id == "c02-e2" else e
                                            for e in t.events]})
    rd = critique.critique(run, moved, pack, transcribe=words, hf=False)
    c = _check(json.loads((rd / "summary.json").read_text()), "sync_sfx")
    assert c["ok"] is False and "c02-e2" in c["detail"] and "now has it at" in c["detail"]


def test_a_line_that_skips_verification_is_not_scored_but_its_take_is_still_found_in_the_mix(tmp_path, fast):
    def psst(data):
        data["chapters"][0]["vo"][0]["verify"] = False

    run, t, pack, words = _film(tmp_path, data_edit=psst)
    rd = critique.critique(run, t, pack, transcribe=lambda p, lang: [] if Path(p).stem == "c01-l1" else words(p, lang),
                           hf=False)
    summary = json.loads((rd / "summary.json").read_text())
    assert _check(summary, "vo_wer")["ok"] is True and _check(summary, "sync_vo")["ok"] is True
    assert [f["code"] for f in summary["findings"] if f["code"] == "vo_not_scored"] == ["vo_not_scored"]
    assert "c01-l1: verify is false" in (rd / "PROMPT.md").read_text()
    sync = json.loads((rd / "evidence" / "sync.json").read_text())
    assert sync["lines"]["rows"][0] == {"line": "c01-l1", "wer": None, "ok": True, "skipped": "verify: false"}
    assert sync["vo"]["rows"][0]["ok"] is True and sync["vo"]["rows"][0]["corr"] > 0.5


def test_the_shot_lists_asr_aliases_are_applied_to_what_the_mix_sounds_like(tmp_path, fast):
    def alias(data):
        data["chapters"][0]["vo"][0]["text"] = "Oleafly money just got safer."
        data["asr_aliases"] = {"Oleafly": ["oh leafly"]}

    run, t, pack, words = _film(tmp_path, data_edit=alias)
    heard = lambda p, lang: say("oh leafly money just got safer.") if Path(p).stem == "c01-l1" else words(p, lang)
    summary = json.loads((critique.critique(run, t, pack, transcribe=heard, hf=False) / "summary.json").read_text())
    assert _check(summary, "vo_wer")["ok"] is True and _check(summary, "sync_vo")["ok"] is True


def test_an_unsourced_number_fails_the_claims_check(tmp_path, fast):
    def number(data):
        data["chapters"][0]["vo"][0]["text"] = "Moving 40 companies just got safer."

    _, _, summary = _round(tmp_path, data_edit=number)
    c = _check(summary, "claims")
    assert c["ok"] is False and "40" in c["detail"]


def test_a_video_with_no_audio_still_gets_a_round_with_the_audio_checks_failing(tmp_path):
    run, rd, summary = _round(tmp_path, audio=False)
    loud = _check(summary, "loudness")
    assert loud["ok"] is False and loud["detail"] == ("the video has no audio track; music planned (pack music mode "
                                                      "'bed') but no bed: run `promo music`")
    assert _check(summary, "sync_vo")["ok"] is False and _check(summary, "tech")["ok"] is True
    assert not (rd / "evidence" / "spectrogram.png").exists()
    assert json.loads((rd / "evidence" / "loudness.json").read_text())["measured"] is None


def test_the_round_judges_the_video_it_is_given(tmp_path, fast):
    run, t, pack, words = _film(tmp_path)
    other = run.renders / "other.mp4"
    shutil.copy(run.renders / "master.mp4", other)
    (run.renders / "master.mp4").unlink()
    rd = critique.critique(run, t, pack, video=other, transcribe=words, hf=False)
    summary = json.loads((rd / "summary.json").read_text())
    assert summary["video"] == str(other.resolve())
    assert summary["video_sha256"] == sha256_file(other)          # what `promo ship` compares with the master
    assert f"Video: {other}" in (rd / "PROMPT.md").read_text()


def test_no_video_is_a_missing_prerequisite_and_makes_no_round(tmp_path):
    run, t, pack, _ = _film(tmp_path)
    (run.renders / "master.mp4").unlink()
    with pytest.raises(critique.MissingMaster, match="promo assemble"):
        critique.critique(run, t, pack, hf=False)
    assert critique.next_round(run) == 1 and not list(run.critic.iterdir())


def test_a_round_that_fails_part_way_leaves_nothing_and_does_not_use_up_its_number(tmp_path, monkeypatch):
    run, t, pack, words = _film(tmp_path)
    monkeypatch.setattr(critique, "phone_sheet", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        critique.critique(run, t, pack, transcribe=words, hf=False)
    assert not list(run.critic.iterdir()) and critique.next_round(run) == 1
    monkeypatch.undo()
    assert critique.critique(run, t, pack, transcribe=words, hf=False) == run.critic / "1"


def test_a_staging_directory_from_another_run_is_neither_counted_nor_touched(tmp_path, fast):
    run, t, pack, words = _film(tmp_path)
    other = run.critic / ".1.other"
    (other / "evidence").mkdir(parents=True)
    (other / "marker").write_text("a concurrent run's")
    rd = critique.critique(run, t, pack, transcribe=words, hf=False)
    assert rd == run.critic / "1" and (other / "marker").read_text() == "a concurrent run's"
    assert critique.next_round(run) == 2 and (rd.stat().st_mode & 0o777) == 0o755       # a mkdtemp directory is 0700


def test_the_round_records_a_relative_video_resolved(tmp_path, fast, monkeypatch):
    """`promo critique --video master.mp4` from renders/: the run is found upwards, the path is relative to here."""
    run, t, pack, words = _film(tmp_path)
    monkeypatch.chdir(run.renders)
    rd = critique.critique(run, t, pack, video=Path("master.mp4"), transcribe=words, hf=False)
    assert json.loads((rd / "summary.json").read_text())["video"] == str((run.renders / "master.mp4").resolve())


def test_rounds_lists_finished_rounds_without_creating_anything(tmp_path):
    run = RunPaths(tmp_path / "nowhere")
    assert critique.rounds(run) == [] and not run.critic.exists()
    run.critic.mkdir(parents=True)
    for name in ("2", "10", ".3.staging", "notes"):
        (run.critic / name).mkdir()
    (run.critic / "7").write_text("a file, not a round")
    assert critique.rounds(run) == [2, 10]
    assert critique.next_round(run) == 11


def test_two_rounds_racing_for_the_same_number_both_land(tmp_path, fast, monkeypatch):
    """Both runs pick round 1 and build it at once; the second to finish takes the next free number."""
    import threading

    run, t, pack, words = _film(tmp_path)
    both_built = threading.Barrier(2, timeout=60)
    build = critique._build_round

    def build_then_wait(*args, **kwargs):
        build(*args, **kwargs)
        both_built.wait()             # neither round lands until both have picked their number and been built

    monkeypatch.setattr(critique, "_build_round", build_then_wait)
    landed, errors = [], []

    def one_round():
        try:
            landed.append(critique.critique(run, t, pack, transcribe=words, hf=False))
        except BaseException as e:   # surfaced below
            errors.append(e)

    threads = [threading.Thread(target=one_round) for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors, errors
    assert sorted(p.name for p in landed) == ["1", "2"]
    for rd in landed:
        assert json.loads((rd / "summary.json").read_text())["round"] == int(rd.name)
        assert (rd / "PROMPT.md").read_text().startswith(f"# Critic round {rd.name}\n")
    assert critique.rounds(run) == [1, 2]
    assert not [p for p in run.critic.iterdir() if p.name.startswith(".")], "no staging folder is left behind"


def test_a_round_gives_up_after_a_bounded_number_of_collisions(tmp_path, fast, monkeypatch):
    run, t, pack, words = _film(tmp_path)
    real_replace = critique.os.replace

    def always_taken(src, dst):
        Path(dst).mkdir(exist_ok=True)
        (Path(dst) / "taken").write_text("by another run")
        return real_replace(src, dst)

    monkeypatch.setattr(critique.os, "replace", always_taken)
    with pytest.raises(critique.CritiqueError, match="tried 5 times"):
        critique.critique(run, t, pack, transcribe=words, hf=False)
    assert not [p for p in run.critic.iterdir() if p.name.startswith(".")], "the staged round is cleaned up"


def _no_sound(data):
    for ch in data["chapters"]:
        ch["vo"] = []
    data["chapters"][0]["shots"][0]["events"][1]["at"] = {"beat": 2}


def _silent_pack(pack):
    """The pack with music mode none and no sounds for any event: a film made with it plans no sound at all."""
    return pack.model_copy(update={"music": {**pack.music, "mode": "none"}, "sfx": {"sfx_gain_db": -12}})


def test_a_film_silent_by_design_passes_the_audio_checks_as_not_applicable(tmp_path, fast):
    run, t, pack, words = _film(tmp_path, data_edit=_no_sound, sfx=False, audio=False)
    silent = _silent_pack(pack)
    assert critique.planned_sound(t, silent) == {"vo_lines": 0, "sfx_placements": 0, "music": False}
    summary = json.loads((critique.critique(run, t, silent, transcribe=words, hf=False) / "summary.json").read_text())
    for name in ("loudness", "sync_sfx"):
        assert _check(summary, name) == {"name": name, "ok": True, "detail": "n/a (no sound planned)"}
    assert _check(summary, "sync_vo")["ok"] is True and _check(summary, "vo_wer")["ok"] is True


def test_planned_sound_with_a_silent_master_fails_and_names_each_missing_step(tmp_path, fast):
    """No VO and no stem or bed on disk is not enough: the pack plans a bed and the events have pack sounds."""
    run, t, pack, words = _film(tmp_path, data_edit=_no_sound, sfx=False, audio=False)
    planned = critique.planned_sound(t, pack)
    assert planned == {"vo_lines": 0, "sfx_placements": 4, "music": True}
    summary = json.loads((critique.critique(run, t, pack, transcribe=words, hf=False) / "summary.json").read_text())
    loud = _check(summary, "loudness")
    assert loud["ok"] is False and loud["detail"].startswith("the video has no audio track")
    assert "SFX planned (4 placements) but no sfx stem: run `promo sfx render`" in loud["detail"]
    assert "music planned (pack music mode 'bed') but no bed: run `promo music`" in loud["detail"]
    sfx = _check(summary, "sync_sfx")
    assert sfx["ok"] is False and "promo sfx render" in sfx["detail"]


def test_the_music_bed_alone_makes_sound_planned(tmp_path, fast):
    run, t, pack, words = _film(tmp_path, data_edit=_no_sound, sfx=False, audio=False)
    bed_only = pack.model_copy(update={"sfx": {"sfx_gain_db": -12}})
    assert critique.missing_sound(run, t, bed_only) == [
        "music planned (pack music mode 'bed') but no bed: run `promo music`"]
    summary = json.loads((critique.critique(run, t, bed_only, transcribe=words, hf=False) / "summary.json").read_text())
    assert _check(summary, "loudness")["ok"] is False
    (run.music_dir / "bed.wav").write_bytes(b"made")                       # the bed exists: nothing is missing
    assert critique.missing_sound(run, t, bed_only) == []


def test_vo_lines_without_takes_are_named(tmp_path, fast):
    run, t, pack, words = _film(tmp_path, sfx=False, audio=False)
    (run.vo_dir / "c02-l2.wav").unlink()
    assert critique.missing_sound(run, t, _silent_pack(pack)) == [
        "VO planned (3 lines) but 1 has no take: run `promo voice --all`"]


def test_a_film_with_sound_planned_but_no_audio_still_fails(tmp_path, fast):
    run, rd, summary = _round(tmp_path, data_edit=_no_sound, audio=False)      # an SFX stem, but a silent master
    loud = _check(summary, "loudness")
    assert loud["ok"] is False and loud["detail"].startswith("the video has no audio track")


def test_loudness_is_not_applicable_only_when_no_sound_is_planned():
    na = critique.loudness_check(None, -14, -1.0, sound_planned=False)[0]
    assert na == {"name": "loudness", "ok": True, "detail": "n/a (no sound planned)"}
    silent = critique.loudness_check({"I": float("-inf"), "LRA": 0.0, "TP": float("-inf")}, -14, -1.0, sound_planned=False)
    assert silent[0]["ok"] is True
    loud = critique.loudness_check({"I": -9.0, "LRA": 1.0, "TP": -0.2}, -14, -1.0, sound_planned=False)[0]
    assert loud["ok"] is False and "-9.0 LUFS" in loud["detail"], "audio that is there is still measured"
    named = critique.loudness_check(None, -14, -1.0, missing=["SFX planned (2 placements) but no sfx stem: run x"])[0]
    assert named["detail"] == "the video has no audio track; SFX planned (2 placements) but no sfx stem: run x"


def test_a_sound_that_would_have_to_be_paid_for_is_left_out_not_bought(tmp_path, monkeypatch):
    from promo.providers import elevenlabs

    run, t, pack, _ = _film(tmp_path)
    bought = []
    monkeypatch.setattr(elevenlabs, "sound_generation", lambda *a, **k: bought.append(a) or b"")
    generated = pack.model_copy(update={"sfx": {**pack.sfx, "click": {"gen": "a soft click", "duration": 0.2}}})
    placements = json.loads((run.sfx_dir / "placements.json").read_text())
    from promo.sfx import Placement

    clips = critique._clips(t, generated, [Placement(**p) for p in placements])
    assert bought == [] and "c01-e2" not in clips                  # the click event: not in the cache, not bought
    assert {"c01-e1", "c02-e1", "c02-e2"} <= set(clips)           # the synthesised ones are made again exactly


def test_a_sound_that_cannot_be_found_in_the_stem_fails_and_only_a_silent_clip_is_a_warning(tmp_path, monkeypatch, fast):
    run, t, pack, words = _film(tmp_path)
    monkeypatch.setattr(critique, "_clips", lambda *a, **k: {})
    summary = json.loads((critique.critique(run, t, pack, transcribe=words, hf=False) / "summary.json").read_text())
    c = _check(summary, "sync_sfx")
    assert c["ok"] is False and "c01-e1: not found in stem" in c["detail"]
    assert not [f for f in summary["findings"] if f["code"] == "sfx_unmeasurable"]
    monkeypatch.setattr(critique, "_clips", lambda timing, pack, placements: {
        p.event: (np.zeros(64), round(p.start_s * SR)) for p in placements})
    summary = json.loads((critique.critique(run, t, pack, transcribe=words, hf=False) / "summary.json").read_text())
    assert _check(summary, "sync_sfx")["ok"] is True
    unmeasured = [f for f in summary["findings"] if f["code"] == "sfx_unmeasurable"]
    assert len(unmeasured) == 1 and "c01-e1" in unmeasured[0]["message"] and "not reported as late" in unmeasured[0]["message"]


def test_the_sfx_check_finds_a_stem_whose_sounds_sit_late_and_says_by_how_much(tmp_path, fast):
    run, t, pack, words = _film(tmp_path)
    stem = sf.read(run.sfx_dir / "track.wav", dtype="float32")[0]
    sf.write(run.sfx_dir / "track.wav", np.roll(stem, round(0.06 * SR)), SR, subtype="PCM_24")     # 60 ms late, all of it
    summary = json.loads((critique.critique(run, t, pack, transcribe=words, hf=False) / "summary.json").read_text())
    c = _check(summary, "sync_sfx")
    assert c["ok"] is False and "c01-e1 +60.0 ms" in c["detail"]


# --- the loudness check ------------------------------------------------------------------------------------------------

def test_loudness_within_a_unit_of_target_passes_and_over_half_a_unit_under_is_still_a_warning():
    ok, findings = critique.loudness_check({"I": -14.2, "LRA": 3.0, "TP": -2.0}, -14, -1.0)
    assert ok["ok"] is True and findings == []
    ok, findings = critique.loudness_check({"I": -14.8, "LRA": 3.0, "TP": -2.0}, -14, -1.0)
    assert ok["ok"] is True
    assert [(f.level, f.code) for f in findings] == [("warning", "loudness_under_target")]
    assert "0.8 LU under the -14 LUFS target" in findings[0].message
    assert "0.8 LU under target" in ok["detail"]


def test_loudness_more_than_a_unit_off_target_or_over_the_true_peak_limit_fails():
    assert critique.loudness_check({"I": -15.6, "LRA": 1.0, "TP": -3.0}, -14, -1.0)[0]["ok"] is False
    assert critique.loudness_check({"I": -12.5, "LRA": 1.0, "TP": -3.0}, -14, -1.0)[0]["ok"] is False
    over_peak, _ = critique.loudness_check({"I": -14.0, "LRA": 1.0, "TP": -0.4}, -14, -1.0)
    assert over_peak["ok"] is False
    assert critique.loudness_check({"I": -14.0, "LRA": 1.0, "TP": -1.0}, -14, -1.0)[0]["ok"] is True
    loud, findings = critique.loudness_check({"I": -12.5, "LRA": 1.0, "TP": -3.0}, -14, -1.0)
    assert findings == []       # being over target is a failure, not the under-target warning


def test_an_sfx_only_mix_that_lands_at_minus_15_6_is_explained_from_the_assemble_report():
    report = {"I": -15.6, "TP": -1.2, "ceiling_db": -2.0, "normalization": "linear-gain+limiter", "gain_db": 7.4}
    check, findings = critique.loudness_check({"I": -15.6, "LRA": 0.5, "TP": -1.2}, -14, -1.0, report)
    assert check["ok"] is False
    msg = findings[0].message
    assert "1.6 LU under the -14 LUFS target" in msg and "applied +7.4 dB" in msg and "-2.0 dBFS" in msg
    assert "too peaky" in msg
    _, plain = critique.loudness_check({"I": -15.6, "LRA": 0.5, "TP": -1.2}, -14, -1.0)
    assert "assemble" not in plain[0].message
    _, only_limiter = critique.loudness_check({"I": -15.6, "LRA": 0.5, "TP": -1.2}, -14, -1.0,
                                              {"normalization": "limiter-only", "gain_db": None})
    assert "only limited the peaks" in only_limiter[0].message


def test_silent_or_absent_audio_fails_the_loudness_check_with_its_own_message():
    assert critique.loudness_check(None, -14, -1.0)[0]["detail"] == "the video has no audio track"
    for silent in (float("-inf"), -70.0):         # what ebur128 reports for nothing, or for what it gates out
        check, findings = critique.loudness_check({"I": silent, "LRA": 0.0, "TP": float("-inf")}, -14, -1.0)
        assert check["ok"] is False and "silent" in check["detail"] and findings == []


def test_the_round_reads_the_assemble_report_next_to_the_video(tmp_path, fast):
    run, t, pack, words = _film(tmp_path, gain_db=-25)
    (run.renders / "master.loudness.json").write_text(json.dumps(
        {"I": -15.6, "TP": -1.2, "ceiling_db": -2.0, "normalization": "linear-gain+limiter", "gain_db": 7.4}))
    rd = critique.critique(run, t, pack, transcribe=words, hf=False)
    summary = json.loads((rd / "summary.json").read_text())
    under = [f for f in summary["findings"] if f["code"] == "loudness_under_target"]
    assert len(under) == 1 and "applied +7.4 dB" in under[0]["message"]
    assert "under the -14 LUFS target" in (rd / "PROMPT.md").read_text().split("## Warnings")[1].split("##")[0]
    shown = json.loads((rd / "evidence" / "loudness.json").read_text())
    assert shown["assemble_report"]["normalization"] == "linear-gain+limiter" and shown["target_lufs"] == -14
    assert shown["measured"]["I"] == pytest.approx(loudness(run.renders / "master.mp4")["I"])


def test_a_silent_master_writes_json_the_next_tool_can_read(tmp_path, fast):
    run, t, pack, words = _film(tmp_path, gain_db=-200)
    rd = critique.critique(run, t, pack, transcribe=words, hf=False)
    text = (rd / "evidence" / "loudness.json").read_text()
    assert "Infinity" not in text and json.loads(text)["measured"]["TP"] is None
    assert "silent" in _check(json.loads((rd / "summary.json").read_text()), "loudness")["detail"]


# --- the determinism check ---------------------------------------------------------------------------------------------

def _verify_json(run, **fields):
    body = {"ok": True, "shots": ["c01-s1", "c02-s1", "c02-s2"], "mismatches": [], "fingerprints": {}}
    body.update(fields)
    run.renders.mkdir(parents=True, exist_ok=True)
    (run.renders / "verify.json").write_text(json.dumps(body))


def test_determinism_passes_with_a_clean_verify_of_every_shot(tmp_path):
    run, t, pack, _ = _film(tmp_path)
    _verify_json(run)
    assert critique._determinism_check(run, t, pack) == {"name": "determinism", "ok": True, "detail": "ok"}


def test_determinism_fails_for_mismatches_a_partial_verify_and_a_file_it_cannot_read(tmp_path):
    run, t, pack, _ = _film(tmp_path)
    _verify_json(run, ok=False, mismatches=[{"shot": "c01-s1", "t_local": 0.5}])
    bad = critique._determinism_check(run, t, pack)
    assert bad["ok"] is False and "c01-s1" in bad["detail"]
    _verify_json(run, shots=["c01-s1"])                      # `promo verify --shot c01-s1` is not every shot
    partial = critique._determinism_check(run, t, pack)
    assert partial["ok"] is False and "not verified: c02-s1, c02-s2" in partial["detail"]
    (run.renders / "verify.json").write_text("{not json")
    assert "cannot be read" in critique._determinism_check(run, t, pack)["detail"]
    (run.renders / "verify.json").write_text("[]")
    assert critique._determinism_check(run, t, pack)["ok"] is False


def test_determinism_fails_for_a_shot_changed_since_it_was_verified(tmp_path):
    from promo.hyperframes import ensure_shot, scaffold_scenes
    from promo.scenes import design_for, shot_hash

    run, t, pack, _ = _film(tmp_path)
    scaffold_scenes(run.scenes)
    for s in t.shots:
        ensure_shot(run.scenes, s.id)
    prints = {s.id: shot_hash(run.scenes, t, s.id, design_for(run, pack), "high", t.fps) for s in t.shots}
    _verify_json(run, fingerprints=prints)
    assert critique._determinism_check(run, t, pack)["ok"] is True
    scene = run.shots_dir / "c02-s2.html"
    scene.write_text(scene.read_text() + "\n<!-- edited -->\n")
    stale = critique._determinism_check(run, t, pack)
    assert stale["ok"] is False and "changed since `promo verify`: c02-s2" in stale["detail"]


# --- the HyperFrames check -----------------------------------------------------------------------------------------------

def _fake_hf(monkeypatch, errors=None):
    """HyperFrames' `check`, which carries its own lint result; `lint` is there to prove it is not called again."""
    errors = errors or {}
    calls = []

    def lint(project):
        calls.append(("lint", project.name))
        return {"ok": True, "errorCount": 0}

    def check(project):
        calls.append(("check", project.name))
        n = errors.get(project.name, 0)
        if n:
            return {"ok": False, "skipped": "lint errors", "lint": {"ok": False, "errorCount": n}}
        return {"ok": True, "lint": {"ok": True, "errorCount": 0}, "layout": {"ok": True}}

    monkeypatch.setattr(hyperframes, "lint", lint)
    monkeypatch.setattr(hyperframes, "check", check)
    return calls


def test_hf_lint_is_run_on_each_shot_project_and_recorded(tmp_path, monkeypatch, fast):
    run, t, pack, words = _film(tmp_path)
    for s in t.shots:
        (run.render_dir / s.id).mkdir(parents=True)
    calls = _fake_hf(monkeypatch)
    rd = critique.critique(run, t, pack, transcribe=words, hf=True)
    assert _check(json.loads((rd / "summary.json").read_text()), "hf_lint") == {"name": "hf_lint", "ok": True, "detail": "ok"}
    assert sorted(calls) == [("check", "c01-s1"), ("check", "c02-s1"), ("check", "c02-s2")]      # lint is not run twice
    hf = json.loads((rd / "evidence" / "hf.json").read_text())
    assert set(hf) == {"c01-s1", "c02-s1", "c02-s2"} and hf["c01-s1"]["layout"] == {"ok": True}
    assert "evidence/hf.json" in (rd / "PROMPT.md").read_text()


def test_hf_lint_errors_fail_by_shot_and_no_projects_at_all_is_not_a_pass(tmp_path, monkeypatch, fast):
    run, t, pack, words = _film(tmp_path)
    (run.render_dir / "c01-s1").mkdir(parents=True)
    (run.render_dir / "c02-s1").mkdir()
    _fake_hf(monkeypatch, errors={"c02-s1": 2})
    c = _check(json.loads((critique.critique(run, t, pack, transcribe=words, hf=True)
                           / "summary.json").read_text()), "hf_lint")
    assert c["ok"] is False and "lint errors in c02-s1" in c["detail"] and "no render project" in c["detail"]
    shutil.rmtree(run.render_dir)
    c = _check(json.loads((critique.critique(run, t, pack, transcribe=words, hf=True)
                           / "summary.json").read_text()), "hf_lint")
    assert c["ok"] is False and "c01-s1, c02-s1, c02-s2" in c["detail"]


def test_hf_is_not_run_without_the_flag(tmp_path, monkeypatch, fast):
    run, t, pack, words = _film(tmp_path)
    calls = _fake_hf(monkeypatch)
    rd = critique.critique(run, t, pack, transcribe=words, hf=False)
    assert calls == [] and not (rd / "evidence" / "hf.json").exists()
    assert "hf_lint" not in {c["name"] for c in json.loads((rd / "summary.json").read_text())["checks"]}


# --- the command ----------------------------------------------------------------------------------------------------------

def _use_words(monkeypatch, words):
    real = critique.critique
    monkeypatch.setattr(critique, "critique", lambda run, t, pack, **kw: real(run, t, pack, transcribe=words, **kw))


def _passing_film(tmp_path, monkeypatch):
    run, t, pack, words = _film(tmp_path, gain_db=0)
    master = run.renders / "master.mp4"
    gain = -14.4 - loudness(master)["I"]      # a little under, so that the true peak stays below -1 dBTP
    tuned = run.renders / "tuned.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(master), "-af", f"volume={gain:.3f}dB", "-c:v", "copy",
                    str(tuned)], check=True)
    tuned.replace(master)
    _verify_json(run)
    _use_words(monkeypatch, words)
    monkeypatch.chdir(run.root)
    return run


def test_critique_command_prints_each_check_and_the_prompt_path_and_exits_0_when_all_pass(tmp_path, monkeypatch, fast):
    run = _passing_film(tmp_path, monkeypatch)
    res = runner.invoke(app, ["critique", "--no-hf"])
    assert res.exit_code == 0, res.output
    for name in ("tech", "loudness", "sync_sfx", "sync_vo", "vo_wer", "claims", "determinism"):
        assert f"PASS  {name}" in res.output
    assert f"critic prompt: {run.critic / '1' / 'PROMPT.md'}" in res.output and "FAIL" not in res.output


def test_critique_command_exits_1_and_says_to_fix_first_when_a_check_fails(tmp_path, monkeypatch, fast):
    run, t, pack, words = _film(tmp_path)
    _use_words(monkeypatch, words)
    monkeypatch.chdir(run.root)
    res = runner.invoke(app, ["critique", "--no-hf"])
    assert res.exit_code == 1
    assert "FAIL  determinism" in res.output and "AUTOMATIC CHECKS FAILED" in res.output
    assert "warning  loudness_under_target" in res.output
    assert (run.critic / "1" / "PROMPT.md").exists()


def test_critique_command_without_a_master_is_exit_2_and_one_line(tmp_path, monkeypatch):
    run, t, pack, _ = _film(tmp_path)
    (run.renders / "master.mp4").unlink()
    monkeypatch.chdir(run.root)
    res = runner.invoke(app, ["critique", "--no-hf"])
    assert res.exit_code == 2 and "promo assemble" in res.output and res.output.count("promo:") == 1
    assert "Traceback" not in res.output and not list(run.critic.iterdir())
    res = runner.invoke(app, ["critique", "--no-hf", "--video", str(tmp_path / "nope.mp4")])
    assert res.exit_code == 2 and "nope.mp4" in res.output


def test_critique_command_without_timing_is_exit_2(tmp_path, monkeypatch):
    run, t, pack, _ = _film(tmp_path)
    run.timing.unlink()
    monkeypatch.chdir(run.root)
    res = runner.invoke(app, ["critique"])
    assert res.exit_code == 2 and "promo timing" in res.output


def test_critique_command_runs_hyperframes_unless_told_not_to(tmp_path, monkeypatch, fast):
    run, t, pack, words = _film(tmp_path)
    for s in t.shots:
        (run.render_dir / s.id).mkdir(parents=True)
    calls = _fake_hf(monkeypatch)
    _use_words(monkeypatch, words)
    monkeypatch.chdir(run.root)
    res = runner.invoke(app, ["critique"])
    assert "PASS  hf_lint" in res.output and calls
    calls.clear()
    assert "hf_lint" not in runner.invoke(app, ["critique", "--no-hf"]).output and not calls


def test_critique_command_failures_are_one_line_exit_1_and_a_bug_is_not_swallowed(tmp_path, monkeypatch):
    from promo.proc import ToolError

    run, t, pack, words = _film(tmp_path)
    monkeypatch.chdir(run.root)
    monkeypatch.setattr(critique, "critique", lambda *a, **k: (_ for _ in ()).throw(
        ToolError("command failed (1): ffmpeg -i x\nInvalid data found")))
    res = runner.invoke(app, ["critique", "--no-hf"])
    assert res.exit_code == 1 and "command failed" in res.output and "logs/last-error.log" in res.output
    assert "Traceback" not in res.output
    monkeypatch.setattr(critique, "critique", lambda *a, **k: (_ for _ in ()).throw(KeyError("a real bug")))
    assert isinstance(runner.invoke(app, ["critique", "--no-hf"]).exception, KeyError)


# --- live: real media, real speech recognition ----------------------------------------------------------------------------

def _copy_run(src: Path, dest: Path) -> RunPaths:
    shutil.copytree(src / "scenes", dest / "scenes", ignore=shutil.ignore_patterns(".render"))
    for name in ("shotlist.json", "timing.json", "design.json"):
        if (src / name).exists():
            shutil.copy(src / name, dest / name)
    return RunPaths(dest)


@pytest.mark.live
def test_live_a_real_silent_1080p_film_gets_a_round(tmp_path):
    """The Oleafly 30 s run's own silent render (copied into tmp; examples/ is only read) through a whole round:
    1080p60 frames, strips at every seam, and the audio checks failing for want of any audio."""
    silent = EXAMPLE / "renders" / "oleafly-30s-silent.mp4"
    if not silent.exists():
        pytest.skip("examples/promo/oleafly-30s is not here")
    run = _copy_run(EXAMPLE, tmp_path / "run")
    run.renders.mkdir(parents=True)
    shutil.copy(silent, run.renders / "master.mp4")
    from promo.shotlist import load_shotlist
    from promo.timing import load_timing

    t = load_timing(run.timing)
    rd = critique.critique(run, t, load_pack(load_shotlist(run.shotlist).pack), hf=False)
    summary = json.loads((rd / "summary.json").read_text())
    assert _check(summary, "tech")["name"] == "tech"
    assert _check(summary, "loudness")["detail"] == "the video has no audio track"
    for s in t.shots[1:]:
        assert (rd / "evidence" / f"strip-seam-{s.id}.png").exists()
    assert (rd / "evidence" / "contact-00.png").exists() and not (rd / "evidence" / "spectrogram.png").exists()


@pytest.mark.live
def test_live_the_template_shot_lints_clean_and_a_broken_one_does_not(tmp_path):
    """The real pinned HyperFrames lint, on a shot built from the harness's own template."""
    from promo.hyperframes import build_shot_project, ensure_shot, scaffold_scenes
    from promo.scenes import design_for

    run = new_run(tmp_path, "d", "saas-kinetic", "16:9", None, "en")
    run.shotlist.write_text(FIX.read_text())
    pack = load_pack("saas-kinetic")
    t = resolve(Shotlist.model_validate(json.loads(FIX.read_text())), Measured(), pack.limits())
    one = t.model_copy(update={"shots": [s for s in t.shots if s.id == "c01-s1"]})
    scaffold_scenes(run.scenes)
    ensure_shot(run.scenes, "c01-s1")
    project = build_shot_project(run.scenes, run.render_dir, t, "c01-s1", design_for(run, pack))
    (run.root / "ev").mkdir()
    assert critique._hf_check(run, one, run.root / "ev") == {"name": "hf_lint", "ok": True, "detail": "ok"}
    html = project / "index.html"
    html.write_text(html.read_text().replace('src="', 'src="../', 1).replace("assets/", "../assets/"))
    broken = critique._hf_check(run, one, run.root / "ev")
    assert broken["ok"] is False and "c01-s1" in broken["detail"]
