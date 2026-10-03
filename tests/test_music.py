import base64
import dataclasses
import itertools
import json
import math
import re
import subprocess
from pathlib import Path

import httpx
import numpy as np
import pytest
import respx
import soundfile as sf
from typer.testing import CliRunner

from promo import music
from promo.cli import app
from promo.config import MissingKeyError
from promo.models import BeatGrid, LineTake
from promo.packs import Limits, load_pack
from promo.proc import ToolError
from music_helpers import groove_mp3, write_groove

from promo.providers import elevenlabs, lyria
from promo.providers.http import ProviderError
from promo.run import new_run
from promo.shotlist import Shotlist
from promo.timing import Measured, TimingError, load_measured, resolve, write_timing

FIX = Path(__file__).parent / "fixtures" / "shotlist_music.json"
runner = CliRunner()


def _fake_key_applies(node) -> bool:
    """Live tests must read the real key from .env, so they never get the fake one."""
    return node.get_closest_marker("live") is None


@pytest.fixture(autouse=True)
def key(request, monkeypatch):
    if _fake_key_applies(request.node):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")


def test_fake_key_skips_live_tests():
    class Node:
        def __init__(self, marker):
            self.marker = marker

        def get_closest_marker(self, name):
            return self.marker if name == "live" else None

    assert _fake_key_applies(Node(None)) is True
    assert _fake_key_applies(Node(object())) is False


def _sl(**edit):
    data = json.loads(FIX.read_text())
    data.update(edit)
    return Shotlist.model_validate(data)


def _click_mp3(tmp_path, seconds=10.0, bpm=120.0) -> bytes:
    sr = 44100
    y = np.zeros(int(seconds * sr), dtype=np.float32)
    for i in range(int(seconds * bpm / 60)):
        t0 = int((0.25 + i * 60 / bpm) * sr)
        k = np.arange(int(0.12 * sr)) / sr
        amp = 0.9 if i % 4 == 0 else 0.3
        seg = amp * np.sin(2 * np.pi * (60 if i % 4 == 0 else 2000) * k) * np.exp(-k * 30)
        y[t0:t0 + len(seg)] += seg[: max(0, min(len(seg), len(y) - t0))]
    wav = tmp_path / "c.wav"
    sf.write(wav, y, sr)
    mp3 = tmp_path / "c.mp3"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-b:a", "192k", str(mp3)], check=True)
    return mp3.read_bytes()


def asked_ms(total_s, bpm=120.0, beats_per_bar=4):
    """What a one-call bed asks the provider for: its length with room for a 6% stretch, plus a bar the conform may
    trim before the music's first downbeat."""
    return round((total_s * 1.06 + beats_per_bar * 60 / bpm) * 1000)


def _no_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY")
    monkeypatch.setattr("promo.config.load_env", lambda path=None: None)
    monkeypatch.setattr("promo.cli.load_env", lambda path=None: None)


# --- the provider ---------------------------------------------------------------------------------------------

@respx.mock
def test_compose_request_shape():
    route = respx.post(elevenlabs.MUSIC_URL).mock(return_value=httpx.Response(200, content=b"ID3mp3"))
    out = elevenlabs.compose("a prompt", 1500)
    assert out == b"ID3mp3"
    req = route.calls[0].request
    body = json.loads(req.content)
    assert body == {"prompt": "a prompt", "music_length_ms": 3000, "model_id": "music_v2_5", "force_instrumental": True}
    assert req.url.params["output_format"] == "mp3_44100_192"
    assert req.headers["xi-api-key"] == "el-test"


@pytest.mark.parametrize("asked, sent", [(1500, 3000), (3000, 3000), (9500.4, 9500), (9_999_999, 600_000)])
@respx.mock
def test_compose_clamps_the_length(asked, sent):
    route = respx.post(elevenlabs.MUSIC_URL).mock(return_value=httpx.Response(200, content=b"ID3mp3"))
    elevenlabs.compose("p", asked)
    assert json.loads(route.calls[0].request.content)["music_length_ms"] == sent


@respx.mock
def test_compose_options_are_sent():
    route = respx.post(elevenlabs.MUSIC_URL).mock(return_value=httpx.Response(200, content=b"ID3mp3"))
    elevenlabs.compose("p", 5000, instrumental=False, model="music_next", output_format="mp3_44100_128")
    req = route.calls[0].request
    body = json.loads(req.content)
    assert body["force_instrumental"] is False and body["model_id"] == "music_next"
    assert req.url.params["output_format"] == "mp3_44100_128"


@respx.mock
def test_sound_generation_request_shape():
    route = respx.post(elevenlabs.SFX_URL).mock(return_value=httpx.Response(200, content=b"ID3sfx"))
    assert elevenlabs.sound_generation("soft UI click", 0.5) == b"ID3sfx"
    req = route.calls[0].request
    body = json.loads(req.content)
    assert body == {"text": "soft UI click", "duration_seconds": 0.5, "prompt_influence": 0.5}
    assert req.url.params["output_format"] == "mp3_44100_128"
    assert req.headers["xi-api-key"] == "el-test"


@respx.mock
def test_401_raises_provider_error_with_message():
    route = respx.post(elevenlabs.MUSIC_URL).mock(return_value=httpx.Response(
        401, json={"detail": {"status": "invalid_api_key", "message": "Invalid API key"}}))
    with pytest.raises(ProviderError) as err:
        elevenlabs.compose("p", 5000)
    assert "elevenlabs: HTTP 401: Invalid API key" in str(err.value)
    assert route.call_count == 1  # no retry on auth errors


@pytest.mark.parametrize("call", [lambda: elevenlabs.compose("p", 5000),
                                  lambda: elevenlabs.sound_generation("click", 0.5)])
@respx.mock
def test_missing_key_makes_no_request(monkeypatch, call):
    _no_key(monkeypatch)
    route = respx.post(url__regex=r"https://api\.elevenlabs\.io/.*").mock(return_value=httpx.Response(200, content=b"x"))
    with pytest.raises(MissingKeyError):
        call()
    assert route.call_count == 0


# --- sections and prompt --------------------------------------------------------------------------------------

def test_sections_are_relative_to_music_start():
    sl = _sl()
    t = resolve(sl, Measured(), Limits())
    secs = music.sections_from_timing(t, sl)
    assert [(s.id, s.start_s, s.dur_s) for s in secs] == [("c01", 0.0, 4.0), ("c02", 4.0, 4.0)]
    assert secs[0].note == "continue the groove"      # a chapter's purpose is for the film, not for the music


def test_section_note_joins_music_cues_or_continues_the_groove():
    data = json.loads(FIX.read_text())
    data["chapters"][1]["shots"][0]["sound"] = {"music_cue": "riser into the drop"}
    data["chapters"][0]["shots"][0]["sound"] = {"music_cue": "filter opens"}
    data["chapters"][0]["shots"].append({**data["chapters"][0]["shots"][0], "id": "c01-s2", "events": [],
                                          "sound": {"music_cue": "snare roll"}})
    sl = Shotlist.model_validate(data)
    secs = music.sections_from_timing(resolve(sl, Measured(), Limits()), sl)
    assert secs[0].note == "filter opens; snare roll"          # the purpose ("Hook") is not in it
    assert secs[1].note == "riser into the drop"
    plain = _sl()
    assert music.sections_from_timing(resolve(plain, Measured(), Limits()), plain)[1].note == "continue the groove"


def test_prompt_lists_timed_sections_and_negatives():
    sl = _sl()
    t = resolve(sl, Measured(), Limits())
    p = music.build_prompt(load_pack("saas-kinetic").music, music.sections_from_timing(t, sl), 92, 9.5)
    assert "92 BPM" in p and "Instrumental" in p
    assert "0:00.0–0:04.0 (c01)" in p and "0:04.0–0:08.0 (c02)" in p
    assert "Avoid: vocals, lyrics" in p
    assert p.splitlines()[-1] == "- 0:08.0–0:09.5: clean ending; no new elements"


def test_prompt_time_format_rolls_over_minutes():
    secs = [music.Section(id="c01", start_s=0.0, dur_s=61.5, note="n")]
    p = music.build_prompt({}, secs, 100, 63.0)
    assert "0:00.0–1:01.5 (c01)" in p and p.splitlines()[-1].startswith("- 1:01.5–1:03.0")
    assert "Avoid: vocals, lyrics" in p  # the default when a pack gives no negatives


@pytest.mark.parametrize("t, text", [(59.96, "1:00.0"), (119.97, "2:00.0"), (0.04, "0:00.0"), (61.5, "1:01.5"),
                                     (-3.0, "0:00.0")])
def test_mmss_never_shows_sixty_seconds(t, text):
    assert music._mmss(t) == text


# --- the bed --------------------------------------------------------------------------------------------------

def test_make_bed_writes_wav_prompt_beats_and_cost(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    t = resolve(sl, Measured(), Limits())
    mp3 = _click_mp3(tmp_path)
    calls = []

    def fake_compose(prompt, length_ms):
        calls.append((prompt, length_ms))
        return mp3

    bed = music.make_bed(run, sl, load_pack("saas-kinetic"), t, compose=fake_compose)
    assert bed == run.music_dir / "bed.wav"
    info = sf.info(str(bed))
    assert info.samplerate == 48000 and info.channels == 2 and info.subtype == "PCM_24"
    assert calls[0][1] == asked_ms(8.3 - 0.3 + 1.5)
    assert "120 BPM" in calls[0][0]  # the shot list's tempo wins over the pack's 92
    assert sf.info(str(bed)).duration == pytest.approx(9.5, abs=1e-3)       # the film's length plus the pack's tail
    assert (run.music_dir / "bed.src.mp3").read_bytes() == mp3
    grid = BeatGrid.model_validate_json((run.music_dir / "beats.json").read_text())
    assert grid.bpm == pytest.approx(120, rel=1e-3)
    assert grid.first_downbeat_s == pytest.approx(0.0, abs=1 / 60)     # bar 1 of the bed is bar 1 of the film
    assert (run.music_dir / "prompt.txt").read_text() == calls[0][0]
    conformed = json.loads((run.music_dir / "conform.json").read_text())
    assert conformed["provider"] == "elevenlabs" and conformed["stretch_ratio"] == pytest.approx(1.0, abs=2e-3)
    assert conformed["shift_s"] == pytest.approx(-0.25, abs=0.03)      # the clicks start a quarter second in
    assert conformed["max_error_s"] <= 1 / 60
    row = json.loads(run.costs.read_text().splitlines()[-1])
    assert row["provider"] == "elevenlabs" and row["usd"] == pytest.approx(12.07 / 60 * 0.15, abs=1e-4)
    assert row["cached"] is False
    music.make_bed(run, sl, load_pack("saas-kinetic"), t, compose=fake_compose)
    assert len(calls) == 1  # cached
    assert json.loads(run.costs.read_text().splitlines()[-1])["cached"] is True
    assert not [p for p in run.music_dir.iterdir() if p.name.startswith(".")]  # no temp files left behind


def _bed_mp3(tmp_path, gain: float) -> bytes:
    """A beat (clicks every half second at 120 BPM) over a sustained pad, scaled by `gain`."""
    sr, seconds = 44100, 10.0
    t = np.arange(int(seconds * sr)) / sr
    y = 0.25 * (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 660 * t))
    for i in range(int(seconds * 2)):
        t0 = int((0.25 + i * 0.5) * sr)
        k = np.arange(int(0.12 * sr)) / sr
        seg = (0.4 if i % 4 == 0 else 0.15) * np.sin(2 * np.pi * (60 if i % 4 == 0 else 2000) * k) * np.exp(-k * 30)
        y[t0:t0 + len(seg)] += seg[: max(0, min(len(seg), len(y) - t0))]
    wav, mp3 = tmp_path / f"bed{gain}.wav", tmp_path / f"bed{gain}.mp3"
    sf.write(wav, (y * gain).astype(np.float32), sr, subtype="FLOAT")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-ac", "2", "-b:a", "192k", str(mp3)], check=True)
    return mp3.read_bytes()


@pytest.mark.parametrize("gain", [1.0, 0.03])   # a hot bed (about -13 LUFS) and a quiet one (about -43 LUFS)
def test_make_bed_normalises_the_bed_to_the_packs_bed_lufs(tmp_path, gain):
    # The provider's delivery level must not decide how the music sits under the voice: bed.wav, the file the mix and
    # the critic both use, lands at the pack's bed_lufs with its arrangement's dynamics kept (one linear gain).
    from promo import audio_analysis as aa

    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    pack = load_pack("saas-kinetic")
    mp3 = _bed_mp3(tmp_path, gain)
    src_wav = tmp_path / "src.wav"
    (tmp_path / "src.mp3").write_bytes(mp3)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp_path / "src.mp3"), str(src_wav)], check=True)
    assert abs(aa.loudness(src_wav)["I"] - pack.music["bed_lufs"]) > 8           # far from the target to start with
    bed = music.make_bed(run, sl, pack, resolve(sl, Measured(), Limits()), compose=lambda p, ms: mp3)
    assert aa.loudness(bed)["I"] == pytest.approx(pack.music["bed_lufs"], abs=0.5)
    assert (run.music_dir / "beats.json").exists()
    assert not [p for p in run.music_dir.iterdir() if p.name.startswith(".")]   # no temp files left behind


def test_make_bed_uses_the_packs_tail_and_tempo_when_the_shot_list_has_none(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl(music={"beats_per_bar": 4})
    sl_timed = _sl()
    t = resolve(sl_timed, Measured(), Limits())
    pack = load_pack("saas-kinetic")
    pack = pack.model_copy(update={"music": {**pack.music, "tail_s": 0.0}})
    calls = []
    music.make_bed(run, sl, pack, t, compose=lambda p, ms: calls.append((p, ms)) or _click_mp3(tmp_path))
    assert calls[0][1] == asked_ms(8.0)       # the bed is conformed to the timing's 120 BPM grid
    assert "92 BPM" in calls[0][0]


_runs = itertools.count(1)


def _rolls(run) -> dict:
    return json.loads((run.music_dir / "roll.json").read_text())


def _bed_prompt(tmp_path, sl, provider="elevenlabs") -> str:
    """The prompt make_bed sends for `sl` under the saas-kinetic pack (a capturing compose; nothing is paid for)."""
    run = new_run(tmp_path, f"demo-{next(_runs)}", "saas-kinetic", "16:9", None, "en")
    t = resolve(_sl(), Measured(), Limits())
    calls = []
    music.make_bed(run, sl, load_pack("saas-kinetic"), t, provider=provider,
                   compose=lambda p, ms: calls.append(p) or _click_mp3(tmp_path))
    return calls[0]


def _styled(**style):
    data = json.loads(FIX.read_text())
    data["music"]["style"] = style
    return Shotlist.model_validate(data)


@pytest.mark.parametrize("provider", ["elevenlabs", "lyria"])
def test_a_film_style_overrides_the_packs_mood(tmp_path, provider):
    pack_mood = load_pack("saas-kinetic").music["mood"]
    prompt = _bed_prompt(tmp_path, _styled(mood="playful, bouncy, hand-made warmth"), provider)
    assert "playful, bouncy, hand-made warmth" in prompt
    assert pack_mood not in prompt
    assert load_pack("saas-kinetic").music["instrumentation"] in prompt     # the lines it did not set stay the pack's


def test_a_film_style_overrides_each_style_line(tmp_path):
    prompt = _bed_prompt(tmp_path, _styled(mood="M1", instrumentation="I1", arrangement="A1", negative="N1"))
    assert "Instrumental M1 track" in prompt and "Instrumentation: I1." in prompt
    assert "Arrangement: A1." in prompt and "Avoid: N1." in prompt
    pack = load_pack("saas-kinetic").music
    assert not any(pack[k] in prompt for k in ("mood", "instrumentation", "arrangement", "negative"))


def test_a_partial_style_keeps_the_packs_other_lines(tmp_path):
    pack = load_pack("saas-kinetic").music
    prompt = _bed_prompt(tmp_path, _styled(negative="harsh synths"))
    assert "Avoid: harsh synths." in prompt and pack["negative"] not in prompt
    assert pack["mood"] in prompt and pack["instrumentation"] in prompt and pack["arrangement"] in prompt


def test_an_unset_style_leaves_the_prompt_exactly_as_it_was(tmp_path):
    sl = _sl()
    assert sl.music.style is None
    t = resolve(sl, Measured(), Limits())
    asked_s = music._asked(9.5, 2.0)        # the film's 9.5 s with its stretch room and a bar, as the provider is asked
    expected = music.build_prompt(load_pack("saas-kinetic").music, music.sections_from_timing(t, sl), 120, asked_s)
    assert _bed_prompt(tmp_path, sl) == expected


def test_an_empty_style_block_is_the_same_cache_key_as_none(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    t = resolve(_sl(), Measured(), Limits())
    pack = load_pack("saas-kinetic")
    music.make_bed(run, _sl(), pack, t, compose=lambda p, ms: _click_mp3(tmp_path))

    def no_call(prompt, length_ms):
        raise AssertionError("an empty style block changed the prompt, so it missed the cache")

    music.make_bed(run, _styled(), pack, t, compose=no_call)


def test_a_style_change_is_a_new_cache_key(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    t = resolve(_sl(), Measured(), Limits())
    calls = []

    def compose(p, ms):
        calls.append(p)
        return _click_mp3(tmp_path)

    pack = load_pack("saas-kinetic")
    music.make_bed(run, _sl(), pack, t, compose=compose)
    music.make_bed(run, _sl(), pack, t, compose=compose)
    assert len(calls) == 1                                  # unchanged: cached
    music.make_bed(run, _styled(mood="playful"), pack, t, compose=compose)
    assert len(calls) == 2 and "playful" in calls[1]        # a style change re-rolls by itself


def test_make_bed_in_none_mode_does_nothing(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    pack = load_pack("saas-kinetic")
    pack = pack.model_copy(update={"music": {**pack.music, "mode": "none"}})

    def boom(prompt, length_ms):
        raise AssertionError("no music should be generated")

    assert music.make_bed(run, sl, pack, resolve(sl, Measured(), Limits()), compose=boom) is None
    assert not (run.music_dir / "bed.wav").exists() and not run.costs.exists()


def test_make_bed_logs_the_paid_call_even_when_the_audio_cannot_be_decoded(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    with pytest.raises(ToolError):
        music.make_bed(run, sl, load_pack("saas-kinetic"), resolve(sl, Measured(), Limits()),
                       compose=lambda p, ms: b"not audio")
    assert json.loads(run.costs.read_text().splitlines()[-1])["cached"] is False
    assert not (run.music_dir / "bed.wav").exists()
    assert not [p for p in run.music_dir.iterdir() if p.name.startswith(".")]


def test_a_bed_that_cannot_be_measured_leaves_the_previous_bed_and_grid_alone(tmp_path, monkeypatch):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    t = resolve(sl, Measured(), Limits())
    mp3 = _click_mp3(tmp_path)
    music.make_bed(run, sl, load_pack("saas-kinetic"), t, compose=lambda p, ms: mp3)
    kept = {n: (run.music_dir / n).read_bytes() for n in ("bed.wav", "beats.json")}
    report = json.loads((run.music_dir / "conform.json").read_text())

    def no_beats(path, beats_per_bar=4, **kw):
        raise ValueError(f"{path}: only 0 beats found; is this music?")

    monkeypatch.setattr("promo.conform.fit_steady_why", no_beats)
    with pytest.raises(music.MusicError, match="is this music"):
        music.make_bed(run, sl, load_pack("saas-kinetic"), t, compose=lambda p, ms: mp3, fresh=True)
    assert {n: (run.music_dir / n).read_bytes() for n in kept} == kept     # bed and grid are as they were
    after = json.loads((run.music_dir / "conform.json").read_text())
    assert {k: v for k, v in after.items() if k != "attempt"} == {k: v for k, v in report.items() if k != "attempt"}
    assert after["attempt"]["ok"] is False and after["attempt"]["segments"][0]["status"] == "failed"
    assert not [p for p in run.music_dir.iterdir() if p.name.startswith(".")]


def test_make_bed_measures_with_the_shot_lists_beats_per_bar(tmp_path, monkeypatch):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    sl = sl.model_copy(update={"music": sl.music.model_copy(update={"beats_per_bar": 3})})
    seen = {}
    from promo import conform
    real = conform.fit_steady_why

    def spy(path, beats_per_bar=4, **kw):
        seen["bpb"] = beats_per_bar
        return real(path, beats_per_bar, **kw)

    monkeypatch.setattr(conform, "fit_steady_why", spy)
    calls = []
    music.make_bed(run, sl, load_pack("saas-kinetic"), resolve(_sl(), Measured(), Limits()),
                   compose=lambda p, ms: calls.append(p) or _click_mp3(tmp_path))
    assert seen["bpb"] == 3 and "3/4" in calls[0]


def test_a_short_bed_is_billed_for_what_it_asks(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    data = json.loads(FIX.read_text())
    data["chapters"] = [{**data["chapters"][0], "bars": 1}]
    sl = Shotlist.model_validate(data)
    pack = load_pack("saas-kinetic")
    pack = pack.model_copy(update={"music": {**pack.music, "tail_s": 0.0}})
    t = resolve(sl, Measured(), Limits())
    assert t.duration_s - t.preroll_s == pytest.approx(2.0)
    calls = []
    music.make_bed(run, sl, pack, t, compose=lambda p, ms: calls.append(ms) or _click_mp3(tmp_path))
    assert calls == [asked_ms(2.0)]          # 4.12 s: a 2 s bed, its stretch room and a bar
    row = json.loads(run.costs.read_text().splitlines()[-1])
    assert row["units"]["music_s"] == 4.12 and row["usd"] == pytest.approx(4.12 / 60 * 0.15)


def test_fresh_moves_to_the_next_roll_and_a_plain_call_keeps_it(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    t = resolve(sl, Measured(), Limits())
    pack = load_pack("saas-kinetic")
    beds = [_click_mp3(tmp_path, bpm=b) for b in (120.0, 118.0, 122.5)]   # all within the 6% a stretch may take
    calls = []

    def compose(prompt, length_ms):
        calls.append(prompt)
        return beds[len(calls) - 1]

    src = run.music_dir / "bed.src.mp3"
    music.make_bed(run, sl, pack, t, compose=compose)
    assert len(calls) == 1 and src.read_bytes() == beds[0] and not (run.music_dir / "roll.json").exists()
    music.make_bed(run, sl, pack, t, compose=compose, fresh=True)
    assert len(calls) == 2 and src.read_bytes() == beds[1] and _rolls(run) == {"0": 1}
    music.make_bed(run, sl, pack, t, compose=compose)
    assert len(calls) == 2 and src.read_bytes() == beds[1]  # the re-rolled bed stays; no fall back to the first
    music.make_bed(run, sl, pack, t, compose=compose, fresh=True)
    assert len(calls) == 3 and src.read_bytes() == beds[2] and _rolls(run) == {"0": 2}
    assert calls[0] == calls[1] == calls[2]  # the prompt is the same; only the roll differs
    assert [json.loads(l)["cached"] for l in run.costs.read_text().splitlines()] == [False, False, True, False]


def test_a_failed_reroll_does_not_move_the_roll(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    t = resolve(sl, Measured(), Limits())
    pack = load_pack("saas-kinetic")

    def rejected(prompt, length_ms):
        raise ProviderError("elevenlabs", 401, "Invalid API key")

    with pytest.raises(ProviderError):
        music.make_bed(run, sl, pack, t, compose=rejected, fresh=True)
    assert not (run.music_dir / "roll.json").exists() and not (run.music_dir / "conform.json").exists()


# --- `promo music` --------------------------------------------------------------------------------------------

def _in_run(tmp_path, monkeypatch, sl_data=None):
    """A run folder with the fixture shot list and no timing.json: `promo music` plans from the shot list."""
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    run.shotlist.write_text(json.dumps(sl_data) if sl_data is not None else FIX.read_text())
    monkeypatch.chdir(run.root)
    return run


def _voice_mode_data(seconds=3.0):
    shot = {"id": "c01-s1", "events": [
        {"id": "c01-e1", "kind": "headline", "at": {"shot": "c01-s1", "edge": "start", "offset": 0.2}}]}
    if seconds is not None:
        shot["seconds"] = seconds
    return {"version": 2, "pack": "saas-kinetic", "language": "en", "timing_mode": "voice", "preroll_s": 0.3,
            "chapters": [{"id": "c01", "purpose": "Hook", "vo": [{"id": "c01-l1", "text": "Hello there world"}],
                          "shots": [shot]}]}


def test_music_rejects_an_unknown_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(app, ["music", "--provider", "suno"])
    assert res.exit_code == 2 and "elevenlabs or lyria" in res.output


@pytest.mark.parametrize("flag, field, chosen", [
    (None, None, "elevenlabs"), (None, "lyria", "lyria"), ("elevenlabs", "lyria", "elevenlabs"),
    ("lyria", None, "lyria"), ("lyria", "elevenlabs", "lyria")])
def test_music_provider_comes_from_the_flag_then_the_shot_list(tmp_path, monkeypatch, flag, field, chosen):
    data = json.loads(FIX.read_text())
    if field:
        data["music"]["provider"] = field
    run = _in_run(tmp_path, monkeypatch, data)
    seen = {}

    def fake(*a, **k):
        seen.update(k)
        return run.music_dir / "bed.wav"

    monkeypatch.setattr("promo.music.make_bed", fake)
    res = runner.invoke(app, ["music", *(["--provider", flag] if flag else [])])
    assert res.exit_code == 0, res.output
    assert seen["provider"] == chosen and f"({chosen})" in res.output


def test_music_help_names_both_providers():
    res = runner.invoke(app, ["music", "--help"])
    assert res.exit_code == 0
    assert "lyria" in res.output and "plan 3" not in res.output and "shotlist.json" in res.output


def _lyria_reply(audio: bytes):
    return httpx.Response(200, json={"status": "completed", "steps": [{"type": "model_output", "content": [
        {"type": "text", "text": "[[A0]]"},
        {"type": "audio", "mime_type": "audio/mpeg", "data": base64.b64encode(audio).decode()}]}]})


@respx.mock
def test_music_with_lyria_from_the_shot_list_makes_a_conformed_bed(tmp_path, monkeypatch):
    data = json.loads(FIX.read_text())
    data["music"]["provider"] = "lyria"
    run = _in_run(tmp_path, monkeypatch, data)
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    monkeypatch.setattr(music, "_words", lambda path: [])
    route = respx.post(lyria.INTERACTIONS_URL).mock(return_value=_lyria_reply(
        groove_mp3(tmp_path, 126.0, 16.0, first_beat_s=0.08)))     # Lyria plays 126 when asked for 120
    res = runner.invoke(app, ["music"])
    assert res.exit_code == 0, res.output
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "lyria-3.5" and body["store"] is False
    assert "Instrumental only, no vocals" in body["input"] and "exactly 120 BPM in 4/4" in body["input"]
    conformed = json.loads((run.music_dir / "conform.json").read_text())
    assert conformed["provider"] == "lyria" and conformed["stretch_ratio"] == pytest.approx(120 / 126, rel=1e-3)
    assert conformed["max_error_s"] <= 1 / 60 and conformed["vocals_words"] == 0
    grid = load_measured(run.root).grid
    assert grid.bpm == pytest.approx(120, rel=1e-3) and grid.first_downbeat_s == pytest.approx(0.0, abs=1 / 60)
    row = json.loads(run.costs.read_text().splitlines()[-1])
    assert (row["provider"], row["model"], row["usd"], row["cached"]) == ("lyria", "lyria-3.5", 0.08, False)
    assert runner.invoke(app, ["music"]).exit_code == 0 and route.call_count == 1      # cached


@respx.mock
def test_music_with_lyria_and_no_gemini_key_is_one_line_and_pays_nothing(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("promo.config.load_env", lambda path=None: None)
    monkeypatch.setattr("promo.cli.load_env", lambda path=None: None)
    route = respx.post(lyria.INTERACTIONS_URL).mock(return_value=_lyria_reply(b"x"))
    res = runner.invoke(app, ["music", "--provider", "lyria"])
    assert res.exit_code == 1 and "GEMINI_API_KEY is missing" in res.output
    assert res.output.count("promo:") == 1 and route.call_count == 0 and not run.costs.exists()


def test_music_outside_a_run_is_a_clean_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(app, ["music"])
    assert res.exit_code == 2 and "no run folder" in res.output


def test_music_with_a_broken_shotlist_is_one_line(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    run.shotlist.write_text("{not json")
    res = runner.invoke(app, ["music"])
    assert res.exit_code == 1 and res.output.count("promo:") == 1 and "Traceback" not in res.output


@respx.mock
def test_music_command_generates_the_bed_and_a_grid(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    route = respx.post(elevenlabs.MUSIC_URL).mock(return_value=httpx.Response(200, content=_click_mp3(tmp_path)))
    res = runner.invoke(app, ["music"])
    assert res.exit_code == 0, res.output
    assert str(run.music_dir / "bed.wav") in res.output and "promo timing" in res.output
    assert "grid     worst bar" in res.output and "limit 16.7 ms" in res.output and "worst beat" in res.output
    assert "sounding bars unmeasured" in res.output
    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content)["music_length_ms"] == asked_ms(9.5)
    assert (run.music_dir / "bed.wav").exists()
    assert load_measured(run.root).grid.bpm == pytest.approx(120, abs=3)


def test_music_in_none_mode_says_so(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch)
    monkeypatch.setattr("promo.music.make_bed", lambda *a, **k: None)
    res = runner.invoke(app, ["music"])
    assert res.exit_code == 0 and "'none'" in res.output


@pytest.mark.parametrize("exc, text", [
    (MissingKeyError("ELEVENLABS_API_KEY"), "ELEVENLABS_API_KEY is missing"),
    (ProviderError("elevenlabs", 401, "Invalid API key"), "HTTP 401"),
    (ToolError("command failed (1): ffmpeg -i bed.src.mp3"), "ffmpeg"),
    (music.MusicError("bed.wav: only 2 beats found; is this music?"), "is this music"),
    (TimingError("chapter c01 has no VO and no shot 'seconds'; cannot size it"), "cannot size it"),
    (OSError("No space left on device"), "No space left"),
])
def test_music_failures_are_one_line(tmp_path, monkeypatch, exc, text):
    _in_run(tmp_path, monkeypatch)

    def fake(*a, **k):
        raise exc

    monkeypatch.setattr("promo.music.make_bed", fake)
    res = runner.invoke(app, ["music"])
    assert res.exit_code == 1 and text in res.output
    assert res.output.count("promo:") == 1 and "Traceback" not in res.output


def test_music_lets_a_genuine_bug_propagate(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch)

    def fake(*a, **k):
        raise ValueError("a bug, not a user error")

    monkeypatch.setattr("promo.music.make_bed", fake)
    res = runner.invoke(app, ["music"])
    assert res.exit_code == 1 and isinstance(res.exception, ValueError)


@respx.mock
def test_music_rerun_after_timing_is_a_cache_hit_and_keeps_the_bed(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    route = respx.post(elevenlabs.MUSIC_URL).mock(return_value=httpx.Response(200, content=_click_mp3(tmp_path)))
    assert runner.invoke(app, ["music"]).exit_code == 0
    prompt = (run.music_dir / "prompt.txt").read_text()
    bed = (run.music_dir / "bed.wav").read_bytes()
    # `promo timing` snaps everything to the bed's measured grid, which the conform put on the plan's: nothing moves
    snapped = resolve(_sl(), load_measured(run.root), Limits())
    nominal = resolve(_sl(), Measured(), Limits())
    assert [(s.start_s, s.end_s) for s in snapped.shots] == [(s.start_s, s.end_s) for s in nominal.shots]
    assert [e.t for e in snapped.events] == pytest.approx([e.t for e in nominal.events], abs=1 / 60)
    write_timing(snapped, run.timing)
    res = runner.invoke(app, ["music"])
    assert res.exit_code == 0, res.output
    assert route.call_count == 1  # the plan ignores the snapped timing.json: same prompt, same key, no second charge
    assert (run.music_dir / "prompt.txt").read_text() == prompt
    assert (run.music_dir / "bed.wav").read_bytes() == bed
    assert [json.loads(l)["cached"] for l in run.costs.read_text().splitlines()] == [False, True]


@respx.mock
def test_music_fresh_rerolls_and_a_plain_rerun_keeps_the_new_bed(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    beds = [_click_mp3(tmp_path, bpm=b) for b in (120.0, 118.0, 122.5)]
    route = respx.post(elevenlabs.MUSIC_URL).mock(side_effect=[httpx.Response(200, content=b) for b in beds])
    src = run.music_dir / "bed.src.mp3"
    assert runner.invoke(app, ["music"]).exit_code == 0
    assert route.call_count == 1 and src.read_bytes() == beds[0]
    assert runner.invoke(app, ["music", "--fresh"]).exit_code == 0
    assert route.call_count == 2 and src.read_bytes() == beds[1]
    assert runner.invoke(app, ["music"]).exit_code == 0
    assert route.call_count == 2 and src.read_bytes() == beds[1]
    assert runner.invoke(app, ["music", "--fresh"]).exit_code == 0
    assert route.call_count == 3 and src.read_bytes() == beds[2]


def test_music_help_documents_fresh():
    res = runner.invoke(app, ["music", "--help"])
    assert res.exit_code == 0 and "--fresh" in res.output and "cached" in res.output


@respx.mock
def test_music_mode_without_bpm_plans_at_the_packs_tempo_and_needs_no_timing_json(tmp_path, monkeypatch):
    data = json.loads(FIX.read_text())
    data["music"] = {"beats_per_bar": 4}
    run = _in_run(tmp_path, monkeypatch, data)
    assert not run.timing.exists()
    route = respx.post(elevenlabs.MUSIC_URL).mock(return_value=httpx.Response(
        200, content=_click_mp3(tmp_path, seconds=16.0, bpm=92.0)))
    res = runner.invoke(app, ["music"])
    assert res.exit_code == 0, res.output
    assert "92 BPM" in (run.music_dir / "prompt.txt").read_text()  # saas-kinetic's tempo
    bar_s = 4 * 60 / 92
    expected_ms = asked_ms(2 * 2 * bar_s + 1.5, bpm=92)  # two chapters of two bars, plus the tail
    assert json.loads(route.calls[0].request.content)["music_length_ms"] == pytest.approx(expected_ms, abs=40)
    assert not run.timing.exists()  # music writes no timing.json
    assert runner.invoke(app, ["music"]).exit_code == 0 and route.call_count == 1


@respx.mock
def test_music_in_voice_mode_follows_the_real_takes_not_timing_json_estimates(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch, _voice_mode_data(seconds=3.0))
    route = respx.post(elevenlabs.MUSIC_URL).mock(return_value=httpx.Response(
        200, content=_click_mp3(tmp_path, seconds=14.0, bpm=92.0)))   # no music.bpm: the pack's 92 BPM
    res = runner.invoke(app, ["music"])
    assert res.exit_code == 0, res.output
    assert json.loads(route.calls[0].request.content)["music_length_ms"] == pytest.approx(asked_ms(3.0 + 1.5, bpm=92),
                                                                                          abs=40)
    run.vo_dir.mkdir(parents=True, exist_ok=True)
    take = LineTake(id="c01-l1", file="audio/vo/c01-l1.wav", duration_s=6.0)
    (run.vo_dir / "c01-l1.json").write_text(take.model_dump_json())
    assert runner.invoke(app, ["music"]).exit_code == 0
    # head pad 0.3 + 6.0 s take + tail pad 0.4, then the bed's own 1.5 s tail
    assert json.loads(route.calls[1].request.content)["music_length_ms"] == pytest.approx(asked_ms(6.7 + 1.5, bpm=92),
                                                                                          abs=80)
    assert route.call_count == 2


@respx.mock
def test_music_with_an_unresolvable_plan_is_one_line_and_pays_nothing(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch, _voice_mode_data(seconds=None))
    data = _voice_mode_data(seconds=None)
    data["chapters"][0]["vo"] = []  # no VO and no shot 'seconds': the chapter cannot be sized
    run.shotlist.write_text(json.dumps(data))
    route = respx.post(elevenlabs.MUSIC_URL).mock(return_value=httpx.Response(200, content=b"x"))
    res = runner.invoke(app, ["music"])
    assert res.exit_code == 1 and "cannot size it" in res.output
    assert res.output.count("promo:") == 1 and "Traceback" not in res.output
    assert route.call_count == 0


@respx.mock
def test_music_without_a_key_is_one_line_and_pays_nothing(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    _no_key(monkeypatch)
    route = respx.post(elevenlabs.MUSIC_URL).mock(return_value=httpx.Response(200, content=b"x"))
    res = runner.invoke(app, ["music"])
    assert res.exit_code == 1 and "ELEVENLABS_API_KEY is missing" in res.output
    assert res.output.count("promo:") == 1 and "Traceback" not in res.output
    assert route.call_count == 0 and not run.costs.exists()


@respx.mock
def test_music_with_a_rejected_key_is_one_line(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    route = respx.post(elevenlabs.MUSIC_URL).mock(return_value=httpx.Response(
        401, json={"detail": {"status": "invalid_api_key", "message": "Invalid API key"}}))
    res = runner.invoke(app, ["music"])
    assert res.exit_code == 1 and "elevenlabs: HTTP 401: Invalid API key" in res.output
    assert res.output.count("promo:") == 1 and "Traceback" not in res.output
    assert route.call_count == 1 and not (run.music_dir / "bed.wav").exists()


@pytest.mark.live
def test_live_ten_seconds_of_music(tmp_path):
    from promo.config import load_env
    load_env()
    audio = elevenlabs.compose("Instrumental upbeat electronic, 100 BPM, bright plucks, soft kick.", 10000)
    p = tmp_path / "m.mp3"
    p.write_bytes(audio)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True, check=True)
    assert float(out.stdout) == pytest.approx(10, abs=1.5)


# --- segments, prompts and the provider-generic bed -----------------------------------------------------------

def test_one_call_when_the_bed_fits():
    [seg] = music.plan_segments(9.5, 2.0, 600.0)
    assert (seg.audio_start_s, seg.seam_s, seg.next_seam_s, seg.audio_end_s) == (0.0, 0.0, 9.5, 9.5)
    assert seg.asked_s == pytest.approx(9.5 * 1.06 + 2.0)


@pytest.mark.parametrize("total, bar, n", [(263.5, 2.0, 2), (363.9, 2.4, 3)])   # the mascot film, the long film
def test_the_two_films_beds_split_into_bar_aligned_segments(total, bar, n):
    segs = music.plan_segments(total, bar, lyria.MAX_CALL_S)
    assert len(segs) == n and all(s.asked_s <= lyria.MAX_CALL_S for s in segs)
    assert segs[0].audio_start_s == 0.0 and segs[-1].audio_end_s == pytest.approx(total)
    for a, b in zip(segs, segs[1:]):
        assert b.seam_s == a.next_seam_s
        assert b.seam_s / bar == pytest.approx(round(b.seam_s / bar))          # on a bar line
        assert b.audio_start_s == pytest.approx(b.seam_s - 2 * bar)            # two bars of lead-in
        assert a.audio_end_s == pytest.approx(a.next_seam_s + bar)             # one bar past the seam


def test_a_seam_moves_onto_a_nearby_section_change():
    assert music.plan_segments(263.5, 2.0, 165.0, [130.0])[1].seam_s == 130.0   # 1 bar from the even split
    assert music.plan_segments(263.5, 2.0, 165.0, [150.0])[1].seam_s == 132.0   # 9 bars away: the even split


def test_a_bed_no_number_of_calls_can_make_is_a_music_error():
    with pytest.raises(music.MusicError, match="cannot be split"):
        music.plan_segments(100.0, 2.0, 5.0)


def test_segment_prompts_share_the_style_and_place_each_part():
    secs = [music.Section(id="c01", start_s=0.0, dur_s=130.0, note="Hook"),
            music.Section(id="c02", start_s=130.0, dur_s=132.0, note="Drop")]
    segs = music.plan_segments(263.5, 2.0, 165.0, [130.0])
    one, two = (music.build_segment_prompt(load_pack("saas-kinetic").music, secs, 120, 4, s, 263.5) for s in segs)

    def style(p):
        return [line for line in p.splitlines() if not line.startswith(("This is part", "Length", "["))]

    assert style(one) == style(two) and style(one)[0].startswith("Instrumental only, no vocals")
    assert "Tempo: exactly 120 BPM in 4/4, one steady tempo from start to end." in one
    assert "part 1 of 2" in one and "Do not end or fade out" in one
    assert "[0:00.0 - 2:10.0] c01: Hook" in one and "[2:10.0 - 2:12.0] c02: Drop" in one
    assert one.splitlines()[-1].endswith("Keep going; no ending.")
    assert "part 2 of 2" in two and "picking up at 2:06.0" in two and "no intro, no fade-in" in two
    keyed = music.build_segment_prompt(load_pack("saas-kinetic").music, secs, 120, 4, segs[1], 263.5, key="Ab major")
    assert "one continuous track in Ab major, picking up" in keyed and style(keyed) == style(one)
    assert "[0:00.0 - 0:04.0] c01 (continuing): Hook" in two and "[0:04.0 - 2:16.0] c02: Drop" in two
    assert two.splitlines()[-1].endswith("Clean ending; no new elements.")


def _lyria_bed(run, sl, t, compose, **kw):
    return music.make_bed(run, sl, load_pack("saas-kinetic"), t, compose=compose, provider="lyria",
                          transcribe=kw.pop("transcribe", lambda p: []), **kw)


def test_a_lyria_bed_is_conformed_cached_and_billed_per_call(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    t = resolve(sl, Measured(), Limits())
    mp3 = groove_mp3(tmp_path, 126.0, 16.0, first_beat_s=0.08)
    calls, heard = [], []

    def compose(prompt, length_ms):
        calls.append((prompt, length_ms))
        return mp3

    bed = _lyria_bed(run, sl, t, compose, transcribe=lambda p: heard.append(p) or ["la"])
    assert sf.info(str(bed)).duration == pytest.approx(9.5, abs=1e-3) and len(heard) == 1
    assert "Instrumental only, no vocals" in calls[0][0] and calls[0][1] == asked_ms(9.5)
    conformed = json.loads((run.music_dir / "conform.json").read_text())
    assert conformed["stretch_ratio"] == pytest.approx(120 / 126, rel=1e-3) and conformed["vocals_words"] == 1
    assert conformed["shift_s"] == pytest.approx(-0.08 * 126 / 120, abs=0.03)
    row = json.loads(run.costs.read_text().splitlines()[-1])
    assert (row["provider"], row["usd"], row["cached"]) == ("lyria", 0.08, False)
    assert list((tmp_path / "cache" / "lyria").rglob("*")) and not (tmp_path / "cache" / "elevenlabs").exists()
    _lyria_bed(run, sl, t, compose)
    assert len(calls) == 1 and json.loads(run.costs.read_text().splitlines()[-1])["cached"] is True


def test_a_lyria_bed_with_vocals_is_a_reroll_and_keeps_the_previous_bed(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    t = resolve(sl, Measured(), Limits())
    _lyria_bed(run, sl, t, lambda p, ms: groove_mp3(tmp_path, 120.0, 16.0))
    before = (run.music_dir / "bed.wav").read_bytes()
    with pytest.raises(music.MusicError, match=r"vocals detected .*4 words.*re-roll"):
        _lyria_bed(run, sl, t, lambda p, ms: groove_mp3(tmp_path, 121.0, 16.0), fresh=True,
                   transcribe=lambda p: ["we", "are", "the", "night"])
    assert (run.music_dir / "bed.wav").read_bytes() == before
    assert _rolls(run) == {"0": 1}        # the sung roll is spent: --fresh moves past it


def test_a_rerolled_bed_that_fails_the_conform_can_be_rerolled_again(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    t = resolve(sl, Measured(), Limits())
    beds = iter([groove_mp3(tmp_path, 120.0, 16.0), groove_mp3(tmp_path, 135.0, 16.0),
                 groove_mp3(tmp_path, 118.0, 16.0)])
    calls = []

    def compose(prompt, length_ms):
        calls.append(prompt)
        return next(beds)

    _lyria_bed(run, sl, t, compose)
    with pytest.raises(music.MusicError, match=r"135.00 BPM, 11.1% off the planned 120 BPM.*re-roll"):
        _lyria_bed(run, sl, t, compose, fresh=True)
    # roll 1 failed: roll 0, cached and passing, comes back for nothing before another call is paid for
    _lyria_bed(run, sl, t, compose, fresh=True)
    assert len(calls) == 2 and _attempt(run)[0]["source"] == "reused from roll 0"
    _lyria_bed(run, sl, t, compose, fresh=True)      # nothing failed this time: a new roll, 2, is asked for
    assert len(calls) == 3 and _rolls(run) == {"0": 2}
    assert json.loads((run.music_dir / "conform.json").read_text())["segments"][0]["measured_bpm"] == \
        pytest.approx(118, rel=1e-3)


def test_each_provider_caches_its_own_bed(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    t = resolve(sl, Measured(), Limits())
    pack = load_pack("saas-kinetic")
    mp3 = groove_mp3(tmp_path, 123.0, 16.0)
    calls = {"elevenlabs": 0, "lyria": 0}

    def counting(name):
        def compose(prompt, length_ms):
            calls[name] += 1
            return mp3
        return compose

    for _ in range(2):
        music.make_bed(run, sl, pack, t, compose=counting("elevenlabs"))
        assert json.loads((run.music_dir / "conform.json").read_text())["stretch_ratio"] == \
            pytest.approx(120 / 123, rel=1e-3)                     # an ElevenLabs bed is conformed too
        _lyria_bed(run, sl, t, counting("lyria"))
    assert calls == {"elevenlabs": 1, "lyria": 1}
    assert [json.loads(l)["provider"] for l in run.costs.read_text().splitlines()] == \
        ["elevenlabs", "lyria", "elevenlabs", "lyria"]


def _long_sl():
    data = json.loads(FIX.read_text())
    data["chapters"][0]["bars"] = 8
    data["chapters"][1]["bars"] = 8
    return Shotlist.model_validate(data)


def test_a_long_bed_comes_in_segments_joined_on_a_downbeat(tmp_path, monkeypatch):
    monkeypatch.setitem(music.PROVIDERS, "lyria", dataclasses.replace(music.PROVIDERS["lyria"], max_call_s=26.0))
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _long_sl()
    t = resolve(sl, Measured(), Limits())                   # 16 bars at 120 BPM: a 33.5 s bed with its tail
    calls = []

    def compose(prompt, length_ms):
        calls.append((prompt, length_ms))
        first = "part 1 of 2" in prompt                    # two calls: their own tempo and voicing each
        return groove_mp3(tmp_path, 123.0 if first else 118.0, length_ms / 1000, first_beat_s=0.1,
                          seed=1 if first else 2, chord=(220.0, 277.2, 329.6) if first else (164.8, 246.9, 370.0))

    bed = _lyria_bed(run, sl, t, compose)
    assert [("part 1 of 2" in p, "part 2 of 2" in p) for p, _ in calls] == [(True, False), (False, True)]
    assert "one continuous track in A major" in calls[1][0]      # part 1's key, heard and passed on
    assert sf.info(str(bed)).duration == pytest.approx(33.5, abs=1e-3)
    conformed = json.loads((run.music_dir / "conform.json").read_text())
    assert [s["src"] for s in conformed["segments"]] == ["bed.src.mp3", "bed.src.2.mp3"]
    assert [s["bed_start_s"] for s in conformed["segments"]] == [0.0, 12.0]
    [seam] = conformed["seams"]
    assert seam["seam_s"] == 16.0                          # the second chapter's first downbeat
    assert seam["level_dip_db"] >= -1.5 and seam["spectral_jump_db"] < 3.0
    assert conformed["max_error_s"] <= 1 / 60                # one grid runs straight through the seam
    grid = BeatGrid.model_validate_json((run.music_dir / "beats.json").read_text())
    assert grid.first_downbeat_s == pytest.approx(0.0, abs=1 / 60)
    assert min(abs(d - 16.0) for d in grid.downbeats) <= 1 / 60
    assert "--- segment 2 of 2 ---" in (run.music_dir / "prompt.txt").read_text()
    rows = [json.loads(l) for l in run.costs.read_text().splitlines()]
    assert [(r["units"]["segment"], r["usd"]) for r in rows] == [("1/2", 0.08), ("2/2", 0.08)]
    _lyria_bed(run, sl, t, compose)
    assert len(calls) == 2                                   # each segment is cached on its own


def test_a_seam_that_dips_or_jumps_is_flagged(tmp_path, monkeypatch):
    seams = [{"seam_s": 132.0, "level_dip_db": -2.4, "level_step_db": -3.0, "spectral_jump_db": 8.2},
             {"seam_s": 250.0, "level_dip_db": -0.4, "level_step_db": 0.1, "spectral_jump_db": 2.0}]
    warnings = music.seam_warnings(seams)
    assert len(warnings) == 2 and all("132.0s" in w for w in warnings)
    assert "dips 2.4 dB" in warnings[0] and "jumps 8.2 dB" in warnings[1] and "re-roll" in warnings[1]


def test_music_prints_the_beds_warnings(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)

    def fake(*a, **k):
        run.music_dir.mkdir(parents=True, exist_ok=True)
        (run.music_dir / "conform.json").write_text(json.dumps({"warnings": ["seam at 132.0s: listen to it"]}))
        return run.music_dir / "bed.wav"

    monkeypatch.setattr("promo.music.make_bed", fake)
    res = runner.invoke(app, ["music"])
    assert res.exit_code == 0 and "warning  seam at 132.0s: listen to it" in res.output


def test_every_later_segment_is_asked_for_the_first_ones_key(tmp_path, monkeypatch):
    monkeypatch.setitem(music.PROVIDERS, "lyria", dataclasses.replace(music.PROVIDERS["lyria"], max_call_s=19.0))
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _long_sl()
    t = resolve(sl, Measured(), Limits())
    prompts = []

    def compose(prompt, length_ms):
        prompts.append(prompt)
        i = len(prompts)
        return groove_mp3(tmp_path, (123.0, 118.0, 121.0)[i - 1], length_ms / 1000, first_beat_s=0.1, seed=i,
                          chord=((220.0, 277.2, 329.6), (164.8, 246.9, 370.0), (220.0, 277.2, 440.0))[i - 1])

    _lyria_bed(run, sl, t, compose)
    assert [f"part {i} of 3" in p for i, p in enumerate(prompts, 1)] == [True, True, True]
    assert "in A major" not in prompts[0]
    assert all("one continuous track in A major, picking up" in p for p in prompts[1:])
    assert len(json.loads((run.music_dir / "conform.json").read_text())["seams"]) == 2


def test_a_one_call_bed_is_judged_on_its_own_span_not_the_providers_ending(tmp_path):
    # Lyria plays on past the bed (59 s for a 46 s ask) and may slow down at its own ending; only the bed counts.
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    t = resolve(sl, Measured(), Limits())
    mp3 = groove_mp3(tmp_path, 120.0, 40.0, first_beat_s=0.05, bpm_at=lambda x: 120.0 if x < 20 else 100.0)
    _lyria_bed(run, sl, t, lambda p, ms: mp3)
    assert json.loads((run.music_dir / "conform.json").read_text())["max_error_s"] <= 1 / 60


# --- fix round 1: the per-bar gate, downbeats across seams, and the minor folds -----------------------------------

from promo import conform as cf  # noqa: E402


def _two_part_compose(tmp_path, **kw):
    def compose(prompt, length_ms):
        first = "part 1 of 2" in prompt
        return groove_mp3(tmp_path, 123.0 if first else 118.0, length_ms / 1000, first_beat_s=0.1,
                          seed=1 if first else 2, chord=(220.0, 277.2, 329.6) if first else (164.8, 246.9, 370.0), **kw)
    return compose


def _late_second_segment(monkeypatch, beats_late=1):
    """Segment 2 comes out of its conform with its downbeat a beat after the planned bar line, as a misread kick
    phase would leave it."""
    real = cf.conform

    def late(src, dst, **kw):
        placed = real(src, dst, **kw)
        if kw.get("what") == "segment 2 of 2":
            cf.shift_head(dst, beats_late * 0.5)
        return placed

    monkeypatch.setattr("promo.conform.conform", late)


def test_a_segment_a_beat_late_is_put_back_on_the_bar_line(tmp_path, monkeypatch):
    monkeypatch.setitem(music.PROVIDERS, "lyria", dataclasses.replace(music.PROVIDERS["lyria"], max_call_s=26.0))
    _late_second_segment(monkeypatch)
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _long_sl()
    _lyria_bed(run, sl, resolve(sl, Measured(), Limits()), _two_part_compose(tmp_path))
    conformed = json.loads((run.music_dir / "conform.json").read_text())
    second = conformed["segments"][1]["downbeat"]
    assert second["found_on_beat"] == 1 and second["moved_beats"] == -1 and second["margin"] >= cf.DOWNBEAT_MARGIN
    assert [d["found_on_beat"] for d in conformed["downbeat_check"]] == [0, 0]
    assert conformed["segments"][0]["downbeat"]["moved_beats"] == 0


def test_a_segment_left_a_beat_late_fails_the_downbeat_check(tmp_path, monkeypatch):
    monkeypatch.setitem(music.PROVIDERS, "lyria", dataclasses.replace(music.PROVIDERS["lyria"], max_call_s=26.0))
    _late_second_segment(monkeypatch)
    monkeypatch.setattr(music, "_align_downbeat", lambda path, **kw: (0, 0, 1.0))     # no correction
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _long_sl()
    with pytest.raises(music.MusicError, match=r"seam 1 \(at 16.0s\): the downbeat is off by 1 beat from the planned "
                                               r"bar lines; re-roll"):
        _lyria_bed(run, sl, resolve(sl, Measured(), Limits()), _two_part_compose(tmp_path))
    assert not (run.music_dir / "bed.wav").exists()


def test_a_downbeat_the_kick_cannot_decide_is_a_warning(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    _lyria_bed(run, sl, resolve(sl, Measured(), Limits()),
               lambda p, ms: groove_mp3(tmp_path, 120.0, 16.0, kick_every_beat=True))      # four on the floor
    conformed = json.loads((run.music_dir / "conform.json").read_text())
    assert conformed["segments"][0]["downbeat"]["margin"] < cf.DOWNBEAT_MARGIN
    assert any(w.startswith("bar 1 at 0.0s: the bed downbeat is a guess") for w in conformed["warnings"])


def test_the_grid_gate_names_the_worst_bar(tmp_path, monkeypatch):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    real = cf.check_grid
    monkeypatch.setattr("promo.conform.check_grid", lambda *a, **k: dataclasses.replace(
        real(*a, **k), max_error_s=0.040, worst_offset_s=-0.040, worst_bar=3, at_s=4.0))
    with pytest.raises(music.MusicError, match=r"bar 3 \(at 4.0s\) of the conformed bed is -40 ms off the planned grid "
                                               r"\(the median of its beats\), more than a frame at 60 fps; re-roll"):
        music.make_bed(run, sl, load_pack("saas-kinetic"), resolve(sl, Measured(), Limits()),
                       compose=lambda p, ms: _click_mp3(tmp_path))


def test_an_off_grid_bar_fails_with_its_offset(tmp_path, monkeypatch):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    real = cf.check_grid
    monkeypatch.setattr("promo.conform.check_grid", lambda *a, **k: dataclasses.replace(
        real(*a, **k), max_error_s=0.100, worst_offset_s=0.100, worst_bar=3, at_s=4.0, off_grid=[(3, 4)]))
    with pytest.raises(music.MusicError, match=r"bar 3 \(at 4.0s\) of the conformed bed is off the grid by \+100 ms: "
                                               r"its drums land outside the ±60 ms each beat is searched in"):
        music.make_bed(run, sl, load_pack("saas-kinetic"), resolve(sl, Measured(), Limits()),
                       compose=lambda p, ms: _click_mp3(tmp_path))


def test_grid_warnings_flag_a_stray_beat_and_too_many_unmeasured_bars(tmp_path):
    src = tmp_path / "g.wav"
    write_groove(src, 120.0, 24.0)
    check = dataclasses.replace(cf.check_grid(src, bpm=120, beats_per_bar=4), raw_max_error_s=0.039, raw_at_s=92.5,
                                bars_measured=9, bars_sounding=13, unmeasured=[(3, 4), (9, 9), (11, 11)])
    one, two = music.grid_warnings(check, 60)
    assert one.startswith("one beat at 92.5s is 39 ms off the grid, over a frame")
    assert two.startswith("4 of 13 sounding bars have too few clear beats") and "(bars 3-4, 9, 11)" in two
    fine = dataclasses.replace(check, raw_max_error_s=0.010, bars_measured=10)       # 3 of 13 is under a quarter
    assert music.grid_warnings(fine, 60) == []


def test_the_grid_numbers_and_the_section_shift_go_into_conform_json(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    _lyria_bed(run, sl, resolve(sl, Measured(), Limits()), lambda p, ms: groove_mp3(tmp_path, 126.0, 16.0))
    conformed = json.loads((run.music_dir / "conform.json").read_text())
    grid = conformed["grid"]
    assert grid["limit_s"] == pytest.approx(1 / 60, abs=1e-6) and grid["bar_max_error_s"] <= grid["limit_s"]
    assert {"worst_bar", "worst_bar_at_s", "raw_max_error_s", "raw_max_error_at_s", "bars_measured", "bars_sounding",
            "unmeasured_bars"} <= set(grid)
    assert conformed["segments"][0]["section_time_scale"] == pytest.approx(126 / 120, abs=1e-3)
    assert "x1.050 (a change the provider put at 1:00.0 of its audio lands at 1:03.0)" in conformed["sections_note"]


@pytest.mark.parametrize("planned", [False, True])
def test_seam_warnings_name_a_level_drop_and_label_a_planned_section_change(planned):
    seams = [{"seam_s": 132.0, "level_dip_db": -0.5, "level_step_db": -7.0, "spectral_jump_db": 8.2,
              "planned_change": planned}]
    step, jump = music.seam_warnings(seams)
    assert "the level drops 7.0 dB across it" in step and "re-roll if it sounds like a restart" in step
    assert ("(at a planned section change)" in step) is planned and ("(at a planned section change)" in jump) is planned
    assert ("(a restart?)" in jump) is not planned


def test_a_seam_on_a_section_change_is_marked_planned():
    segs = music.plan_segments(263.5, 2.0, 165.0, [130.0])
    assert segs[1].on_section and not music.plan_segments(263.5, 2.0, 165.0, [150.0])[1].on_section


def test_seams_count_bars_from_the_plans_bar_one():
    segs = music.plan_segments(263.5, 2.0, 165.0, origin_s=0.5)
    assert segs[1].seam_s == pytest.approx(132.5) and segs[1].audio_start_s == pytest.approx(128.5)
    assert (segs[1].seam_s - 0.5) / 2.0 == pytest.approx(round((segs[1].seam_s - 0.5) / 2.0))


def test_an_elevenlabs_bed_too_long_for_one_call_is_refused(tmp_path, monkeypatch):
    monkeypatch.setitem(music.PROVIDERS, "elevenlabs",
                        dataclasses.replace(music.PROVIDERS["elevenlabs"], max_call_s=20.0))
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _long_sl()

    def never(prompt, length_ms):
        raise AssertionError("no call for a bed that cannot be made in one")

    with pytest.raises(music.MusicError, match=r"elevenlabs makes a bed in one call of at most 20s.*--provider lyria"):
        music.make_bed(run, sl, load_pack("saas-kinetic"), resolve(sl, Measured(), Limits()), compose=never)


def test_a_delivery_that_falls_silent_before_the_film_ends_is_a_reroll(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _long_sl()                                          # a 32 s film
    mp3 = groove_mp3(tmp_path, 120.0, 24.0, tail_silence_s=16.0)     # 40 s long, but music for only 24 s of it
    with pytest.raises(music.MusicError, match=r"lyria returned 2\d\.\ds of music for the bed, which needs 32.0s"):
        _lyria_bed(run, sl, resolve(sl, Measured(), Limits()), lambda p, ms: mp3)


def test_a_failed_reroll_leaves_the_previous_sources_with_the_previous_bed(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    t = resolve(sl, Measured(), Limits())
    first = groove_mp3(tmp_path, 120.0, 16.0, seed=1)
    _lyria_bed(run, sl, t, lambda p, ms: first)
    with pytest.raises(music.MusicError, match="vocals detected"):
        _lyria_bed(run, sl, t, lambda p, ms: groove_mp3(tmp_path, 121.0, 16.0, seed=2), fresh=True,
                   transcribe=lambda p: ["la", "la", "la", "la"])
    assert (run.music_dir / "bed.src.mp3").read_bytes() == first
    assert not [p for p in run.music_dir.iterdir() if p.name.startswith(".")]


def test_estimate_key_is_none_for_audio_it_cannot_judge(tmp_path):
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros((4800, 2), np.float32), 48000)
    assert cf.estimate_key(silent) is None


def test_promo_beats_on_a_conformed_bed_keeps_the_plans_bar_one(tmp_path, monkeypatch):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    data = json.loads(run.shotlist.read_text())
    data["music"]["bpm"] = 120                                # the tempo the bed was made at
    run.shotlist.write_text(json.dumps(data))
    monkeypatch.chdir(run.root)
    # beat 0 of the file is the weak kick on beat 3; the strong kick (what analyze calls bar 1) is a second in
    from music_helpers import SR as GROOVE_SR, groove_samples
    y, _, _ = groove_samples(120.0, 20.0, first_beat_s=-1.0)
    run.music_dir.mkdir(parents=True, exist_ok=True)
    sf.write(run.music_dir / "bed.wav", y, GROOVE_SR, subtype="FLOAT")
    assert runner.invoke(app, ["beats"]).exit_code == 0
    assert BeatGrid.model_validate_json((run.music_dir / "beats.json").read_text()).first_downbeat_s == \
        pytest.approx(1.0, abs=0.01)
    (run.music_dir / "conform.json").write_text(json.dumps({"origin_s": 0.0}))
    res = runner.invoke(app, ["beats"])
    assert res.exit_code == 0, res.output
    grid = BeatGrid.model_validate_json((run.music_dir / "beats.json").read_text())
    assert grid.first_downbeat_s == pytest.approx(0.0, abs=0.01) and grid.downbeats[1] == pytest.approx(2.0, abs=0.01)
    import re
    assert "stays where the conform put it" in re.sub(r"[\s│]+", " ", runner.invoke(app, ["beats", "--help"]).output)


# --- R101: music prompts carry musical words only -----------------------------------------------------------------

PURPOSE = "The writing assistant: Ask AI; your coding agent; 14 agents; a tough reviewer persona: Reviewer #2"


@pytest.mark.parametrize("provider", ["elevenlabs", "lyria"])
def test_a_chapter_purpose_never_reaches_the_prompt(tmp_path, provider):
    data = json.loads(FIX.read_text())
    data["chapters"][0]["purpose"] = PURPOSE
    prompt = _bed_prompt(tmp_path, Shotlist.model_validate(data), provider)
    for word in ("Ask AI", "agent", "Reviewer", "writing assistant", "persona"):
        assert word not in prompt
    assert "continue the groove" in prompt


def test_the_notes_of_a_section_are_its_cues_with_the_denylist_taken_out():
    data = json.loads(FIX.read_text())
    data["chapters"][0]["shots"][0]["sound"] = {"music_cue": "Ask AI riser; Claude and GPT-4.5 agents hit; GitHub drop"}
    data["chapters"][1]["shots"][0]["sound"] = {"music_cue": "AI agent LLM Gemini"}
    sl = Shotlist.model_validate(data)
    one, two = (s.note for s in music.sections_from_timing(resolve(sl, Measured(), Limits()), sl))
    assert one == "Ask riser; and hit; drop"
    assert two == "continue the groove"                # nothing musical left once the denylist is out


@pytest.mark.parametrize("text, clean", [
    ("AI", ""), ("A.I. riser", "riser"), ("an AI-powered bass drop", "an bass drop"),
    ("coding agents and an agentic lift", "coding and an lift"), ("OpenAI, Anthropic and Mistral", "and"),
    ("ChatGPT-4o then gpt5.1 pad", "then pad"), ("build to the drop", "build to the drop"),
    ("said: Claude", "said"), ("a trail", "a trail"), ("Maine rain", "Maine rain"),
])
def test_clean_note_strips_whole_words_only(text, clean):
    assert music.clean_note(text) == clean


def test_the_denylist_is_documented_where_it_lives():
    assert {"AI", "agent", "agents", "Claude", "OpenAI", "GitHub"} <= set(music.NOTE_DENYLIST)
    assert "denylist" in music.clean_note.__doc__.lower() and "NOTE_DENYLIST" in music.clean_note.__doc__


# --- R101: per-segment rolls, per-segment re-roll, reuse from the cache -----------------------------------------------

A_MAJOR, E_CHORD, F_MAJOR = (220.0, 277.2, 329.6), (164.8, 246.9, 370.0), (174.6, 220.0, 261.6)


class Script:
    """A mocked 2-part compose: one queue of deliveries per part (each a function of the asked length in ms, so
    it can be a good groove or a bad one). Records which part each call was for."""

    def __init__(self, tmp_path, part1=(), part2=()):
        self.tmp, self.q, self.calls = tmp_path, {1: list(part1), 2: list(part2)}, []

    def __call__(self, prompt, length_ms):
        part = 1 if "part 1 of 2" in prompt else 2
        self.calls.append((part, prompt))
        return self.q[part].pop(0)(length_ms)

    def good(self, part, bpm=None, chord=A_MAJOR, seed=1):
        bpm = bpm or (123.0 if part == 1 else 118.0)
        return lambda ms: groove_mp3(self.tmp, bpm, ms / 1000, first_beat_s=0.1, seed=seed + part, chord=chord)

    def bad(self):      # 135 BPM: 12.5% off the 120 planned, over the 6% a stretch may take
        return lambda ms: groove_mp3(self.tmp, 135.0, ms / 1000, first_beat_s=0.1, seed=9)

    @property
    def parts(self):
        return [p for p, _ in self.calls]


def _two_calls(monkeypatch, max_call_s=26.0):
    monkeypatch.setitem(music.PROVIDERS, "lyria", dataclasses.replace(music.PROVIDERS["lyria"], max_call_s=max_call_s))


def _long_run(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _long_sl()
    return run, sl, resolve(sl, Measured(), Limits())


def _attempt(run) -> list[dict]:
    return json.loads((run.music_dir / "conform.json").read_text())["attempt"]["segments"]


def test_fresh_rerolls_only_the_segment_that_failed_its_own_checks(tmp_path, monkeypatch):
    _two_calls(monkeypatch)
    run, sl, t = _long_run(tmp_path)
    script = Script(tmp_path)
    script.q = {1: [script.good(1)], 2: [script.bad(), script.good(2)]}
    with pytest.raises(music.MusicError, match=r"segment 2 of 2 .*135.00 BPM, 11.1% off the planned 120 BPM.*re-roll"):
        _lyria_bed(run, sl, t, script)
    assert script.parts == [1, 2] and not (run.music_dir / "bed.wav").exists()
    one, two = _attempt(run)
    assert (one["status"], one["roll"], one["source"]) == ("passed", 0, "fresh")
    assert (two["status"], two["roll"]) == ("failed", 0) and "135.00 BPM" in two["reason"]
    assert not (run.music_dir / "roll.json").exists()         # nothing has been re-rolled yet

    bed = _lyria_bed(run, sl, t, script, fresh=True)
    assert script.parts == [1, 2, 2]                       # segment 1 passed: it is not asked for again
    assert _rolls(run) == {"1": 1}
    one, two = _attempt(run)
    assert (one["status"], one["roll"], one["source"]) == ("passed", 0, "cached")
    assert (two["status"], two["roll"], two["source"]) == ("passed", 1, "fresh")
    assert sf.info(str(bed)).duration == pytest.approx(33.5, abs=1e-3)
    rows = [json.loads(l) for l in run.costs.read_text().splitlines()]
    assert [(r["units"]["segment"], r["cached"]) for r in rows] == [("1/2", False), ("2/2", False), ("1/2", True),
                                                                      ("2/2", False)]
    report = json.loads((run.music_dir / "conform.json").read_text())
    assert [(s["roll"], s["source"], s["passed"]) for s in report["segments"]] == [(0, "cached", True), (1, "fresh", True)]
    assert report["attempt"]["ok"] is True


def test_fresh_with_no_recorded_attempt_rerolls_every_segment(tmp_path, monkeypatch):
    _two_calls(monkeypatch)
    run, sl, t = _long_run(tmp_path)
    script = Script(tmp_path)
    script.q = {1: [script.good(1), script.good(1, 121.0)], 2: [script.good(2), script.good(2, 119.0)]}
    _lyria_bed(run, sl, t, script)
    (run.music_dir / "conform.json").unlink()               # nothing recorded
    _lyria_bed(run, sl, t, script, fresh=True)
    assert script.parts == [1, 2, 1, 2] and _rolls(run) == {"0": 1, "1": 1}


def test_fresh_after_a_passing_bed_rerolls_every_segment(tmp_path, monkeypatch):
    _two_calls(monkeypatch)
    run, sl, t = _long_run(tmp_path)
    script = Script(tmp_path)
    script.q = {1: [script.good(1), script.good(1, 121.0)], 2: [script.good(2), script.good(2, 119.0)]}
    _lyria_bed(run, sl, t, script)
    _lyria_bed(run, sl, t, script, fresh=True)
    assert script.parts == [1, 2, 1, 2] and _rolls(run) == {"0": 1, "1": 1}


def _three_states(tmp_path, monkeypatch, part1_roll1_chord):
    """Roll 0 of both parts passes; at roll 1 part 1 passes (in `part1_roll1_chord`) and part 2 fails. Returns the
    run, the script and what is needed for the next `--fresh`."""
    _two_calls(monkeypatch)
    run, sl, t = _long_run(tmp_path)
    script = Script(tmp_path)
    script.q = {1: [script.good(1), script.good(1, 121.0, chord=part1_roll1_chord, seed=5)],
                2: [script.good(2), script.bad()]}
    _lyria_bed(run, sl, t, script)
    with pytest.raises(music.MusicError, match="segment 2 of 2"):
        _lyria_bed(run, sl, t, script, fresh=True)
    assert script.parts == [1, 2, 1, 2]
    assert [s["status"] for s in _attempt(run)] == ["passed", "failed"] and _rolls(run) == {"0": 1, "1": 1}
    return run, sl, t, script


def test_a_failed_segment_is_replaced_by_a_cached_one_asked_for_the_same_key(tmp_path, monkeypatch):
    run, sl, t, script = _three_states(tmp_path, monkeypatch, A_MAJOR)     # part 1 is in A major at both rolls
    assert "in A major" in script.calls[1][1] and "in A major" in script.calls[3][1]
    before = len(script.calls)
    bed = _lyria_bed(run, sl, t, script, fresh=True)
    assert len(script.calls) == before                                    # nothing paid: roll 0 of part 2 passes
    one, two = _attempt(run)
    assert (one["roll"], one["source"]) == (1, "cached") and (two["roll"], two["source"]) == (0, "reused from roll 0")
    assert two["status"] == "passed" and sf.info(str(bed)).duration == pytest.approx(33.5, abs=1e-3)
    assert _rolls(run) == {"0": 1, "1": 1}
    rows = [json.loads(l) for l in run.costs.read_text().splitlines()][-2:]
    assert [(r["units"]["segment"], r["cached"], r["units"].get("roll")) for r in rows] == \
        [("1/2", True, 1), ("2/2", True, 0)]


def test_a_cached_segment_asked_for_another_key_is_not_reused(tmp_path, monkeypatch):
    run, sl, t, script = _three_states(tmp_path, monkeypatch, F_MAJOR)     # part 1 came back in F at roll 1
    key_of = lambda prompt: re.search(r"one continuous track in (\w+ \w+)", prompt).group(1)
    assert key_of(script.calls[1][1]) == "A major" and key_of(script.calls[3][1]) != "A major"
    script.q[2] = [script.good(2, chord=F_MAJOR, seed=7)]
    before = len(script.calls)
    _lyria_bed(run, sl, t, script, fresh=True)
    assert script.parts[before:] == [2]                                    # roll 0 of part 2 asked for A: not used
    assert key_of(script.calls[-1][1]) == key_of(script.calls[3][1])
    one, two = _attempt(run)
    assert (two["roll"], two["source"]) == (2, "fresh") and _rolls(run) == {"0": 1, "1": 2}


@pytest.mark.parametrize("with_cues", [True, False])
def test_a_cached_delivery_asked_with_a_purpose_in_its_notes_is_still_found(tmp_path, with_cues):
    # Before R101 the notes carried the chapter purposes (and the cues, if the shot list had any), so the paid
    # deliveries are cached under those prompts. Nothing of them is sent again.
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    data = json.loads(FIX.read_text())
    data["chapters"][0]["shots"][0]["sound"] = {"music_cue": "filter opens"}
    sl = Shotlist.model_validate(data)
    t = resolve(sl, Measured(), Limits())
    pack = load_pack("saas-kinetic")
    total = round(t.duration_s - t.preroll_s + pack.music["tail_s"], 6)
    [seg] = music.plan_segments(total, 2.0, lyria.MAX_CALL_S)
    old = music.build_segment_prompt(pack.music, music.sections_from_timing(t, sl, purposes=True, cues=with_cues),
                                     120, 4, seg, total)
    new = music.build_segment_prompt(pack.music, music.sections_from_timing(t, sl), 120, 4, seg, total)
    assert "Hook" in old and ("filter opens" in old) is with_cues and "Hook" not in new
    key = music.segment_cache_key(music.PROVIDERS["lyria"], old, round(seg.asked_s * 1000), 0, 0, 1)
    from promo.providers import cache
    cache.cache_put("lyria", key, groove_mp3(tmp_path, 122.0, 16.0, first_beat_s=0.1))

    def never(prompt, length_ms):
        raise AssertionError("the paid delivery was in the cache")

    _lyria_bed(run, sl, t, never)
    [seg_row] = _attempt(run)
    assert (seg_row["source"], seg_row["usd"]) == ("cached", 0.0)
    assert (run.music_dir / "prompt.txt").read_text() == old               # the prompt the audio was asked with


def test_a_legacy_roll_txt_becomes_roll_json_with_every_segment_at_that_roll(tmp_path, monkeypatch):
    _two_calls(monkeypatch)
    run, sl, t = _long_run(tmp_path)
    run.music_dir.mkdir(parents=True, exist_ok=True)
    (run.music_dir / "roll.txt").write_text("2")
    script = Script(tmp_path)
    script.q = {1: [script.bad()], 2: []}
    with pytest.raises(music.MusicError, match="segment 1 of 2"):
        _lyria_bed(run, sl, t, script)
    assert _rolls(run) == {"0": 2, "1": 2} and not (run.music_dir / "roll.txt").exists()
    from promo.providers import cache
    prov = music.PROVIDERS["lyria"]
    first = _plan(sl, t)[0]
    asked = prov.clamp_ms(first.asked_s * 1000)
    assert cache.cache_get("lyria", music.segment_cache_key(prov, script.calls[0][1], asked, 2, 0, 2))
    assert not cache.cache_get("lyria", music.segment_cache_key(prov, script.calls[0][1], asked, 0, 0, 2))


def _plan(sl, t):
    """The segments make_bed plans for `sl` (the saas-kinetic pack, 120 BPM in 4/4)."""
    pack = load_pack("saas-kinetic")
    total = round(t.duration_s - t.preroll_s + pack.music["tail_s"], 6)
    origin = round(t.beats[0] - t.preroll_s, 6) if t.beats else 0.0
    starts = [s.start_s for s in music.sections_from_timing(t, sl)]
    return music.plan_segments(total, 2.0, music.PROVIDERS["lyria"].max_call_s, starts, origin)


def test_a_provider_error_is_not_a_failed_segment_and_fresh_carries_on(tmp_path, monkeypatch):
    _two_calls(monkeypatch)
    run, sl, t = _long_run(tmp_path)
    script = Script(tmp_path)

    def blocked(prompt, length_ms):
        if "part 2 of 2" in prompt:
            raise ProviderError("lyria", 400, "Request blocked for an unspecified policy reason")
        return script.good(1)(length_ms)

    with pytest.raises(ProviderError, match="blocked"):
        _lyria_bed(run, sl, t, blocked)
    one, two = _attempt(run)
    assert (one["status"], two["status"]) == ("passed", "error") and "blocked" in two["reason"]
    script.q = {1: [], 2: [script.good(2)]}
    _lyria_bed(run, sl, t, script, fresh=True)           # nothing failed its checks: this carries on, part 1 stays
    assert script.parts == [2] and not (run.music_dir / "roll.json").exists()
    assert [s["source"] for s in _attempt(run)] == ["cached", "fresh"]


def test_a_first_segment_that_fails_stops_before_paying_for_the_rest(tmp_path, monkeypatch):
    _two_calls(monkeypatch)
    run, sl, t = _long_run(tmp_path)
    script = Script(tmp_path)
    script.q = {1: [script.bad()], 2: []}
    with pytest.raises(music.MusicError, match="segment 1 of 2"):
        _lyria_bed(run, sl, t, script)
    assert script.parts == [1]
    assert [s["status"] for s in _attempt(run)] == ["failed", "pending"]


def test_the_grid_gate_blames_the_segment_it_fell_in(tmp_path, monkeypatch):
    _two_calls(monkeypatch)
    run, sl, t = _long_run(tmp_path)
    real = cf.check_grid
    monkeypatch.setattr("promo.conform.check_grid", lambda *a, **k: dataclasses.replace(
        real(*a, **k), max_error_s=0.040, worst_offset_s=0.040, worst_bar=9, at_s=24.0))
    script = Script(tmp_path)
    script.q = {1: [script.good(1)], 2: [script.good(2)]}
    with pytest.raises(music.MusicError, match=r"bar 9 \(at 24.0s\)"):
        _lyria_bed(run, sl, t, script)
    one, two = _attempt(run)
    assert (one["status"], two["status"]) == ("passed", "failed") and "bar 9" in two["reason"]
    n = len(script.calls)
    script.q = {1: [], 2: [script.good(2, 119.0)]}
    monkeypatch.setattr("promo.conform.check_grid", real)
    _lyria_bed(run, sl, t, script, fresh=True)                   # only the blamed segment is asked again
    assert script.parts[n:] == [2]


def test_every_segment_with_drums_off_the_grid_is_blamed_not_only_the_worst_bars(tmp_path, monkeypatch):
    _two_calls(monkeypatch)
    run, sl, t = _long_run(tmp_path)
    real = cf.check_grid
    monkeypatch.setattr("promo.conform.check_grid", lambda *a, **k: dataclasses.replace(
        real(*a, **k), max_error_s=0.250, worst_offset_s=-0.250, worst_bar=3, at_s=4.0, off_grid=[(3, 4), (10, 11)]))
    script = Script(tmp_path)
    script.q = {1: [script.good(1), script.good(1, 121.0)], 2: [script.good(2), script.good(2, 119.0)]}
    with pytest.raises(music.MusicError, match=r"bar 3 \(at 4.0s\) .*off the grid by -250 ms.*\(segments 1, 2 are off the grid\)"):
        _lyria_bed(run, sl, t, script)
    one, two = _attempt(run)
    assert (one["status"], two["status"]) == ("failed", "failed") and "bars 10-11 (at 18.0s)" in two["reason"]
    monkeypatch.setattr("promo.conform.check_grid", real)
    n = len(script.calls)
    _lyria_bed(run, sl, t, script, fresh=True)                     # both are asked for again, in one run
    assert script.parts[n:] == [1, 2]


def test_a_downbeat_off_at_a_seam_blames_the_segment_after_it(tmp_path, monkeypatch):
    _two_calls(monkeypatch)
    _late_second_segment(monkeypatch)
    monkeypatch.setattr(music, "_align_downbeat", lambda path, **kw: (0, 0, 1.0))
    run, sl, t = _long_run(tmp_path)
    script = Script(tmp_path)
    script.q = {1: [script.good(1)], 2: [script.good(2)]}
    with pytest.raises(music.MusicError, match=r"seam 1 \(at 16.0s\): the downbeat is off by 1 beat"):
        _lyria_bed(run, sl, t, script)
    assert [s["status"] for s in _attempt(run)] == ["passed", "failed"]


# --- R101: the steadiness tolerance follows the frame -------------------------------------------------------------

def _wander_bed(tmp_path, ms):
    """A 120 BPM groove whose beats swing `ms` either side of a straight grid, a slow wave 16 s long."""
    return groove_mp3(tmp_path, 120.0, 24.0, first_beat_s=0.1, seed=3,
                      offset_at=lambda k: (ms / 1000) * math.sin(2 * math.pi * k / 32))


def test_make_bed_passes_one_frame_as_the_steadiness_tolerance(tmp_path, monkeypatch):
    from promo import conform

    seen = []
    real = conform.fit_steady_why

    def spy(path, beats_per_bar=4, **kw):
        seen.append(kw.get("steady_tol_s"))
        return real(path, beats_per_bar, **kw)

    monkeypatch.setattr(conform, "fit_steady_why", spy)
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    _lyria_bed(run, sl, resolve(sl, Measured(), Limits()), lambda p, ms: groove_mp3(tmp_path, 120.0, 16.0))
    assert seen and all(x == pytest.approx(1 / 60) for x in seen)


def test_a_sub_frame_wander_still_makes_a_bed(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    _lyria_bed(run, sl, resolve(sl, Measured(), Limits()), lambda p, ms: _wander_bed(tmp_path, 14))   # reads 11.5 ms
    assert (run.music_dir / "bed.wav").exists()                                  # over 10 ms, under a frame


def test_a_wander_over_a_frame_still_fails(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    with pytest.raises(music.MusicError, match=r"no steady beat .*wander \d+ ms|\d+ ms off the planned grid"):
        _lyria_bed(run, sl, resolve(sl, Measured(), Limits()), lambda p, ms: _wander_bed(tmp_path, 22))
    assert not (run.music_dir / "bed.wav").exists()


# --- R101: the per-segment table ----------------------------------------------------------------------------------

ROWS = [
    {"segment": "1/3", "index": 0, "roll": 2, "source": "reused from roll 1", "status": "passed", "reason": "",
     "usd": 0.0, "tried": [{"roll": 2, "reason": "segment 1 of 3 has no steady beat"}]},
    {"segment": "2/3", "index": 1, "roll": 3, "source": "fresh", "status": "failed",
     "reason": "segment 2 of 3 has no steady beat to fit to the 120 BPM grid; re-roll it with `promo music --fresh`",
     "usd": 0.08, "tried": []},
    {"segment": "3/3", "index": 2, "roll": 0, "source": "", "status": "pending", "reason": "", "usd": 0.0, "tried": []},
]


def test_the_segment_table_names_roll_source_result_reason_and_cost():
    lines = music.segment_table(ROWS)
    assert lines[0].split()[:5] == ["segment", "roll", "source", "result", "cost"]
    one, two, three = lines[1:4]
    assert one.split()[:2] == ["1/3", "2"] and "reused from roll 1" in one and " pass " in one and "$0.00" in one
    assert "roll 2 failed: segment 1 of 3 has no steady beat" in one
    assert two.split()[:2] == ["2/3", "3"] and "fresh" in two and "FAIL" in two and "$0.08" in two
    assert "has no steady beat to fit to the 120 BPM grid" in two and "re-roll it with" not in two
    assert "not reached" in three
    assert lines[-1].startswith("cost") and "$0.08" in lines[-1]


@respx.mock
def test_promo_music_prints_the_table_on_every_run(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    route = respx.post(elevenlabs.MUSIC_URL).mock(return_value=httpx.Response(200, content=_click_mp3(tmp_path)))
    res = runner.invoke(app, ["music"])
    assert res.exit_code == 0, res.output
    first = [l for l in res.output.splitlines() if l.startswith("1/1")]
    assert len(first) == 1 and "fresh" in first[0] and "pass" in first[0] and "$0.03" in first[0]
    assert "segment  roll  source" in res.output
    again = runner.invoke(app, ["music"])
    assert route.call_count == 1 and "cached" in [l for l in again.output.splitlines() if l.startswith("1/1")][0]
    assert "$0.00" in again.output


@respx.mock
def test_promo_music_prints_the_table_and_one_error_line_when_a_segment_fails(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    respx.post(elevenlabs.MUSIC_URL).mock(return_value=httpx.Response(200, content=_click_mp3(tmp_path, bpm=135.0)))
    res = runner.invoke(app, ["music"])
    assert res.exit_code == 1 and res.output.count("promo:") == 1
    row = [l for l in res.output.splitlines() if l.startswith("1/1")][0]
    assert "FAIL" in row and "135.00 BPM" in row


def test_promo_music_help_says_what_fresh_rerolls():
    import re

    text = re.sub(r"[\s│]+", " ", runner.invoke(app, ["music", "--help"]).output)
    assert "only the segments that failed" in text and "all of them" in text


# --- R103: `promo music --fresh --segment N` ------------------------------------------------------------------------


def _passing_bed(tmp_path, monkeypatch):
    """A 2-segment bed that has passed once (both parts at roll 0, part 1 in A major). Returns what a re-roll needs."""
    _two_calls(monkeypatch)
    run, sl, t = _long_run(tmp_path)
    script = Script(tmp_path)
    script.q = {1: [script.good(1)], 2: [script.good(2)]}
    _lyria_bed(run, sl, t, script)
    assert script.parts == [1, 2] and not (run.music_dir / "roll.json").exists()
    return run, sl, t, script


def test_fresh_segment_rerolls_the_named_segment_that_passed_and_keeps_the_other_cached(tmp_path, monkeypatch):
    run, sl, t, script = _passing_bed(tmp_path, monkeypatch)
    script.q = {1: [], 2: [script.good(2, 119.0, seed=4)]}
    bed = _lyria_bed(run, sl, t, script, fresh=True, segments=[2])
    assert script.parts == [1, 2, 2]                       # part 1 passed and was not named: no new call for it
    assert _rolls(run) == {"1": 1}                         # roll.json: only the named segment moved
    one, two = _attempt(run)
    assert (one["status"], one["roll"], one["source"], one["usd"]) == ("passed", 0, "cached", 0.0)
    assert (two["status"], two["roll"], two["source"], two["usd"]) == ("passed", 1, "fresh", 0.08)
    assert sf.info(str(bed)).duration == pytest.approx(33.5, abs=1e-3)
    rows = [json.loads(l) for l in run.costs.read_text().splitlines()][2:]
    assert [(r["units"]["segment"], r["units"]["roll"], r["cached"]) for r in rows] == [("1/2", 0, True),
                                                                                       ("2/2", 1, False)]
    report = json.loads((run.music_dir / "conform.json").read_text())
    assert [(s["roll"], s["source"]) for s in report["segments"]] == [(0, "cached"), (1, "fresh")]
    assert report["attempt"]["re_rolled"] == [1] and report["attempt"]["fresh"] is True
    _lyria_bed(run, sl, t, script)                         # a plain run keeps the new roll and pays nothing
    assert script.parts == [1, 2, 2] and _rolls(run) == {"1": 1}


def test_fresh_segment_one_leaves_the_later_segment_on_its_cached_roll(tmp_path, monkeypatch):
    run, sl, t, script = _passing_bed(tmp_path, monkeypatch)
    script.q = {1: [script.good(1, 121.0, seed=5)], 2: []}          # the same key (A major): part 2's prompt is unchanged
    _lyria_bed(run, sl, t, script, fresh=True, segments=[1])
    assert script.parts == [1, 2, 1] and _rolls(run) == {"0": 1}
    one, two = _attempt(run)
    assert (one["roll"], one["source"]) == (1, "fresh") and (two["roll"], two["source"]) == (0, "cached")


def test_fresh_segment_re_rolls_every_named_segment_and_only_those(tmp_path, monkeypatch):
    monkeypatch.setitem(music.PROVIDERS, "lyria", dataclasses.replace(music.PROVIDERS["lyria"], max_call_s=19.0))
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _long_sl()
    t = resolve(sl, Measured(), Limits())
    calls = []
    chords = {1: (220.0, 277.2, 329.6), 2: (164.8, 246.9, 370.0), 3: (220.0, 277.2, 440.0)}

    def compose(prompt, length_ms):
        part = int(re.search(r"part (\d) of 3", prompt).group(1))
        calls.append(part)
        return groove_mp3(tmp_path, (123.0, 118.0, 121.0)[part - 1] + 0.5 * calls.count(part), length_ms / 1000,
                          first_beat_s=0.1, seed=part + 10 * calls.count(part), chord=chords[part])

    _lyria_bed(run, sl, t, compose)
    assert calls == [1, 2, 3]
    _lyria_bed(run, sl, t, compose, fresh=True, segments=[3, 1])     # in any order
    assert sorted(calls[3:]) == [1, 3] and _rolls(run) == {"0": 1, "2": 1}
    rows = _attempt(run)
    assert [(r["roll"], r["source"]) for r in rows] == [(1, "fresh"), (0, "cached"), (1, "fresh")]


def test_fresh_segment_asks_for_a_new_roll_even_when_a_cached_one_would_have_passed(tmp_path, monkeypatch):
    # Plain `--fresh` after _three_states takes part 2's roll 0 back from the cache for free (see the R101 test above);
    # naming the segment is a request for a new draw, so it pays for one.
    run, sl, t, script = _three_states(tmp_path, monkeypatch, A_MAJOR)
    script.q[2] = [script.good(2, 119.0, seed=8)]
    before = len(script.calls)
    _lyria_bed(run, sl, t, script, fresh=True, segments=[2])
    assert script.parts[before:] == [2]
    one, two = _attempt(run)
    assert (one["roll"], one["source"]) == (1, "cached") and (two["roll"], two["source"]) == (2, "fresh")
    assert _rolls(run) == {"0": 1, "1": 2}


def test_fresh_segment_that_is_out_of_range_is_refused_before_anything_is_paid(tmp_path, monkeypatch):
    run, sl, t, script = _passing_bed(tmp_path, monkeypatch)
    with pytest.raises(music.SegmentError, match=r"--segment 3: this bed has 2 segments, numbered 1 to 2"):
        _lyria_bed(run, sl, t, script, fresh=True, segments=[3])
    with pytest.raises(music.SegmentError, match=r"--segment 0, 5: this bed has 2 segments"):
        _lyria_bed(run, sl, t, script, fresh=True, segments=[5, 0])
    assert script.parts == [1, 2] and len(run.costs.read_text().splitlines()) == 2 and not (run.music_dir / "roll.json").exists()
    assert isinstance(music.SegmentError("x"), music.MusicError)


def test_fresh_segment_on_a_one_segment_bed_says_so(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    sl = _sl()
    with pytest.raises(music.SegmentError, match=r"--segment 2: this bed is one segment \(--segment 1\)"):
        _lyria_bed(run, sl, resolve(sl, Measured(), Limits()), lambda p, ms: pytest.fail("paid"), fresh=True,
                   segments=[2])


def test_segments_without_fresh_are_refused(tmp_path, monkeypatch):
    run, sl, t, script = _passing_bed(tmp_path, monkeypatch)
    with pytest.raises(music.SegmentError, match="--segment needs --fresh"):
        _lyria_bed(run, sl, t, script, segments=[2])
    assert script.parts == [1, 2]


def test_promo_music_segment_without_fresh_is_exit_two(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(app, ["music", "--segment", "2"])
    assert res.exit_code == 2 and "--segment needs --fresh" in res.output


@pytest.mark.parametrize("n", ["0", "-1"])
def test_promo_music_segment_counts_from_one(tmp_path, monkeypatch, n):
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(app, ["music", "--fresh", "--segment", n])
    assert res.exit_code == 2 and "count from 1" in res.output


@respx.mock
def test_promo_music_segment_out_of_range_is_exit_two_and_pays_nothing(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    route = respx.post(elevenlabs.MUSIC_URL).mock(return_value=httpx.Response(200, content=_click_mp3(tmp_path)))
    res = runner.invoke(app, ["music", "--fresh", "--segment", "2"])
    assert res.exit_code == 2 and "this bed is one segment" in res.output and "promo:" in res.output
    assert route.call_count == 0 and not run.costs.exists()


def test_promo_music_passes_the_named_segments_to_make_bed(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    seen = {}

    def fake(*a, **k):
        seen.update(k)
        run.music_dir.mkdir(parents=True, exist_ok=True)
        (run.music_dir / "conform.json").write_text("{}")
        return run.music_dir / "bed.wav"

    monkeypatch.setattr("promo.music.make_bed", fake)
    res = runner.invoke(app, ["music", "--fresh", "--segment", "3", "--segment", "1", "--segment", "3"])
    assert res.exit_code == 0, res.output
    assert seen["fresh"] is True and seen["segments"] == [1, 3]       # sorted, once each
    runner.invoke(app, ["music", "--fresh"])
    assert seen["segments"] is None


def test_promo_music_help_documents_segment():
    text = re.sub(r"[\s│]+", " ", runner.invoke(app, ["music", "--help"]).output).lower()
    assert "--segment" in text and "needs --fresh" in text and "numbered as the table prints them" in text
    assert "exactly this segment" in text and "whether or not it passed" in text


def _seams(monkeypatch, **measured):
    """What the seams measure: nothing worth a warning, but for the numbers given (the synthetic parts' own chords
    would dip the crossfade a little)."""
    quiet = {"level_dip_db": -0.2, "level_step_db": 0.0, "spectral_jump_db": 0.5, **measured}
    monkeypatch.setattr("promo.conform.seam_report", lambda path, seam_s, **kw: {"seam_s": seam_s, **quiet})


def test_a_seam_warning_names_the_segment_to_re_roll(tmp_path, monkeypatch):
    _two_calls(monkeypatch)
    _seams(monkeypatch, spectral_jump_db=9.0)
    run, sl, t = _long_run(tmp_path)
    script = Script(tmp_path)
    script.q = {1: [script.good(1)], 2: [script.good(2)]}
    _lyria_bed(run, sl, t, script)
    report = json.loads((run.music_dir / "conform.json").read_text())
    assert any("seam at 16.0s: a band's level jumps 9.0 dB" in w for w in report["warnings"])
    assert report["seam_rerolls"] == [{"seam_s": 16.0, "segment": 2}]
    assert music.seam_advice(report) == ["seam 16.0 s: re-roll with --fresh --segment 2"]


def test_a_guessed_downbeat_at_a_seam_names_the_segment_after_it(tmp_path, monkeypatch):
    _two_calls(monkeypatch)
    _seams(monkeypatch)
    run, sl, t = _long_run(tmp_path)
    script = Script(tmp_path)
    script.q = {1: [script.good(1)],
                2: [lambda ms: groove_mp3(tmp_path, 118.0, ms / 1000, first_beat_s=0.1, seed=3, kick_every_beat=True)]}
    _lyria_bed(run, sl, t, script)
    report = json.loads((run.music_dir / "conform.json").read_text())
    assert any("the segment 2 downbeat is a guess" in w for w in report["warnings"])
    assert report["seam_rerolls"] == [{"seam_s": 16.0, "segment": 2}]


def test_a_bed_with_no_seam_warning_has_no_advice(tmp_path, monkeypatch):
    _seams(monkeypatch)
    run, sl, t, script = _passing_bed(tmp_path, monkeypatch)
    report = json.loads((run.music_dir / "conform.json").read_text())
    assert report["seam_rerolls"] == [] and music.seam_advice(report) == []
    assert music.seam_advice({}) == []                         # a report from before R103


def test_a_bar_one_guess_is_not_a_seam_and_gets_no_advice(tmp_path, monkeypatch):
    _two_calls(monkeypatch)
    _seams(monkeypatch)
    run, sl, t = _long_run(tmp_path)
    script = Script(tmp_path)
    script.q = {1: [lambda ms: groove_mp3(tmp_path, 123.0, ms / 1000, first_beat_s=0.1, seed=3, kick_every_beat=True)],
                2: [script.good(2)]}
    _lyria_bed(run, sl, t, script)
    report = json.loads((run.music_dir / "conform.json").read_text())
    assert any(w.startswith("bar 1 at") and "is a guess" in w for w in report["warnings"])
    assert report["seam_rerolls"] == []


def test_promo_music_prints_which_segment_to_re_roll_for_each_warned_seam(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)

    def fake(*a, **k):
        run.music_dir.mkdir(parents=True, exist_ok=True)
        (run.music_dir / "conform.json").write_text(json.dumps({
            "warnings": ["seam at 246.0s: a band's level jumps 9.0 dB across it (a restart?); listen to it"],
            "seam_rerolls": [{"seam_s": 246.0, "segment": 3}, {"seam_s": 380.5, "segment": 4}]}))
        return run.music_dir / "bed.wav"

    monkeypatch.setattr("promo.music.make_bed", fake)
    res = runner.invoke(app, ["music"])
    assert res.exit_code == 0, res.output
    assert "advice   seam 246.0 s: re-roll with --fresh --segment 3" in res.output
    assert "advice   seam 380.5 s: re-roll with --fresh --segment 4" in res.output
    assert res.output.index("warning  seam at 246.0s") < res.output.index("advice   seam 246.0 s")


def test_promo_music_prints_no_advice_when_no_seam_warns(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)

    def fake(*a, **k):
        run.music_dir.mkdir(parents=True, exist_ok=True)
        (run.music_dir / "conform.json").write_text(json.dumps({"warnings": [], "seam_rerolls": []}))
        return run.music_dir / "bed.wav"

    monkeypatch.setattr("promo.music.make_bed", fake)
    assert "advice" not in runner.invoke(app, ["music"]).output
