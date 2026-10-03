import json
import shutil
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from promo import ref
from promo.models import Caption


def _ref_video(tmp_path):
    a, b, out = tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "ref 🎬.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=3",
                    "-f", "lavfi", "-i", "sine=duration=3:sample_rate=48000", "-pix_fmt", "yuv420p", "-shortest", str(a)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x180:r=30:d=3",
                    "-f", "lavfi", "-i", "sine=duration=3:sample_rate=48000", "-pix_fmt", "yuv420p", "-shortest", str(b)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(a), "-i", str(b), "-filter_complex",
                    "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]", "-map", "[v]", "-map", "[a]", str(out)], check=True)
    return out


def test_ingest_measures_and_scaffolds_style_guide(tmp_path):
    words = [Caption(text=w, startMs=500 + i * 300, endMs=500 + i * 300 + 250) for i, w in enumerate("one two three four".split())]
    out = ref.ingest(_ref_video(tmp_path), tmp_path / "audit", transcribe=lambda p, lang: words)
    m = json.loads((out / "metrics.json").read_text())
    assert m["video"]["width"] == 320 and m["video"]["fps"] == 30
    assert m["cuts"] == pytest.approx([3.0], abs=0.1)
    assert m["voice"]["words"] == 4 and m["voice"]["coverage"] == pytest.approx(1.0 / 6.0, abs=0.02)
    assert "I" in m["audio"]["loudness"]
    guide = (out / "style_guide.md").read_text()
    for heading in ("## Palette", "## Type", "## Shot lengths", "## Transitions", "## Camera", "## Texture",
                    "## Text in and out", "## Signature moves", "## Voice", "## Music", "## Sync"):
        assert heading in guide
    assert "grammar of the reference, never its content" in guide
    assert list(out.glob("contact-*.png"))


def test_pitch_of_a_known_tone():
    sr = 16000
    t = np.arange(sr) / sr
    x = (0.5 * np.sin(2 * np.pi * 210 * t)).astype(np.float32)
    words = [Caption(text="a", startMs=0, endMs=1000)]
    assert ref.pitch_hz(x, sr, words) == pytest.approx(210, rel=0.03)


def _words(*spans):
    return [Caption(text=f"w{i}", startMs=a, endMs=b) for i, (a, b) in enumerate(spans)]


def test_voice_metrics_pace_is_measured_inside_the_lines(tmp_path):
    # two lines of two words (a 1.45 s gap between them) in a 6 s video: 4 words, 1.1 s inside lines
    words = _words((500, 750), (800, 1050), (2500, 2750), (2800, 3050))
    out = ref.ingest(_ref_video(tmp_path), tmp_path / "audit", transcribe=lambda p, lang: words)
    v = json.loads((out / "metrics.json").read_text())["voice"]
    assert v["lines"] == 2 and v["line_gaps_s"] == [1.45]
    assert v["spoken_wpm"] == round(4 / 1.1 * 60) and v["overall_wpm"] == round(4 / 6 * 60)
    assert json.loads((out / "transcript.json").read_text())[0]["text"] == "w0"
    guide = (out / "style_guide.md").read_text()
    assert "4 words, coverage 17%" in guide and "218 wpm inside lines" in guide


def test_transcribe_gets_the_16k_wav_and_the_language(tmp_path):
    seen = []

    def fake(path, lang):
        seen.append((path.name, lang, path.exists()))
        return []

    ref.ingest(_ref_video(tmp_path), tmp_path / "audit", language="de", transcribe=fake)
    assert seen == [("audio16k.wav", "de", True)]
    m = json.loads((tmp_path / "audit" / "metrics.json").read_text())
    assert m["voice"]["words"] == 0 and m["voice"]["median_pitch_hz"] is None
    assert "no speech found" in (tmp_path / "audit" / "style_guide.md").read_text()


def test_default_transcriber_is_whisper_in_english_unless_told(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ref.asr, "transcribe", lambda path, lang, engine=None: calls.append((lang, engine)) or [])
    video = _ref_video(tmp_path)
    ref.ingest(video, tmp_path / "a")
    ref.ingest(video, tmp_path / "b", language="fr")
    assert calls == [("en", "whisper"), ("fr", "whisper")]


def test_crop_and_a_folder_with_spaces_and_emoji(tmp_path):
    from PIL import Image

    out = ref.ingest(_ref_video(tmp_path), tmp_path / "audit 🎬 crop", crop="160:90:0:0", transcribe=lambda p, lang: [])
    assert Image.open(out / "frames" / "f-0000.png").size == (480, 270)      # the crop, scaled to 480 wide
    assert 12 <= len(list((out / "frames").glob("f-*.png"))) <= 13            # every 0.5 s of a 6 s video
    assert (out / "contact-00.png").exists() and not (out / "contact-01.png").exists()


def test_bad_inputs_are_ref_errors(tmp_path):
    video = _ref_video(tmp_path)
    for crop in ("1280x720", "1:2:3", "a:b:c:d", "1:2:3:4,vflip"):
        with pytest.raises(ref.RefError, match="W:H:X:Y"):
            ref.ingest(video, tmp_path / "x", crop=crop)
    with pytest.raises(ref.RefError, match="no such video"):
        ref.ingest(tmp_path / "missing.mp4", tmp_path / "x")
    audio_only = tmp_path / "tone.m4a"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=duration=1", str(audio_only)], check=True)
    with pytest.raises(ref.RefError, match="no video stream"):
        ref.ingest(audio_only, tmp_path / "x")


def test_a_silent_video_has_no_audio_metrics(tmp_path):
    silent = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=2",
                    "-pix_fmt", "yuv420p", str(silent)], check=True)

    def never(path, lang):
        raise AssertionError("there is no audio to transcribe")

    out = ref.ingest(silent, tmp_path / "audit", transcribe=never)
    m = json.loads((out / "metrics.json").read_text())
    assert m["audio"] == {"loudness": None, "bpm": None, "bpm_confidence": None, "bpm_candidates": [], "bpm_tracker": None}
    assert m["voice"]["words"] == 0
    assert json.loads((out / "transcript.json").read_text()) == []
    assert "none (no audio track)" in (out / "style_guide.md").read_text()
    assert not (out / "spectrogram.png").exists() and (out / "contact-00.png").exists()


def test_silence_does_not_put_infinity_in_the_json(tmp_path):
    quiet = tmp_path / "quiet.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=30:duration=2",
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-pix_fmt", "yuv420p", "-shortest", str(quiet)], check=True)
    out = ref.ingest(quiet, tmp_path / "audit", transcribe=lambda p, lang: [])
    text = (out / "metrics.json").read_text()
    assert "Infinity" not in text and "NaN" not in text
    assert json.loads(text)["audio"]["loudness"]["TP"] is None                # a silent peak is -inf dBFS
    assert "TP n/a dBTP" in (out / "style_guide.md").read_text()


def test_pitch_ignores_words_past_the_end_of_the_audio():
    sr = 16000
    x = (0.5 * np.sin(2 * np.pi * 140 * np.arange(sr) / sr)).astype(np.float32)
    words = [Caption(text="a", startMs=-50, endMs=1400), Caption(text="b", startMs=5000, endMs=6000)]
    assert ref.pitch_hz(x, sr, words) == pytest.approx(140, rel=0.03)


def test_pitch_of_noise_and_silence_is_none():
    sr = 16000
    words = [Caption(text="a", startMs=0, endMs=1000)]
    assert ref.pitch_hz(np.zeros(sr, dtype=np.float32), sr, words) is None
    noise = np.random.default_rng(1).normal(size=sr).astype(np.float32)
    assert ref.pitch_hz(noise, sr, words) is None
    assert ref.pitch_hz(noise, sr, []) is None


def test_cli_ref_defaults_the_folder_and_passes_options(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from promo.cli import app

    calls = []
    monkeypatch.setattr(ref, "ingest", lambda video, out, **kw: calls.append((video, out, kw)) or out)
    video = tmp_path / "my ref.mp4"
    res = CliRunner().invoke(app, ["ref", str(video), "--crop", "10:10:0:0", "--lang", "pt"])
    assert res.exit_code == 0, res.output
    assert calls == [(video, tmp_path / "my ref-audit", {"crop": "10:10:0:0", "language": "pt"})]
    assert res.output.strip() == str(tmp_path / "my ref-audit")
    CliRunner().invoke(app, ["ref", str(video), "--out", str(tmp_path / "elsewhere")])
    assert calls[-1][1] == tmp_path / "elsewhere"


def test_cli_ref_errors_are_one_line(tmp_path):
    from typer.testing import CliRunner

    from promo.cli import app

    runner = CliRunner()
    res = runner.invoke(app, ["ref", str(tmp_path / "missing.mp4")])
    assert res.exit_code == 2 and "promo: no such video" in res.output and "Traceback" not in res.output
    res = runner.invoke(app, ["ref", str(_ref_video(tmp_path)), "--crop", "wide", "--out", str(tmp_path / "o")])
    assert res.exit_code == 2 and "W:H:X:Y" in res.output
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    res = runner.invoke(app, ["ref", str(broken), "--out", str(tmp_path / "o2")])
    assert res.exit_code == 1 and res.output.startswith("promo: command failed") and "Traceback" not in res.output


def _mix(bpm, seconds=24.0, seed=0):
    """A kick on every beat and a quiet hat between them, under what makes a beat tracker slip: a mid-band stab every
    dotted quarter (2/3 of the tempo) and phrases of syllable-rate noise bursts, like a voice-over."""
    sr, rng = 16000, np.random.default_rng(seed)
    n, beat = int(seconds * sr), 60.0 / bpm
    y = np.zeros(n)
    t = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * np.cumsum(50 + 60 * np.exp(-t * 40)) / sr) * np.exp(-t * 14)
    hat = rng.normal(size=int(0.04 * sr)) * np.exp(-np.arange(int(0.04 * sr)) / sr * 150)
    stab = (np.sin(2 * np.pi * 1200 * t[:1920]) + np.sin(2 * np.pi * 1800 * t[:1920])) * np.exp(-t[:1920] * 25)

    def put(at_s, sound, gain):
        a = int(at_s * sr)
        if 0 <= a < n:
            y[a:a + len(sound)] += gain * sound[:n - a]

    for i in range(int((seconds - 0.5) / beat)):
        put(0.2 + i * beat, kick, 0.8)
        put(0.2 + (i + 0.5) * beat, hat, 0.12)
    for j in range(int((seconds - 0.5) / (1.5 * beat))):
        put(0.5 + j * 1.5 * beat, stab, 0.35)
    at = 0.5
    while at < seconds - 2:
        for _ in range(rng.integers(8, 14)):
            d = rng.uniform(0.12, 0.22)
            spectrum = np.fft.rfft(rng.normal(size=int(d * sr)))
            freqs = np.fft.rfftfreq(int(d * sr), 1 / sr)
            spectrum[(freqs < 250) | (freqs > 3200)] = 0
            burst = np.fft.irfft(spectrum, int(d * sr)) * np.hanning(int(d * sr))
            put(at, burst / np.abs(burst).max(), 0.5)
            at += d + rng.uniform(0.02, 0.1)
        at += rng.uniform(0.5, 1.2)
    return (y / (np.abs(y).max() * 1.05)).astype(np.float32)


def _clicks(bpm, seconds=12.0):
    """Mid-band ticks on every beat over an irregular low rumble: the whole spectrum has a pulse and the kick band does not."""
    sr = 16000
    rng = np.random.default_rng(5)
    y = np.zeros(int(seconds * sr))
    tick = np.sin(2 * np.pi * 1800 * np.arange(int(0.05 * sr)) / sr) * np.hanning(int(0.05 * sr))
    for i in range(int((seconds - 0.5) / (60 / bpm))):
        a = int((0.2 + i * 60 / bpm) * sr)
        y[a:a + len(tick)] += tick
    spectrum = np.fft.rfft(rng.normal(size=len(y)))
    spectrum[np.fft.rfftfreq(len(y), 1 / sr) > 120] = 0
    rumble = np.fft.irfft(spectrum, len(y))
    return (y + 0.4 * rumble / np.abs(rumble).max()).astype(np.float32)


@pytest.mark.parametrize("bpm", [70, 90, 100, 120, 128, 140])
def test_tempo_is_the_kick_refined_tracker_reading_under_voice_over_and_stabs(bpm):
    t = ref.tempo(_mix(bpm))
    assert t["bpm"] == pytest.approx(bpm, abs=0.3) and t["confidence"] == "kick-band"
    assert t["candidates"][0] == t["bpm"] and t["tracker"] is not None
    assert not any(c == pytest.approx(bpm * 2 / 3, rel=0.01) for c in t["candidates"])       # never the 2/3 reading


@pytest.mark.parametrize("bpm", [90, 120, 140])
@pytest.mark.parametrize("off", [0.975, 1.025])
def test_the_kick_band_corrects_a_tracker_that_is_a_couple_of_percent_off(bpm, off):
    t = ref.tempo(_mix(bpm), tracker=bpm * off)           # 117.45 for 120 BPM was the reference's reading
    assert t["bpm"] == pytest.approx(bpm, abs=0.1) and t["tracker"] == pytest.approx(bpm * off)


def test_a_doubled_tracker_is_halved_only_with_a_strong_kick_and_a_landing_in_range():
    assert ref.tempo(_mix(70), tracker=140.0)["bpm"] == pytest.approx(70, abs=0.1)         # half 70: in [70, 180)
    assert ref.tempo(_mix(85), tracker=170.0)["bpm"] == pytest.approx(85, abs=0.1)
    # a tracker at 240 would halve to 120 (in range), but a half of 90 or 100 of a 45 or 50 reading is out of range
    assert ref.tempo(_mix(45), tracker=90.0)["bpm"] == pytest.approx(90, abs=0.3)          # half is 45: not considered
    assert ref.tempo(_mix(100), tracker=100.0)["bpm"] == pytest.approx(100, abs=0.1)       # no level change without cause


def test_a_weak_kick_leaves_the_tracker_as_it_is_and_says_so():
    t = ref.tempo(_clicks(120), tracker=120.0)
    assert t["bpm"] == 120.0 and t["confidence"] == "tracker-only"            # not refined: the kick band is empty
    assert t["candidates"] == [120.0, 60.0, 240.0]
    assert ref.tempo(_clicks(100), tracker=101.0)["bpm"] == 101.0


def test_no_credible_pulse_gives_none_with_the_reason():
    noise = np.random.default_rng(3).normal(size=16000 * 12).astype(np.float32) * 0.1
    t = ref.tempo(noise, tracker=120.0)
    assert t["bpm"] is None and t["confidence"] is None and t["error"] == "no credible pulse found"
    tone = (0.5 * np.sin(2 * np.pi * 220 * np.arange(16000 * 12) / 16000)).astype(np.float32)
    assert ref.tempo(tone, tracker=118.0)["bpm"] is None
    silent = ref.tempo(np.zeros(16000 * 12, dtype=np.float32))
    assert silent["bpm"] is None and silent["tracker"] is None and silent["error"].startswith("no beat tracker reading (")
    assert "/" not in silent["error"]                                           # no temporary file name in it


def test_a_clip_too_short_for_a_tempo_is_none_not_an_error():
    y = _mix(120, seconds=12)
    for seconds in (1.0, 4.0, 4.2, 4.4, 4.9):
        t = ref.tempo(y[: int(seconds * 16000)])
        assert t["bpm"] is None and t["error"] == "audio too short for a tempo"
    assert ref.tempo(y[: 16000 * 6])["bpm"] == pytest.approx(120, abs=0.3)


# A real 120 BPM reference video, opt-in: PROMO_REAL_REFERENCE=/path/to/video.mp4
REAL_REFERENCE = Path(os.environ.get("PROMO_REAL_REFERENCE", "/nonexistent"))


@pytest.mark.skipif(not REAL_REFERENCE.exists(), reason="set PROMO_REAL_REFERENCE to a 120 BPM reference video")
def test_tempo_of_the_pocketsflow_reference_is_120_not_the_trackers_117_5(tmp_path):
    from promo.audio_analysis import decode_mono

    wav = tmp_path / "a16.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(REAL_REFERENCE), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                   check=True)
    t = ref.tempo(decode_mono(wav, 16000), wav=wav)
    assert t["bpm"] == pytest.approx(120.0, abs=0.1) and t["confidence"] == "kick-band"
    assert t["tracker"] == pytest.approx(117.45, abs=0.1)
    assert not any(c == pytest.approx(81, abs=1.5) for c in t["candidates"])                # 80-81 is the dotted-quarter reading


STOCK_BEDS = Path.home() / "Downloads"
# (file, the level the tracker reads and a person hears): the kick-only reading an earlier version used put these at 30 to 60
BEDS = [("loksii-no-copyright-music-211881.mp3", 120.0), ("moodmode-no-copyright-music-201745.mp3", 120.0),
        ("sub_clair-kids-581150.mp3", 95.0), ("Aylex - Brand (freetouse.com).mp3", 150.0),
        ("lnplusmusic-background-background-music-597024.mp3", 110.0)]


@pytest.mark.parametrize("name,bpm", BEDS)
def test_real_stock_beds_keep_the_trackers_level(tmp_path, name, bpm):
    if not (STOCK_BEDS / name).exists():
        pytest.skip(f"{name} is not on this machine")
    from promo.audio_analysis import decode_mono

    wav = tmp_path / "bed.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "8", "-t", "45", "-i", str(STOCK_BEDS / name), "-vn", "-ac", "1",
                    "-ar", "16000", str(wav)], check=True)
    t = ref.tempo(decode_mono(wav, 16000), wav=wav)
    assert t["bpm"] == pytest.approx(bpm, rel=0.015) and t["bpm"] == pytest.approx(t["tracker"], rel=0.015)


def _video_with_sound(tmp_path, samples):
    import soundfile as sf

    wav, out = tmp_path / "mix.wav", tmp_path / "mix.mp4"
    sf.write(wav, samples, 16000)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=30:duration=12",
                    "-i", str(wav), "-pix_fmt", "yuv420p", "-shortest", str(out)], check=True)
    return out


def test_ingest_reports_the_tempo_its_basis_its_candidates_and_the_trackers_reading(tmp_path):
    out = ref.ingest(_video_with_sound(tmp_path, _mix(120, seconds=12)), tmp_path / "audit", transcribe=lambda p, lang: [])
    audio = json.loads((out / "metrics.json").read_text())["audio"]
    assert audio["bpm"] == pytest.approx(120, abs=0.3) and audio["bpm_confidence"] == "kick-band"
    assert audio["bpm_candidates"][0] == audio["bpm"] and audio["bpm_tracker"] == pytest.approx(120, rel=0.03)
    assert "bpm_error" not in audio
    line = [ln for ln in (out / "style_guide.md").read_text().splitlines() if ln.startswith("- Audio:")][0]
    assert f"tempo {audio['bpm']:.1f} BPM (kick band; candidates {audio['bpm']:.1f}, " in line
    assert f"beat tracker {audio['bpm_tracker']:.1f})" in line


def test_ingest_says_so_when_there_is_no_credible_pulse(tmp_path):
    out = ref.ingest(_ref_video(tmp_path), tmp_path / "audit", transcribe=lambda p, lang: [])     # a sine tone
    audio = json.loads((out / "metrics.json").read_text())["audio"]
    assert audio["bpm"] is None and audio["bpm_confidence"] is None and audio["bpm_candidates"] == []
    assert audio["bpm_error"] == "no credible pulse found"
    assert "tempo n/a (no credible pulse found; beat tracker " in (out / "style_guide.md").read_text()


def test_ingest_says_when_the_tracker_alone_gives_the_tempo(tmp_path):
    out = ref.ingest(_video_with_sound(tmp_path, _clicks(120)), tmp_path / "audit", transcribe=lambda p, lang: [])
    audio = json.loads((out / "metrics.json").read_text())["audio"]
    assert audio["bpm_confidence"] == "tracker-only" and audio["bpm"] == audio["bpm_tracker"]
    assert "tracker only: the kick band is too weak to refine it" in (out / "style_guide.md").read_text()


def _rotated(tmp_path):
    base, rot = tmp_path / "base.mp4", tmp_path / "rot.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=2",
                    "-pix_fmt", "yuv420p", str(base)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-display_rotation", "90", "-i", str(base), "-c", "copy", str(rot)],
                   check=True)
    return base, rot


def test_crop_must_fit_the_frame(tmp_path):
    video = _ref_video(tmp_path)                                    # 320x180
    for crop in ("0:90:0:0", "160:0:0:0", "321:180:0:0", "160:90:200:0", "160:90:0:100"):
        with pytest.raises(ref.RefError, match=r"does not fit the 320x180 frame"):
            ref.ingest(video, tmp_path / "x", crop=crop)
    assert not (tmp_path / "x").exists()                            # refused before anything was written
    out = ref.ingest(video, tmp_path / "ok", crop="320:180:0:0", transcribe=lambda p, lang: [])   # exactly the frame
    assert (out / "contact-00.png").exists()


def test_crop_of_a_rotated_phone_video_is_checked_against_the_upright_frame(tmp_path):
    from PIL import Image

    _, rot = _rotated(tmp_path)
    with pytest.raises(ref.RefError, match=r"does not fit the 180x320 frame"):
        ref.ingest(rot, tmp_path / "x", crop="320:180:0:0")
    silent = ref.ingest(rot, tmp_path / "ok", crop="180:320:0:0", transcribe=lambda p, lang: [])
    assert Image.open(silent / "frames" / "f-0000.png").size == (480, 854)
    assert json.loads((silent / "metrics.json").read_text())["video"]["width"] == 180


def test_cli_ref_without_ffmpeg_or_ffprobe_is_one_line_exit_2(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from promo.cli import app

    video, real_ffprobe = _ref_video(tmp_path), shutil.which("ffprobe")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))
    res = CliRunner().invoke(app, ["ref", str(video), "--out", str(tmp_path / "o")])
    assert res.exit_code == 2 and res.output.strip() == "promo: ffprobe not found (install ffmpeg)"
    (bin_dir / "ffprobe").symlink_to(real_ffprobe)
    res = CliRunner().invoke(app, ["ref", str(video), "--out", str(tmp_path / "o")])
    assert res.exit_code == 2 and res.output.strip() == "promo: ffmpeg not found (install ffmpeg)"
