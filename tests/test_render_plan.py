import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from promo.models import LineTake
from promo.packs import Limits
from promo.render import plan_ffmpeg
from promo.shotlist import Shotlist
from promo.timeline import Media, build_timeline
from promo.timing import Measured, resolve
from promo.voice import trim_and_normalize

FIX = Path(__file__).parent / "fixtures" / "shotlist_music.json"
SR = 48000


def test_plan_has_inputs_graph_and_two_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr("promo.timeline.duration_s", lambda p: 20.0)
    monkeypatch.setattr("promo.timeline.sha256_file", lambda p: "x")
    takes = {l: LineTake(id=l, file=f"audio/vo/{l}.wav", duration_s=1.0) for l in ("c01-l1", "c02-l1", "c02-l2")}
    t = resolve(Shotlist.model_validate(json.loads(FIX.read_text())), Measured(takes=takes), Limits())
    d = tmp_path / "Codex Projects 🎬"
    media = Media(shots={"c01-s1": d / "a b.mp4", "c02-s1": d / "c.mp4"},
                  vo={l: d / f"{l}.wav" for l in takes}, music=d / "bed.wav", sfx=d / "sfx.wav",
                  overlay=d / "captions.mov")
    tl = build_timeline(t, media)
    cmd = plan_ffmpeg(tl, width=1920, height=1080, fps=60, bg="#F2F3EC", duck_db=7, fade_out_s=1.0,
                      picture_out=d / "p.mp4", mix_out=d / "m.wav")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "concat=n=3:v=1:a=0[vbase]" in graph
    assert "color=c=0xF2F3EC" in " ".join(cmd)          # the missing shot becomes a colour source
    assert "overlay=0:0:enable='between(t,0.000000,8.300000)'" in graph
    assert "adelay=delays=800:all=1" in graph
    assert "sidechaincompress=threshold=0.03:ratio=1.498:attack=15:release=350" in graph   # a 7 dB duck
    assert "atrim=0:8.300000[mix]" in graph
    assert str(d) not in graph                            # paths never enter the graph
    assert cmd.count("-i") == 2 + 1 + 3 + 1 + 1 + 1       # 2 shots, 1 gap colour, 3 VO, overlay, music, sfx
    assert cmd[-1] == str(d / "m.wav") and str(d / "p.mp4") in cmd
    assert cmd[cmd.index("-crf") + 1] == "16"


# --- behaviour: run the planned command on tiny real media and measure what comes out ---------------------------

def _timing():
    takes = {l: LineTake(id=l, file=f"audio/vo/{l}.wav", duration_s=s)
             for l, s in (("c01-l1", 1.5), ("c02-l1", 1.0), ("c02-l2", 1.4))}
    return resolve(Shotlist.model_validate(json.loads(FIX.read_text())), Measured(takes=takes), Limits())


def _sine(path: Path, seconds: float, freq: float, amp: float, channels: int = 1) -> Path:
    t = np.arange(round(seconds * SR)) / SR
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, np.stack([x] * channels, 1) if channels > 1 else x, SR, subtype="PCM_24")
    return path


def _shot(path: Path, seconds: float, colour: str = "gray", fps: int = 60) -> Path:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"color=c={colour}:s=64x36:r={fps}:d={seconds}", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def _render(t, media, tmp_path, *, fps: int = 30, bg: str = "#F2F3EC", duck_db: float = 7, fade_out_s: float = 1.0,
            draft: bool = True):
    pic, mix = tmp_path / "pic.mp4", tmp_path / "mix.wav"
    cmd = plan_ffmpeg(build_timeline(t, media), width=64, height=36, fps=fps, bg=bg, duck_db=duck_db,
                      fade_out_s=fade_out_s, picture_out=pic, mix_out=mix, draft=draft)
    subprocess.run(cmd, check=True)
    audio, sr = sf.read(mix, dtype="float64", always_2d=True)
    assert sr == SR
    return pic, audio


def _rms(x: np.ndarray, a: float, b: float) -> float:
    seg = x[round(a * SR):round(b * SR)]
    return float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0


def _frames(pic: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(pic)], capture_output=True, text=True,
                         check=True).stdout
    return int(out.strip())


def test_mono_sfx_reaches_both_channels_at_its_own_level(tmp_path):
    # The SFX stem is mono. It must land on both channels at unity: the stem's own samples, with no -3 dB pan law
    # (ffmpeg's default mono->stereo upmix) and nothing on one side only.
    t = _timing()
    stem = np.zeros(round(t.duration_s * SR), dtype=np.float32)
    stem[2 * SR:3 * SR] = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(SR) / SR)
    sf.write(tmp_path / "sfx.wav", stem, SR, subtype="PCM_24")
    _, audio = _render(t, Media(sfx=tmp_path / "sfx.wav"), tmp_path)
    assert audio.shape == (round(t.duration_s * SR), 2)
    assert np.array_equal(audio[:, 0], audio[:, 1])
    assert np.max(np.abs(audio[:, 0] - stem)) < 1e-4
    assert 20 * np.log10(np.max(np.abs(audio[:, 0]))) == pytest.approx(20 * np.log10(0.5), abs=0.01)


def test_stereo_music_passes_through_unchanged(tmp_path):
    t = _timing()
    left = 0.25 * np.sin(2 * np.pi * 220 * np.arange(10 * SR) / SR)
    sf.write(tmp_path / "bed.wav", np.stack([left, np.zeros_like(left)], 1), SR, subtype="PCM_24")
    _, audio = _render(t, Media(music=tmp_path / "bed.wav"), tmp_path, fade_out_s=0)   # no end fade: every sample
    pre = round(t.preroll_s * SR)
    assert np.max(np.abs(audio[:pre])) == 0                                  # the bed starts after the pre-roll
    assert np.max(np.abs(audio[pre:, 0] - left[:len(audio) - pre])) < 1e-4  # left stays left, at its level, to the end
    assert np.max(np.abs(audio[:, 1])) < 1e-6                               # nothing leaks into the right


def test_bed_shorter_than_the_film_fades_out_and_is_padded_with_silence(tmp_path):
    # The bed may not cover the snapped film. It must not cut the mix short, fail, or stop on a click: it fades
    # out before its end and the rest of the film is silence (here the VO carries on over it).
    t = _timing()
    bed = _sine(tmp_path / "bed.wav", 3.0, 110, 0.3, channels=2)
    vo = {l.id: _sine(tmp_path / f"{l.id}.wav", l.end_s - l.start_s, 2000, 0.2) for l in t.lines}
    _, audio = _render(t, Media(music=bed), tmp_path)
    end = t.preroll_s + 3.0
    assert len(audio) == round(t.duration_s * SR)
    assert _rms(audio[:, 0], 0.5, 1.5) == pytest.approx(0.3 / np.sqrt(2), rel=0.02)    # full level before the fade
    assert _rms(audio[:, 0], end - 0.1, end) < 0.1 * _rms(audio[:, 0], 0.5, 1.5)       # faded by its end
    assert np.max(np.abs(audio[round(end * SR):])) == 0                               # silence after it

    _, audio = _render(t, Media(music=bed, vo=vo), tmp_path)
    last = t.lines[-1]
    assert len(audio) == round(t.duration_s * SR)
    assert _rms(audio[:, 0], last.start_s + 0.1, last.end_s - 0.1) > 0.1               # VO after the bed survives


def test_music_carries_on_after_the_last_vo_line(tmp_path):
    # sidechaincompress stops when either input stops. The VO key ends with the last line (6.9 s here), so it is
    # padded to the film length; otherwise the ducked music, and the mix, would go silent there.
    t = _timing()
    bed = _sine(tmp_path / "bed.wav", 12.0, 110, 0.3, channels=2)
    vo = {l.id: _sine(tmp_path / f"{l.id}.wav", l.end_s - l.start_s, 2000, 0.2) for l in t.lines}
    _, audio = _render(t, Media(music=bed, vo=vo), tmp_path)
    assert t.lines[-1].end_s < t.duration_s - 1.3
    # From just after the last line to where the end-of-film fade begins: still ducking as the compressor releases,
    # so not at full level, but nowhere near the silence a cut-off bed leaves.
    assert _rms(audio[:, 0], t.lines[-1].end_s + 0.05, t.duration_s - 1.0) > 0.5 * 0.3 / np.sqrt(2)


def test_picture_frames_come_from_absolute_times_at_any_fps(tmp_path):
    # A cut on an odd 60 fps frame falls between two 30 fps frames. Each segment is rounded from its absolute start
    # and end, so the draft picture has exactly round(duration * 30) frames and no drift builds up.
    t = _timing()
    cut = round(4.3 * 60 + 1) / 60
    t.shots[0].end_s = t.shots[1].start_s = cut
    shots = {s.id: _shot(tmp_path / f"{s.id}.mp4", s.end_s - s.start_s) for s in t.shots[:2]}
    pic, _ = _render(t, Media(shots=shots), tmp_path, fps=30)
    assert _frames(pic) == round(t.duration_s * 30)


def test_a_short_shot_render_holds_its_last_frame_so_later_shots_keep_their_place(tmp_path):
    t = _timing()
    shots = {"c01-s1": _shot(tmp_path / "c01-s1.mp4", 4.0, "black"),          # 0.3 s shorter than its slot
             "c02-s1": _shot(tmp_path / "c02-s1.mp4", 2.0, "white")}
    pic, _ = _render(t, Media(shots=shots), tmp_path, fps=30)
    assert _frames(pic) == round(t.duration_s * 30)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(pic), "-vf", "scale=8:4", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    luma = np.frombuffer(raw, np.uint8).reshape(-1, 32).mean(axis=1)
    first_white = int(np.argmax(luma > 128))
    assert first_white == round(t.shot("c02-s1").start_s * 30)


def test_background_must_be_a_hex_colour(tmp_path):
    t = _timing()
    with pytest.raises(ValueError, match="background"):
        plan_ffmpeg(build_timeline(t, Media()), width=64, height=36, fps=30, bg="rgba(0,0,0,0.5)", duck_db=7,
                    fade_out_s=1.0, picture_out=tmp_path / "p.mp4", mix_out=tmp_path / "m.wav")


def test_a_60fps_overlay_on_a_30fps_draft_adds_no_frame(tmp_path):
    # The captions overlay is rendered at the timeline rate (60). On a 30 fps draft, overlay would keep emitting
    # frames at the overlay's own times once the base picture ends, one past the end of the film.
    t = _timing()
    ov = tmp_path / "captions.mov"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"color=c=white@0.5:s=64x36:r=60:d={t.duration_s},format=yuva444p10le",
                    "-c:v", "prores_ks", "-profile:v", "4444", str(ov)], check=True)
    shots = {s.id: _shot(tmp_path / f"{s.id}.mp4", s.end_s - s.start_s) for s in t.shots}
    pic, _ = _render(t, Media(shots=shots, overlay=ov), tmp_path, fps=30)
    assert _frames(pic) == round(t.duration_s * 30)


def _speech(seconds: float, f0: float, seed: int) -> np.ndarray:
    """Speech-like VO: glottal-pulse harmonics under 150-300 ms syllables with short gaps. Normalised like a real take,
    it ducks a bed within 0.15 dB of recorded speech at every depth from 4 to 10 dB (checked against macOS `say`
    takes normalised by promo.voice), so it stands in for a voice in the duck calibration test."""
    rng = np.random.default_rng(seed)
    n = round(seconds * SR)
    t = np.arange(n) / SR
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.08 * np.sin(2 * np.pi * 0.7 * t))) / SR
    x = sum(np.cos(h * phase) / h for h in range(1, 25))
    env, k = np.zeros(n), 0
    while k < n:
        length = int(rng.uniform(0.15, 0.3) * SR)
        m = min(length, n - k)
        env[k:k + m] = np.sin(np.pi * np.arange(m) / length) ** 0.5 * rng.uniform(0.7, 1.0)
        k += length + int(rng.uniform(0.01, 0.04) * SR)
    return (x * env).astype(np.float32)


@pytest.mark.parametrize("depth", [6, 8])
def test_the_bed_ducks_by_the_packs_depth_under_speech(tmp_path, depth):
    t = _timing()
    vo = {}
    for i, line in enumerate(t.lines):
        raw = tmp_path / f"{line.id}.raw.wav"
        sf.write(raw, _speech(line.end_s - line.start_s, (190, 210, 180)[i], i), SR, subtype="FLOAT")
        vo[line.id] = trim_and_normalize(raw, tmp_path / f"{line.id}.wav")      # -16 LUFS, as a real take
    bed = _sine(tmp_path / "bed.wav", 12.0, 220, 0.25, channels=2)
    _, both = _render(t, Media(vo=vo, music=bed), tmp_path, duck_db=depth)
    _, voice = _render(t, Media(vo=vo), tmp_path, duck_db=depth)
    music = both[:, 0] - voice[:, 0]                       # the buses sum at unity, so this is the ducked bed alone
    w = 2400                                                # 50 ms windows
    m = len(music) // w
    level = lambda x: 20 * np.log10(np.sqrt((x[:m * w].reshape(m, w) ** 2).mean(1)) + 1e-12)
    speaking = level(voice[:, 0]) > -40
    duck = -(level(music) - 20 * np.log10(0.25 / np.sqrt(2)))
    assert np.median(duck[speaking]) == pytest.approx(depth, abs=1.0)


def test_no_duck_when_the_depth_is_zero_and_a_depth_out_of_range_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("promo.timeline.duration_s", lambda p: 20.0)
    monkeypatch.setattr("promo.timeline.sha256_file", lambda p: "x")
    tl = build_timeline(_timing(), Media(music=tmp_path / "bed.wav"))
    kw = dict(width=64, height=36, fps=30, bg="#000000", fade_out_s=1.0, picture_out=tmp_path / "p.mp4",
              mix_out=tmp_path / "m.wav")
    assert "sidechaincompress" not in " ".join(plan_ffmpeg(tl, duck_db=0, **kw))
    for bad in (-1, 19, float("nan")):
        with pytest.raises(ValueError, match="duck_db"):
            plan_ffmpeg(tl, duck_db=bad, **kw)


def test_a_bed_longer_than_the_film_fades_out_before_the_film_ends(tmp_path):
    # The bed is generated a tail longer than the film and clamped at the film end; it must not stop dead there.
    t = _timing()
    bed = _sine(tmp_path / "bed.wav", 12.0, 220, 0.3, channels=2)
    _, audio = _render(t, Media(music=bed), tmp_path)
    steady = _rms(audio[:, 0], 2.0, 3.0)
    assert steady == pytest.approx(0.3 / np.sqrt(2), rel=0.02)
    assert 20 * np.log10(_rms(audio[:, 0], t.duration_s - 0.02, t.duration_s) / steady) <= -40


@pytest.mark.parametrize("draft", [False, True])
def test_gaps_are_bt709_and_the_picture_is_tagged_bt709(tmp_path, draft):
    # The shots are BT.709 (HyperFrames tags all four), so a generated gap must be converted with the same matrix,
    # and the picture must say so; ffmpeg's default conversion of a colour source uses BT.601 and drops the tags.
    pic, _ = _render(_timing(), Media(), tmp_path, bg="#1E40AF", draft=draft)
    tags = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=color_range,color_space,color_transfer,color_primaries", "-of", "json", str(pic)],
                          capture_output=True, text=True, check=True).stdout
    stream = json.loads(tags)["streams"][0]
    assert (stream["color_range"], stream["color_space"], stream["color_transfer"], stream["color_primaries"]) == \
        ("tv", "bt709", "bt709", "bt709")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(pic), "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "yuv444p", "-"], capture_output=True, check=True).stdout
    plane = len(raw) // 3
    yuv = [raw[i * plane + plane // 2] for i in range(3)]          # the centre pixel of each plane
    assert yuv == pytest.approx([71, 181, 109], abs=1)                 # BT.709 limited range for #1E40AF
