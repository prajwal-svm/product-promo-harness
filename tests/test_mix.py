import subprocess

import numpy as np
import pytest
import soundfile as sf

from promo import audio_analysis as aa
from promo.mix import loudnorm_measure, master_audio, mux


@pytest.fixture
def quiet_sine(tmp_path):
    # Both channels carry the sine at full level. `-ac 2` would not: ffmpeg upmixes mono at -3 dB per channel.
    p = tmp_path / "in.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=6:sample_rate=48000", "-af", "pan=stereo|c0=c0|c1=c0", str(p)],
                   check=True)
    return p


def test_measure_reports_input_loudness(quiet_sine):
    m = loudnorm_measure(quiet_sine, -14, -2.0)
    # a 1/8-amplitude sine reads -21 LUFS per channel; the identical second channel adds 3 dB
    assert float(m["input_i"]) == pytest.approx(-18, abs=1.0)


def test_master_hits_target_and_ceiling(quiet_sine, tmp_path):
    out = master_audio(quiet_sine, tmp_path / "m.wav", target_i=-14.0, ceiling_db=-2.0)
    m = aa.loudness(out)
    assert m["I"] == pytest.approx(-14, abs=0.7)
    assert m["TP"] <= -1.9


def test_mux_makes_aac_mp4(quiet_sine, tmp_path):
    pic = tmp_path / "pic.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=6",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(pic)], check=True)
    out = mux(pic, quiet_sine, tmp_path / "o.mp4")
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,sample_rate", "-of", "csv=p=0",
                            str(out)], capture_output=True, text=True, check=True).stdout
    assert "h264" in probe and "aac,48000" in probe


def test_silence_is_limited_not_normalised(tmp_path):
    src = tmp_path / "silent.wav"
    sf.write(src, np.zeros((4 * 48000, 2), np.float32), 48000, subtype="FLOAT")
    out = master_audio(src, tmp_path / "m.wav")
    audio, sr = sf.read(out, always_2d=True)
    assert sr == 48000 and audio.shape == (4 * 48000, 2) and not audio.any()


def test_master_does_not_move_the_sound_in_time(tmp_path):
    # A limiter's look-ahead delays its output unless it is compensated; any delay here is an A/V offset.
    sr, n = 48000, 6 * 48000
    x = 0.1 * np.sin(2 * np.pi * 220 * np.arange(n) / sr) + np.random.default_rng(1).normal(0, 0.02, n)
    x[2 * sr:2 * sr + 24] += 0.9                     # a click at 2.000 s, well over the ceiling
    src = tmp_path / "click.wav"
    sf.write(src, np.stack([x, x], 1).astype(np.float32), sr, subtype="FLOAT")
    audio, _ = sf.read(master_audio(src, tmp_path / "m.wav"), always_2d=True)
    assert len(audio) == n
    jump = np.abs(np.diff(audio[sr:3 * sr, 0]))
    assert (sr + int(np.argmax(jump))) / sr == pytest.approx(2.0, abs=0.0005)
    assert 20 * np.log10(np.max(np.abs(audio))) <= -2.0 + 0.05


def test_mux_keeps_every_picture_frame(tmp_path):
    # Picture and mix are exact by construction, so the mux must keep every frame it is given. -shortest would
    # trim this picture's surplus frame (253 -> 252) and so hide a length error instead of letting it show.
    pic, wav = tmp_path / "pic.mp4", tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=30",
                    "-frames:v", "253", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(pic)], check=True)
    sf.write(wav, np.zeros((round(8.4 * 48000), 2), np.float32), 48000, subtype="PCM_24")
    out = mux(pic, wav, tmp_path / "o.mp4", draft=True)
    frames = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                             "stream=nb_read_frames", "-of", "csv=p=0", str(out)], capture_output=True, text=True,
                            check=True).stdout.strip()
    assert frames == "253"


def test_master_keeps_a_quiet_section_quiet(tmp_path):
    # A float pre-master: a bed at about -11 LUFS with a planned -10 dB "dark beat" from 10 to 20 s, and clicks
    # above 0 dBFS. loudnorm's pass 2 cannot stay linear here (true peak + gain > TP), and its dynamic mode brings
    # this section up to -7.5 dB. The master applies the measured gain linearly and leaves the peaks to the limiter.
    sr, n = 48000, 30 * 48000
    t = np.arange(n) / sr
    bed = 0.3 * (np.sin(2 * np.pi * 220 * t) + 0.6 * np.sin(2 * np.pi * 330 * t))
    bed[10 * sr:20 * sr] *= 10 ** (-10 / 20)
    clicks = np.zeros(n)
    for at in np.arange(1.5, 30, 3.0):
        k = round(at * sr)
        clicks[k:k + 480] = 1.5 * np.random.default_rng(int(at * 10)).uniform(-1, 1, 480) * np.exp(-np.arange(480) / 80)
    x = bed + clicks
    src = tmp_path / "pre.wav"
    sf.write(src, np.stack([x, x], 1).astype(np.float32), sr, subtype="FLOAT")
    assert float(loudnorm_measure(src, -14, -2.0)["input_tp"]) > 0       # peaks above full scale, as in a real mix
    audio, _ = sf.read(master_audio(src, tmp_path / "m.wav"), always_2d=True)
    rms = lambda a, b: np.sqrt(np.mean(audio[round(a * sr):round(b * sr), 0] ** 2))
    # windows clear of the clicks (every 3 s from 1.5 s) and of the limiter's release
    loud = np.mean([rms(a, a + 1.0) for a in (2.2, 5.2, 23.2, 26.2)])
    quiet = np.mean([rms(a, a + 1.0) for a in (11.2, 14.2, 17.2)])
    assert 20 * np.log10(quiet / loud) == pytest.approx(-10, abs=0.5)
    assert aa.loudness(tmp_path / "m.wav")["I"] == pytest.approx(-14, abs=0.7)
