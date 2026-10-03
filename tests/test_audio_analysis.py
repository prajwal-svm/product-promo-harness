import subprocess

import numpy as np
import pytest

from promo import audio_analysis as aa


@pytest.fixture
def sine(tmp_path):
    p = tmp_path / "sine 🎵.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=1000:duration=5",
                    "-ar", "48000", str(p)], check=True)
    return p


def test_loudness_of_known_sine(sine):
    m = aa.loudness(sine)
    # ffmpeg's sine source has amplitude 1/8: peak -18.06 dBFS, a 1 kHz sine reads 3.01 dB below its peak in LUFS
    assert m["I"] == pytest.approx(-21.1, abs=0.5)
    assert m["TP"] == pytest.approx(-18.1, abs=0.5)


def test_decode_and_duration(sine):
    x = aa.decode_mono(sine, sr=16000)
    assert x.dtype == np.float32 and len(x) == pytest.approx(80000, abs=50)
    assert aa.duration_s(sine) == pytest.approx(5.0, abs=0.01)


def test_peak_offset():
    x = np.zeros(4800, dtype=np.float32)
    x[1200] = -0.9
    assert aa.peak_offset_s(x, 48000) == pytest.approx(0.025)


def test_onsets_find_clicks():
    sr = 48000
    x = np.zeros(sr * 3, dtype=np.float32)
    for t in (0.5, 1.25, 2.0):
        i = int(t * sr)
        x[i:i + 480] = np.hanning(480) * 0.8
    found = aa.onset_times(x, sr)
    for t in (0.5, 1.25, 2.0):
        assert min(abs(f - t) for f in found) < 0.012


def test_spectrogram_and_silences(tmp_path, sine):
    out = aa.spectrogram(sine, tmp_path / "s.png")
    assert out.exists() and out.stat().st_size > 1000
    gap = tmp_path / "gap.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=duration=1:sample_rate=48000",
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono:d=2",
                    "-f", "lavfi", "-i", "sine=duration=1:sample_rate=48000",
                    "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1", str(gap)], check=True)
    s = aa.silences(gap, min_s=1.0)
    assert len(s) == 1 and s[0][0] == pytest.approx(1.0, abs=0.05) and s[0][1] == pytest.approx(3.0, abs=0.05)
