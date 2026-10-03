"""Small synthetic voice, mix and recognition fakes shared by the evidence and critique tests."""

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from promo.models import Caption

SR = 48000


def voice(seconds, seed):
    """Speech-like audio: a harmonic tone with a wandering pitch and a syllable envelope; no two takes are alike. (Noise
    would not do: AAC cannot keep it, and a real voice survives AAC with a correlation of 0.8 or more.)"""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 120 + 35 * np.sin(2 * np.pi * (0.9 + 0.13 * seed) * t + seed) + 20 * np.sin(2 * np.pi * 2.3 * t + 2 * seed)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    tone = sum(np.sin(h * phase + rng.uniform(0, 6.28)) / h for h in range(1, 14))
    return (0.4 * tone * (0.65 + 0.35 * np.sin(2 * np.pi * (3.1 + 0.2 * seed) * t + seed))).astype(np.float32)


def mix(path, parts, seconds=9.0, bed=0.01, gain_db=0.0):
    """A WAV (48 kHz mono) of a quiet noise bed with each (audio, start_s, gain) part laid on it, then `gain_db`."""
    buf = np.random.default_rng(99).standard_normal(int(seconds * SR)) * bed
    for audio, start, gain in parts:
        a = round(start * SR)
        buf[a:a + len(audio)] += audio[: len(buf) - a] * gain
    sf.write(path, (buf * 10 ** (gain_db / 20)).astype(np.float32), SR, subtype="FLOAT")
    return path


def with_audio(wav, path, seconds=8.3):
    """A 320x180 30 fps test-pattern video of `seconds` with the WAV (AAC) as its sound; no WAV, no sound."""
    audio = ["-i", str(wav), "-c:a", "aac"] if wav else ["-an"]
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc=size=320x180:rate=30:duration={seconds}", *audio, "-pix_fmt", "yuv420p", "-shortest",
                    str(path)], check=True)
    return Path(path)


def say(text, start=0.0, per_word=0.3, lag_ms=0):
    """The words of `text` as a recogniser would time them, 0.3 s a word from `start`."""
    return [Caption(text=w, startMs=round((start + i * per_word) * 1000) + lag_ms,
                    endMs=round((start + i * per_word + 0.25) * 1000) + lag_ms) for i, w in enumerate(text.split())]
