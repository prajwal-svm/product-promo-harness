"""Loudness, decoding, peaks, onsets, silences and spectrograms (ffmpeg + numpy)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from promo.proc import atomic_path, run


def decode_mono(path: Path, sr: int = 48000) -> np.ndarray:
    out = run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], text=False)
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def duration_s(path: Path) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path])
    return float(out.stdout.strip())


def loudness(path: Path) -> dict:
    err = run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"]).stderr
    tail = err[err.rfind("Summary:"):]

    def grab(label: str) -> float:
        m = re.search(rf"{label}\s+(-?\d+(?:\.\d+)?|-inf)", tail)
        if not m:
            raise ValueError(f"could not read {label} from ebur128 summary of {path}")
        return float("-inf") if m.group(1) == "-inf" else float(m.group(1))

    return {"I": grab("I:"), "LRA": grab("LRA:"), "TP": grab("Peak:")}


def peak_offset_s(samples: np.ndarray, sr: int) -> float:
    return float(np.argmax(np.abs(samples))) / sr


def onset_times(samples: np.ndarray, sr: int, *, hop: int = 256, win: int = 1024, z: float = 2.0) -> list[float]:
    if len(samples) < win * 2:
        return []
    n = (len(samples) - win) // hop
    frames = np.lib.stride_tricks.as_strided(
        samples, shape=(n, win), strides=(samples.strides[0] * hop, samples.strides[0]))
    spec = np.abs(np.fft.rfft(frames * np.hanning(win), axis=1))
    flux = np.maximum(0.0, np.diff(np.log1p(spec * 10), axis=0)).sum(axis=1)
    thr = flux.mean() + z * flux.std()
    out, last = [], -10**9
    min_gap = int(0.05 * sr / hop)
    for i in range(1, len(flux) - 1):
        if flux[i] > thr and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1] and i - last >= min_gap:
            a, b = i * hop, min(len(samples), (i + 1) * hop + win)
            out.append((a + int(np.argmax(np.abs(samples[a:b])))) / sr)  # the transient's peak sample
            last = i
    return out


def spectrogram(path: Path, out_png: Path, width: int = 1600, height: int = 420) -> Path:
    with atomic_path(out_png) as tmp:
        run(["ffmpeg", "-v", "error", "-y", "-i", path, "-lavfi",
             f"showspectrumpic=s={width}x{height}:legend=1:scale=log:fscale=log", tmp])
    return out_png


def silences(path: Path, noise_db: float = -50.0, min_s: float = 1.0) -> list[tuple[float, float]]:
    err = run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af",
               f"silencedetect=n={noise_db}dB:d={min_s}", "-f", "null", "-"]).stderr
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end: (-?[\d.]+)", err)]
    return list(zip(starts, ends))
