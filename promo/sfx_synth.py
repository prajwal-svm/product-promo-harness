"""Deterministic procedural sound effects (numpy)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

from promo.proc import atomic_path

SR = 48000
KINDS = ("click", "tick", "key", "pop", "thump", "whoosh", "riser", "chime", "impact", "sub_boom")
MIN_DURATION_S = 0.001
MAX_DURATION_S = 60.0
MIN_PITCH_HZ = 20.0
MAX_PITCH_HZ = 12000.0  # a chime's top partial is twice its pitch, which must stay under Nyquist (SR / 2)


class SfxError(ValueError):
    """A sound that cannot be made: an unknown kind, an impossible parameter, a bad pack entry or bad audio."""


def _t(duration: float) -> np.ndarray:
    return np.arange(int(duration * SR), dtype=np.float64) / SR


def _norm(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x)))
    return (x * (peak / m) if m > 0 else x).astype(np.float32)


def _lowpass(x: np.ndarray, n: int) -> np.ndarray:
    k = np.ones(max(1, n)) / max(1, n)
    return np.convolve(x, k, mode="same")


def _check_pitch(name: str, hz: float) -> None:
    if not (isinstance(hz, (int, float)) and math.isfinite(hz) and MIN_PITCH_HZ <= hz <= MAX_PITCH_HZ):
        raise SfxError(f"{name} must be between {MIN_PITCH_HZ:g} and {MAX_PITCH_HZ:g} Hz (got {hz!r})")


def _pair(name: str, pair) -> tuple[float, float]:
    try:
        lo, hi = pair
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError) as e:
        raise SfxError(f"{name} takes two numbers [low, high] (got {pair!r})") from e
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise SfxError(f"{name} takes two finite numbers (got {pair!r})")
    return lo, hi


def _value_pitch(value: float, value_range, pitch_range) -> float:
    lo, hi = _pair("value_range", value_range)
    p_lo, p_hi = _pair("pitch_range", pitch_range)
    _check_pitch("pitch_range", p_lo)
    _check_pitch("pitch_range", p_hi)
    frac = 0.0 if hi == lo else min(1.0, max(0.0, (value - lo) / (hi - lo)))
    return float(np.exp(np.log(p_lo) + frac * (np.log(p_hi) - np.log(p_lo))))


def synth(kind: str, *, seed: int = 0, duration: float | None = None, pitch: float | None = None,
          value: float | None = None, value_range=(0.0, 100.0), pitch_range=(1000.0, 4400.0)) -> np.ndarray:
    """One sound as float32 mono at SR, peak 0.9. The same arguments always give the same samples.

    `value` (on `value_range`) sets the pitch on a log scale across `pitch_range`, unless `pitch` is given.
    """
    if kind not in KINDS:
        raise SfxError(f"unknown sfx kind '{kind}' (one of {', '.join(KINDS)})")
    if duration is not None and not (math.isfinite(duration) and MIN_DURATION_S <= duration <= MAX_DURATION_S):
        raise SfxError(f"duration must be between {MIN_DURATION_S:g} and {MAX_DURATION_S:g} s (got {duration!r})")
    if pitch is not None:
        _check_pitch("pitch", pitch)
    rng = np.random.default_rng(seed)
    if value is not None and pitch is None:
        pitch = _value_pitch(value, value_range, pitch_range)
    if kind == "click":
        t = _t(duration or 0.05)
        x = np.sin(2 * np.pi * (pitch or 1800) * t) * np.exp(-t * 90) + 0.05 * rng.standard_normal(len(t)) * np.exp(-t * 200)
    elif kind == "tick":
        t = _t(duration or 0.04)
        x = np.sin(2 * np.pi * (pitch or 2400) * t) * np.exp(-t * 120)
    elif kind == "key":
        t = _t(duration or 0.08)
        noise = np.diff(rng.standard_normal(len(t) + 1))
        x = noise * np.exp(-t * 60) * 0.6 + np.sin(2 * np.pi * 180 * t) * np.exp(-t * 40)
    elif kind == "pop":
        t = _t(duration or 0.15)
        f = (pitch or 600) + 900 * t / t[-1]
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 30)
    elif kind == "thump":
        t = _t(duration or 0.5)
        f = 90 - 60 * t / t[-1]
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)
    elif kind == "whoosh":
        d = duration or 0.35
        t = _t(d)
        noise = _lowpass(rng.standard_normal(len(t)), 6)
        rise = np.clip(t / (0.8 * d), 0, 1) ** 2
        fall = np.exp(-np.clip(t - 0.8 * d, 0, None) * 40)
        x = noise * rise * fall
    elif kind == "riser":
        d = duration or 1.5
        t = _t(d)
        f = 200 + 1800 * (t / d) ** 2
        x = (np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.6 + _lowpass(rng.standard_normal(len(t)), 3) * 0.4) * (t / d) ** 2
    elif kind == "chime":
        t = _t(duration or 1.2)
        base = pitch or 880
        x = sum(np.sin(2 * np.pi * base * m * t) / m for m in (1, 1.5, 2)) * np.exp(-t * 3)
    elif kind == "impact":
        t = _t(duration or 0.8)
        x = rng.standard_normal(len(t)) * np.exp(-t * 25) * 0.5 + np.sin(2 * np.pi * 55 * t) * np.exp(-t * 6)
    else:  # sub_boom
        t = _t(duration or 1.5)
        f = 55 - 15 * t / t[-1]
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3)
    return _norm(np.asarray(x, dtype=np.float64))


def write_wav(samples: np.ndarray, path: Path, sr: int = SR) -> Path:
    """Write 24-bit WAV through a temp file, so a failure leaves no half-written sound.

    The file is opened here rather than by libsndfile, so an unwritable place is an OSError.
    """
    with atomic_path(Path(path)) as tmp:
        with open(tmp, "wb") as f:
            sf.write(f, samples, sr, subtype="PCM_24", format="WAV")
    return Path(path)
