"""Fit a generated bed to the film's planned beat grid, whichever provider made it.

The film is cut to a planned grid: beat k at `origin + k * 60 / bpm` seconds from the bed's start. A provider's bed
comes back near that tempo, not on it (Lyria 3.5 played 126 BPM when asked for 120), and its first downbeat lands
wherever the model put it. `conform` measures the bed's own straight grid (promo.beats.fit_steady), time-stretches it
to the planned tempo with ffmpeg's `atempo` (this ffmpeg has no rubberband), then trims or pads its head so a
downbeat lands on the film's bar 1. `verify` measures the result against the plan. `stitch` joins beds made in
segments with an equal-power crossfade one beat long, centred on a downbeat.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

import librosa

from promo import beats as _beats
from promo.beats import STEADY_TOL_S, SteadyFit, beat_errors, fit_steady_why, kick_phase_scores
from promo.proc import atomic_path, run as run_cmd

MAX_STRETCH = 0.06      # atempo is clean for small ratios; a bed further off than this is re-rolled
SR = 48000
HEAD_PAD_MS = 250
SOUNDING_DB = 30.0      # a bar more than this far under the median bar's level is a quiet passage, not music to check
DOWNBEAT_MARGIN = 0.10  # the kick on the downbeat must beat the next beat's by this share, or the bar line is a guess
OFFSET_AGREE_S = 0.020  # an off-grid bar's hits agree on one offset to within this (a moved section moves them all)
REROLL = "re-roll it with `promo music --fresh`"


class ConformError(RuntimeError):
    """The bed cannot be fitted to the planned grid; a re-roll is the fix."""


@dataclass(frozen=True)
class Placement:
    """How one bed (or segment) was fitted: its measured tempo, the atempo factor (planned tempo over measured), the
    head shift in seconds (+ pads silence, - trims audio), the length written, and how much of it is the provider's
    audio (the rest, if any, is silence padded at the end)."""

    measured_bpm: float
    ratio: float
    shift_s: float
    seconds: float
    available_s: float


@dataclass(frozen=True)
class GridCheck:
    """The conformed bed against the planned grid, bar by bar (ruling R97).

    Each beat's error is where its clear attack lands minus where the plan puts it. A bar's error is the median of
    its beats' errors, and the bar is measured when at least 3 of its beats (all of them in a shorter bar) have a
    clear attack. `max_error_s` is the worst measured bar's (`worst_bar`, 1-based, starting at `at_s`): the gate.
    `raw_max_error_s` is the worst single beat (at `raw_at_s`): reported and warned on, not gated, because a real
    bed's stray notes land 30-40 ms off a beat that is on the grid. `unmeasured` lists (first, last) 1-based bar
    ranges that sound but have too few clear beats (a filtered breakdown); `bars_sounding` leaves out quiet passages.
    `fit` is the bed's own straight grid (fit_steady), the grid beats.json is written from.

    A bar whose strong drum hits land outside the ±60 ms each beat is searched in is off the grid, not a breakdown,
    when fewer than 3 of its beats are detected there and at least 3 strong hits elsewhere in their beats agree on
    one offset (within OFFSET_AGREE_S, around the beat cycle, so -250 and +250 ms are the same half beat). Its error
    is that offset, it is measured, and it fails the gate. Hits that do not agree (16th-note figures in a drumless
    build) leave the bar unmeasured. `off_grid` lists those bars; `worst_offset_s` is the worst bar's signed error
    (+ late)."""

    max_error_s: float
    at_s: float
    worst_bar: int
    raw_max_error_s: float
    raw_at_s: float
    bars_measured: int
    bars_sounding: int
    unmeasured: list[tuple[int, int]]
    fit: SteadyFit
    worst_offset_s: float = 0.0
    off_grid: list[tuple[int, int]] = ()

    @property
    def coverage(self) -> float:
        return round(self.bars_measured / max(1, self.bars_sounding), 3)


def stretch_ratio(measured_bpm: float, planned_bpm: float, what: str = "the bed") -> float:
    ratio = planned_bpm / measured_bpm
    if abs(ratio - 1.0) > MAX_STRETCH + 1e-9:
        raise ConformError(
            f"{what} came back at {measured_bpm:.2f} BPM, {abs(ratio - 1) * 100:.1f}% off the planned "
            f"{planned_bpm:g} BPM; a stretch beyond {MAX_STRETCH * 100:g}% would be audible, so {REROLL}")
    return ratio


def _steady(path: Path, bpm: float, beats_per_bar: int, what: str, seconds: float | None = None,
            steady_tol_s: float = STEADY_TOL_S) -> SteadyFit:
    try:
        fit, why = fit_steady_why(path, beats_per_bar, bpm_hint=bpm, seconds=seconds, steady_tol_s=steady_tol_s)
    except ValueError as e:     # no beats at all
        raise ConformError(f"{what}: {e}; {REROLL}") from e
    if fit is None:
        raise ConformError(f"{what} has no steady beat to fit to the {bpm:g} BPM grid ({why}); {REROLL}")
    return fit


def conform(src: Path, dst: Path, *, bpm: float, beats_per_bar: int, origin_s: float = 0.0,
            length_s: float | None = None, span_s: float | None = None, what: str = "the bed",
            steady_tol_s: float = STEADY_TOL_S) -> Placement:
    """Write `src` to `dst` (48 kHz stereo float) at `bpm`, with its first downbeat at `origin_s`, cut or padded to
    `length_s`. Whatever comes before that downbeat (the model's leading silence, a pickup the prompt asked it not
    to play) is trimmed; when `origin_s` is later than the downbeat, silence is padded in front instead.

    `span_s`, when given, is how much of the result is used: the tempo is measured on that much music (and at
    least 8 bars), so a provider that plays on past it, and slows into its own ending there, is not held to it.
    `steady_tol_s` is how far the bed's 2-bar windows may wander from one straight grid (a film passes one frame)."""
    bar = beats_per_bar * 60.0 / bpm
    measure = None if span_s is None else max(span_s, 8 * bar) * (1 + MAX_STRETCH) + 2 * bar
    fit = _steady(src, bpm, beats_per_bar, what, measure, steady_tol_s)
    ratio = stretch_ratio(fit.bpm, bpm, what)
    dst = Path(dst)
    stretched = dst.with_name(f".{dst.stem}.stretch.wav")
    try:
        # A quarter second of silence in front first: atempo starts its output a few tens of ms into the input,
        # which would clip a downbeat on the very first sample.
        filters = [f"adelay={HEAD_PAD_MS}:all=1"] + ([f"atempo={ratio:.8f}"] if abs(ratio - 1.0) > 1e-6 else [])
        run_cmd(["ffmpeg", "-v", "error", "-y", "-i", src, "-af", ",".join(filters), "-ar", str(SR), "-ac", "2",
                 "-c:a", "pcm_f32le", stretched])
        # atempo shifts the audio by a few tens of ms, so measure where the downbeats landed rather than predict it
        after = _steady(stretched, bpm, beats_per_bar, what,
                        None if measure is None else HEAD_PAD_MS / 1000 + max(span_s, 8 * bar) + 2 * bar,
                        steady_tol_s)
        shift = origin_s - after.downbeats[0]
        audio, _ = sf.read(stretched, dtype="float32", always_2d=True)
    finally:
        stretched.unlink(missing_ok=True)
    n_shift = round(shift * SR)
    audio = np.concatenate([np.zeros((n_shift, audio.shape[1]), np.float32), audio]) if n_shift > 0 else audio[-n_shift:]
    # the provider's music, not its trailing silence or the tail of a fade: where the sound ends
    available = _beats._sound_span(_beats._mono22k(audio, SR))[1]
    if length_s is not None:
        n = round(length_s * SR)
        audio = audio[:n] if len(audio) >= n else np.concatenate([audio, np.zeros((n - len(audio), audio.shape[1]),
                                                                                   np.float32)])
    with atomic_path(dst) as tmp:
        sf.write(tmp, audio, SR, subtype="FLOAT", format="WAV")
    # the shift as the delivered audio sees it: the head pad came in before the stretch
    return Placement(measured_bpm=fit.bpm, ratio=round(ratio, 6), shift_s=round(n_shift / SR + HEAD_PAD_MS / 1000 / ratio, 6),
                     seconds=round(len(audio) / SR, 6), available_s=round(available, 6))


def shift_head(path: Path, seconds: float) -> None:
    """Move `path`'s audio later by `seconds` (silence in front) or, for a negative value, earlier (trim its head)."""
    audio, _ = sf.read(path, dtype="float32", always_2d=True)
    n = round(abs(seconds) * SR)
    audio = np.concatenate([np.zeros((n, audio.shape[1]), np.float32), audio]) if seconds > 0 else audio[n:]
    with atomic_path(Path(path)) as tmp:
        sf.write(tmp, audio, SR, subtype="FLOAT", format="WAV")


def downbeat_phase(path: Path, *, bpm: float, beats_per_bar: int, start_s: float,
                   end_s: float | None = None) -> tuple[int, float]:
    """With planned bar lines at `start_s` + m bars: (which beat of the bar, 0 .. beats_per_bar-1, has the most kick
    energy over [start_s, end_s); the margin, its lead over the runner-up as a share of it). A margin under
    DOWNBEAT_MARGIN is a guess (a kick on every beat)."""
    y = _beats._load(path, end_s)
    period = 60.0 / bpm
    stop = len(y) / _beats.SR if end_s is None else min(end_s, len(y) / _beats.SR)
    times = start_s + np.arange(max(0, int((stop - start_s) / period))) * period
    if len(times) < beats_per_bar:
        return 0, 0.0
    scores = kick_phase_scores(y, times, beats_per_bar)
    order = np.argsort(scores)[::-1]
    best = float(scores[order[0]])
    return int(order[0]), round((best - float(scores[order[1]])) / best, 4) if best > 0 else 0.0


def fit_length(src: Path, dst: Path, length_s: float) -> None:
    """Copy `src` to `dst` cut, or padded with silence, to exactly `length_s` (48 kHz stereo float)."""
    audio, _ = sf.read(src, dtype="float32", always_2d=True)
    n = round(length_s * SR)
    audio = audio[:n] if len(audio) >= n else np.concatenate([audio, np.zeros((n - len(audio), audio.shape[1]),
                                                                               np.float32)])
    with atomic_path(Path(dst)) as tmp:
        sf.write(tmp, audio, SR, subtype="FLOAT", format="WAV")


def _ranges(bars: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for b in bars:
        if out and b == out[-1][1] + 1:
            out[-1] = (out[-1][0], b)
        else:
            out.append((b, b))
    return out


def _shared_offset(offsets: np.ndarray, period: float, need: int) -> float | None:
    """The one offset at least `need` of `offsets` agree on (within OFFSET_AGREE_S, measured around the beat
    cycle), wrapped into (-period/2, period/2], or None when they do not agree. The centre is the offset most others
    agree with, so one stray hit does not drag it."""
    offsets = np.asarray(offsets, dtype=float)

    def dev(centre: float) -> np.ndarray:
        return (offsets - centre + period / 2) % period - period / 2

    best = max(offsets, key=lambda c: int((np.abs(dev(c)) <= OFFSET_AGREE_S).sum()), default=None)
    if best is None:
        return None
    close = np.abs(dev(best)) <= OFFSET_AGREE_S
    if int(close.sum()) < need:
        return None
    moved = best + float(np.mean(dev(best)[close]))
    return float((moved + period / 2) % period - period / 2)


def check_grid(path: Path, *, bpm: float, beats_per_bar: int, origin_s: float = 0.0,
               seconds: float | None = None, steady_tol_s: float = STEADY_TOL_S) -> GridCheck:
    """Measure `path` against the planned grid (beat k at origin_s + k * 60 / bpm), bar by bar; see GridCheck.
    `seconds` limits it to the file's first so many. Raises ConformError when the bed has no steady beat (within
    `steady_tol_s`) or no bar clear enough to measure."""
    fit = _steady(path, bpm, beats_per_bar, "the conformed bed", seconds, steady_tol_s)
    period = 60.0 / bpm
    n = max(0, int(np.floor((fit.grid[-1] + period / 2 - origin_s) / period)) + 1)
    planned = origin_s + np.arange(n) * period
    errors, off, level = beat_errors(path, planned, bpm, seconds)
    need = min(3, beats_per_bar)
    bars = [slice(b * beats_per_bar, min(n, (b + 1) * beats_per_bar)) for b in range(-(-n // beats_per_bar))]
    bar_db = np.array([10 * np.log10(float(np.mean(10 ** (level[sl] / 10)))) for sl in bars])
    loud = bar_db >= float(np.median(bar_db)) - SOUNDING_DB if len(bars) else np.zeros(0, bool)
    bar_err: dict[int, float] = {}
    unmeasured, off_grid = [], []
    for b, sl in enumerate(bars):
        seen = errors[sl][np.isfinite(errors[sl])]
        away = off[sl][np.isfinite(off[sl])]
        if len(seen) >= min(need, sl.stop - sl.start):
            bar_err[b] = float(np.median(seen))
        elif len(away) > len(seen) and (moved := _shared_offset(away, period, min(need, sl.stop - sl.start))) is not None:
            bar_err[b] = moved                         # strong drums, all moved by one offset: off the grid
            off_grid.append(b + 1)
        elif loud[b]:
            unmeasured.append(b + 1)
    if not bar_err:
        raise ConformError(f"the conformed bed has no bar with {need} clear beats to check against the grid; "
                           f"{REROLL}")
    worst = max(bar_err, key=lambda b: abs(bar_err[b]))
    seen_all = np.flatnonzero(np.isfinite(errors))
    raw = int(seen_all[np.argmax(np.abs(errors[seen_all]))]) if len(seen_all) else 0
    return GridCheck(max_error_s=round(abs(bar_err[worst]), 6), at_s=round(origin_s + worst * beats_per_bar * period, 3),
                     worst_bar=worst + 1,
                     raw_max_error_s=round(abs(float(errors[raw])), 6) if len(seen_all) else 0.0,
                     raw_at_s=round(float(planned[raw]), 3), bars_measured=len(bar_err),
                     bars_sounding=len(bar_err) + len(unmeasured), unmeasured=_ranges(unmeasured), fit=fit,
                     worst_offset_s=round(bar_err[worst], 6), off_grid=_ranges(off_grid))


def stitch(parts: list[tuple[Path, float]], seams_s: list[float], dst: Path, *, beat_s: float,
           length_s: float) -> None:
    """Lay each (file, start time) part on one timeline and join part i to part i+1 at `seams_s[i]` with an
    equal-power crossfade one beat long, centred on the seam. Every part is 48 kHz stereo float."""
    if len(seams_s) != len(parts) - 1:
        raise ValueError("one seam between each pair of parts")
    n_total = round(length_s * SR)
    out = np.zeros((n_total, 2), np.float32)
    half = round(beat_s * SR / 2)
    for i, (path, start) in enumerate(parts):
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != SR:
            raise ValueError(f"{path}: {sr} Hz, expected {SR}")
        a0 = round(start * SR)
        gain = np.ones(len(audio), np.float32)
        idx = np.arange(len(audio)) + a0                    # timeline sample of each part sample
        if i > 0:     # fade in across the seam before this part
            c = round(seams_s[i - 1] * SR)
            x = np.clip((idx - (c - half)) / (2 * half), 0.0, 1.0)
            gain *= np.sin(0.5 * np.pi * x).astype(np.float32)
        if i < len(parts) - 1:   # fade out across the seam after it
            c = round(seams_s[i] * SR)
            x = np.clip((idx - (c - half)) / (2 * half), 0.0, 1.0)
            gain *= np.cos(0.5 * np.pi * x).astype(np.float32)
        lo, hi = max(0, a0), min(n_total, a0 + len(audio))
        if hi > lo:
            out[lo:hi] += audio[lo - a0:hi - a0] * gain[lo - a0:hi - a0, None]
    with atomic_path(Path(dst)) as tmp:
        sf.write(tmp, out, SR, subtype="FLOAT", format="WAV")


_BAND_EDGES = (40, 150, 400, 1000, 2500, 6000, 16000)


def _db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(x ** 2) + 1e-12))


def _bands_db(x: np.ndarray) -> np.ndarray:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.array([10 * np.log10(spec[(f >= lo) & (f < hi)].sum() + 1e-12)
                     for lo, hi in zip(_BAND_EDGES, _BAND_EDGES[1:])])


def seam_report(path: Path, seam_s: float, *, beat_s: float, bar_s: float, bars: int = 3) -> dict:
    """How a join sounds, measured.

    `level_dip_db`: the quietest one-beat window through the crossfade against the median of the same windows in
    the `bars` bars either side (negative is a dip; one bar alone varies too much to judge by). `level_step_db`: the
    bars after the seam against the bars before it. `spectral_jump_db`: the largest change of six bands' levels
    from the bars before the seam to the bars after it, where a restart, a new key or a thinner mix shows.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)

    def win(t0: float, length: float) -> np.ndarray:
        return mono[max(0, round(t0 * sr)):max(0, round((t0 + length) * sr))]

    def side(t0: float, sign: int) -> list[np.ndarray]:
        return [w for k in range(1, bars + 1) if len(w := win(t0 + sign * k * bar_s, beat_s)) == round(beat_s * sr)]

    dips = []
    for o in np.arange(-0.5, 0.5, 0.125) * beat_s:
        t0 = seam_s + o - beat_s / 2
        near = [_db(w) for w in side(t0, -1) + side(t0, +1)]
        if near:
            dips.append(_db(win(t0, beat_s)) - float(np.median(near)))
    span = bars * bar_s
    before, after = win(seam_s - span, span), win(seam_s, span)
    full = round(span * sr)
    if len(before) == full and len(after) == full:
        step = _db(after) - _db(before)
        jump = float(np.max(np.abs(_bands_db(after) - _bands_db(before))))
    else:
        step = jump = 0.0
    return {"seam_s": round(seam_s, 4), "level_dip_db": round(min(dips), 2) if dips else 0.0,
            "level_step_db": round(step, 2), "spectral_jump_db": round(jump, 2)}


# Krumhansl-Kessler key profiles (C major, C minor), rotated to each tonic.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_TONICS = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


def estimate_key(path: Path) -> str | None:
    """The key a stretch of music is in ("Ab major"), from the chroma of its harmonic part, or None when the audio
    cannot be read or is too short to tell. A later segment is asked for the key its first segment came back in, so
    two calls do not meet at the seam in different keys."""
    try:
        y, sr = librosa.load(str(path), sr=22050, mono=True)
        if len(y) < 4 * sr or float(np.sqrt(np.mean(y ** 2))) < 1e-4:
            return None                       # under 4 s, or silence: nothing to hear a key in
        chroma = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(y), sr=sr, hop_length=2048).mean(axis=1)
    except (ValueError, RuntimeError, librosa.util.exceptions.ParameterError):
        return None
    if not np.isfinite(chroma).all() or float(chroma.std()) == 0.0:
        return None
    scores = [(float(np.corrcoef(np.roll(profile, i), chroma)[0, 1]), f"{_TONICS[i]} {mode}")
              for i in range(12) for profile, mode in ((_MAJOR, "major"), (_MINOR, "minor"))]
    return max(scores)[1]
