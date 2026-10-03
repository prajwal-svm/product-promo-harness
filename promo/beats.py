"""Tempo, beats, onsets and real downbeats of a music bed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import maximum_filter1d

from promo.models import BeatGrid
from promo.proc import atomic_path

HOP = 512
SR = 22050
# The steady-grid fit reads attacks on a 2.9 ms hop. A spectral-flux peak comes about half an FFT window before
# the attack it marks; ATTACK_BIAS_S puts the grid back on the attacks (calibrated on synthetic kicks and snares).
FINE_HOP = 64
FINE_FFT = 512
ATTACK_BIAS_S = 0.0079
# A bed is steady when its 2-bar windows keep one tempo: at most STEADY_OFF of the windows with sound lose the
# beat period (their onsets, folded at the period, peak less than PERIODIC_PEAK times their mean: a tempo change or
# no beat), at least STEADY_COVERAGE of all windows have clear beats to measure, and 90% of those sit within
# the tolerance of the grid (each window's median attack). A single attack may stray further than a window's median:
# a syncopated note, or atempo's overlap-add moving a transient by 10-15 ms. The caller sets the tolerance (a film
# passes one frame: a wander under a frame is steady enough to cut to); STEADY_TOL_S is the default for one that does
# not.
STEADY_TOL_S = 0.010
STEADY_OFF = 0.15
STEADY_COVERAGE = 0.5
PERIODIC_PEAK = 1.6
QUIET = 0.3            # a window with less than this share of the median window's onset energy is a quiet passage
WINDOW_BARS = 2


def _refined_bpm(beat_times: np.ndarray, coarse_bpm: float) -> float:
    """librosa's tempo is quantised to integer-frame lags (117.45 or 123.05 for a true 120), which is
    up to 2.5% off. The tracked beats average out that quantisation, so measure the tempo on them.
    Gaps far from the coarse period (a skipped or doubled beat) are left out of the mean."""
    gaps = np.diff(beat_times)
    period = 60.0 / coarse_bpm
    kept = gaps[np.abs(gaps - period) < 0.15 * period]
    return 60.0 / float(kept.mean()) if len(kept) >= 3 else coarse_bpm


def _regularize(beats: list[float], period: float) -> list[float]:
    """Make the beat list a true beat count: entry i is beat i. The tracker can drop a beat (a gap of
    about 2 periods) or double one (two beats under half a period apart); either shifts every later
    index by one. Fill a gap of g with round(g/period)-1 evenly spaced beats, and drop the later of
    two beats that are too close."""
    out: list[float] = []
    for t in beats:
        if out:
            prev = out[-1]
            gap = t - prev
            if gap < 0.5 * period:
                continue
            if gap > 1.5 * period:
                n = round(gap / period)
                out.extend([prev + k * gap / n for k in range(1, n)])
        out.append(float(t))
    return out


def _load(path: Path, seconds: float | None = None) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, mono=True, duration=seconds)
    return y


def _track(y: np.ndarray, path: Path, beats_per_bar: int) -> tuple[float, np.ndarray, np.ndarray]:
    """librosa's tracker: (refined bpm, regularised beat times, onset envelope)."""
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=HOP, units="frames")
    bpm = float(np.atleast_1d(tempo)[0])
    if len(beat_frames) < beats_per_bar:
        raise ValueError(f"{path}: only {len(beat_frames)} beats found; is this music?")
    beat_times = librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP)
    bpm = _refined_bpm(beat_times, bpm)
    beat_times = np.array(_regularize(list(beat_times), 60.0 / bpm))
    if len(beat_times) < beats_per_bar:
        raise ValueError(f"{path}: only {len(beat_times)} beats found; is this music?")
    return bpm, beat_times, env


# --- the steady grid ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SteadyFit:
    """One straight beat grid that a steady bed's attacks sit on.

    `grid` runs from the file's first beat to its last (where it has sound); `downbeats` is every bar's beat 1.
    `windows` holds (start time, median attack offset from the grid) for each 2-bar window with clear attacks;
    `coverage` is the share of windows that had them.
    """

    bpm: float
    grid: list[float]
    downbeats: list[float]
    downbeat_index: int
    windows: list[tuple[float, float]]
    coverage: float

    @property
    def period(self) -> float:
        return 60.0 / self.bpm


def _flux(band: np.ndarray) -> np.ndarray:
    """Spectral flux of a band: the rise of its total log energy. A whole band rises on a kick or a snare; one
    loud bin of a held chord, or a hat's hiss, barely moves it."""
    e = np.log1p(1000.0 * band.sum(axis=0))
    return np.maximum(0.0, np.diff(e, prepend=e[0]))


def _norm(x: np.ndarray) -> np.ndarray:
    pos = x[x > 0]
    return x / (float(np.median(pos)) if len(pos) else 1.0)


def _envelopes(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fine onset envelopes: (kick and snare, everything). Hats are left out of the first: the beat is where the
    kick and snare play, and a loud hat on every "and" must not pull the grid half a beat over."""
    spec = np.abs(librosa.stft(y, n_fft=FINE_FFT, hop_length=FINE_HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=FINE_FFT)
    def band(lo: float, hi: float) -> np.ndarray:
        return _norm(_flux(spec[(freqs >= lo) & (freqs < hi)]))

    def clip(x: np.ndarray) -> np.ndarray:   # one huge onset (music starting out of silence) must not outweigh the beat
        return np.minimum(x, 3.0 * float(np.percentile(x, 99)))

    # the mid range in three bands, so a snare or a clap still shows over a loud held chord in another band
    beat = band(0, 150) + (band(150, 500) + band(500, 1500) + band(1500, 5000)) / 3
    return clip(beat), clip(beat + band(5000, SR))


def _autocorr_period(env: np.ndarray, period: float, seconds: float) -> float:
    """The beat period from where the envelope repeats at lags of 8, 16, 32 ... beats (each lag searched within a
    fifth of a beat of the last estimate), fitted through the origin. Exact to a few parts per million."""
    fr = SR / FINE_HOP
    x = env - env.mean()
    n_all = len(x)
    pts: list[tuple[int, float]] = []
    for n in (8, 16, 32, 64, 128, 256):
        if n * period > 0.5 * seconds:
            break
        centre = n * period * fr
        w = max(3, int(0.2 * period * fr))
        lo, hi = max(1, int(centre - w)), min(n_all - 2, int(centre + w))
        if hi - lo < 3:
            break
        ac = np.array([np.dot(x[:-lag], x[lag:]) / (n_all - lag) for lag in range(lo, hi + 1)])
        k = int(np.argmax(ac))
        d = 0.0
        if 0 < k < len(ac) - 1:
            den = ac[k - 1] - 2 * ac[k] + ac[k + 1]
            d = 0.5 * (ac[k - 1] - ac[k + 1]) / den if den < 0 else 0.0
        pts.append((n, (lo + k + d) / fr))
        period = pts[-1][1] / n
    if not pts:
        return period
    ns = np.array([p[0] for p in pts], dtype=float)
    lags = np.array([p[1] for p in pts])
    return float(np.dot(ns, lags) / np.dot(ns, ns))


def _fold_phase(env: np.ndarray, period: float, bins: int = 96) -> float:
    """Where in the beat the envelope is strongest, in seconds after a multiple of the period."""
    t = np.arange(len(env)) * FINE_HOP / SR
    idx = (((t / period) % 1.0) * bins).astype(int) % bins
    hist = np.bincount(idx, weights=env, minlength=bins)
    smooth = np.convolve(np.r_[hist[-2:], hist, hist[:2]], np.ones(5) / 5, mode="valid")
    return (int(np.argmax(smooth)) + 0.5) / bins * period


def _attacks(env: np.ndarray, grid: np.ndarray, window: float) -> tuple[np.ndarray, np.ndarray]:
    """The strongest attack within `window` of each grid time (sub-hop, by a parabola), and its strength."""
    fr = SR / FINE_HOP
    times, strength = np.full(len(grid), np.nan), np.zeros(len(grid))
    for i, t in enumerate(grid):
        a, b = max(1, int(np.floor((t - window) * fr))), min(len(env) - 2, int(np.ceil((t + window) * fr)))
        if b <= a:
            continue
        j = a + int(np.argmax(env[a:b + 1]))
        left, mid, right = env[j - 1], env[j], env[j + 1]
        den = left - 2 * mid + right
        d = 0.5 * (left - right) / den if den < 0 else 0.0
        times[i], strength[i] = (j + d) / fr + ATTACK_BIAS_S, mid
    return times, strength


def _mono22k(audio: np.ndarray, sr: int) -> np.ndarray:
    """(n, channels) float audio at `sr` as the mono 22.05 kHz signal the measurements read."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    return librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=SR) if sr != SR else mono


def _sound_span(y: np.ndarray) -> tuple[float, float]:
    """First and last moment the file is within 40 dB of its loudest 93 ms."""
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    if not len(rms) or rms.max() <= 0:
        return 0.0, 0.0
    loud = np.flatnonzero(rms >= rms.max() * 10 ** (-40 / 20))
    return float(loud[0] * HOP / SR), float((loud[-1] + 1) * HOP / SR)


_LEVELS = (1.0, 2.0, 0.5, 1.5, 2 / 3, 3.0, 1 / 3, 4 / 3, 0.75)


def _nearest_level(bpm: float, hint: float | None) -> float:
    """The tracker's tempo, moved to the metrical level (x2, x1/2, x3/2 ...) nearest the tempo the bed was asked
    for: a tracker that counts a 126 BPM groove at 84 or 252 has the right beat at the wrong level."""
    if not hint:
        return bpm
    return min((bpm * f for f in _LEVELS), key=lambda b: abs(np.log(b / hint)))


def _fit(y: np.ndarray, bpm: float, beats_per_bar: int, bpm_hint: float | None = None,
         steady_tol_s: float = STEADY_TOL_S) -> tuple[SteadyFit | None, str]:
    """(the bed's straight grid, ""), or (None, why it has none). 90% of the 2-bar windows must sit within
    `steady_tol_s` of the grid."""
    seconds = len(y) / SR
    bpm = _nearest_level(bpm, bpm_hint)
    beat_env, all_env = _envelopes(y)
    period = _autocorr_period(all_env, 60.0 / bpm, seconds)
    phase = _fold_phase(beat_env, period) + ATTACK_BIAS_S
    start, end = _sound_span(y)
    window = min(0.12 * period, 0.06)

    def grid_for(origin: float, p: float) -> tuple[np.ndarray, np.ndarray]:
        k = np.arange(int(np.ceil((max(start - p / 4, -0.005) - origin) / p)), int(np.floor((end - origin) / p)) + 1)
        return k, origin + k * p

    k, grid = grid_for(phase, period)
    if len(grid) < 4 * beats_per_bar:
        return None, "it is under 4 bars long"
    for _ in range(2):    # least squares on the clear attacks, twice: the second pass starts from the first's grid
        times, strength = _attacks(beat_env, grid, window)
        clear = strength >= 0.75 * float(np.median(strength))
        keep = clear & np.isfinite(times)
        if keep.sum() < 2 * beats_per_bar:
            return None, "it has too few clear beats to fit a grid to"
        for _ in range(3):
            slope, icpt = np.polyfit(k[keep], times[keep], 1)
            res = times - (icpt + k * slope)
            spread = 1.4826 * float(np.median(np.abs(res[keep])))
            keep = keep & (np.abs(res) <= max(4 * spread, 0.004))
            if keep.sum() < 2 * beats_per_bar:
                return None, "too few of its beats agree on one grid"
        origin = float(icpt) % float(slope)
        k, grid = grid_for(origin, float(slope))
        period = float(slope)
    times, strength = _attacks(beat_env, grid, window)
    clear = (strength >= 0.75 * float(np.median(strength))) & np.isfinite(times)
    size = WINDOW_BARS * beats_per_bar
    fr = SR / FINE_HOP
    spans = [(i, slice(max(0, int(grid[i] * fr)), int(min(grid[min(i + size, len(grid) - 1)], seconds) * fr)))
             for i in range(0, len(grid) - size // 2, size)]
    energy = [float(beat_env[f].sum()) for _, f in spans]
    median_energy = float(np.median(energy)) if energy else 0.0
    windows, total, sounding, off_at = [], len(spans), 0, []
    for (i, frames), e in zip(spans, energy):
        if e < QUIET * median_energy:
            continue                                   # a quiet passage: nothing to judge
        sounding += 1
        t = np.arange(frames.start, frames.start + len(beat_env[frames])) / fr
        fold = np.bincount(((t / period % 1.0) * 24).astype(int) % 24, weights=beat_env[frames], minlength=24)
        if fold.max() < PERIODIC_PEAK * max(float(fold.mean()), 1e-12):
            off_at.append(float(grid[i]))              # onsets, but not at this period: the tempo moved
            continue
        sl = slice(i, i + size)
        if int(clear[sl].sum()) * 2 >= min(size, len(grid) - i):
            windows.append((round(float(grid[i]), 4), float(np.median((times[sl] - grid[sl])[clear[sl]]))))
    errs = np.abs([w[1] for w in windows])
    on = len(windows)
    if len(off_at) > STEADY_OFF * max(1, sounding):
        where = ", ".join(f"{t:.0f}s" for t in off_at[:8]) + (" ..." if len(off_at) > 8 else "")
        return None, (f"its 2-bar windows lose the beat period in {len(off_at)} of {sounding} (at {where}): a tempo "
                      f"change, or a long beatless passage")
    if on < min(3, total) or on < STEADY_COVERAGE * total:
        return None, f"only {on} of its {total} 2-bar windows have clear beats to measure"
    if float(np.percentile(errs, 90)) > steady_tol_s:
        return None, (f"its beats wander {float(np.percentile(errs, 90)) * 1000:.0f} ms from one straight grid "
                      f"(90th percentile of its 2-bar windows)")
    coverage = on / max(1, total)
    first = _kick_phase(y, grid, beats_per_bar)
    beats = [round(float(t), 4) for t in grid]
    return SteadyFit(bpm=round(60.0 / period, 4), grid=beats, downbeats=beats[first::beats_per_bar],
                     downbeat_index=first, windows=windows, coverage=round(coverage, 3)), ""


def fit_steady_why(path: Path, beats_per_bar: int = 4, bpm_hint: float | None = None,
                   seconds: float | None = None, steady_tol_s: float = STEADY_TOL_S) -> tuple[SteadyFit | None, str]:
    """fit_steady, with the reason when there is no steady grid (for the error a person reads)."""
    y = _load(path, seconds)
    bpm, _, _ = _track(y, path, beats_per_bar)
    return _fit(y, bpm, beats_per_bar, bpm_hint, steady_tol_s)


def fit_steady(path: Path, beats_per_bar: int = 4, bpm_hint: float | None = None,
               seconds: float | None = None, steady_tol_s: float = STEADY_TOL_S) -> SteadyFit | None:
    """The bed's one straight grid, or None when it has no steady tempo (or too few clear beats to tell).
    `bpm_hint` (the tempo the bed was asked for) picks the metrical level; `seconds` measures only the file's first
    so many seconds; `steady_tol_s` is how far (90% of) its 2-bar windows may wander from the straight grid. Raises
    ValueError, as analyze does, when no beats can be found at all."""
    return fit_steady_why(path, beats_per_bar, bpm_hint, seconds, steady_tol_s)[0]


def kick_phase_scores(y: np.ndarray, beat_times: np.ndarray, beats_per_bar: int) -> np.ndarray:
    """Mean kick (low-band, 150 Hz and down) onset strength at each beat of the bar, counting from the first of
    `beat_times`: the downbeat is the beat with the most."""
    beat_frames = np.round(np.asarray(beat_times) * SR / HOP).astype(int)
    spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    low = spec[freqs <= 150].sum(axis=0)
    low_flux = np.maximum(0.0, np.diff(low, prepend=low[0]))
    beat_frames = np.clip(beat_frames, 0, len(low_flux) - 1)
    scores = []
    for phase in range(beats_per_bar):
        total = 0.0
        for f in beat_frames[phase::beats_per_bar]:
            total += float(low_flux[max(0, f - 2):f + 3].max())
        scores.append(total / max(1, len(beat_frames[phase::beats_per_bar])))
    return np.array(scores)


def _kick_phase(y: np.ndarray, beat_times: np.ndarray, beats_per_bar: int) -> int:
    """Which beat of the bar (0 .. beats_per_bar-1, counting from the first beat) has the most kick energy."""
    return int(np.argmax(kick_phase_scores(y, beat_times, beats_per_bar)))


DETECT_REL = 0.4    # a beat is detected when its drum transient is at least this share of the median beat's


def beat_errors(path: Path, planned: np.ndarray, bpm: float,
                seconds: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each planned beat time: (how far its detected attack lands from it, NaN where none is detected; how far
    the nearest strong drum hit elsewhere in its beat lands, for a beat with none detected, else NaN; the level in
    dB over the beat, for telling a quiet passage from a sounding one).

    The attack's time is read as fit_steady reads it: the strongest kick-and-snare onset within 0.12 beat (60 ms at
    most). It counts as detected when its drum transient (the linear rise of 0-5 kHz energy) is at least DETECT_REL
    of the median beat's. Soft or filtered drums (a breakdown) fall under that, and their smeared attacks read
    15-20 ms late, so they are left unmeasured rather than judged. But drums that are strong and simply land outside
    that window (a section 100 ms, or half a beat, off the grid) must not pass for a breakdown: for a beat with no
    detected attack, the whole beat period (half a beat either way) is searched for a hit that strong, and the
    nearest one's offset is the second value."""
    y = _load(path, seconds)
    beat_env, _ = _envelopes(y)
    planned = np.asarray(planned, dtype=float)
    if not len(planned):
        return np.zeros(0), np.zeros(0), np.zeros(0)
    period = 60.0 / bpm
    window = min(0.12 * period, 0.06)
    times, _ = _attacks(beat_env, planned, window)
    spec = np.abs(librosa.stft(y, n_fft=FINE_FFT, hop_length=FINE_HOP))
    energy = spec[librosa.fft_frequencies(sr=SR, n_fft=FINE_FFT) < 5000].sum(axis=0)
    rise = np.maximum(0.0, np.diff(energy, prepend=energy[0]))
    fr = SR / FINE_HOP
    transient = np.array([float(rise[max(0, int(t * fr) - 3):int(t * fr) + 4].max()) if np.isfinite(t)
                          and int(t * fr) - 3 < len(rise) else 0.0 for t in times])
    strong = DETECT_REL * float(np.median(transient))
    detected = np.isfinite(times) & (transient >= strong)
    # strong drum hits anywhere: onsets (the kick-and-snare onset envelope's peaks, each the largest within 50 ms,
    # so a noisy decay's wobbles are not hits) whose transient is as strong as a detected beat's, timed as above
    span = int(0.05 * fr)
    candidates = np.flatnonzero((beat_env > 0) & (beat_env >= maximum_filter1d(beat_env, size=2 * span + 1)))
    hits = np.array([i for i in candidates if float(rise[max(0, i - 3):i + 4].max()) >= strong], dtype=float)
    onsets_s, _ = _attacks(beat_env, hits / fr, 0.5 / fr) if len(hits) else (np.zeros(0), None)
    off = np.full(len(planned), np.nan)
    for i, t in enumerate(planned):
        if detected[i]:
            continue
        # half a beat either way, and 15 ms more: a hit exactly half a beat off must not fall out by rounding
        near = onsets_s[(np.abs(onsets_s - t) <= period / 2 + 0.015) & (np.abs(onsets_s - t) > window)]
        if len(near):
            off[i] = float(near[np.argmin(np.abs(near - t))]) - t
    level = np.array([10 * np.log10(float(np.mean(seg ** 2)) + 1e-12) if len(seg := y[max(0, int(t * SR)):
                      max(0, int((t + period) * SR))]) else -120.0 for t in planned])
    return np.where(detected, times - planned, np.nan), off, level


def onsets(path: Path) -> list[float]:
    """Onset peaks (seconds) of any length of audio, as analyze reports them."""
    env = librosa.onset.onset_strength(y=_load(path), sr=SR, hop_length=HOP)
    return [round(float(t), 4) for t in librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, units="time")]


def analyze(path: Path, beats_per_bar: int = 4, bpm_hint: float | None = None,
            steady_tol_s: float = STEADY_TOL_S) -> BeatGrid:
    """The bed's beat grid. A steady bed (one tempo, as generated beds are) gets its straight grid from fit_steady,
    exact to a millisecond and from its first beat; any other bed keeps librosa's tracked beats. `bpm_hint`, the
    tempo the bed was asked for, picks the steady grid's metrical level; `steady_tol_s` is fit_steady's tolerance."""
    y = _load(path)
    bpm, beat_times, env = _track(y, path, beats_per_bar)
    fit, _ = _fit(y, bpm, beats_per_bar, bpm_hint, steady_tol_s)
    if fit is not None:
        bpm, beats, downbeats = fit.bpm, fit.grid, fit.downbeats
    else:   # the tracker's own beats, with downbeats at the beat phase that has the most kick energy
        beats = [round(float(t), 4) for t in beat_times]
        downbeats = beats[_kick_phase(y, beat_times, beats_per_bar)::beats_per_bar]
    onsets = [round(float(t), 4) for t in librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, units="time")]
    return BeatGrid(bpm=round(bpm, 3), beats=beats, downbeats=downbeats, onsets=onsets,
                    first_downbeat_s=downbeats[0], beats_per_bar=beats_per_bar)


def write_grid(grid: BeatGrid, path: Path) -> None:
    with atomic_path(Path(path)) as tmp:
        tmp.write_text(grid.model_dump_json(indent=2))
