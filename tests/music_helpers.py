"""Synthetic grooves with known beat times, for the beat-grid, conform and stitching tests."""

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def _kick(sr):
    k = np.arange(int(0.2 * sr)) / sr
    return 0.9 * np.sin(2 * np.pi * (50 + 80 * np.exp(-k * 30)) * k) * np.exp(-k * 12)


def _noise_hit(rng, sr, seconds, decay_s, gain, bright=False):
    n = int(seconds * sr)
    noise = rng.standard_normal(n + 2)
    if bright:   # a hat: the noise differenced twice, so its energy sits up top as a real hat's does
        noise = np.diff(noise, 2) / 2
    return noise[:n] * np.exp(-np.arange(n) / (decay_s * sr)) * gain


def _soften(seg, gain):
    """A drum hit through a low-pass (a 2 ms moving average, twice) and turned down: a filtered breakdown's drums."""
    box = np.ones(88) / 88
    return np.convolve(np.convolve(seg, box)[: len(seg)], box)[: len(seg)] * gain


def groove_samples(bpm, seconds, *, first_beat_s=0.0, sr=SR, beats_per_bar=4, hats_offbeat=0.12, seed=0,
                   bpm_at=None, pad=0.08, chord=(220.0, 277.2, 329.6), offset_at=None, filtered_at=None,
                   kick_every_beat=False, tail_silence_s=0.0):
    """A stereo float groove: kick on 1 (and 3), snare on 2 and 4, a hat on every off-beat, over a held chord.

    Beat k sits at first_beat_s + k * 60 / bpm (beat 0 is a downbeat). `bpm_at(t)`, when given, sets the tempo
    beat by beat instead, for music that is not steady. `offset_at(k)` moves beat k's hits by that many seconds
    (jitter, a stray note, a section off the grid). `filtered_at(t)` true makes the drums of the beat at t soft and
    low-passed, with no hat (a filtered breakdown). `kick_every_beat` plays the same kick on every beat (four on the
    floor: no beat of the bar stands out). `tail_silence_s` adds that much silence after the music.
    Returns (samples (n, 2), the beat times, the downbeat times).
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    y = pad * sum(np.sin(2 * np.pi * f * t) * (0.8 ** i) for i, f in enumerate(chord)) * (t >= first_beat_s)
    kick = _kick(sr)
    beats, k, tb = [], 0, first_beat_s

    def put(start, seg):
        a = int(round(start * sr))
        if a >= n:
            return
        if a < 0:
            seg, a = seg[-a:], 0
        y[a:a + len(seg)] += seg[: n - a]

    while tb < seconds:
        if tb >= 0:
            beats.append(tb)
        phase = k % beats_per_bar
        at = tb + (offset_at(k) if offset_at else 0.0)
        soft = bool(filtered_at and filtered_at(tb))
        if kick_every_beat:
            hit = kick
        else:
            hit = (kick if phase == 0 else 0.7 * kick) if phase in (0, 2) else _noise_hit(rng, sr, 0.12, 0.03, 0.35)
        put(at, _soften(hit, 0.12) if soft else hit)
        period = 60.0 / (bpm_at(tb) if bpm_at else bpm)
        if hats_offbeat and not soft:
            put(at + period / 2, _noise_hit(rng, sr, 0.03, 0.005, hats_offbeat, bright=True))
        tb += period
        k += 1
    downbeats = [b for i, b in enumerate(beats) if (round((b - first_beat_s) * bpm / 60) % beats_per_bar) == 0] \
        if bpm_at is None else []
    st = np.stack([y, 0.95 * y], 1).astype(np.float32)
    if tail_silence_s:
        st = np.concatenate([st, np.zeros((int(tail_silence_s * sr), 2), np.float32)])
    return st, beats, downbeats


def write_groove(path, bpm, seconds, **kw):
    """Write a groove to `path` (WAV, float) or, for a .mp3 path, through an MP3 encode. Returns (beats, downbeats)."""
    path = Path(path)
    st, beats, downbeats = groove_samples(bpm, seconds, **kw)
    wav = path if path.suffix == ".wav" else path.with_suffix(".src.wav")
    sf.write(wav, st, SR, subtype="FLOAT")
    if path.suffix == ".mp3":
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-b:a", "192k", str(path)], check=True)
        wav.unlink()
    return beats, downbeats


def groove_mp3(tmp_path, bpm, seconds, name=None, **kw) -> bytes:
    p = Path(tmp_path) / (name or f"groove-{bpm}-{seconds}-{kw.get('first_beat_s', 0.0)}-{kw.get('seed', 0)}.mp3")
    write_groove(p, bpm, seconds, **kw)
    return p.read_bytes()
