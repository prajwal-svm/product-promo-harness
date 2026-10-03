"""Reference ingest: measure a reference video and scaffold style_guide.md (the grammar, never the content)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from promo import asr
from promo.audio_analysis import decode_mono, duration_s, loudness, spectrogram
from promo.evidence import contact_sheet, frames_at, has_audio
from promo.models import Caption
from promo.proc import atomic_path, run

FRAME_EVERY_S = 0.5
SHEET_FRAMES = 36          # contact-00.png holds the first 36 frames (6 x 6), and so on
LINE_GAP_S = 0.6           # words closer than this belong to one spoken line
CROP_RE = re.compile(r"(\d+):(\d+):(\d+):(\d+)")

# Tempo (see `tempo`): the beat tracker decides the metrical level, the kick band refines the period.
TEMPO_WIN = 2048                 # analysis window in samples
TEMPO_BLOCK = 4096               # frames per block, so a long reference never holds its whole spectrogram
KICK_HZ = (30, 130)
REFINE = 0.03                    # the tracker's value is searched +-3% around, in
REFINE_STEP_BPM = 0.05           # steps of 0.05 BPM
KICK_GATE = 1.5                  # kick-band z-score on the pulses, above chance, below which the kick cannot refine a tempo
PULSE_GATE = 0.7                 # the same on the whole spectrum, below which there is no credible pulse at all
LEVEL_RANGE = (70, 180)          # a half or double of the tempo is only considered when it lands in [70, 180), to the nearest BPM
LEVEL_RATIO = 1.8                # ...and the kick band scores it at least this many times the tracker's level (a kick on every other pulse scores 2.0)
LEVEL_MIN_SCORE = 3.0            # ...and at least this high (a clean synthetic kick scores 4 to 7; the real beds whose half lands in range top out at 2.5)
MIN_TEMPO_S = 5.0

STYLE_GUIDE = """# Style guide: {name}

Measured by `promo ref`. Take the grammar of the reference, never its content, logos or characters.

## Measured
- Video: {w}x{h} at {fps:g} fps, {dur:.1f} s; hard cuts at {cuts}.
- Audio: {audio}.
- Voice: {voice}.

## Palette
(hex values by role: bg, bg_alt, accent, accent_ink, dark, surface, ink, muted; sampled from crops, tuned by eye)

## Type
(family candidates, weight, tracking, case, size as % of frame height, how the key word is marked)

## Shot lengths
(chapter and shot lengths; where the key beat sits)

## Transitions
(each transition type and the carrier object that becomes the next scene)

## Camera
(moves, parallax, depth, what stays locked)

## Texture
(grain, vignette, blur, depth of field)

## Text in and out
(how words enter and exit; lag behind the voice)

## Signature moves
(the 5 to 8 moves that make this reference recognisable)

## Voice
(register, pace, coverage, gaps, delivery notes)

## Music
(tempo, instrumentation, arrangement, loudness and loudness range)

## Sync
(what lands on beats, on words, and on visual events)
"""


class RefError(RuntimeError):
    """The input is not something `promo ref` can ingest (a missing file, no video stream, a malformed crop)."""


def _probe(video: Path) -> dict:
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height,r_frame_rate,avg_frame_rate:stream_side_data=rotation:stream_tags=rotate",
               "-of", "json", video]).stdout
    streams = json.loads(out)["streams"]
    if not streams:
        raise RefError(f"{video} has no video stream")
    s = streams[0]
    # A phone recording has a variable frame rate: the average is what it plays at, r_frame_rate is only its finest tick.
    rate = next((r for r in (s.get("avg_frame_rate"), s["r_frame_rate"]) if r and not r.startswith("0/")), "0/1")
    num, den = rate.split("/")
    # ffmpeg turns a rotated phone video upright before any filter, so the frame (and a crop of it) is the upright one.
    turns = [d["rotation"] for d in s.get("side_data_list", []) if "rotation" in d] or [s.get("tags", {}).get("rotate", 0)]
    upright = abs(int(float(turns[0]))) % 180 == 90
    width, height = (s["height"], s["width"]) if upright else (s["width"], s["height"])
    return {"width": width, "height": height, "fps": round(float(num) / float(den or 1), 3),
            "duration_s": round(duration_s(video), 3)}


def _check_crop(crop: str, info: dict) -> None:
    w, h, x, y = (int(n) for n in CROP_RE.fullmatch(crop).groups())
    if not (w > 0 and h > 0 and x + w <= info["width"] and y + h <= info["height"]):
        raise RefError(f"--crop {crop} does not fit the {info['width']}x{info['height']} frame: W and H must be above 0, "
                       f"X+W at most {info['width']} and Y+H at most {info['height']}")


def _cuts(video: Path) -> list[float]:
    err = run(["ffmpeg", "-hide_banner", "-nostats", "-i", video, "-vf", "scdet=threshold=10", "-an", "-f", "null", "-"]).stderr
    return [round(float(t), 3) for t in re.findall(r"lavfi\.scd\.time: ([\d.]+)", err)]


def pitch_hz(samples: np.ndarray, sr: int, words: list[Caption]) -> float | None:
    """Median fundamental frequency over the voiced frames of the spoken words, by autocorrelation (70-350 Hz)."""
    f0 = []
    lo, hi = int(sr / 350), int(sr / 70)
    for w in words:
        a, b = max(0, int(w.startMs / 1000 * sr)), min(len(samples), int(w.endMs / 1000 * sr))
        for i in range(a, b - 1024, 512):
            fr = samples[i:i + 1024] * np.hanning(1024)
            fr = fr - fr.mean()
            ac = np.correlate(fr, fr, "full")[1023:]
            if ac[0] <= 0:
                continue
            k = lo + int(np.argmax(ac[lo:hi]))
            if ac[k] / ac[0] > 0.45:
                f0.append(sr / k)
    return float(np.median(f0)) if f0 else None


def _kick_flux(samples: np.ndarray, sr: int) -> np.ndarray:
    """Positive spectral flux of the 30-130 Hz band, one value per 10 ms frame."""
    hop = sr // 100
    x = np.ascontiguousarray(samples, dtype=np.float32)
    x = x / (float(np.abs(x).max()) or 1.0)
    frames = (len(x) - TEMPO_WIN) // hop + 1
    freqs = np.fft.rfftfreq(TEMPO_WIN, 1 / sr)
    band = (freqs >= KICK_HZ[0]) & (freqs <= KICK_HZ[1])
    window = np.hanning(TEMPO_WIN).astype(np.float32)
    rises, prev = [], None
    for a in range(0, frames, TEMPO_BLOCK):       # in blocks, so a long reference never holds its whole spectrogram
        b = min(frames, a + TEMPO_BLOCK)
        block = np.lib.stride_tricks.as_strided(x[a * hop:], shape=(b - a, TEMPO_WIN),
                                                strides=(x.strides[0] * hop, x.strides[0]))
        logs = np.log1p(np.abs(np.fft.rfft(block * window, axis=1))[:, band])
        if prev is not None:
            logs = np.vstack([prev, logs])
        prev = logs[-1:]
        rises.append(np.maximum(0.0, np.diff(logs, axis=0)).sum(axis=1))
    return np.concatenate(rises)


def _pulse_score(env: np.ndarray, rate: float, bpm: float) -> float:
    """The best mean of a z-scored envelope on an evenly spaced pulse train of this tempo, over every phase."""
    period = 60.0 * rate / bpm
    pulses = int((len(env) - 1 - period) / period) + 1
    if pulses < 2:
        return 0.0
    phases = np.arange(max(1, round(period))) * (period / max(1, round(period)))
    at = phases[:, None] + np.arange(pulses)[None, :] * period
    return float(np.interp(at, np.arange(len(env)), env).mean(axis=1).max())


def _refine(env: np.ndarray, rate: float, centre: float) -> tuple[float, float]:
    """(score, tempo) of the best pulse train within +-3% of `centre`, in 0.05 BPM steps.

    The score is the pulse-train score minus what a train that cannot be aligned with the envelope (a tempo 7%, 9%
    and 13% off; their median) gets from the same pulse count. Picking the best phase of a sparse train (few pulses,
    a slow tempo) is lucky by itself, and a plain score would favour a slow train over the fast one it divides."""
    grid = np.arange(centre * (1 - REFINE), centre * (1 + REFINE) + 1e-9, REFINE_STEP_BPM)
    scores = [_pulse_score(env, rate, b) for b in grid]
    best = int(np.argmax(scores))
    chance = float(np.median([_pulse_score(env, rate, grid[best] * f) for f in (0.93, 1.07, 1.13)]))
    return scores[best] - chance, float(grid[best])


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (float(x.std()) or 1.0)


def _beat_tracker(samples: np.ndarray, sr: int, wav: Path | None) -> tuple[float | None, str | None]:
    """`beats.analyze`'s tempo for the audio (it takes a file), or the reason it found none."""
    import tempfile

    import soundfile as sf

    from promo.beats import analyze

    with tempfile.TemporaryDirectory() as d:
        path = wav if wav is not None else Path(d) / "tracker.wav"
        try:
            if wav is None:
                sf.write(path, samples, sr)
            return analyze(path).bpm, None
        except Exception as e:  # not music, or too short
            return None, str(e).replace(str(path), "the audio")


def _levels(bpm: float, tracker: float | None) -> list[float]:
    """The tempo, the tracker's own reading and the half and double of the tempo: the readings a designer may want."""
    out: list[float] = []
    for b in (bpm, tracker, bpm / 2, bpm * 2):
        if b is not None and all(abs(b - o) / o > 0.01 for o in out):
            out.append(b)
    return [round(b, 2) for b in out]


def tempo(samples: np.ndarray, sr: int = 16000, *, wav: Path | None = None, tracker: float | None = None) -> dict:
    """The tempo of a reference's audio, which may carry a voice-over over the music.

    The beat tracker (`beats.analyze`) decides the metrical level: on real music it picks beat, half or double
    correctly, but on a mix it can be 2% off the true period (117.45 for a 120 BPM reference). The kick band
    (30-130 Hz, which speech hardly touches) then refines the period. Picking the level from the kick instead,
    as an earlier version did, put 14 of 25 real stock beds at the wrong level, 8 of them at about a quarter of
    their tempo: real kicks are strongest on beat 1, so a once-per-bar pulse train scores best.
    - The tracker's tempo is searched +-3% around, in 0.05 BPM steps, for the pulse train that sits highest on the
      kick band's flux (z-scored, above what an unaligned train gets by chance).
    - Half and double of the result are considered only if they land in [70, 180) and the kick band scores them
      at least 1.8 times the tracker's level and at least 3.0. Never a quarter, 2/3 or 3/2.
    - A kick score below 1.5 cannot refine the tracker's tempo: it is reported as it is, `confidence`
      "tracker-only", provided the whole spectrum has a credible pulse at that tempo (score 0.7 or more: noise,
      voice-over alone and ambient pads measure under 0.5, real beds 1.0 to 3.2). Otherwise `bpm` is None.
    `tracker` gives the beat tracker's tempo instead of running it (`wav` is the audio as a file, to save writing
    one). Returns `bpm`, `confidence` ("kick-band", "tracker-only" or None), `candidates` (the tempo first), `tracker`
    and `error` (why `bpm` is None, or why the tracker found nothing).
    """
    out: dict = {"bpm": None, "confidence": None, "candidates": [], "tracker": None, "error": None}
    if len(samples) < MIN_TEMPO_S * sr:
        out["error"] = "audio too short for a tempo"
        return out
    error = None
    if tracker is None:
        tracker, error = _beat_tracker(samples, sr, wav)
    if tracker is None:
        out["error"] = f"no beat tracker reading ({error})"
        return out
    out["tracker"] = round(tracker, 2)
    import librosa

    rate = sr / (sr // 100)
    kick = _zscore(_kick_flux(samples, sr))
    level = _refine(kick, rate, tracker)
    for m in (0.5, 2.0):
        alt = _refine(kick, rate, level[1] * m)
        if (LEVEL_RANGE[0] <= round(alt[1]) < LEVEL_RANGE[1] and alt[0] >= LEVEL_MIN_SCORE
                and alt[0] >= LEVEL_RATIO * level[0] and alt[0] > level[0]):
            level = alt
    if level[0] >= KICK_GATE:
        out.update(bpm=round(level[1], 2), confidence="kick-band", candidates=_levels(level[1], tracker))
        return out
    full = _zscore(librosa.onset.onset_strength(y=np.asarray(samples, dtype=np.float32), sr=sr, hop_length=sr // 100))
    if _refine(full, rate, tracker)[0] >= PULSE_GATE:
        out.update(bpm=round(tracker, 2), confidence="tracker-only", candidates=_levels(tracker, None))
        return out
    out["error"] = "no credible pulse found"
    return out


def _frames(video: Path, info: dict, out_dir: Path, crop: str | None) -> list[tuple[float, Path]]:
    """One frame every 0.5 s, 480 px wide (cropped first when asked), as (time, png)."""
    if not crop:
        times = [float(t) for t in np.arange(0.0, info["duration_s"], FRAME_EVERY_S)]
        return frames_at(video, times, out_dir, 480)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("f-*.png"):
        old.unlink()
    run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vf", f"fps={1 / FRAME_EVERY_S:g},crop={crop},scale=480:-2",
         "-start_number", "0", out_dir / "f-%04d.png"])
    return [(i * FRAME_EVERY_S, p) for i, p in enumerate(sorted(out_dir.glob("f-*.png")))]


def _lines(words: list[Caption]) -> list[list[Caption]]:
    lines: list[list[Caption]] = []
    for w in words:
        if lines and w.startMs - lines[-1][-1].endMs <= LINE_GAP_S * 1000:
            lines[-1].append(w)
        else:
            lines.append([w])
    return lines


def _voice_metrics(words: list[Caption], samples: np.ndarray, duration: float) -> dict:
    lines = _lines(words)
    speech = sum((w.endMs - w.startMs) / 1000 for w in words)
    in_lines = sum((ln[-1].endMs - ln[0].startMs) / 1000 for ln in lines)
    return {
        "words": len(words), "lines": len(lines), "speech_s": round(speech, 2),
        "coverage": round(speech / duration, 3) if duration else 0,
        # within a line the pace includes the small gaps between words, as the pack's pace_wpm does
        "spoken_wpm": round(len(words) / in_lines * 60) if in_lines else None,
        "overall_wpm": round(len(words) / duration * 60) if duration else None,
        "line_gaps_s": [round((b[0].startMs - a[-1].endMs) / 1000, 2) for a, b in zip(lines, lines[1:])],
        "median_pitch_hz": pitch_hz(samples, 16000, words),
    }


def _finite(x: float) -> float | None:
    return x if math.isfinite(x) else None      # silence measures -inf, which JSON cannot hold


def _tempo_metrics(samples: np.ndarray, wav: Path) -> dict:
    """`bpm`, how sure we are (`bpm_confidence`), `bpm_candidates` and the beat tracker's own reading `bpm_tracker`."""
    t = tempo(samples, wav=wav)
    out = {"bpm": t["bpm"], "bpm_confidence": t["confidence"], "bpm_candidates": t["candidates"],
           "bpm_tracker": t["tracker"]}
    if t["error"]:
        out["bpm_error"] = t["error"]
    return out


def _audio_metrics(video: Path, out: Path) -> tuple[dict, Path, np.ndarray]:
    audio: dict = {"loudness": {k: _finite(v) for k, v in loudness(video).items()}}
    spectrogram(video, out / "spectrogram.png")
    wav = out / "audio16k.wav"
    run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "16000", wav])
    samples = decode_mono(wav, 16000)
    audio.update(_tempo_metrics(samples, wav))
    return audio, wav, samples


def _write(path: Path, text: str) -> None:
    with atomic_path(path) as tmp:
        tmp.write_text(text)


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:g}"


def _tempo_phrase(audio: dict) -> str:
    bpm, tracker = audio.get("bpm"), audio.get("bpm_tracker")
    notes = []
    if bpm:
        notes.append("kick band" if audio.get("bpm_confidence") == "kick-band"
                     else "tracker only: the kick band is too weak to refine it")
    elif audio.get("bpm_error"):
        notes.append(audio["bpm_error"])
    if len(audio.get("bpm_candidates") or []) > 1:
        notes.append("candidates " + ", ".join(f"{c:.1f}" for c in audio["bpm_candidates"]))
    if tracker:
        notes.append(f"beat tracker {tracker:.1f}")
    return (f"{bpm:.1f} BPM" if bpm else "n/a") + (f" ({'; '.join(notes)})" if notes else "")


def _audio_line(audio: dict) -> str:
    loud = audio["loudness"]
    if loud is None:
        return "none (no audio track)"
    return (f"{_num(loud['I'])} LUFS, LRA {_num(loud['LRA'])} LU, TP {_num(loud['TP'])} dBTP; "
            f"tempo {_tempo_phrase(audio)}")


def _voice_line(voice: dict) -> str:
    if not voice["words"]:
        return "no speech found"
    pitch = f"{voice['median_pitch_hz']:.0f} Hz (measured, not heard)" if voice["median_pitch_hz"] else "n/a"
    return (f"{voice['words']} words, coverage {voice['coverage']:.0%}, {_num(voice['spoken_wpm'])} wpm inside lines, "
            f"{_num(voice['overall_wpm'])} wpm overall, median pitch {pitch}")


def ingest(video: Path, out: Path, *, crop: str | None = None, language: str | None = None, transcribe=None) -> Path:
    """Measure `video` into the folder `out`: metrics.json, style_guide.md (a skeleton with the measured facts filled in),
    contact-NN.png sheets of a frame every 0.5 s (cropped to W:H:X:Y when `crop` is given), spectrogram.png, the
    16 kHz audio the voice was measured on, and transcript.json (word times from `transcribe(wav, language)`,
    Whisper in English by default)."""
    video, out = Path(video), Path(out)
    if not video.is_file():
        raise RefError(f"no such video: {video}")
    if crop and not CROP_RE.fullmatch(crop):
        raise RefError(f"--crop wants W:H:X:Y in pixels, e.g. 1280:720:320:180 (got '{crop}')")
    info = _probe(video)
    if crop:
        _check_crop(crop, info)
    out.mkdir(parents=True, exist_ok=True)
    cuts = _cuts(video)
    items = _frames(video, info, out / "frames", crop)
    for i in range(0, len(items), SHEET_FRAMES):
        contact_sheet([(f"{t:.1f}s", p) for t, p in items[i:i + SHEET_FRAMES]], out / f"contact-{i // SHEET_FRAMES:02d}.png")

    if has_audio(video):
        audio, wav, samples = _audio_metrics(video, out)
        tr = transcribe or (lambda p, lang: asr.transcribe(p, lang or "en", engine="whisper"))
        words = tr(wav, language)
    else:                                            # a silent screen recording: nothing to measure
        audio = {"loudness": None, "bpm": None, "bpm_confidence": None, "bpm_candidates": [], "bpm_tracker": None}
        words, samples = [], np.zeros(0, dtype=np.float32)
    _write(out / "transcript.json", json.dumps([w.model_dump() for w in words], indent=2))
    voice = _voice_metrics(words, samples, info["duration_s"])
    _write(out / "metrics.json", json.dumps({"video": info, "cuts": cuts, "audio": audio, "voice": voice}, indent=2))
    _write(out / "style_guide.md", STYLE_GUIDE.format(
        name=video.stem, w=info["width"], h=info["height"], fps=info["fps"], dur=info["duration_s"],
        cuts=", ".join(f"{c:.2f}s" for c in cuts) or "none", audio=_audio_line(audio), voice=_voice_line(voice)))
    return out
