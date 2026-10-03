"""Critic evidence: frames, sheets, strips, technical checks and sync at planned events."""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from promo import asr
from promo.audio_analysis import decode_mono
from promo.proc import ToolError, atomic_path, run
from promo.sfx import Placement
from promo.sfx_synth import SR
from promo.timing import Timing
from promo.voice import score_take

SFX_WINDOW_S = 0.15        # the widest look either side of an event for its sound
SFX_SILENT = 1e-5          # a stem below this (-100 dBFS) is silent, not a sound
SFX_CORR_MIN = 0.5         # a sound found with a lower correlation than this is not measurable, not late...
SFX_AMPLITUDE_MIN = 0.25   # ...and neither is one found at under this share of its level (a faint tail with its shape)
VO_SEARCH_S = 0.5          # how far either side of its planned start a take is looked for in the mix
VO_CORR_MIN = 0.5          # a take found with a lower correlation than this is not in the mix
LINE_WER_MAX = 0.15        # per line, on the final mix (voice.py's own gate on the isolated take is 0.1)
SEEK_BACKOFF_S = 0.001     # see _frame


def _video_span(video: Path) -> tuple[float, float]:
    """(seconds, frames per second) of the first video stream."""
    info = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=duration,r_frame_rate:format=duration", "-of", "json", video]).stdout)
    if not info.get("streams"):
        raise ToolError(f"{video} has no video stream")
    stream = info["streams"][0]
    num, _, den = stream["r_frame_rate"].partition("/")
    raw = stream.get("duration", info["format"]["duration"])
    return float(raw), float(num) / float(den or 1)


def has_audio(video: Path) -> bool:
    out = run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type",
               "-of", "csv=p=0", video]).stdout
    return bool(out.strip())


def _frame(video: Path, t: float, out: Path, width: int | None) -> Path:
    # -ss lands on the first frame at or after the time it is given. Aim a millisecond early so a time that is
    # exactly on a frame, but rounded up to three decimals, does not skip to the next frame.
    vf = ["-vf", f"scale={width}:-2"] if width else []
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, t - SEEK_BACKOFF_S):.3f}", "-i", video,
         "-frames:v", "1", *vf, out])
    if not out.exists():
        raise ToolError(f"ffmpeg wrote no frame at {t:.3f}s of {video}")
    return out


def frames_at(video: Path, times: list[float], out_dir: Path, width: int = 480) -> list[tuple[float, Path]]:
    """One PNG per time, `width` px wide. A time past the last frame gives the last frame."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seconds, fps = _video_span(video)
    last = max(0.0, seconds - 1.0 / fps)
    return [(t, _frame(video, min(t, last), out_dir / f"f-{i:04d}.png", width)) for i, t in enumerate(times)]


def contact_sheet(items: list[tuple[str, Path]], out: Path, cols: int = 6) -> Path:
    if not items:
        raise ValueError("a contact sheet needs at least one frame")
    font = ImageFont.load_default(size=20)
    thumbs = [(label, Image.open(p).convert("RGB")) for label, p in items]
    w, h = thumbs[0][1].size
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (w + 4), rows * (h + 4)), "gray")
    for i, (label, im) in enumerate(thumbs):
        d = ImageDraw.Draw(im)
        box = d.textbbox((0, 0), label, font=font)
        d.rectangle([0, 0, box[2] + 10, box[3] + 6], fill="black")
        d.text((5, 3), label, fill="yellow", font=font)
        sheet.paste(im, ((i % cols) * (w + 4), (i // cols) * (h + 4)))
    with atomic_path(out) as tmp:
        sheet.save(tmp, format="PNG")
    return out


def strip(video: Path, center_t: float, fps: int, out: Path, n: int = 12, width: int = 320) -> Path:
    """`n` consecutive frames: n/2 before `center_t`, then `center_t` itself and the frames after it."""
    times = [center_t + (i - n // 2) / fps for i in range(n)]
    with tempfile.TemporaryDirectory() as d:
        items = frames_at(video, times, Path(d), width)
        return contact_sheet([(f"{t:.3f}", p) for t, p in items], out, cols=n)


SHEET_FRAMES = 36      # frames per sheet; the contact sheets, the phone sheet and the master sheet all page at this


def _paged_sheets(video: Path, out: Path, times: list[float], width: int, cols: int, label: str) -> Path:
    """The frames at `times`, 36 to a sheet: the first sheet is `out`, the next ones `<stem>-01<suffix>`, `-02`, ..."""
    first = out
    for page, i in enumerate(range(0, len(times), SHEET_FRAMES)):
        target = out if page == 0 else out.with_name(f"{out.stem}-{page:02d}{out.suffix}")
        with tempfile.TemporaryDirectory() as d:
            items = frames_at(video, times[i:i + SHEET_FRAMES], Path(d), width)
            contact_sheet([(label.format(t=t), p) for t, p in items], target, cols=cols)
    return first


def phone_sheet(video: Path, out: Path, every_s: float = 1.0) -> Path:
    """One 360 px frame every `every_s`, 36 to a sheet: `out` is the first sheet, later ones are `<stem>-01<suffix>`."""
    seconds, _ = _video_span(video)
    return _paged_sheets(video, out, [float(t) for t in np.arange(0.0, seconds, every_s)], 360, 5, "{t:.1f}s")


def master_sheet(video: Path, out: Path, fps: float = 2.0) -> Path:
    """Frames of the encoded master at `fps`, 36 to a sheet: `out` is the first sheet, later ones are `<stem>-01<suffix>`."""
    seconds, _ = _video_span(video)
    return _paged_sheets(video, out, [float(t) for t in np.arange(0.0, seconds, 1.0 / fps)], 320, 8, "{t:.1f}s")


def fastest_action_t(video: Path) -> float:
    """The time of the biggest frame-to-frame change (10 fps, 96x54 grey): where the strip is most worth a look."""
    raw = run(["ffmpeg", "-v", "error", "-i", video, "-vf", "fps=10,scale=96:54,format=gray", "-f", "rawvideo", "-"],
              text=False).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 54, 96).astype(np.float32)
    if len(frames) < 2:
        return 0.0
    diff = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    return float(np.argmax(diff) + 1) / 10.0


def tech_checks(video: Path, *, width: int, height: int, fps: int, max_static_s: float) -> dict:
    probe = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height,r_frame_rate,codec_name", "-of", "default=nw=1", video]).stdout
    info = dict(line.split("=", 1) for line in probe.strip().splitlines())
    if "width" not in info:
        return {"ok": False, "issues": ["no video stream"], "codec": None}
    issues = []
    if (int(info["width"]), int(info["height"])) != (width, height):
        issues.append(f"canvas is {info['width']}x{info['height']}, expected {width}x{height}")
    num, den = info["r_frame_rate"].split("/")
    if abs(float(num) / float(den) - fps) > 0.01:
        issues.append(f"fps is {info['r_frame_rate']}, expected {fps}")
    err = run(["ffmpeg", "-hide_banner", "-nostats", "-i", video, "-vf",
               f"blackdetect=d=0.5:pic_th=0.98:pix_th=0.02,freezedetect=n=-60dB:d={max_static_s}",
               "-an", "-f", "null", "-"]).stderr
    for s, e in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", err):
        issues.append(f"black from {float(s):.2f}s to {float(e):.2f}s")
    for s in re.findall(r"freeze_start: ([\d.]+)", err):
        issues.append(f"frozen picture from {float(s):.2f}s (longer than {max_static_s}s)")
    return {"ok": not issues, "issues": issues, "codec": info.get("codec_name")}


def _correlate(window: np.ndarray, clip: np.ndarray) -> tuple[int, float, float]:
    """Where `clip` sits best in `window`: (the sample it starts on, the normalised correlation there, -1 to 1, and the
    amplitude it is there at, 1.0 being the clip as it is).

    The place is where the clip's matched-filter output is largest, so a loud copy beats a faint tail that merely has its
    shape. The normalised correlation does not depend on the level, so a take under a bed or a sound after a limiter still
    scores; the amplitude says how loud the match is."""
    w, c = np.asarray(window, dtype=np.float64), np.asarray(clip, dtype=np.float64)
    span = len(w) - len(c) + 1
    energy_c = float(c @ c)
    if span < 1 or energy_c <= 0:
        return 0, 0.0, 0.0
    n = 1 << (len(w) + len(c) - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(w, n) * np.conj(np.fft.rfft(c, n)), n)[:span]
    squares = np.concatenate(([0.0], np.cumsum(w * w)))
    energy_w = squares[len(c):len(c) + span] - squares[:span]
    k = int(np.argmax(corr))
    return k, float(corr[k] / np.sqrt(max(energy_w[k], 1e-18) * energy_c)), float(corr[k] / energy_c)


def _half_window_s(times_of_same_sound: list[float], t: float) -> float:
    """How far either side of an event to look for its sound: half the gap to the nearest other event of the same sound
    (a different sound cannot be taken for it), at most SFX_WINDOW_S."""
    others = [abs(x - t) for x in times_of_same_sound if abs(x - t) > 0.0005]
    return min(SFX_WINDOW_S, min(others) / 2) if others else SFX_WINDOW_S


def sfx_report(stem: Path | None, placements: Sequence[Placement], fps: int, *,
               clips: Mapping[str, tuple[np.ndarray, int]] | None = None,
               event_times: Mapping[str, float] | None = None) -> dict:
    """Where each placed sound really is in the SFX stem, against the sample it should start on.

    `clips` maps an event id to its sound as it sits in the stem and the sample it should start on (`sfx.event_clip`).
    Each sound is found by cross-correlating it into the stem around its planned place, after taking out every OTHER
    sound where the stem says it was placed (the stem is rebuilt from the same clips, scaled to fit). A neighbour, loud,
    long or the same sound, therefore cannot be mistaken for it. A row is ok when the sound is found within one frame
    plus 2 ms of its place. The clips are made again from the same seeded synth or cache as the stem, so a sound that
    cannot be found (correlation under 0.5, or under a quarter of its level), or whose clip cannot be made again, is
    "not found in stem" and FAILS; `unmeasurable` (not a failure, a warning) is only for a clip that is itself silent.
    A stem that is silent where a sound belongs, or ends before it, fails. With `event_times` (timing.json's event id ->
    time) a placement for an event that has moved or gone is stale and fails too.
    """
    tol_ms = 1000.0 / fps + 2
    if stem is None or not Path(stem).exists():
        return {"ok": False, "tolerance_ms": round(tol_ms, 2), "events": [],
                "detail": "no SFX stem (audio/sfx/track.wav); run `promo sfx render`"}
    clips = clips or {}
    data = decode_mono(Path(stem), SR).astype(np.float64)
    n = len(data)
    rebuilt = np.zeros(n)
    stem_start: dict[str, tuple[np.ndarray, int]] = {}
    for p in placements:
        if p.event in clips:
            s0 = round(p.start_s * SR)
            fit = clips[p.event][0][: max(0, n - s0)]
            rebuilt[s0:s0 + len(fit)] += fit
            stem_start[p.event] = (fit, s0)
    energy = float(rebuilt @ rebuilt)
    scale = min(1.0, max(0.0, float(data @ rebuilt) / energy)) if energy > 0 else 1.0   # a stem scaled down to not clip
    times_by_sound: dict[str, list[float]] = {}
    for p in placements:
        times_by_sound.setdefault(p.sfx, []).append(p.t)
    rows = []
    for p in placements:
        half = _half_window_s(times_by_sound[p.sfx], p.t)
        row: dict = {"event": p.event, "t": p.t, "window_ms": round(half * 1000, 1)}
        rows.append(row)
        if event_times is not None:
            now = event_times.get(p.event)
            if now is None:
                row.update(ok=False, offset_ms=None, issue="the event is no longer in timing.json; run `promo sfx render`")
                continue
            if abs(now - p.t) > 0.0015:
                row.update(ok=False, offset_ms=None, issue=f"timing.json now has it at {now:.3f}s, the stem was rendered "
                                                           f"for {p.t:.3f}s; run `promo sfx render`")
                continue
        if p.event not in clips:
            row.update(ok=False, offset_ms=None,
                       issue="not found in stem: its sound could not be made again from the pack or the cache to look "
                             "for it; run `promo sfx render`")
            continue
        clip, expected = clips[p.event]
        fit, s0 = stem_start[p.event]
        r = round(half * SR)
        lo, hi = max(0, expected - r), min(n, expected + r + len(clip))
        if expected >= n or hi - lo < len(fit) or len(fit) == 0:
            row.update(ok=False, offset_ms=None,
                       issue=f"the SFX stem is {n / SR:.3f}s long and has no audio at {p.t:.3f}s")
            continue
        if float(np.max(np.abs(fit))) < SFX_SILENT:        # a silent sound has no shape to look for
            row.update(ok=True, offset_ms=None, unmeasurable=True, issue="its sound is silent: there is nothing to find")
            continue
        window = data[lo:hi]
        if float(np.max(np.abs(window))) < SFX_SILENT:
            row.update(ok=False, offset_ms=None, issue=f"the SFX stem is silent at {p.t:.3f}s")
            continue
        others = rebuilt[lo:hi].copy()
        a, b = max(lo, s0), min(hi, s0 + len(fit))
        if b > a:
            others[a - lo:b - lo] -= fit[a - s0:b - s0]           # every sound but this one, where the stem has it
        k, corr, amplitude = _correlate(window - scale * others, fit)
        row["corr"] = round(corr, 3)
        if corr < SFX_CORR_MIN or amplitude < SFX_AMPLITUDE_MIN * scale:
            row.update(ok=False, offset_ms=None,
                       issue=f"not found in stem near {p.t:.3f}s (correlation {corr:.2f}): the stem was not made from "
                             f"this pack's sounds, or by this timing; run `promo sfx render`")
            continue
        offset = (lo + k - expected) / SR * 1000
        row.update(offset_ms=round(offset, 2), ok=abs(offset) <= tol_ms)
    return {"ok": all(r["ok"] for r in rows), "tolerance_ms": round(tol_ms, 2), "events": rows}


def _window_wav(video: Path, start_s: float, duration_s: float, out: Path) -> Path:
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, start_s):.4f}", "-t", f"{duration_s:.4f}", "-i", video,
         "-vn", "-ac", "1", "-ar", "16000", out])
    return out


def _percentile(sorted_values: list[float], q: float) -> float | None:
    return sorted_values[min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1))))] if sorted_values else None


def vo_report(timing: Timing, video: Path, language: str, *, takes: Mapping[str, Path] | None = None,
              transcribe=asr.transcribe, aliases: Mapping[str, Sequence[str]] | None = None,
              skip_lines: Collection[str] = ()) -> dict:
    """The voice on the final mix, line by line (spec 9.8).

    `vo` is the sync: each line's own take (`takes`: line id -> audio/vo/<id>.wav) is cross-correlated into the mix
    within 0.5 s of the line's planned start. A line is ok when it is found (correlation of 0.5 or more) within one
    frame of where timing.json puts it. `asr_offsets_ms` is information only: how far recognised word starts are from the
    take's word starts, which is bounded by how well the recogniser times words, not by the mix.

    `lines` is the WER: the stretch of the mix from the line's start for the take's length is cut out, resampled for
    recognition and scored with `voice.score_take`, the function the takes themselves are scored with (aliases read
    as the script's word, numbers and spelling variants normalised, inline audio tags dropped). A line in `skip_lines`
    (`verify: false`: speech recognition cannot score it) is left out of the WER.
    """
    aliases, takes = aliases or {}, takes or {}
    tol_ms = 1000.0 / timing.fps
    audible = has_audio(video)
    mix = decode_mono(video, SR) if audible else np.zeros(0, dtype=np.float32)
    sync_rows, wer_rows, offsets = [], [], []
    with tempfile.TemporaryDirectory() as d:
        for line in timing.lines:
            length = line.end_s - line.start_s
            row: dict = {"line": line.id}
            sync_rows.append(row)
            take = takes.get(line.id)
            if not audible:
                row.update(offset_ms=None, corr=None, ok=False, issue="the video has no audio track")
            elif take is None or not Path(take).exists():
                row.update(offset_ms=None, corr=None, ok=False,
                           issue="no voice take for it, so nothing of it is in the mix; run `promo voice`")
            else:
                clip = decode_mono(Path(take), SR)
                center, r = round(line.start_s * SR), round(VO_SEARCH_S * SR)
                lo, hi = max(0, center - r), min(len(mix), center + r + len(clip))
                k, corr, _ = _correlate(mix[lo:hi], clip)
                if hi - lo < len(clip) or corr < VO_CORR_MIN:
                    row.update(offset_ms=None, corr=round(corr, 3), ok=False,
                               issue=f"its take was not found within {VO_SEARCH_S} s of {line.start_s:.2f}s "
                                     f"(correlation {corr:.2f})")
                else:
                    offset = (lo + k - center) / SR * 1000
                    row.update(offset_ms=round(offset, 2), corr=round(corr, 3), ok=abs(offset) <= tol_ms)
            if line.id in skip_lines:
                wer_rows.append({"line": line.id, "wer": None, "ok": True, "skipped": "verify: false"})
                continue
            if not audible or length <= 0:
                wer_rows.append({"line": line.id, "wer": 1.0, "ok": False})
                continue
            wav = _window_wav(video, line.start_s, length, Path(d) / f"{line.id}.wav")
            score, words = score_take(wav, line.text, aliases, language, length * 1000, transcribe)
            wer_rows.append({"line": line.id, "wer": round(score, 3), "ok": score <= LINE_WER_MAX})
            if len(words) == len(line.words):          # the take's words are on the take's clock: the window starts with it
                base = round(line.start_s * 1000)
                offsets += [abs(w.startMs - (t.startMs - base)) for w, t in zip(words, line.words)]
    offsets.sort()
    return {
        "vo": {"ok": all(r["ok"] for r in sync_rows), "tolerance_ms": round(tol_ms, 2), "min_corr": VO_CORR_MIN,
               "rows": sync_rows,
               "asr_offsets_ms": {"words": len(offsets), "median": _percentile(offsets, 0.5),
                                  "p95": _percentile(offsets, 0.95)}},
        "lines": {"ok": all(r["ok"] for r in wer_rows), "rows": wer_rows},
    }


def sync_report(timing: Timing, stem: Path | None, placements: Sequence[Placement], video: Path, language: str, *,
                transcribe=asr.transcribe, aliases: Mapping[str, Sequence[str]] | None = None,
                skip_lines: Collection[str] = (), takes: Mapping[str, Path] | None = None,
                clips: Mapping[str, tuple[np.ndarray, int]] | None = None) -> dict:
    """Sync at the planned events: `sfx` (each sound found in the stem), `vo` (each take found in the final mix) and
    `lines` (WER per line on the final mix). See `sfx_report` and `vo_report`."""
    sfx = sfx_report(stem, placements, timing.fps, clips=clips, event_times={e.id: e.t for e in timing.events})
    return {"sfx": sfx, **vo_report(timing, video, language, takes=takes, transcribe=transcribe, aliases=aliases,
                                    skip_lines=skip_lines)}
