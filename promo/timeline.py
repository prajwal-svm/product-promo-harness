"""timing.json + rendered media → an OpenTimelineIO edit."""

from __future__ import annotations

from pathlib import Path

import opentimelineio as otio
from pydantic import BaseModel

from promo.audio_analysis import duration_s
from promo.proc import atomic_path, sha256_file
from promo.run import RunPaths
from promo.timing import TimedShot, Timing

AUDIO_RATE = 48000


class TimelineError(ValueError):
    """The timing cannot be laid out on the tracks (clips that overlap on one track)."""


class Media(BaseModel):
    shots: dict[str, Path] = {}
    vo: dict[str, Path] = {}
    music: Path | None = None
    sfx: Path | None = None
    overlay: Path | None = None


def _n(seconds: float, rate: float) -> int:
    """Whole frames (video) or samples (audio) at `seconds`. Positions are rounded once, from absolute
    times, and lengths are differences of those, so rounding never accumulates along a track."""
    return round(seconds * rate)


def _clip(name: str, path: Path, count: int, rate: float, meta: dict) -> otio.schema.Clip:
    clip = otio.schema.Clip(
        name=name,
        media_reference=otio.schema.ExternalReference(target_url=str(path)),
        source_range=otio.opentime.TimeRange(otio.opentime.RationalTime(0, rate), otio.opentime.RationalTime(count, rate)),
    )
    clip.metadata["promo"] = {**meta, "sha256": sha256_file(path)}
    return clip


def _gap(count: int, rate: float, name: str = "", meta: dict | None = None) -> otio.schema.Gap:
    gap = otio.schema.Gap(
        name=name,
        source_range=otio.opentime.TimeRange(otio.opentime.RationalTime(0, rate), otio.opentime.RationalTime(count, rate)),
    )
    if meta:
        gap.metadata["promo"] = meta
    return gap


def _frames(s: TimedShot, fps: float) -> int:
    return _n(s.end_s, fps) - _n(s.start_s, fps)


def _existing(path: Path) -> Path | None:
    return path if path.exists() else None


def collect_media(run: RunPaths, timing: Timing, *, draft: bool = False) -> Media:
    shot_dir = run.renders / ("draft/shots" if draft else "shots")
    shots = {s.id: shot_dir / f"{s.id}.mp4" for s in timing.shots
             if (shot_dir / f"{s.id}.mp4").exists() and _frames(s, timing.fps) > 0}
    vo = {l.id: run.root / l.file for l in timing.lines if l.file and (run.root / l.file).exists()}
    return Media(
        shots=shots,
        vo=vo,
        music=_existing(run.music_dir / "bed.wav"),
        sfx=_existing(run.sfx_dir / "track.wav"),
        overlay=_existing(run.captions_dir / "overlay" / "captions.mov"),
    )


def build_timeline(timing: Timing, media: Media, name: str = "promo") -> otio.schema.Timeline:
    """Lay timing.json and the rendered media out on tracks, every item at its absolute time.

    V1 runs the whole film. Every item that is not a Clip is filled with the pack background by the
    renderer: a shot without a render is a Gap named `missing:<shot>` (metadata shot + missing), and
    time that no shot covers (a hole between chapters, or after the last shot) is an unnamed Gap.
    Items that would overlap on one track raise TimelineError.
    """
    fps, total = timing.fps, timing.duration_s
    tl = otio.schema.Timeline(name=name)
    tl.metadata["promo"] = {"duration_s": total, "fps": fps, "width": timing.width, "height": timing.height,
                            "preroll_s": timing.preroll_s}

    v1 = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    cursor = 0
    for s in timing.shots:
        start, end = _n(s.start_s, fps), _n(s.end_s, fps)
        if start < cursor - 1:
            raise TimelineError(f"shot {s.id} starts at {s.start_s:.3f}s, before the previous shot ends "
                                f"at {cursor / fps:.3f}s; one video track cannot overlap")
        start = max(start, cursor)
        if end <= start:  # no frames to show, including a shot the one-frame bump above has used up
            continue
        if start > cursor:  # nothing is scheduled here; unnamed, and the renderer fills it with the background
            v1.append(_gap(start - cursor, fps))
        src = media.shots.get(s.id)
        if src is not None:
            v1.append(_clip(s.id, src, end - start, fps, {"shot": s.id, "engine": s.engine}))
        else:
            v1.append(_gap(end - start, fps, f"missing:{s.id}", {"shot": s.id, "missing": True}))
        cursor = end
    if cursor < _n(total, fps):  # V1 always runs the whole film
        v1.append(_gap(_n(total, fps) - cursor, fps))
    tl.tracks.append(v1)

    if media.overlay is not None:
        v2 = otio.schema.Track(name="V2", kind=otio.schema.TrackKind.Video)
        v2.append(_clip("captions", media.overlay, _n(total, fps), fps, {"role": "captions"}))
        tl.tracks.append(v2)

    a1 = otio.schema.Track(name="A1 VO", kind=otio.schema.TrackKind.Audio)
    cursor = 0
    for line in sorted(timing.lines, key=lambda l: l.start_s):
        path = media.vo.get(line.id)
        if path is None:
            continue
        start, end = _n(line.start_s, AUDIO_RATE), _n(min(line.end_s, total), AUDIO_RATE)
        if start < cursor - 1:
            raise TimelineError(f"VO line {line.id} starts at {line.start_s:.3f}s, before the previous line ends "
                                f"at {cursor / AUDIO_RATE:.3f}s; one audio track cannot overlap")
        start = max(start, cursor)
        if end <= start:
            continue
        if start > cursor:
            a1.append(_gap(start - cursor, AUDIO_RATE))
        a1.append(_clip(line.id, path, end - start, AUDIO_RATE, {"line": line.id}))
        cursor = end
    tl.tracks.append(a1)

    a2 = otio.schema.Track(name="A2 Music", kind=otio.schema.TrackKind.Audio)
    if media.music is not None:
        pre = _n(timing.preroll_s, AUDIO_RATE)
        a2.append(_gap(pre, AUDIO_RATE))
        length = min(_n(duration_s(media.music), AUDIO_RATE), _n(total, AUDIO_RATE) - pre)
        a2.append(_clip("music", media.music, length, AUDIO_RATE, {"role": "music"}))
    tl.tracks.append(a2)

    a3 = otio.schema.Track(name="A3 SFX", kind=otio.schema.TrackKind.Audio)
    if media.sfx is not None:
        length = min(_n(duration_s(media.sfx), AUDIO_RATE), _n(total, AUDIO_RATE))
        a3.append(_clip("sfx", media.sfx, length, AUDIO_RATE, {"role": "sfx"}))
    tl.tracks.append(a3)
    return tl


def write_timeline(tl: otio.schema.Timeline, path: Path) -> Path:
    path = Path(path)
    with atomic_path(path) as tmp:
        otio.adapters.write_to_file(tl, str(tmp))
    return path


def read_timeline(path: Path) -> otio.schema.Timeline:
    return otio.adapters.read_from_file(str(path))
