"""Assemble: OTIO edit → picture + pre-master mix → mastered MP4 with measured true peak."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Callable

from promo.audio_analysis import loudness
from promo.mix import loudnorm_measure, master_audio, master_gain_db, mux
from promo.packs import Pack
from promo.proc import ToolError, atomic_path, run, sha256_file, write_sidecar
from promo.render import plan_ffmpeg
from promo.run import RunPaths
from promo.timeline import build_timeline, collect_media, write_timeline
from promo.timing import Timing

DRAFT_FPS = 30
RENDER_TIMEOUT_S = 3600
CEILING_DB = -2.0
CEILING_STEP_DB = 0.5
CEILING_RETRIES = 3


class AssembleError(RuntimeError):
    """The film cannot be assembled, or its master cannot be brought within the true-peak limit."""


class MissingRenders(AssembleError):
    """No shot has been rendered for this mode yet: a missing prerequisite, not a failure."""


class StaleRenders(MissingRenders):
    """Shot renders made for other inputs than the current ones: `promo scenes` has to run first."""


def _master_loop(pre: Path, picture: Path, final: Path, work: Path, *, target_i: float, tp_limit: float,
                 draft: bool) -> dict:
    """Measure the mix once, then master (one linear gain, then the limiter), mux and measure the true peak on the
    MP4 itself (AAC adds inter-sample overshoot). While it is above `tp_limit`, lower the limiter ceiling 0.5 dB and
    redo the master and the mux, at most 3 times."""
    measured = loudnorm_measure(pre, target_i, CEILING_DB)
    gain = master_gain_db(measured, target_i)
    master = work / ("master-draft.wav" if draft else "master.wav")
    ceiling = CEILING_DB
    for attempt in range(CEILING_RETRIES + 1):
        master_audio(pre, master, target_i=target_i, ceiling_db=ceiling, measured=measured)
        mux(picture, master, final, draft=draft)
        m = loudness(final)
        if m["TP"] <= tp_limit + 1e-9:
            return {**m, "ceiling_db": ceiling, "normalization": "limiter-only" if gain is None else "linear-gain+limiter",
                    "gain_db": gain}
        if attempt < CEILING_RETRIES:
            ceiling -= CEILING_STEP_DB
    raise AssembleError(f"true peak {m['TP']:.2f} dBTP is still above {tp_limit} after lowering the limiter to "
                        f"{ceiling:.1f} dBFS; check the mix for a hot SFX or music clip")


def _design(run_paths: RunPaths, pack: Pack) -> dict:
    """The run's design (design.json, which the shots render with), else the pack's tokens."""
    from promo.scenes import design_for

    try:
        return design_for(run_paths, pack)
    except ValueError as e:  # JSONDecodeError and UnicodeDecodeError are both ValueErrors
        raise AssembleError(f"design.json cannot be read ({type(e).__name__}); fix or delete it and run again") from e


def _background(design: dict, pack: Pack) -> str:
    tokens = design.get("tokens") if isinstance(design, dict) else None
    colors = tokens.get("colors") if isinstance(tokens, dict) else None
    bg = colors.get("bg") if isinstance(colors, dict) else None
    return bg or pack.tokens.get("colors", {}).get("bg", "#000000")


def check_shots(run_paths: RunPaths, timing: Timing, design: dict, shots: dict[str, Path], *,
                draft: bool) -> tuple[list[str], list[str]]:
    """(stale, unchecked) shot ids among the rendered HyperFrames `shots`.

    Stale: the render's sidecar hash differs from what `promo scenes` would hash for the current timing, scene file,
    lib and design (the same hash, quality and fps it renders with), or its scene file is gone. Unchecked: the render
    has no readable sidecar, so it was not made by `promo scenes` and cannot be compared.
    """
    from promo.scenes import shot_hash

    quality, fps = ("draft", DRAFT_FPS) if draft else ("high", timing.fps)
    stale, unchecked = [], []
    for s in timing.shots:
        if s.engine != "hyperframes" or s.id not in shots:
            continue
        try:
            recorded = json.loads(shots[s.id].with_name(shots[s.id].name + ".render.json").read_text())["inputs"]["hash"]
        except (OSError, ValueError, KeyError, TypeError):
            unchecked.append(s.id)
            continue
        try:
            current = shot_hash(run_paths.scenes, timing, s.id, design, quality, fps)
        except OSError:  # the scene file is gone: the render cannot be made again as it is
            current = None
        if recorded != current:
            stale.append(s.id)
    return stale, unchecked


def _ffmpeg_version() -> str:
    first = run(["ffmpeg", "-version"]).stdout.split("\n", 1)[0]   # "ffmpeg version 8.1.1 Copyright ..."
    parts = first.split()
    return parts[2] if len(parts) > 2 else "unknown"


def _json_safe(report: dict) -> dict:
    """-inf (a silent master) is not JSON; write it as null."""
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in report.items()}


def assemble(run_paths: RunPaths, timing: Timing, pack: Pack, *, draft: bool = False, allow_stale: bool = False,
             warn: Callable[[str], None] | None = None) -> Path:
    """Write timeline.otio, render it, master the mix and mux renders/master.mp4 (with --draft: renders/draft/
    animatic.mp4 at half size and 30 fps, from renders/draft/shots). The loudness measured on the MP4 goes next to
    it as <name>.loudness.json. A master that fails the true-peak check never replaces the previous one.

    Shot renders made for other inputs stop the master (StaleRenders) unless `allow_stale`; for the draft, and with
    `allow_stale`, they are a warning passed to `warn`, as are renders that cannot be checked.
    """
    warn = warn or (lambda message: None)
    media = collect_media(run_paths, timing, draft=draft)
    if not media.shots:
        where = "renders/draft/shots" if draft else "renders/shots"
        raise MissingRenders(f"no shot renders in {where}; run `promo scenes{' --draft' if draft else ''}` first")
    design = _design(run_paths, pack)
    stale, unchecked = check_shots(run_paths, timing, design, media.shots, draft=draft)
    if stale:
        what = (f"stale shot renders: {', '.join(stale)} ({'their' if len(stale) > 1 else 'its'} timing, scene or "
                f"design changed since {'they were' if len(stale) > 1 else 'it was'} rendered)")
        if not (draft or allow_stale):
            raise StaleRenders(f"{what}; run `promo scenes` first, or pass --allow-stale")
        warn(f"{what}; assembling {'them' if len(stale) > 1 else 'it'} anyway")
    if unchecked:
        warn(f"no render record for {', '.join(unchecked)}, so {'they' if len(unchecked) > 1 else 'it'} cannot be "
             f"checked against the current timing")
    tl = build_timeline(timing, media)
    otio_path = run_paths.renders / "draft" / "timeline.otio" if draft else run_paths.timeline
    write_timeline(tl, otio_path)
    w, h = (timing.width // 2, timing.height // 2) if draft else (timing.width, timing.height)
    w, h = w - w % 2, h - h % 2
    fps = DRAFT_FPS if draft else timing.fps
    work = run_paths.mix_dir
    work.mkdir(parents=True, exist_ok=True)
    picture = work / ("picture-draft.mp4" if draft else "picture.mp4")
    pre = work / ("mix_pre-draft.wav" if draft else "mix_pre.wav")
    try:
        cmd = plan_ffmpeg(tl, width=w, height=h, fps=fps, bg=_background(design, pack),
                          duck_db=pack.music.get("duck_db"), fade_out_s=pack.music.get("fade_out_s"),
                          picture_out=picture, mix_out=pre, draft=draft)
    except ValueError as e:
        raise AssembleError(f"pack {pack.name}: {e}") from e
    try:
        run(cmd, timeout=RENDER_TIMEOUT_S)
    except ToolError as e:  # the command line holds the whole graph: lead with ffmpeg's own last word instead
        lines = [l for l in str(e).splitlines() if l.strip()]
        reason = lines[-1].strip() if len(lines) > 1 else "see the log"
        raise ToolError(f"ffmpeg could not render {otio_path.name}: {reason}\n{e}") from e
    except subprocess.TimeoutExpired as e:
        raise ToolError(f"ffmpeg did not finish rendering {otio_path.name} in {RENDER_TIMEOUT_S // 60} minutes") from e

    final = run_paths.renders / ("draft/animatic.mp4" if draft else "master.mp4")
    staged = final.with_name(f".{final.stem}.staging{final.suffix}")
    try:
        report = _master_loop(pre, picture, staged, work, target_i=pack.music.get("lufs", -14),
                              tp_limit=pack.music.get("true_peak", -1.0), draft=draft)
        os.replace(staged, final)
    finally:
        staged.unlink(missing_ok=True)
    with atomic_path(final.with_name(final.stem + ".loudness.json")) as tmp:
        tmp.write_text(json.dumps(_json_safe(report), indent=2))
    write_sidecar(final, tool="ffmpeg", version=_ffmpeg_version(), flags=["draft"] if draft else ["master"],
                  inputs={"timeline": sha256_file(otio_path)})
    return final
