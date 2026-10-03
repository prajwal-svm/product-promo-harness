"""Render HyperFrames shots one project at a time; skip shots whose inputs did not change."""

from __future__ import annotations

import json
from pathlib import Path

from promo.hyperframes import (
    HF_VERSION,
    build_shot_project,
    ensure_shot,
    render,
    scaffold_scenes,
    snapshot,
    timing_slice,
)
from promo.packs import Pack
from promo.proc import sha256_hex
from promo.run import RunPaths
from promo.timing import Timing


def design_for(run: RunPaths, pack: Pack) -> dict:
    if run.design.exists():
        return json.loads(run.design.read_text())
    return {"pack": pack.name, "tokens": pack.tokens}


def shot_hash(scenes_dir: Path, timing: Timing, shot_id: str, design: dict, quality: str, fps: int) -> str:
    parts = [HF_VERSION, quality, str(fps), json.dumps(timing_slice(timing, shot_id), sort_keys=True),
             json.dumps(design, sort_keys=True), (scenes_dir / "shots" / f"{shot_id}.html").read_text()]
    for sub in ("lib",):
        for f in sorted((scenes_dir / sub).rglob("*")):
            if f.is_file():
                parts.append(f.read_bytes().hex())
    return sha256_hex(*parts)


def _selected(timing: Timing, only: list[str] | None, sample: tuple[float, float] | None) -> list[str]:
    ids = [s.id for s in timing.shots if s.engine == "hyperframes"]
    if only:
        ids = [i for i in ids if i in only]
    if sample:
        a, b = sample
        ids = [s.id for s in timing.shots if s.id in ids and s.start_s < b and s.end_s > a]
    return ids


def render_shots(run: RunPaths, timing: Timing, pack: Pack, *, mode: str = "final", only: list[str] | None = None,
                 sample: tuple[float, float] | None = None, force: bool = False) -> list[Path]:
    design = design_for(run, pack)
    scaffold_scenes(run.scenes)
    draft = mode == "draft"
    fps = 30 if draft else timing.fps
    quality = "draft" if draft else "high"
    out_dir = run.renders / ("draft/shots" if draft else "shots")
    outs = []
    for shot_id in _selected(timing, only, sample):
        ensure_shot(run.scenes, shot_id)
        out = out_dir / f"{shot_id}.mp4"
        digest = shot_hash(run.scenes, timing, shot_id, design, quality, fps)
        side = out.with_name(out.name + ".render.json")
        if not force and out.exists() and side.exists() and json.loads(side.read_text())["inputs"].get("hash") == digest:
            outs.append(out)
            continue
        project = build_shot_project(run.scenes, run.render_dir, timing, shot_id, design)
        shot = timing.shot(shot_id)
        outs.append(render(project, out, fps=fps, width=timing.width, height=timing.height,
                           duration=shot.duration_s, quality=quality, inputs={"hash": digest}))
    return outs


def stills(run: RunPaths, timing: Timing, pack: Pack) -> list[Path]:
    design = design_for(run, pack)
    scaffold_scenes(run.scenes)
    out: list[Path] = []
    for shot_id in _selected(timing, None, None):
        ensure_shot(run.scenes, shot_id)
        shot = timing.shot(shot_id)
        last = shot.duration_s - 1 / timing.fps
        times = {0.0, round(shot.duration_s / 2, 3), round(last, 3)}
        times |= {round(min(last, e.t_local + 0.25), 3) for e in timing.events_in(shot_id)}
        project = build_shot_project(run.scenes, run.render_dir, timing, shot_id, design)
        out += snapshot(project, sorted(times), run.renders / "stills" / shot_id)
    return out
