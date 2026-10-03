"""The only door to HyperFrames: pinned version, quiet env, per-shot projects, render/lint/check/snapshot."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from promo.config import packs_dir
from promo.proc import atomic_path, run, write_sidecar
from promo.timing import Timing

HF_VERSION = "0.8.98"
TEMPLATE_DIR = packs_dir() / "_core" / "scene-template"


class HyperframesError(RuntimeError):
    pass


def hf_cmd(*args) -> list[str]:
    return ["npx", "--yes", f"hyperframes@{HF_VERSION}", *[str(a) for a in args]]


def hf_env() -> dict:
    env = dict(os.environ)
    env.update(HYPERFRAMES_SKIP_SKILLS="1", HYPERFRAMES_NO_TELEMETRY="1", HYPERFRAMES_NO_UPDATE_CHECK="1")
    return env


def scaffold_scenes(scenes_dir: Path) -> None:
    for sub in ("lib", "assets"):
        shutil.copytree(TEMPLATE_DIR / sub, scenes_dir / sub, dirs_exist_ok=True)
    (scenes_dir / "shots").mkdir(parents=True, exist_ok=True)


def ensure_shot(scenes_dir: Path, shot_id: str) -> Path:
    path = scenes_dir / "shots" / f"{shot_id}.html"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((TEMPLATE_DIR / "shots" / "_shot-basic.html").read_text().replace("{{SHOT_ID}}", shot_id))
    return path


def _num(x: float) -> str:
    return f"{x:.6f}".rstrip("0").rstrip(".")


def stamp_root(html: str, shot_id: str, *, duration: float, fps: int, width: int, height: int) -> str:
    m = re.search(rf'<div\b[^>]*\bdata-composition-id="{re.escape(shot_id)}"[^>]*>', html)
    if not m:
        raise HyperframesError(f'shot {shot_id}: no root element with data-composition-id="{shot_id}"')
    tag = m.group(0)
    for attr, val in (("data-start", "0"), ("data-duration", _num(duration)), ("data-fps", str(fps)),
                      ("data-width", str(width)), ("data-height", str(height))):
        if re.search(rf'\s{attr}="[^"]*"', tag):
            tag = re.sub(rf'(\s){attr}="[^"]*"', rf'\1{attr}="{val}"', tag)
        else:
            tag = tag[:-1] + f' {attr}="{val}">'
    out = html[: m.start()] + tag + html[m.end():]
    return re.sub(r'(<meta name="viewport" content=")[^"]*(")', rf"\g<1>width={width}, height={height}\g<2>", out)


def timing_slice(timing: Timing, shot_id: str) -> dict:
    shot = timing.shot(shot_id)
    words = [w.model_dump() for line in timing.lines for w in line.words
             if shot.start_s - 0.001 <= w.startMs / 1000 < shot.end_s]
    return {
        "shot": shot.model_dump(),
        "chapter": timing.chapter(shot.chapter).model_dump(),
        "events": [e.model_dump() for e in timing.events_in(shot_id)],
        "words": words,
        "canvas": [timing.width, timing.height, timing.fps, timing.aspect],
        "limits": timing.limits.model_dump(),
    }


def motion_assertions(timing: Timing, shot_id: str, html: str) -> dict:
    shot = timing.shot(shot_id)
    assertions = []
    if timing.shots[0].id == shot_id and "data-hero" in html:
        assertions.append({"kind": "appearsBy", "selector": "[data-hero]", "bySec": 0.5})
    if 'class="stage"' in html:
        assertions.append({"kind": "keepsMoving", "withinSelector": ".stage", "maxStaticSec": timing.limits.max_dead_s})
    return {"duration": shot.duration_s, "assertions": assertions}


def _link_tree(src: Path, dst: Path) -> None:
    def link(s, d):
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)
    shutil.copytree(src, dst, copy_function=link)


def build_shot_project(scenes_dir: Path, render_dir: Path, timing: Timing, shot_id: str, design: dict) -> Path:
    src = scenes_dir / "shots" / f"{shot_id}.html"
    if not src.exists():
        raise HyperframesError(f"missing {src}; run `promo scenes --scaffold` or write the shot")
    dest = render_dir / shot_id
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for sub in ("lib", "assets"):
        if (scenes_dir / sub).exists():
            _link_tree(scenes_dir / sub, dest / sub)
    shot = timing.shot(shot_id)
    html = stamp_root(src.read_text(), shot_id, duration=shot.duration_s, fps=timing.fps,
                      width=timing.width, height=timing.height)
    (dest / "index.html").write_text(html)
    (dest / "timing.json").write_text(timing.model_dump_json())
    (dest / "design.json").write_text(json.dumps(design))
    (dest / "index.motion.json").write_text(json.dumps(motion_assertions(timing, shot_id, html), indent=2))
    return dest


def estimate_frame_bytes(duration: float, fps: int, width: int, height: int) -> int:
    return int(duration * fps * width * height * 4 * 1.1)


def ensure_disk(path: Path, need_bytes: int, reserve_gb: float = 5.0) -> None:
    free = shutil.disk_usage(path).free
    if need_bytes + reserve_gb * 1e9 > free:
        raise HyperframesError(
            f"not enough disk for this render: needs ~{need_bytes / 1e9:.1f} GB of frame storage plus a "
            f"{reserve_gb:g} GB reserve, but {free / 1e9:.1f} GB is free at {path}. Free space or render fewer shots.")


def render(project: Path, out: Path, *, fps: int, width: int, height: int, duration: float, quality: str = "high",
           fmt: str = "mp4", gpu: bool = False, workers: int | None = None, inputs: dict | None = None) -> Path:
    ensure_disk(Path(tempfile.gettempdir()), estimate_frame_bytes(duration, fps, width, height))
    flags = ["--format", fmt, "--fps", str(fps), "--quality", quality,
             "--browser-gpu" if gpu else "--no-browser-gpu", "--quiet"]
    if workers:
        flags += ["--workers", str(workers)]
    with atomic_path(out) as tmp:
        run(hf_cmd("render", project, "--output", tmp, *flags), env=hf_env(), timeout=3600)
    write_sidecar(out, tool="hyperframes", version=HF_VERSION, flags=flags, inputs=inputs or {})
    return out


def _json_out(stdout: str) -> dict:
    start = stdout.find("{")
    if start < 0:
        raise HyperframesError(f"no JSON in hyperframes output: {stdout[:200]}")
    return json.loads(stdout[start:])


def lint(project: Path) -> dict:
    return _json_out(run(hf_cmd("lint", project, "--json"), env=hf_env(), check=False, timeout=300).stdout)


def check(project: Path, *, at_transitions: bool = True) -> dict:
    lint_res = lint(project)
    if lint_res.get("errorCount", 0) > 0:
        return {"ok": False, "skipped": "lint errors", "lint": lint_res}
    args = ["check", project, "--json", "--no-browser-gpu"] + (["--at-transitions"] if at_transitions else [])
    return _json_out(run(hf_cmd(*args), env=hf_env(), check=False, timeout=900).stdout)


def snapshot(project: Path, times: list[float], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run(hf_cmd("snapshot", project, "--at", ",".join(f"{t:.3f}" for t in times), "--no-end",
               "--describe", "false", "--no-browser-gpu", "-o", out_dir), env=hf_env(), timeout=900)
    return sorted(out_dir.glob("frame-*.png"))
