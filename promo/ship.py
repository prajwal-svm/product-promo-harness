"""promo ship: poster, critic gate, README, reveal."""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

from promo.audio_analysis import duration_s
from promo.critique import rounds
from promo.proc import atomic_path, run, sha256_file
from promo.run import RunPaths

RENDERS_HEADING = "## Renders"


class ShipError(RuntimeError):
    """The film is not cleared to ship."""


class NoMaster(ShipError):
    """There is no master to ship yet: a missing prerequisite, not a failed gate."""


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _same_file(r: RunPaths, named: str, master: Path) -> bool:
    p = Path(named)
    return (p if p.is_absolute() else r.root / p).resolve() == master.resolve()


def critic_gate(r: RunPaths, master: Path, master_sha256: str | None = None) -> tuple[int | None, tuple[str, str] | None]:
    """The latest critic round, and None when it cleared this master, else (why, what to do).

    A round clears the master when its automatic checks passed (summary.json `ok`), the critic subagent passed it
    (scores.json `pass`) and it judged this very master: the SHA-256 the round recorded of the video it judged is the
    master's. A round from before the hash was recorded must have judged renders/master.mp4, made no earlier than the
    round.
    """
    n = max(rounds(r), default=None)
    if n is None:
        return None, ("no critic round yet", "run `promo critique`, then the critic subagent")
    rd = r.critic / str(n)
    summary = _read_json(rd / "summary.json")
    if summary is None:
        return n, (f"critic round {n} has no readable summary.json", "run `promo critique` again")
    judged = summary.get("video")
    judged = judged if isinstance(judged, str) and judged else None
    other = judged is not None and not _same_file(r, judged, master)
    if isinstance(summary.get("video_sha256"), str):
        if summary["video_sha256"] != (master_sha256 or sha256_file(master)):
            if other:
                return n, (f"critic round {n} judged {judged}, not renders/master.mp4",
                           "run `promo critique` on the master")
            return n, (f"renders/master.mp4 changed after critic round {n}", "run `promo critique` again")
    elif other:
        return n, (f"critic round {n} judged {judged}, not renders/master.mp4", "run `promo critique` on the master")
    elif master.stat().st_mtime > (rd / "summary.json").stat().st_mtime:
        return n, (f"renders/master.mp4 changed after critic round {n}", "run `promo critique` again")
    if summary.get("ok") is not True:
        return n, (f"critic round {n} failed its automatic checks", "fix them and run `promo critique` again")
    scores = _read_json(rd / "scores.json")
    if scores is None:
        return n, (f"critic round {n} has no scores.json yet",
                   f"the critic subagent writes it next to critic/{n}/PROMPT.md")
    if scores.get("pass") is not True:
        return n, (f"critic round {n} did not pass (scores.json pass is not true)",
                   "fix the problems in its review.md, re-render and re-critique")
    return n, None


def _render_key(line: str) -> str:
    """What makes two Renders lines the same shipment: the master's hash and the outcome, not the date."""
    return re.sub(r"^- \d{4}-\d{2}-\d{2}: ", "", line)


def _add_render_line(readme: Path, line: str, title: str) -> None:
    """Add `line` under the README's Renders section (made at the end if missing). The same master shipped with the
    same outcome again, on any day, is kept as one line."""
    text = readme.read_text() if readme.exists() else f"# {title}\n"
    lines = text.rstrip("\n").split("\n")
    if any(_render_key(x) == _render_key(line) for x in lines if x.startswith("- ")):
        return
    if RENDERS_HEADING not in lines:
        lines += ["", RENDERS_HEADING, "", line]
    else:
        head = lines.index(RENDERS_HEADING)
        end = next((i for i in range(head + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
        while end > head + 1 and not lines[end - 1].strip():
            end -= 1
        lines[end:end] = [line] if end > head + 1 else ["", line]
    with atomic_path(readme) as tmp:
        tmp.write_text("\n".join(lines) + "\n")


def ship(r: RunPaths, *, reveal: bool = False, force: bool = False) -> dict:
    """Write renders/poster.png (frame 0, the thumbnail pre-roll) and a Renders line in the run README, behind the
    critic gate. `force` ships past a failed gate and says so in the README. `reveal` shows the master in Finder."""
    master = r.renders / "master.mp4"
    if not master.is_file():
        raise NoMaster("no renders/master.mp4; run `promo assemble` first")
    master_sha256 = sha256_file(master)
    n, failed = critic_gate(r, master, master_sha256)
    if failed and not force:
        why, todo = failed
        raise ShipError(f"{why}; {todo}, or use --force")
    note = f"shipped without a passing critic round (--force: {failed[0]})." if failed else f"passed critic round {n}."
    poster = r.renders / "poster.png"
    with atomic_path(poster) as tmp:
        run(["ffmpeg", "-v", "error", "-y", "-i", master, "-frames:v", "1", "-update", "1", tmp])
    readme = r.root / "README.md"
    line = (f"- {_dt.date.today().isoformat()}: `renders/master.mp4` ({duration_s(master):.1f} s, sha256 "
            f"{master_sha256[:12]}), poster `renders/poster.png`; {note}")
    _add_render_line(readme, line, r.root.name)
    if reveal:
        run(["open", "-R", master])
    return {"master": master, "poster": poster, "readme": readme, "critic_round": n, "passed": failed is None,
            "why": failed[0] if failed else None}
