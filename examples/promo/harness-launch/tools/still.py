"""Fast look: snapshot film times (absolute seconds) through the harness's own project builder.

    uv run --project ../../.. python tools/still.py 1.2 3.4 40.0 [--out DIR]
Each time is mapped to the shot that contains it. Frames land in renders/look/<label>-<t>.png.
"""
import json
import shutil
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN.parents[2]))
from promo.hyperframes import build_shot_project, snapshot  # noqa: E402
from promo.timing import load_timing  # noqa: E402


def main(argv):
    out = RUN / "renders" / "look"
    if "--out" in argv:
        i = argv.index("--out"); out = Path(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    times = sorted(float(a) for a in argv)   # snapshot returns frames in time order
    timing = load_timing(RUN / "timing.json")
    design = json.loads((RUN / "design.json").read_text()) if (RUN / "design.json").exists() else {}
    by_shot = {}
    for t in times:
        s = next((s for s in timing.shots if s.start_s <= t < s.end_s), timing.shots[-1])
        by_shot.setdefault(s.id, []).append(t)
    out.mkdir(parents=True, exist_ok=True)
    for sid, ts in by_shot.items():
        s = timing.shot(sid)
        proj = build_shot_project(RUN / "scenes", RUN / "scenes" / ".render", timing, sid, design)
        local = [round(min(s.duration_s - 1 / 60, max(0.0, t - s.start_s)), 3) for t in ts]
        tmp = out / ("_" + sid)
        if tmp.exists():
            shutil.rmtree(tmp)
        frames = snapshot(proj, local, tmp)
        for f, t in zip(sorted(frames), sorted(ts)):
            dst = out / f"{t:07.3f}.png"
            shutil.move(str(f), dst)
            print(dst)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1:])
