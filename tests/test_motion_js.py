import json
import shutil
import subprocess

import pytest

from promo.config import packs_dir

MOTION = packs_dir() / "_core" / "scene-template" / "lib" / "motion.js"

SCRIPT = """
globalThis.window = globalThis;
require(process.argv[1]);
const M = window.PromoMotion;
const heavy = M.ease("heavy");
const r1 = M.rng(42), r2 = M.rng(42);
const out = {
  s0: M.spring(0), sBig: M.spring(5), sOver: Math.max(...Array.from({length: 400}, (_, i) => M.spring(i / 200, 220, 14))),
  heavy1: heavy(1), heavyDur: heavy.duration, snappyDur: M.ease("snappy").duration,
  track: M.track(10, [[0, 0], [1, 100], [2, 40]]),
  ind: M.indicator(1.05, [[0, 0], [1, 300]]),
  swapMid: M.swapAlpha(0.5, 0, 1), swapEarly: M.swapAlpha(0.05, 0, 1),
  rngSame: [r1(), r1(), r1()].join() === [r2(), r2(), r2()].join(),
  loop: M.loopT(-0.5, 2),
};
let threw = false; try { M.ease("nope"); } catch (e) { threw = true; }
out.threw = threw;
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node needed")
def test_motion_js_contract():
    res = subprocess.run(["node", "-e", SCRIPT, str(MOTION)], capture_output=True, text=True, check=True)
    o = json.loads(res.stdout)
    assert o["s0"] == 0 and o["sBig"] == pytest.approx(1, abs=1e-4)
    assert o["sOver"] > 1.0          # playful overshoots
    assert o["heavy1"] == 1
    assert o["snappyDur"] < o["heavyDur"]
    assert o["track"] == pytest.approx(40, abs=0.01)
    assert o["ind"]["left"] <= o["ind"]["right"]
    assert o["swapMid"] == 1 and o["swapEarly"] == 0
    assert o["rngSame"] is True
    assert o["loop"] == 1.5
    assert o["threw"] is True


ACCURACY = """
globalThis.window = globalThis;
require(process.argv[1]);
const M = window.PromoMotion;
// Reference: integrate x'' = k(1 - x) - d x' with a tiny semi-implicit Euler step.
const DT = 1e-5;
const TIMES = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5];
function worstError(k, d) {
  const at = new Map(TIMES.map((t) => [Math.round(t / DT), t]));
  let x = 0, v = 0, worst = 0;
  for (let n = 1; n <= Math.round(TIMES[TIMES.length - 1] / DT); n++) {
    v += (k * (1 - x) - d * v) * DT;
    x += v * DT;
    if (at.has(n)) worst = Math.max(worst, Math.abs(x - M.spring(at.get(n), k, d)));
  }
  return worst;
}
const peak = (k, d) => Math.max(...Array.from({ length: 3001 }, (_, i) => M.spring(i / 1000, k, d)));
console.log(JSON.stringify({
  err: {
    overdamped: worstError(100, 60),
    heavy: worstError(...M.PRESETS.heavy),
    default: worstError(...M.PRESETS.default),
    snappy: worstError(...M.PRESETS.snappy),
    playful: worstError(...M.PRESETS.playful),
    critical: worstError(100, 20),
  },
  heavyPeak: peak(...M.PRESETS.heavy),
  defaultPeak: peak(...M.PRESETS.default),
}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node needed")
def test_spring_matches_numeric_integration_in_every_damping_regime():
    res = subprocess.run(["node", "-e", ACCURACY, str(MOTION)], capture_output=True, text=True, check=True)
    o = json.loads(res.stdout)
    for name, err in o["err"].items():
        assert err < 1e-3, f"{name} spring is off by {err}"
    # Type springs (heavy, default) must not overshoot at all.
    assert o["heavyPeak"] <= 1.0 + 1e-9
    assert o["defaultPeak"] <= 1.0 + 1e-9
