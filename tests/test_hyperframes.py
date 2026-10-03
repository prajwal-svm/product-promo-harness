import json
from collections import namedtuple
from pathlib import Path

import pytest

from promo import hyperframes as hf
from promo.packs import Limits
from promo.shotlist import Shotlist
from promo.timing import Measured, resolve

FIX = Path(__file__).parent / "fixtures" / "shotlist_music.json"


@pytest.fixture
def timing():
    return resolve(Shotlist.model_validate(json.loads(FIX.read_text())), Measured(), Limits())


def test_cmd_is_pinned_and_env_is_quiet():
    assert hf.hf_cmd("lint", "x") == ["npx", "--yes", "hyperframes@0.8.98", "lint", "x"]
    env = hf.hf_env()
    assert env["HYPERFRAMES_SKIP_SKILLS"] == "1" and env["HYPERFRAMES_NO_TELEMETRY"] == "1"
    assert env["HYPERFRAMES_NO_UPDATE_CHECK"] == "1"


def test_stamp_root_replaces_and_adds_attributes():
    html = '<meta name="viewport" content="width=1920, height=1080" />\n<div id="root" data-composition-id="c01-s1" data-duration="1">'
    out = hf.stamp_root(html, "c01-s1", duration=4.3, fps=60, width=1080, height=1920)
    assert 'data-duration="4.3"' in out and 'data-fps="60"' in out and 'data-start="0"' in out
    assert 'data-width="1080"' in out and 'data-height="1920"' in out
    assert 'content="width=1080, height=1920"' in out
    with pytest.raises(hf.HyperframesError, match='data-composition-id="nope"'):
        hf.stamp_root(html, "nope", duration=1, fps=60, width=1920, height=1080)


def test_build_shot_project(tmp_path, timing):
    scenes = tmp_path / "scenes"
    hf.scaffold_scenes(scenes)
    hf.ensure_shot(scenes, "c01-s1")
    hf.ensure_shot(scenes, "c02-s1")
    p1 = hf.build_shot_project(scenes, scenes / ".render", timing, "c01-s1", {"tokens": {"colors": {"bg": "#fff"}}})
    html = (p1 / "index.html").read_text()
    assert 'data-composition-id="c01-s1"' in html and 'data-duration="4.3"' in html
    assert (p1 / "lib" / "motion.js").exists() and (p1 / "assets" / "vendor" / "gsap.min.js").exists()
    assert json.loads((p1 / "timing.json").read_text())["duration_s"] == 8.3
    assert json.loads((p1 / "design.json").read_text())["tokens"]["colors"]["bg"] == "#fff"
    m1 = json.loads((p1 / "index.motion.json").read_text())
    kinds = [a["kind"] for a in m1["assertions"]]
    assert "appearsBy" in kinds and "keepsMoving" in kinds  # first shot carries the hero check
    p2 = hf.build_shot_project(scenes, scenes / ".render", timing, "c02-s1", {})
    assert "appearsBy" not in [a["kind"] for a in json.loads((p2 / "index.motion.json").read_text())["assertions"]]


def test_timing_slice_only_changes_with_its_shot(timing):
    a = hf.timing_slice(timing, "c02-s1")
    t2 = timing.model_copy(deep=True)
    t2.events[-1].t_local += 0.1  # c02-e2 lives in c02-s2
    assert hf.timing_slice(t2, "c02-s1") == a
    assert hf.timing_slice(t2, "c02-s2") != hf.timing_slice(timing, "c02-s2")


def test_render_refuses_when_disk_too_small(tmp_path, monkeypatch):
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(hf.shutil, "disk_usage", lambda p: Usage(1, 1, 3 * 10**9))
    monkeypatch.setattr(hf, "run", lambda *a, **k: pytest.fail("render must not start"))
    with pytest.raises(hf.HyperframesError, match="not enough disk"):
        hf.render(tmp_path, tmp_path / "o.mp4", fps=60, width=1920, height=1080, duration=10)


def test_render_command_and_sidecar(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw.get("env")
        Path(cmd[cmd.index("--output") + 1]).write_bytes(b"mp4")

    monkeypatch.setattr(hf, "run", fake_run)
    out = hf.render(tmp_path, tmp_path / "shots" / "c01-s1.mp4", fps=60, width=1920, height=1080, duration=1.0,
                    inputs={"hash": "abc"})
    cmd = seen["cmd"]
    assert cmd[:4] == ["npx", "--yes", "hyperframes@0.8.98", "render"]
    assert "--no-browser-gpu" in cmd and cmd[cmd.index("--fps") + 1] == "60" and cmd[cmd.index("--quality") + 1] == "high"
    assert seen["env"]["HYPERFRAMES_NO_TELEMETRY"] == "1"
    side = json.loads((out.parent / "c01-s1.mp4.render.json").read_text())
    assert side["inputs"] == {"hash": "abc"} and side["version"] == "0.8.98"


def test_check_skips_on_lint_errors(tmp_path, monkeypatch):
    class R:
        stdout = 'notice line\n{"errorCount": 1, "warningCount": 0, "findings": [{"code": "missing_gsap_script"}]}'
    monkeypatch.setattr(hf, "run", lambda *a, **k: R())
    res = hf.check(tmp_path)
    assert res["ok"] is False and res["skipped"] == "lint errors"
    assert res["lint"]["findings"][0]["code"] == "missing_gsap_script"


def test_snapshot_flags(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        out = Path(cmd[cmd.index("-o") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame-00-at-0.1s.png").write_bytes(b"png")

    monkeypatch.setattr(hf, "run", fake_run)
    shots = hf.snapshot(tmp_path, [0.1], tmp_path / "snaps")
    cmd = seen["cmd"]
    assert "--no-end" in cmd and cmd[cmd.index("--describe") + 1] == "false" and "--no-browser-gpu" in cmd
    assert cmd[cmd.index("--at") + 1] == "0.100"
    assert [p.name for p in shots] == ["frame-00-at-0.1s.png"]
