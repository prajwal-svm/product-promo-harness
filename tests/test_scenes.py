import json
import shutil
import subprocess
from pathlib import Path

import pytest

from promo import scenes
from promo.packs import load_pack
from promo.run import new_run
from promo.shotlist import Shotlist
from promo.timing import Measured, resolve

FIX = Path(__file__).parent / "fixtures" / "shotlist_music.json"


def _setup(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    run.shotlist.write_text(FIX.read_text())
    pack = load_pack("saas-kinetic")
    t = resolve(Shotlist.model_validate(json.loads(FIX.read_text())), Measured(), pack.limits(), pack.text_mode())
    return run, pack, t


def test_design_for_falls_back_to_pack_tokens(tmp_path):
    run, pack, _ = _setup(tmp_path)
    d = scenes.design_for(run, pack)
    assert d["tokens"]["colors"]["accent"] == "#C6F135" and d["pack"] == "saas-kinetic"


def test_render_shots_skips_unchanged(tmp_path, monkeypatch):
    run, pack, t = _setup(tmp_path)
    calls = []

    def fake_render(project, out, **kw):
        calls.append((project.name, kw["fps"], kw["quality"]))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mp4")
        from promo.proc import write_sidecar
        write_sidecar(out, tool="hyperframes", version="0.8.98", flags=[], inputs=kw["inputs"])
        return out

    monkeypatch.setattr(scenes, "render", fake_render)
    outs = scenes.render_shots(run, t, pack, mode="final")
    assert [p.name for p in outs] == ["c01-s1.mp4", "c02-s1.mp4", "c02-s2.mp4"]
    assert calls[0] == ("c01-s1", 60, "high")
    scenes.render_shots(run, t, pack, mode="final")
    assert len(calls) == 3                      # nothing changed → nothing re-rendered
    (run.shots_dir / "c02-s2.html").write_text((run.shots_dir / "c02-s2.html").read_text() + "\n<!-- edit -->")
    scenes.render_shots(run, t, pack, mode="final")
    assert [c[0] for c in calls[3:]] == ["c02-s2"]  # only the edited shot
    scenes.render_shots(run, t, pack, mode="draft", only=["c01-s1"])
    assert calls[-1] == ("c01-s1", 30, "draft")
    assert (run.renders / "draft" / "shots" / "c01-s1.mp4").exists()


def test_sample_selects_overlapping_shots(tmp_path, monkeypatch):
    run, pack, t = _setup(tmp_path)
    seen = []
    monkeypatch.setattr(scenes, "render", lambda project, out, **kw: seen.append(project.name) or out)
    scenes.render_shots(run, t, pack, mode="sample", sample=(5.0, 7.0), force=True)
    assert seen == ["c02-s1", "c02-s2"]


@pytest.mark.live
def test_live_render_lint_and_snapshot(tmp_path):
    run, pack, t = _setup(tmp_path)
    outs = scenes.render_shots(run, t, pack, mode="final", only=["c01-s1"])
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(outs[0])],
                           capture_output=True, text=True, check=True)
    info = json.loads(probe.stdout)
    assert info["streams"][0]["width"] == 1920 and info["streams"][0]["r_frame_rate"] == "60/1"
    assert float(info["format"]["duration"]) == pytest.approx(4.3, abs=0.05)
    from promo import hyperframes as hf
    project = run.render_dir / "c01-s1"
    assert hf.lint(project)["errorCount"] == 0
    snaps = hf.snapshot(project, [0.1, 1.0], tmp_path / "snaps")
    assert len(snaps) == 2


def _cli_run(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from promo.cli import app
    from promo.timing import write_timing

    run, pack, t = _setup(tmp_path)
    monkeypatch.chdir(run.root)
    return CliRunner(), app, run, t, write_timing


def test_cli_scenes_needs_timing_first(tmp_path, monkeypatch):
    runner, app, _, _, _ = _cli_run(tmp_path, monkeypatch)
    res = runner.invoke(app, ["scenes", "--scaffold"])
    assert res.exit_code == 2 and "promo timing" in res.output


def test_cli_scenes_scaffold_creates_shot_files(tmp_path, monkeypatch):
    runner, app, run, t, write_timing = _cli_run(tmp_path, monkeypatch)
    write_timing(t, run.timing)
    res = runner.invoke(app, ["scenes", "--scaffold"])
    assert res.exit_code == 0, res.output
    assert [Path(p).name for p in res.output.split()] == ["c01-s1.html", "c02-s1.html", "c02-s2.html"]
    assert 'data-composition-id="c02-s2"' in (run.shots_dir / "c02-s2.html").read_text()
    assert (run.scenes / "lib" / "motion.js").exists()


def test_cli_scenes_rejects_a_malformed_sample(tmp_path, monkeypatch):
    runner, app, run, t, write_timing = _cli_run(tmp_path, monkeypatch)
    write_timing(t, run.timing)
    for bad in ("12", "9-3", "a-b"):
        res = runner.invoke(app, ["scenes", "--sample", bad])
        assert res.exit_code == 2 and "--sample" in res.output, bad
