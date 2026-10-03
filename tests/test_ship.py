import json
import os
import re
import subprocess

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from promo.cli import app
from promo.proc import sha256_file
from promo.run import new_run
from promo.ship import NoMaster, ShipError, ship

runner = CliRunner()


def _master(run):
    run.renders.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=2",
                    "-pix_fmt", "yuv420p", str(run.renders / "master.mp4")], check=True)


def _round(run, n, ok=True, passed=True, video=None, scores=True, sha="master"):
    """A finished round. Like `promo critique`, it records the hash of the video it judged (the master by default);
    `sha=None` is a round from before the hash was recorded."""
    d = run.critic / str(n)
    d.mkdir(parents=True)
    summary = {"ok": ok, "checks": []}
    if video is not None:
        summary["video"] = str(video)
    if sha == "master":
        sha = sha256_file(run.renders / "master.mp4") if (run.renders / "master.mp4").exists() else "0" * 64
    if sha is not None:
        summary["video_sha256"] = sha
    (d / "summary.json").write_text(json.dumps(summary))
    if scores:
        (d / "scores.json").write_text(json.dumps({"pass": passed}))


def _age_master(run, seconds=60):
    """Make the master older than everything written after it, as it is when a critic round judged it."""
    master = run.renders / "master.mp4"
    st = master.stat()
    os.utime(master, (st.st_atime - seconds, st.st_mtime - seconds))


def _pixels(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)


def test_ship_refuses_without_a_passing_critic(tmp_path):
    run = new_run(tmp_path, "s", "saas-kinetic", "16:9", None, "en")
    _master(run)
    with pytest.raises(ShipError, match="no critic round"):
        ship(run)
    _round(run, 1, ok=True, passed=False)
    with pytest.raises(ShipError, match="round 1"):
        ship(run)


def test_ship_writes_poster_and_readme(tmp_path):
    run = new_run(tmp_path, "s", "saas-kinetic", "16:9", None, "en")
    _master(run)
    _round(run, 1, ok=False)
    _round(run, 2, ok=True, passed=True)
    out = ship(run)
    assert out["poster"].exists() and out["master"].name == "master.mp4"
    assert out["critic_round"] == 2 and out["passed"] is True
    readme = (run.root / "README.md").read_text()
    assert "## Renders" in readme and "critic round 2" in readme
    assert readme.startswith("# s\n"), "the scaffolded README is kept, the section is appended"


def test_force_ship_is_recorded(tmp_path):
    run = new_run(tmp_path, "s", "saas-kinetic", "16:9", None, "en")
    _master(run)
    out = ship(run, force=True)
    assert out["passed"] is False and out["critic_round"] is None
    readme = (run.root / "README.md").read_text()
    assert "shipped without a passing critic round" in readme and "no critic round yet" in readme


def test_force_with_a_passing_round_records_the_pass(tmp_path):
    run = new_run(tmp_path, "s", "saas-kinetic", "16:9", None, "en")
    _master(run)
    _round(run, 1)
    out = ship(run, force=True)
    assert out["passed"] is True
    readme = (run.root / "README.md").read_text()
    assert "passed critic round 1" in readme and "without a passing" not in readme


def test_poster_is_the_first_frame(tmp_path):
    run = new_run(tmp_path, "s", "saas-kinetic", "16:9", None, "en")
    _master(run)
    _round(run, 1)
    poster = ship(run)["poster"]
    assert poster == run.renders / "poster.png"
    master = run.renders / "master.mp4"
    first, later = tmp_path / "first.png", tmp_path / "later.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(master), "-vf", r"select=eq(n\,0)", "-frames:v", "1",
                    str(first)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1", "-i", str(master), "-frames:v", "1", str(later)],
                   check=True)
    got = _pixels(poster)
    assert got.shape == (180, 320, 3)
    assert np.array_equal(got, _pixels(first))
    assert not np.array_equal(got, _pixels(later))
    assert not list(run.renders.glob(".poster*")), "no temporary file is left behind"


def test_ship_without_a_master_is_a_missing_prerequisite(tmp_path):
    run = new_run(tmp_path, "s", "saas-kinetic", "16:9", None, "en")
    _round(run, 1)
    with pytest.raises(NoMaster, match="promo assemble"):
        ship(run)
    with pytest.raises(NoMaster):
        ship(run, force=True)


def test_ship_refuses_a_round_without_scores(tmp_path):
    run = new_run(tmp_path, "s", "saas-kinetic", "16:9", None, "en")
    _master(run)
    _round(run, 1, scores=False)
    with pytest.raises(ShipError, match="no scores.json"):
        ship(run)


def test_ship_refuses_a_round_that_failed_its_automatic_checks(tmp_path):
    run = new_run(tmp_path, "s", "saas-kinetic", "16:9", None, "en")
    _master(run)
    _round(run, 1, ok=False, passed=True)
    with pytest.raises(ShipError, match="round 1 failed its automatic checks"):
        ship(run)


def test_ship_refuses_a_round_that_judged_another_video(tmp_path):
    """An animatic critic round (`promo critique --video renders/draft/animatic.mp4`) does not clear the master."""
    run = new_run(tmp_path, "s", "saas-kinetic", "16:9", None, "en")
    _master(run)
    _round(run, 1, video=run.renders / "draft" / "animatic.mp4", sha="a" * 64)
    with pytest.raises(ShipError, match="judged .*animatic.mp4"):
        ship(run)
    _round(run, 2, video=run.renders / "draft" / "animatic.mp4", sha=None)    # a round from before the hash
    with pytest.raises(ShipError, match="judged .*animatic.mp4"):
        ship(run)


def test_a_round_on_a_copy_of_the_master_clears_it(tmp_path):
    """The hash decides, not the name: `promo critique --video` on a byte-identical copy judged this master."""
    run = new_run(tmp_path, "s", "saas-kinetic", "16:9", None, "en")
    _master(run)
    _round(run, 1, video=run.renders / "copy.mp4")
    assert ship(run)["passed"] is True


def test_a_copied_or_touched_master_still_ships(tmp_path):
    """cp -R without -p, rsync without -t and touch make the master newer than the round; the bytes are the same."""
    run = new_run(tmp_path, "s", "saas-kinetic", "16:9", None, "en")
    _master(run)
    _round(run, 1)
    summary = run.critic / "1" / "summary.json"
    st = summary.stat()
    os.utime(summary, (st.st_atime - 3600, st.st_mtime - 3600))   # the round now looks an hour older than the master
    assert ship(run)["passed"] is True


def test_an_older_render_restored_over_the_master_is_refused(tmp_path):
    """cp -p, a Time Machine restore or a mv of an earlier render keeps an old mtime; the hash still differs."""
    run = new_run(tmp_path, "s", "saas-kinetic", "16:9", None, "en")
    _master(run)
    _round(run, 1)
    old = tmp_path / "old.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=3",
                    "-pix_fmt", "yuv420p", str(old)], check=True)
    os.utime(old, (1_000_000_000, 1_000_000_000))                  # 2001: older than the round
    os.replace(old, run.renders / "master.mp4")
    with pytest.raises(ShipError, match="changed after critic round 1"):
        ship(run)


def test_ship_accepts_a_round_that_named_the_master(tmp_path):
    run = new_run(tmp_path, "s", "saas-kinetic", "16:9", None, "en")
    _master(run)
    _round(run, 1, video=run.renders / "master.mp4")
    _round(run, 2, video="renders/master.mp4")   # `promo critique --video renders/master.mp4` from the run folder
    assert ship(run)["critic_round"] == 2


def test_ship_refuses_a_master_assembled_after_a_round_without_a_hash(tmp_path):
    """A round from before the hash was recorded falls back to comparing the times."""
    run = new_run(tmp_path, "s", "saas-kinetic", "16:9", None, "en")
    _master(run)
    _round(run, 1, sha=None)
    summary = run.critic / "1" / "summary.json"
    st = summary.stat()
    os.utime(summary, (st.st_atime - 60, st.st_mtime - 60))   # the master is now newer than the round
    with pytest.raises(ShipError, match="changed after critic round 1"):
        ship(run)


def test_shipping_twice_keeps_one_renders_section(tmp_path):
    run = new_run(tmp_path, "s", "saas-kinetic", "16:9", None, "en")
    _master(run)
    _round(run, 1)
    ship(run)
    ship(run)
    readme = (run.root / "README.md").read_text()
    assert readme.count("## Renders") == 1
    assert readme.count("passed critic round 1") == 1, "the same master shipped twice is one line"


def test_the_same_master_shipped_on_a_later_day_is_still_one_line(tmp_path):
    run = new_run(tmp_path, "s", "saas-kinetic", "16:9", None, "en")
    _master(run)
    _round(run, 1)
    ship(run)
    readme = run.root / "README.md"
    readme.write_text(re.sub(r"^- \d{4}-\d{2}-\d{2}:", "- 2001-09-09:", readme.read_text(), flags=re.M))
    ship(run)
    text = readme.read_text()
    assert text.count("passed critic round 1") == 1 and "- 2001-09-09:" in text
    ship(run, force=True)                    # the same master and the same outcome, so still one line
    assert readme.read_text().count("sha256 ") == 1


def test_a_new_master_adds_a_line_under_the_same_section(tmp_path):
    run = new_run(tmp_path, "s", "saas-kinetic", "16:9", None, "en")
    _master(run)
    _round(run, 1)
    ship(run)
    readme = run.root / "README.md"
    readme.write_text(readme.read_text() + "\n## Notes\n\nKept by hand.\n")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=3",
                    "-pix_fmt", "yuv420p", str(run.renders / "master.mp4")], check=True)
    _age_master(run)
    _round(run, 2)
    ship(run)
    text = readme.read_text()
    assert text.count("## Renders") == 1
    section = text.split("## Renders", 1)[1].split("## Notes", 1)[0]
    assert "critic round 1" in section and "critic round 2" in section
    assert text.rstrip().endswith("Kept by hand."), "a section written after Renders stays where it was"


def _cli_run(tmp_path, monkeypatch):
    run = new_run(tmp_path, "s", "saas-kinetic", "16:9", None, "en")
    monkeypatch.chdir(run.root)
    return run


def test_cli_ship_exit_codes(tmp_path, monkeypatch):
    run = _cli_run(tmp_path, monkeypatch)
    res = runner.invoke(app, ["ship"])
    assert res.exit_code == 2 and "promo assemble" in res.output
    _master(run)
    res = runner.invoke(app, ["ship"])
    assert res.exit_code == 1 and "no critic round" in res.output and "--force" in res.output
    _round(run, 1)
    res = runner.invoke(app, ["ship"])
    assert res.exit_code == 0, res.output
    assert "master.mp4" in res.output and "poster.png" in res.output


def test_cli_force_warns(tmp_path, monkeypatch):
    run = _cli_run(tmp_path, monkeypatch)
    _master(run)
    res = runner.invoke(app, ["ship", "--force"])
    assert res.exit_code == 0, res.output
    assert "without a passing critic round" in res.output


def test_cli_reveal_opens_the_master_in_finder(tmp_path, monkeypatch):
    run = _cli_run(tmp_path, monkeypatch)
    _master(run)
    _round(run, 1)
    bin_dir, log = tmp_path / "fake-bin", tmp_path / "open.log"
    bin_dir.mkdir()
    (bin_dir / "open").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{log}"\n')
    (bin_dir / "open").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    res = runner.invoke(app, ["ship", "--reveal"])
    assert res.exit_code == 0, res.output
    assert log.read_text().splitlines() == ["-R", str(run.renders / "master.mp4")]
