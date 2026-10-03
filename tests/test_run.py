import json

import pytest

from promo.run import RunNotFound, find_run, new_run
from promo.shotlist import load_shotlist


def test_new_run_creates_valid_skeleton(tmp_path):
    run = new_run(tmp_path, "launch-v1", "saas-kinetic", "9:16", 45, "en")
    assert run.root == tmp_path / "promo" / "launch-v1"
    sl = load_shotlist(run.shotlist)
    assert sl.canvas.width == 1080 and sl.canvas.height == 1920 and sl.canvas.fps == 60
    assert sl.timing_mode == "music" and sl.music.bpm == 92
    assert sl.voice.name == "Kore"
    assert (run.root / ".gitignore").read_text().startswith("renders/")
    assert "45 s" in run.brief.read_text()
    for d in (run.vo_dir, run.music_dir, run.sfx_dir, run.shots_dir, run.critic):
        assert d.is_dir()


def test_new_run_refuses_existing(tmp_path):
    new_run(tmp_path, "a", "saas-kinetic", "16:9", None, "en")
    with pytest.raises(FileExistsError):
        new_run(tmp_path, "a", "saas-kinetic", "16:9", None, "en")


def test_new_run_rejects_bad_slug_and_aspect(tmp_path):
    with pytest.raises(ValueError, match="slug"):
        new_run(tmp_path, "Bad Slug", "saas-kinetic", "16:9", None, "en")
    with pytest.raises(ValueError, match="aspect"):
        new_run(tmp_path, "ok", "saas-kinetic", "4:3", None, "en")


def test_find_run_walks_up(tmp_path):
    run = new_run(tmp_path, "x", "saas-kinetic", "16:9", None, "en")
    deep = run.root / "scenes" / "shots"
    assert find_run(deep).root == run.root


def test_find_run_outside_a_run(tmp_path):
    with pytest.raises(RunNotFound, match="promo new"):
        find_run(tmp_path)
