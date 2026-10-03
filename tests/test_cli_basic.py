import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from promo.cli import app

FIX = Path(__file__).parent / "fixtures" / "shotlist_music.json"
runner = CliRunner()


def _run_with_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(app, ["new", "demo"])
    assert res.exit_code == 0, res.output
    run = tmp_path / "promo" / "demo"
    shutil.copy(FIX, run / "shotlist.json")
    monkeypatch.chdir(run)
    return run


def test_packs_lists_saas_kinetic():
    res = runner.invoke(app, ["packs"])
    assert res.exit_code == 0
    assert "saas-kinetic" in res.output and "ad-sparse" in res.output


def test_timing_writes_file_and_reports(tmp_path, monkeypatch):
    run = _run_with_fixture(tmp_path, monkeypatch)
    res = runner.invoke(app, ["timing"])
    assert res.exit_code == 0, res.output
    data = json.loads((run / "timing.json").read_text())
    assert data["duration_s"] == 8.3
    assert "0 errors" in res.output and "no_take" in res.output


def test_timing_fails_on_errors(tmp_path, monkeypatch):
    run = _run_with_fixture(tmp_path, monkeypatch)
    data = json.loads((run / "shotlist.json").read_text())
    data["chapters"][0]["shots"][0]["events"][1]["at"]["word"] = "banana"
    (run / "shotlist.json").write_text(json.dumps(data))
    res = runner.invoke(app, ["timing"])
    assert res.exit_code == 1
    assert "word_not_found" in res.output


def test_lint_script_fails_on_unsourced_number(tmp_path, monkeypatch):
    run = _run_with_fixture(tmp_path, monkeypatch)
    data = json.loads((run / "shotlist.json").read_text())
    data["chapters"][1]["vo"][0]["claims"] = []
    (run / "shotlist.json").write_text(json.dumps(data))
    res = runner.invoke(app, ["lint-script"])
    assert res.exit_code == 1
    assert "unsourced_number" in res.output


def test_lint_script_text_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "copy.md"
    f.write_text("Imagine a seamless world.")
    res = runner.invoke(app, ["lint-script", "--text", str(f), "--register", "professor"])
    assert res.exit_code == 1
    assert "rhetorical_opener" in res.output and "hype_word" in res.output


def test_script_writes_markdown(tmp_path, monkeypatch):
    run = _run_with_fixture(tmp_path, monkeypatch)
    runner.invoke(app, ["timing"])
    res = runner.invoke(app, ["script"])
    assert res.exit_code == 0, res.output
    assert "## Transition map" in (run / "script.md").read_text()


def test_commands_outside_a_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(app, ["timing"])
    assert res.exit_code == 2
    assert "promo new" in res.output


def test_fresh_scaffold_passes_its_own_timing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["new", "demo"]).exit_code == 0
    monkeypatch.chdir(tmp_path / "promo" / "demo")
    res = runner.invoke(app, ["timing"])
    assert res.exit_code == 0, res.output
    assert "0 errors" in res.output


def test_timing_unknown_pack_is_a_clean_error(tmp_path, monkeypatch):
    run = _run_with_fixture(tmp_path, monkeypatch)
    data = json.loads((run / "shotlist.json").read_text())
    data["pack"] = "no-such-pack"
    (run / "shotlist.json").write_text(json.dumps(data))
    res = runner.invoke(app, ["timing"])
    assert res.exit_code == 1
    assert "promo:" in res.output and "no-such-pack" in res.output
    assert "Traceback" not in res.output


def test_lint_script_and_script_reject_invalid_json(tmp_path, monkeypatch):
    run = _run_with_fixture(tmp_path, monkeypatch)
    (run / "shotlist.json").write_text("{not json")
    for cmd in ("lint-script", "script"):
        res = runner.invoke(app, [cmd])
        assert res.exit_code == 1, (cmd, res.output)
        assert "promo:" in res.output and "Traceback" not in res.output


def test_lint_script_and_script_reject_invalid_schema(tmp_path, monkeypatch):
    run = _run_with_fixture(tmp_path, monkeypatch)
    (run / "shotlist.json").write_text(json.dumps({"version": 2}))
    for cmd in ("lint-script", "script"):
        res = runner.invoke(app, [cmd])
        assert res.exit_code == 1, (cmd, res.output)
        assert "promo:" in res.output and "Traceback" not in res.output


def test_lint_script_text_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(app, ["lint-script", "--text", str(tmp_path / "missing.md")])
    assert res.exit_code == 2
    assert "promo:" in res.output and "Traceback" not in res.output


def test_lint_script_and_script_outside_a_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for cmd in ("lint-script", "script"):
        res = runner.invoke(app, [cmd])
        assert res.exit_code == 2, (cmd, res.output)
        assert "promo new" in res.output


def test_script_leaves_no_temp_file(tmp_path, monkeypatch):
    run = _run_with_fixture(tmp_path, monkeypatch)
    assert runner.invoke(app, ["script"]).exit_code == 0
    assert (run / "script.md").is_file()
    assert not list(run.glob(".script.tmp*"))


def test_lint_script_warns_about_verify_false_and_still_exits_0(tmp_path, monkeypatch):
    run = _run_with_fixture(tmp_path, monkeypatch)
    data = json.loads((run / "shotlist.json").read_text())
    data["chapters"][0]["vo"][0]["verify"] = False
    (run / "shotlist.json").write_text(json.dumps(data))
    res = runner.invoke(app, ["lint-script"])
    assert res.exit_code == 0, res.output
    assert "warning" in res.output and "verify_skipped" in res.output and "c01-l1: verify is false" in res.output
    assert "\n0 errors, " in res.output
