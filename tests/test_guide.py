import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from promo.cli import app
from promo.config import packs_dir
from promo.guide import write_guide
from promo.packs import BANNED_IDS, load_pack
from promo.run import new_run

FIX = Path(__file__).parent / "fixtures" / "shotlist_music.json"
runner = CliRunner()


def test_guide_pastes_doctrine_pack_and_rules(tmp_path):
    run = new_run(tmp_path, "d", "saas-kinetic", "16:9", None, "en")
    path = write_guide(run, load_pack("saas-kinetic"))
    text = path.read_text()
    assert path == run.guide
    assert "# Motion doctrine (core)" in text                    # verbatim doctrine
    assert "## Render contract" in text
    assert "Lime 3D props, sharp, layered by scale and parallax" in text    # re-allowed item with its reason
    assert "idle-sine-loop" not in text.split("## Banned for this run")[1].split("## Re-allowed")[0]
    assert "# saas-kinetic" in text and "#C6F135" in text         # pack.md and tokens
    assert "Edit only your own shot files" in text
    assert "window.__timelines" in text and "PromoStage.load" in text


def test_guide_asks_for_no_blur_and_no_vignette(tmp_path):
    """The user's rules: no blur of anything and a clean frame. The guide every scene subagent gets must not ask for
    either (it once budgeted blurred elements and called the finishing layer a vignette)."""
    text = write_guide(new_run(tmp_path, "d", "saas-kinetic", "16:9", None, "en"), load_pack("saas-kinetic")).read_text()
    assert "No blur of anything" in text and "whips without" in text
    assert "blurred elements" not in text and "peak blur" not in text and "Heavy blur" not in text
    assert "nobody removes it" not in text and "grain + vignette" not in text
    assert "blur-in" not in text, "the starter shot no longer blurs, so there is nothing to replace"
    assert "no vignette, inner glow, feathered edge or grey scrim" in text


def _section(text, start, end):
    return text.split(start)[1].split(end)[0]


def test_guide_lists_every_banned_item_the_pack_did_not_re_allow(tmp_path):
    pack = load_pack("saas-kinetic")
    text = write_guide(new_run(tmp_path, "d", "saas-kinetic", "16:9", None, "en"), pack).read_text()
    banned = _section(text, "## Banned for this run", "## Re-allowed")
    for item in BANNED_IDS:
        assert (f"`{item}`" in banned) == (item not in pack.allowed()), item
    assert "- `centred-title-on-gradient`: A centred title on a gradient background" in banned    # banned.md's own wording
    reallowed = text.split("## Re-allowed by this pack")[1].split("# saas-kinetic")[0]
    assert "- `shockwave-ring`: One thin lime ring wipe between beats" in reallowed


def test_guide_pastes_the_sources_verbatim(tmp_path):
    pack = load_pack("saas-kinetic")
    text = write_guide(new_run(tmp_path, "d", "saas-kinetic", "16:9", None, "en"), pack).read_text()
    assert (packs_dir() / "_core" / "doctrine.md").read_text().strip() in text
    assert (pack.root / "pack.md").read_text().strip() in text
    assert json.dumps({"pack": "saas-kinetic", "tokens": pack.tokens}, indent=2) in text
    for heading in ("## Rules for scene work", "## HyperFrames gotchas", "## Helpers available in every shot",
                    "## Design tokens (design.json)"):
        assert heading in text
    assert "```json\n{" in text and "PromoStage.finish(tl, S)" in text


def test_guide_uses_the_runs_design_json_when_there_is_one(tmp_path):
    run = new_run(tmp_path, "d", "saas-kinetic", "16:9", None, "en")
    run.design.write_text(json.dumps({"pack": "saas-kinetic", "tokens": {"colors": {"accent": "#123456"}}}))
    text = write_guide(run, load_pack("saas-kinetic")).read_text()
    assert "#123456" in text and "#C6F135" not in _section(text, "## Design tokens", "## HyperFrames gotchas")


def test_guide_with_nothing_re_allowed(tmp_path):
    base = load_pack("saas-kinetic")
    pack = base.model_copy(update={"structure": {**base.structure, "allow": []}})
    text = write_guide(new_run(tmp_path, "d", "saas-kinetic", "16:9", None, "en"), pack).read_text()
    assert "## Re-allowed by this pack\n\n- (none)\n" in text and "`idle-sine-loop`" in _section(
        text, "## Banned for this run", "## Re-allowed")


def test_guide_is_rewritten_in_place(tmp_path):
    run = new_run(tmp_path, "d", "saas-kinetic", "16:9", None, "en")
    run.guide.write_text("stale")
    write_guide(run, load_pack("saas-kinetic"))
    assert run.guide.read_text().startswith("# ANIMATION_GUIDE for this run")
    assert not list(run.root.glob(".ANIMATION_GUIDE.tmp*"))


def _run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["new", "demo"]).exit_code == 0
    run = tmp_path / "promo" / "demo"
    shutil.copy(FIX, run / "shotlist.json")
    monkeypatch.chdir(run)
    return run


def test_cli_guide_writes_the_file_and_prints_its_path(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch)
    res = runner.invoke(app, ["guide"])
    assert res.exit_code == 0, res.output
    assert res.output.strip() == str(run.resolve() / "ANIMATION_GUIDE.md")
    assert "Edit only your own shot files" in (run / "ANIMATION_GUIDE.md").read_text()


def test_cli_guide_outside_a_run_is_exit_2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(app, ["guide"])
    assert res.exit_code == 2 and "promo new" in res.output


def test_cli_guide_with_a_bad_shotlist_or_pack_is_one_line_exit_1(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch)
    data = json.loads((run / "shotlist.json").read_text())
    (run / "shotlist.json").write_text(json.dumps({**data, "pack": "no-such-pack"}))
    res = runner.invoke(app, ["guide"])
    assert res.exit_code == 1 and "no pack named 'no-such-pack'" in res.output and "Traceback" not in res.output
    (run / "shotlist.json").write_text("{ nope")
    res = runner.invoke(app, ["guide"])
    assert res.exit_code == 1 and "not valid JSON" in res.output
    assert not (run / "ANIMATION_GUIDE.md").exists()
