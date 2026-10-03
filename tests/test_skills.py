import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import typer.main
import yaml

from promo.cli import app
from promo.config import HARNESS_ROOT

SKILLS = ["promo-video", "promo-script", "promo-voice", "promo-captions", "promo-critic"]
INSTALL = HARNESS_ROOT / "scripts" / "install.sh"


def _skill(name: str) -> str:
    return (HARNESS_ROOT / "skills" / name / "SKILL.md").read_text()


@pytest.mark.parametrize("name", SKILLS)
def test_skill_frontmatter(name):
    """Strict YAML (PyYAML, skill validators), not only what Claude Code's forgiving parser accepts."""
    text = _skill(name)
    m = re.match(r"^---\n(name: .+\ndescription: .+\n)---\n", text)
    assert m, "frontmatter must be name + description"
    front = yaml.safe_load(m.group(1))
    assert set(front) == {"name", "description"}
    assert front["name"] == name
    assert 80 < len(front["description"]) < 1024


def test_entry_skill_names_the_harness_and_every_gate_command():
    text = _skill("promo-video")
    assert "<harness>/scripts/install.sh" in text and "readlink ~/.claude/skills/promo-video" in text
    for cmd in ("promo new", "promo lint-script", "promo timing", "promo voice audition", "promo voice --all",
                "promo music", "promo beats", "promo sfx render", "promo guide", "promo scenes --stills",
                "promo scenes --draft", "promo assemble --draft", "promo scenes --sample", "promo verify",
                "promo captions", "promo assemble", "--allow-stale", "promo critique", "promo ship --reveal",
                "promo doctor", "promo ref", "promo script"):
        assert cmd in text, cmd
    assert "cut, make or produce a promo video" in text


def test_entry_skill_runs_the_pipeline_in_order():
    text = _skill("promo-video")
    order = ["promo new", "promo lint-script", "promo voice --all", "promo music", "promo beats", "promo sfx render",
             "promo scenes --stills", "promo captions", "promo assemble`", "promo verify`", "promo critique`",
             "promo ship --reveal"]
    gates = text[text.index("## Gates"):]
    places = [gates.find(cmd) for cmd in order]
    assert -1 not in places, dict(zip(order, places))
    assert places == sorted(places), dict(zip(order, places))


HARD_RULES = ["22 px", "no blur, anywhere", "depth of field", "motion blur", "vignette", "scrim", "pointer", "0.3-0.5 s", "never invent", "2-4 s", "sparse",
              "hero scale", "vendored", "humanizer", "em dash", "not X but Y",
              "hyperframes-agent-render-gotchas", "ffmpeg-filtergraph-mix-master-gotchas"]


@pytest.mark.parametrize("name", ["promo-video", "promo-critic"])
def test_quality_rules_are_hard_rules(name):
    text = _skill(name).lower()
    for rule in HARD_RULES:
        assert rule.lower() in text, rule


def test_voice_skill_teaches_both_providers():
    text = _skill("promo-voice")
    for needle in ("supertonic", "Supertonic", "F2", "no key", "gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts",
                   "GEMINI_API_KEY", '"pronounce"', '"Kubernetes": "koo-ber-NET-eez"', '"asr_aliases"', '"verify": false',
                   "promo voice audition", "promo voice --all", "--force"):
        assert needle in text, needle
    description = yaml.safe_load(text.split("---\n")[1])["description"]
    assert "Supertonic" in description and "Gemini" in description


@pytest.mark.parametrize("name", SKILLS)
def test_skills_have_no_em_dashes(name):
    assert "—" not in _skill(name)


def _cli_tree():
    """{command: click command} for every `promo` command and `promo <group> <sub>` subcommand."""
    root = typer.main.get_command(app)
    tree = {}
    for name, cmd in root.commands.items():
        tree[name] = cmd
        for sub, subcmd in getattr(cmd, "commands", {}).items():
            tree[f"{name} {sub}"] = subcmd
    return tree


def _options(cmd) -> set[str]:
    return {opt for p in cmd.params for opt in (*p.opts, *p.secondary_opts) if opt.startswith("--")} | {"--help"}


def _promo_commands(text: str) -> list[str]:
    """Every `promo ...` command the text shows: in inline code spans, and on the lines of fenced code blocks (a line
    can chain several with && or ;, and end in a # comment)."""
    fenced = re.findall(r"^```[^\n]*\n(.*?)^```", text, flags=re.M | re.S)
    inline = re.sub(r"^```[^\n]*\n.*?^```", "", text, flags=re.M | re.S)
    out = re.findall(r"`(promo [^`]+)`", inline)
    for block in fenced:
        for line in block.splitlines():
            line = re.sub(r"\s+#.*$", "", line)
            out += [part.strip() for part in re.split(r"&&|;|\|\|", line) if part.strip().startswith("promo ")]
    return out


def test_promo_commands_are_found_in_fences_too():
    text = "Run `promo timing`.\n\n```bash\npromo new x --pack p   # scaffold\npromo voice --all && promo ship\n```\n"
    assert _promo_commands(text) == ["promo timing", "promo new x --pack p", "promo voice --all", "promo ship"]


@pytest.mark.parametrize("name", SKILLS + ["README"])
def test_every_promo_command_in_the_docs_exists(name):
    """Every `promo ...` in a code span or a fenced block names a real command and only its real options."""
    text = (HARNESS_ROOT / "README.md").read_text() if name == "README" else _skill(name)
    tree = _cli_tree()
    spans = _promo_commands(text)
    assert spans, "no promo commands found"
    for span in spans:
        words = span.split()
        if words[1] == "--help":
            continue
        if len(words) > 2 and f"{words[1]} {words[2]}" in tree:
            key = f"{words[1]} {words[2]}"
        else:
            key = words[1]
        assert key in tree, f"{span}: no command `promo {key}`"
        for opt in re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", span):
            assert opt in _options(tree[key]), f"{span}: `promo {key}` has no {opt}"


def test_install_script_is_idempotent():
    text = INSTALL.read_text()
    assert "uv tool install --editable" in text and "ln -sfn" in text
    assert "launch-video" not in text, "other skills are never touched"
    assert "set -euo pipefail" in text
    assert os.access(INSTALL, os.X_OK), "install.sh must be executable"


# install.sh in a sandbox: a copy of the harness under a path with a space, a fake `uv` that only logs, and a
# temporary HOME standing in for the user's. Nothing outside tmp_path is touched.

FAKE_UV = """#!/bin/sh
for a in "$@"; do printf '%s\\t' "$a"; done >> "$FAKE_UV_LOG"
echo >> "$FAKE_UV_LOG"
"""


@pytest.fixture
def sandbox(tmp_path):
    harness = tmp_path / "my harness"
    (harness / "scripts").mkdir(parents=True)
    shutil.copy2(INSTALL, harness / "scripts" / "install.sh")
    shutil.copytree(HARNESS_ROOT / "skills", harness / "skills")
    shutil.copy2(HARNESS_ROOT / ".env.example", harness / ".env.example")
    home = tmp_path / "home"
    old = home / ".agents" / "skills" / "launch-video"
    old.mkdir(parents=True)
    (old / "SKILL.md").write_text("---\nname: launch-video\n---\n")
    skills = home / ".claude" / "skills"
    skills.mkdir(parents=True)
    (skills / "launch-video").symlink_to("../../.agents/skills/launch-video")   # relative, as on the real Mac
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(FAKE_UV)
    (bin_dir / "uv").chmod(0o755)
    return {"root": tmp_path, "harness": harness, "home": home, "skills": skills, "old": old,
            "bin": bin_dir, "log": tmp_path / "uv.log"}


def _install(sb, **env):
    bash = "/bin/bash" if Path("/bin/bash").exists() else "bash"   # macOS ships bash 3.2: the script must run on it
    full = {"PATH": f"{sb['bin']}{os.pathsep}/usr/bin:/bin", "HOME": str(sb["home"]), "FAKE_UV_LOG": str(sb["log"]),
            **env}
    return subprocess.run([bash, str(sb["harness"] / "scripts" / "install.sh")], env=full, capture_output=True,
                          text=True, cwd=sb["root"])


def _uv_calls(sb):
    return [line.rstrip("\t").split("\t") for line in sb["log"].read_text().splitlines() if line]


def test_install_links_every_skill_and_leaves_other_skills_alone(sandbox):
    res = _install(sandbox)
    assert res.returncode == 0, res.stdout + res.stderr
    for name in SKILLS:
        link = sandbox["skills"] / name
        assert link.is_symlink(), name
        assert Path(os.readlink(link)) == sandbox["harness"] / "skills" / name
        assert (link / "SKILL.md").read_text() == _skill(name)
        assert f"linked skill {name}" in res.stdout
    assert (sandbox["skills"] / "launch-video").is_symlink(), "another skill's link is left alone"
    assert (sandbox["old"] / "SKILL.md").exists()
    harness = str(sandbox["harness"])
    assert ["sync", "--extra", "dev"] in _uv_calls(sandbox)
    assert ["tool", "install", "--editable", harness, "--python", "3.12", "--force"] in _uv_calls(sandbox)
    env_file = sandbox["harness"] / ".env"
    assert env_file.read_text() == (sandbox["harness"] / ".env.example").read_text()
    assert env_file.stat().st_mode & 0o777 == 0o600, ".env will hold keys: readable by its owner only"
    assert "promo doctor" in res.stdout


def test_install_twice_changes_nothing_the_second_time(sandbox):
    assert _install(sandbox).returncode == 0
    (sandbox["harness"] / ".env").write_text("GEMINI_API_KEY=kept\n")
    stale = sandbox["skills"] / "promo-voice"
    stale.unlink()
    stale.symlink_to(sandbox["root"])      # an old link to somewhere else is replaced
    before = sorted(p.name for p in sandbox["skills"].iterdir())
    res = _install(sandbox)
    assert res.returncode == 0, res.stdout + res.stderr
    assert sorted(p.name for p in sandbox["skills"].iterdir()) == before
    assert Path(os.readlink(stale)) == sandbox["harness"] / "skills" / "promo-voice"
    assert (sandbox["harness"] / ".env").read_text() == "GEMINI_API_KEY=kept\n", ".env is never overwritten"
    assert "launch-video" not in res.stdout
    for name in SKILLS:     # ln without -n would follow the old link and put a new one inside its target
        assert not (sandbox["harness"] / "skills" / name / name).exists(), name
    assert not (sandbox["root"] / "promo-voice").exists()


def test_install_honours_promo_skills_dir(sandbox):
    custom = sandbox["root"] / "elsewhere" / "skills"
    res = _install(sandbox, PROMO_SKILLS_DIR=str(custom))
    assert res.returncode == 0, res.stdout + res.stderr
    assert sorted(p.name for p in custom.iterdir()) == sorted(SKILLS)
    assert (sandbox["skills"] / "launch-video").is_symlink(), "the HOME skills folder is not touched"
    assert not (sandbox["skills"] / "promo-video").exists()


def test_install_ignores_a_generic_skills_dir_another_tool_exports(sandbox):
    other = sandbox["root"] / "other-tool" / "skills"
    res = _install(sandbox, SKILLS_DIR=str(other))
    assert res.returncode == 0, res.stdout + res.stderr
    assert not other.exists()
    assert (sandbox["skills"] / "promo-video").is_symlink() and (sandbox["skills"] / "launch-video").is_symlink()


def test_install_never_replaces_a_real_folder(sandbox):
    real = sandbox["skills"] / "promo-critic"
    real.mkdir()
    (real / "SKILL.md").write_text("mine")
    launch = sandbox["skills"] / "launch-video"
    launch.unlink()
    launch.mkdir()                          # a real launch-video folder, not a link: leave it alone
    res = _install(sandbox)
    assert res.returncode == 1
    assert "promo-critic" in res.stderr and "not a link" in res.stderr
    assert (real / "SKILL.md").read_text() == "mine" and not (real / "promo-critic").exists()
    assert launch.is_dir() and not launch.is_symlink()
    for name in set(SKILLS) - {"promo-critic"}:
        assert (sandbox["skills"] / name).is_symlink(), "the other skills are still linked"


def test_install_reports_a_failed_uv_step(sandbox):
    (sandbox["bin"] / "uv").write_text("#!/bin/sh\necho 'uv: no network' >&2\nexit 3\n")
    res = _install(sandbox)
    assert res.returncode != 0
    assert not (sandbox["skills"] / "promo-video").exists(), "set -e stops before linking half an install"
