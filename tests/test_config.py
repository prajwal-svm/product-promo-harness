import pytest

from promo import config


def test_harness_root_is_repo(repo_root):
    assert config.HARNESS_ROOT == repo_root
    assert (config.HARNESS_ROOT / "pyproject.toml").exists()


def test_require_key_reads_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DEMO_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("DEMO_KEY=abc123\n")
    config.load_env(env)
    assert config.require_key("DEMO_KEY") == "abc123"


def test_missing_key_message_points_at_env(monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    with pytest.raises(config.MissingKeyError) as err:
        config.require_key("NOPE_KEY")
    msg = str(err.value)
    assert "NOPE_KEY is missing" in msg
    assert ".env.example" in msg


def test_env_file_is_gitignored(repo_root):
    lines = (repo_root / ".gitignore").read_text().splitlines()
    assert ".env" in lines


def test_cache_dir_respects_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMO_CACHE_DIR", str(tmp_path / "c"))
    d = config.cache_dir("gemini")
    assert d == tmp_path / "c" / "gemini"
    assert d.is_dir()
