"""Harness root, .env loading, API keys and the cache directory."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

HARNESS_ROOT = Path(__file__).resolve().parents[1]


class MissingKeyError(RuntimeError):
    def __init__(self, name: str):
        super().__init__(
            f"promo: {name} is missing. Add it to {HARNESS_ROOT / '.env'} (see .env.example)."
        )
        self.name = name


def load_env(path: Path | None = None) -> None:
    """Load the harness .env without overriding variables already set."""
    env_path = path or HARNESS_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def require_key(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        load_env()
        value = os.environ.get(name, "").strip()
    if not value:
        raise MissingKeyError(name)
    return value


def cache_dir(provider: str) -> Path:
    base = Path(os.environ.get("PROMO_CACHE_DIR", str(Path.home() / ".cache" / "promo")))
    d = base / provider
    d.mkdir(parents=True, exist_ok=True)
    return d


def packs_dir() -> Path:
    return HARNESS_ROOT / "packs"


def templates_dir() -> Path:
    return HARNESS_ROOT / "templates"
