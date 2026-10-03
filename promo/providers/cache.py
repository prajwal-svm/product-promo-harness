"""Content-hash cache for paid results, so a rerun never pays twice."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from promo.config import cache_dir
from promo.proc import atomic_path


def cache_path(provider: str, key: str) -> Path:
    return cache_dir(provider) / key[:2] / key


def cache_get(provider: str, key: str) -> bytes | None:
    p = cache_path(provider, key)
    return p.read_bytes() if p.is_file() else None


def cache_put(provider: str, key: str, data: bytes) -> Path:
    p = cache_path(provider, key)
    with atomic_path(p) as tmp:
        tmp.write_bytes(data)
    return p


def cached_call(provider: str, key: str, fn: Callable[[], bytes]) -> tuple[bytes, bool]:
    hit = cache_get(provider, key)
    if hit is not None:
        return hit, True
    data = fn()
    cache_put(provider, key, data)
    return data, False
