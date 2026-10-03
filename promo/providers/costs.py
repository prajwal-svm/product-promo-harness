"""Per-run cost log (costs.jsonl)."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path


def log_cost(path: Path | None, *, provider: str, model: str, units: dict, usd: float, cached: bool) -> None:
    if path is None:
        return
    row = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "provider": provider, "model": model, "units": units,
        "usd": round(usd, 6), "cached": cached,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def total_usd(path: Path) -> float:
    if not Path(path).exists():
        return 0.0
    total = 0.0
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if not row.get("cached"):
            total += row.get("usd", 0.0)
    return total
