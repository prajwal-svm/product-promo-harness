"""Design-system packs: load, merge with core defaults, list."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from pydantic import BaseModel

from promo.config import packs_dir

SECTIONS = ("tokens", "structure", "voice", "music", "sfx", "captions")

BANNED_IDS = (
    "centred-title-on-gradient",
    "everything-fades-in",
    "corner-labels",
    "frame-borders",
    "ui-glow",
    "particle-burst",
    "shockwave-ring",
    "rgb-split",
    "unmotivated-shake",
    "lens-flare",
    "neon-glow",
    "grid-floor",
    "bouncy-type",
    "idle-sine-loop",
    "black-to-accent-fade",
    "dead-beat",
    "css-outline",
)


class PackError(ValueError):
    pass


class Limits(BaseModel):
    min_read_s: float = 0.8
    max_dead_s: float = 4.0
    hook_by_s: float = 2.0
    head_pad_s: float = 0.3
    tail_pad_s: float = 0.4
    line_gap_s: float = 0.3
    text_lag_s: float = 0.08
    pace_wpm: float = 170.0


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for key, value in b.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Pack(BaseModel):
    name: str
    root: Path
    tokens: dict
    structure: dict
    voice: dict
    music: dict
    sfx: dict
    captions: dict

    def limits(self) -> Limits:
        values = dict(self.tokens.get("limits", {}))
        if "payoff_interval_s" in self.structure:
            values["max_dead_s"] = self.structure["payoff_interval_s"]
        if "pace_wpm" in self.voice:
            values["pace_wpm"] = self.voice["pace_wpm"]
        return Limits(**values)

    def register(self) -> str:
        return self.voice.get("register", "ad-sparse")

    def text_mode(self) -> str:
        return self.structure.get("text_mode", "paraphrase")

    def timing_mode(self) -> str:
        return self.structure.get("timing_mode", "music")

    def allowed(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for item in self.structure.get("allow", []):
            if item.get("id") not in BANNED_IDS:
                raise PackError(f"pack {self.name}: allow entry '{item.get('id')}' is not a banned id")
            if not item.get("reason"):
                raise PackError(f"pack {self.name}: allow entry '{item['id']}' needs a reason")
            out[item["id"]] = item["reason"]
        return out

    def caption_mode(self, aspect: str) -> str:
        return self.captions.get("mode", {}).get(aspect, "none")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise PackError(f"{path}: invalid JSON ({e})") from e


def load_pack(name: str) -> Pack:
    root = packs_dir() / name
    if name.startswith("_") or not root.is_dir():
        raise PackError(f"no pack named '{name}' in {packs_dir()}")
    core = _read_json(packs_dir() / "_core" / "defaults.json")
    sections = {}
    for section in SECTIONS:
        merged = core.get(section, {})
        f = root / f"{section}.json"
        if f.exists():
            merged = deep_merge(merged, _read_json(f))
        sections[section] = merged
    return Pack(name=name, root=root, **sections)


def list_packs() -> list[Pack]:
    return [load_pack(p.name) for p in sorted(packs_dir().iterdir()) if p.is_dir() and not p.name.startswith("_")]
