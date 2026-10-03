import re

import pytest

from promo import packs
from promo.config import packs_dir


def test_load_saas_kinetic_merges_core_defaults():
    p = packs.load_pack("saas-kinetic")
    assert p.tokens["colors"]["accent"] == "#C6F135"          # from the pack
    assert p.tokens["fonts"]["display"]["family"] == "PromoDisplay"  # from core
    assert p.music["duck_db"] == 7                              # core default kept
    assert p.music["bed_lufs"] == -22 and p.music["fade_out_s"] == 1.0
    assert "duck" not in p.music                                # the depth in dB is the only duck setting
    assert p.music["bpm"] == 92
    assert p.register() == "ad-sparse"
    assert p.text_mode() == "paraphrase"
    assert p.timing_mode() == "music"


def test_limits_take_payoff_interval_and_pace():
    lim = packs.load_pack("saas-kinetic").limits()
    assert lim.max_dead_s == 4.0
    assert lim.pace_wpm == 190
    assert lim.text_lag_s == 0.08


def test_allowed_items_have_reasons():
    allowed = packs.load_pack("saas-kinetic").allowed()
    assert set(allowed) == {"idle-sine-loop", "shockwave-ring"}
    assert all(reason for reason in allowed.values())


def test_allow_must_reference_a_banned_id(tmp_path, monkeypatch):
    fake = tmp_path / "packs"
    (fake / "_core").mkdir(parents=True)
    (fake / "_core" / "defaults.json").write_text((packs_dir() / "_core" / "defaults.json").read_text())
    (fake / "bad").mkdir()
    (fake / "bad" / "structure.json").write_text('{"allow": [{"id": "not-a-rule", "reason": "x"}]}')
    monkeypatch.setattr(packs, "packs_dir", lambda: fake)
    with pytest.raises(packs.PackError, match="not-a-rule"):
        packs.load_pack("bad").allowed()


def test_list_packs_skips_core():
    names = [p.name for p in packs.list_packs()]
    assert "saas-kinetic" in names
    assert "_core" not in names


def test_unknown_pack():
    with pytest.raises(packs.PackError, match="no pack named 'nope'"):
        packs.load_pack("nope")


def test_banned_ids_match_banned_md():
    text = (packs_dir() / "_core" / "banned.md").read_text()
    ids = [i for i in re.findall(r"^\| ([a-z0-9-]+) \|", text, flags=re.M) if i != "id"]
    assert tuple(ids) == packs.BANNED_IDS


def test_caption_mode_per_aspect():
    p = packs.load_pack("saas-kinetic")
    assert p.caption_mode("16:9") == "none"
    assert p.caption_mode("9:16") == "subtitle"


def test_deep_merge_does_not_mutate():
    a = {"x": {"y": 1, "z": 2}}
    b = {"x": {"y": 3}}
    out = packs.deep_merge(a, b)
    assert out == {"x": {"y": 3, "z": 2}}
    assert a == {"x": {"y": 1, "z": 2}}


BLUR = re.compile(r"filter\s*:\s*[\"']?\s*blur|backdrop-?filter\s*:|backdropFilter|blur\(", re.I)


def test_nothing_in_the_packs_or_the_scene_template_blurs():
    """The user's rule is absolute: no blur of anything. Every pack file (the scene template, the caption overlay,
    pack data and docs) is free of blur filters, backdrop filters and blur() in tweens. Vendored libraries are not
    ours to edit and are skipped."""
    hits = []
    for f in packs_dir().rglob("*"):
        if f.is_file() and "vendor" not in f.parts and f.suffix in (".html", ".css", ".js", ".json", ".md"):
            for n, line in enumerate(f.read_text().splitlines(), 1):
                if BLUR.search(line):
                    hits.append(f"{f.relative_to(packs_dir())}:{n}: {line.strip()}")
    assert not hits, "\n".join(hits)


def test_no_pack_promises_depth_of_field():
    hits = [f"{f.relative_to(packs_dir())}" for d in packs_dir().iterdir() if d.is_dir() and d.name != "_core"
            for f in d.iterdir() if f.is_file() and re.search(r"depth[ -]of[ -]field", f.read_text(), re.I)]
    assert not hits, hits
