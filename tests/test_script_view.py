import json
from pathlib import Path

from promo.packs import Limits
from promo.script_view import render_script
from promo.shotlist import Shotlist
from promo.timing import Measured, resolve

FIX = Path(__file__).parent / "fixtures" / "shotlist_music.json"


def test_script_view_has_sections_and_times():
    sl = Shotlist.model_validate(json.loads(FIX.read_text()))
    t = resolve(sl, Measured(), Limits())
    md = render_script(sl, t)
    assert md.startswith("# Script")
    for heading in ("## Voiceover", "## On screen", "## Transition map", "## Sound plan"):
        assert heading in md
    assert "| c01-l1 | 0.80 |" in md
    assert "the check badge becomes the dashboard dot" in md


def test_script_view_without_timing():
    sl = Shotlist.model_validate(json.loads(FIX.read_text()))
    md = render_script(sl, None)
    assert "| c01-l1 | – |" in md
