import json
import math
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import pytest

from promo import captions
from promo import hyperframes as hf
from promo.models import Caption
from promo.packs import Limits, deep_merge, load_pack
from promo.proc import ToolError, write_sidecar
from promo.run import new_run
from promo.shotlist import Shotlist
from promo.timing import Measured, resolve, write_timing

FIX = Path(__file__).parent / "fixtures" / "shotlist_music.json"
OVERLAY = Path(__file__).parents[1] / "packs" / "_core" / "caption-overlay" / "index.html"


def W(text, a, b):
    return Caption(text=text, startMs=a, endMs=b)


def test_pages_break_on_sentence_pause_and_width():
    words = [W("Need", 0, 200), W("a", 200, 300), W("reversal?", 300, 700),
             W("Request", 800, 1100), W("it.", 1100, 1300),
             W("Then", 2000, 2200), W("wait", 2200, 2400)]
    ps = captions.pages(words, max_chars=32)
    assert [p.text for p in ps] == ["Need a reversal?", "Request it.", "Then wait"]
    long = [W(f"word{i}", i * 100, i * 100 + 90) for i in range(12)]
    ps = captions.pages(long, max_chars=16, max_lines=2)
    assert all(len(p.words) <= 4 for p in ps)   # "word0 word1" = 11 chars → 2 words a line, 2 lines


def test_pages_break_only_when_the_pause_is_longer_than_pause_ms():
    words = [W("one", 0, 100), W("two", 350, 450), W("three", 701, 800)]
    ps = captions.pages(words, max_chars=32, pause_ms=250)
    assert [p.text for p in ps] == ["one two", "three"]   # a 250 ms gap stays, a 251 ms gap breaks


def test_pages_of_no_words_is_empty():
    assert captions.pages([], max_chars=32) == []


def test_min_duration_never_overlaps_next_page():
    ps = captions.pages([W("Hi.", 0, 200), W("There.", 450, 800)], max_chars=32, min_ms=700)
    assert (ps[0].startMs, ps[0].endMs) == (0, 450)
    assert (ps[1].startMs, ps[1].endMs) == (450, 1150)


def test_srt_and_vtt():
    ps = captions.pages([W("Hello", 1060, 1500), W("there.", 1500, 2980)], max_chars=32)
    assert captions.to_srt(ps) == "1\n00:00:01,060 --> 00:00:02,980\nHello there.\n"
    assert captions.to_vtt(ps) == "WEBVTT\n\n00:00:01.060 --> 00:00:02.980\nHello there.\n"


def test_srt_numbers_cues_and_separates_them_with_a_blank_line():
    ps = captions.pages([W("One.", 0, 800), W("Two.", 900, 1700)], max_chars=32)
    assert captions.to_srt(ps) == ("1\n00:00:00,000 --> 00:00:00,800\nOne.\n\n"
                                   "2\n00:00:00,900 --> 00:00:01,700\nTwo.\n")


def test_timestamps_roll_over_minutes_and_hours():
    ps = [captions.Page(text="late", startMs=3_723_004, endMs=3_725_000, words=[])]
    assert captions.to_srt(ps).splitlines()[1] == "01:02:03,004 --> 01:02:05,000"


def test_vtt_escapes_markup_but_srt_keeps_the_text():
    ps = captions.pages([W("R&D", 0, 400), W("<10%", 400, 900)], max_chars=32)
    assert "R&amp;D &lt;10%" in captions.to_vtt(ps)
    assert "R&D <10%" in captions.to_srt(ps)


def test_resolve_style_maps_roles():
    style = captions.resolve_style({"color": "ink", "highlight": "accent", "size_px": 54},
                                   {"tokens": {"colors": {"ink": "#111111", "accent": "#C6F135"}}})
    assert style == {"color": "#111111", "highlight": "#C6F135", "size_px": 54}


def test_resolve_style_keeps_literals_and_survives_a_design_without_tokens():
    assert captions.resolve_style({"color": "#fff", "highlight": "accent"}, {}) == {"color": "#fff", "highlight": "accent"}


def _timing(aspect="16:9"):
    data = json.loads(FIX.read_text())
    if aspect != "16:9":
        w, h = {"9:16": (1080, 1920), "1:1": (1080, 1080)}[aspect]
        data["canvas"] = {"aspect": aspect, "width": w, "height": h, "fps": 60}
    return resolve(Shotlist.model_validate(data), Measured(), Limits())


def test_words_from_timing_is_every_word_in_time_order():
    t = _timing()
    words = captions.words_from_timing(t)
    assert [w.text for w in words][:3] == ["Moving", "company", "money"]
    assert len(words) == sum(len(line.words) for line in t.lines)
    assert [w.startMs for w in words] == sorted(w.startMs for w in words)


def test_build_writes_sidecars_and_skips_overlay_at_16x9(tmp_path):
    run = new_run(tmp_path, "d", "saas-kinetic", "16:9", None, "en")
    out = captions.build(run, _timing(), load_pack("saas-kinetic"), render_fn=lambda *a, **k: pytest.fail("no overlay"))
    assert out["srt"].read_text().startswith("1\n")
    assert json.loads(out["words"].read_text())[0]["text"] == "Moving"
    assert "overlay" not in out


def test_build_sidecars_agree_and_leave_no_temp_files(tmp_path):
    run = new_run(tmp_path, "d", "saas-kinetic", "16:9", None, "en")
    out = captions.build(run, _timing(), load_pack("saas-kinetic"), render_fn=lambda *a, **k: pytest.fail("no overlay"))
    assert out["vtt"].read_text().startswith("WEBVTT\n\n")
    words = [Caption.model_validate(w) for w in json.loads(out["words"].read_text())]
    assert words == captions.words_from_timing(_timing())
    assert sorted(p.name for p in run.captions_dir.iterdir()) == ["captions.srt", "captions.vtt", "words.json"]


def test_build_renders_alpha_overlay_at_9x16(tmp_path):
    run = new_run(tmp_path, "v", "saas-kinetic", "9:16", None, "en")
    seen = {}

    def fake_render(project, out, **kw):
        seen.update(kw, project=project)
        out.write_bytes(b"mov")
        return out

    out = captions.build(run, _timing("9:16"), load_pack("saas-kinetic"), render_fn=fake_render)
    assert out["overlay"].name == "captions.mov"
    assert out["overlay"] == run.captions_dir / "overlay" / "captions.mov"   # the path timeline.collect_media reads
    assert seen["fmt"] == "mov" and seen["fps"] == 60 and seen["height"] == 1920
    assert seen["width"] == 1080 and seen["duration"] == 8.3 and seen["quality"] == "high"
    html = (seen["project"] / "index.html").read_text()
    assert 'data-composition-id="captions"' in html and 'data-height="1920"' in html
    assert 'data-duration="8.3"' in html and 'content="width=1080, height=1920"' in html
    data = json.loads((seen["project"] / "captions.json").read_text())
    assert data["style"]["color"].startswith("#") and data["band"] == [0.70, 0.86]
    assert data["style"]["highlight"] == "#C6F135" and data["style"]["size_px"] == 54
    assert data["pages"] and data["pages"][0]["words"][0]["text"] == "Moving"
    assert (seen["project"] / "assets" / "vendor" / "gsap.min.js").exists()
    assert (seen["project"] / "assets" / "fonts" / "InterVariable.woff2").exists()


def test_build_uses_the_run_design_json_colours_when_present(tmp_path):
    run = new_run(tmp_path, "v", "saas-kinetic", "9:16", None, "en")
    run.design.write_text(json.dumps({"tokens": {"colors": {"ink": "#010203", "accent": "#0A0B0C"}}}))
    seen = {}

    def fake_render(project, out, **kw):
        seen["project"] = project
        out.write_bytes(b"mov")
        return out

    captions.build(run, _timing("9:16"), load_pack("saas-kinetic"), render_fn=fake_render)
    data = json.loads((seen["project"] / "captions.json").read_text())
    assert (data["style"]["color"], data["style"]["highlight"]) == ("#010203", "#0A0B0C")


def test_build_with_no_words_writes_empty_sidecars_and_skips_the_overlay(tmp_path):
    run = new_run(tmp_path, "v", "saas-kinetic", "9:16", None, "en")
    silent = _timing("9:16").model_copy(update={"lines": []})
    out = captions.build(run, silent, load_pack("saas-kinetic"), render_fn=lambda *a, **k: pytest.fail("nothing to caption"))
    assert out["srt"].read_text() == "" and out["vtt"].read_text() == "WEBVTT\n\n"
    assert json.loads(out["words"].read_text()) == [] and "overlay" not in out


def test_build_removes_a_stale_overlay_it_does_not_replace(tmp_path):
    run = new_run(tmp_path, "v", "saas-kinetic", "9:16", None, "en")
    overlay = run.captions_dir / "overlay" / "captions.mov"
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(b"old")
    side = overlay.with_name(overlay.name + ".render.json")
    side.write_text("{}")
    silent = _timing("9:16").model_copy(update={"lines": []})
    captions.build(run, silent, load_pack("saas-kinetic"), render_fn=lambda *a, **k: pytest.fail("nothing to caption"))
    assert not overlay.exists() and not side.exists()   # timeline.collect_media would put it on V2 otherwise


def test_overlay_template_is_unpainted_and_has_no_outline():
    html = OVERLAY.read_text()
    assert "outline" not in html and "text-stroke" not in html    # legibility is the soft shadow only
    assert "textShadow" in html and "st.shadow" in html and "st.plate" in html
    lines = [line.strip() for line in html.splitlines()]
    assert "background" not in next(line for line in lines if line.startswith("html, body"))
    assert "background" not in next(line for line in lines if line.startswith("#root"))
    assert html.rindex("window.__timelines") > html.rindex("appendChild")   # registered after all setup
    assert "return tl" not in html


# --- sentence breaks and page timing ---------------------------------------------------------------------

@pytest.mark.parametrize("closing", ['safe."', "safe.)", 'safe?"', "safe!'", "safe\u2026", "safe.\u201d", "safe?)"])
def test_a_stop_followed_by_a_closing_quote_or_bracket_ends_the_sentence(closing):
    ps = captions.pages([W("Stay", 0, 300), W(closing, 300, 800), W("Next", 850, 1200)], max_chars=32)
    assert [p.text for p in ps] == [f"Stay {closing}", "Next"]


@pytest.mark.parametrize("abbr", ["e.g.", "i.e.", "U.S.", "Fig.", "Eq.", "fig.", "a.m."])
def test_common_abbreviations_do_not_end_a_sentence(abbr):
    ps = captions.pages([W("See", 0, 200), W(abbr, 200, 500), W("three", 520, 900)], max_chars=32)
    assert [p.text for p in ps] == [f"See {abbr} three"]


def test_et_al_does_not_end_a_sentence_but_a_lone_al_does():
    ps = captions.pages([W("Smith", 0, 300), W("et", 300, 400), W("al.", 400, 700), W("found", 720, 1000)], max_chars=32)
    assert [p.text for p in ps] == ["Smith et al. found"]
    ps = captions.pages([W("Call", 0, 300), W("Al.", 300, 700), W("Then", 720, 1000)], max_chars=32)
    assert [p.text for p in ps] == ["Call Al.", "Then"]


def test_a_word_that_merely_contains_a_stop_does_not_end_a_sentence():
    ps = captions.pages([W("Version", 0, 300), W("3.5", 300, 600), W("ships", 620, 900)], max_chars=32)
    assert [p.text for p in ps] == ["Version 3.5 ships"]


def test_every_page_ends_no_later_than_the_next_page_starts():
    # ASR jitter: "one." claims to end at 900 ms but the next page starts at 600 ms.
    ps = captions.pages([W("one.", 0, 900), W("two.", 600, 1500), W("three.", 1400, 3000)], max_chars=32, min_ms=100)
    assert [(p.startMs, p.endMs) for p in ps] == [(0, 600), (600, 1400), (1400, 3000)]


def test_the_last_page_ends_by_the_film_duration():
    ps = captions.pages([W("Almost", 0, 400), W("over.", 400, 2600)], max_chars=32, end_ms=2000)
    assert (ps[-1].startMs, ps[-1].endMs) == (0, 2000)
    short = captions.pages([W("Bye.", 1500, 1600)], max_chars=32, min_ms=700, end_ms=1800)
    assert (short[0].startMs, short[0].endMs) == (1500, 1800)   # the minimum duration yields to the end of the film


def test_no_page_ends_after_the_film_not_only_the_last_one():
    # "a long." runs to 4800 ms but the film ends at 4000 ms, and a later page exists, so it is not the last raw page.
    words = [W("a", 3300, 3450), W("long.", 3500, 4800), W("late.", 5000, 5200)]
    ps = captions.pages(words, max_chars=32, end_ms=4000)
    assert [(p.text, p.startMs, p.endMs) for p in ps] == [("a long.", 3300, 4000)]
    assert all(p.endMs <= 4000 for p in captions.pages(words, max_chars=32, end_ms=4000, min_ms=5000))


def test_pages_with_no_length_are_dropped():
    # A page that starts where the next one does, and one that starts after the film ends.
    ps = captions.pages([W("a.", 500, 600), W("b.", 500, 900), W("late.", 5000, 5200)], max_chars=32, end_ms=4000)
    assert [(p.text, p.startMs, p.endMs) for p in ps] == [("b.", 500, 1200)]


# --- fitting the line length to the font -----------------------------------------------------------------

def test_fit_max_chars_never_exceeds_the_pack_limit_or_the_text_box():
    # 1080 px wide, 8% margins, 54 px type: 907 px of text box, and GLYPH_EM em per character.
    derived = math.floor(1080 * (1 - 2 * captions.BOX_MARGIN) / (54 * captions.GLYPH_EM) + 1e-9)
    assert captions.fit_max_chars(26, width_px=1080, size_px=54) == derived == 21
    assert captions.fit_max_chars(30, width_px=1080, size_px=54) == 21
    assert captions.fit_max_chars(18, width_px=1080, size_px=54) == 18          # the pack limit when it is the tighter one
    assert captions.fit_max_chars(32, width_px=1920, size_px=54) == 32
    assert captions.fit_max_chars(26, width_px=1080, size_px=54, plate=True) < derived
    assert captions.fit_max_chars(22, width_px=1080, size_px=90) < captions.fit_max_chars(22, width_px=1080, size_px=60)
    assert captions.fit_max_chars(22, width_px=1080, size_px=4000) == 1         # always room for a word


def test_fit_max_chars_ignores_a_size_it_cannot_use():
    for bad in (None, 0, -5, "54", True):
        assert captions.fit_max_chars(26, width_px=1080, size_px=bad) == 26


def _captured_pages(tmp_path, aspect, pack, slug="p"):
    run = new_run(tmp_path, slug, "saas-kinetic", aspect, None, "en")
    seen = {}

    def fake_render(project, out, **kw):
        seen["project"] = project
        out.write_bytes(b"mov")
        return out

    captions.build(run, _timing(aspect), pack, render_fn=fake_render)
    return json.loads((seen["project"] / "captions.json").read_text())


def _long_words_timing(aspect, n=24, text="WOMEN MAKE MOMMY WOW SWIMMING MEMORANDUM"):
    words = [Caption(text=w, startMs=i * 300, endMs=i * 300 + 280) for i, w in
             enumerate((f"{text} " * n).split()[:n])]
    line = _timing(aspect).lines[0].model_copy(update={"words": words, "estimated": False})
    return _timing(aspect).model_copy(update={"lines": [line], "duration_s": 12.0})


def test_a_square_film_at_54px_uses_the_derived_limit_not_the_packs_26(tmp_path):
    pack = load_pack("saas-kinetic")
    assert pack.captions["max_chars"]["1:1"] == 26 and pack.captions["style"]["size_px"] == 54
    run = new_run(tmp_path, "sq", "saas-kinetic", "1:1", None, "en")
    seen = {}

    def fake_render(project, out, **kw):
        seen["project"] = project
        out.write_bytes(b"mov")
        return out

    captions.build(run, _long_words_timing("1:1"), pack, render_fn=fake_render)
    data = json.loads((seen["project"] / "captions.json").read_text())
    derived = captions.fit_max_chars(26, width_px=1080, size_px=54)
    assert derived < 26 and data["pages"]
    for page in data["pages"]:
        assert captions._line_count([Caption(**w) for w in page["words"]], derived) <= 2


def test_a_bigger_font_makes_pages_shorter(tmp_path):
    pack = load_pack("saas-kinetic")
    big = pack.model_copy(update={"captions": deep_merge(pack.captions, {"style": {"size_px": 90}})})
    run_a = new_run(tmp_path, "a", "saas-kinetic", "9:16", None, "en")
    run_b = new_run(tmp_path, "b", "saas-kinetic", "9:16", None, "en")
    seen = {}

    def fake_render(project, out, **kw):
        seen.setdefault("projects", []).append(project)
        out.write_bytes(b"mov")
        return out

    t = _long_words_timing("9:16")
    captions.build(run_a, t, pack, render_fn=fake_render)
    captions.build(run_b, t, big, render_fn=fake_render)
    pages_54, pages_90 = (json.loads((pr / "captions.json").read_text())["pages"] for pr in seen["projects"])
    assert len(pages_90) > len(pages_54)


def test_the_sidecars_keep_the_packs_limit_when_there_is_no_overlay(tmp_path):
    # 16:9 has no overlay, so a huge font size must not narrow the pack's 32 characters per line.
    pack = load_pack("saas-kinetic")
    huge = pack.model_copy(update={"captions": deep_merge(pack.captions, {"style": {"size_px": 200}})})
    run = new_run(tmp_path, "w", "saas-kinetic", "16:9", None, "en")
    out = captions.build(run, _long_words_timing("16:9", text="ABCDEFGHIJ KLMNOPQRSTU"), huge,
                         render_fn=lambda *a, **k: pytest.fail("no overlay"))
    first_cue = out["srt"].read_text().splitlines()[2]
    assert first_cue == "ABCDEFGHIJ KLMNOPQRSTU ABCDEFGHIJ KLMNOPQRSTU"   # two lines of two words (22 characters each)


# --- plate and shadow -------------------------------------------------------------------------------------

def test_resolve_style_maps_plate_and_shadow_roles_too():
    design = {"tokens": {"colors": {"dark": "#1E3510", "shadow": "rgba(0,0,0,0.35)"}}}
    style = captions.resolve_style({"plate": "dark", "shadow": "shadow", "color": "#fff"}, design)
    assert style == {"plate": "#1E3510", "shadow": "rgba(0,0,0,0.35)", "color": "#fff"}
    assert captions.resolve_style({"shadow": "none"}, design) == {"shadow": "none"}


def test_resolve_style_survives_malformed_design_tokens():
    for bad in ({"tokens": []}, {"tokens": {"colors": []}}, [], {"tokens": None}):
        assert captions.resolve_style({"color": "ink"}, bad) == {"color": "ink"}


@pytest.mark.parametrize("value", [None, "", "  ", "none", "transparent", "rgba(0,0,0,0.0)", "rgba(0, 0, 0, 0)",
                                   "rgba(0,0,0,.0)", "#00000000", "#0000"])
def test_an_invisible_plate_is_no_plate(value):
    assert not captions._has_plate(value)


@pytest.mark.parametrize("value", ["#1E3510", "rgba(0,0,0,0.5)", "rgba(0,0,0,0.05)", "#1E351080", "black"])
def test_a_visible_plate_is_a_plate(value):
    assert captions._has_plate(value)


def test_plate_and_shadow_reach_the_overlay_template(tmp_path):
    pack = load_pack("saas-kinetic")
    styled = pack.model_copy(update={"captions": deep_merge(pack.captions, {"style": {"plate": "dark", "shadow": "shadow"}})})
    data = _captured_pages(tmp_path, "9:16", styled)
    assert data["style"]["plate"] == pack.tokens["colors"]["dark"] and data["style"]["shadow"] == pack.tokens["colors"]["shadow"]
    assert data["box"] == {"margin": captions.BOX_MARGIN, "plate_pad_x_em": captions.PLATE_PAD_X_EM}
    html = OVERLAY.read_text()
    assert "st.plate" in html and "st.shadow" in html and "plate_pad_x_em" in html   # the template consumes them


def test_the_packs_transparent_default_plate_is_dropped_and_the_shadow_stays_unset(tmp_path):
    pack = load_pack("saas-kinetic")
    assert pack.captions["style"]["plate"] == "rgba(0,0,0,0.0)"
    data = _captured_pages(tmp_path, "9:16", pack)
    assert "plate" not in data["style"] and "shadow" not in data["style"]   # the template's soft black shadow applies


def test_a_plate_narrows_the_text_box_so_pages_get_shorter(tmp_path):
    pack = load_pack("saas-kinetic")
    plated = pack.model_copy(update={"captions": deep_merge(pack.captions, {"style": {"plate": "dark"}})})
    run_a = new_run(tmp_path, "a", "saas-kinetic", "9:16", None, "en")
    run_b = new_run(tmp_path, "b", "saas-kinetic", "9:16", None, "en")
    seen = []

    def fake_render(project, out, **kw):
        seen.append(project)
        out.write_bytes(b"mov")
        return out

    # Two words that make a 21-character line: it fits the plain text box (21) but not the plated one (19).
    t = _long_words_timing("9:16", text="ABCDEFGHIJ KLMNOPQRST")
    captions.build(run_a, t, pack, render_fn=fake_render)
    captions.build(run_b, t, plated, render_fn=fake_render)
    plain, with_plate = (json.loads((pr / "captions.json").read_text())["pages"] for pr in seen)
    assert len(with_plate) > len(plain)


# --- re-rendering only when the inputs changed -------------------------------------------------------------

def _recording_render(calls):
    def fake_render(project, out, **kw):
        calls.append(kw)
        out.write_bytes(b"mov")
        write_sidecar(out, tool="hyperframes", version="0.8.98", flags=[], inputs=kw["inputs"])
        return out
    return fake_render


def test_a_second_run_with_the_same_inputs_does_not_render_again(tmp_path):
    run = new_run(tmp_path, "v", "saas-kinetic", "9:16", None, "en")
    pack, calls = load_pack("saas-kinetic"), []
    first = captions.build(run, _timing("9:16"), pack, render_fn=_recording_render(calls))
    assert len(calls) == 1 and len(calls[0]["inputs"]["hash"]) == 64
    second = captions.build(run, _timing("9:16"), pack, render_fn=lambda *a, **k: pytest.fail("inputs unchanged"))
    assert second["overlay"] == first["overlay"] and second["overlay"].read_bytes() == b"mov"


def test_a_changed_word_renders_again_with_a_new_hash(tmp_path):
    run = new_run(tmp_path, "v", "saas-kinetic", "9:16", None, "en")
    pack, calls = load_pack("saas-kinetic"), []
    captions.build(run, _timing("9:16"), pack, render_fn=_recording_render(calls))
    t = _timing("9:16")
    line = t.lines[0].model_copy(update={"words": [t.lines[0].words[0].model_copy(update={"text": "Shifting"}),
                                                   *t.lines[0].words[1:]]})
    captions.build(run, t.model_copy(update={"lines": [line, *t.lines[1:]]}), pack, render_fn=_recording_render(calls))
    assert len(calls) == 2 and calls[0]["inputs"]["hash"] != calls[1]["inputs"]["hash"]


@pytest.mark.parametrize("change", ["style", "band", "canvas", "duration", "template"])
def test_every_render_input_is_part_of_the_hash(tmp_path, monkeypatch, change):
    run = new_run(tmp_path, "v", "saas-kinetic", "9:16", None, "en")
    pack, t, calls = load_pack("saas-kinetic"), _timing("9:16"), []
    captions.build(run, t, pack, render_fn=_recording_render(calls))
    if change == "style":
        pack = pack.model_copy(update={"captions": deep_merge(pack.captions, {"style": {"weight": 800}})})
    elif change == "band":
        pack = pack.model_copy(update={"tokens": deep_merge(pack.tokens, {"caption_band": {"9:16": [0.6, 0.8]}})})
    elif change == "canvas":
        t = t.model_copy(update={"fps": 30})
    elif change == "duration":
        t = t.model_copy(update={"duration_s": 9.0})
    else:
        tpl = tmp_path / "index.html"
        tpl.write_text(OVERLAY.read_text() + "\n<!-- edit -->")
        monkeypatch.setattr(captions, "OVERLAY_TEMPLATE", tpl)
    captions.build(run, t, pack, render_fn=_recording_render(calls))
    assert len(calls) == 2


def test_a_missing_or_corrupt_sidecar_means_render_again(tmp_path):
    run = new_run(tmp_path, "v", "saas-kinetic", "9:16", None, "en")
    pack, calls = load_pack("saas-kinetic"), []
    out = captions.build(run, _timing("9:16"), pack, render_fn=_recording_render(calls))
    side = out["overlay"].with_name("captions.mov.render.json")
    side.write_text("{not json")
    captions.build(run, _timing("9:16"), pack, render_fn=_recording_render(calls))
    side.unlink()
    captions.build(run, _timing("9:16"), pack, render_fn=_recording_render(calls))
    out["overlay"].unlink()
    captions.build(run, _timing("9:16"), pack, render_fn=_recording_render(calls))
    assert len(calls) == 4


def test_a_failed_render_with_changed_inputs_leaves_no_stale_overlay(tmp_path):
    run = new_run(tmp_path, "v", "saas-kinetic", "9:16", None, "en")
    pack, calls = load_pack("saas-kinetic"), []
    out = captions.build(run, _timing("9:16"), pack, render_fn=_recording_render(calls))

    def refuse(*a, **k):
        raise hf.HyperframesError("not enough disk for this render")

    changed = _timing("9:16").model_copy(update={"duration_s": 9.0})
    with pytest.raises(hf.HyperframesError):
        captions.build(run, changed, pack, render_fn=refuse)
    assert not out["overlay"].exists() and not out["overlay"].with_name("captions.mov.render.json").exists()


def test_a_corrupt_design_json_after_the_inputs_changed_leaves_no_stale_overlay(tmp_path):
    run = new_run(tmp_path, "v", "saas-kinetic", "9:16", None, "en")
    pack, calls = load_pack("saas-kinetic"), []
    out = captions.build(run, _timing("9:16"), pack, render_fn=_recording_render(calls))
    assert out["overlay"].exists()
    changed = _timing("9:16").model_copy(update={"duration_s": 9.0})
    run.design.write_text("{not json")
    with pytest.raises(captions.CaptionsError, match="design.json cannot be read"):
        captions.build(run, changed, pack, render_fn=lambda *a, **k: pytest.fail("must fail before rendering"))
    assert not out["overlay"].exists() and not out["overlay"].with_name("captions.mov.render.json").exists()


def _fail_writes_to(monkeypatch, name):
    real = captions._write

    def write(path, text):
        if path.name == name:
            raise OSError(28, "No space left on device")
        return real(path, text)

    monkeypatch.setattr(captions, "_write", write)


@pytest.mark.parametrize("sidecar", ["words.json", "captions.srt", "captions.vtt"])
def test_a_sidecar_write_failing_after_the_inputs_changed_leaves_no_stale_overlay(tmp_path, monkeypatch, sidecar):
    run = new_run(tmp_path, "v", "saas-kinetic", "9:16", None, "en")
    pack, calls = load_pack("saas-kinetic"), []
    out = captions.build(run, _timing("9:16"), pack, render_fn=_recording_render(calls))
    _fail_writes_to(monkeypatch, sidecar)
    changed = _timing("9:16").model_copy(update={"duration_s": 9.0})
    with pytest.raises(OSError, match="No space left"):
        captions.build(run, changed, pack, render_fn=lambda *a, **k: pytest.fail("must fail before rendering"))
    assert not out["overlay"].exists() and not out["overlay"].with_name("captions.mov.render.json").exists()


def test_a_sidecar_write_failing_with_unchanged_inputs_keeps_the_valid_overlay(tmp_path, monkeypatch):
    run = new_run(tmp_path, "v", "saas-kinetic", "9:16", None, "en")
    pack, calls = load_pack("saas-kinetic"), []
    out = captions.build(run, _timing("9:16"), pack, render_fn=_recording_render(calls))
    _fail_writes_to(monkeypatch, "captions.srt")
    with pytest.raises(OSError):
        captions.build(run, _timing("9:16"), pack, render_fn=lambda *a, **k: pytest.fail("inputs unchanged"))
    assert out["overlay"].read_bytes() == b"mov"


def test_a_run_that_changes_nothing_never_reaches_a_failing_render(tmp_path):
    # The disk check lives in render, so an unchanged overlay must not even get that far.
    run = new_run(tmp_path, "v", "saas-kinetic", "9:16", None, "en")
    pack, calls = load_pack("saas-kinetic"), []
    captions.build(run, _timing("9:16"), pack, render_fn=_recording_render(calls))

    def refuse(*a, **k):
        raise hf.HyperframesError("not enough disk for this render")

    again = captions.build(run, _timing("9:16"), pack, render_fn=refuse)
    assert again["overlay"].read_bytes() == b"mov"


def test_a_stale_overlay_is_still_removed_when_the_pack_stops_asking_for_one(tmp_path):
    run = new_run(tmp_path, "v", "saas-kinetic", "9:16", None, "en")
    pack, calls = load_pack("saas-kinetic"), []
    out = captions.build(run, _timing("9:16"), pack, render_fn=_recording_render(calls))
    off = pack.model_copy(update={"captions": deep_merge(pack.captions, {"mode": {"9:16": "none"}})})
    again = captions.build(run, _timing("9:16"), off, render_fn=lambda *a, **k: pytest.fail("mode none"))
    assert "overlay" not in again and not out["overlay"].exists()


# --- promo captions ---------------------------------------------------------------------------------------

def _cli_run(tmp_path, monkeypatch, aspect="16:9", voiced=True):
    from typer.testing import CliRunner

    from promo.cli import app

    run = new_run(tmp_path, "demo", "saas-kinetic", aspect, None, "en")
    run.shotlist.write_text(FIX.read_text())
    t = _timing(aspect)
    if voiced:   # every line has a take, so nothing is estimated
        t = t.model_copy(update={"lines": [line.model_copy(update={"estimated": False}) for line in t.lines]})
    write_timing(t, run.timing)
    monkeypatch.chdir(run.root)
    return CliRunner(), app, run


def test_cli_captions_needs_timing_first(tmp_path, monkeypatch):
    runner, app, run = _cli_run(tmp_path, monkeypatch)
    run.timing.unlink()
    res = runner.invoke(app, ["captions"])
    assert res.exit_code == 2 and "promo timing" in res.output and "Traceback" not in res.output


def test_cli_captions_rejects_estimated_lines(tmp_path, monkeypatch):
    runner, app, run = _cli_run(tmp_path, monkeypatch, voiced=False)
    res = runner.invoke(app, ["captions"])
    assert res.exit_code == 2 and "promo voice --all" in res.output
    assert not (run.captions_dir / "captions.srt").exists()


def test_cli_captions_reports_a_corrupt_timing_file_in_one_line(tmp_path, monkeypatch):
    runner, app, run = _cli_run(tmp_path, monkeypatch)
    run.timing.write_text("{not json")
    res = runner.invoke(app, ["captions"])
    assert res.exit_code == 1 and res.output.startswith("promo: timing.json cannot be read")
    assert "Traceback" not in res.output


def test_cli_captions_writes_sidecars_at_16x9(tmp_path, monkeypatch):
    runner, app, run = _cli_run(tmp_path, monkeypatch)
    res = runner.invoke(app, ["captions"])
    assert res.exit_code == 0, res.output
    printed = {line.split()[0]: Path(line.split()[1]) for line in res.output.splitlines()}
    assert set(printed) == {"words", "srt", "vtt"}
    assert all(p.exists() for p in printed.values()) and printed["srt"] == run.captions_dir / "captions.srt"


def test_cli_captions_without_a_run_folder_exits_2(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from promo.cli import app

    monkeypatch.chdir(tmp_path)
    res = CliRunner().invoke(app, ["captions"])
    assert res.exit_code == 2 and "no run folder" in res.output


def test_cli_captions_disk_too_small_is_one_line_exit_1(tmp_path, monkeypatch):
    runner, app, run = _cli_run(tmp_path, monkeypatch, aspect="9:16")
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(hf.shutil, "disk_usage", lambda p: Usage(1, 1, 3 * 10**9))
    monkeypatch.setattr(hf, "run", lambda *a, **k: pytest.fail("render must not start"))
    res = runner.invoke(app, ["captions"])
    assert res.exit_code == 1 and res.output.startswith("promo: not enough disk")
    assert "Traceback" not in res.output and len(res.output.strip().splitlines()) == 1
    assert (run.captions_dir / "captions.srt").exists()   # the sidecars were written before the render refused


def test_cli_captions_failed_render_is_one_line_exit_1(tmp_path, monkeypatch):
    runner, app, run = _cli_run(tmp_path, monkeypatch, aspect="9:16")

    def boom(*a, **k):
        raise ToolError("command failed (1): npx --yes hyperframes@0.8.98 render\nChrome crashed\nat frame 12")

    monkeypatch.setattr(hf, "run", boom)
    res = runner.invoke(app, ["captions"])
    lines = res.output.strip().splitlines()
    assert res.exit_code == 1
    assert lines == ["promo: command failed (1): npx --yes hyperframes@0.8.98 render (full output: logs/last-error.log)"]
    assert (run.root / "logs" / "last-error.log").read_text() == (
        "command failed (1): npx --yes hyperframes@0.8.98 render\nChrome crashed\nat frame 12\n")
    assert not (run.captions_dir / "overlay" / "captions.mov").exists()


def test_tool_error_keeps_its_full_text_for_debugging(tmp_path, monkeypatch):
    runner, app, run = _cli_run(tmp_path, monkeypatch, aspect="9:16")
    err = ToolError("command failed (1): npx x\nChrome crashed")
    monkeypatch.setattr(hf, "run", lambda *a, **k: (_ for _ in ()).throw(err))
    runner.invoke(app, ["captions"])
    assert str(err) == "command failed (1): npx x\nChrome crashed"   # only the printed message is shortened


def test_the_error_log_holds_the_command_exit_code_and_stderr_tail(tmp_path, monkeypatch):
    from promo.proc import run as run_tool

    runner, app, run = _cli_run(tmp_path, monkeypatch, aspect="9:16")
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(hf.shutil, "disk_usage", lambda p: Usage(10**12, 0, 10**12))
    # The real subprocess runner builds the ToolError: exit code 3 and a stderr of 50 lines.
    script = "import sys\nfor i in range(50): print('stderr line', i, file=sys.stderr)\nsys.exit(3)"
    monkeypatch.setattr(hf, "run", lambda cmd, **kw: run_tool([sys.executable, "-c", script], **{k: v for k, v in kw.items() if k != "env"}))
    res = runner.invoke(app, ["captions"])
    lines = res.output.strip().splitlines()
    assert res.exit_code == 1 and len(lines) == 1
    assert lines[0].startswith("promo: command failed (3): ") and lines[0].endswith("(full output: logs/last-error.log)")
    assert "stderr line" not in res.output
    log = (run.root / "logs" / "last-error.log").read_text()
    assert log.splitlines()[0].startswith("command failed (3): ")        # the command and its exit code
    assert "stderr line 49" in log and "stderr line 10" in log             # the stderr tail (the last 40 lines)
    assert "stderr line 9\n" not in log
    assert sorted(p.name for p in (run.root / "logs").iterdir()) == ["last-error.log"]   # no temp file left behind


def test_the_logged_path_is_relative_to_the_current_directory(tmp_path, monkeypatch):
    runner, app, run = _cli_run(tmp_path, monkeypatch, aspect="9:16")
    monkeypatch.setattr(hf, "run", lambda *a, **k: (_ for _ in ()).throw(ToolError("command failed (1): npx x\nboom")))
    monkeypatch.chdir(run.scenes)   # a subfolder of the run: the log is one level up
    res = runner.invoke(app, ["captions"])
    assert res.output.strip() == "promo: command failed (1): npx x (full output: ../logs/last-error.log)"
    assert (run.root / "logs" / "last-error.log").read_text().endswith("boom\n")


def test_the_logged_path_is_absolute_when_it_cannot_be_made_relative(tmp_path, monkeypatch, capsys):
    from promo import cli

    runner, app, run = _cli_run(tmp_path, monkeypatch, aspect="9:16")

    def other_drive(path, start):
        raise ValueError("path is on mount 'C:', start on mount 'D:'")

    monkeypatch.setattr("os.path.relpath", other_drive)
    with pytest.raises(cli.typer.Exit):
        raise cli._provider_failure(ToolError("command failed (1): npx x\nboom"))
    assert capsys.readouterr().err.strip() == f"promo: command failed (1): npx x (full output: {run.root / 'logs' / 'last-error.log'})"


def test_a_later_failure_replaces_the_error_log(tmp_path, monkeypatch):
    runner, app, run = _cli_run(tmp_path, monkeypatch, aspect="9:16")
    for n in (1, 2):
        monkeypatch.setattr(hf, "run", lambda *a, n=n, **k: (_ for _ in ()).throw(ToolError(f"command failed ({n}): npx x\ntail {n}")))
        runner.invoke(app, ["captions"])
    assert (run.root / "logs" / "last-error.log").read_text() == "command failed (2): npx x\ntail 2\n"


def test_outside_a_run_folder_the_error_is_its_first_line_alone(tmp_path, monkeypatch):
    from promo import cli

    monkeypatch.chdir(tmp_path)
    with pytest.raises(cli.typer.Exit) as exc:
        raise cli._provider_failure(ToolError("command failed (1): npx x\nChrome crashed"))
    assert exc.value.exit_code == 1
    assert not (tmp_path / "logs").exists()


def test_an_unwritable_error_log_falls_back_to_the_first_line(tmp_path, monkeypatch, capsys):
    from promo import cli

    runner, app, run = _cli_run(tmp_path, monkeypatch, aspect="9:16")
    (run.root / "logs").write_text("a file where the folder should be")
    with pytest.raises(cli.typer.Exit):
        raise cli._provider_failure(ToolError("command failed (1): npx x\nChrome crashed"))
    assert capsys.readouterr().err.strip() == "promo: command failed (1): npx x"


def test_logs_are_gitignored_in_new_runs(tmp_path):
    run = new_run(tmp_path, "g", "saas-kinetic", "16:9", None, "en")
    assert "logs/" in (run.root / ".gitignore").read_text().splitlines()


def test_a_one_line_tool_error_is_printed_as_it_is(tmp_path, monkeypatch):
    runner, app, run = _cli_run(tmp_path, monkeypatch, aspect="9:16")
    monkeypatch.setattr(hf, "run", lambda *a, **k: (_ for _ in ()).throw(ToolError("command failed (1): npx x")))
    res = runner.invoke(app, ["captions"])
    assert res.output.strip() == "promo: command failed (1): npx x"


def test_cli_captions_without_node_is_a_missing_prerequisite_exit_2(tmp_path, monkeypatch):
    runner, app, run = _cli_run(tmp_path, monkeypatch, aspect="9:16")
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(hf.shutil, "disk_usage", lambda p: Usage(10**12, 0, 10**12))
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))   # the real subprocess call cannot find npx
    res = runner.invoke(app, ["captions"])
    assert res.exit_code == 2 and res.output.strip() == "promo: HyperFrames needs Node.js (npx not found)"


def test_cli_captions_reports_an_unreadable_design_json_in_one_line(tmp_path, monkeypatch):
    runner, app, run = _cli_run(tmp_path, monkeypatch, aspect="9:16")
    run.design.write_text("{not json")
    monkeypatch.setattr(hf, "run", lambda *a, **k: pytest.fail("render must not start"))
    res = runner.invoke(app, ["captions"])
    assert res.exit_code == 1 and "Traceback" not in res.output
    assert res.output.strip().startswith("promo: design.json cannot be read (JSONDecodeError)")
    assert len(res.output.strip().splitlines()) == 1


def test_cli_scenes_shortens_a_tool_error_the_same_way(tmp_path, monkeypatch):
    from promo import scenes as sc

    runner, app, run = _cli_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sc, "render_shots", lambda *a, **k: (_ for _ in ()).throw(ToolError("command failed (1): npx x\nboom")))
    res = runner.invoke(app, ["scenes"])
    assert res.exit_code == 1 and res.output.strip() == "promo: command failed (1): npx x (full output: logs/last-error.log)"
    assert (run.root / "logs" / "last-error.log").read_text().endswith("boom\n")


@pytest.mark.live
def test_live_overlay_has_alpha(tmp_path):
    run = new_run(tmp_path, "v", "saas-kinetic", "9:16", None, "en")
    # Three seconds is enough to carry the first spoken line and keeps the frame storage near 1.6 GB.
    t = _timing("9:16")
    short = t.model_copy(update={"duration_s": 3.0, "lines": t.lines[:1]})
    out = captions.build(run, short, load_pack("saas-kinetic"))
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=codec_name,pix_fmt,nb_frames", "-of", "csv=p=0", str(out["overlay"])],
                           capture_output=True, text=True, check=True).stdout
    assert "prores" in probe and "yuva" in probe
    assert probe.strip().endswith("180")   # 3 s at 60 fps
