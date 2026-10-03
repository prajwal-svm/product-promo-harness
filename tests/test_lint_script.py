import json
from pathlib import Path

import pytest

from promo.lint_script import lint_shotlist, lint_text, numbers_in
from promo.shotlist import Shotlist

FIX = Path(__file__).parent / "fixtures" / "shotlist_music.json"


def codes(findings, level=None):
    return [f.code for f in findings if level is None or f.level == level]


def test_numbers_in_normalises():
    assert numbers_in("Reversals over $10,000? About 45% of 3.50") == ["10000", "45", "3.5"]


def test_fixture_is_clean():
    sl = Shotlist.model_validate(json.loads(FIX.read_text()))
    assert codes(lint_shotlist(sl, "ad-sparse", "paraphrase"), "error") == []


def test_em_dash_and_not_x_but_y():
    f = lint_text("It's not a tool, it's a teammate — really.", "professor")
    assert {"em_dash", "not_x_but_y"} <= set(codes(f, "error"))
    assert "not_x_but_y" in codes(lint_text("Not just faster but safer.", "professor"))
    assert "not_x_but_y" in codes(lint_text("This isn't just a dashboard.", "professor"))


def test_openers_and_hype():
    assert "rhetorical_opener" in codes(lint_text("Imagine a world without spreadsheets.", "professor"))
    assert "rhetorical_opener" not in codes(lint_text("Meet Signet.", "ad-sparse"))
    assert "hype_word" in codes(lint_text("A seamless, revolutionary workflow.", "professor"))


def test_unsourced_number():
    data = json.loads(FIX.read_text())
    data["chapters"][1]["vo"][0]["claims"] = []
    sl = Shotlist.model_validate(data)
    errs = [f for f in lint_shotlist(sl, "ad-sparse", "paraphrase") if f.code == "unsourced_number"]
    assert errs and errs[0].ref == "c02-l1"


def test_line_too_long_only_in_ad_sparse():
    text = "one two three four five six seven eight nine ten eleven twelve thirteen."
    assert "line_too_long" in codes(lint_text(text, "ad-sparse"))
    assert "line_too_long" not in codes(lint_text(text, "professor"))


def test_fragment_run_level_depends_on_register():
    text = "Fast. Simple. Secure. Done."
    assert "fragment_run" in codes(lint_text(text, "professor"), "error")
    assert "fragment_run" in codes(lint_text(text, "ad-sparse"), "warning")


def test_triplet_and_density_warnings():
    assert "triplet" in codes(lint_text("Plan, build, and ship.", "professor"), "warning")
    assert "number_density" in codes(lint_text("We did 3 things in 2 days for 5 teams.", "professor"), "warning")


def test_sentence_variance_for_professor():
    flat = "The cat sat down now. The dog ran off now. The bird flew up now. The fish swam by now. The cow lay down now."
    assert "sentence_variance" in codes(lint_text(flat, "professor"), "warning")


def test_headline_duplicating_vo():
    data = json.loads(FIX.read_text())
    data["chapters"][0]["headline"]["text"] = "Moving company money just got safer."
    sl = Shotlist.model_validate(data)
    assert "text_duplicates_vo" in codes(lint_shotlist(sl, "ad-sparse", "paraphrase"), "warning")
    data["chapters"][0]["shots"][0]["text_mode"] = "verbatim-sync"
    sl2 = Shotlist.model_validate(data)
    assert "text_duplicates_vo" not in codes(lint_shotlist(sl2, "ad-dense", "verbatim-sync"))


# Fix 1: Typographic apostrophes
def test_typographic_apostrophes():
    # Curly apostrophe should be normalized and flagged as not_x_but_y
    f = lint_text("It’s not a tool, it’s a teammate.", "professor")
    assert "not_x_but_y" in codes(f, "error")
    # Test line length with curly quotes (10 words, should not flag)
    f = lint_text("Don’t wait, it’s what’s next and we’re here right now.", "ad-sparse")
    assert "line_too_long" not in codes(f)
    # Test hype word with "in today's fast-paced"
    f = lint_text("In today’s fast-paced world.", "professor")
    assert "hype_word" in codes(f, "error")


# Fix 2: not_x_but_y comprehensive tests
@pytest.mark.parametrize("text", [
    "It's not a tool, it's a teammate.",
    "Signet isn't a tool, it's a teammate.",
    "We're not just faster, we're safer.",
    "You're not buying software, you're buying certainty.",
    "It's not about speed. It's about trust.",
    "This isn't a tool. It's a teammate.",
    "It's not a tool but a teammate.",
    "Not just faster but safer.",
    "This isn't just a dashboard.",
    "Acme isn't a tool, it's a teammate.",
    "Stripe isn't a tool. It's a teammate.",
    "Our platform isn't a tool, it's a teammate.",
    "The product is not a tool, it's a teammate.",
    "Audits aren't a chore, they're a habit.",
    "It's not a simple tool but a teammate.",
])
def test_not_x_but_y_must_flag(text):
    f = lint_text(text, "professor")
    assert "not_x_but_y" in codes(f, "error"), f"Should flag: {text}"


@pytest.mark.parametrize("text", [
    "Your platform isn't a tool, it's a teammate.",
    "Her job isn't easy, it's rewarding.",
    "Websites aren't a chore, they're a habit.",
    "Items aren't a chore, they're a habit.",
    "Healthcare isn't a cost, it's a foundation.",
    "Heroes aren't born, they're made.",
])
def test_not_x_but_y_must_flag_noun_subject(text):
    f = lint_text(text, "professor")
    assert "not_x_but_y" in codes(f, "error"), f"Should flag: {text}"


@pytest.mark.parametrize("text", [
    "It's not live yet, but it's close.",
    "It's not cheap, but it's worth it.",
    "Reversals over $10,000? Risk has to sign off.",
    "Don't guess. Measure.",
    "You're not alone. Every team has this problem, and it's fixable.",
    "It's not free. Pricing starts at five dollars, and that's fair.",
    "We're not hiding anything. That's the point.",
    "Don't wait, it's what's next and we're here right now.",
    "Don't wait, it's live.",
    "I'm not a tool, it's a teammate.",
    "I am not done, it's fine.",
])
def test_not_x_but_y_must_not_flag(text):
    f = lint_text(text, "professor")
    assert "not_x_but_y" not in codes(f, "error"), f"Should not flag: {text}"


# Fix 3: hype_word inflections and AI_VOCAB
def test_hype_word_inflections():
    assert "hype_word" in codes(lint_text("Supercharged teams.", "professor"), "error")
    assert "hype_word" in codes(lint_text("It unleashed growth.", "professor"), "error")
    assert "hype_word" in codes(lint_text("Revolutionizing audits.", "professor"), "error")
    assert "hype_word" in codes(lint_text("Empowered users.", "professor"), "error")


def test_hype_word_spacing():
    assert "hype_word" in codes(lint_text("A game changer.", "professor"), "error")
    assert "hype_word" in codes(lint_text("Cutting edge tools.", "professor"), "error")


def test_hype_word_tapestry_synergy():
    # Test y → ies inflections
    assert "hype_word" in codes(lint_text("Rich tapestries.", "professor"), "error")
    assert "hype_word" in codes(lint_text("Synergies everywhere.", "professor"), "error")


def test_hype_word_multi_word():
    # Test multi-word hype phrase inflections
    assert "hype_word" in codes(lint_text("Unlocking the power of data.", "professor"), "error")


def test_ai_vocab_warning():
    f = lint_text("A pivotal moment.", "professor")
    assert "ai_vocab" in codes(f, "warning")
    assert "hype_word" not in codes(f, "error")


# Fix 4: numbers_in magnitude suffixes
def test_numbers_in_basic_cases():
    assert numbers_in("10 rows and 10k users") == ["10", "10000"]
    assert numbers_in("15 teams and 5k users") == ["15", "5000"]
    assert numbers_in("5mn users") == ["5000000"]
    assert numbers_in("1.001k") == ["1001"]
    assert numbers_in("5min") == ["5"]


def test_numbers_in_magnitude_suffixes():
    assert numbers_in("$10k and 1.5M users and 2bn rows") == ["10000", "1500000", "2000000000"]


def test_unsourced_number_with_magnitude_suffix():
    data = json.loads(FIX.read_text())
    # Add a claim for $10k (normalized to 10000)
    data["claims"][0]["text"] = "Reversals over $10,000 need a Risk sign-off"
    data["chapters"][1]["vo"][0]["text"] = "Reversals over $10k?"
    data["chapters"][1]["vo"][0]["claims"] = ["k1"]
    sl = Shotlist.model_validate(data)
    errs = [f for f in lint_shotlist(sl, "ad-sparse", "paraphrase") if f.code == "unsourced_number"]
    assert not errs, "Should not flag $10k as unsourced when claim has $10,000"


# Fix 5: lint_text on multi-sentence prose
def test_lint_text_multi_sentence_line_rules_per_sentence():
    # Three sentences of ≤12 words each should not trigger line_too_long
    text = "One two three. Four five six. Seven eight nine."
    f = lint_text(text, "ad-sparse")
    assert "line_too_long" not in codes(f)


def test_lint_text_rhetorical_opener_per_sentence():
    text = "Audits are slow. Imagine no audits."
    f = lint_text(text, "professor")
    assert "rhetorical_opener" in codes(f, "error")


def test_verify_false_is_a_warning_not_an_error():
    data = json.loads(FIX.read_text())
    sl = Shotlist.model_validate(data)
    assert "verify_skipped" not in codes(lint_shotlist(sl, "ad-sparse", "paraphrase"))
    data["chapters"][0]["vo"][0]["verify"] = False
    found = [f for f in lint_shotlist(Shotlist.model_validate(data), "ad-sparse", "paraphrase")
             if f.code == "verify_skipped"]
    assert [(f.level, f.ref) for f in found] == [("warning", "c01-l1")]
    assert "c01-l1" in found[0].message and "Psst" in found[0].message
    assert codes(lint_shotlist(Shotlist.model_validate(data), "ad-sparse", "paraphrase"), "error") == []
