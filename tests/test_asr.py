import json
import shutil
import subprocess

import httpx
import pytest
import respx

from promo import asr
from promo.models import Caption

PARAKEET_TOKENS = [(" M", 1.2, 1.36), ("ov", 1.36, 1.52), ("ing", 1.52, 1.68), (" comp", 1.68, 1.92), ("any", 1.92, 2.08),
                   (" m", 2.08, 2.16), ("one", 2.16, 2.24), ("y", 2.24, 2.32), (" just", 2.4, 2.56), (" got", 2.56, 2.8),
                   (" sa", 2.8, 3.12), ("fer", 3.12, 3.44), (".", 3.84, 4.16)]


def test_merge_tokens_builds_words_and_keeps_punctuation_off_the_clock():
    words = asr.merge_tokens(PARAKEET_TOKENS)
    assert [w.text for w in words] == ["Moving", "company", "money", "just", "got", "safer."]
    assert (words[0].startMs, words[0].endMs) == (1200, 1680)
    assert words[-1].endMs == 3440


def test_wer_normalises_currency_and_numbers():
    assert asr.wer("Reversals over $10,000?", "reversals over ten thousand dollars") == 0.0
    # "a twelve thousand five hundred dollars reversal" (7 words) vs "...dollar..." = one substitution
    assert asr.wer("A $12,500.00 reversal", "a twelve thousand five hundred dollar reversal") == pytest.approx(1 / 7)
    assert asr.normalize_words("45% faster") == ["forty", "five", "percent", "faster"]


def test_wer_treats_typographic_apostrophes_as_straight():
    assert asr.wer("It\u2019s live.", "it's live") == 0.0
    assert asr.wer("\u201cDon\u2019t stop,\u201d she said", "don't stop she said") == 0.0


def test_wer_expands_magnitude_suffixes():
    assert asr.wer("Reversals over $10k?", "reversals over ten thousand dollars") == 0.0
    assert asr.normalize_words("1.5M users") == ["one", "million", "five", "hundred", "thousand", "users"]
    assert asr.normalize_words("2bn") == ["two", "billion"]
    # Not attached, or followed by another letter: left alone.
    assert asr.normalize_words("10 k") == ["ten", "k"]
    assert asr.normalize_words("5kg") == ["five", "kg"]


def test_wer_reads_ordinals_and_years_aloud():
    assert asr.wer("Our 1st launch in 2026", "our first launch in twenty twenty-six") == 0.0
    assert asr.normalize_words("the 22nd and 3rd") == ["the", "twenty", "second", "third"]
    # A comma, a currency sign, a percent sign or a decimal point makes it a plain number, not a year.
    assert asr.normalize_words("1,999 users") == ["one", "thousand", "nine", "hundred", "ninety", "nine", "users"]
    assert asr.normalize_words("$2000") == ["two", "thousand", "dollars"]
    assert asr.normalize_words("1999%") == ["one", "thousand", "nine", "hundred", "ninety", "nine", "percent"]
    assert asr.normalize_words("in 1905") == ["in", "nineteen", "oh", "five"]
    assert asr.normalize_words("2099 and 2100") == ["twenty", "ninety", "nine", "two", "thousand", "one", "hundred"]


def test_years_and_ordinals_in_other_languages():
    assert asr.normalize_words("en 2026", "fr") == ["en", "deux", "mille", "vingt", "six"]
    assert asr.normalize_words("le 3rd", "fr") == ["le", "troisième"]
    # A language without a year form (hi has no num2words support, vi has no to_year) reads the cardinal.
    assert asr.normalize_words("2026", "hi") == ["two", "thousand", "twenty", "six"]
    assert asr.normalize_words("2026", "vi") == asr.normalize_words("2,026", "vi")
    assert asr.normalize_words("3rd", "hi") == ["third"]
    # num2words' Japanese "year" is an era name (令和八年), so a bare year stays a cardinal.
    assert asr.normalize_words("2026", "ja") == asr.normalize_words("2,026", "ja")


def test_wer_counts_edits():
    assert asr.wer("a b c d", "a x c") == pytest.approx(0.5)
    assert asr.wer("", "") == 0.0


def test_normalize_unicode_and_character_languages():
    assert asr.normalize_words("Café crème, s'il vous plaît.", "fr") == ["café", "crème", "s'il", "vous", "plaît"]
    assert asr.normalize_words("東京タワー", "ja") == ["東", "京", "タ", "ワ", "ー"]


def test_engine_selection(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(asr, "_transcribe_parakeet", lambda p: calls.append("parakeet") or [])
    monkeypatch.setattr(asr, "_transcribe_whisper", lambda p, lang: calls.append(f"whisper:{lang}") or [])
    asr.transcribe(tmp_path / "a.wav", "en")
    asr.transcribe(tmp_path / "a.wav", "hi")
    assert calls == ["parakeet", "whisper:hi"]
    with pytest.raises(ValueError, match="unknown ASR engine"):
        asr.transcribe(tmp_path / "a.wav", "en", engine="nope")


@pytest.mark.parametrize("tag", ["en", "EN", "en-GB", "en_US", "en-us", " en "])
def test_every_english_tag_goes_to_parakeet(monkeypatch, tmp_path, tag):
    calls = []
    monkeypatch.setattr(asr, "_transcribe_parakeet", lambda p: calls.append("parakeet") or [])
    monkeypatch.setattr(asr, "_transcribe_whisper", lambda p, lang: calls.append(f"whisper:{lang}") or [])
    asr.transcribe(tmp_path / "a.wav", tag)
    assert calls == ["parakeet"]


@pytest.mark.parametrize("tag, primary", [("fr", "fr"), ("pt-BR", "pt"), ("PT_br", "pt"), ("zh-Hans-CN", "zh")])
def test_whisper_gets_only_the_primary_subtag(monkeypatch, tmp_path, tag, primary):
    import sys
    import types

    seen = {}
    fake = types.SimpleNamespace(transcribe=lambda path, **kw: seen.update(kw) or {"segments": []})
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
    assert asr.transcribe(tmp_path / "a.wav", tag) == []
    assert seen["language"] == primary


def test_an_unsupported_whisper_language_is_a_readable_value_error(monkeypatch, tmp_path):
    import sys
    import types

    def refuse(path, **kw):
        raise ValueError(f"Unsupported language: {kw['language']}")

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=refuse))
    with pytest.raises(ValueError, match=r"Whisper\) cannot transcribe language 'xx-YY': Unsupported language: xx"):
        asr.transcribe(tmp_path / "a.wav", "xx-YY")


def test_region_tags_get_the_same_normalisation_as_their_language():
    assert asr.normalize_words("organise the colour", "EN") == ["organize", "the", "color"]
    assert asr.normalize_words("organise", "en_GB") == ["organize"]
    assert asr.normalize_words("東京", "ja-JP") == ["東", "京"]                    # character tokens, as for "ja"
    assert asr.normalize_words("2026", "ja-JP") == asr.normalize_words("2,026", "ja-JP")


@respx.mock
def test_scribe_parses_words(monkeypatch, tmp_path):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF0000WAVE")
    route = respx.post(asr.SCRIBE_URL).mock(return_value=httpx.Response(200, json={"words": [
        {"text": "Hello", "start": 0.1, "end": 0.4, "type": "word"},
        {"text": " ", "start": 0.4, "end": 0.45, "type": "spacing"},
        {"text": "there.", "start": 0.45, "end": 0.8, "type": "word"},
    ]}))
    words = asr.transcribe(wav, "en", engine="scribe")
    assert [(w.text, w.startMs, w.endMs) for w in words] == [("Hello", 100, 400), ("there.", 450, 800)]
    sent = route.calls[0].request
    assert sent.headers["xi-api-key"] == "test-key"
    assert b"scribe_v2" in sent.content and b"timestamps_granularity" in sent.content


# --- US/UK spelling ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("british, american", [
    ("colour", "color"), ("colourful", "colorful"), ("favourite", "favorite"), ("behaviour", "behavior"),
    ("neighbourhood", "neighborhood"), ("honoured", "honored"),
    ("normalise", "normalize"), ("normalised", "normalized"), ("organisation", "organization"),
    ("organisations", "organizations"), ("organise", "organize"), ("unorganised", "unorganized"),
    ("realise", "realize"), ("emphasise", "emphasize"), ("analyse", "analyze"), ("analysed", "analyzed"),
    ("centre", "center"), ("centres", "centers"), ("centred", "centered"), ("metre", "meter"),
    ("kilometres", "kilometers"), ("litre", "liter"), ("theatre", "theater"),
    ("catalogue", "catalog"), ("catalogues", "catalogs"), ("cataloguing", "cataloging"),
])
def test_us_spelling_rewrites_the_british_forms_it_names(british, american):
    assert asr.us_spelling(british) == american
    assert asr.us_spelling(american) == american                         # the American form is already normal


@pytest.mark.parametrize("word", [
    "our", "hour", "your", "four", "pour", "flour", "tour", "advise", "promise", "surprise", "exercise", "otherwise",
    "rise", "noise", "precise", "enterprise", "expertise", "organ", "organic", "normal", "realism", "analysis",
    "emphasis", "figure", "entire", "more", "there", "literature", "litter", "metered", "theatrical", "colony",
])
def test_us_spelling_leaves_other_words_alone(word):
    assert asr.us_spelling(word) == word


def test_wer_treats_british_and_american_spelling_as_the_same_word():
    assert asr.wer("Dark mode, your own colour, and twenty-one languages.",
                   "Dark mode, your own color and twenty one languages.") == 0.0
    assert asr.wer("Bring your favourite editor keys.", "Bring your favorite editor keys.") == 0.0
    assert asr.wer("Normalise the centre column.", "normalize the center column") == 0.0
    assert asr.wer("Pick a colour.", "Pick a collar.") == pytest.approx(1 / 3)       # a real error still counts


def test_spelling_rules_are_for_english_only():
    assert asr.normalize_words("Nous organisons le centre.", "fr") == ["nous", "organisons", "le", "centre"]
    assert asr.normalize_words("organise", "fr") == ["organise"]
    assert asr.normalize_words("organise", "en-GB") == ["organize"]


# --- ASR aliases -------------------------------------------------------------------------------------------------

def _words(text, step=100):
    return [Caption(text=t, startMs=i * step, endMs=i * step + 90, timestampMs=i * step)
            for i, t in enumerate(text.split())]


def _texts(words):
    return [w.text for w in words]


ALIASES = {"Oleafly": ["oh leafly", "o leafly", "o'leafly"], "busywork": ["busy work"], "CV": ["c v"]}


def test_alias_replaces_a_one_word_mishearing():
    out = asr.apply_aliases(_words("Meet Oleefly today."), {"Oleafly": ["oleefly"]}, "Meet Oleafly today.")
    assert _texts(out) == ["Meet", "Oleafly", "today."]
    assert [(w.startMs, w.endMs) for w in out] == [(0, 90), (100, 190), (200, 290)]


def test_alias_matches_several_asr_words_and_keeps_the_clock_and_punctuation():
    words = _words("Oh, Leafly puts all of it here.")
    out = asr.apply_aliases(words, ALIASES, "Oleafly puts all of it here.")
    assert _texts(out) == ["Oleafly", "puts", "all", "of", "it", "here."]
    assert (out[0].startMs, out[0].endMs, out[0].timestampMs) == (0, 190, 0)         # first start to last end
    assert asr.wer("Oleafly puts all of it here.", " ".join(_texts(out))) == 0.0
    out = asr.apply_aliases(_words("Meet oh Leafly."), ALIASES, "Meet Oleafly.")
    assert _texts(out) == ["Meet", "Oleafly."]                                          # the full stop stays on the word


def test_alias_is_matched_on_normalised_tokens():
    out = asr.apply_aliases(_words("Search O\u2019Leafly now"), ALIASES, "Search Oleafly now")   # curly apostrophe
    assert _texts(out) == ["Search", "Oleafly", "now"]
    out = asr.apply_aliases(_words("OH LEAFLY!"), ALIASES, "Oleafly!")
    assert _texts(out) == ["Oleafly!"]
    out = asr.apply_aliases(_words("I'll take the busy work."), ALIASES, "I'll take the busywork.")
    assert _texts(out) == ["I'll", "take", "the", "busywork."]
    out = asr.apply_aliases(_words("the ATS friendly C V templates"), ALIASES, "the ATS-friendly CV templates")
    assert _texts(out) == ["the", "ATS", "friendly", "CV", "templates"]


def test_alias_keeps_the_longest_match_and_the_lowest_confidence():
    words = _words("oh leafly")
    words[0].confidence, words[1].confidence = 0.9, 0.4
    out = asr.apply_aliases(words, {"Oleafly": ["leafly", "oh leafly"]}, "Oleafly")
    assert _texts(out) == ["Oleafly"] and out[0].confidence == 0.4


@pytest.mark.parametrize("heard", ["Oh no leafly", "pro leafly", "oh and leafly", "leaflet", "oh leaflets",
                                   "Oh, so, Leafly"])
def test_alias_does_not_swallow_unrelated_words(heard):
    words = _words(heard)
    assert _texts(asr.apply_aliases(words, ALIASES, "Oleafly is here")) == _texts(words)


def test_alias_only_replaces_the_words_it_matched():
    out = asr.apply_aliases(_words("So oh Leafly now works well"), ALIASES, "So Oleafly now works well")
    assert _texts(out) == ["So", "Oleafly", "now", "works", "well"]


def test_alias_is_ignored_on_a_line_that_does_not_say_the_word():
    words = _words("Quick tips for your paper.")
    assert _texts(asr.apply_aliases(words, {"Typst": ["tips"]}, "Quick tips for your paper.")) == _texts(words)
    assert asr.active_aliases("Quick tips for your paper.", {"Typst": ["tips"]}) == {}
    assert asr.active_aliases("Write in Typst.", {"Typst": ["tips"], "CV": ["c v"]}) == {"Typst": ["tips"]}


def test_alias_does_not_rewrite_more_words_than_the_script_has():
    script = "Typst has tips."                                       # one Typst, and a real "tips"
    out = asr.apply_aliases(_words("tipst has tips."), {"Typst": ["tipst", "tips"]}, script)
    assert _texts(out) == ["Typst", "has", "tips."]
    out = asr.apply_aliases(_words("Typst has tips."), {"Typst": ["tipst", "tips"]}, script)   # already right
    assert _texts(out) == ["Typst", "has", "tips."]
    out = asr.apply_aliases(_words("Oleafly then oh Leafly"), ALIASES, "Oleafly then Oleafly")
    assert _texts(out) == ["Oleafly", "then", "Oleafly"]                                   # two in the script: two allowed


def test_no_aliases_or_no_words_change_nothing():
    words = _words("Oh Leafly")
    assert _texts(asr.apply_aliases(words, {}, "Oleafly")) == ["Oh", "Leafly"]
    assert asr.apply_aliases([], ALIASES, "Oleafly") == []
    assert _texts(asr.apply_aliases(words, {"Oleafly": [""]}, "Oleafly")) == ["Oh", "Leafly"]


def test_alias_prefers_the_replacement_that_lowers_the_edit_distance():
    # one Typst in the script; ASR wrote the real word "tips" twice, and the second one is the mishearing
    script = "Typst has tips for you."
    out = asr.apply_aliases(_words("tips has tips for you."), {"Typst": ["tips"]}, script)
    assert _texts(out) == ["Typst", "has", "tips", "for", "you."]
    out = asr.apply_aliases(_words("tipst has tips for you."), {"Typst": ["tips", "tipst"]}, script)
    assert _texts(out) == ["Typst", "has", "tips", "for", "you."]                  # the alias that is not a real word
    # a replacement that would not lower the distance is left alone
    out = asr.apply_aliases(_words("Typst has tips for you."), {"Typst": ["tips"]}, script)
    assert _texts(out) == ["Typst", "has", "tips", "for", "you."]


def test_alias_with_and_in_it_can_never_match():
    # normalisation drops "and", so "rock and roll" is the tokens rock + roll, and ASR's "and" word sits between them
    words = _words("rock and roll")
    assert _texts(asr.apply_aliases(words, {"R&R": ["rock and roll"]}, "R&R")) == ["rock", "and", "roll"]
    assert "and" not in asr.normalize_words("rock and roll")
    assert asr.normalize_words("rock and roll", keep_and=True) == ["rock", "and", "roll"]


# --- aligning the script's words to what ASR heard ---------------------------------------------------------------

def _heard(spec):
    out = []
    for part in spec.split():
        text, span = part.rsplit(":", 1)
        a, b = span.split("-")
        out.append(Caption(text=text, startMs=int(a), endMs=int(b), timestampMs=int(a), confidence=0.9))
    return out


def _aligned(script, spec, duration=4000):
    return [(w.text, w.startMs, w.endMs) for w in asr.align_to_script(script.split(), _heard(spec), duration)]


def test_align_gives_one_word_per_script_word_with_the_asr_clock():
    assert _aligned("Moving money safer.", "moving:0-300 money:300-600 safer:700-1000") == [
        ("Moving", 0, 300), ("money", 300, 600), ("safer.", 700, 1000)]


def test_align_hyphenated_script_word_runs_from_its_first_part_to_its_last():
    assert _aligned("stay read-only.", "stay:0-300 read:400-700 only:700-1000") == [("stay", 0, 300), ("read-only.", 400, 1000)]
    assert _aligned("stay read-only.", "stay:0-300 read-only:400-1000") == [("stay", 0, 300), ("read-only.", 400, 1000)]


def test_align_script_word_ASR_split_into_several_words_or_merged_into_one():
    assert _aligned("the ATS-friendly CV", "the:0-100 ATS:100-400 friendly:400-800 C:800-900 V:900-1000")[:2] == [
        ("the", 0, 100), ("ATS-friendly", 100, 800)]
    merged = _aligned("the chores: twenty-two converters,", "the:0-100 chores,22:100-1000 converters,:1000-1500")
    assert merged[1][1] == 100 and merged[1][2] == merged[2][1] and merged[2][2] == 1000   # the merged word is shared
    assert merged[3] == ("converters,", 1000, 1500)


def test_align_keeps_and_so_an_and_anchor_has_its_own_time():
    assert _aligned("colour, and twenty-one", "color:0-400 and:400-500 twenty:500-800 one:800-900") == [
        ("colour,", 0, 400), ("and", 400, 500), ("twenty-one", 500, 900)]


def test_align_a_dropped_script_word_shares_the_gap_and_a_misheard_one_keeps_the_heard_time():
    assert _aligned("Ask Oleafly then wait", "Ask:0-300 then:900-1100 wait:1100-1400") == [
        ("Ask", 0, 300), ("Oleafly", 300, 900), ("then", 900, 1100), ("wait", 1100, 1400)]
    assert _aligned("a repo, a zip", "a:0-100 ripple:100-500 a:600-700 zip:700-1000") == [
        ("a", 0, 100), ("repo,", 100, 500), ("a", 600, 700), ("zip", 700, 1000)]


def test_align_edges_and_the_empty_cases():
    assert _aligned("Hello there", "there:1000-1400", 3000) == [("Hello", 0, 1000), ("there", 1000, 1400)]
    assert _aligned("Hello there", "Hello:0-400", 3000) == [("Hello", 0, 400), ("there", 400, 3000)]
    words = asr.align_to_script(["Hello", "there", "friend"], [], 3000)                  # nothing heard: spread it
    assert [(w.text, w.startMs) for w in words] == [("Hello", 0), ("there", 938), ("friend", 1875)]
    assert words[-1].endMs == 3000
    assert asr.align_to_script([], _heard("a:0-100"), 1000) == []
    assert [(w.text, w.startMs, w.endMs) for w in asr.align_to_script(["Ask", "—", "wait"], _heard("Ask:0-300 wait:600-900"),
                                                                       900)] == [("Ask", 0, 300), ("—", 300, 600),
                                                                                 ("wait", 600, 900)]


def test_align_keeps_the_lowest_confidence_of_the_asr_words_a_script_word_covers():
    heard = _heard("read:0-300 only:300-600")
    heard[1].confidence = 0.4
    (word,) = asr.align_to_script(["read-only."], heard, 600)
    assert (word.text, word.startMs, word.endMs, word.confidence) == ("read-only.", 0, 600, 0.4)


def test_align_in_a_character_language_uses_the_same_tokens_as_wer():
    words = asr.align_to_script(["東京タワー"], _heard("東:0-100 京:100-200 タ:200-300 ワ:300-400 ー:400-500"), 500, "ja")
    assert [(w.text, w.startMs, w.endMs) for w in words] == [("東京タワー", 0, 500)]


def test_align_times_the_parts_of_a_hyphenated_word_from_what_asr_heard():
    (word,) = asr.align_to_script(["read-only."], _heard("read:100-600 only.:900-1000"), 1200)
    assert (word.startMs, word.endMs) == (100, 1000)
    assert [(p.text, p.startMs, p.endMs) for p in word.parts] == [("read", 100, 600), ("only.", 900, 1000)]
    words = asr.align_to_script("the ATS-friendly CV".split(), _heard("the:0-100 ATS:100-400 friendly:700-900 CV:900-1000"), 1000)
    assert [(p.text, p.startMs) for p in words[1].parts] == [("ATS", 100), ("friendly", 700)]
    assert words[0].parts == [] and words[2].parts == []                        # only a hyphenated word has parts


def test_align_parts_come_from_a_merged_asr_word_by_letters_and_are_dropped_when_a_part_was_not_heard():
    (word,) = asr.align_to_script(["twenty-two"], _heard("chores,22:0-900"), 900)       # tokens chores, twenty, two
    assert [p.text for p in word.parts] == ["twenty", "two"]
    assert word.parts[0].endMs <= word.parts[1].startMs + 1 and word.parts[1].endMs == 900
    (word,) = asr.align_to_script(["read-only."], _heard("read:100-600"), 1000)         # "only" was not heard at all
    assert word.parts == [] and (word.startMs, word.endMs) == (100, 600)


def test_a_word_serialises_without_parts_exactly_as_before():
    plain = Caption(text="a", startMs=0, endMs=1)
    assert "parts" not in plain.model_dump() and "parts" not in plain.model_dump_json()
    with_parts = asr.align_to_script(["read-only."], _heard("read:0-300 only.:300-600"), 600)[0]
    again = Caption.model_validate_json(with_parts.model_dump_json())
    assert again == with_parts and len(again.parts) == 2


def test_align_clamps_every_time_to_the_take():
    heard = _heard("Hello:0-300 read:400-700 only:700-5000 world:5000-6000")
    heard[0] = heard[0].model_copy(update={"startMs": -200})                           # ASR times can fall outside the take
    words = asr.align_to_script(["Hello", "read-only", "world"], heard, 1000)
    assert all(0 <= w.startMs <= w.endMs <= 1000 for w in words)
    assert all(0 <= p.startMs <= p.endMs <= 1000 for w in words for p in w.parts)
    assert [(w.startMs, w.endMs) for w in words] == [(0, 300), (400, 1000), (1000, 1000)]
    assert [w.timestampMs for w in words] == [w.startMs for w in words]
    spread = asr.align_to_script(["a", "b"], [], 500)
    assert spread[-1].endMs == 500
    assert [(w.startMs, w.endMs) for w in asr.align_to_script(["a"], _heard("a:900-1200"), 0)] == [(0, 0)]


@pytest.mark.live
@pytest.mark.skipif(shutil.which("say") is None, reason="macOS `say` needed")
def test_parakeet_live_on_system_voice(tmp_path):
    aiff, wav = tmp_path / "s.aiff", tmp_path / "s.wav"
    subprocess.run(["say", "-o", str(aiff), "Moving company money just got safer."], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(wav)], check=True)
    words = asr.transcribe(wav, "en")
    hyp = " ".join(w.text for w in words)
    assert asr.wer("Moving company money just got safer.", hyp) <= 0.2
    assert all(w.endMs >= w.startMs for w in words)
