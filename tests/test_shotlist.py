import copy
import json
from pathlib import Path

import pytest

from promo.shotlist import (
    BeatAnchor,
    ShotAnchor,
    ShotlistError,
    WordAnchor,
    load_shotlist,
)

FIX = Path(__file__).parent / "fixtures" / "shotlist_music.json"


def _write(tmp_path, data) -> Path:
    p = tmp_path / "shotlist.json"
    p.write_text(json.dumps(data))
    return p


def test_fixture_loads_and_anchors_are_typed():
    sl = load_shotlist(FIX)
    assert sl.timing_mode == "music"
    line = sl.chapters[0].vo[0]
    assert isinstance(line.at, BeatAnchor) and line.at.beat == 1
    ev = sl.chapters[0].shots[0].events[1]
    assert isinstance(ev.at, WordAnchor) and ev.at.word == "safer" and ev.at.offset == -0.03
    assert isinstance(sl.chapters[1].shots[1].events[0].at, ShotAnchor)
    read = sl.chapters[0].shots[0].reads[0]
    assert isinstance(read.from_, BeatAnchor)


def test_helpers():
    sl = load_shotlist(FIX)
    assert [l.id for l in sl.all_lines()] == ["c01-l1", "c02-l1", "c02-l2"]
    assert [s.id for s in sl.all_shots()] == ["c01-s1", "c02-s1", "c02-s2"]
    assert sl.chapter_of_shot("c02-s2").id == "c02"
    assert sl.chapter_of_line("c01-l1").id == "c01"


def test_duplicate_ids_rejected(tmp_path):
    data = json.loads(FIX.read_text())
    data["chapters"][1]["shots"][1]["id"] = "c01-s1"
    with pytest.raises(ShotlistError, match="duplicate id 'c01-s1'"):
        load_shotlist(_write(tmp_path, data))


def test_unknown_claim_rejected(tmp_path):
    data = json.loads(FIX.read_text())
    data["chapters"][1]["vo"][0]["claims"] = ["k9"]
    with pytest.raises(ShotlistError, match="unknown claim 'k9'"):
        load_shotlist(_write(tmp_path, data))


def test_word_anchor_to_unknown_line_rejected(tmp_path):
    data = json.loads(FIX.read_text())
    data["chapters"][0]["shots"][0]["events"][1]["at"]["line"] = "c09-l1"
    with pytest.raises(ShotlistError, match="unknown line 'c09-l1'"):
        load_shotlist(_write(tmp_path, data))


def test_music_mode_requires_bars(tmp_path):
    data = json.loads(FIX.read_text())
    del data["chapters"][1]["bars"]
    with pytest.raises(ShotlistError, match="c02 needs 'bars' in music mode"):
        load_shotlist(_write(tmp_path, data))


@pytest.mark.parametrize("provider", ["elevenlabs", "lyria"])
def test_music_provider_is_optional_and_named(tmp_path, provider):
    data = json.loads(FIX.read_text())
    assert load_shotlist(_write(tmp_path, data)).music.provider is None
    data["music"]["provider"] = provider
    assert load_shotlist(_write(tmp_path, data)).music.provider == provider


def test_an_unknown_music_provider_is_rejected(tmp_path):
    data = json.loads(FIX.read_text())
    data["music"]["provider"] = "suno"
    with pytest.raises(ShotlistError, match="music.provider"):
        load_shotlist(_write(tmp_path, data))


def test_music_style_is_optional_and_overrides_named_lines(tmp_path):
    data = json.loads(FIX.read_text())
    assert load_shotlist(_write(tmp_path, data)).music.style is None
    data["music"]["style"] = {"mood": "playful, bouncy, warm"}
    style = load_shotlist(_write(tmp_path, data)).music.style
    assert style is not None and style.mood == "playful, bouncy, warm"
    assert style.instrumentation is None and style.arrangement is None and style.negative is None
    data["music"]["style"] = {"mood": "m", "instrumentation": "i", "arrangement": "a", "negative": "n"}
    assert load_shotlist(_write(tmp_path, data)).music.style.model_dump() == {
        "mood": "m", "instrumentation": "i", "arrangement": "a", "negative": "n"}


def test_an_unknown_music_style_key_is_rejected(tmp_path):
    data = json.loads(FIX.read_text())
    data["music"]["style"] = {"mood": "m", "tempo": "fast"}
    with pytest.raises(ShotlistError, match="music.style.tempo"):
        load_shotlist(_write(tmp_path, data))


def test_bad_anchor_shape_is_readable(tmp_path):
    data = json.loads(FIX.read_text())
    data["chapters"][0]["vo"][0]["at"] = {"beats": 1}
    with pytest.raises(ShotlistError) as err:
        load_shotlist(_write(tmp_path, data))
    assert "chapters.0.vo.0.at" in str(err.value)


def test_chapter_needs_a_shot(tmp_path):
    data = json.loads(FIX.read_text())
    data["chapters"][0]["shots"] = []
    with pytest.raises(ShotlistError, match="shots"):
        load_shotlist(_write(tmp_path, data))


def test_vo_line_verifies_by_default_and_can_opt_out(tmp_path):
    data = json.loads(FIX.read_text())
    assert [l.verify for l in load_shotlist(_write(tmp_path, data)).all_lines()] == [True, True, True]
    data["chapters"][0]["vo"][0]["verify"] = False
    assert [l.verify for l in load_shotlist(_write(tmp_path, data)).all_lines()] == [False, True, True]


def test_verify_must_be_a_boolean(tmp_path):
    data = json.loads(FIX.read_text())
    data["chapters"][0]["vo"][0]["verify"] = "sometimes"
    with pytest.raises(ShotlistError, match="chapters.0.vo.0.verify"):
        load_shotlist(_write(tmp_path, data))


def test_voice_block_takes_a_provider_and_its_options(tmp_path):
    data = json.loads(FIX.read_text())
    data["voice"] = {"provider": "supertonic", "name": "F2", "model": "supertonic-2", "steps": 12, "speed": 1.25,
                     "silence_s": 0.3}
    v = load_shotlist(_write(tmp_path, data)).voice
    assert (v.provider, v.name, v.steps, v.speed, v.silence_s) == ("supertonic", "F2", 12, 1.25, 0.3)
    data["voice"] = {"name": "Kore"}
    v = load_shotlist(_write(tmp_path, data)).voice
    assert v.provider is None and v.steps is None                      # an unset provider falls back to the pack's


@pytest.mark.parametrize("voice, where", [
    ({"provider": "elevenlabs"}, "voice.provider"),
    ({"provider": "supertonic", "speed": 2.5}, "voice.speed"),
    ({"provider": "supertonic", "speed": 0.5}, "voice.speed"),
    ({"provider": "supertonic", "steps": 0}, "voice.steps"),
    ({"provider": "supertonic", "silence_s": -1}, "voice.silence_s"),
    ({"provider": "gemini", "steps": 8}, "Supertonic options"),
])
def test_bad_voice_block_is_readable(tmp_path, voice, where):
    data = json.loads(FIX.read_text())
    data["voice"] = voice
    with pytest.raises(ShotlistError, match=where):
        load_shotlist(_write(tmp_path, data))


def test_asr_aliases_map_a_word_to_what_asr_writes(tmp_path):
    data = json.loads(FIX.read_text())
    assert load_shotlist(_write(tmp_path, data)).asr_aliases == {}
    data["asr_aliases"] = {"Oleafly": ["oh leafly", "o'leafly"], "CV": ["c v"]}
    assert load_shotlist(_write(tmp_path, data)).asr_aliases == data["asr_aliases"]


@pytest.mark.parametrize("aliases", [{"Oleafly": []}, {"Oleafly": [""]}, {"": ["oh leafly"]}, {"Oleafly": "oh leafly"}])
def test_bad_asr_aliases_are_readable(tmp_path, aliases):
    data = json.loads(FIX.read_text())
    data["asr_aliases"] = aliases
    with pytest.raises(ShotlistError, match="asr_aliases"):
        load_shotlist(_write(tmp_path, data))


@pytest.mark.parametrize("value", [{"Typst": "tips"}, {"Typst": []}, {"Typst": [" "]}, {"  ": ["tips"]}, ["Typst"], "Typst"])
def test_check_aliases_rejects_what_the_shot_list_rejects(value):
    from promo.shotlist import check_aliases

    with pytest.raises(ValueError, match="asr_aliases"):
        check_aliases(value)
    assert check_aliases({"Typst": ["tips", "tipst"]}) == {"Typst": ["tips", "tipst"]}
    assert check_aliases({}) == {}
