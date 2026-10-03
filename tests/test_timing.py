import json
from pathlib import Path

import pytest

from promo.models import BeatGrid, Caption, LineTake
from promo.packs import Limits
from promo.shotlist import Shotlist, load_shotlist
from promo.timing import Measured, TimingError, load_timing, resolve, write_timing

FIX = Path(__file__).parent / "fixtures" / "shotlist_music.json"
LIM = Limits()  # defaults: head .3, tail .4, gap .3, pace 170, dead 4, hook 2, min read .8


def _sl(mutate=None) -> Shotlist:
    data = json.loads(FIX.read_text())
    if mutate:
        mutate(data)
    return Shotlist.model_validate(data)


def _codes(t, level=None):
    return [f.code for f in t.findings if level is None or f.level == level]


def _take_c01() -> LineTake:
    words = [("Moving", 0, 280), ("company", 280, 620), ("money", 620, 880),
             ("just", 880, 1000), ("got", 1000, 1120), ("safer.", 1120, 1200)]
    return LineTake(id="c01-l1", file="audio/vo/c01-l1.wav", duration_s=1.2,
                    words=[Caption(text=w, startMs=a, endMs=b) for w, a, b in words])


def test_music_mode_chapters_lines_shots_events():
    t = resolve(_sl(), Measured(takes={"c01-l1": _take_c01()}), LIM)
    assert [(c.id, c.start_s, c.end_s) for c in t.chapters] == [("c01", 0.3, 4.3), ("c02", 4.3, 8.3)]
    assert t.duration_s == 8.3
    l1 = t.line("c01-l1")
    assert l1.start_s == pytest.approx(0.8) and l1.end_s == pytest.approx(2.0)
    assert l1.words[5].startMs == 1920  # 800 + 1120
    shots = {s.id: (s.start_s, s.end_s) for s in t.shots}
    assert shots == {"c01-s1": (0.0, 4.3), "c02-s1": (4.3, 6.3), "c02-s2": (6.3, 8.3)}
    ev = {e.id: e for e in t.events}
    assert ev["c01-e1"].t == pytest.approx(0.3)
    assert ev["c01-e2"].t == pytest.approx(0.8 + 1.12 - 0.03)          # word anchor
    assert ev["c02-e1"].t == pytest.approx(4.55)                        # beat 0.5 in c02
    assert ev["c02-e2"].t == pytest.approx(6.7) and ev["c02-e2"].t_local == pytest.approx(0.4)
    assert t.errors() == []


def test_after_anchor_and_estimated_lines_warn():
    t = resolve(_sl(), Measured(), LIM)
    l1, l2 = t.line("c02-l1"), t.line("c02-l2")
    assert l1.estimated and l2.estimated
    assert l2.start_s == pytest.approx(l1.end_s + 0.2)
    assert _codes(t, "warning").count("no_take") == 3


def test_measured_grid_moves_the_origin():
    beats = [round(0.1 + 0.5 * i, 3) for i in range(40)]
    grid = BeatGrid(bpm=120, beats=beats, downbeats=beats[1::4], first_downbeat_s=0.6)
    t = resolve(_sl(), Measured(grid=grid), LIM)
    assert t.chapters[0].start_s == pytest.approx(0.9)   # 0.6 + preroll 0.3
    assert t.chapters[0].end_s == pytest.approx(4.9)


def test_voice_mode_and_beat_snap():
    data = {
        "version": 2, "pack": "saas-kinetic", "timing_mode": "voice", "preroll_s": 0.3,
        "music": {"bpm": 100},
        "chapters": [{"id": "c1", "vo": [{"id": "l1", "text": "one two"}, {"id": "l2", "text": "three four five"}],
                      "shots": [{"id": "s1", "events": [{"id": "e1", "kind": "headline", "at": {"t": 0.0}}]}]}],
    }
    sl = Shotlist.model_validate(data)
    takes = {"l1": LineTake(id="l1", file="a.wav", duration_s=1.0), "l2": LineTake(id="l2", file="b.wav", duration_s=1.5)}
    t = resolve(sl, Measured(takes=takes), LIM)
    assert t.line("l1").start_s == pytest.approx(0.6)
    assert t.line("l2").start_s == pytest.approx(1.9)
    # vo end 3.4 + tail 0.4 = 3.8 → next beat at 100 bpm from origin 0.3 is 3.9
    assert t.chapters[0].end_s == pytest.approx(3.9)
    assert "t_anchor" in _codes(t, "warning")


def test_voice_mode_without_vo_or_seconds_cannot_resolve():
    data = {"version": 2, "pack": "saas-kinetic", "timing_mode": "voice",
            "chapters": [{"id": "c1", "shots": [{"id": "s1"}]}]}
    with pytest.raises(TimingError, match="c1 has no VO"):
        resolve(Shotlist.model_validate(data), Measured(), LIM)


def test_vo_overrun_is_error():
    long = LineTake(id="c01-l1", file="x.wav", duration_s=3.5)
    t = resolve(_sl(), Measured(takes={"c01-l1": long}), LIM)
    errs = [f for f in t.errors() if f.code == "vo_overrun"]
    assert errs and errs[0].ref == "c01-l1"


def test_word_not_found_is_error():
    def m(d):
        d["chapters"][0]["shots"][0]["events"][1]["at"]["word"] = "banana"
    t = resolve(_sl(m), Measured(takes={"c01-l1": _take_c01()}), LIM)
    assert "word_not_found" in _codes(t, "error")


def test_event_outside_its_shot():
    def m(d):
        d["chapters"][1]["shots"][0]["events"][0]["at"] = {"beat": 5}
    t = resolve(_sl(m), Measured(), LIM)
    assert "event_outside_shot" in _codes(t, "error")


def test_dead_time_and_late_hook():
    def m(d):
        d["chapters"][1]["shots"][0]["events"] = []
        d["chapters"][1]["shots"][1]["events"] = []
        d["chapters"][0]["shots"][0]["events"][0]["at"] = {"beat": 6}
    t = resolve(_sl(m), Measured(takes={"c01-l1": _take_c01()}), LIM)
    codes = _codes(t, "error")
    assert "dead_time" in codes      # from the last event to 8.3 s
    assert "late_hook" not in codes  # the word event at 1.89 s is still inside the hook


def test_late_hook():
    def m(d):
        d["chapters"][0]["shots"][0]["events"] = [{"id": "c01-e1", "kind": "headline", "at": {"beat": 6}}]
    t = resolve(_sl(m), Measured(), LIM)
    assert "late_hook" in _codes(t, "error")


def test_short_and_overlapping_reads():
    def m(d):
        d["chapters"][0]["shots"][0]["reads"] = [
            {"what": "a", "from": {"beat": 2}, "to": {"beat": 3}},
            {"what": "b", "from": {"beat": 2.5}, "to": {"beat": 5}},
        ]
    t = resolve(_sl(m), Measured(), LIM)
    codes = _codes(t, "error")
    assert "read_too_short" in codes and "reads_overlap" in codes


def test_lines_overlap():
    def m(d):
        d["chapters"][1]["vo"][1]["at"] = {"beat": 0.5}
    t = resolve(_sl(m), Measured(), LIM)
    assert "lines_overlap" in _codes(t, "error")


def test_timing_round_trip(tmp_path):
    t = resolve(_sl(), Measured(takes={"c01-l1": _take_c01()}), LIM)
    p = tmp_path / "timing.json"
    write_timing(t, p)
    back = load_timing(p)
    assert back == t
    assert back.beats[0] == pytest.approx(0.3) and back.beats[1] == pytest.approx(0.8)


def _two_shot_chapter(bpm: float, fps: int) -> Shotlist:
    """One bar; the second shot starts on beat 1 and has an event on that very beat."""
    return Shotlist.model_validate({
        "version": 2, "pack": "saas-kinetic", "canvas": {"fps": fps}, "preroll_s": 0.3, "music": {"bpm": bpm},
        "chapters": [{"id": "c1", "bars": 1, "shots": [
            {"id": "s1", "beats": 1, "events": [{"id": "e1", "kind": "headline", "at": {"beat": 0}}]},
            {"id": "s2", "events": [{"id": "e2", "kind": "tick", "at": {"beat": 1}}]},
        ]}],
    })


def test_event_on_shot_start_at_non_frame_aligned_tempo():
    # 92 bpm at 30 fps: beat 1 is 0.952174 s but the shot boundary rounds up to frame 29 (0.966667 s).
    t = resolve(_two_shot_chapter(92, 30), Measured(), LIM)
    assert t.findings == []
    assert t.shot("s2").start_s == pytest.approx(29 / 30)
    e2 = next(e for e in t.events if e.id == "e2")
    assert e2.t == pytest.approx(0.3 + 60 / 92, abs=1e-4)  # the exact time (4 dp), not the frame, is kept
    assert e2.t_local == 0.0


@pytest.mark.parametrize("fps", [30, 60])
def test_event_on_shot_start_for_every_integer_tempo(fps):
    for bpm in range(60, 181):
        t = resolve(_two_shot_chapter(bpm, fps), Measured(), LIM)
        assert "event_outside_shot" not in _codes(t), (bpm, fps)
        assert all(e.t_local >= 0 for e in t.events), (bpm, fps)


def test_voice_shot_seconds_not_on_a_frame_boundary():
    data = {"version": 2, "pack": "saas-kinetic", "timing_mode": "voice", "canvas": {"fps": 30}, "preroll_s": 0.3,
            "chapters": [{"id": "c1", "shots": [
                {"id": "s1", "seconds": 2.25, "events": [{"id": "e1", "kind": "headline", "at": {"t": 0.0}}]}]}]}
    t = resolve(Shotlist.model_validate(data), Measured(), LIM)  # 0.3 + 2.25 = 2.55 s is half a frame
    c1 = t.chapter("c1")
    assert c1.end_s >= c1.start_s + 2.25 - 1e-6
    assert t.shot("s1").end_s == c1.end_s
    assert t.errors() == []


def test_voice_chapter_start_off_the_frame_grid_is_absorbed_by_the_last_shot():
    # preroll 0.45 s is 13.5 frames at 30 fps, so the chapter start rounds up by half a frame.
    data = {"version": 2, "pack": "saas-kinetic", "timing_mode": "voice", "canvas": {"fps": 30}, "preroll_s": 0.45,
            "chapters": [{"id": "c1", "shots": [
                {"id": "s1", "seconds": 1.0, "events": [{"id": "e1", "kind": "headline", "at": {"t": 0.0}}]},
                {"id": "s2", "seconds": 1.25}]}]}
    t = resolve(Shotlist.model_validate(data), Measured(), LIM)
    c1 = t.chapter("c1")
    assert t.shot("s2").end_s == c1.end_s
    assert t.shot("s2").duration_s == pytest.approx(1.25, abs=1 / 30)
    assert t.errors() == []


@pytest.mark.parametrize("bpm", range(60, 181, 7))
def test_voice_later_chapters_at_non_frame_aligned_tempo(bpm):
    data = {"version": 2, "pack": "saas-kinetic", "timing_mode": "voice", "canvas": {"fps": 30}, "preroll_s": 0.3,
            "music": {"bpm": bpm},
            "chapters": [{"id": f"c{i}", "shots": [
                {"id": f"s{i}", "seconds": 2.25, "events": [{"id": f"e{i}", "kind": "tick", "at": {"t": 0.0}}]}]}
                for i in (1, 2, 3)]}
    t = resolve(Shotlist.model_validate(data), Measured(), LIM)
    assert "event_outside_shot" not in _codes(t), bpm
    assert all(e.t_local >= 0 for e in t.events), bpm
    assert [s.end_s for s in t.shots[:-1]] == [s.start_s for s in t.shots[1:]]
    for c in t.chapters:
        assert c.end_s - c.start_s >= 2.25 - 1e-6, bpm


def _anchor_t(words, anchor_word, nth=1):
    take = LineTake(id="c01-l1", file="audio/vo/c01-l1.wav", duration_s=3.0,
                    words=[Caption(text=w, startMs=a, endMs=b) for w, a, b in words])

    def mutate(d):
        d["chapters"][0]["vo"][0]["text"] = " ".join(w for w, _, _ in words)   # the take is of this line's copy
        d["chapters"][0]["shots"][0]["events"][1]["at"].update(word=anchor_word, nth=nth, offset=0.0)

    t = resolve(_sl(mutate), Measured(takes={"c01-l1": take}), LIM)
    ev = {e.id: e for e in t.events}["c01-e2"]
    return t, round(ev.t - t.line("c01-l1").start_s, 3)


def test_word_anchor_names_a_hyphenated_word_or_a_part_of_it():
    words = [("They", 0, 300), ("stay", 300, 700), ("read-only.", 800, 1600)]
    for anchor, expected in (("read-only", 0.8), ("read-only.", 0.8), ("read", 0.8), ("only", 1.2)):
        t, at = _anchor_t(words, anchor)
        assert at == expected and "word_not_found" not in _codes(t), anchor


def test_word_anchor_counts_hyphen_parts_as_words_in_time_order():
    words = [("well-known", 0, 1000), ("and", 1000, 1200), ("well", 1200, 1600)]
    assert _anchor_t(words, "well", 1)[1] == 0.0
    assert _anchor_t(words, "well", 2)[1] == 1.2                       # the second "well" is the plain word
    assert "word_not_found" in _codes(_anchor_t(words, "well", 3)[0], "error")
    assert _anchor_t(words, "known")[1] == pytest.approx(4 / 9, abs=0.001)


def test_word_anchor_on_a_part_fires_where_asr_timed_it_and_falls_back_to_letters():
    from promo.models import WordPart

    word = Caption(text="ATS-friendly", startMs=0, endMs=1100,
                   parts=[WordPart(text="ATS", startMs=0, endMs=300), WordPart(text="friendly", startMs=500, endMs=1100)])
    take = LineTake(id="c01-l1", file="audio/vo/c01-l1.wav", duration_s=1.2, words=[word])
    assert _anchor_with(take, "friendly") == 0.5                          # ASR's time, not 3/11 of the way (0.3 s)
    assert _anchor_with(take.model_copy(update={"words": [word.model_copy(update={"parts": []})]}),
                        "friendly") == pytest.approx(0.3, abs=0.001)         # no part times: by letters


def test_timed_line_words_carry_their_parts_on_the_film_clock():
    from promo.models import WordPart

    word = Caption(text="read-only.", startMs=100, endMs=900,
                   parts=[WordPart(text="read", startMs=100, endMs=500), WordPart(text="only.", startMs=500, endMs=900)])
    def mutate(d):
        d["chapters"][0]["vo"][0]["text"] = "read-only."   # the take is of this line's copy

    t = resolve(_sl(mutate), Measured(takes={"c01-l1": LineTake(id="c01-l1", file="f.wav", duration_s=1.0, words=[word])}), LIM)
    line = t.line("c01-l1")
    assert [(p.text, p.startMs - round(line.start_s * 1000)) for p in line.words[0].parts] == [("read", 100), ("only.", 500)]
    assert load_timing_roundtrip(t).line("c01-l1").words[0].parts[1].startMs == line.words[0].parts[1].startMs


def load_timing_roundtrip(t):
    from promo.timing import Timing

    return Timing.model_validate_json(t.model_dump_json())


def _anchor_with(take, anchor_word):
    def mutate(d):
        d["chapters"][0]["vo"][0]["text"] = " ".join(w.text for w in take.words)   # the take is of this line's copy
        d["chapters"][0]["shots"][0]["events"][1]["at"].update(word=anchor_word, offset=0.0)

    t = resolve(_sl(mutate), Measured(takes={"c01-l1": take}), LIM)
    return round({e.id: e for e in t.events}["c01-e2"].t - t.line("c01-l1").start_s, 3)


def test_a_take_with_a_problem_is_no_take_at_all():
    bad = LineTake(id="c01-l1", file="audio/vo/c01-l1.wav", duration_s=0.0, wer=1.0, problem="every take came back silent")
    t = resolve(_sl(), Measured(takes={"c01-l1": bad}), LIM)
    line = t.line("c01-l1")
    assert line.estimated and line.file is None and line.end_s - line.start_s > 1.0       # a real, estimated length
    assert len(line.words) == 6 and line.words[-1].endMs > line.words[0].startMs
    warn = [f for f in t.findings if f.code == "no_take" and f.ref == "c01-l1"]
    assert len(warn) == 1 and "every take came back silent" in warn[0].message and "estimated" in warn[0].message


def test_a_take_of_older_copy_is_no_take_at_all():
    # the script line changed after it was voiced: the take on disk says the old words, so its audio and word times
    # are not this line's (an anchor on a new word would not resolve, and the old copy would play)
    words = [("Moving", 0, 280), ("money", 280, 620), ("got", 620, 1000), ("safer.", 1000, 1200)]
    old = LineTake(id="c01-l1", file="audio/vo/c01-l1.wav", duration_s=1.2,
                   words=[Caption(text=w, startMs=a, endMs=b) for w, a, b in words])
    t = resolve(_sl(), Measured(takes={"c01-l1": old}), LIM)
    line = t.line("c01-l1")
    assert line.estimated and line.file is None
    assert [w.text for w in line.words] == "Moving company money just got safer.".split()
    warn = [f for f in t.findings if f.code == "stale_take" and f.ref == "c01-l1"]
    assert len(warn) == 1 and "estimated" in warn[0].message


def test_a_take_without_words_is_still_used():
    t = resolve(_sl(), Measured(takes={"c01-l1": LineTake(id="c01-l1", file="a.wav", duration_s=1.0)}), LIM)
    assert not t.line("c01-l1").estimated and "stale_take" not in _codes(t)


def test_load_measured_keeps_the_record_but_timing_does_not_use_it(tmp_path):
    from promo.timing import load_measured

    vo = tmp_path / "audio" / "vo"
    vo.mkdir(parents=True)
    bad = LineTake(id="c01-l1", file="audio/vo/c01-l1.wav", duration_s=0.0, wer=1.0, problem="every take came back silent")
    (vo / "c01-l1.json").write_text(bad.model_dump_json())
    measured = load_measured(tmp_path)
    assert measured.takes["c01-l1"].problem
    assert resolve(_sl(), measured, LIM).line("c01-l1").estimated
