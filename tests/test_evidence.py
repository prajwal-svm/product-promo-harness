import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from audio_helpers import SR, mix as _mix, say as _say, voice as _voice, with_audio as _with_audio
from promo import evidence as ev
from promo import sfx as sfx_mod
from promo import sfx_synth
from promo.models import Caption, LineTake
from promo.packs import Limits, load_pack
from promo.sfx import Placement, local_peak_s, render_stem
from promo.shotlist import Shotlist
from promo.timing import Measured, TimedLine, resolve

FIX = Path(__file__).parent / "fixtures" / "shotlist_music.json"
MASCOT = Path(__file__).resolve().parents[1] / "examples" / "promo" / "oleafly-mascot"


def _fake_key_applies(node) -> bool:
    """Live tests must read the real key from .env, so they never get the fake one."""
    return node.get_closest_marker("live") is None


@pytest.fixture(autouse=True)
def keys(request, monkeypatch):
    if _fake_key_applies(request.node):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")
        monkeypatch.setenv("GEMINI_API_KEY", "gm-test")


def test_fake_key_skips_live_tests():
    class Node:
        def __init__(self, marker):
            self.marker = marker

        def get_closest_marker(self, name):
            return self.marker if name == "live" else None

    assert _fake_key_applies(Node(None)) is True
    assert _fake_key_applies(Node(object())) is False


def _video(path, src="testsrc=size=320x180:rate=30:duration=4", audio="sine=duration=4:sample_rate=48000"):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", src, "-f", "lavfi", "-i", audio,
                    "-pix_fmt", "yuv420p", "-shortest", str(path)], check=True)
    return path


def test_sheets(tmp_path):
    v = _video(tmp_path / "v.mp4")
    items = ev.frames_at(v, [0.5, 1.0, 1.5], tmp_path / "f")
    sheet = ev.contact_sheet([(f"{t:.2f}s", p) for t, p in items], tmp_path / "c.png", cols=2)
    assert Image.open(sheet).size[0] > 480
    s = ev.strip(v, 2.0, 30, tmp_path / "s.png")
    assert Image.open(s).size[0] >= 12 * 320
    assert Image.open(ev.phone_sheet(v, tmp_path / "p.png")).size[0] >= 360
    assert ev.master_sheet(v, tmp_path / "m.png").exists()
    assert 0.0 <= ev.fastest_action_t(v) <= 4.0


def test_tech_checks_flags_black_and_freeze(tmp_path):
    ok = ev.tech_checks(_video(tmp_path / "ok.mp4"), width=320, height=180, fps=30, max_static_s=4.0)
    assert ok["ok"] is True
    black = ev.tech_checks(_video(tmp_path / "b.mp4", src="color=c=black:s=320x180:r=30:d=4"),
                           width=320, height=180, fps=30, max_static_s=2.0)
    assert black["ok"] is False
    assert any("black" in i for i in black["issues"]) and any("frozen" in i for i in black["issues"])
    wrong = ev.tech_checks(_video(tmp_path / "w.mp4"), width=1920, height=1080, fps=60, max_static_s=4.0)
    assert any("canvas" in i for i in wrong["issues"])


def _timing():
    words = [Caption(text=w, startMs=i * 180, endMs=i * 180 + 160) for i, w in
             enumerate("Moving company money just got safer.".split())]
    takes = {"c01-l1": LineTake(id="c01-l1", file="a.wav", duration_s=1.2, words=words)}
    data = json.loads(FIX.read_text())
    data["canvas"] = {"aspect": "16:9", "width": 320, "height": 180, "fps": 30}
    return resolve(Shotlist.model_validate(data), Measured(takes=takes), Limits())


# --- sound helpers -----------------------------------------------------------------------------------------------------

def _film_with_voice(tmp_path, *, shift_s=0.0, with_takes=True):
    """A timing with a take for every line (audio on disk, words on the take's clock), the stem for its events, and a
    master that holds the takes at their planned starts (+ shift_s), the stem and a bed."""
    data = json.loads(FIX.read_text())
    data["canvas"] = {"aspect": "16:9", "width": 320, "height": 180, "fps": 30}
    shotlist = Shotlist.model_validate(data)
    takes, files = {}, {}
    for k, line in enumerate(shotlist.all_lines()):
        words = _say(line.text)
        length = len(words) * 0.3
        files[line.id] = tmp_path / f"{line.id}.wav"
        sf.write(files[line.id], _voice(length, 10 + k), SR, subtype="FLOAT")
        takes[line.id] = LineTake(id=line.id, file=f"{line.id}.wav", duration_s=length, words=words)
    t = resolve(shotlist, Measured(takes=takes if with_takes else {}), Limits())
    stem, placements, _ = render_stem(t, load_pack("saas-kinetic"), tmp_path / "track.wav")
    stem_audio = sf.read(stem, dtype="float32")[0]
    parts = [(stem_audio, 0.0, 1.0)]
    if with_takes:
        parts += [(sf.read(files[l.id], dtype="float32")[0], l.start_s + shift_s, 0.5) for l in t.lines]
    video = _with_audio(_mix(tmp_path / "mix.wav", parts), tmp_path / "v.mp4")
    return t, stem, placements, video, files


def _by_line(t):
    """A recogniser that hears each line's own text, whichever window it is asked about (windows are named by line)."""
    text = {l.id: l.text for l in t.lines}
    return lambda path, lang: _say(text[Path(path).stem])


def _clips(t):
    pack = load_pack("saas-kinetic")
    return {e.id: sfx_mod.event_clip(e, pack) for e in t.events}


def test_sync_report(tmp_path):
    t, stem, placements, v, files = _film_with_voice(tmp_path)
    takes = files
    rep = ev.sync_report(t, stem, placements, v, "en", transcribe=_by_line(t), takes=takes, clips=_clips(t))
    assert rep["sfx"]["ok"] and rep["vo"]["ok"] and rep["lines"]["ok"]
    # each take is found by correlation, to within the AAC encoder's own few-ms delay, not by recognition timing
    assert all(abs(r["offset_ms"]) < 10 and r["corr"] > 0.6 for r in rep["vo"]["rows"])      # under a bed, the stem and AAC
    assert [r["line"] for r in rep["vo"]["rows"]] == [l.id for l in t.lines]
    # a take that sits 50 ms late (more than a frame at 30 fps) is late, and by how much
    (tmp_path / "late").mkdir()
    late_t, late_stem, late_p, late_v, late_files = _film_with_voice(tmp_path / "late", shift_s=0.05)
    late = ev.sync_report(late_t, late_stem, late_p, late_v, "en", transcribe=_by_line(late_t), takes=late_files,
                          clips=_clips(late_t))
    assert late["vo"]["ok"] is False
    assert [round(r["offset_ms"]) for r in late["vo"]["rows"]] == pytest.approx([50, 50, 50], abs=10)
    # recognition timing is information only: words heard 150 ms late do not fail the sync
    lagged = ev.sync_report(t, stem, placements, v, "en", takes=takes, clips=_clips(t),
                            transcribe=lambda p, lang: _say({l.id: l.text for l in t.lines}[Path(p).stem], lag_ms=150))
    assert lagged["vo"]["ok"] is True and lagged["vo"]["asr_offsets_ms"]["p95"] == pytest.approx(150, abs=1)
    shifted = [p.model_copy(update={"t": p.t + 0.1}) for p in placements]
    assert ev.sync_report(t, stem, shifted, v, "en", transcribe=_by_line(t), takes=takes, clips=_clips(t))["sfx"]["ok"] is False


# --- frames land on the frame they are labelled with ---------------------------------------------------------------

def _red_then_blue(path):
    """30 fps, one second of red then one of blue: the cut is between frame 29 and frame 30, at 1.0 s."""
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=64x64:r=30:d=1",
                    "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=30:d=1", "-filter_complex",
                    "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]", "-pix_fmt", "yuv420p", "-g", "5", str(path)],
                   check=True)
    return path


def _cell_colours(sheet, n, cell=64, pad=4):
    im = Image.open(sheet).convert("RGB")
    return ["red" if (px := im.getpixel((k * (cell + pad) + cell // 2, cell - 8)))[0] > px[2] else "blue"
            for k in range(n)]


def test_strip_cells_show_the_frame_they_are_labelled_with(tmp_path):
    v = _red_then_blue(tmp_path / "cut.mp4")
    # centre 1.0 s is the first blue frame; 12 cells run from 6 frames before it to 5 after, and every label time
    # is rounded to 3 decimals (0.9667 -> 0.967), which must not push a cell on to the next frame
    cells = _cell_colours(ev.strip(v, 1.0, 30, tmp_path / "s.png", width=64), 12)
    assert cells == ["red"] * 6 + ["blue"] * 6


def test_a_time_past_the_last_frame_gives_the_last_frame(tmp_path):
    v = _red_then_blue(tmp_path / "cut.mp4")
    [(_, last)] = ev.frames_at(v, [99.0], tmp_path / "f", width=64)
    assert Image.open(last).convert("RGB").getpixel((32, 32))[2] > 150


def test_has_audio_and_a_video_with_no_picture(tmp_path):
    v = _video(tmp_path / "v.mp4")
    assert ev.has_audio(v)
    silent = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(v), "-an", "-c:v", "copy", str(silent)], check=True)
    assert not ev.has_audio(silent)
    wav = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(v), "-vn", str(wav)], check=True)
    assert ev.tech_checks(wav, width=320, height=180, fps=30, max_static_s=4.0) == \
        {"ok": False, "issues": ["no video stream"], "codec": None}


# --- SFX: each placed sound is found in the stem by cross-correlation -------------------------------------------------------

def _burst(freq, tau_s, amp, seconds=0.4):
    k = np.arange(int(seconds * SR)) / SR
    return amp * np.cos(2 * np.pi * freq * k) * np.exp(-k / tau_s)


def _start(clip, t):
    return round(t * SR) - int(np.argmax(np.abs(clip)))


def _sounds(tmp_path, spec, *, late_s=None, scale=1.0, seconds=3.0):
    """spec: [(name, clip, t)]. The stem has each clip peaking at its t (late_s: {index: seconds} moves one later);
    returns (stem path, placements, clips as sfx.event_clip gives them)."""
    buf = np.zeros(int(seconds * SR))
    placements, clips = [], {}
    for i, (name, clip, t) in enumerate(spec):
        start = _start(clip, t)
        where = start + round((late_s or {}).get(i, 0.0) * SR)
        buf[where:where + len(clip)] += clip
        placements.append(Placement(event=f"e{i}", sfx=name, t=t, start_s=start / SR))
        clips[f"e{i}"] = (clip, start)
    sf.write(tmp_path / "stem.wav", (buf * scale).astype(np.float32), SR, subtype="FLOAT")
    return tmp_path / "stem.wav", placements, clips


def _offsets(report):
    return [r["offset_ms"] for r in report["events"]]


def test_local_peak_with_no_samples_in_its_window_is_none_not_an_error():
    data = np.zeros(SR, dtype=np.float32)
    assert local_peak_s(data, SR, 5.0) is None
    assert local_peak_s(data, SR, 0.5) is not None


def test_a_soft_tick_right_after_a_loud_sound_with_a_long_decay_is_found_at_its_own_place(tmp_path):
    loud, tick = _burst(500, 0.08, 0.8), _burst(3000, 0.01, 0.05)
    stem, placements, clips = _sounds(tmp_path, [("boom", loud, 1.0), ("tick", tick, 1.1)])
    data = sf.read(stem, dtype="float32")[0]
    assert local_peak_s(data, SR, 1.1) == pytest.approx(1.0, abs=0.001)     # the loudest sample near the tick is the boom's
    rep = ev.sfx_report(stem, placements, 60, clips=clips)
    assert rep["ok"] is True and _offsets(rep) == pytest.approx([0.0, 0.0], abs=0.05)
    assert all(r["corr"] > 0.99 for r in rep["events"])


def test_a_tick_thirty_ms_late_reads_plus_thirty_whatever_sits_next_to_it(tmp_path):
    loud, tick = _burst(500, 0.08, 0.8), _burst(3000, 0.01, 0.05)
    stem, placements, clips = _sounds(tmp_path, [("boom", loud, 1.0), ("tick", tick, 1.1)], late_s={1: 0.03})
    rep = ev.sfx_report(stem, placements, 60, clips=clips)             # one frame at 60 fps is 16.7 ms
    assert _offsets(rep)[0] == pytest.approx(0.0, abs=0.05) and _offsets(rep)[1] == pytest.approx(30.0, abs=0.05)
    assert [r["ok"] for r in rep["events"]] == [True, False] and rep["ok"] is False


def test_the_same_sound_twice_a_tenth_of_a_second_apart_is_told_apart(tmp_path):
    tick = _burst(2500, 0.02, 0.4)
    stem, placements, clips = _sounds(tmp_path, [("tick", tick, 1.0), ("tick", tick, 1.1)], late_s={0: 0.03})
    rep = ev.sfx_report(stem, placements, 60, clips=clips)
    assert _offsets(rep)[0] == pytest.approx(30.0, abs=0.05) and _offsets(rep)[1] == pytest.approx(0.0, abs=0.05)
    assert [r["window_ms"] for r in rep["events"]] == [50.0, 50.0]      # half the gap to the other tick: no mix-up


def test_overlapping_real_sounds_are_each_found_exactly(tmp_path):
    # a 1.5 s sub boom and a 0.8 s impact 0.1 s later share their low end: a bare correlation picks the wrong place
    boom, impact = sfx_synth.synth("sub_boom") * 0.25, sfx_synth.synth("impact") * 0.25
    stem, placements, clips = _sounds(tmp_path, [("sub", boom, 1.0), ("impact", impact, 1.1)], seconds=4.0)
    rep = ev.sfx_report(stem, placements, 60, clips=clips)
    assert rep["ok"] is True and _offsets(rep) == pytest.approx([0.0, 0.0], abs=0.05)
    (tmp_path / "x").mkdir()
    late_stem, late_p, late_clips = _sounds(tmp_path / "x", [("sub", boom, 1.0), ("impact", impact, 1.1)],
                                            late_s={1: 0.03}, seconds=4.0)
    late = ev.sfx_report(late_stem, late_p, 60, clips=late_clips)
    # the boom is read through the late impact's leftover in the stem, which a 55 Hz tone's broad peak turns into a
    # fraction of a millisecond; the late impact itself is exact
    assert _offsets(late) == pytest.approx([0.0, 30.0], abs=0.5)


def test_a_stem_scaled_down_to_stay_under_full_scale_is_still_found(tmp_path):
    stem, placements, clips = _sounds(tmp_path, [("a", _burst(900, 0.05, 0.9), 0.5), ("b", _burst(2000, 0.03, 0.9), 0.56)],
                                      scale=0.55)
    rep = ev.sfx_report(stem, placements, 60, clips=clips)
    assert rep["ok"] is True and _offsets(rep) == pytest.approx([0.0, 0.0], abs=0.05)


def test_sounds_at_one_time_are_both_found_and_do_not_narrow_each_others_window(tmp_path):
    stem, placements, clips = _sounds(tmp_path, [("a", _burst(900, 0.05, 0.5), 1.0), ("b", _burst(2200, 0.03, 0.5), 1.0)])
    rep = ev.sfx_report(stem, placements, 60, clips=clips)
    assert rep["ok"] is True and [r["window_ms"] for r in rep["events"]] == [150.0, 150.0]


def test_a_sound_that_is_not_in_the_stem_is_not_found_and_fails_and_is_never_called_late(tmp_path):
    stem, placements, clips = _sounds(tmp_path, [("a", _burst(900, 0.05, 0.5), 1.0)])
    other = _burst(2700, 0.04, 0.5)                       # the stem was not made from this clip
    rep = ev.sfx_report(stem, placements, 60, clips={"e0": (other, _start(other, 1.0))})
    row = rep["events"][0]
    assert rep["ok"] is False and row["ok"] is False and row["offset_ms"] is None and row["corr"] < 0.5
    assert "not found in stem" in row["issue"] and "unmeasurable" not in row
    gone = ev.sfx_report(stem, placements, 60, clips={})["events"][0]    # its sound cannot be made again to look for it
    assert gone["ok"] is False and "not found in stem" in gone["issue"] and "could not be made again" in gone["issue"]


def test_only_a_silent_clip_is_unmeasurable_and_that_is_not_a_failure(tmp_path):
    stem, placements, clips = _sounds(tmp_path, [("a", _burst(900, 0.05, 0.5), 1.0)])
    silent = np.zeros(4800)
    rep = ev.sfx_report(stem, placements, 60, clips={"e0": (silent, round(1.0 * SR))})
    row = rep["events"][0]
    assert rep["ok"] is True and row["unmeasurable"] is True and "silent" in row["issue"]


def test_an_event_past_the_end_of_the_stem_is_a_finding_not_a_crash(tmp_path):
    clip = _burst(900, 0.05, 0.5)
    stem, placements, clips = _sounds(tmp_path, [("a", clip, 1.0)], seconds=3.0)
    placements.append(Placement(event="e1", sfx="a", t=5.0, start_s=_start(clip, 5.0) / SR))
    clips["e1"] = (clip, _start(clip, 5.0))
    rep = ev.sfx_report(stem, placements, 30, clips=clips)
    assert rep["ok"] is False and rep["events"][0]["ok"] is True
    late = rep["events"][1]
    assert late["ok"] is False and late["offset_ms"] is None and "3.000s long" in late["issue"]


def test_a_stem_that_is_silent_at_an_event_says_so(tmp_path):
    clip = _burst(900, 0.05, 0.5)
    stem, placements, clips = _sounds(tmp_path, [("a", clip, 1.0)])
    placements.append(Placement(event="e1", sfx="a", t=2.0, start_s=_start(clip, 2.0) / SR))
    clips["e1"] = (clip, _start(clip, 2.0))
    rep = ev.sfx_report(stem, placements, 30, clips=clips)
    assert rep["events"][1]["ok"] is False and "silent" in rep["events"][1]["issue"]


def test_no_stem_fails_the_sfx_check_with_a_pointer(tmp_path):
    for stem in (None, tmp_path / "missing.wav"):
        rep = ev.sfx_report(stem, _placed(1.0), 30)
        assert rep["ok"] is False and "promo sfx render" in rep["detail"]


def _placed(*times):
    return [Placement(event=f"e{i}", sfx="click", t=t, start_s=t) for i, t in enumerate(times)]


def test_a_placement_for_an_event_that_has_moved_or_gone_is_stale(tmp_path):
    stem, placements, clips = _sounds(tmp_path, [("a", _burst(900, 0.05, 0.5), 1.0), ("b", _burst(2200, 0.03, 0.5), 2.0)])
    rep = ev.sfx_report(stem, placements, 30, clips=clips, event_times={"e0": 1.0, "e1": 2.4})
    assert rep["events"][0]["ok"] is True
    assert rep["events"][1]["ok"] is False and "now has it at 2.400s" in rep["events"][1]["issue"]
    gone = ev.sfx_report(stem, placements[:1], 30, clips=clips, event_times={})
    assert gone["ok"] is False and "no longer in timing.json" in gone["events"][0]["issue"]


# --- the voice on the final mix ---------------------------------------------------------------------------------------------

def _lines(*specs):
    """Timed lines [(id, text, start_s)] at 0.3 s a word, with a take of that length on disk for each (see _takes)."""
    return [TimedLine(id=i, chapter="c01", text=text, start_s=start, end_s=start + 0.3 * len(text.split()),
                      words=_say(text, start)) for i, text, start in specs]


def _takes(tmp_path, lines):
    files = {}
    for k, line in enumerate(lines):
        files[line.id] = tmp_path / f"take-{line.id}.wav"
        sf.write(files[line.id], _voice(line.end_s - line.start_s, 40 + k), SR, subtype="FLOAT")
    return files


def _vo(tmp_path, lines, heard, *, shifts=None, takes=True, **kw):
    """The report for `lines` whose takes are laid in a mix at their starts (+ shifts[id]); `heard` maps a line id to the
    words a recogniser would write for the window cut around it (None: the line's own words)."""
    files = _takes(tmp_path, lines)
    mix = _mix(tmp_path / "m.wav", [(sf.read(files[l.id], dtype="float32")[0], l.start_s + (shifts or {}).get(l.id, 0.0), 0.5)
                                    for l in lines])
    t = _timing().model_copy(update={"lines": lines})
    return ev.vo_report(t, _with_audio(mix, tmp_path / "v.mp4"), "en", takes=files if takes else None,
                        transcribe=lambda p, lang: (heard or {}).get(Path(p).stem) or _say(
                            {l.id: l.text for l in lines}[Path(p).stem]), **kw)


def test_each_take_is_found_in_the_mix_to_the_sample_and_the_search_is_per_line(tmp_path):
    lines = _lines(("l1", "Moving money just got safer.", 1.0), ("l2", "Risk has to sign off.", 3.5))
    rep = _vo(tmp_path, lines, None, shifts={"l2": 0.012})
    rows = rep["vo"]["rows"]
    assert rep["vo"]["ok"] is True and rows[0]["offset_ms"] == pytest.approx(0, abs=8) and rows[0]["corr"] > 0.8
    assert rows[1]["offset_ms"] == pytest.approx(12, abs=8)           # within one frame (33 ms at 30 fps), and reported
    assert rep["vo"]["tolerance_ms"] == pytest.approx(33.33, abs=0.01) and rep["vo"]["min_corr"] == 0.5


def test_a_take_more_than_a_frame_late_in_the_mix_fails_and_says_by_how_much(tmp_path):
    lines = _lines(("l1", "Moving money just got safer.", 1.0), ("l2", "Risk has to sign off.", 3.5))
    rep = _vo(tmp_path, lines, None, shifts={"l1": 0.08})
    assert rep["vo"]["ok"] is False
    assert rep["vo"]["rows"][0]["offset_ms"] == pytest.approx(80, abs=8) and rep["vo"]["rows"][0]["ok"] is False
    assert rep["vo"]["rows"][1]["ok"] is True


def test_a_take_that_is_not_in_the_mix_is_not_found(tmp_path):
    lines = _lines(("l1", "Moving money just got safer.", 1.0))
    files = _takes(tmp_path, lines)
    mix = _mix(tmp_path / "m.wav", [(_voice(1.5, 77), 1.0, 0.5)])        # a different voice is there instead
    t = _timing().model_copy(update={"lines": lines})
    rep = ev.vo_report(t, _with_audio(mix, tmp_path / "v.mp4"), "en", takes=files, transcribe=lambda p, l: _say(lines[0].text))
    row = rep["vo"]["rows"][0]
    assert rep["vo"]["ok"] is False and row["offset_ms"] is None and row["corr"] < 0.5 and "not found within 0.5 s" in row["issue"]


def test_a_line_with_no_take_has_nothing_in_the_mix_and_fails_by_name(tmp_path):
    lines = _lines(("l1", "Moving money just got safer.", 1.0))
    rep = _vo(tmp_path, lines, None, takes=False)
    assert rep["vo"]["ok"] is False and "no voice take" in rep["vo"]["rows"][0]["issue"]


def test_the_wer_window_is_the_takes_own_length_from_its_start_and_scored_by_the_takes_own_function(tmp_path):
    from promo.voice import score_take

    lines = _lines(("l1", "Moving money just got safer.", 1.0), ("l2", "Risk has to sign off.", 3.5))
    seen = {}

    def hear(path, lang):
        info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate,channels:format=duration",
                                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout)
        seen[Path(path).stem] = (float(info["format"]["duration"]), info["streams"][0]["sample_rate"], info["streams"][0]["channels"])
        return _say({l.id: l.text for l in lines}[Path(path).stem], lag_ms=40)

    files = _takes(tmp_path, lines)
    mix = _mix(tmp_path / "m.wav", [(sf.read(files[l.id], dtype="float32")[0], l.start_s, 0.5) for l in lines])
    t = _timing().model_copy(update={"lines": lines})
    rep = ev.vo_report(t, _with_audio(mix, tmp_path / "v.mp4"), "en", takes=files, transcribe=hear)
    assert seen["l1"][0] == pytest.approx(1.5, abs=0.06) and seen["l2"][0] == pytest.approx(1.5, abs=0.06)
    assert seen["l1"][1:] == ("16000", 1)                               # resampled for recognition: 16 kHz mono
    expected = score_take(Path("x"), lines[0].text, {}, "en", 1500, lambda p, l: _say(lines[0].text, lag_ms=40))[0]
    assert rep["lines"]["rows"][0] == {"line": "l1", "wer": round(expected, 3), "ok": True}


def test_numbers_spelling_variants_and_inline_audio_tags_do_not_count_against_a_line(tmp_path):
    lines = _lines(("l1", "Reversals over $10,000?", 1.0), ("l2", "The colour <short pause> is organised.", 3.0))
    heard = {"l1": _say("Reversals over ten thousand dollars"), "l2": _say("The color is organized.")}
    rep = _vo(tmp_path, lines, heard)
    assert [r["wer"] for r in rep["lines"]["rows"]] == [0.0, 0.0] and rep["lines"]["ok"] is True


def test_a_known_mishearing_is_read_as_the_scripts_word_when_aliases_are_given(tmp_path):
    lines = _lines(("l1", "Oleafly puts it all in one place.", 1.0))
    heard = {"l1": _say("oh leafly puts it all in one place.")}
    without = _vo(tmp_path, lines, heard)
    assert without["lines"]["rows"][0]["wer"] > 0.15 and without["lines"]["ok"] is False
    (tmp_path / "again").mkdir()
    rep = _vo(tmp_path / "again", lines, heard, aliases={"Oleafly": ["oh leafly"]})
    assert rep["lines"]["rows"][0] == {"line": "l1", "wer": 0.0, "ok": True}


def test_a_line_that_skips_verification_is_not_scored_for_wer_but_is_still_found_in_the_mix(tmp_path):
    lines = _lines(("psst", "Psst.", 1.0), ("l2", "Moving money just got safer.", 2.0))
    rep = _vo(tmp_path, lines, {"psst": _say("Shh and sh."), "l2": _say("Moving money just got safer.")}, skip_lines={"psst"})
    assert rep["lines"]["rows"][0] == {"line": "psst", "wer": None, "ok": True, "skipped": "verify: false"}
    assert rep["lines"]["rows"][1]["wer"] == 0.0 and rep["lines"]["ok"] is True
    assert [r["ok"] for r in rep["vo"]["rows"]] == [True, True]           # the take of "Psst." is in the mix all the same
    (tmp_path / "again").mkdir()
    unskipped = _vo(tmp_path / "again", lines, {"psst": _say("Shh and sh.")})
    assert unskipped["lines"]["ok"] is False


def test_a_line_that_was_not_heard_fails_by_name(tmp_path):
    lines = _lines(("l1", "Moving money just got safer.", 1.0), ("l2", "Risk has to sign off.", 3.5))
    rep = _vo(tmp_path, lines, {"l2": [Caption(text="hmm", startMs=0, endMs=200)]})
    assert [r["ok"] for r in rep["lines"]["rows"]] == [True, False] and rep["lines"]["rows"][1]["wer"] == 1.0


def test_recognition_word_times_are_information_only(tmp_path):
    lines = _lines(("l1", "Moving money just got safer.", 1.0), ("l2", "Risk has to sign off.", 3.5))
    heard = {"l1": _say("Moving money just got safer.", lag_ms=150), "l2": _say("Risk has to sign off.", lag_ms=150)}
    rep = _vo(tmp_path, lines, heard)
    info = rep["vo"]["asr_offsets_ms"]
    assert rep["vo"]["ok"] is True and info["words"] == 10 and info["median"] == info["p95"] == 150


def test_a_video_with_no_audio_is_not_transcribed_and_fails(tmp_path):
    lines = _lines(("l1", "Moving money just got safer.", 1.0))
    t = _timing().model_copy(update={"lines": lines})
    silent = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=3",
                    "-an", "-pix_fmt", "yuv420p", str(silent)], check=True)

    def boom(path, lang):
        raise AssertionError("nothing to transcribe")

    rep = ev.vo_report(t, silent, "en", takes=_takes(tmp_path, lines), transcribe=boom)
    assert rep["vo"]["ok"] is False and rep["lines"]["ok"] is False
    assert rep["vo"]["rows"][0]["issue"] == "the video has no audio track"


def test_a_film_with_no_voice_has_nothing_to_fail(tmp_path):
    rep = _vo(tmp_path, [], None)
    assert rep["vo"]["ok"] is True and rep["lines"] == {"ok": True, "rows": []}


def test_the_phone_and_master_sheets_page_at_36_frames(tmp_path):
    v = _video(tmp_path / "long.mp4", src="testsrc=size=160x90:rate=10:duration=40", audio="sine=duration=40:sample_rate=48000")
    ev.phone_sheet(v, tmp_path / "phone.png", every_s=1.0)        # 40 frames: 36, then 4
    first, second = Image.open(tmp_path / "phone.png"), Image.open(tmp_path / "phone-01.png")
    assert first.size == (5 * (360 + 4), 8 * (202 + 4)) and second.size == (5 * (360 + 4), 1 * (202 + 4))
    assert not (tmp_path / "phone-02.png").exists()
    ev.master_sheet(v, tmp_path / "master.png", fps=2.0)           # 80 frames: 36, 36, 8
    assert [(tmp_path / f"master{s}.png").exists() for s in ("", "-01", "-02", "-03")] == [True, True, True, False]


def test_score_take_is_the_one_scoring_path_for_a_new_take_a_stored_one_and_the_critic(tmp_path, monkeypatch):
    import io

    from promo import voice
    from promo.shotlist import VOLine

    heard = lambda p, lang: _say("oh leafly puts it all in one place.")
    aliases = {"Oleafly": ["oh leafly"]}
    wer, words = voice.score_take(Path("x.wav"), "Oleafly puts it <short pause> all in one place.", aliases, "en", 3000, heard)
    assert wer == 0.0 and [w.text for w in words] == "Oleafly puts it all in one place.".split()      # script spelling, no tag
    assert ev.score_take is voice.score_take                                                       # the critic's window too

    calls = []
    real = voice.score_take
    monkeypatch.setattr(voice, "score_take", lambda *a, **k: calls.append(a[1]) or real(*a, **k))
    line = VOLine(id="l1", text="Oleafly puts it all in one place.")
    buf = io.BytesIO()
    sf.write(buf, _voice(2.0, 5), SR, format="WAV", subtype="PCM_16")
    take = voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={}, language="en",
                                 vo_dir=tmp_path / "vo", synth=lambda text, *, voice, style="", model="": buf.getvalue(),
                                 transcribe=heard, asr_aliases=aliases)
    assert calls == [line.text] and take.wer == 0.0 and [w.text for w in take.words][0] == "Oleafly"
    stored = voice._rescore(take.model_copy(update={"wer": 0.9, "words": []}), line, vo_dir=tmp_path / "vo", language="en",
                            aliases=aliases, transcribe=heard)
    assert calls == [line.text, line.text] and stored.wer == 0.0 and [w.text for w in stored.words] == line.text.split()


@pytest.mark.live
def test_live_real_speech_is_found_in_the_mix_and_scored_line_by_line_with_real_recognition(tmp_path):
    """The first minute of the Oleafly mascot run's own VO (examples/ is only read), with that run's timing, takes, shot
    list and aliases, through the real local ASR. Each take is found by correlation within a millisecond of its planned
    start, and each line's window of the mix scores with the takes' own function: recognition timing agrees with the
    take's to the millisecond because it is the same audio cut to the same window. (One long recognition pass over the
    minute drifts by up to 0.4 s a word, which is why the windows are per line.)"""
    from promo.shotlist import load_shotlist
    from promo.timing import load_timing
    from promo.voice import voice_settings

    if not (MASCOT / "renders" / "vo-preview.wav").exists():
        pytest.skip("examples/promo/oleafly-mascot is not here")
    sl, t = load_shotlist(MASCOT / "shotlist.json"), load_timing(MASCOT / "timing.json")
    t = t.model_copy(update={"lines": [l for l in t.lines if l.end_s <= 59.5]})
    wav = tmp_path / "vo.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-t", "60", "-i", str(MASCOT / "renders" / "vo-preview.wav"),
                    str(wav)], check=True)
    skip = [l.id for l in sl.all_lines() if not l.verify]
    assert skip == ["l01"]
    rep = ev.vo_report(t, wav, "en", takes={l.id: MASCOT / l.file for l in t.lines},
                       aliases=voice_settings(sl, load_pack(sl.pack)).aliases, skip_lines=skip)
    rows = rep["vo"]["rows"]
    assert [r["line"] for r in rows] == [l.id for l in t.lines] and rep["vo"]["ok"] is True
    assert max(abs(r["offset_ms"]) for r in rows) < 2 and min(r["corr"] for r in rows) > 0.95
    assert rep["lines"]["rows"][0] == {"line": "l01", "wer": None, "ok": True, "skipped": "verify: false"}
    assert rep["lines"]["ok"] is True
    assert rep["vo"]["asr_offsets_ms"]["words"] > 100 and rep["vo"]["asr_offsets_ms"]["p95"] <= 40
