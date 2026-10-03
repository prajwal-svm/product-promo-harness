import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from promo.models import LineTake
from promo.packs import Limits
from promo.run import RunPaths
from promo.shotlist import Shotlist
from promo.timeline import AUDIO_RATE, Media, TimelineError, build_timeline, collect_media, read_timeline, write_timeline
from promo.timing import Measured, resolve

FIX = Path(__file__).parent / "fixtures" / "shotlist_music.json"


def _wav(path: Path, seconds: float):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"sine=duration={seconds}:sample_rate=48000",
                    str(path)], check=True)
    return path


@pytest.fixture
def setup(tmp_path):
    takes = {lid: LineTake(id=lid, file=f"audio/vo/{lid}.wav", duration_s=1.0) for lid in ("c01-l1", "c02-l1", "c02-l2")}
    t = resolve(Shotlist.model_validate(json.loads(FIX.read_text())), Measured(takes=takes), Limits())
    shots = {}
    for sid in ("c01-s1", "c02-s1"):  # c02-s2 missing on purpose
        p = tmp_path / f"{sid}.mp4"
        p.write_bytes(b"fake")
        shots[sid] = p
    media = Media(shots=shots, vo={lid: _wav(tmp_path / f"{lid}.wav", 1.0) for lid in takes},
                  music=_wav(tmp_path / "bed.wav", 12.0), sfx=_wav(tmp_path / "sfx.wav", 8.3), overlay=None)
    return t, media


def test_tracks_and_video_items(setup):
    t, media = setup
    tl = build_timeline(t, media)
    assert [tr.name for tr in tl.tracks] == ["V1", "A1 VO", "A2 Music", "A3 SFX"]
    v1 = tl.tracks[0]
    assert [type(i).__name__ for i in v1] == ["Clip", "Clip", "Gap"]
    assert v1[2].name == "missing:c02-s2"
    assert v1[2].metadata["promo"] == {"shot": "c02-s2", "missing": True}
    assert v1.duration().to_seconds() == pytest.approx(8.3)
    assert v1[0].metadata["promo"]["shot"] == "c01-s1"
    assert tl.metadata["promo"]["fps"] == 60


def test_vo_positions_match_timing(setup):
    t, media = setup
    tl = build_timeline(t, media)
    a1 = tl.tracks[1]
    starts = [a1.range_of_child_at_index(i).start_time.to_seconds() for i, item in enumerate(a1)
              if type(item).__name__ == "Clip"]
    assert starts == pytest.approx([l.start_s for l in t.lines], abs=1 / AUDIO_RATE)


def test_music_starts_after_preroll_and_is_trimmed(setup):
    t, media = setup
    a2 = build_timeline(t, media).tracks[2]
    assert type(a2[0]).__name__ == "Gap" and a2[0].duration().to_seconds() == pytest.approx(0.3)
    assert a2[1].source_range.duration.to_seconds() == pytest.approx(8.0)


def test_overlay_track_when_present(setup, tmp_path):
    t, media = setup
    ov = tmp_path / "captions.mov"
    ov.write_bytes(b"mov")
    tl = build_timeline(t, media.model_copy(update={"overlay": ov}))
    assert [tr.name for tr in tl.tracks][:2] == ["V1", "V2"]


def test_round_trip(setup, tmp_path):
    t, media = setup
    p = write_timeline(build_timeline(t, media), tmp_path / "time line 🎬.otio")
    back = read_timeline(p)
    assert back.tracks[0][0].media_reference.target_url == str(media.shots["c01-s1"])


def test_clips_carry_source_hash_and_timeline_metadata(setup):
    t, media = setup
    tl = build_timeline(t, media)
    clip = tl.tracks[0][0]
    assert clip.metadata["promo"]["sha256"] == hashlib.sha256(b"fake").hexdigest()
    assert clip.metadata["promo"]["engine"] == "hyperframes"
    assert [c.metadata["promo"]["line"] for c in tl.tracks[1] if type(c).__name__ == "Clip"] == [l.id for l in t.lines]
    assert tl.tracks[2][1].metadata["promo"]["role"] == "music"
    assert tl.tracks[3][0].metadata["promo"]["role"] == "sfx"
    assert tl.metadata["promo"] == {"duration_s": t.duration_s, "fps": 60, "width": 1920, "height": 1080,
                                    "preroll_s": 0.3}


def test_audio_rate_is_48k_and_video_rate_is_fps(setup):
    t, media = setup
    tl = build_timeline(t, media)
    assert tl.tracks[0][0].source_range.duration.rate == 60
    for track in tl.tracks[1:]:
        assert all(i.source_range.duration.rate == AUDIO_RATE for i in track)


def test_missing_optional_audio_leaves_empty_tracks(setup):
    t, media = setup
    tl = build_timeline(t, media.model_copy(update={"music": None, "sfx": None, "vo": {}}))
    assert [tr.name for tr in tl.tracks] == ["V1", "A1 VO", "A2 Music", "A3 SFX"]
    assert [len(tr) for tr in tl.tracks[1:]] == [0, 0, 0]


def test_vo_position_error_does_not_accumulate(setup, tmp_path):
    # Many short lines at odd times: each clip must start on its own nearest sample, not drift.
    t, _ = setup
    proto = t.lines[0]
    t.lines = [proto.model_copy(update={"id": f"l{k}", "start_s": 0.3 + k * 0.1000013, "end_s": 0.3 + k * 0.1000013 + 0.0500017})
               for k in range(60)]
    wav = _wav(tmp_path / "v.wav", 0.1)
    tl = build_timeline(t, Media(vo={l.id: wav for l in t.lines}))
    a1 = next(tr for tr in tl.tracks if tr.name == "A1 VO")
    starts = [a1.range_of_child_at_index(i).start_time.to_seconds() for i, x in enumerate(a1) if type(x).__name__ == "Clip"]
    assert starts == pytest.approx([l.start_s for l in t.lines], abs=0.5 / AUDIO_RATE + 1e-9)


def test_overlapping_vo_lines_are_an_error(setup):
    t, media = setup
    t.lines[1].start_s = t.lines[0].end_s - 0.5
    with pytest.raises(TimelineError, match=t.lines[1].id):
        build_timeline(t, media)


def test_timeline_error_is_a_value_error():
    assert issubclass(TimelineError, ValueError)


@pytest.mark.parametrize("gap_samples", [0, 1, -1])
def test_vo_line_touching_the_previous_one_is_accepted(setup, gap_samples):
    # Starting exactly at, or within one sample of, the previous end is not an overlap.
    t, media = setup
    t.lines[1].start_s = t.lines[0].end_s + gap_samples / AUDIO_RATE
    t.lines[1].end_s = t.lines[1].start_s + 1.0
    a1 = build_timeline(t, media).tracks[1]
    assert [c.metadata["promo"]["line"] for c in a1 if type(c).__name__ == "Clip"][:2] == [t.lines[0].id, t.lines[1].id]


def test_vo_line_two_samples_before_the_previous_end_is_an_error(setup):
    t, media = setup
    t.lines[1].start_s = t.lines[0].end_s - 2 / AUDIO_RATE
    with pytest.raises(TimelineError, match=t.lines[1].id):
        build_timeline(t, media)


def test_collect_media_finds_rendered_files_and_skips_absent_ones(tmp_path):
    run = RunPaths(tmp_path / "Codex Projects 🎬" / "promo" / "demo")
    takes = {lid: LineTake(id=lid, file=f"audio/vo/{lid}.wav", duration_s=1.0) for lid in ("c01-l1", "c02-l1", "c02-l2")}
    t = resolve(Shotlist.model_validate(json.loads(FIX.read_text())), Measured(takes=takes), Limits())
    for d in (run.renders / "shots", run.renders / "draft" / "shots", run.vo_dir, run.music_dir, run.sfx_dir,
              run.captions_dir / "overlay"):
        d.mkdir(parents=True)
    (run.renders / "shots" / "c01-s1.mp4").write_bytes(b"a")
    (run.renders / "draft" / "shots" / "c02-s1.mp4").write_bytes(b"b")
    (run.vo_dir / "c01-l1.wav").write_bytes(b"v")
    (run.music_dir / "bed.wav").write_bytes(b"m")

    final = collect_media(run, t)
    assert final.shots == {"c01-s1": run.renders / "shots" / "c01-s1.mp4"}
    assert final.vo == {"c01-l1": run.root / "audio/vo/c01-l1.wav"}
    assert final.music == run.music_dir / "bed.wav"
    assert final.sfx is None and final.overlay is None

    draft = collect_media(run, t, draft=True)
    assert draft.shots == {"c02-s1": run.renders / "draft" / "shots" / "c02-s1.mp4"}


def _v1_starts(tl):
    v1 = tl.tracks[0]
    return {i.metadata["promo"]["shot"]: v1.range_of_child_at_index(k).start_time.to_seconds()
            for k, i in enumerate(v1) if type(i).__name__ == "Clip"}


def test_v1_places_each_shot_at_its_start_when_chapters_are_not_contiguous(setup):
    # c02 starts at bar 2.25 (4.8 s), half a second after c01 ends (4.3 s). Timing allows this, so V1 has
    # to carry the hole; otherwise c02 plays 0.5 s early against VO, music and SFX.
    t, media = setup
    d = json.loads(FIX.read_text())
    d["chapters"][1]["start"] = {"bar": 2.25}
    takes = {lid: LineTake(id=lid, file=f"audio/vo/{lid}.wav", duration_s=1.0) for lid in ("c01-l1", "c02-l1", "c02-l2")}
    t = resolve(Shotlist.model_validate(d), Measured(takes=takes), Limits(max_dead_s=30))
    assert t.shot("c02-s1").start_s == pytest.approx(4.8) and t.duration_s == pytest.approx(8.8)
    v1 = build_timeline(t, media).tracks[0]
    assert [type(i).__name__ for i in v1] == ["Clip", "Gap", "Clip", "Gap"]
    assert v1[1].name == "" and v1[1].duration().to_seconds() == pytest.approx(0.5)
    assert v1.range_of_child_at_index(2).start_time.to_seconds() == pytest.approx(t.shot("c02-s1").start_s, abs=1 / 60)
    assert v1[3].name == "missing:c02-s2"
    assert v1.duration().to_seconds() == pytest.approx(t.duration_s)


def test_v1_runs_the_whole_film_when_the_last_shot_ends_early(setup):
    t, media = setup
    t.shots[-1].end_s = t.duration_s - 0.5
    v1 = build_timeline(t, media).tracks[0]
    assert type(v1[-1]).__name__ == "Gap" and v1[-1].duration().to_seconds() == pytest.approx(0.5)
    assert v1.duration().to_seconds() == pytest.approx(t.duration_s)


def test_overlapping_shots_are_an_error(setup):
    t, media = setup
    t.shots[1].start_s = t.shots[0].end_s - 0.5
    with pytest.raises(TimelineError, match=t.shots[1].id):
        build_timeline(t, media)


def test_shot_starting_within_one_frame_of_the_previous_end_is_accepted(setup):
    t, media = setup
    t.shots[1].start_s = t.shots[0].end_s - 1 / 60
    v1 = build_timeline(t, media).tracks[0]
    assert [type(i).__name__ for i in v1] == ["Clip", "Clip", "Gap"]
    assert v1.duration().to_seconds() == pytest.approx(t.duration_s)


def test_zero_frame_shot_is_skipped(setup, tmp_path):
    t, media = setup
    t.shots.append(t.shots[-1].model_copy(update={"id": "c02-s3", "start_s": t.duration_s, "end_s": t.duration_s}))
    v1 = build_timeline(t, media).tracks[0]
    assert [type(i).__name__ for i in v1] == ["Clip", "Clip", "Gap"]
    assert v1.duration().to_seconds() == pytest.approx(t.duration_s)

    run = RunPaths(tmp_path / "run")
    (run.renders / "shots").mkdir(parents=True)
    for sid in ("c01-s1", "c02-s3"):
        (run.renders / "shots" / f"{sid}.mp4").write_bytes(b"x")
    assert list(collect_media(run, t).shots) == ["c01-s1"]


def test_shot_with_no_frames_left_after_the_one_frame_bump_is_skipped(setup):
    # A hand-edited one-frame shot that starts one frame early is allowed, then bumped to the previous end. That
    # leaves it no frames, so it is skipped like any zero-frame shot, never laid down as a zero-frame Clip.
    t, media = setup
    t.shots[1].start_s = t.shots[0].end_s - 1 / 60
    t.shots[1].end_s = t.shots[0].end_s
    v1 = build_timeline(t, media).tracks[0]
    assert all(item.duration().value > 0 for item in v1)
    assert [i.name for i in v1 if type(i).__name__ == "Clip"] == ["c01-s1"]
    assert v1.duration().to_seconds() == pytest.approx(t.duration_s)
