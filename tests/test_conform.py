import numpy as np
import pytest
import soundfile as sf

from music_helpers import write_groove

from promo import conform as cf

FPS = 60
FRAME = 1 / FPS


def _bed(tmp_path, bpm, seconds=40.0, first=0.0, seed=0, name=None, **kw):
    src = tmp_path / (name or f"src-{bpm}-{seed}.mp3")
    beats, downbeats = write_groove(src, bpm, seconds, first_beat_s=first, seed=seed, **kw)
    return src, beats, downbeats


# --- the conform math -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("bpm", [114, 117, 120, 123, 126])
def test_conform_lands_every_beat_within_a_frame_of_the_plan(tmp_path, bpm):
    first = float(np.random.default_rng(bpm).uniform(0.0, 1.5))   # wherever the model put its first downbeat
    src, _, _ = _bed(tmp_path, bpm, first=first, seed=bpm)
    dst = tmp_path / "bed.wav"
    placed = cf.conform(src, dst, bpm=120, beats_per_bar=4, origin_s=0.0, length_s=30.0)
    assert placed.measured_bpm == pytest.approx(bpm, rel=2e-4)
    assert placed.ratio == pytest.approx(120 / bpm, rel=2e-4)
    assert placed.shift_s == pytest.approx(-first * bpm / 120, abs=0.03)    # the head before the first downbeat goes
    info = sf.info(str(dst))
    assert (info.samplerate, info.channels) == (48000, 2) and info.duration == pytest.approx(30.0, abs=1e-4)
    check = cf.check_grid(dst, bpm=120, beats_per_bar=4, origin_s=0.0)
    assert check.max_error_s <= FRAME
    assert check.fit.bpm == pytest.approx(120, rel=3e-4)
    assert check.fit.downbeats[0] == pytest.approx(0.0, abs=FRAME)


@pytest.mark.parametrize("bpm", [112.5, 127.5])      # 6.7% and 5.9% fast of 120 ... and the other way round
def test_conform_refuses_a_stretch_beyond_six_percent(tmp_path, bpm):
    src, _, _ = _bed(tmp_path, bpm, seconds=24.0, seed=7)
    if abs(120 / bpm - 1) <= cf.MAX_STRETCH:
        cf.conform(src, tmp_path / "ok.wav", bpm=120, beats_per_bar=4, length_s=10.0)
        return
    with pytest.raises(cf.ConformError, match=r"re-roll") as err:
        cf.conform(src, tmp_path / "no.wav", bpm=120, beats_per_bar=4, length_s=10.0)
    assert "112.50 BPM" in str(err.value) and "6.7%" in str(err.value)
    assert not (tmp_path / "no.wav").exists()


@pytest.mark.parametrize("measured, ok", [(120 / 1.06, True), (120 / 0.94, True), (120 / 1.0601, False),
                                          (120 / 0.9399, False)])
def test_stretch_ratio_limit_is_six_percent_either_way(measured, ok):
    if ok:
        assert abs(cf.stretch_ratio(measured, 120) - 1) == pytest.approx(0.06, abs=1e-9)
    else:
        with pytest.raises(cf.ConformError, match="re-roll"):
            cf.stretch_ratio(measured, 120)


def test_conform_pads_when_bar_one_comes_after_the_first_downbeat(tmp_path):
    src, _, _ = _bed(tmp_path, 120, seconds=24.0, first=0.2, seed=8)
    dst = tmp_path / "bed.wav"
    placed = cf.conform(src, dst, bpm=120, beats_per_bar=4, origin_s=0.5, length_s=16.0)
    assert placed.shift_s == pytest.approx(0.3, abs=0.03)
    audio, _ = sf.read(dst, always_2d=True)
    assert np.abs(audio[: int(0.25 * 48000)]).max() < 1e-6      # silence padded at the head
    assert cf.check_grid(dst, bpm=120, beats_per_bar=4, origin_s=0.5).fit.downbeats[0] == pytest.approx(0.5, abs=FRAME)


def test_conform_reports_how_much_music_it_had(tmp_path):
    src, _, _ = _bed(tmp_path, 126, seconds=20.0, seed=9)
    placed = cf.conform(src, tmp_path / "bed.wav", bpm=120, beats_per_bar=4, length_s=30.0)
    assert placed.available_s == pytest.approx(20.0 * 126 / 120, abs=0.1)   # the rest of the 30 s is padding
    assert placed.seconds == pytest.approx(30.0)


def test_music_without_a_steady_beat_is_a_reroll(tmp_path):
    src, _, _ = _bed(tmp_path, 100, seconds=30.0, seed=5, bpm_at=lambda t: 100.0 if t < 15 else 130.0)
    with pytest.raises(cf.ConformError, match=r"no steady beat to fit to the 100 BPM grid \(its 2-bar windows lose the "
                                              r"beat period in \d of \d \(at (1|2)\ds.*\).*re-roll"):
        cf.conform(src, tmp_path / "bed.wav", bpm=100, beats_per_bar=4)


def test_check_grid_catches_a_bed_off_the_plan(tmp_path):
    src, _, _ = _bed(tmp_path, 120, seconds=24.0, first=0.03, seed=10, name="late.wav")
    check = cf.check_grid(src, bpm=120, beats_per_bar=4, origin_s=0.0)
    assert check.max_error_s == pytest.approx(0.03, abs=0.003) and check.max_error_s > FRAME


# --- the per-bar gate (ruling R97): the worst bar's median beat error, at most a frame ----------------------------

def test_four_bars_off_the_grid_fail_the_bar_gate(tmp_path):
    # the reviewer's probe: 4 bars of a 2-minute bed sit 40 ms late (2.4 frames), then the music is back on the grid
    src, _, _ = _bed(tmp_path, 120, seconds=120.0, seed=12, name="section.wav",
                     offset_at=lambda k: 0.040 if 40 <= k < 56 else 0.0)            # bars 11-14, 20-28 s
    check = cf.check_grid(src, bpm=120, beats_per_bar=4, origin_s=0.0)
    assert check.max_error_s == pytest.approx(0.040, abs=0.004) and check.max_error_s > FRAME
    assert 11 <= check.worst_bar <= 14 and 20.0 <= check.at_s < 28.0
    assert check.raw_max_error_s >= check.max_error_s


@pytest.mark.parametrize("late", [0.100, 0.250])          # 6 frames; half a beat at 120 BPM
def test_bars_further_off_than_the_attack_search_fail_as_off_grid(tmp_path, late):
    # past the ±60 ms each beat is searched in, the drums would read as a breakdown: they are found in the whole beat
    src, _, _ = _bed(tmp_path, 120, seconds=120.0, seed=15, name=f"far-{late}.wav",
                     offset_at=lambda k: late if 40 <= k < 56 else 0.0)            # bars 11-14, 20-28 s
    check = cf.check_grid(src, bpm=120, beats_per_bar=4, origin_s=0.0)
    assert check.max_error_s == pytest.approx(late, abs=0.006) and check.max_error_s > FRAME
    assert 11 <= check.worst_bar <= 14 and abs(check.worst_offset_s) == pytest.approx(late, abs=0.006)
    assert check.off_grid == [(11, 14)] and not any(a <= 12 <= b for a, b in check.unmeasured)


def test_jitter_and_a_stray_note_pass_the_bar_gate(tmp_path):
    # real music: every beat a few ms off (6 ms spread) and, now and then, a stray note 39 ms late
    rng = np.random.default_rng(13)
    jitter = rng.normal(0.0, 0.006, 400)
    src, _, _ = _bed(tmp_path, 120, seconds=60.0, seed=13, name="jitter.wav",
                     offset_at=lambda k: 0.039 if k % 23 == 11 else float(jitter[k]))
    check = cf.check_grid(src, bpm=120, beats_per_bar=4, origin_s=0.0)
    assert check.max_error_s <= FRAME
    assert check.raw_max_error_s > FRAME                     # the stray note is seen, as a single beat
    assert check.bars_measured >= 0.9 * check.bars_sounding


def test_a_filtered_breakdown_is_unmeasured_not_failed(tmp_path):
    # 4 bars of soft, low-passed drums that also read late (as P1's breakdown did): too few clear beats to judge
    src, _, _ = _bed(tmp_path, 120, seconds=60.0, seed=14, name="breakdown.wav",
                     filtered_at=lambda t: 24.0 <= t < 32.0, offset_at=lambda k: 0.030 if 48 <= k < 64 else 0.0)
    check = cf.check_grid(src, bpm=120, beats_per_bar=4, origin_s=0.0)
    assert check.max_error_s <= FRAME
    assert (13, 16) in check.unmeasured and check.off_grid == []     # bars 13-16, 1-based: no strong hit anywhere
    assert check.bars_measured == check.bars_sounding - 4


def test_check_grid_catches_a_slipped_beat(tmp_path):
    from music_helpers import SR, groove_samples

    y, _, _ = groove_samples(120.0, 40.0, first_beat_s=0.0, seed=11)
    cut = int(20.0 * SR)                                     # 30 ms of nothing on bar 11: every later beat is late
    y = np.concatenate([y[:cut], np.zeros((int(0.03 * SR), 2), np.float32), y[cut:]])
    p = tmp_path / "slip.wav"
    sf.write(p, y, SR, subtype="FLOAT")
    try:
        check = cf.check_grid(p, bpm=120, beats_per_bar=4, origin_s=0.0)
    except cf.ConformError:
        return                                               # or it is not one steady grid at all
    assert check.max_error_s > FRAME and check.at_s > 20.0


# --- stitching segments ---------------------------------------------------------------------------------------

def _segment(tmp_path, bpm, seconds, seed, start_s, chord):
    """A segment as another call would deliver it (its own tempo, its own voicing of the chord), conformed to 120 BPM
    with its first downbeat at its start. Two calls never share a sustained note's phase, so the voicings differ:
    the same sine at opposite phase in both would cancel through any crossfade."""
    src, _, _ = _bed(tmp_path, bpm, seconds=seconds, first=0.11, seed=seed, chord=chord)
    dst = tmp_path / f"seg-{seed}.wav"
    cf.conform(src, dst, bpm=120, beats_per_bar=4, origin_s=0.0)
    return dst, start_s


def test_stitched_segments_join_on_a_downbeat_without_a_dip_or_a_grid_slip(tmp_path):
    bar = 2.0
    seam = 10 * bar                                    # bar 11 starts at 20 s
    lead = 2 * bar
    a = _segment(tmp_path, 123, 26.0, seed=21, start_s=0.0, chord=(220.0, 277.2, 329.6))        # bars 1-11, and over
    b = _segment(tmp_path, 117, 30.0, seed=22, start_s=seam - lead, chord=(164.8, 246.9, 370.0))  # 2 bars of lead-in
    bed = tmp_path / "bed.wav"
    cf.stitch([a, b], [seam], bed, beat_s=0.5, length_s=40.0)
    y, sr = sf.read(bed, always_2d=True)
    assert len(y) == 40 * 48000
    # level: no one-beat window through the crossfade is more than 1.5 dB under the same window a bar either side
    report = cf.seam_report(bed, seam, beat_s=0.5, bar_s=bar)
    assert report["level_dip_db"] >= -1.5 and abs(report["level_step_db"]) < 1.5
    # spectrum: the bar after the seam has the same balance as the bar before it (no restart)
    assert report["spectral_jump_db"] < 3.0
    # grid: one straight 120 BPM grid runs through the seam, and the seam is on one of its downbeats
    check = cf.check_grid(bed, bpm=120, beats_per_bar=4, origin_s=0.0)
    assert check.max_error_s <= FRAME
    downbeats = np.array(check.fit.downbeats)
    assert np.min(np.abs(downbeats - seam)) <= FRAME


def test_seam_report_hears_a_restart(tmp_path):
    # the second part starts thin (its first bars are just the pad, as a model's intro would be): a jump at the seam
    a = _segment(tmp_path, 120, 26.0, seed=31, start_s=0.0, chord=(220.0, 277.2, 329.6))
    src, _, _ = _bed(tmp_path, 120, seconds=30.0, first=0.11, seed=32, chord=(164.8, 246.9, 370.0))
    b = tmp_path / "intro.wav"
    cf.conform(src, b, bpm=120, beats_per_bar=4)
    y, sr = sf.read(b, always_2d=True)
    y[: int(8.0 * sr)] *= 0.1                          # two bars of lead-in and two after the seam, much quieter
    sf.write(b, y, sr, subtype="FLOAT")
    bed = tmp_path / "bed.wav"
    cf.stitch([a, (b, 16.0)], [20.0], bed, beat_s=0.5, length_s=40.0)
    report = cf.seam_report(bed, 20.0, beat_s=0.5, bar_s=2.0, bars=2)
    assert report["level_step_db"] < -6 and report["spectral_jump_db"] > 6


def test_stitch_needs_one_seam_per_join(tmp_path):
    with pytest.raises(ValueError):
        cf.stitch([(tmp_path / "a.wav", 0.0), (tmp_path / "b.wav", 1.0)], [], tmp_path / "x.wav", beat_s=0.5,
                  length_s=2.0)


def _chords(path, chords, seconds_each=2.0, sr=48000):
    t = np.arange(int(seconds_each * sr)) / sr
    y = np.concatenate([sum(np.sin(2 * np.pi * f * t) for f in chord) / len(chord) for chord in chords])
    sf.write(path, np.stack([y, y], 1).astype(np.float32) * 0.3, sr, subtype="FLOAT")
    return path


def test_estimate_key_names_a_progressions_key(tmp_path):
    g, c, d, em = (196.0, 246.9, 293.7), (261.6, 329.6, 392.0), (293.7, 370.0, 440.0), (164.8, 196.0, 246.9)
    assert cf.estimate_key(_chords(tmp_path / "g.wav", [g, c, d, g, em, c, d, g])) == "G major"
    am, dm, e = (220.0, 261.6, 329.6), (293.7, 349.2, 440.0), (329.6, 415.3, 493.9)
    assert cf.estimate_key(_chords(tmp_path / "a.wav", [am, dm, e, am, dm, e, am, am])) == "A minor"


# --- the steadiness tolerance follows the frame ------------------------------------------------------------------

def _wandering(tmp_path, ms):
    src = tmp_path / f"wander-{ms}.mp3"
    write_groove(src, 120.0, 40.0, first_beat_s=0.3, seed=11,
                 offset_at=lambda k: (ms / 1000) * (1 if (k // 8) % 2 else -1))
    return src


def test_conform_calls_a_sub_frame_wander_steady_only_when_told_the_frame(tmp_path):
    src = _wandering(tmp_path, 12)
    with pytest.raises(cf.ConformError, match=r"no steady beat .*wander 1\d ms"):
        cf.conform(src, tmp_path / "a.wav", bpm=120, beats_per_bar=4, length_s=30.0)
    placed = cf.conform(src, tmp_path / "b.wav", bpm=120, beats_per_bar=4, length_s=30.0, steady_tol_s=FRAME)
    assert placed.measured_bpm == pytest.approx(120, rel=1e-3)


def test_conform_still_refuses_a_20_ms_wander_at_a_frame(tmp_path):
    src = _wandering(tmp_path, 20)
    with pytest.raises(cf.ConformError, match=r"no steady beat .*wander 2\d ms"):
        cf.conform(src, tmp_path / "c.wav", bpm=120, beats_per_bar=4, length_s=30.0, steady_tol_s=FRAME)
