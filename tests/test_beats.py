import json

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from promo.beats import STEADY_TOL_S, _refined_bpm, _regularize, analyze, write_grid
from promo.cli import app
from promo.models import BeatGrid
from promo.run import new_run


def _click_track(path, bpm=120.0, start=0.25, bars=8, kick_phase=1, sr=22050):
    beat = 60.0 / bpm
    n_beats = bars * 4
    y = np.zeros(int((start + n_beats * beat + 1.0) * sr), dtype=np.float32)
    rng = np.random.default_rng(0)
    for i in range(n_beats):
        t0 = int((start + i * beat) * sr)
        hat = rng.standard_normal(int(0.02 * sr)).astype(np.float32)
        hat = np.diff(hat, prepend=0.0) * 0.15 * np.hanning(len(hat))
        y[t0:t0 + len(hat)] += hat
        if i % 4 == kick_phase:
            k = np.arange(int(0.15 * sr)) / sr
            y[t0:t0 + len(k)] += 0.9 * np.sin(2 * np.pi * 60 * k) * np.exp(-k * 25)
    sf.write(path, y, sr)
    return [start + i * beat for i in range(n_beats) if i % 4 == kick_phase]


def test_analyze_finds_tempo_and_real_downbeats(tmp_path):
    p = tmp_path / "clicks.wav"
    kicks = _click_track(p)
    g = analyze(p)
    assert g.bpm == pytest.approx(120, abs=2)
    assert min(abs(g.first_downbeat_s - k) for k in kicks[:2]) < 0.06
    spacing = np.diff(g.downbeats)
    assert np.allclose(spacing, 2.0, atol=0.06)
    assert g.first_downbeat_s in g.beats


def test_grid_round_trip(tmp_path):
    g = BeatGrid(bpm=120, beats=[0.5, 1.0], downbeats=[0.5], first_downbeat_s=0.5)
    p = tmp_path / "beats.json"
    write_grid(g, p)
    assert BeatGrid.model_validate_json(p.read_text()) == g


def _assert_even(beats, period=0.5, tol=0.05):
    gaps = np.diff(beats)
    assert np.all(np.abs(gaps - period) <= tol * period), gaps


def test_regularize_fills_dropped_beats():
    beats = [i * 0.5 for i in range(32) if i not in (10, 20, 21, 22)]  # one missing, then three in a row
    out = _regularize(beats, 0.5)
    assert len(out) == 32
    _assert_even(out)
    assert out[0] == beats[0] and out[-1] == beats[-1]


def test_regularize_drops_doubled_beat():
    beats = [i * 0.5 for i in range(32)]
    beats.insert(16, 7.6)  # the tracker fired twice on beat 15 (t=7.5)
    out = _regularize(beats, 0.5)
    assert len(out) == 32 and 7.6 not in out
    _assert_even(out)


def test_regularize_leaves_clean_beats_alone():
    beats = [0.25 + i * 0.4876 + (i % 2) * 0.0232 for i in range(20)]  # frame-quantised jitter
    assert _regularize(beats, 0.5) == beats


def test_refined_bpm_undoes_frame_quantisation():
    frame = 512 / 22050
    beats = np.round(np.arange(32) * 0.5 / frame) * frame
    beats = np.delete(beats, 10)  # one skipped beat must not bias the mean
    assert _refined_bpm(beats, 117.454) == pytest.approx(120, abs=0.3)


def test_refined_bpm_falls_back_with_too_few_gaps():
    assert _refined_bpm(np.array([0.0, 0.5]), 117.454) == 117.454


def test_analyze_grid_is_evenly_spaced(tmp_path):
    p = tmp_path / "clicks.wav"
    _click_track(p)
    g = analyze(p)
    _assert_even(g.beats, period=0.5, tol=0.06)
    assert g.downbeats == g.beats[g.beats.index(g.first_downbeat_s)::4]


def test_beats_cli_reports_non_music_as_error(tmp_path, monkeypatch):
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(3 * 22050, dtype=np.float32), 22050)
    monkeypatch.chdir(tmp_path)
    res = CliRunner().invoke(app, ["beats", str(silent)])
    assert res.exit_code == 1
    assert "promo:" in res.output and "Traceback" not in res.output


def _beats_run(tmp_path, monkeypatch, beats_per_bar=3):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    data = json.loads(run.shotlist.read_text())
    data["music"]["beats_per_bar"] = beats_per_bar
    run.shotlist.write_text(json.dumps(data))
    _click_track(run.music_dir / "bed.wav")
    monkeypatch.chdir(run.root)
    return run


hints: list = []
tols: list = []


def _spy_on_analyze(monkeypatch):
    seen = []
    hints.clear()
    tols.clear()
    real = analyze

    def spy(path, beats_per_bar=4, bpm_hint=None, steady_tol_s=None):
        seen.append(beats_per_bar)
        hints.append(bpm_hint)
        tols.append(steady_tol_s)
        return real(path, beats_per_bar, bpm_hint=bpm_hint, steady_tol_s=steady_tol_s)

    monkeypatch.setattr("promo.beats.analyze", spy)
    return seen


def test_beats_cli_uses_the_shot_lists_beats_per_bar(tmp_path, monkeypatch):
    run = _beats_run(tmp_path, monkeypatch, beats_per_bar=3)
    seen = _spy_on_analyze(monkeypatch)
    res = CliRunner().invoke(app, ["beats"])
    assert res.exit_code == 0, res.output
    assert seen == [3] and hints == [json.loads(run.shotlist.read_text())["music"].get("bpm")]
    assert tols == [pytest.approx(1 / 60)]            # a run's tolerance is its frame, as `promo music` judged it
    grid = BeatGrid.model_validate_json((run.music_dir / "beats.json").read_text())
    assert grid.beats_per_bar == 3


def test_beats_cli_keeps_common_time_without_a_usable_shot_list(tmp_path, monkeypatch):
    run = _beats_run(tmp_path, monkeypatch, beats_per_bar=3)
    run.shotlist.write_text("{not json")
    seen = _spy_on_analyze(monkeypatch)
    res = CliRunner().invoke(app, ["beats"])
    assert res.exit_code == 0, res.output
    assert seen == [4]


def test_beats_cli_outside_a_run_keeps_common_time(tmp_path, monkeypatch):
    wav = tmp_path / "clicks.wav"
    _click_track(wav)
    monkeypatch.chdir(tmp_path)
    seen = _spy_on_analyze(monkeypatch)
    res = CliRunner().invoke(app, ["beats", str(wav)])
    assert res.exit_code == 0, res.output
    assert seen == [4] and tols == [STEADY_TOL_S]       # outside a run there is no frame: the default


# --- the steady grid: tempo and phase to the millisecond -------------------------------------------------------

from music_helpers import write_groove  # noqa: E402

from promo.beats import fit_steady  # noqa: E402


@pytest.mark.parametrize("bpm, first", [(117.0, 0.613), (126.0, 0.0), (92.0, 1.271)])
def test_fit_steady_measures_tempo_and_phase_to_a_millisecond(tmp_path, bpm, first):
    p = tmp_path / "g.wav"
    beats, downbeats = write_groove(p, bpm, 40.0, first_beat_s=first, seed=int(bpm))
    fit = fit_steady(p, bpm_hint=bpm)
    assert fit is not None
    assert fit.bpm == pytest.approx(bpm, rel=1e-4)
    grid = np.array(fit.grid)
    truth = np.array(beats)
    k = np.round((grid - first) * bpm / 60).astype(int)
    assert np.max(np.abs(grid - (first + k * 60 / bpm))) < 0.002
    assert fit.downbeats[0] == pytest.approx(downbeats[0], abs=0.002)
    assert grid[0] == pytest.approx(truth[0], abs=0.002)   # the grid starts at the file's first beat


def test_analyze_is_not_fooled_by_loud_offbeat_hats(tmp_path):
    # librosa's tracker locks onto the loudest periodic element; loud hats on the "and"s put it half a beat off.
    p = tmp_path / "hats.wav"
    beats, downbeats = write_groove(p, 120.0, 30.0, first_beat_s=0.25, hats_offbeat=0.9, seed=3)
    g = analyze(p, bpm_hint=120)
    errs = [min(abs(b - t) for t in beats) for b in g.beats]
    assert max(errs) < 0.003
    assert g.first_downbeat_s == pytest.approx(downbeats[0], abs=0.003)


def test_analyze_grid_reaches_a_beat_on_the_first_sample(tmp_path):
    p = tmp_path / "head.wav"
    write_groove(p, 120.0, 20.0, first_beat_s=0.0, seed=4)
    g = analyze(p, bpm_hint=120)
    assert g.first_downbeat_s == pytest.approx(0.0, abs=0.002)
    assert g.bpm == pytest.approx(120.0, rel=1e-4)


def test_analyze_keeps_the_tracked_beats_when_the_tempo_moves(tmp_path):
    # A bed that jumps from 100 to 130 BPM has no single grid, so analyze returns librosa's tracked beats.
    from promo.beats import _load, _track

    p = tmp_path / "ramp.wav"
    write_groove(p, 100.0, 30.0, first_beat_s=0.3, bpm_at=lambda t: 100.0 if t < 15 else 130.0, seed=5)
    assert fit_steady(p, bpm_hint=100) is None and fit_steady(p) is None
    _, tracked, _ = _track(_load(p), p, 4)
    assert analyze(p).beats == [round(float(t), 4) for t in tracked]


# --- the steadiness tolerance is the caller's (a film passes one frame) ----------------------------------------------

def _wandering(tmp_path, ms, name="wander.wav"):
    """A 120 BPM groove whose 2-bar windows sit alternately `ms` early and late of one straight grid."""
    p = tmp_path / name
    write_groove(p, 120.0, 48.0, first_beat_s=0.3, seed=11, offset_at=lambda k: (ms / 1000) * (1 if (k // 8) % 2 else -1))
    return p


def test_a_sub_frame_wander_is_steady_when_the_tolerance_is_a_frame(tmp_path):
    p = _wandering(tmp_path, 12)           # reads about 14 ms: under a frame (16.7 ms at 60 fps), over the default 10 ms
    assert fit_steady(p, bpm_hint=120) is None                          # the default tolerance
    fit = fit_steady(p, bpm_hint=120, steady_tol_s=1 / 60)
    assert fit is not None and fit.bpm == pytest.approx(120, rel=1e-3)


def test_a_wander_over_a_frame_is_not_steady_whatever_the_tolerance_up_to_a_frame(tmp_path):
    from promo.beats import fit_steady_why

    p = _wandering(tmp_path, 20)
    fit, why = fit_steady_why(p, bpm_hint=120, steady_tol_s=1 / 60)
    assert fit is None and "its beats wander 2" in why and "ms from one straight grid" in why


def test_analyze_takes_the_same_tolerance(tmp_path):
    from promo.beats import _load, _track

    p = _wandering(tmp_path, 12, "wander2.wav")
    _, tracked, _ = _track(_load(p), p, 4)
    assert analyze(p, bpm_hint=120).beats == [round(float(t), 4) for t in tracked]      # no straight grid at 10 ms
    straight = np.diff(analyze(p, bpm_hint=120, steady_tol_s=1 / 60).beats)
    assert np.ptp(straight) < 1e-3                                                      # one straight grid
