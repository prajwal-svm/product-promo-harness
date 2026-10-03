import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from promo import assemble as asm
from promo import audio_analysis as aa
from promo.cli import app
from promo.models import LineTake
from promo.packs import load_pack
from promo.proc import ToolError
from promo.run import new_run
from promo.sfx import render_stem
from promo.shotlist import Shotlist
from promo.timing import Measured, resolve, write_timing

FIX = Path(__file__).parent / "fixtures" / "shotlist_music.json"


def _ff(*args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *map(str, args)], check=True)


def test_assemble_in_path_with_spaces_and_emoji(tmp_path):
    project = tmp_path / "Codex Projects" / "My App 🎬"
    run = new_run(project, "launch", "saas-kinetic", "16:9", None, "en")
    data = json.loads(FIX.read_text())
    data["canvas"] = {"aspect": "16:9", "width": 320, "height": 180, "fps": 30}
    run.shotlist.write_text(json.dumps(data))
    takes = {}
    for lid, secs in (("c01-l1", 1.5), ("c02-l1", 1.0), ("c02-l2", 1.4)):
        _ff("-f", "lavfi", "-i", f"sine=frequency=300:duration={secs}:sample_rate=48000", run.vo_dir / f"{lid}.wav")
        takes[lid] = LineTake(id=lid, file=f"audio/vo/{lid}.wav", duration_s=secs)
    pack = load_pack("saas-kinetic")
    t = resolve(Shotlist.model_validate(data), Measured(takes=takes), pack.limits(), pack.text_mode())
    write_timing(t, run.timing)
    (run.renders / "shots").mkdir(parents=True, exist_ok=True)
    for s in t.shots[:2]:  # the third shot stays missing → background colour
        _ff("-f", "lavfi", "-i", f"testsrc=size=320x180:rate=30:duration={s.duration_s}", "-pix_fmt", "yuv420p",
            run.renders / "shots" / f"{s.id}.mp4")
    _ff("-f", "lavfi", "-i", "sine=frequency=110:duration=12:sample_rate=48000", "-ac", "2", run.music_dir / "bed.wav")
    render_stem(t, pack, run.sfx_dir / "track.wav")

    out = asm.assemble(run, t, pack)

    assert out == run.renders / "master.mp4"
    assert aa.duration_s(out) == pytest.approx(8.3, abs=0.06)
    m = aa.loudness(out)
    assert m["TP"] <= -1.0 and m["I"] == pytest.approx(-14, abs=1.5)
    report = json.loads((run.renders / "master.loudness.json").read_text())
    assert report["TP"] == pytest.approx(m["TP"], abs=0.05)
    assert run.timeline.exists()
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,r_frame_rate", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True, check=True).stdout.strip()
    assert probe == "320,30/1"


def test_true_peak_retry_lowers_ceiling(tmp_path, monkeypatch):
    ceilings = []
    monkeypatch.setattr(asm, "loudnorm_measure", lambda *a, **kw: {"input_i": "-20.0"})   # measured once, up front
    monkeypatch.setattr(asm, "master_audio", lambda src, dst, **kw: ceilings.append(kw["ceiling_db"]) or dst)
    monkeypatch.setattr(asm, "mux", lambda pic, aud, out, **kw: out)
    readings = iter([{"I": -14, "LRA": 3, "TP": -0.6}, {"I": -14, "LRA": 3, "TP": -1.2}])
    monkeypatch.setattr(asm, "loudness", lambda p: next(readings))
    assert asm._master_loop(tmp_path / "pre.wav", tmp_path / "pic.mp4", tmp_path / "o.mp4", tmp_path,
                            target_i=-14, tp_limit=-1.0, draft=False)["TP"] == -1.2
    assert ceilings == [-2.0, -2.5]


def test_true_peak_gives_up(tmp_path, monkeypatch):
    monkeypatch.setattr(asm, "loudnorm_measure", lambda *a, **kw: {"input_i": "-20.0"})
    monkeypatch.setattr(asm, "master_audio", lambda src, dst, **kw: dst)
    monkeypatch.setattr(asm, "mux", lambda pic, aud, out, **kw: out)
    monkeypatch.setattr(asm, "loudness", lambda p: {"I": -14, "LRA": 3, "TP": -0.2})
    with pytest.raises(asm.AssembleError, match="true peak"):
        asm._master_loop(tmp_path / "pre.wav", tmp_path / "pic.mp4", tmp_path / "o.mp4", tmp_path,
                         target_i=-14, tp_limit=-1.0, draft=False)


# --- picture and sound stay together, in the master and in the draft made from 60 fps shots ----------------------

def _music_run(tmp_path, fps=60):
    run = new_run(tmp_path / "proj", "sync", "saas-kinetic", "16:9", None, "en")
    data = json.loads(FIX.read_text())
    data["canvas"] = {"aspect": "16:9", "width": 320, "height": 180, "fps": fps}
    run.shotlist.write_text(json.dumps(data))
    pack = load_pack("saas-kinetic")
    t = resolve(Shotlist.model_validate(data), Measured(), pack.limits(), pack.text_mode())
    t.findings = []
    write_timing(t, run.timing)
    return run, t, pack


def _first_bright_frame(mp4: Path, after: int = 0) -> int:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-vf", "scale=8:4", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    luma = np.frombuffer(raw, np.uint8).reshape(-1, 32).mean(axis=1)
    return after + int(np.argmax(luma[after:] > 128))


def _first_burst_s(mp4: Path) -> float:
    x = aa.decode_mono(mp4)
    jump = np.abs(np.diff(x))                       # the noise burst, not the 110 Hz bed underneath it
    return int(np.argmax(jump > 0.3 * jump.max())) / 48000


@pytest.mark.parametrize("draft", [False, True])
def test_picture_and_sound_stay_in_sync(tmp_path, draft):
    # Shots are rendered at 60 fps (the timeline rate) even for the draft, which must still come out at 30 fps and
    # half size. A noise burst in the SFX stem sits exactly on the cut from a black shot to a white one; in the MP4
    # the first white frame and the burst must agree to within a frame.
    run, t, pack = _music_run(tmp_path)
    shot_dir = run.renders / ("draft/shots" if draft else "shots")
    shot_dir.mkdir(parents=True, exist_ok=True)
    for s, colour in zip(t.shots[:2], ("black", "white")):
        _ff("-f", "lavfi", "-i", f"color=c={colour}:s=320x180:r=60:d={s.duration_s}", "-pix_fmt", "yuv420p",
            shot_dir / f"{s.id}.mp4")
    cut = t.shot("c02-s1").start_s
    sr = 48000
    bed = 0.05 * np.sin(2 * np.pi * 110 * np.arange(10 * sr) / sr)
    sf.write(run.music_dir / "bed.wav", np.stack([bed, bed], 1), sr, subtype="PCM_24")
    stem = np.zeros(round(t.duration_s * sr))
    stem[round(cut * sr):round(cut * sr) + 1200] = np.random.default_rng(3).uniform(-0.5, 0.5, 1200)
    sf.write(run.sfx_dir / "track.wav", stem, sr, subtype="PCM_24")

    out = asm.assemble(run, t, pack, draft=draft)

    fps = 30 if draft else 60
    assert out == run.renders / ("draft/animatic.mp4" if draft else "master.mp4")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True, check=True).stdout.strip()
    assert probe == ("160,90,30/1" if draft else "320,180,60/1")
    assert aa.duration_s(out) == pytest.approx(t.duration_s, abs=0.03)
    video_cut = _first_bright_frame(out) / fps
    audio_cut = _first_burst_s(out)
    assert video_cut == pytest.approx(cut, abs=0.5 / fps)
    assert audio_cut == pytest.approx(cut, abs=0.005)
    assert abs(video_cut - audio_cut) < 1 / fps
    report = json.loads(out.with_name(out.stem + ".loudness.json").read_text())
    assert report["TP"] <= -1.0
    assert report["normalization"] == "linear-gain+limiter" and isinstance(report["gain_db"], float)
    assert (run.renders / "draft" / "timeline.otio").exists() == draft and run.timeline.exists() == (not draft)


def test_a_film_with_no_sound_yet_is_limited_not_normalised(tmp_path):
    # An early animatic can have shots and nothing to hear. The master chain skips loudnorm, and the report's
    # -inf true peak is written as null, so the JSON stays valid.
    run, t, pack = _music_run(tmp_path)
    (run.renders / "draft" / "shots").mkdir(parents=True)
    _ff("-f", "lavfi", "-i", f"color=c=gray:s=320x180:r=60:d={t.shots[0].duration_s}", "-pix_fmt", "yuv420p",
        run.renders / "draft" / "shots" / "c01-s1.mp4")
    out = asm.assemble(run, t, pack, draft=True)
    report = json.loads(out.with_name("animatic.loudness.json").read_text())
    assert report["TP"] is None and report["ceiling_db"] == -2.0
    assert report["normalization"] == "limiter-only" and report["gain_db"] is None
    assert not aa.decode_mono(out).any()
    assert aa.duration_s(out) == pytest.approx(t.duration_s, abs=0.03)


# --- stale shot renders --------------------------------------------------------------------------------------------

def _hf_shots(run, t, pack, *, draft=False, stale=()):
    """Two shot renders with the sidecar `promo scenes` writes; the `stale` ones carry the hash of other inputs."""
    from promo.hyperframes import ensure_shot, scaffold_scenes
    from promo.proc import write_sidecar
    from promo.scenes import design_for, shot_hash

    scaffold_scenes(run.scenes)
    out_dir = run.renders / ("draft/shots" if draft else "shots")
    out_dir.mkdir(parents=True, exist_ok=True)
    fps, quality = (30, "draft") if draft else (t.fps, "high")
    for s in t.shots[:2]:
        ensure_shot(run.scenes, s.id)
        out = out_dir / f"{s.id}.mp4"
        _ff("-f", "lavfi", "-i", f"color=c=gray:s=320x180:r={fps}:d={s.duration_s}", "-pix_fmt", "yuv420p", out)
        digest = "0" * 64 if s.id in stale else shot_hash(run.scenes, t, s.id, design_for(run, pack), quality, fps)
        write_sidecar(out, tool="hyperframes", version="0.8.98", flags=[], inputs={"hash": digest})


def test_a_stale_shot_render_stops_the_master_unless_allowed(tmp_path):
    run, t, pack = _music_run(tmp_path)
    _hf_shots(run, t, pack, stale=("c02-s1",))
    with pytest.raises(asm.StaleRenders, match=r"c02-s1.*run `promo scenes` first") as e:
        asm.assemble(run, t, pack)
    assert "c01-s1" not in str(e.value)
    assert not (run.renders / "master.mp4").exists()            # refused before anything was rendered
    warnings = []
    out = asm.assemble(run, t, pack, allow_stale=True, warn=warnings.append)
    assert out.exists() and len(warnings) == 1 and "c02-s1" in warnings[0]


def test_a_stale_shot_render_only_warns_in_the_draft(tmp_path):
    run, t, pack = _music_run(tmp_path)
    _hf_shots(run, t, pack, draft=True, stale=("c01-s1",))
    warnings = []
    out = asm.assemble(run, t, pack, draft=True, warn=warnings.append)
    assert out.exists() and len(warnings) == 1 and "c01-s1" in warnings[0] and "c02-s1" not in warnings[0]


def test_current_shot_renders_pass_without_a_warning(tmp_path):
    run, t, pack = _music_run(tmp_path)
    _hf_shots(run, t, pack)
    warnings = []
    asm.assemble(run, t, pack, warn=warnings.append)
    assert warnings == []


def test_a_render_without_a_record_is_flagged_as_unchecked_not_stale(tmp_path):
    # A render with no sidecar was not made by `promo scenes`, so its inputs cannot be compared: say so, and go on.
    run, t, pack = _music_run(tmp_path)
    _hf_shots(run, t, pack)
    (run.renders / "shots" / "c01-s1.mp4.render.json").unlink()
    warnings = []
    asm.assemble(run, t, pack, warn=warnings.append)
    assert len(warnings) == 1 and "c01-s1" in warnings[0] and "cannot" in warnings[0]


def test_cli_assemble_with_a_stale_render_exits_2_and_allow_stale_goes_on(tmp_path, monkeypatch):
    run, t, pack = _music_run(tmp_path)
    _hf_shots(run, t, pack, stale=("c01-s1",))
    monkeypatch.chdir(run.root)
    res = CliRunner().invoke(app, ["assemble"])
    assert res.exit_code == 2
    assert _err_lines(res) == ["promo: stale shot renders: c01-s1 (its timing, scene or design changed since it was "
                               "rendered); run `promo scenes` first, or pass --allow-stale"]
    res = CliRunner().invoke(app, ["assemble", "--allow-stale"])
    assert res.exit_code == 0, res.output
    assert res.stdout.splitlines()[0].startswith("warning  stale shot renders: c01-s1")
    assert (run.renders / "master.mp4").exists()


# --- the command ---------------------------------------------------------------------------------------------------

def _cli_run(tmp_path, monkeypatch, *, timing=True, shots=("c01-s1",)):
    run, t, pack = _music_run(tmp_path)
    if not timing:
        run.timing.unlink()
    (run.renders / "shots").mkdir(parents=True, exist_ok=True)
    for sid in shots:
        (run.renders / "shots" / f"{sid}.mp4").write_bytes(b"x")
    monkeypatch.chdir(run.root)
    return run, t


def _err_lines(res) -> list[str]:
    return [l for l in res.stderr.splitlines() if l.strip()]


def test_cli_assemble_without_timing_exits_2(tmp_path, monkeypatch):
    _cli_run(tmp_path, monkeypatch, timing=False)
    res = CliRunner().invoke(app, ["assemble"])
    assert res.exit_code == 2 and _err_lines(res) == ["promo: run `promo timing` first"]


@pytest.mark.parametrize("draft", [False, True])
def test_cli_assemble_without_any_shot_render_exits_2(tmp_path, monkeypatch, draft):
    _cli_run(tmp_path, monkeypatch, shots=())
    res = CliRunner().invoke(app, ["assemble", *(["--draft"] if draft else [])])
    assert res.exit_code == 2
    assert _err_lines(res) == [f"promo: no shot renders in {'renders/draft/shots' if draft else 'renders/shots'}; "
                               f"run `promo scenes{' --draft' if draft else ''}` first"]


def test_cli_assemble_refuses_timing_with_errors(tmp_path, monkeypatch):
    run, t = _cli_run(tmp_path, monkeypatch)
    data = json.loads(run.timing.read_text())
    data["findings"] = [{"level": "error", "code": "late_hook", "message": "x"}]
    run.timing.write_text(json.dumps(data))
    res = CliRunner().invoke(app, ["assemble"])
    assert res.exit_code == 1 and _err_lines(res)[0].startswith("promo: timing.json has errors")


def test_cli_assemble_overlapping_shots_is_one_line_exit_1(tmp_path, monkeypatch):
    run, t = _cli_run(tmp_path, monkeypatch)
    t.shots[1].start_s = t.shots[0].end_s - 0.5
    write_timing(t, run.timing)
    res = CliRunner().invoke(app, ["assemble"])
    assert res.exit_code == 1
    lines = _err_lines(res)
    assert len(lines) == 1 and lines[0].startswith("promo: shot c02-s1 starts at")


def test_cli_assemble_ffmpeg_failure_is_one_line_exit_1(tmp_path, monkeypatch):
    run, _ = _cli_run(tmp_path, monkeypatch)

    def boom(cmd, **kw):
        raise ToolError("command failed (1): ffmpeg -v error -y -i a.mp4 -filter_complex [0:v]...\n"
                        "[fc#0 @ 0x1] Error reinitializing filters!\nError opening output files: Invalid argument")

    monkeypatch.setattr(asm, "run", boom)
    res = CliRunner().invoke(app, ["assemble"])
    assert res.exit_code == 1
    assert _err_lines(res) == ["promo: ffmpeg could not render timeline.otio: Error opening output files: "
                               "Invalid argument (full output: logs/last-error.log)"]
    assert "Error reinitializing filters!" in (run.root / "logs" / "last-error.log").read_text()


def test_cli_assemble_prints_the_master_and_its_loudness(tmp_path, monkeypatch):
    run, _ = _cli_run(tmp_path, monkeypatch)

    def fake(run_paths, timing, pack, *, draft=False, allow_stale=False, warn=None):
        out = run_paths.renders / "master.mp4"
        out.write_bytes(b"mp4")
        out.with_name("master.loudness.json").write_text('{"I": -14.0, "LRA": 3.0, "TP": -1.4, "ceiling_db": -2.0}')
        return out

    monkeypatch.setattr(asm, "assemble", fake)
    res = CliRunner().invoke(app, ["assemble"])
    assert res.exit_code == 0, res.output
    assert res.output.splitlines()[0] == str(run.renders / "master.mp4")
    assert '"TP": -1.4' in res.output and "not by ear" in res.output
