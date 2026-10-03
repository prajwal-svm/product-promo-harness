"""One critic round: the evidence pack, automatic checks, and the prompt for a fresh critic subagent."""

from __future__ import annotations

import errno
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from promo import asr, hyperframes
from promo.audio_analysis import loudness, spectrogram
from promo.config import packs_dir
from promo.evidence import (
    contact_sheet,
    fastest_action_t,
    frames_at,
    has_audio,
    master_sheet,
    phone_sheet,
    strip,
    sync_report,
    tech_checks,
)
from promo.lint_script import lint_shotlist
from promo.models import Finding
from promo.packs import Pack
from promo.proc import sha256_file
from promo.run import RunPaths
from promo.sfx import Placement, SfxError, event_clip
from promo.shotlist import load_shotlist
from promo.timing import Timing
from promo.verify import stale_shots, verifiable_shots
from promo.voice import voice_settings

LUFS_TOLERANCE = 1.0      # the loudness check fails beyond this many LU from the target...
LUFS_UNDER_WARN = 0.5     # ...and warns when the film is more than this far under it
SILENT_LUFS = -70.0       # ebur128 reports this, or -inf, when nothing is loud enough to measure
NO_SOUND = "n/a (no sound planned)"

PROMPT = """Open the contact sheets, the strips and the phone sheet in evidence/ and look at them properly.
Be a harsh motion director, not a proud author. Score 1-10: hook in the first 2 s · readability at phone
size · motion quality (springs, no dead frames) · variety (a new thing every 2-4 s) · composition · brand
accuracy · sound sync. List the 3 biggest problems with timestamps. Hunt specifically for: text overlapping
during swaps, anything sliding instead of easing, corner labels and frame borders, centred-on-gradient shots,
blurry scaled text, a dead beat with nothing happening, a stutter at a seam or loop point, and every item on
the pack checklist, the common failures and the banned list below. Would a motion designer put each shot on
their reel?"""

COMMON_FAILURES = [
    "On-screen text repeating the VO (unless the shot's text mode is verbatim-sync)",
    "Constant speed with no holds",
    "Reads that flash by",
    "Twinning (paired elements moving identically)",
    "Snapped state changes",
    "Dead stretches",
    "An empty frame 0",
    "Overlays moving with the camera",
    "Key action inside the caption band",
    "Invented UI or unsourced numbers",
    "Any blur: a blur filter, depth of field, motion blur, a blurred background or a blur-in entrance",
    "A vignette, white inner shadow, glow, feathered edge or grey scrim",
]


class CritiqueError(RuntimeError):
    """The round cannot be built from this run's files."""


class MissingMaster(CritiqueError):
    """There is no video to judge yet: a missing prerequisite, not a failure."""


ROUND_ATTEMPTS = 5   # how many round numbers a finished round tries when concurrent rounds take the ones it picked


def rounds(run: RunPaths) -> list[int]:
    """The numbers of the finished rounds in critic/, in order; nothing is created. A staging folder is never one."""
    if not run.critic.is_dir():
        return []
    return sorted(int(p.name) for p in run.critic.iterdir() if p.is_dir() and p.name.isdigit())


def next_round(run: RunPaths) -> int:
    run.critic.mkdir(parents=True, exist_ok=True)
    return max(rounds(run), default=0) + 1


def _pack_checklist(pack: Pack) -> list[str]:
    md = (pack.root / "pack.md").read_text() if (pack.root / "pack.md").exists() else ""
    m = re.search(r"## Quality checklist\n(.*?)(\n## |\Z)", md, flags=re.S)
    return [l[2:].strip() for l in (m.group(1).splitlines() if m else []) if l.startswith("- ")]


def _banned(pack: Pack) -> list[str]:
    text = (packs_dir() / "_core" / "banned.md").read_text()
    allowed = pack.allowed()
    rows = re.findall(r"^\| ([a-z0-9-]+) \| (.+?) \|$", text, flags=re.M)
    return [f"{i}: {desc}" for i, desc in rows if i != "id" and i not in allowed]


def planned_sound(timing: Timing, pack: Pack) -> dict:
    """The sound the film plans: its VO lines, its SFX placements (events with a pack sound, one each in the stem) and
    a music bed (the pack's music mode is not `none`). A film silent by design plans none of the three."""
    return {"vo_lines": len(timing.lines),
            "sfx_placements": sum(1 for ev in timing.events if isinstance(pack.sfx.get(ev.sfx or ev.kind), dict)),
            "music": pack.music.get("mode", "bed") != "none"}


def missing_sound(run: RunPaths, timing: Timing, pack: Pack) -> list[str]:
    """Each planned sound whose takes, stem or bed is not on disk, with the step that makes it."""
    planned, out = planned_sound(timing, pack), []
    if planned["vo_lines"]:
        no_take = [l.id for l in timing.lines if not l.file or not (run.root / l.file).is_file()]
        if no_take:
            out.append(f"VO planned ({planned['vo_lines']} lines) but {len(no_take)} "
                       f"{'has' if len(no_take) == 1 else 'have'} no take: run `promo voice --all`")
    if planned["sfx_placements"] and not (run.sfx_dir / "track.wav").exists():
        out.append(f"SFX planned ({planned['sfx_placements']} placements) but no sfx stem: run `promo sfx render`")
    if planned["music"] and not (run.music_dir / "bed.wav").exists():
        out.append(f"music planned (pack music mode '{pack.music.get('mode', 'bed')}') but no bed: run `promo music`")
    return out


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _finite(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def loudness_check(loud: dict | None, target: float, tp_limit: float, report: dict | None = None, *,
                   sound_planned: bool = True, missing: list[str] | tuple = ()) -> tuple[dict, list[Finding]]:
    """The loudness check and its findings, from the loudness measured on the MP4 (`loud`, None without an audio
    track) and, when there is one, the report `promo assemble` wrote next to it (`report`).

    It passes within 1 LU of the target with the true peak at or under the limit. A film more than 0.5 LU under the
    target is also a warning, whether or not it passes: a mix that is only short sounds is too peaky to reach the
    target under the limiter's ceiling, and the report says what `assemble` did about it. A film silent by design
    (`sound_planned` false: no VO lines, no SFX placements, pack music mode `none`) passes with no audio or silent
    audio. With sound planned, missing or silent audio fails, naming each planned sound that is `missing` and the step
    that makes it.
    """
    i = _finite(loud.get("I")) if loud is not None else None
    if not sound_planned and (i is None or i <= SILENT_LUFS):
        return _check("loudness", True, NO_SOUND), []
    if loud is None:
        return _check("loudness", False, "; ".join(["the video has no audio track", *missing])), []
    tp = loud.get("TP")
    if i is None or i <= SILENT_LUFS:
        return _check("loudness", False, "; ".join(["the audio is silent: there is no loudness to measure", *missing])), []
    ok = abs(i - target) <= LUFS_TOLERANCE and isinstance(tp, (int, float)) and tp <= tp_limit + 1e-9
    detail = f"I {i} LUFS (target {target}), TP {tp} dBTP (limit {tp_limit})"
    findings: list[Finding] = []
    under = target - i
    if under > LUFS_UNDER_WARN:
        how, report = "", report or {}
        gain, ceiling = _finite(report.get("gain_db")), report.get("ceiling_db")
        if report.get("normalization") == "linear-gain+limiter" and gain is not None:
            how = (f" (assemble applied {gain:+.1f} dB, then limited the peaks at {ceiling} dBFS and so took it back: "
                   f"a mix of short sounds with no bed or VO is too peaky to reach the target under that ceiling)")
        elif report.get("normalization") == "limiter-only":
            how = " (assemble found no loudness to correct, so it only limited the peaks)"
        message = f"integrated loudness is {under:.1f} LU under the {target} LUFS target{how}"
        findings.append(Finding(level="warning", code="loudness_under_target", message=message))
        detail += f"; {under:.1f} LU under target"
    return _check("loudness", ok, detail), findings


def _determinism_check(run: RunPaths, timing: Timing, pack: Pack) -> dict:
    f = run.renders / "verify.json"
    if not f.exists():
        return _check("determinism", False, "not run: `promo verify`")
    v = _read_json(f)
    if v is None or not isinstance(v.get("shots"), list):
        return _check("determinism", False, "renders/verify.json cannot be read; run `promo verify` again")
    if not v.get("ok"):
        return _check("determinism", False, json.dumps(v.get("mismatches")))
    missing = [i for i in verifiable_shots(timing) if i not in v["shots"]]
    if missing:
        return _check("determinism", False, f"not verified: {', '.join(missing)}; run `promo verify`")
    stale = stale_shots(run, timing, pack, v.get("fingerprints") or {})
    if stale:
        return _check("determinism", False, f"changed since `promo verify`: {', '.join(stale)}; run it again")
    return _check("determinism", True, "ok")


def _hf_check(run: RunPaths, timing: Timing, ev: Path) -> dict:
    """HyperFrames' own check of each shot's render project. `check` runs lint first and carries its result, so lint
    is not run again here."""
    results, unlinted = {}, []
    for s in timing.shots:
        if s.engine != "hyperframes":
            continue
        project = run.render_dir / s.id
        if not project.exists():
            unlinted.append(s.id)
            continue
        results[s.id] = hyperframes.check(project)
    (ev / "hf.json").write_text(json.dumps(results, indent=2))
    bad = [sid for sid, r in results.items() if (r.get("lint") or {}).get("errorCount", 0) > 0]
    parts = [f"lint errors in {', '.join(bad)}"] if bad else []
    if unlinted:
        parts.append(f"no render project in scenes/.render for {', '.join(unlinted)}")
    return _check("hf_lint", not bad and (bool(results) or not unlinted), "; ".join(parts) or "ok")


def _sfx_detail(sfx: dict) -> str:
    if "detail" in sfx:
        return sfx["detail"]
    late = [r for r in sfx["events"] if not r["ok"]]
    if not late:
        found = sum(not r.get("unmeasurable") for r in sfx["events"])
        return f"{found} of {len(sfx['events'])} found within {sfx['tolerance_ms']} ms"
    return "; ".join(f"{r['event']}: {r['issue']}" if "issue" in r else f"{r['event']} {r['offset_ms']:+.1f} ms"
                     for r in late)


def _vo_detail(vo: dict) -> str:
    late = [r for r in vo["rows"] if not r["ok"]]
    if not late:
        return f"{len(vo['rows'])} lines found within {vo['tolerance_ms']} ms"
    return "; ".join(f"{r['line']}: {r['issue']}" if "issue" in r else f"{r['line']} {r['offset_ms']:+.1f} ms"
                     for r in late)


def _no_paid_sound(prompt: str, duration: float) -> bytes:
    raise SfxError("a generated sound that is not in the cache would be a paid call; run `promo sfx render`")


def _clips(timing: Timing, pack: Pack, placements: list[Placement]) -> dict:
    """Each placed event's sound as it sits in the stem, made again (synthesised or read from the cache; never paid for).
    An event whose sound cannot be made again is left out."""
    events, memo, out = {e.id: e for e in timing.events}, {}, {}
    for p in placements:
        if p.event in events:
            try:
                got = event_clip(events[p.event], pack, gen=_no_paid_sound, memo=memo)
            except SfxError:
                continue
            if got is not None:
                out[p.event] = got
    return out


def _read_placements(run: RunPaths) -> tuple[list[Placement], str | None]:
    """(placements, why they cannot be used). No stem is reported by the SFX check itself."""
    f = run.sfx_dir / "placements.json"
    if not (run.sfx_dir / "track.wav").exists():
        return [], None
    try:
        return [Placement(**p) for p in json.loads(f.read_text())], None
    except FileNotFoundError:
        return [], "no audio/sfx/placements.json beside the stem; run `promo sfx render`"
    except (OSError, ValueError, TypeError) as e:
        return [], f"audio/sfx/placements.json cannot be read ({type(e).__name__}); run `promo sfx render`"


def critique(run: RunPaths, timing: Timing, pack: Pack, *, video: Path | None = None,
             transcribe=asr.transcribe, hf: bool = True) -> Path:
    """Build critic/<n>/: the evidence, summary.json (the automatic checks) and PROMPT.md. The round appears whole
    or not at all, so a failure part way never uses up a round number."""
    video = Path(video) if video else run.renders / "master.mp4"
    if not video.is_file():
        raise MissingMaster(f"no video at {video}; run `promo assemble` first, or pass --video")
    sl = load_shotlist(run.shotlist)
    n = next_round(run)
    stage = Path(tempfile.mkdtemp(dir=run.critic, prefix=f".{n}."))   # its own name, so concurrent runs cannot clobber it
    stage.chmod(0o755)
    (stage / "evidence").mkdir()
    try:
        _build_round(run, timing, pack, sl, video, n, stage, transcribe=transcribe, hf=hf)
        return _land(run, stage, n)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _land(run: RunPaths, stage: Path, n: int) -> Path:
    """Move the finished round into critic/<n>/. When a concurrent round has taken n meanwhile, the round is renumbered
    to the next free number and tried again, at most ROUND_ATTEMPTS times."""
    for _ in range(ROUND_ATTEMPTS):
        rd = run.critic / str(n)
        try:
            os.replace(stage, rd)    # a rename onto a folder that holds anything fails; nothing is ever merged
            return rd
        except OSError as e:
            if e.errno not in (errno.ENOTEMPTY, errno.EEXIST) or not rd.exists():
                raise
        n = next_round(run)
        _renumber(stage, n)
    raise CritiqueError(f"concurrent critic rounds kept taking the next number; tried {ROUND_ATTEMPTS} times")


def _renumber(stage: Path, n: int) -> None:
    summary = json.loads((stage / "summary.json").read_text())
    summary["round"] = n
    (stage / "summary.json").write_text(json.dumps(summary, indent=2))
    prompt = (stage / "PROMPT.md").read_text()
    (stage / "PROMPT.md").write_text(re.sub(r"\A# Critic round \d+", f"# Critic round {n}", prompt))


def _build_round(run: RunPaths, timing: Timing, pack: Pack, sl, video: Path, n: int, rd: Path, *,
                 transcribe, hf: bool) -> None:
    ev = rd / "evidence"
    audio = has_audio(video)
    on_beats = [b for b in timing.beats if b <= timing.duration_s - 0.02] if timing.timing_mode == "music" else []
    times = on_beats or [float(t) for t in np.arange(0, timing.duration_s, 0.5)]
    with tempfile.TemporaryDirectory() as d:
        frames = frames_at(video, times, Path(d))
        for i in range(0, len(frames), 36):
            contact_sheet([(f"{t:.2f}s", p) for t, p in frames[i:i + 36]], ev / f"contact-{i // 36:02d}.png")
    for s in timing.shots[1:]:
        strip(video, s.start_s, timing.fps, ev / f"strip-seam-{s.id}.png")
    strip(video, fastest_action_t(video), timing.fps, ev / "strip-fastest.png")
    phone_sheet(video, ev / "phone.png")
    master_sheet(video, ev / "master.png")
    loud = None
    if audio:
        spectrogram(video, ev / "spectrogram.png")
        loud = loudness(video)
    target, tp_limit = pack.music.get("lufs", -14), pack.music.get("true_peak", -1.0)
    report = _read_json(video.with_name(video.stem + ".loudness.json"))
    (ev / "loudness.json").write_text(json.dumps(
        {"measured": {k: _finite(v) for k, v in loud.items()} if loud else None, "target_lufs": target,
         "true_peak_limit_dbtp": tp_limit, "assemble_report": report}, indent=2))

    stem = run.sfx_dir / "track.wav"
    silent_by_design = not any(planned_sound(timing, pack).values())
    placements, placements_problem = _read_placements(run)
    skipped = [line.id for line in sl.all_lines() if not line.verify]
    takes = {l.id: run.root / l.file for l in timing.lines if l.file}
    sync = sync_report(timing, stem, placements, video, sl.language, transcribe=transcribe,
                       aliases=voice_settings(sl, pack).aliases, skip_lines=skipped, takes=takes,
                       clips=_clips(timing, pack, placements))
    if placements_problem:
        sync["sfx"] = {"ok": False, "events": [], "detail": placements_problem}
    elif silent_by_design:
        sync["sfx"] = {"ok": True, "events": [], "detail": NO_SOUND}
    (ev / "sync.json").write_text(json.dumps(sync, indent=2))
    tech = tech_checks(video, width=timing.width, height=timing.height, fps=timing.fps,
                       max_static_s=timing.limits.max_dead_s)
    (ev / "tech.json").write_text(json.dumps(tech, indent=2))
    claims = [f.message for f in lint_shotlist(sl, pack.register(), pack.text_mode()) if f.code == "unsourced_number"]

    loud_check, findings = loudness_check(loud, target, tp_limit, report, sound_planned=not silent_by_design,
                                          missing=missing_sound(run, timing, pack))
    unmeasured = [r["event"] for r in sync["sfx"]["events"] if r.get("unmeasurable")]
    if unmeasured:
        findings.append(Finding(level="warning", code="sfx_unmeasurable", message=(
            f"{', '.join(unmeasured)}: the sound could not be found in the stem to measure it (it is not reported as "
            f"late); `promo sfx render` makes the stem again")))
    if skipped:
        findings.append(Finding(level="warning", code="vo_not_scored", message=(
            f"{', '.join(skipped)}: verify is false, so speech recognition did not score these lines on the mix")))
    checks = [
        _check("tech", tech["ok"], "; ".join(tech["issues"]) or "ok"),
        loud_check,
        _check("sync_sfx", sync["sfx"]["ok"], _sfx_detail(sync["sfx"])),
        _check("sync_vo", sync["vo"]["ok"], _vo_detail(sync["vo"])),
        _check("vo_wer", sync["lines"]["ok"],
               ", ".join(f"{r['line']} {r['wer']}" for r in sync["lines"]["rows"] if not r["ok"]) or "ok"),
        _check("claims", not claims, "; ".join(claims) or "ok"),
        _determinism_check(run, timing, pack),
    ]
    if hf:
        checks.append(_hf_check(run, timing, ev))
    summary = {"round": n, "video": str(video.resolve()), "video_sha256": sha256_file(video),
               "ok": all(c["ok"] for c in checks), "checks": checks, "findings": [f.model_dump() for f in findings]}
    (rd / "summary.json").write_text(json.dumps(summary, indent=2))
    evidence = sorted(p.relative_to(rd).as_posix() for p in ev.glob("*") if p.is_file())
    body = [
        f"# Critic round {n}", "", f"Video: {video}", "", PROMPT, "",
        "## Pack checklist", *[f"- {x}" for x in _pack_checklist(pack)], "",
        "## Common failures", *[f"- {x}" for x in COMMON_FAILURES], "",
        "## Banned (not re-allowed by this pack)", *[f"- {x}" for x in _banned(pack)], "",
        "## Automatic checks", *[f"- {'PASS' if c['ok'] else 'FAIL'} {c['name']}: {c['detail']}" for c in checks], "",
        *(["## Warnings", *[f"- {f.message}" for f in findings], ""] if findings else []),
        "## Evidence", *[f"- {e}" for e in evidence], "",
        "## Write back",
        "Write `review.md` (per-shot verdict and concrete fixes) and `scores.json` next to this file:",
        '```json\n{"scores": {"hook": 0, "readability": 0, "motion": 0, "variety": 0, "composition": 0, "brand": 0,'
        ' "sync": 0}, "problems": [{"t": 0.0, "shot": "", "problem": "", "fix": ""}], "checklist": {}, "pass": false}\n```',
        "Pass only if every score is 8 or more and every checklist item passes. Audio was checked by measurement, not by ear.",
    ]
    (rd / "PROMPT.md").write_text("\n".join(body) + "\n")
