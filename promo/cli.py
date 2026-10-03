"""The `promo` command line. Each module registers its commands here."""

from __future__ import annotations

import typer

from promo.config import load_env

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Product promo harness.")


@app.callback()
def main() -> None:
    """Build showreel-grade promo videos from a run folder."""
    load_env()


@app.command()
def version() -> None:
    """Print the harness version."""
    from promo import __version__

    typer.echo(__version__)


from pathlib import Path
from typing import TYPE_CHECKING, Optional

from promo.models import Finding

if TYPE_CHECKING:
    from promo.packs import Pack
    from promo.run import RunPaths
    from promo.shotlist import Shotlist
    from promo.timing import Timing


def _run_or_exit() -> "RunPaths":
    from promo.run import RunNotFound, find_run

    try:
        return find_run()
    except RunNotFound as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2)


def _load_run_context() -> "tuple[RunPaths, Shotlist, Pack]":
    """Find the run folder, then load its shot list and pack; a bad file is a one-line error, exit 1."""
    from promo.packs import PackError, load_pack
    from promo.shotlist import ShotlistError, load_shotlist

    run = _run_or_exit()
    try:
        sl = load_shotlist(run.shotlist)
        pack = load_pack(sl.pack)
    except (ShotlistError, PackError) as e:
        typer.echo(f"promo: {e}", err=True)
        raise typer.Exit(1)
    return run, sl, pack


def print_findings(findings: list[Finding]) -> tuple[int, int]:
    errors = [f for f in findings if f.level == "error"]
    warnings = [f for f in findings if f.level == "warning"]
    for f in errors + warnings:
        typer.echo(f"{f.level:<8} {f.code:<20} {f.message}")
    typer.echo(f"{len(errors)} errors, {len(warnings)} warnings")
    return len(errors), len(warnings)


@app.command()
def new(
    slug: str,
    pack: str = typer.Option("saas-kinetic", help="Design-system pack."),
    aspect: str = typer.Option("16:9", help="16:9, 9:16 or 1:1."),
    length: Optional[float] = typer.Option(None, help="Target length in seconds."),
    lang: str = typer.Option("en", help="Narration language code."),
) -> None:
    """Scaffold a run folder at ./promo/<slug>."""
    from promo.run import new_run

    try:
        run = new_run(Path.cwd(), slug, pack, aspect, length, lang)
    except (ValueError, FileExistsError) as e:
        typer.echo(f"promo: {e}", err=True)
        raise typer.Exit(2)
    typer.echo(str(run.root))


@app.command()
def packs() -> None:
    """List design-system packs."""
    from promo.packs import list_packs

    for p in list_packs():
        typer.echo(f"{p.name:<22} register={p.register():<10} text={p.text_mode():<14} "
                   f"timing={p.timing_mode():<6} fps={p.tokens.get('fps', 60)}")


@app.command()
def timing() -> None:
    """Resolve shotlist.json + measured audio into timing.json."""
    from promo.timing import TimingError, load_measured, resolve, write_timing

    run, sl, pack = _load_run_context()
    try:
        t = resolve(sl, load_measured(run.root), pack.limits(), pack.text_mode())
    except TimingError as e:
        typer.echo(f"promo: {e}", err=True)
        raise typer.Exit(1)
    write_timing(t, run.timing)
    errors, _ = print_findings(t.findings)
    typer.echo(f"timing.json: {t.duration_s:.2f}s, {len(t.shots)} shots, {len(t.events)} events")
    if errors:
        raise typer.Exit(1)


@app.command("lint-script")
def lint_script_cmd(
    text: Optional[Path] = typer.Option(None, help="Lint a prose file instead of the run's shotlist."),
    register: Optional[str] = typer.Option(None, help="Register (ad-sparse, ad-dense, professor)."),
) -> None:
    """Anti-Claudism, register and claims checks."""
    from promo.lint_script import lint_shotlist, lint_text

    if text is not None:
        try:
            body = text.read_text()
        except OSError as e:
            typer.echo(f"promo: {e}", err=True)
            raise typer.Exit(2)
        findings = lint_text(body, register or "professor", ref=text.name)
    else:
        _, sl, pack = _load_run_context()
        findings = lint_shotlist(sl, register or pack.register(), pack.text_mode())
    errors, _ = print_findings(findings)
    if errors:
        raise typer.Exit(1)


@app.command()
def script() -> None:
    """Write script.md, the readable view of the shot list."""
    from promo.proc import atomic_path
    from promo.script_view import render_script
    from promo.timing import load_timing

    run, sl, _ = _load_run_context()
    t = load_timing(run.timing) if run.timing.exists() else None
    with atomic_path(run.script_md) as tmp:
        tmp.write_text(render_script(sl, t))
    typer.echo(str(run.script_md))


@app.command()
def beats(wav: Optional[Path] = typer.Argument(None, help="Music file (default: audio/music/bed.wav).")) -> None:
    """Measure tempo, beats and downbeats; writes audio/music/beats.json in a run.

    On a bed `promo music` made (conform.json beside it), bar 1 stays where the conform put it, on the plan's bar
    1, rather than the beat the kick energy picks: the two can disagree on a kick that falls on every beat, and
    `promo timing` would then move the film by beats.
    """
    from promo.beats import STEADY_TOL_S, analyze, write_grid
    from promo.run import RunNotFound, find_run

    try:
        run = find_run()
    except RunNotFound:
        run = None
    src = wav or (run.music_dir / "bed.wav" if run else None)
    if src is None or not Path(src).exists():
        typer.echo(f"promo: no music file ({src})", err=True)
        raise typer.Exit(2)
    beats_per_bar, bpm_hint, tol = 4, None, STEADY_TOL_S
    if run is not None:
        from promo.shotlist import ShotlistError, load_shotlist

        try:
            sl = load_shotlist(run.shotlist)
            beats_per_bar, bpm_hint = sl.music.beats_per_bar, sl.music.bpm   # the planned tempo picks the metrical level
            tol = 1.0 / sl.canvas.fps      # as `promo music` judged the bed: a wander under a frame is steady
        except (ShotlistError, OSError):
            pass  # no usable shot list: keep the common time
    try:
        grid = analyze(Path(src), beats_per_bar, bpm_hint=bpm_hint, steady_tol_s=tol)
    except ValueError as e:
        typer.echo(f"promo: {e}", err=True)
        raise typer.Exit(1)
    if run is not None:
        conform_json = run.music_dir / "conform.json"
        if Path(src).resolve() == (run.music_dir / "bed.wav").resolve() and conform_json.exists():
            import json

            from promo.music import phase_grid

            try:
                grid = phase_grid(grid, float(json.loads(conform_json.read_text())["origin_s"]), beats_per_bar)
            except (OSError, ValueError, KeyError, TypeError):
                pass   # an unreadable report: keep the measured bar phase
        write_grid(grid, run.music_dir / "beats.json")
    typer.echo(f"bpm {grid.bpm:.2f}, {len(grid.beats)} beats, first downbeat {grid.first_downbeat_s:.3f}s")


def _parse_sample(text: str) -> tuple[float, float]:
    """`12-15` -> (12.0, 15.0); anything else is a one-line usage error, exit 2."""
    try:
        a, b = (float(x) for x in text.split("-"))
    except ValueError:
        a = b = -1.0
    if not 0 <= a < b:
        typer.echo(f"promo: --sample wants A-B seconds with A < B, e.g. 12-15 (got '{text}')", err=True)
        raise typer.Exit(2)
    return a, b


@app.command()
def scenes(
    scaffold: bool = typer.Option(False, help="Create missing shot files from the base template and stop."),
    stills: bool = typer.Option(False, help="Snapshot key frames of every shot."),
    draft: bool = typer.Option(False, help="Animatic quality: 30 fps, draft encode."),
    sample: Optional[str] = typer.Option(None, help="Render only shots overlapping A-B seconds, e.g. 12-15."),
    shot: Optional[list[str]] = typer.Option(None, "--shot", help="Only these shot ids."),
    force: bool = typer.Option(False, help="Re-render even if inputs are unchanged."),
) -> None:
    """Render HyperFrames shots (one project per shot)."""
    from promo import scenes as sc
    from promo.hyperframes import HyperframesError, ensure_shot, scaffold_scenes
    from promo.proc import ToolError
    from promo.timing import load_timing

    run, _, pack = _load_run_context()
    rng = _parse_sample(sample) if sample else None
    if not run.timing.exists():
        typer.echo("promo: run `promo timing` first", err=True)
        raise typer.Exit(2)
    t = load_timing(run.timing)
    if t.errors():
        typer.echo("promo: timing.json has errors; fix them (`promo timing`) before rendering", err=True)
        raise typer.Exit(1)
    try:
        if scaffold:
            scaffold_scenes(run.scenes)
            for s in t.shots:
                if s.engine == "hyperframes":
                    typer.echo(str(ensure_shot(run.scenes, s.id)))
            return
        if stills:
            for p in sc.stills(run, t, pack):
                typer.echo(str(p))
            return
        mode = "draft" if draft else ("sample" if rng else "final")
        for p in sc.render_shots(run, t, pack, mode=mode, only=shot or None, sample=rng, force=force):
            typer.echo(str(p))
    except (HyperframesError, ToolError, OSError) as e:
        raise _provider_failure(e)


voice_app = typer.Typer(
    help="Voice-over takes and auditions. Supertonic's output varies from draw to draw and the cache pins each draw: "
         "--force re-scores the same audio for free, and changing steps or speed in the shot list's voice block gives "
         "a fresh draw.",
    invoke_without_command=True, no_args_is_help=False)
app.add_typer(voice_app, name="voice")


ERROR_LOG = "logs/last-error.log"


def _save_error_log(text: str) -> str | None:
    """Write the full failure text to <run>/logs/last-error.log; return its path as shown to the user (relative to
    the current directory), or None outside a run folder or if it cannot be written."""
    import os

    from promo.proc import atomic_path
    from promo.run import RunNotFound, find_run

    try:
        path = find_run().root / ERROR_LOG
    except RunNotFound:
        return None
    try:
        with atomic_path(path) as tmp:
            tmp.write_text(text.rstrip("\n") + "\n")
    except OSError:
        return None
    try:
        return os.path.relpath(path, Path.cwd())
    except (ValueError, OSError):   # another drive, or a current directory that no longer exists
        return str(path)


def _provider_failure(e: Exception) -> "typer.Exit":
    """A missing key, a rejected key or a failed tool is one line, exit 1; paid takes stay cached.

    A ToolError carries the command, its exit code and the tail of its stderr. The full text goes to
    <run>/logs/last-error.log and the user gets the first line and the log's relative path (the first line alone
    outside a run folder); the exception itself keeps the full text. Missing Node (no `npx` to launch) is a
    missing prerequisite, exit 2, and so are a missing `ffmpeg` or `ffprobe` and a missing `supertonic`.
    """
    from promo.proc import ToolError
    from promo.providers.supertonic import SupertonicNotInstalled

    if isinstance(e, FileNotFoundError) and e.filename in ("npx", "node"):
        typer.echo("promo: HyperFrames needs Node.js (npx not found)", err=True)
        return typer.Exit(2)
    if isinstance(e, FileNotFoundError) and e.filename in ("ffmpeg", "ffprobe"):
        typer.echo(f"promo: {e.filename} not found (install ffmpeg)", err=True)
        return typer.Exit(2)
    if isinstance(e, SupertonicNotInstalled):
        typer.echo(f"promo: {e}", err=True)
        return typer.Exit(2)
    msg = str(e)
    if isinstance(e, ToolError) and len(msg.splitlines()) > 1:
        first = msg.splitlines()[0]
        msg = f"{first} (full output: {log})" if (log := _save_error_log(msg)) else first
    typer.echo(msg if msg.startswith("promo:") else f"promo: {msg}", err=True)
    return typer.Exit(1)


@voice_app.callback()
def voice_main(
    ctx: typer.Context,
    line: Optional[list[str]] = typer.Option(None, "--line", help="Only these line ids (repeatable)."),
    all_: bool = typer.Option(False, "--all", help="Every VO line."),
    force: bool = typer.Option(False, help="Re-take even if unchanged (cached audio is reused)."),
) -> None:
    """Synthesise, verify and store VO takes (Gemini 3.8 TTS, or Supertonic on this Mac)."""
    if ctx.invoked_subcommand:
        return
    if not line and not all_:
        typer.echo("promo voice: pass --all or --line ID", err=True)
        raise typer.Exit(2)
    from promo.config import MissingKeyError
    from promo.proc import ToolError
    from promo.providers.http import ProviderError
    from promo.voice import MAX_WER, synthesize_all

    run, sl, pack = _load_run_context()
    unknown = sorted(set(line or []) - {ln.id for ln in sl.all_lines()})
    if unknown:
        typer.echo(f"promo: no VO line with id {', '.join(unknown)} in shotlist.json", err=True)
        raise typer.Exit(2)
    try:
        takes = synthesize_all(run, sl, pack, only=line or None, force=force)
    except (MissingKeyError, ProviderError, ToolError, ValueError) as e:   # ValueError: a bad setting, e.g. a language
        raise _provider_failure(e)
    bad = [t for t in takes if t.wer > MAX_WER or t.problem]
    for t in takes:
        flag = f"  CHECK ({t.problem})" if t.problem else ("  CHECK" if t.wer > MAX_WER else "")
        score = f"WER {t.wer:.2f}" if t.verified else "WER  n/a"
        typer.echo(f"{t.id:<10} {t.duration_s:6.2f}s  {score}  takes {t.takes}{flag}")
    typer.echo("Narration was verified by speech recognition, not auditioned by the model.")
    skipped = [t.id for t in takes if not t.verified]
    if skipped:
        typer.echo(f"Not verified (verify: false): {', '.join(skipped)}. Listen to these.")
    if bad:
        typer.echo(f"{len(bad)} line(s) still mismatch or are silent after retakes: fix the text or `pronounce`, or list "
                   "what speech recognition mishears in `asr_aliases` (rescoring is free: `promo voice --all --force` "
                   "reuses the cached audio), then rerun.", err=True)
        raise typer.Exit(1)


@voice_app.command("audition")
def voice_audition(
    text: str,
    voices: Optional[str] = typer.Option(None, help="Comma-separated voice names (default: the pack's voices)."),
    style: Optional[str] = typer.Option(None, help="Style direction for Gemini (default: the pack's); Supertonic has none."),
) -> None:
    """Render one sentence in several voices for the user to choose."""
    from promo.config import MissingKeyError
    from promo.packs import PackError
    from promo.proc import ToolError
    from promo.providers.http import ProviderError
    from promo.voice import audition, audition_voices, voice_settings

    run, sl, pack = _load_run_context()
    try:
        settings = voice_settings(sl, pack)
    except PackError as e:
        raise _provider_failure(e)
    names = voices.split(",") if voices is not None else audition_voices(settings, pack)
    names = [n for n in (n.strip() for n in names) if n]
    if not names:
        eg = "F1,F2,M1" if settings.provider == "supertonic" else "Kore,Aoede"
        typer.echo(f"promo: --voices needs at least one voice name, e.g. --voices {eg}", err=True)
        raise typer.Exit(2)
    try:
        paths = audition(text, names, style or pack.voice.get("style", ""), run.auditions_dir, costs_path=run.costs,
                         settings=settings, language=sl.language)
    except (MissingKeyError, ProviderError, ToolError) as e:
        raise _provider_failure(e)
    for p in paths:
        typer.echo(str(p))
    typer.echo("Listen and pick one, then set shotlist.json → voice.name.")


@app.command("music")
def music_cmd(
    provider: Optional[str] = typer.Option(None, help="elevenlabs or lyria (Lyria 3.5 on GEMINI_API_KEY). Default: "
                                           "shotlist.json music.provider, else elevenlabs."),
    fresh: bool = typer.Option(False, "--fresh", help="Re-roll, even though this plan's bed is cached: only the "
                               "segments that failed their checks in the last attempt, or all of them when none did "
                               "(no attempt recorded, or the last one made a bed). Costs one call per segment "
                               "re-rolled; plain `promo music` then keeps the new bed."),
    segment: Optional[list[int]] = typer.Option(None, "--segment", help="With --fresh: re-roll exactly this segment, "
                                                "whether or not it passed, and keep every other segment's roll. "
                                                "Repeat it for several. Numbered as the table prints them (from 1). "
                                                "Needs --fresh."),
) -> None:
    """Generate the music bed from the shot list's section plan, fit it to the planned beat grid, and measure it.

    The bed is stretched to the planned tempo and shifted so a downbeat lands on bar 1 (conform.json says by how
    much); a bed more than 6% off tempo, off the grid by more than a frame, or (Lyria) with vocals is a re-roll.
    A long bed comes in segments, each with its own roll (audio/music/roll.json) and its own checks. Every run
    prints a table of the segments: roll, source (fresh, cached or reused from roll N), pass or fail with the
    reason, and cost. A segment that fails first tries the other cached rolls asked for the same key; `--fresh`
    then re-rolls only the segments that failed. `--fresh --segment N` re-rolls exactly the segments named, and
    after a run each seam that still has a warning says which segment to re-roll for it.
    The plan comes from shotlist.json and the VO takes, never from timing.json, so running it again after
    `promo timing` returns the cached bed and pays nothing.
    """
    from promo.config import MissingKeyError
    from promo.music import (DEFAULT_PROVIDER, PROVIDERS, MusicError, SegmentError, make_bed, seam_advice,
                             segment_table, tempo)
    from promo.proc import ToolError
    from promo.providers.http import ProviderError
    from promo.timing import Measured, TimingError, load_measured, resolve

    if provider is not None and provider not in PROVIDERS:
        typer.echo(f"promo: --provider must be {' or '.join(PROVIDERS)} (got '{provider}')", err=True)
        raise typer.Exit(2)
    if segment and not fresh:
        typer.echo("promo: --segment needs --fresh: it names the segments to re-roll", err=True)
        raise typer.Exit(2)
    if segment and min(segment) < 1:
        typer.echo(f"promo: --segment {min(segment)}: segments count from 1, as `promo music` prints them", err=True)
        raise typer.Exit(2)
    named = sorted(set(segment)) if segment else None
    run, sl, pack = _load_run_context()
    chosen = provider or sl.music.provider or DEFAULT_PROVIDER
    try:
        # The section plan on the nominal grid: no measured beat grid, so it is the same before and after
        # `promo timing` has snapped to this bed. A music-mode shot list without music.bpm plans at the pack's tempo.
        plan_sl = sl
        if sl.timing_mode == "music" and sl.music.bpm is None:
            plan_sl = sl.model_copy(update={"music": sl.music.model_copy(update={"bpm": tempo(sl, pack)})})
        plan = resolve(plan_sl, Measured(takes=load_measured(run.root).takes), pack.limits(), pack.text_mode())
        bed = make_bed(run, sl, pack, plan, fresh=fresh, provider=chosen, segments=named,
                       on_attempt=lambda rows: [typer.echo(line) for line in segment_table(rows)])
    except SegmentError as e:      # a usage error found once the bed is planned: nothing was paid for
        typer.echo(f"promo: {e}", err=True)
        raise typer.Exit(2)
    except (MissingKeyError, ProviderError, ToolError, MusicError, TimingError, OSError) as e:
        raise _provider_failure(e)
    if bed is None:
        typer.echo("pack music mode is 'none': no bed generated")
        return
    import json

    try:
        report = json.loads((bed.parent / "conform.json").read_text())
    except (OSError, ValueError):
        report = {}
    grid = report.get("grid")
    if grid:
        unmeasured = grid["bars_sounding"] - grid["bars_measured"]
        typer.echo(f"grid     worst bar {grid['bar_max_error_s'] * 1000:.1f} ms off at {grid['worst_bar_at_s']:.1f}s "
                   f"(bar {grid['worst_bar']}; limit {grid['limit_s'] * 1000:.1f} ms), worst beat "
                   f"{grid['raw_max_error_s'] * 1000:.1f} ms at {grid['raw_max_error_at_s']:.1f}s, {unmeasured} of "
                   f"{grid['bars_sounding']} sounding bars unmeasured")
    for w in report.get("warnings", []):
        typer.echo(f"warning  {w}")
    for line in seam_advice(report):
        typer.echo(f"advice   {line}")
    typer.echo(f"{bed} ({chosen})\nNow run `promo timing` again to snap to the measured beat grid.")


sfx_app = typer.Typer(help="Sound effects.", no_args_is_help=True)
app.add_typer(sfx_app, name="sfx")


@sfx_app.command("render")
def sfx_render() -> None:
    """Render audio/sfx/track.wav: one sound per event, peaks on event times."""
    from pydantic import ValidationError

    from promo.config import MissingKeyError
    from promo.proc import ToolError
    from promo.providers.http import ProviderError
    from promo.sfx import SfxError, render_stem
    from promo.timing import load_timing

    run, _, pack = _load_run_context()
    if not run.timing.exists():
        typer.echo("promo: run `promo timing` first", err=True)
        raise typer.Exit(2)  # a missing prerequisite, as in scenes and beats; a corrupt file is exit 1 below
    try:
        timing = load_timing(run.timing)
    except (ValidationError, UnicodeDecodeError, OSError) as e:
        typer.echo(f"promo: timing.json cannot be read ({type(e).__name__}); run `promo timing` again", err=True)
        raise typer.Exit(1)
    try:
        stem, placements, warnings = render_stem(timing, pack, run.sfx_dir / "track.wav", costs_path=run.costs)
    except (MissingKeyError, ProviderError, ToolError, SfxError, OSError) as e:
        raise _provider_failure(e)
    for w in warnings:
        typer.echo(f"warning  {w}")
    typer.echo(f"{stem} ({len(placements)} sounds)")


@sfx_app.command("synth")
def sfx_synth_cmd(kind: str, out: Path = typer.Option(..., help="Output WAV."),
                  pitch: Optional[float] = typer.Option(None, help="Pitch in Hz."),
                  duration: Optional[float] = typer.Option(None, help="Length in seconds."),
                  value: Optional[float] = typer.Option(None, help="Value 0-100; the pitch follows it on a log scale.")) -> None:
    """Write one procedural sound for auditioning."""
    from promo.sfx_synth import SfxError, synth, write_wav

    if out.suffix.lower() != ".wav":
        typer.echo(f"promo: --out must be a .wav file (got '{out}')", err=True)
        raise typer.Exit(2)
    try:
        samples = synth(kind, pitch=pitch, duration=duration, value=value)
    except SfxError as e:
        typer.echo(f"promo: {e}", err=True)
        raise typer.Exit(2)
    try:
        typer.echo(str(write_wav(samples, out)))
    except OSError as e:
        raise _provider_failure(e)


@sfx_app.command("gen")
def sfx_gen_cmd(prompt: str, duration: float = typer.Option(0.5, help="Length in seconds."),
                out: Path = typer.Option(..., help="Output MP3.")) -> None:
    """Generate one sound with ElevenLabs (cached) and write it as MP3.

    Inside a run folder the call is logged in costs.jsonl.
    """
    from promo.config import MissingKeyError
    from promo.proc import atomic_path
    from promo.providers.http import ProviderError
    from promo.run import RunNotFound, find_run
    from promo.sfx import SfxError, generate

    if out.suffix.lower() != ".mp3":
        typer.echo(f"promo: --out must be an .mp3 file, which is what ElevenLabs returns (got '{out}')", err=True)
        raise typer.Exit(2)
    try:
        costs_path = find_run().costs
    except RunNotFound:
        costs_path = None  # auditioning outside a run: nothing to log the cost to
    try:
        audio = generate(prompt, duration, costs_path=costs_path)
        with atomic_path(out) as tmp:
            tmp.write_bytes(audio)
    except SfxError as e:
        typer.echo(f"promo: {e}", err=True)
        raise typer.Exit(2)
    except (MissingKeyError, ProviderError, OSError) as e:
        raise _provider_failure(e)
    typer.echo(str(out))


@app.command()
def captions() -> None:
    """Word-level captions: SRT/VTT always; an alpha overlay when the pack wants subtitles."""
    from pydantic import ValidationError

    from promo.captions import CaptionsError, build
    from promo.hyperframes import HyperframesError
    from promo.proc import ToolError
    from promo.timing import load_timing

    run, _, pack = _load_run_context()
    if not run.timing.exists():
        typer.echo("promo: run `promo timing` first", err=True)
        raise typer.Exit(2)
    try:
        t = load_timing(run.timing)
    except (ValidationError, UnicodeDecodeError, OSError) as e:
        typer.echo(f"promo: timing.json cannot be read ({type(e).__name__}); run `promo timing` again", err=True)
        raise typer.Exit(1)
    if any(line.estimated for line in t.lines):
        typer.echo("promo: some lines have no voice take yet; run `promo voice --all` then `promo timing`", err=True)
        raise typer.Exit(2)
    try:
        out = build(run, t, pack)
    except (CaptionsError, HyperframesError, ToolError, OSError) as e:
        raise _provider_failure(e)
    for name, path in out.items():
        typer.echo(f"{name:<8} {path}")


@app.command()
def assemble(
    draft: bool = typer.Option(False, help="Animatic: half size, 30 fps, fast encode."),
    allow_stale: bool = typer.Option(False, "--allow-stale",
                                     help="Assemble the master even if some shot renders are out of date."),
) -> None:
    """Build timeline.otio, render it, and master the mix (true peak measured on the MP4).

    Shot renders made for an older timing, scene or design stop the master (exit 2) unless --allow-stale; for
    --draft they are a warning.
    """
    from pydantic import ValidationError

    from promo.assemble import AssembleError, MissingRenders, assemble as do_assemble
    from promo.music import MusicError
    from promo.proc import ToolError
    from promo.sfx import SfxError
    from promo.timeline import TimelineError
    from promo.timing import load_timing

    run, _, pack = _load_run_context()
    if not run.timing.exists():
        typer.echo("promo: run `promo timing` first", err=True)
        raise typer.Exit(2)
    try:
        t = load_timing(run.timing)
    except (ValidationError, UnicodeDecodeError, OSError) as e:
        typer.echo(f"promo: timing.json cannot be read ({type(e).__name__}); run `promo timing` again", err=True)
        raise typer.Exit(1)
    if t.errors():
        typer.echo("promo: timing.json has errors; fix them (`promo timing`) before assembling", err=True)
        raise typer.Exit(1)
    try:
        out = do_assemble(run, t, pack, draft=draft, allow_stale=allow_stale,
                          warn=lambda message: typer.echo(f"warning  {message}"))
    except MissingRenders as e:
        typer.echo(f"promo: {e}", err=True)
        raise typer.Exit(2)
    except (TimelineError, SfxError, MusicError, AssembleError, ToolError, OSError) as e:
        raise _provider_failure(e)
    report = out.with_name(out.stem + ".loudness.json").read_text()
    typer.echo(f"{out}\n{report}\nAudio checked by measurement, not by ear.")


def _timing_or_exit(run: "RunPaths") -> "Timing":
    """timing.json, loaded: a missing one is exit 2 (run `promo timing`), an unreadable one exit 1."""
    from pydantic import ValidationError

    from promo.timing import load_timing

    if not run.timing.exists():
        typer.echo("promo: run `promo timing` first", err=True)
        raise typer.Exit(2)
    try:
        return load_timing(run.timing)
    except (ValidationError, UnicodeDecodeError, OSError) as e:
        typer.echo(f"promo: timing.json cannot be read ({type(e).__name__}); run `promo timing` again", err=True)
        raise typer.Exit(1)


@app.command()
def verify(
    shot: Optional[list[str]] = typer.Option(None, "--shot", help="Only these shot ids (repeatable)."),
    frames: int = typer.Option(6, min=1, help="Frames per shot."),
    allow_overlaps: bool = typer.Option(False, "--allow-overlaps",
                                        help="Report overlapping tweens on one property as warnings, not failures."),
) -> None:
    """Scene-state determinism: the same inline styles whatever the seek order, and no overlapping tweens."""
    from promo.hyperframes import HyperframesError
    from promo.proc import ToolError
    from promo.verify import MissingScenes, VerifyError, missing_scenes, verify as do_verify

    run, _, pack = _load_run_context()
    t = _timing_or_exit(run)
    if t.errors():
        typer.echo("promo: timing.json has errors; fix them (`promo timing`) before verifying", err=True)
        raise typer.Exit(1)
    missing = missing_scenes(run, t, shot or None)
    if missing:
        typer.echo(f"promo: no scene file for {', '.join(missing)}; run `promo scenes --scaffold`", err=True)
        raise typer.Exit(2)
    try:
        res = do_verify(run, t, pack, only=shot or None, n=frames, allow_overlaps=allow_overlaps)
    except MissingScenes as e:
        typer.echo(f"promo: {e}", err=True)
        raise typer.Exit(2)
    except (VerifyError, HyperframesError, ToolError, OSError) as e:
        raise _provider_failure(e)
    typer.echo(f"{'ok' if res.ok else 'FAIL'}: {len(res.shots)} shots checked, {len(res.mismatches)} failures, "
               f"{len(res.warnings)} warnings")
    for m in res.mismatches:
        extra = f" [pixels: largest difference {m['max_diff']} of 255 on {m['pixels']}]" if m.get("pixels") else ""
        typer.echo(f"  FAIL  {m['message']}{extra}")
    for w in res.warnings:
        typer.echo(f"  warn  {w['message']}")
    if not res.ok:
        typer.echo(f"The captures of the failing shots are in {run.renders / 'verify'}")
        raise typer.Exit(1)


@app.command()
def critique(
    video: Optional[Path] = typer.Option(None, help="Video to judge (default renders/master.mp4)."),
    no_hf: bool = typer.Option(False, "--no-hf", help="Skip HyperFrames lint/check."),
) -> None:
    """Build a critic round: evidence, automatic checks, and PROMPT.md for a fresh critic subagent."""
    import json

    from promo.config import MissingKeyError
    from promo.critique import CritiqueError, MissingMaster, critique as do_critique
    from promo.hyperframes import HyperframesError
    from promo.packs import PackError
    from promo.proc import ToolError
    from promo.providers.http import ProviderError
    from promo.verify import VerifyError

    run, _, pack = _load_run_context()
    t = _timing_or_exit(run)
    try:
        rd = do_critique(run, t, pack, video=video, hf=not no_hf)
    except MissingMaster as e:
        typer.echo(f"promo: {e}", err=True)
        raise typer.Exit(2)
    except (CritiqueError, VerifyError, HyperframesError, MissingKeyError, ProviderError, ToolError, PackError,
            OSError) as e:
        raise _provider_failure(e)
    summary = json.loads((rd / "summary.json").read_text())
    for c in summary["checks"]:
        typer.echo(f"{'PASS' if c['ok'] else 'FAIL'}  {c['name']:<12} {c['detail']}")
    for f in summary["findings"]:
        typer.echo(f"warning  {f['code']:<20} {f['message']}")
    typer.echo(f"critic prompt: {rd / 'PROMPT.md'}")
    if not summary["ok"]:
        typer.echo("AUTOMATIC CHECKS FAILED: fix these before asking the critic subagent.", err=True)
        raise typer.Exit(1)


@app.command()
def ship(
    reveal: bool = typer.Option(False, "--reveal", help="Reveal the master in Finder."),
    force: bool = typer.Option(False, "--force", help="Ship without a passing critic round (recorded in the README)."),
) -> None:
    """Poster (frame 0), the critic gate and a Renders line in the run README; optionally reveal in Finder.

    The latest critic round must have passed its automatic checks and the critic's scores, on this master.
    """
    from promo.proc import ToolError
    from promo.ship import NoMaster, ShipError, ship as do_ship

    run = _run_or_exit()
    try:
        out = do_ship(run, reveal=reveal, force=force)
    except NoMaster as e:
        typer.echo(f"promo: {e}", err=True)
        raise typer.Exit(2)
    except ShipError as e:
        typer.echo(f"promo: {e}", err=True)
        raise typer.Exit(1)
    except (ToolError, OSError) as e:
        raise _provider_failure(e)
    typer.echo(f"master {out['master']}\nposter {out['poster']}\nreadme {out['readme']}")
    if not out["passed"]:
        typer.echo(f"warning  shipped without a passing critic round ({out['why']}); the README says so")


@app.command()
def guide() -> None:
    """Write ANIMATION_GUIDE.md for scene subagents."""
    from promo.guide import write_guide
    from promo.packs import PackError

    run, _, pack = _load_run_context()
    try:
        typer.echo(str(write_guide(run, pack)))
    except (PackError, OSError) as e:
        raise _provider_failure(e)


@app.command()
def ref(
    video: Path,
    out: Optional[Path] = typer.Option(None, help="Output folder (default: <video stem>-audit next to it)."),
    crop: Optional[str] = typer.Option(None, help="Crop W:H:X:Y, e.g. a laptop screen inside a phone recording."),
    lang: Optional[str] = typer.Option(None, help="Narration language code (default en)."),
) -> None:
    """Reference ingest: frames, sheets, audio and voice metrics, style_guide.md skeleton."""
    from promo.proc import ToolError
    from promo.ref import RefError, ingest

    dest = out or video.with_name(video.stem + "-audit")
    try:
        typer.echo(str(ingest(video, dest, crop=crop, language=lang)))
    except RefError as e:
        typer.echo(f"promo: {e}", err=True)
        raise typer.Exit(2)
    except (ToolError, ValueError, OSError) as e:   # ValueError: a language Whisper cannot transcribe
        raise _provider_failure(e)


@app.command()
def doctor(live: bool = typer.Option(False, help="Also ping each API key (a free key-validation call, never a paid one).")) -> None:
    """Check keys, tools, models and disk."""
    from promo.doctor import run_checks

    checks = run_checks(live=live)
    for c in checks:
        mark = "ok  " if c.ok else ("WARN" if c.level == "warning" else ("info" if c.level == "info" else "FAIL"))
        typer.echo(f"{mark}  {c.name:<34} {c.detail}")
    errors = sum(1 for c in checks if not c.ok and c.level == "error")
    typer.echo(f"{errors} errors, {sum(1 for c in checks if not c.ok and c.level == 'warning')} warnings")
    if errors:
        raise typer.Exit(1)
