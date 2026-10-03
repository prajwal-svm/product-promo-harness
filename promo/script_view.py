"""A readable, derived view of the shot list: script.md."""

from __future__ import annotations

from promo.shotlist import Shotlist
from promo.timing import Timing


def _t(value: float | None) -> str:
    return "–" if value is None else f"{value:.2f}"


def render_script(sl: Shotlist, timing: Timing | None) -> str:
    starts = {l.id: l.start_s for l in timing.lines} if timing else {}
    shot_spans = {s.id: (s.start_s, s.end_s) for s in timing.shots} if timing else {}
    out = ["# Script", "", f"Pack `{sl.pack}` · {sl.canvas.aspect} · {sl.timing_mode} mode · language `{sl.language}`", ""]
    out += ["## Voiceover", "", "| line | start (s) | text | style |", "| --- | --- | --- | --- |"]
    for line in sl.all_lines():
        out.append(f"| {line.id} | {_t(starts.get(line.id))} | {line.text} | {line.style} |")
    out += ["", "## On screen", ""]
    for ch in sl.chapters:
        head = ch.headline.text if ch.headline else "(no headline)"
        key = f" (key: *{ch.headline.key}*)" if ch.headline and ch.headline.key else ""
        out.append(f"- **{ch.id}** {head}{key}. {ch.purpose}".rstrip())
        for shot in ch.shots:
            span = shot_spans.get(shot.id)
            when = f" [{span[0]:.2f}–{span[1]:.2f}s]" if span else ""
            out.append(f"  - {shot.id}{when}: {shot.visual.composition or '-'}; camera: {shot.visual.camera or '-'}")
    out += ["", "## Transition map", "", "| from | kind | carrier | meaning |", "| --- | --- | --- | --- |"]
    for shot in sl.all_shots():
        tr = shot.transition_out
        if tr:
            out.append(f"| {shot.id} | {tr.kind} | {tr.carrier} | {tr.meaning} |")
    out += ["", "## Sound plan", ""]
    for shot in sl.all_shots():
        cue = shot.sound.music_cue or ""
        sfx = ", ".join(f"{e.kind}→{e.sfx}" for e in shot.events if e.sfx)
        if cue or sfx:
            out.append(f"- {shot.id}: {cue}{'; ' if cue and sfx else ''}{sfx}")
    return "\n".join(out) + "\n"
