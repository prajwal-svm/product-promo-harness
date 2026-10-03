"""The music bed: a timed prompt from the shot list, generation (ElevenLabs or Lyria), then a bed conformed to the
film's planned beat grid.

A long bed comes in segments, one provider call each, and each segment has its own roll number (audio/music/roll.json),
its own checks and its own re-roll: `promo music --fresh` asks again only for the segments that failed their checks,
and `promo music --fresh --segment N` asks again for exactly the segments named, whether or not they passed.
A music prompt carries musical words only: a section's note is its shots' `sound.music_cue` values with the
NOTE_DENYLIST (AI, agents, model and brand names) taken out, never the chapter's purpose.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Collection

from pydantic import BaseModel

from promo import conform as cf
from promo.audio_analysis import loudness
from promo.beats import SteadyFit, onsets as onset_times, write_grid
from promo.models import BeatGrid
from promo.packs import Pack
from promo.proc import ToolError, atomic_path, run as run_cmd, sha256_hex
from promo.providers import cache, costs, elevenlabs, lyria
from promo.run import RunPaths
from promo.shotlist import Shotlist
from promo.timing import Timing


class MusicError(RuntimeError):
    """The generated audio could not be used (no beats in it, off the planned grid, vocals in it ...)."""


class SegmentError(MusicError):
    """`--segment` was used wrongly: without `--fresh`, or naming a segment the bed does not have. A usage error
    (exit 2 in the CLI), found before anything is paid for."""


class Section(BaseModel):
    """One chapter's stretch of the bed. Times are seconds from the music start, not the film start."""

    id: str
    start_s: float
    dur_s: float
    note: str


def _mmss(t: float) -> str:
    m, s = divmod(round(max(0.0, t), 1), 60)  # round first, so 59.96 reads 1:00.0 and not 0:60.0
    return f"{int(m)}:{s:04.1f}"


# Words a music prompt must not carry. Lyria answered a prompt with HTTP 400 "Request blocked for an unspecified
# policy reason" when its section lines quoted chapter purposes with product words ("Ask AI", "your coding agent; 14
# agents", "Reviewer #2"). So a section's note is only its shots' `sound.music_cue` values, and these are taken out of
# those too: AI, agents, model names and brand names. A term matches as a whole word (so "trail" is safe), and a
# version or a hyphenated compound goes with it ("GPT-4.5", "AI-powered").
NOTE_DENYLIST = (
    "AI", "AIs", "A.I.", "LLM", "LLMs", "agent", "agents", "agentic", "chatbot", "chatbots", "copilot",
    "ChatGPT", "GPT", "OpenAI", "Claude", "Anthropic", "Gemini", "Llama", "Mistral", "DeepSeek", "Codex", "Cursor",
    "Google", "Microsoft", "Apple", "GitHub", "Notion", "Slack",
)
_DENY = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(w) for w in sorted(NOTE_DENYLIST, key=len, reverse=True))
                   + r")(?:[-.]?\d[\w.]*)?(?:-\w+)*(?!\w)", re.IGNORECASE)
NO_CUE = "continue the groove"


def clean_note(text: str) -> str:
    """`text` without the terms of NOTE_DENYLIST (the denylist for music prompts), its `;`-separated parts tidied
    of the punctuation and spaces the removal leaves. Empty when nothing musical is left."""
    parts = []
    for frag in _DENY.sub(" ", text).split(";"):
        frag = re.sub(r"\s+", " ", frag)
        frag = re.sub(r"\s+([,.:])", r"\1", frag).strip(" ,.:;-/")
        if frag:
            parts.append(frag)
    return "; ".join(parts)


def sections_from_timing(timing: Timing, sl: Shotlist, *, purposes: bool = False,
                         cues: bool = True) -> list[Section]:
    """One section per chapter. Its note is its shots' music cues, cleaned (clean_note) and joined, else "continue
    the groove": the chapter's purpose is about the picture and never reaches the prompt.

    `purposes=True` rebuilds the notes as they were before ruling R101 (the purpose first, then the cues, nothing
    taken out; `cues=False` leaves the cues off, as a shot list without any had them). It is never sent: make_bed
    uses it only to find, in the cache, deliveries that were paid for with those prompts."""
    out = []
    for ch_t, ch in zip(timing.chapters, sl.chapters):
        shot_cues = [s.sound.music_cue for s in ch.shots if s.sound.music_cue] if cues else []
        if purposes:
            note = "; ".join(x for x in [ch.purpose, *shot_cues] if x) or NO_CUE
        else:
            note = "; ".join(c for c in (clean_note(x) for x in shot_cues) if c) or NO_CUE
        out.append(Section(id=ch.id, start_s=round(ch_t.start_s - timing.preroll_s, 3),
                           dur_s=round(ch_t.end_s - ch_t.start_s, 3), note=note))
    return out


def build_prompt(music: dict, sections: list[Section], bpm: float, total_s: float, beats_per_bar: int = 4) -> str:
    mood = music.get("mood", "")
    lines = [f"Instrumental {mood} track at {bpm:g} BPM in {beats_per_bar}/4." if mood
             else f"Instrumental track at {bpm:g} BPM in {beats_per_bar}/4."]
    if music.get("instrumentation"):
        lines.append(f"Instrumentation: {music['instrumentation']}.")
    if music.get("arrangement"):
        lines.append(f"Arrangement: {music['arrangement']}.")
    lines.append(f"Avoid: {music.get('negative', 'vocals, lyrics')}.")
    lines.append(f"Total length {total_s:.1f} seconds. Change sections exactly at these times:")
    for s in sections:
        lines.append(f"- {_mmss(s.start_s)}–{_mmss(s.start_s + s.dur_s)} ({s.id}): {s.note}")
    end = sections[-1].start_s + sections[-1].dur_s if sections else 0.0
    lines.append(f"- {_mmss(end)}–{_mmss(total_s)}: clean ending; no new elements")
    return "\n".join(lines)


# --- segments -------------------------------------------------------------------------------------------------

LEAD_BARS = 2      # a later segment starts this many bars before its seam, so the crossfade has settled music
TAIL_BARS = 1      # an earlier segment runs this many bars past its seam
SNAP_BARS = 4      # a seam moves up to this far to sit on a section change


@dataclass(frozen=True)
class Segment:
    """One provider call's share of the bed, in bed seconds. Its audio starts at `audio_start_s` (a downbeat) and
    runs to `audio_end_s`; it is heard from `seam_s` (where it takes over) to `next_seam_s`. `asked_s` covers the
    span plus a stretch of up to MAX_STRETCH and a bar the conform may trim before the first downbeat.
    `on_section` says its seam sits on a planned section change."""

    index: int
    count: int
    audio_start_s: float
    seam_s: float
    next_seam_s: float
    audio_end_s: float
    asked_s: float
    on_section: bool = False


def _asked(span_s: float, bar_s: float) -> float:
    return round(span_s * (1 + cf.MAX_STRETCH) + bar_s, 3)


def plan_segments(total_s: float, bar_s: float, max_call_s: float, section_starts_s: list[float] = (),
                  origin_s: float = 0.0) -> list[Segment]:
    """Split a bed of `total_s` into the fewest bar-aligned segments one call each can make. Seams sit on the
    plan's bar lines (counted from `origin_s`, the plan's bar 1), on a section change when one is within SNAP_BARS
    of the even split."""
    if _asked(total_s, bar_s) <= max_call_s:
        return [Segment(0, 1, 0.0, 0.0, round(total_s, 3), round(total_s, 3), _asked(total_s, bar_s))]
    n_bars = math.ceil((total_s - origin_s) / bar_s - 1e-9)
    section_bars = sorted({round((s - origin_s) / bar_s) for s in section_starts_s if origin_s < s < total_s})
    for n in range(2, n_bars // (LEAD_BARS + TAIL_BARS + 1) + 1):
        for snap in (True, False):
            seams = []
            for i in range(1, n):
                bar = round(i * n_bars / n)
                near = [b for b in section_bars if abs(b - bar) <= SNAP_BARS] if snap else []
                seams.append(min(near, key=lambda b: abs(b - bar)) if near else bar)
            if seams != sorted(set(seams)) or seams[0] <= LEAD_BARS or seams[-1] >= n_bars - TAIL_BARS:
                continue
            segs = []
            edges = [None, *seams, None]
            for i in range(n):
                seam = origin_s + edges[i] * bar_s if edges[i] is not None else 0.0
                nxt = origin_s + edges[i + 1] * bar_s if edges[i + 1] is not None else total_s
                start = seam - (LEAD_BARS * bar_s if i else 0.0)
                end = nxt + TAIL_BARS * bar_s if i < n - 1 else total_s
                segs.append(Segment(i, n, round(start, 3), round(seam, 3), round(nxt, 3), round(end, 3),
                                    _asked(end - start, bar_s), on_section=bool(i and edges[i] in section_bars)))
            if all(s.asked_s <= max_call_s for s in segs):
                return segs
    raise MusicError(f"a {total_s:.1f}s bed cannot be split into calls of at most {max_call_s:g}s")


def build_segment_prompt(music: dict, sections: list[Section], bpm: float, beats_per_bar: int, seg: Segment,
                         total_s: float, key: str | None = None) -> str:
    """A Lyria prompt for one segment: the same style lines in every segment, then where this part sits in the
    track (and, for a later part, the `key` the first part came back in), then its sections in the segment's own
    time ([m:ss - m:ss], as Lyria's docs write timestamps)."""
    lines = ["Instrumental only, no vocals: no singing, no lyrics, no spoken words."]
    if music.get("mood"):
        lines.append(f"Style: {music['mood']}.")
    lines.append(f"Tempo: exactly {bpm:g} BPM in {beats_per_bar}/4, one steady tempo from start to end.")
    if music.get("key"):
        lines.append(f"Key: {music['key']}.")
    if music.get("instrumentation"):
        lines.append(f"Instrumentation: {music['instrumentation']}.")
    if music.get("arrangement"):
        lines.append(f"Arrangement: {music['arrangement']}.")
    lines.append(f"Avoid: {music.get('negative', 'vocals, lyrics')}.")
    last = seg.index == seg.count - 1
    local_end = seg.audio_end_s - seg.audio_start_s
    if seg.count == 1:
        lines.append("Start on the first downbeat at 0:00.")
    elif seg.index == 0:
        lines.append(f"This is part 1 of {seg.count} of one continuous {_mmss(total_s)} track. Start on the first "
                     f"downbeat at 0:00. Do not end or fade out: at {_mmss(local_end)} the music carries on in full.")
    else:
        lines.append(f"This is part {seg.index + 1} of {seg.count} of one continuous track"
                     f"{f' in {key}' if key else ''}, picking up at "
                     f"{_mmss(seg.audio_start_s)} of it. Start at 0:00 on a downbeat, already in full swing: no "
                     f"intro, no fade-in, no count-in; the same key, sounds and instruments as the rest of the track."
                     + ("" if last else f" Do not end or fade out: at {_mmss(local_end)} the music carries on."))
    lines.append(f"Length {_mmss(seg.asked_s)} ({seg.asked_s:.0f} seconds). Sections, in this part's own time:")
    for s in sections:
        a, b = max(s.start_s, seg.audio_start_s), min(s.start_s + s.dur_s, seg.audio_end_s)
        if b - a <= 1e-6:
            continue
        cont = " (continuing)" if s.start_s < seg.audio_start_s - 1e-6 else ""
        lines.append(f"[{_mmss(a - seg.audio_start_s)} - {_mmss(b - seg.audio_start_s)}] {s.id}{cont}: {s.note}")
    end = min(seg.audio_end_s, sections[-1].start_s + sections[-1].dur_s if sections else 0.0) - seg.audio_start_s
    tail = "Clean ending; no new elements." if last else "Keep going; no ending."
    lines.append(f"[{_mmss(end)} - {_mmss(seg.asked_s)}] {tail}")
    return "\n".join(lines)


# --- providers ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MusicProvider:
    """What make_bed needs to know about a music API. `max_call_s`: the longest bed one call is asked for.
    `call_usd(asked_s)`: one call's price. `clamp_ms`: the length a call is asked (and billed) for.
    `check_vocals`: the API has no instrumental switch, so the bed is listened to for words.
    `segment_prompts`: prompts in the Lyria shape (build_segment_prompt), even for one call."""

    name: str
    model: str
    compose: Callable[[str, float], bytes]
    max_call_s: float
    call_usd: Callable[[float], float]
    clamp_ms: Callable[[float], int]
    check_vocals: bool
    segment_prompts: bool


PROVIDERS: dict[str, MusicProvider] = {
    "elevenlabs": MusicProvider(
        "elevenlabs", elevenlabs.MUSIC_MODEL, elevenlabs.compose, elevenlabs.MUSIC_MAX_MS / 1000,
        lambda s: s / 60 * elevenlabs.MUSIC_USD_PER_MIN, elevenlabs.clamp_length_ms, False, False),
    "lyria": MusicProvider(
        "lyria", lyria.MUSIC_MODEL, lyria.compose, lyria.MAX_CALL_S, lambda s: lyria.call_usd(lyria.MUSIC_MODEL),
        lambda ms: int(round(ms)), True, True),
}
DEFAULT_PROVIDER = "elevenlabs"
MAX_VOICE_WORDS = 3     # ASR hears a word or two in some instrumental beds; more than this is singing
# A seam is flagged (in conform.json and by `promo music`) when its crossfade dips more than this, or a band's level
# jumps more than this across it. Probe 2's real seam: a 1.3 dB dip, and 4.1 dB less air (6-16 kHz) after it,
# where the bed's own section changes move a band by up to 7 dB.
SEAM_DIP_DB = -1.5
SEAM_STEP_DB = -6.0
SEAM_JUMP_DB = 6.0
UNMEASURED_SHARE = 0.25   # more than this share of the sounding bars unmeasured is a warning (ruling R97)


BED_PEAK_DB = -1.0


def normalise_bed(src: Path, dst: Path, lufs: float) -> Path:
    """Write `src` to `dst` (48 kHz stereo s24) at `lufs` LUFS integrated, so the provider's delivery level never
    decides how the music sits under the voice.

    One linear gain keeps the arrangement's dynamics (a single-pass loudnorm, as for VO, would level a planned
    drop). A limiter at -1 dBFS, at 4x the sample rate, only catches peaks that a large boost pushes past full scale.
    """
    measured = loudness(src)["I"]
    gain = lufs - measured if math.isfinite(measured) and measured > -70 else 0.0   # -70 LUFS: nothing to measure
    af = (f"volume={gain:.2f}dB,aresample=192000,alimiter=limit={10 ** (BED_PEAK_DB / 20):.4f}:level=false:"
          f"latency=true,aresample=48000")
    with atomic_path(dst) as tmp:
        run_cmd(["ffmpeg", "-v", "error", "-y", "-i", src, "-af", af, "-ar", "48000", "-c:a", "pcm_s24le", tmp])
    return dst


def tempo(sl: Shotlist, pack: Pack) -> float:
    """The tempo the bed is asked for: the shot list's, else the pack's."""
    return sl.music.bpm or pack.music.get("bpm", 96)


def segment_cache_key(prov: MusicProvider, prompt: str, length_ms: int, roll: int, index: int, count: int) -> str:
    """The cache key of one delivery: provider, model, the prompt, the length asked, the segment's roll and which
    segment of how many it is. A different prompt (another key line, another style) is another delivery."""
    return sha256_hex(f"{prov.name}-music", prov.model, prompt, str(length_ms), f"roll={roll}",
                      f"segment={index + 1}/{count}")


def _roll_file(run: RunPaths) -> Path:
    return run.music_dir / "roll.json"


def load_rolls(run: RunPaths, count: int) -> dict[int, int]:
    """Each segment's roll, from audio/music/roll.json ({segment index: roll}: the newest roll asked for that
    segment; a segment not in it is on roll 0). A legacy roll.txt, one roll for the whole bed, becomes "every
    segment at that roll", written to roll.json and removed."""
    try:
        raw = json.loads(_roll_file(run).read_text())
        return {int(k): max(0, int(v)) for k, v in raw.items()}
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    legacy = run.music_dir / "roll.txt"
    try:
        roll = max(0, int(legacy.read_text().strip()))
    except (OSError, ValueError):
        return {}
    rolls = {i: roll for i in range(count)}
    save_rolls(run, rolls)
    legacy.unlink(missing_ok=True)
    return rolls


def save_rolls(run: RunPaths, rolls: dict[int, int]) -> None:
    with atomic_path(_roll_file(run)) as tmp:
        tmp.write_text(json.dumps({str(k): v for k, v in sorted(rolls.items())}))


def _words(path: Path) -> list[str]:
    """What speech recognition hears in the bed (Parakeet, which does not loop on music as Whisper can)."""
    from promo.asr import transcribe

    return [w.text for w in transcribe(path, "en")]


def _src_name(i: int) -> str:
    return "bed.src.mp3" if i == 0 else f"bed.src.{i + 1}.mp3"


def _conformed_grid(fit: SteadyFit, origin_s: float, beats_per_bar: int, onsets: list[float],
                    length_s: float) -> BeatGrid:
    """beats.json for a conformed bed: its measured straight grid from the plan's bar 1 (`origin_s`) to the bed's
    end. The conform put a downbeat there, so the bar phase is the plan's; a bar 1 on the first sample, which the
    measurement cannot hear the start of, still counts."""
    period, first = 60.0 / fit.bpm, fit.grid[0]
    k0 = round((origin_s - first) / period)
    k_end = min(len(fit.grid) - 1, math.floor((length_s - first) / period + 1e-9))
    beats = [round(first + k * period, 4) for k in range(k0, k_end + 1)]
    return BeatGrid(bpm=fit.bpm, beats=beats, downbeats=beats[::beats_per_bar], onsets=onsets,
                    first_downbeat_s=beats[0], beats_per_bar=beats_per_bar)


def seam_warnings(seams: list[dict]) -> list[str]:
    """What a person should listen to at the seams. A big band jump or level drop where the plan changes section
    anyway is labelled so, not hidden: it may be the change, or a restart that happens to sit there."""
    out = []
    for r in seams:
        at = f"seam at {r['seam_s']:.1f}s"
        planned = " (at a planned section change)" if r.get("planned_change") else ""
        if r["level_dip_db"] < SEAM_DIP_DB:
            out.append(f"{at}: the crossfade dips {-r['level_dip_db']:.1f} dB; listen to it")
        if r.get("level_step_db", 0.0) < SEAM_STEP_DB:
            out.append(f"{at}: the level drops {-r['level_step_db']:.1f} dB across it{planned}; listen to it, and "
                       f"re-roll if it sounds like a restart")
        if r["spectral_jump_db"] > SEAM_JUMP_DB:
            out.append(f"{at}: a band's level jumps {r['spectral_jump_db']:.1f} dB across it{planned or ' (a restart?)'}"
                       f"; listen to it, and re-roll if it sounds like one")
    return out


def seam_advice(report: dict) -> list[str]:
    """For each seam a person was told to listen to (a downbeat that is a guess, a level dip or step, a band jump),
    the segment to re-roll: the later segment of the seam, since the earlier one is what it joins on to."""
    return [f"seam {r['seam_s']:.1f} s: re-roll with --fresh --segment {r['segment']}"
            for r in report.get("seam_rerolls", [])]


def _bars(ranges: list[tuple[int, int]]) -> str:
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in ranges)


def grid_warnings(check: cf.GridCheck, fps: int) -> list[str]:
    """The grid check's numbers a person should hear for themselves: a single beat over a frame off (its bar is
    within a frame, or the gate would have failed), and more than a quarter of the sounding bars unmeasured."""
    out = []
    frame = 1.0 / fps
    if check.raw_max_error_s > frame:
        out.append(f"one beat at {check.raw_at_s:.1f}s is {check.raw_max_error_s * 1000:.0f} ms off the grid, over a "
                   f"frame (its bar's median is within one); listen there")
    unmeasured = check.bars_sounding - check.bars_measured
    if unmeasured > UNMEASURED_SHARE * check.bars_sounding:
        out.append(f"{unmeasured} of {check.bars_sounding} sounding bars have too few clear beats to check against "
                   f"the grid (bars {_bars(check.unmeasured)}); listen to them")
    return out


def phase_grid(grid: BeatGrid, origin_s: float, beats_per_bar: int) -> BeatGrid:
    """`grid` with bar 1 at the beat nearest `origin_s` (the plan's bar 1, which the conform put on a downbeat),
    instead of the beat its kick energy picks. Unchanged when no beat is within half a beat of it."""
    if not grid.beats:
        return grid
    period = 60.0 / grid.bpm
    i0 = min(range(len(grid.beats)), key=lambda i: abs(grid.beats[i] - origin_s))
    if abs(grid.beats[i0] - origin_s) > period / 2:
        return grid
    beats = grid.beats[i0:]
    return grid.model_copy(update={"beats": beats, "downbeats": beats[::beats_per_bar], "first_downbeat_s": beats[0]})


def _align_downbeat(path: Path, *, bpm: float, beats_per_bar: int, end_s: float) -> tuple[int, int, float]:
    """A later segment's bar lines must be the plan's. Its audio starts on a planned bar line; find which beat of
    that bar carries its kick-energy downbeat, and when that is not beat 1 (and the kick says so clearly), move the
    audio by the fewest beats that put it there. Returns (the beat it was found on, beats moved, the margin)."""
    beat, margin = cf.downbeat_phase(path, bpm=bpm, beats_per_bar=beats_per_bar, start_s=0.0, end_s=end_s)
    if beat == 0 or margin < cf.DOWNBEAT_MARGIN:
        return beat, 0, margin
    moved = -beat if beat <= beats_per_bar // 2 else beats_per_bar - beat     # trim `beat` beats, or pad the rest
    cf.shift_head(path, moved * 60.0 / bpm)
    return beat, moved, margin




# --- one attempt: what happened to each segment ------------------------------------------------------------------

_REROLL_TAIL = re.compile(r"[;,]?\s*(?:so )?re-roll (?:it|them) with `promo music --fresh`[^;]*$")


def _plain(reason: str) -> str:
    """A failure reason without its closing "re-roll it with `promo music --fresh`" (the table says that once)."""
    return _REROLL_TAIL.sub("", reason).strip()


@dataclass
class SegmentRun:
    """One segment in one run of make_bed: the roll its audio is on, where it came from (fresh: paid for now;
    cached: found at its own roll; reused from roll N: found at another), and how it did. `status`: passed, failed
    (its own checks, or the whole bed's checks at its place in the bed), error (the provider or a tool failed before
    there was audio to judge) or pending (not reached). `tried` lists the rolls that failed before it passed."""

    index: int
    count: int
    roll: int | None = None
    source: str = ""
    status: str = "pending"
    reason: str = ""
    usd: float = 0.0
    tried: list[dict] = field(default_factory=list)

    def row(self) -> dict:
        return {"segment": f"{self.index + 1}/{self.count}", "index": self.index, "roll": self.roll,
                "source": self.source, "status": self.status, "reason": self.reason, "usd": round(self.usd, 6),
                "tried": list(self.tried)}


def segment_table(rows: list[dict]) -> list[str]:
    """The lines `promo music` prints on every run: for each segment its roll, its source (fresh, cached, reused
    from roll N), pass or fail with the reason, and the cost of this run."""
    head = ("segment", "roll", "source", "result", "cost", "detail")
    cells = []
    for r in rows:
        result = {"passed": "pass", "failed": "FAIL", "error": "error"}.get(r["status"], "-")
        if r["status"] in ("failed", "error"):
            detail = _plain(r.get("reason", ""))
            others = [str(t["roll"]) for t in r.get("tried", []) if t["roll"] != r["roll"]]
            if others:
                detail += f" (rolls {', '.join(others)} from the cache failed too)"
        else:
            detail = "; ".join(f"roll {t['roll']} failed: {t['reason']}" for t in r.get("tried", []))
        cells.append((r["segment"], "-" if r["roll"] is None or r["status"] == "pending" else str(r["roll"]),
                      r["source"] or ("not reached" if r["status"] == "pending" else "-"), result,
                      f"${r['usd']:.2f}", detail))
    widths = [max(len(head[i]), *(len(c[i]) for c in cells)) for i in range(5)]

    def line(c) -> str:
        return ("  ".join(x.ljust(w) for x, w in zip(c[:5], widths)) + (f"  {c[5]}" if c[5] else "")).rstrip()

    total = sum(r["usd"] for r in rows)
    return [line(head), *(line(c) for c in cells), f"cost     ${total:.2f} this run"]


def _load_report(run: RunPaths) -> dict:
    try:
        report = json.loads((run.music_dir / "conform.json").read_text())
    except (OSError, ValueError):
        return {}
    return report if isinstance(report, dict) else {}


def _last_attempt(run: RunPaths, count: int) -> list[dict]:
    """The segment rows of the last recorded attempt (conform.json's `attempt`), or [] when there is none or it was
    for a bed split into a different number of segments."""
    try:
        rows = _load_report(run)["attempt"]["segments"]
    except (KeyError, TypeError):
        return []
    if not isinstance(rows, list) or len(rows) != count or not all(isinstance(r, dict) for r in rows):
        return []
    return rows


@dataclass(frozen=True)
class _Eval:
    ok: bool
    reason: str = ""
    placement: cf.Placement | None = None
    phase: dict | None = None
    words: int | None = None


@dataclass(frozen=True)
class _Won:
    """A segment's delivery that passed its own checks: how it was fitted, its downbeat, the words ASR heard in it
    (None when the provider needs no listening for vocals) and the prompt it was asked with."""

    placement: cf.Placement
    phase: dict
    words: int | None
    prompt: str


class _Bed:
    """One run of make_bed: the plan, then each segment (its roll, its delivery, its own checks), then the joined
    bed's checks, then the files."""

    def __init__(self, run: RunPaths, sl: Shotlist, pack: Pack, timing: Timing, prov: MusicProvider,
                 compose: Callable[[str, float], bytes], transcribe: Callable[[Path], list[str]] | None,
                 fresh: bool, on_attempt: Callable[[list[dict]], None] | None,
                 segments: Collection[int] | None = None) -> None:
        self.run, self.prov, self.compose, self.transcribe = run, prov, compose, transcribe
        self.fresh, self.on_attempt = fresh, on_attempt
        self.bpb = sl.music.beats_per_bar
        self.asked_bpm = tempo(sl, pack)
        self.grid_bpm = timing.bpm or self.asked_bpm
        self.beat_s = 60.0 / self.grid_bpm
        self.bar_s = self.bpb * self.beat_s
        self.origin = round(timing.beats[0] - timing.preroll_s, 6) if timing.beats else 0.0
        self.total = round(timing.duration_s - timing.preroll_s + pack.music.get("tail_s", 1.5), 6)
        self.film_end = timing.duration_s - timing.preroll_s
        self.fps = timing.fps
        self.frame = 1.0 / timing.fps       # the steadiness tolerance, and the grid gate's: a frame
        self.bed_lufs = pack.music["bed_lufs"]
        self.sections = sections_from_timing(timing, sl)
        # The notes of the prompts paid for before R101: the purposes and the cues, or the purposes alone (a shot list
        # that had no cues yet, as the mascot film's had when its rolls were paid for).
        self.legacy_sections = [sections_from_timing(timing, sl, purposes=True),
                                sections_from_timing(timing, sl, purposes=True, cues=False)]
        self.segs = plan_segments(self.total, self.bar_s, prov.max_call_s, [s.start_s for s in self.sections],
                                  self.origin)
        if len(self.segs) > 1 and not prov.segment_prompts:
            raise MusicError(f"{prov.name} makes a bed in one call of at most {prov.max_call_s:.0f}s, and this one "
                             f"needs {_asked(self.total, self.bar_s):.0f}s with its stretch room; use --provider "
                             f"lyria, which makes long beds in segments, or a shorter film")
        self.only = self._named(segments)
        # The film's own style lines win over the pack's; the prompt is in the cache key, so a change re-rolls.
        self.cfg = {**pack.music, **(sl.music.style.model_dump(exclude_none=True) if sl.music.style else {})}
        n, d = len(self.segs), run.music_dir
        self.pending = [d / f".pending.{_src_name(i)}" for i in range(n)]
        self.decoded = [d / f".seg{i}.src.wav" for i in range(n)]
        self.work = [d / f".seg{i}.wav" for i in range(n)]
        self.heard = [d / f".seg{i}.heard.wav" for i in range(n)]
        self.joined = d / ".bed.f32.wav"      # float, so a hot delivery is not clipped before its level comes down
        self.staged = d / ".bed.staged.wav"
        self.results = [SegmentRun(i, n) for i in range(n)]
        self.current: int | None = None
        self.rolls: dict[int, int] = {}
        self.preferred: dict[int, int] = {}      # the roll each segment's audio was on when it last passed
        self.failed_roll: dict[int, int] = {}    # the roll each segment last failed on
        self.bump: set[int] = set()              # segments `--fresh` re-rolls
        self.bump_failed = False                 # ... because they failed (so a cached passing roll is looked for)

    def _named(self, segments: Collection[int] | None) -> set[int] | None:
        """The segments `--segment` names, as indices (it counts from 1, as the table prints them), or None when it
        names none. A number the bed has no segment for is a usage error, before anything is paid for."""
        if not segments:
            return None
        n = len(self.segs)
        bad = sorted({s for s in segments if not 1 <= s <= n})
        if bad:
            where = "this bed is one segment (--segment 1)" if n == 1 else \
                f"this bed has {n} segments, numbered 1 to {n} as `promo music` prints them"
            raise SegmentError(f"--segment {', '.join(map(str, bad))}: {where}")
        return {s - 1 for s in segments}

    # --- the run -------------------------------------------------------------------------------------------------

    def make(self) -> Path:
        ok, error = False, ""
        try:
            bed = self._build()
            ok = True
            return bed
        except cf.ConformError as e:
            error = str(e)
            raise MusicError(error) from e
        except Exception as e:
            error = str(e)
            if self.current is not None and self.results[self.current].status == "pending":
                self.results[self.current].status, self.results[self.current].reason = "error", error
            raise
        finally:
            for p in (*self.pending, *self.decoded, *self.work, *self.heard, self.joined, self.staged):
                p.unlink(missing_ok=True)
            if not ok:
                self._store_failed_attempt(error)
            if self.on_attempt and self._checked():
                self.on_attempt([r.row() for r in self.results])

    def _checked(self) -> bool:
        return any(r.status in ("passed", "failed") for r in self.results)

    def _attempt(self, ok: bool, error: str) -> dict:
        return {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "fresh": self.fresh, "ok": ok,
                "error": error, "re_rolled": sorted(self.bump), "segments": [r.row() for r in self.results]}

    def _store_failed_attempt(self, error: str) -> None:
        """A failed run leaves bed.wav, beats.json and the report of the bed on disk as they were; only the
        report's `attempt` (which segment failed, and why) is replaced, so the next `--fresh` knows what to re-roll."""
        if not self._checked():
            return
        report = _load_report(self.run)
        report["attempt"] = self._attempt(False, error)
        with atomic_path(self.run.music_dir / "conform.json") as tmp:
            tmp.write_text(json.dumps(report, indent=2))

    def _plan_rerolls(self) -> None:
        """Which segments `--fresh` asks for again: those `--segment` names, whether or not they passed (a new roll
        for each, none taken back from the cache); else those that failed in the last attempt; all of them when none
        did (no attempt recorded, or the last one made a bed); none when that attempt was cut short (a provider
        error) with nothing failed, since then there is nothing to replace, only work to finish. Every other
        segment is as it is in an ordinary run: on the roll it last passed on."""
        last = _last_attempt(self.run, len(self.segs))
        for r in last:
            if isinstance(r.get("index"), int) and isinstance(r.get("roll"), int):
                if r.get("status") == "passed":
                    self.preferred[r["index"]] = r["roll"]
                elif r.get("status") == "failed":
                    self.failed_roll[r["index"]] = r["roll"]
        if not self.fresh:
            return
        if self.only is not None:
            self.bump, self.bump_failed = set(self.only), False
            return
        failed = {r["index"] for r in last if r.get("status") == "failed" and isinstance(r.get("index"), int)}
        if failed:
            self.bump, self.bump_failed = failed, True
        elif not any(r.get("status") in ("error", "pending") for r in last):
            self.bump = set(range(len(self.segs)))

    # --- one segment ---------------------------------------------------------------------------------------------

    def _prompt(self, sections: list[Section], seg: Segment, key: str | None) -> str:
        if self.prov.segment_prompts:
            return build_segment_prompt(self.cfg, sections, self.asked_bpm, self.bpb, seg, self.total, key=key)
        return build_prompt(self.cfg, sections, self.asked_bpm, seg.asked_s, self.bpb)

    def _prompts(self, seg: Segment, key: str | None) -> list[str]:
        """The prompt to send, then the prompts as they were before R101 (the purposes in their notes): a delivery paid
        for with one of those is as good, and is found in the cache by it. The key line is the same in all of them,
        so a delivery asked for another key than segment 1's in use is not a match."""
        out: list[str] = []
        for sections in (self.sections, *self.legacy_sections):
            p = self._prompt(sections, seg, key)
            if p not in out:
                out.append(p)
        return out

    def _units(self, seg: Segment, length_ms: int, roll: int) -> dict:
        return {"music_s": round(length_ms / 1000, 3), "segment": f"{seg.index + 1}/{seg.count}", "roll": roll}

    def _fetch(self, seg: Segment, prompts: list[str], length_ms: int, roll: int, may_pay: bool,
               res: SegmentRun) -> tuple[bytes, str, bool] | None:
        """(audio, the prompt it was asked with, from the cache) for this segment at `roll`, or None when it is not
        cached and `may_pay` is false. A new call is logged as soon as its audio is in hand: the money is spent."""
        for p in prompts:
            hit = cache.cache_get(self.prov.name, segment_cache_key(self.prov, p, length_ms, roll, seg.index, seg.count))
            if hit is not None:
                return hit, p, True
        if not may_pay:
            return None
        audio = self.compose(prompts[0], length_ms)
        cache.cache_put(self.prov.name, segment_cache_key(self.prov, prompts[0], length_ms, roll, seg.index, seg.count),
                        audio)
        usd = self.prov.call_usd(length_ms / 1000)
        costs.log_cost(self.run.costs, provider=self.prov.name, model=self.prov.model,
                       units=self._units(seg, length_ms, roll), usd=usd, cached=False)
        res.usd += usd
        return audio, prompts[0], False

    def _resolve(self, seg: Segment, key: str | None) -> _Won | None:
        """Get a delivery for `seg` that passes its own checks, or None. Which rolls, in order:

        - an ordinary run: the roll its audio was last on (cached, or paid for when it has none), then, if that fails,
          every other roll in the cache;
        - `--fresh` for a segment that failed: the rolls in the cache first (not the one that failed), for free; then
          a new roll, paid for;
        - `--fresh` for every segment (a bed that passed, or none recorded): a new roll only.

        A cached delivery counts only if it was asked with the prompt of now, which names segment 1's key."""
        i, res = seg.index, self.results[seg.index]
        prompts = self._prompts(seg, key)
        length_ms = self.prov.clamp_ms(seg.asked_s * 1000)
        current = self.rolls.get(i, 0)
        cached = [r for r in range(current, -1, -1)]
        if i in self.bump and self.bump_failed:
            home = current + 1
            order = [(r, False) for r in cached if r != self.failed_roll.get(i)] + [(home, True)]
        elif i in self.bump:
            home = current + 1
            order = [(home, True)]
        else:
            home = self.preferred.get(i, current)
            order = [(home, True)] + [(r, False) for r in cached if r != home]
        reasons: dict[int, str] = {}
        for roll, may_pay in order:
            got = self._fetch(seg, prompts, length_ms, roll, may_pay, res)
            if got is None:
                continue
            audio, prompt, hit = got
            if roll > self.rolls.get(i, 0):         # a new roll is paid for: keep it, so a failure is re-rolled past it
                self.rolls[i] = roll
                save_rolls(self.run, self.rolls)
            ev = self._evaluate(seg, audio, res)
            if ev.ok:
                if hit:
                    costs.log_cost(self.run.costs, provider=self.prov.name, model=self.prov.model,
                                   units=self._units(seg, length_ms, roll), usd=self.prov.call_usd(length_ms / 1000),
                                   cached=True)
                res.status, res.roll, res.reason = "passed", roll, ""
                res.source = f"reused from roll {roll}" if roll != home else ("cached" if hit else "fresh")
                return _Won(ev.placement, ev.phase, ev.words, prompt)
            reasons[roll] = ev.reason
            res.tried.append({"roll": roll, "reason": _plain(ev.reason)})
        res.status = "failed"
        res.roll = home if home in reasons else list(reasons)[-1]
        res.reason = reasons[res.roll]
        res.source = "fresh" if res.usd else "cached"
        return None

    def _evaluate(self, seg: Segment, audio: bytes, res: SegmentRun) -> _Eval:
        """A delivery's own checks: it decodes; it has a steady beat, within the stretch the conform may take, that
        is fitted to the planned grid; its downbeat is on the planned bar line (moved by whole beats when it is
        not and the kick says so clearly); it has music for as long as the film needs it; and, for a provider with
        no instrumental switch, nobody sings in it."""
        i, n = seg.index, seg.count
        what = "the bed" if n == 1 else f"segment {i + 1} of {n}"
        src, dec, out = self.pending[i], self.decoded[i], self.work[i]
        with atomic_path(src) as tmp:
            tmp.write_bytes(audio)
        try:
            run_cmd(["ffmpeg", "-v", "error", "-y", "-i", src, "-ar", "48000", "-ac", "2", "-c:a", "pcm_f32le", dec])
        except ToolError:
            res.status, res.reason = "failed", f"{what}: the delivery cannot be decoded as audio; {cf.REROLL}"
            raise
        try:
            placed = cf.conform(dec, out, bpm=self.grid_bpm, beats_per_bar=self.bpb,
                                origin_s=self.origin if i == 0 else 0.0,
                                span_s=seg.audio_end_s - seg.audio_start_s, what=what, steady_tol_s=self.frame)
        except cf.ConformError as e:
            return _Eval(False, str(e))
        last = i == n - 1
        heard_end = (self.film_end if last else seg.next_seam_s) - seg.audio_start_s
        if i == 0:   # its first downbeat is bar 1 by the conform; how clearly the kick says so
            found, margin = cf.downbeat_phase(out, bpm=self.grid_bpm, beats_per_bar=self.bpb, start_s=self.origin,
                                              end_s=heard_end)
            moved = 0
        else:
            found, moved, margin = _align_downbeat(out, bpm=self.grid_bpm, beats_per_bar=self.bpb, end_s=heard_end)
        available = placed.available_s + moved * self.beat_s
        needed = (self.film_end if last else seg.next_seam_s + self.beat_s) - seg.audio_start_s
        if available < needed - 1e-3:
            return _Eval(False, f"{self.prov.name} returned {available:.1f}s of music for {what}, which needs "
                                f"{needed:.1f}s; {cf.REROLL}")
        words = None
        if self.prov.check_vocals:
            cf.fit_length(out, self.heard[i], heard_end)        # what is heard of it, not what the provider played on
            heard = (self.transcribe or _words)(self.heard[i])
            words = len(heard)
            if words > MAX_VOICE_WORDS:
                return _Eval(False, f"vocals detected in the {self.prov.name} {'bed' if n == 1 else what} ({words} "
                                    f"words, \"{' '.join(heard[:12])}\"); {cf.REROLL}")
        return _Eval(True, "", placed, {"found_on_beat": found, "moved_beats": moved, "margin": margin}, words)

    # --- the joined bed ------------------------------------------------------------------------------------------

    def _segment_at(self, t: float) -> int:
        """The segment heard at bed time `t`."""
        return max((s.index for s in self.segs if s.index == 0 or t >= s.seam_s), default=0)

    def _blame(self, index: int, message: str) -> None:
        self.results[index].status, self.results[index].reason = "failed", message

    def _failure(self, failed: list[int]) -> str:
        rows = [self.results[i] for i in failed]
        if len(rows) == 1:
            return rows[0].reason
        return (f"{'; '.join(_plain(r.reason) for r in rows)}; {len(rows)} segments failed their checks: re-roll them "
                f"with `promo music --fresh` (it re-rolls only these)")

    def _build(self) -> Path:
        run, prov, segs, n = self.run, self.prov, self.segs, len(self.segs)
        grid_bpm, bpb, origin, film_end, frame = self.grid_bpm, self.bpb, self.origin, self.film_end, self.frame
        run.music_dir.mkdir(parents=True, exist_ok=True)
        self.rolls = load_rolls(run, n)
        self._plan_rerolls()
        wons: list[_Won | None] = [None] * n
        failed: list[int] = []
        music_key: str | None = None
        for seg in segs:
            self.current = seg.index
            if seg.index == 1 and not self.cfg.get("key"):    # later parts are asked for the first part's key
                music_key = cf.estimate_key(self.decoded[0])
            won = self._resolve(seg, music_key)
            if won is None:
                failed.append(seg.index)
                if seg.index == 0:       # the rest are asked for its key: nothing to pay for until it passes
                    break
            else:
                wons[seg.index] = won
        self.current = None
        if failed:
            raise MusicError(self._failure(failed))
        placements, phases = [w.placement for w in wons], [w.phase for w in wons]

        if n == 1:
            # judged on the bed's own span, but on at least 8 bars: a short bed has too few to measure alone
            check = cf.check_grid(self.work[0], bpm=grid_bpm, beats_per_bar=bpb, origin_s=origin,
                                  seconds=max(self.total, 8 * self.bar_s), steady_tol_s=frame)
            cf.fit_length(self.work[0], self.joined, self.total)
        else:
            cf.stitch([(w, s.audio_start_s) for w, s in zip(self.work, segs)], [s.seam_s for s in segs[1:]],
                      self.joined, beat_s=self.beat_s, length_s=self.total)
            check = cf.check_grid(self.joined, bpm=grid_bpm, beats_per_bar=bpb, origin_s=origin, steady_tol_s=frame)
        if check.max_error_s > frame + 1e-9:
            off_grid = any(a <= check.worst_bar <= b for a, b in check.off_grid)
            how = ("is off the grid by {:+.0f} ms: its drums land outside the ±60 ms each beat is searched in"
                   if off_grid else "is {:+.0f} ms off the planned grid (the median of its beats)")
            message = (f"bar {check.worst_bar} (at {check.at_s:.1f}s) of the conformed bed "
                       f"{how.format(check.worst_offset_s * 1000)}, more than a frame at {self.fps} fps; {cf.REROLL}")
            blamed = {self._segment_at(check.at_s): message}
            for first, last_bar in check.off_grid:      # every segment with drums off the grid, not only the worst bar's
                at = origin + (first - 1) * self.bar_s
                blamed.setdefault(self._segment_at(at), f"bars {first}-{last_bar} (at {at:.1f}s) of the conformed bed "
                                                         f"are off the grid: its drums land outside the ±60 ms each "
                                                         f"beat is searched in; {cf.REROLL}")
            for index, why in blamed.items():
                self._blame(index, why)
            if len(blamed) > 1:
                message += f" (segments {', '.join(str(i + 1) for i in sorted(blamed))} are off the grid)"
            raise MusicError(message)
        downbeats, off_beat = [], []
        for seg in segs:   # after stitching: every segment's bars are the plan's, or the bed is re-rolled
            start = origin if seg.index == 0 else seg.seam_s
            end = film_end if seg.index == n - 1 else seg.next_seam_s
            found, margin = cf.downbeat_phase(self.joined, bpm=grid_bpm, beats_per_bar=bpb, start_s=start, end_s=end)
            downbeats.append({"from_s": round(start, 3), "found_on_beat": found, "margin": margin})
            if found and margin >= cf.DOWNBEAT_MARGIN:
                where = "bar 1" if seg.index == 0 else f"seam {seg.index} (at {seg.seam_s:.1f}s)"
                off_beat.append((seg.index, f"{where}: the downbeat is off by {found} beat{'s' if found > 1 else ''} "
                                            f"from the planned bar lines; {cf.REROLL}"))
        if off_beat:
            for index, why in off_beat:
                self._blame(index, why)
            raise MusicError(off_beat[0][1])
        seams = [{**cf.seam_report(self.joined, s.seam_s, beat_s=self.beat_s, bar_s=self.bar_s),
                  "planned_change": s.on_section} for s in segs[1:]]
        normalise_bed(self.joined, self.staged, self.bed_lufs)
        bed = run.music_dir / "bed.wav"
        # the bed passed: its sources, bed.wav and its grid replace the previous set together
        for i, src in enumerate(self.pending):
            src.replace(run.music_dir / _src_name(i))
        for stale in run.music_dir.glob("bed.src.*.mp3"):     # segments of an earlier, longer plan
            if stale.name not in {_src_name(i) for i in range(n)}:
                stale.unlink()
        self.staged.replace(bed)

        prompts = [w.prompt for w in wons]
        with atomic_path(run.music_dir / "prompt.txt") as tmp:
            tmp.write_text(prompts[0] if n == 1 else "\n\n".join(
                f"--- segment {i + 1} of {n} ---\n{p}" for i, p in enumerate(prompts)))
        grid_path = run.music_dir / "beats.json"
        grid_path.unlink(missing_ok=True)  # a grid of the previous bed must not outlive it if the new one cannot be measured
        write_grid(_conformed_grid(check.fit, origin, bpb, onset_times(bed), self.total), grid_path)
        warnings = grid_warnings(check, self.fps)
        for seg, ph in zip(segs, phases):
            if ph["margin"] < cf.DOWNBEAT_MARGIN:
                where = (f"bar 1 at {origin:.1f}s" if seg.index == 0 else f"seam at {seg.seam_s:.1f}s")
                warnings.append(f"{where}: the {'bed' if n == 1 else f'segment {seg.index + 1}'} downbeat is a "
                                f"guess (its kick is as strong on other beats, margin {ph['margin']:.2f}); listen that "
                                f"the bars line up there")
        warnings += seam_warnings(seams)
        # the segment to re-roll for each warned seam: the later one (the first segment has no seam of its own)
        warned = {seg.index for seg, ph in zip(segs, phases) if seg.index and ph["margin"] < cf.DOWNBEAT_MARGIN}
        warned |= {seg.index for seg, row in zip(segs[1:], seams) if seam_warnings([row])}
        seam_rerolls = [{"seam_s": segs[i].seam_s, "segment": i + 1} for i in sorted(warned)]
        words = [w.words for w in wons if w.words is not None]
        report = {
            "provider": prov.name, "model": prov.model, "planned_bpm": grid_bpm, "beats_per_bar": bpb,
            "origin_s": origin, "fps": self.fps, "stretch": "atempo",
            "stretch_ratio": placements[0].ratio, "shift_s": placements[0].shift_s,
            "grid": {
                "limit_s": round(frame, 6),
                "bar_max_error_s": check.max_error_s, "bar_max_error_frames": round(check.max_error_s * self.fps, 3),
                "worst_bar": check.worst_bar, "worst_bar_at_s": check.at_s,
                "raw_max_error_s": check.raw_max_error_s, "raw_max_error_at_s": check.raw_at_s,
                "bars_measured": check.bars_measured, "bars_sounding": check.bars_sounding,
                "unmeasured_bars": [list(r) for r in check.unmeasured],
                "off_grid_bars": [list(r) for r in check.off_grid],
            },
            "max_error_s": check.max_error_s, "measured_bpm": check.fit.bpm,
            "sections_note": "; ".join(
                f"segment {i + 1}: the stretch moves its sections x{1 / p.ratio:.3f} (a change the provider put at "
                f"1:00.0 of its audio lands at {_mmss(60.0 / p.ratio)})" for i, p in enumerate(placements)),
            "segments": [{"src": _src_name(seg.index), "roll": r.roll, "source": r.source, "passed": True,
                          "reason": "", "bed_start_s": seg.audio_start_s, "seam_s": seg.seam_s,
                          "asked_s": seg.asked_s, "measured_bpm": p.measured_bpm, "stretch_ratio": p.ratio,
                          "section_time_scale": round(1 / p.ratio, 4), "shift_s": p.shift_s,
                          "available_s": p.available_s, "downbeat": ph}
                         for seg, r, p, ph in zip(segs, self.results, placements, phases)],
            "downbeat_check": downbeats,
            "seams": seams,
            **({"vocals_words": sum(words)} if words else {}),
            "warnings": warnings,
            "seam_rerolls": seam_rerolls,
            "attempt": self._attempt(True, ""),
        }
        with atomic_path(run.music_dir / "conform.json") as tmp:
            tmp.write_text(json.dumps(report, indent=2))
        return bed


def make_bed(run: RunPaths, sl: Shotlist, pack: Pack, timing: Timing,
             compose: Callable[[str, float], bytes] | None = None, *, fresh: bool = False,
             provider: str = DEFAULT_PROVIDER, transcribe: Callable[[Path], list[str]] | None = None,
             on_attempt: Callable[[list[dict]], None] | None = None,
             segments: Collection[int] | None = None) -> Path | None:
    """Generate the bed with `provider` (cached by provider, model, prompt, length, roll and segment), conform it to
    the planned grid, convert it to 48 kHz stereo s24 at the pack's bed_lufs, and write its measured grid.

    The bed starts at `timing.preroll_s` in the film; its length runs to the film's end plus the pack's tail. The
    planned grid is the timing's (beat k at `timing.beats[k]`), or the asked tempo from the bed's start when the
    film has none. A bed longer than one call can make comes in bar-aligned segments, each conformed on its own,
    its downbeat put on the plan's bar line, and joined on a downbeat.

    Each segment is checked on its own (a steady beat within a frame's wander, a stretch of at most 6%, a downbeat on
    the plan's bar line, music for as long as the film needs it, no singing) and the joined bed is checked as a
    whole (every measured bar's median beat within a frame of the plan, ruling R97; every segment's bars the plan's).
    bed.wav, bed.src*.mp3, beats.json, conform.json and prompt.txt are written only when it passes; a failure
    leaves the previous set as it was, and records which segments failed in conform.json's `attempt`.
    Returns None when the pack's music mode is 'none'.

    The same plan always returns the same cached bed. Every segment has its own roll number (part of the cache key,
    kept in audio/music/roll.json as soon as the new roll's audio is paid for). `fresh=True` re-rolls only the
    segments that failed in the last attempt (all of them when none did: nothing recorded, or the last attempt made
    a bed). A failed segment first tries the other rolls in the cache, asked for the same prompt (so for the same
    key as segment 1 now in use), and is re-rolled only if none of them passes. `segments` (with `fresh=True`) names
    the segments to re-roll instead, numbered from 1 as the table prints them: exactly those get a new roll, whether
    or not they passed, and every other segment keeps the roll it is on; a number the bed has no segment for, or
    `segments` without `fresh`, is a SegmentError before anything is paid for. conform.json's `seam_rerolls` names,
    for each seam that has a warning, the segment to re-roll for it. `on_attempt` is called with each segment's row
    (roll, source, result, reason, cost) when the run ends, whether it made a bed or not.
    """
    if pack.music.get("mode", "bed") == "none":
        return None
    if provider not in PROVIDERS:
        raise MusicError(f"unknown music provider '{provider}' (use {' or '.join(PROVIDERS)})")
    if segments and not fresh:
        raise SegmentError("--segment needs --fresh: it names the segments to re-roll")
    prov = PROVIDERS[provider]
    return _Bed(run, sl, pack, timing, prov, compose or prov.compose, transcribe, fresh, on_attempt, segments).make()
