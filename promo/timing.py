"""Resolve the authored shot list plus measured audio into timing.json."""

from __future__ import annotations

import math
import re
from pathlib import Path

from pydantic import BaseModel

from promo.models import BeatGrid, Caption, Finding, LineTake, WordPart
from promo.packs import Limits
from promo.proc import atomic_path
from promo.shotlist import (
    AfterAnchor,
    BarAnchor,
    BeatAnchor,
    Chapter,
    ShotAnchor,
    Shotlist,
    TimeAnchor,
    WordAnchor,
    script_text,
)


class TimingError(ValueError):
    """The shot list cannot be resolved at all."""


class TimedLine(BaseModel):
    id: str
    chapter: str
    text: str
    start_s: float
    end_s: float
    file: str | None = None
    estimated: bool = False
    words: list[Caption] = []


class TimedEvent(BaseModel):
    id: str
    shot: str
    chapter: str
    kind: str
    t: float
    t_local: float
    sfx: str | None = None
    value: float | None = None


class TimedRead(BaseModel):
    shot: str
    what: str
    start_s: float
    end_s: float


class TimedShot(BaseModel):
    id: str
    chapter: str
    start_s: float
    end_s: float
    engine: str
    source: str | None = None
    text_mode: str

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 6)


class TimedChapter(BaseModel):
    id: str
    start_s: float
    end_s: float
    headline: dict | None = None


class Timing(BaseModel):
    version: int = 1
    duration_s: float
    fps: int
    width: int
    height: int
    aspect: str
    preroll_s: float
    timing_mode: str
    bpm: float | None
    beats_per_bar: int
    beats: list[float]
    limits: Limits
    chapters: list[TimedChapter]
    shots: list[TimedShot]
    lines: list[TimedLine]
    events: list[TimedEvent]
    reads: list[TimedRead]
    findings: list[Finding]

    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    def shot(self, shot_id: str) -> TimedShot:
        return next(s for s in self.shots if s.id == shot_id)

    def line(self, line_id: str) -> TimedLine:
        return next(l for l in self.lines if l.id == line_id)

    def chapter(self, chapter_id: str) -> TimedChapter:
        return next(c for c in self.chapters if c.id == chapter_id)

    def events_in(self, shot_id: str) -> list[TimedEvent]:
        return [e for e in self.events if e.shot == shot_id]


class Measured(BaseModel):
    takes: dict[str, LineTake] = {}
    grid: BeatGrid | None = None


class _Grid:
    def __init__(self, sl: Shotlist, grid: BeatGrid | None):
        self.bpb = sl.music.beats_per_bar
        self.pre = sl.preroll_s
        self.beats: list[float] = []
        if grid is not None and grid.beats:
            self.bpm: float | None = grid.bpm
            i0 = min(range(len(grid.beats)), key=lambda i: abs(grid.beats[i] - grid.first_downbeat_s))
            self.beats = [round(b + self.pre, 6) for b in grid.beats[i0:]]
        else:
            self.bpm = sl.music.bpm
        self.beat_s = 60.0 / self.bpm if self.bpm else None
        self.origin = self.beats[0] if self.beats else self.pre

    def time_of_beat(self, k: float) -> float:
        if self.beat_s is None:
            raise TimingError("no tempo: set music.bpm in shotlist.json or run `promo music` first")
        if not self.beats:
            return self.origin + k * self.beat_s
        n = len(self.beats)
        if k <= 0:
            return self.beats[0] + k * self.beat_s
        i = int(math.floor(k))
        if i + 1 < n:
            return self.beats[i] + (k - i) * (self.beats[i + 1] - self.beats[i])
        return self.beats[-1] + (k - (n - 1)) * self.beat_s

    def next_beat_at_or_after(self, t: float) -> float:
        if self.beat_s is None:
            return t
        k = max(0, math.ceil((t - self.origin) / self.beat_s - 1e-9))
        while self.time_of_beat(k) < t - 1e-6:
            k += 1
        while k > 0 and self.time_of_beat(k - 1) >= t - 1e-6:
            k -= 1
        return self.time_of_beat(k)


def _norm(word: str) -> str:
    return re.sub(r"[^\w]+", "", word.lower())


def _anchorable(words: list[Caption]) -> list[tuple[str, float]]:
    """(normalised text, start in ms) of everything a word anchor can name, in time order.

    A line's words are the script's own (see promo.asr.align_to_script), so an anchor names a script word. A hyphenated
    word can also be named by its parts ("read" or "only" in "read-only"), the way scoring reads it: a part starts where
    ASR timed it (`Caption.parts`), or, when the take has no part times, where its letters fall in the word.
    """
    out: list[tuple[str, float]] = []
    for w in words:
        out.append((_norm(w.text), w.startMs))
        if w.parts:
            out.extend((_norm(p.text), p.startMs) for p in w.parts)
            continue
        pieces = [p for p in re.split(r"[-\u2010\u2011]", w.text) if _norm(p)]
        if len(pieces) > 1:
            total, done = sum(len(_norm(p)) for p in pieces), 0
            for piece in pieces:
                out.append((_norm(piece), w.startMs + (w.endMs - w.startMs) * done / total))
                done += len(_norm(piece))
    return out


def _estimated_words(text: str, duration: float) -> list[Caption]:
    words = text.split()
    step = duration / max(1, len(words))
    return [Caption(text=w, startMs=round(i * step * 1000), endMs=round((i + 1) * step * 1000)) for i, w in enumerate(words)]


class _Resolver:
    def __init__(self, sl: Shotlist, measured: Measured, limits: Limits, text_mode_default: str):
        self.sl = sl
        self.measured = measured
        self.lim = limits
        self.text_mode_default = text_mode_default
        self.grid = _Grid(sl, measured.grid)
        self.findings: list[Finding] = []
        self.lines: dict[str, TimedLine] = {}
        self.shots: dict[str, TimedShot] = {}
        self.ch_ctx: dict[str, tuple[float, float | None]] = {}
        self.fps = sl.canvas.fps

    def q(self, t: float) -> float:
        return round(round(t * self.fps) / self.fps, 6)

    def q_up(self, t: float) -> float:
        """Round up to a whole frame, so a span sized from its content never shrinks below it."""
        return round(math.ceil(t * self.fps - 1e-3) / self.fps, 6)

    def err(self, code: str, message: str, ref: str | None = None) -> None:
        self.findings.append(Finding(level="error", code=code, message=message, ref=ref))

    def warn(self, code: str, message: str, ref: str | None = None) -> None:
        self.findings.append(Finding(level="warning", code=code, message=message, ref=ref))

    def word_time(self, line: TimedLine, a: WordAnchor, where: str) -> float:
        hits = [start for text, start in _anchorable(line.words) if text == _norm(a.word)]
        if len(hits) >= a.nth:
            return hits[a.nth - 1] / 1000
        self.err("word_not_found", f"{where}: word '{a.word}' (occurrence {a.nth}) not found in {line.id}: \"{line.text}\"", where)
        return line.start_s

    def anchor_time(self, a, ch_id: str, where: str) -> float:
        ch_start, ch_start_beat = self.ch_ctx[ch_id]
        g = self.grid
        if isinstance(a, TimeAnchor):
            self.warn("t_anchor", f"{where}: uses a raw time; prefer a beat or word anchor", where)
            return ch_start + a.t
        if isinstance(a, (BeatAnchor, BarAnchor)):
            beats = a.beat if isinstance(a, BeatAnchor) else a.bar * g.bpb
            if ch_start_beat is not None:
                return g.time_of_beat(ch_start_beat + beats)
            if g.beat_s is None:
                raise TimingError(f"{where}: beat anchor needs a tempo (music.bpm)")
            return ch_start + beats * g.beat_s
        if isinstance(a, AfterAnchor):
            if a.after not in self.lines:
                raise TimingError(f"{where}: line '{a.after}' must be placed before it is referenced")
            return self.lines[a.after].end_s + a.gap
        if isinstance(a, WordAnchor):
            if a.line not in self.lines:
                raise TimingError(f"{where}: line '{a.line}' must be placed before it is referenced")
            return self.word_time(self.lines[a.line], a, where) + a.offset
        if isinstance(a, ShotAnchor):
            if a.shot not in self.shots:
                raise TimingError(f"{where}: shot '{a.shot}' is not resolved yet")
            s = self.shots[a.shot]
            return (s.start_s if a.edge == "start" else s.end_s) + a.offset
        raise TimingError(f"{where}: unsupported anchor {a!r}")

    def place_lines(self, ch: Chapter, ch_start: float) -> float | None:
        prev_end: float | None = None
        for line in ch.vo:
            take = self.measured.takes.get(line.id)
            # a take's words are the script's own (promo.asr.align_to_script), so a take whose words differ was voiced
            # from older copy: its audio says the old line and its word times are not this line's
            stale = take is not None and bool(take.words) and [w.text for w in take.words] != script_text(line.text).split()
            if take is not None and not take.problem and not stale:   # an unusable or stale take is no take at all
                dur, words, file, est = take.duration_s, take.words, take.file, False
            else:
                dur = max(1, len(line.text.split())) / self.lim.pace_wpm * 60.0
                words, file, est = _estimated_words(line.text, dur), None, True
                if stale and not take.problem:
                    self.warn("stale_take", f"{line.id}: the take was voiced from older copy; duration estimated at {dur:.2f}s", line.id)
                else:
                    why = f"no usable voice take ({take.problem})" if take is not None else "no voice take yet"
                    self.warn("no_take", f"{line.id}: {why}; duration estimated at {dur:.2f}s", line.id)
            if line.at is None:
                start = ch_start + self.lim.head_pad_s if prev_end is None else prev_end + self.lim.line_gap_s
            else:
                start = self.anchor_time(line.at, ch.id, line.id)
            base = start * 1000
            self.lines[line.id] = TimedLine(
                id=line.id, chapter=ch.id, text=line.text, start_s=round(start, 6), end_s=round(start + dur, 6),
                file=file, estimated=est,
                words=[Caption(text=w.text, startMs=round(base + w.startMs), endMs=round(base + w.endMs),
                               timestampMs=round(base + w.startMs), confidence=w.confidence,
                               parts=[WordPart(text=p.text, startMs=round(base + p.startMs), endMs=round(base + p.endMs))
                                      for p in w.parts]) for w in words],
            )
            prev_end = start + dur
        return prev_end

    def chapters(self) -> list[TimedChapter]:
        out: list[TimedChapter] = []
        cursor_beat, cursor_s = 0.0, self.sl.preroll_s
        for ch in self.sl.chapters:
            if self.sl.timing_mode == "music":
                start_beat = ch.start.bar * self.grid.bpb if isinstance(ch.start, BarAnchor) else cursor_beat
                end_beat = start_beat + ch.bars * self.grid.bpb
                start_s, end_s = self.grid.time_of_beat(start_beat), self.grid.time_of_beat(end_beat)
                cursor_beat = end_beat
                self.ch_ctx[ch.id] = (start_s, start_beat)
                self.place_lines(ch, start_s)
                for line in ch.vo:
                    tl = self.lines[line.id]
                    if tl.end_s > end_s - self.lim.tail_pad_s + 1e-6:
                        self.err("vo_overrun", f"{line.id} ends at {tl.end_s:.2f}s but chapter {ch.id} ends at {end_s:.2f}s "
                                 f"(tail pad {self.lim.tail_pad_s}s); shorten the line or give the chapter more bars", line.id)
            else:
                start_s = cursor_s
                self.ch_ctx[ch.id] = (start_s, None)
                last = self.place_lines(ch, start_s)
                vo_end = last + self.lim.tail_pad_s if last is not None else start_s
                end_s = max(vo_end, start_s + sum(s.seconds or 0 for s in ch.shots))
                if end_s <= start_s + 1e-9:
                    raise TimingError(f"chapter {ch.id} has no VO and no shot 'seconds'; cannot size it")
                end_s = self.q_up(self.grid.next_beat_at_or_after(end_s))
                cursor_s = end_s
            out.append(TimedChapter(id=ch.id, start_s=self.q(start_s), end_s=self.q(end_s),
                                    headline=ch.headline.model_dump() if ch.headline else None))
        return out

    def split_shots(self, ch: Chapter, tch: TimedChapter) -> list[tuple[float, float]]:
        _, start_beat = self.ch_ctx[ch.id]
        music = self.sl.timing_mode == "music" and start_beat is not None
        total = ch.bars * self.grid.bpb if music else tch.end_s - tch.start_s
        length_of = (lambda s: s.beats) if music else (lambda s: s.seconds)
        fixed = sum(length_of(s) for s in ch.shots if length_of(s) is not None)
        free = [s for s in ch.shots if length_of(s) is None]
        rest = total - fixed
        # A voice-mode chapter length is a whole number of frames, so it can fall short of the
        # authored seconds by under a frame (a chapter start that is off the frame grid). The
        # last shot absorbs that shortfall. Music-mode lengths are exact beats.
        slack = 0.0 if music else 1.0 / self.fps
        if rest < -max(1e-6, slack):
            raise TimingError(f"chapter {ch.id}: shot lengths add up to {fixed}, but the chapter has {total}")
        deficit, rest = min(rest, 0.0), max(rest, 0.0)
        each = rest / len(free) if free else 0.0
        lengths = [length_of(s) if length_of(s) is not None else each for s in ch.shots]
        if not free and rest > 1e-6:
            lengths[-1] += rest
        lengths[-1] = max(0.0, lengths[-1] + deficit)
        spans, pos = [], (start_beat if music else tch.start_s)
        for length in lengths:
            a, b = pos, pos + length
            spans.append((self.grid.time_of_beat(a), self.grid.time_of_beat(b)) if music else (a, b))
            pos = b
        quantised = [(self.q(a), self.q(b)) for a, b in spans]
        if music:
            return quantised
        return [(min(a, tch.end_s), min(b, tch.end_s)) for a, b in quantised]

    def run(self) -> Timing:
        chapters = self.chapters()
        first = True
        for ch, tch in zip(self.sl.chapters, chapters):
            for shot, (s0, s1) in zip(ch.shots, self.split_shots(ch, tch)):
                self.shots[shot.id] = TimedShot(
                    id=shot.id, chapter=ch.id, start_s=0.0 if first else s0, end_s=s1, engine=shot.engine,
                    source=shot.source, text_mode=shot.text_mode or self.text_mode_default)
                first = False
        events: list[TimedEvent] = []
        reads: list[TimedRead] = []
        for ch in self.sl.chapters:
            for shot in ch.shots:
                ts = self.shots[shot.id]
                for ev in shot.events:
                    t = self.anchor_time(ev.at, ch.id, ev.id)
                    # Shot edges are whole frames but the exact event time is kept for audio, so
                    # judge membership by the frame the event lands on.
                    frame = round(t * self.fps)
                    if not (round(ts.start_s * self.fps) <= frame < round(ts.end_s * self.fps)):
                        self.err("event_outside_shot", f"{ev.id} at {t:.2f}s is outside {shot.id} [{ts.start_s:.2f}, {ts.end_s:.2f})", ev.id)
                    events.append(TimedEvent(id=ev.id, shot=shot.id, chapter=ch.id, kind=ev.kind, t=round(t, 4),
                                             t_local=max(0.0, round(t - ts.start_s, 4)), sfx=ev.sfx, value=ev.value))
                shot_reads = []
                for rd in shot.reads:
                    r0 = self.anchor_time(rd.from_, ch.id, f"{shot.id} read")
                    r1 = self.anchor_time(rd.to, ch.id, f"{shot.id} read")
                    if r1 - r0 < self.lim.min_read_s - 1e-6:
                        self.err("read_too_short", f"{shot.id}: read '{rd.what}' lasts {r1 - r0:.2f}s (< {self.lim.min_read_s}s)", shot.id)
                    shot_reads.append(TimedRead(shot=shot.id, what=rd.what, start_s=round(r0, 4), end_s=round(r1, 4)))
                shot_reads.sort(key=lambda r: r.start_s)
                for a, b in zip(shot_reads, shot_reads[1:]):
                    if b.start_s < a.end_s - 1e-6:
                        self.err("reads_overlap", f"{shot.id}: reads '{a.what}' and '{b.what}' overlap", shot.id)
                reads.extend(shot_reads)
        self.validate(chapters, events)
        duration = chapters[-1].end_s
        beats: list[float] = []
        if self.grid.beat_s:
            k = 0
            while (bt := self.grid.time_of_beat(k)) <= duration + 1e-6:
                beats.append(round(bt, 6))
                k += 1
        sl = self.sl
        return Timing(
            duration_s=duration, fps=sl.canvas.fps, width=sl.canvas.width, height=sl.canvas.height,
            aspect=sl.canvas.aspect, preroll_s=sl.preroll_s, timing_mode=sl.timing_mode, bpm=self.grid.bpm,
            beats_per_bar=self.grid.bpb, beats=beats, limits=self.lim, chapters=chapters,
            shots=list(self.shots.values()), lines=list(self.lines.values()),
            events=sorted(events, key=lambda e: e.t), reads=reads, findings=self.findings,
        )

    def validate(self, chapters: list[TimedChapter], events: list[TimedEvent]) -> None:
        lines = sorted(self.lines.values(), key=lambda l: l.start_s)
        for a, b in zip(lines, lines[1:]):
            if b.start_s < a.end_s - 1e-6:
                self.err("lines_overlap", f"{b.id} starts at {b.start_s:.2f}s before {a.id} ends at {a.end_s:.2f}s", b.id)
        times = sorted(e.t for e in events)
        hook_limit = self.sl.preroll_s + self.lim.hook_by_s
        if not times or times[0] > hook_limit + 1e-6:
            first = f"{times[0]:.2f}s" if times else "never"
            self.err("late_hook", f"first event happens at {first}; the hook must land by {hook_limit:.2f}s")
        points = [self.sl.preroll_s, *times, chapters[-1].end_s]
        for a, b in zip(points, points[1:]):
            if b - a > self.lim.max_dead_s + 1e-6:
                self.err("dead_time", f"nothing happens between {a:.2f}s and {b:.2f}s (> {self.lim.max_dead_s}s)")


def resolve(sl: Shotlist, measured: Measured, limits: Limits, text_mode_default: str = "paraphrase") -> Timing:
    return _Resolver(sl, measured, limits, text_mode_default).run()


def write_timing(t: Timing, path: Path) -> None:
    with atomic_path(Path(path)) as tmp:
        tmp.write_text(t.model_dump_json(indent=2))


def load_timing(path: Path) -> Timing:
    return Timing.model_validate_json(Path(path).read_text())


def load_measured(run_dir: Path) -> Measured:
    takes: dict[str, LineTake] = {}
    vo_dir = Path(run_dir) / "audio" / "vo"
    if vo_dir.is_dir():
        for f in sorted(vo_dir.glob("*.json")):
            take = LineTake.model_validate_json(f.read_text())
            takes[take.id] = take
    grid = None
    beats = Path(run_dir) / "audio" / "music" / "beats.json"
    if beats.exists():
        grid = BeatGrid.model_validate_json(beats.read_text())
    return Measured(takes=takes, grid=grid)
