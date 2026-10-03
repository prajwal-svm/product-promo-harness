"""The authored shot list: schema and cross-reference validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

TextMode = Literal["paraphrase", "verbatim-sync", "labels-only", "none"]
Engine = Literal["hyperframes", "manim", "revideo", "footage", "gen", "still"]


def check_aliases(value: object, where: str = "asr_aliases") -> dict[str, list[str]]:
    """`{word: [what ASR writes, ...]}` with no blank entries, or a ValueError naming `where`.

    A bare string is rejected rather than read letter by letter.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{where} must map each word to a list of what ASR writes")
    for canonical, heard in value.items():
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError(f"{where} has an empty word")
        if (not isinstance(heard, list) or not heard
                or not all(isinstance(h, str) and h.strip() for h in heard)):
            raise ValueError(f"{where}['{canonical}'] needs a non-empty list of what ASR writes, not {heard!r}")
    return value


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class BarAnchor(_Strict):
    bar: float


class BeatAnchor(_Strict):
    beat: float


class WordAnchor(_Strict):
    line: str
    word: str
    nth: int = 1
    offset: float = 0.0


class AfterAnchor(_Strict):
    after: str
    gap: float = 0.0


class ShotAnchor(_Strict):
    shot: str
    edge: Literal["start", "end"]
    offset: float = 0.0


class TimeAnchor(_Strict):
    t: float


Anchor = Union[BarAnchor, BeatAnchor, WordAnchor, AfterAnchor, ShotAnchor, TimeAnchor]


class Claim(_Strict):
    id: str
    text: str
    source: str


class VOLine(_Strict):
    id: str
    text: str
    style: str = ""
    at: Anchor | None = None
    claims: list[str] = []
    # False skips ASR and the WER retake for a line speech recognition cannot score, such as "Psst.".
    # The pace and clip checks still run; `promo lint-script` warns, so it stays rare.
    verify: bool = True


class Event(_Strict):
    id: str
    kind: str
    at: Anchor
    sfx: str | None = None
    value: float | None = None


class Read(_Strict):
    what: str
    from_: Anchor = Field(alias="from")
    to: Anchor


class Visual(_Strict):
    composition: str = ""
    objects: list[str] = []
    camera: str = ""
    locked: list[str] = []


class Motion(_Strict):
    entrance: str = ""
    primary: str = ""
    secondary: str = ""
    exit: str = ""


class Sound(_Strict):
    music_cue: str | None = None
    ambience: str | None = None


class Transition(_Strict):
    kind: str = "cut"
    carrier: str = ""
    meaning: str = ""


class Handoff(_Strict):
    element: str
    x: float
    y: float
    scale: float = 1.0
    opacity: float = 1.0
    direction: str = ""
    speed_px_s: float = 0.0
    blur_px: float = 0.0


class Shot(_Strict):
    id: str
    engine: Engine = "hyperframes"
    source: str | None = None
    text_mode: TextMode | None = None
    beats: float | None = None
    seconds: float | None = None
    visual: Visual = Visual()
    motion: Motion = Motion()
    reads: list[Read] = []
    events: list[Event] = []
    sound: Sound = Sound()
    transition_out: Transition | None = None
    handoff_in: Handoff | None = None
    handoff_out: Handoff | None = None


class Headline(_Strict):
    text: str
    key: str | None = None


class Chapter(_Strict):
    id: str
    purpose: str = ""
    headline: Headline | None = None
    start: Anchor | None = None
    bars: float | None = None
    vo: list[VOLine] = []
    shots: list[Shot] = Field(min_length=1)


class Canvas(_Strict):
    aspect: str = "16:9"
    width: int = 1920
    height: int = 1080
    fps: int = 60


class MusicStyle(_Strict):
    """Per-film overrides of the pack's music style lines; unset fields keep the pack's."""

    mood: str | None = None
    instrumentation: str | None = None
    arrangement: str | None = None
    negative: str | None = None


class MusicSpec(_Strict):
    bpm: float | None = None
    beats_per_bar: int = 4
    source: str | None = None
    # Who makes the bed with `promo music` when --provider is not given; unset is ElevenLabs.
    provider: Literal["elevenlabs", "lyria"] | None = None
    style: MusicStyle | None = None


class VoiceChoice(_Strict):
    """Which voice speaks. Unset fields fall back to the pack's voice config, then to the provider's defaults.

    `style` is Gemini's style prompt. `steps`, `speed` and `silence_s` are Supertonic's options.
    """

    provider: Literal["gemini", "supertonic"] | None = None
    name: str | None = None
    style: str | None = None
    model: str | None = None
    steps: int | None = Field(default=None, ge=1)
    speed: float | None = Field(default=None, ge=0.7, le=2.0)
    silence_s: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _options_match_provider(self) -> "VoiceChoice":
        if self.provider == "gemini" and any(v is not None for v in (self.steps, self.speed, self.silence_s)):
            raise ValueError("steps, speed and silence_s are Supertonic options; provider is 'gemini'")
        return self


class Shotlist(_Strict):
    version: Literal[2] = 2
    pack: str
    language: str = "en"
    canvas: Canvas = Canvas()
    timing_mode: Literal["music", "voice"] = "music"
    preroll_s: float = 0.3
    music: MusicSpec = MusicSpec()
    voice: VoiceChoice = VoiceChoice()
    pronounce: dict[str, str] = {}
    # Known ways speech recognition writes a word the script spells differently, when the audio is right:
    # {"Oleafly": ["oh leafly", "o'leafly"]}. Scoring then reads them as the script's word. A pack's
    # `voice.asr_aliases` supplies defaults; this wins per word.
    asr_aliases: dict[str, list[str]] = {}
    claims: list[Claim] = []
    chapters: list[Chapter] = Field(min_length=1)

    @field_validator("asr_aliases", mode="before")
    @classmethod
    def _aliases_are_words(cls, value: object) -> object:
        return check_aliases(value)

    def all_lines(self) -> list[VOLine]:
        return [line for ch in self.chapters for line in ch.vo]

    def all_shots(self) -> list[Shot]:
        return [shot for ch in self.chapters for shot in ch.shots]

    def chapter_of_shot(self, shot_id: str) -> Chapter:
        for ch in self.chapters:
            if any(s.id == shot_id for s in ch.shots):
                return ch
        raise KeyError(shot_id)

    def chapter_of_line(self, line_id: str) -> Chapter:
        for ch in self.chapters:
            if any(l.id == line_id for l in ch.vo):
                return ch
        raise KeyError(line_id)

    @model_validator(mode="after")
    def _cross_references(self) -> "Shotlist":
        seen: set[str] = set()

        def claim_id(kind: str, value: str) -> None:
            if value in seen:
                raise ValueError(f"duplicate id '{value}' ({kind})")
            seen.add(value)

        line_ids = {l.id for l in self.all_lines()}
        shot_ids = {s.id for s in self.all_shots()}
        claim_ids = {c.id for c in self.claims}

        def check_anchor(a: Anchor | None, where: str) -> None:
            if isinstance(a, WordAnchor) and a.line not in line_ids:
                raise ValueError(f"{where}: unknown line '{a.line}'")
            if isinstance(a, AfterAnchor) and a.after not in line_ids:
                raise ValueError(f"{where}: unknown line '{a.after}'")
            if isinstance(a, ShotAnchor) and a.shot not in shot_ids:
                raise ValueError(f"{where}: unknown shot '{a.shot}'")

        for ch in self.chapters:
            claim_id("chapter", ch.id)
            if self.timing_mode == "music" and ch.bars is None:
                raise ValueError(f"chapter {ch.id} needs 'bars' in music mode")
            if self.timing_mode == "music" and ch.start is not None and not isinstance(ch.start, BarAnchor):
                raise ValueError(f"chapter {ch.id}: in music mode 'start' must be a bar anchor")
            for line in ch.vo:
                claim_id("line", line.id)
                check_anchor(line.at, line.id)
                if isinstance(line.at, ShotAnchor):
                    raise ValueError(f"{line.id}: VO lines cannot anchor to a shot")
                for c in line.claims:
                    if c not in claim_ids:
                        raise ValueError(f"{line.id}: unknown claim '{c}'")
            for shot in ch.shots:
                claim_id("shot", shot.id)
                for ev in shot.events:
                    claim_id("event", ev.id)
                    check_anchor(ev.at, ev.id)
                for rd in shot.reads:
                    check_anchor(rd.from_, f"{shot.id} read")
                    check_anchor(rd.to, f"{shot.id} read")
        return self


class ShotlistError(ValueError):
    pass


def load_shotlist(path: Path) -> Shotlist:
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError as e:
        raise ShotlistError(f"{path}: not valid JSON ({e})") from e
    try:
        return Shotlist.model_validate(data)
    except ValidationError as e:
        lines = []
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"])
            lines.append(f"  {loc}: {err['msg']}")
        raise ShotlistError(f"{path}: invalid shot list\n" + "\n".join(lines)) from e


def script_text(text: str) -> str:
    """The words a listener hears: the line without its inline audio tags (`<short pause>`, `<breath>`).

    The tags stay in the text sent to TTS; ASR never transcribes them, so WER, pace and word alignment
    all run against this.
    """
    t = re.sub(r"\s*<[^>]*>\s*(?=[.,!?;:\u2026])", "", text)  # a tag before punctuation: keep the punctuation attached
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", t)).strip()
