"""Records shared across modules."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_serializer


class WordPart(BaseModel):
    """One hyphen-separated part of a script word ("read" in "read-only.") with the time ASR gave it."""

    text: str
    startMs: int
    endMs: int


class Caption(BaseModel):
    """One timed word (Remotion's Caption shape). Times are integer milliseconds.

    `parts` is set only on a hyphenated script word whose parts ASR timed ("ATS-friendly"), so a word anchor on
    "friendly" fires where it was spoken. A word without parts serialises exactly as before.
    """

    text: str
    startMs: int
    endMs: int
    timestampMs: int | None = None
    confidence: float | None = None
    parts: list[WordPart] = []

    @model_serializer(mode="wrap")
    def _omit_empty_parts(self, handler):
        data = handler(self)
        if not data.get("parts"):
            data.pop("parts", None)
        return data


class BeatGrid(BaseModel):
    """Measured tempo grid of a music file. Times are seconds from the file start."""

    bpm: float
    beats: list[float]
    downbeats: list[float]
    onsets: list[float] = []
    first_downbeat_s: float
    beats_per_bar: int = 4


class LineTake(BaseModel):
    """The accepted TTS take for one VO line. Word times are ms from the take start.

    `style` is Gemini's style prompt; `params` holds the numeric options of a provider with no style prompt
    (Supertonic: steps, speed, silence_s). `verified` is False for a line authored with `verify: false`, whose
    word times are a proportional split and whose `wer` is 0.0 because no ASR ran. `words` are the script's own
    words (spelling and punctuation as written) on the ASR clock, one per whitespace-separated script word. `problem`
    is a reason the line cannot be used as it stands (every take silent); it is empty for a usable take.
    """

    id: str
    file: str
    duration_s: float
    words: list[Caption] = []
    wer: float = 0.0
    takes: int = 1
    provider: str = "gemini"
    voice: str = ""
    model: str = ""
    style: str = ""
    params: dict[str, int | float] = Field(default_factory=dict)
    verified: bool = True
    problem: str = ""


class Finding(BaseModel):
    level: Literal["error", "warning"]
    code: str
    message: str
    ref: str | None = None
