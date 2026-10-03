"""The SFX stem: one sound per event, each sound's peak placed on its event time."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from pydantic import BaseModel

from promo.audio_analysis import decode_mono
from promo.packs import Pack
from promo.proc import ToolError, atomic_path, sha256_hex
from promo.providers import cache, costs, elevenlabs
from promo.providers.http import ProviderError
from promo.sfx_synth import SR, SfxError, synth, write_wav
from promo.timing import Timing

DEFAULT_GAIN_DB = -10
DEFAULT_GEN_S = 0.5

__all__ = ["Placement", "SfxError", "event_clip", "gen_key", "generate", "local_peak_s", "render_stem"]


class Placement(BaseModel):
    event: str
    sfx: str
    t: float
    start_s: float


def local_peak_s(samples: np.ndarray, sr: int, t: float, before: float = 0.15, after: float = 0.15) -> float | None:
    """The time of the loudest sample within `before` s ahead of `t` and `after` s behind it, or None when that window
    holds no samples (`t` is past the end of the audio). A caller with a neighbouring sound inside the window narrows
    it: the loudest sample there may be the neighbour's."""
    a = max(0, int((t - before) * sr))
    b = min(len(samples), int((t + after) * sr))
    if b <= a:
        return None
    return (a + int(np.argmax(np.abs(samples[a:b])))) / sr


def _decode(path: Path, what: str) -> np.ndarray:
    try:
        sound = decode_mono(path, SR)
    except ToolError as e:  # keep ffmpeg's own last line, not the whole command and its stderr tail
        raise SfxError(f"{what}: ffmpeg could not decode it ({str(e).strip().splitlines()[-1]})") from e
    if sound.size == 0:
        raise SfxError(f"{what}: it holds no audio")
    return sound


def _decode_bytes(data: bytes, what: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sfx.mp3"
        p.write_bytes(data)
        return _decode(p, what)


def gen_key(prompt: str, duration: float) -> str:
    """The cache key of one generated sound. It carries the provider's output format and prompt influence, so
    changing either default can never serve audio made with the old one."""
    return sha256_hex("elevenlabs-sfx", elevenlabs.SFX_OUTPUT_FORMAT, str(elevenlabs.SFX_PROMPT_INFLUENCE),
                      prompt, str(float(duration)))


def generate(prompt: str, duration: float, *, gen: Callable[[str, float], bytes] = elevenlabs.sound_generation,
             costs_path: Path | None = None) -> bytes:
    """One ElevenLabs sound as MP3 bytes, cached by prompt and length so a rerun never pays twice.

    The paid call is logged as soon as the audio is in hand, before anything can fail on it.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise SfxError("the sound prompt is empty")
    if not (isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0):
        raise SfxError(f"duration must be a number of seconds above 0 (got {duration!r})")

    def fetch() -> bytes:
        audio = gen(prompt, duration)
        if not audio:  # raised before the cache write, so an empty body is never kept
            raise ProviderError("elevenlabs", 200, f"returned an empty sound for '{prompt}'")
        return audio

    audio, hit = cache.cached_call("elevenlabs", gen_key(prompt, duration), fetch)
    # usd stays 0: the spec gives no price for sound generation (ElevenLabs bills it in plan credits), so none is
    # invented. `unpriced` says so in the row, and the units record what was asked for.
    costs.log_cost(costs_path, provider="elevenlabs", model="sound-generation",
                   units={"sfx_s": duration, "unpriced": True}, usd=0.0, cached=hit)
    return audio


def _seed(event_id: str) -> int:
    """A noise seed from the event's id: the same event always sounds the same, whatever else is in the film."""
    return int(sha256_hex("sfx-seed", event_id)[:16], 16)


def _num(spec: dict, key: str, default: float | None = None) -> float | None:
    value = spec.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SfxError(f"{key} must be a number (got {value!r})")
    return float(value)


def _load(spec: dict, pack: Pack, gen: Callable, costs_path: Path | None) -> np.ndarray:
    """A sound that does not depend on the event: a sample file or a generated sound."""
    if "file" in spec:
        name = spec["file"]
        if not isinstance(name, str) or not name:
            raise SfxError(f"file must be a path relative to the pack folder (got {name!r})")
        path = pack.root / name
        if not path.is_file():
            raise SfxError(f"file not found: {path}")
        return _decode(path, str(path))
    if "gen" in spec:
        prompt, duration = spec["gen"], _num(spec, "duration", DEFAULT_GEN_S)
        audio = generate(prompt, duration, gen=gen, costs_path=costs_path)
        try:
            return _decode_bytes(audio, f"generated sound '{prompt}'")
        except SfxError as e:  # the cache must not keep audio that cannot be used: a rerun would fail the same way
            entry = cache.cache_path("elevenlabs", gen_key(prompt, duration))
            try:
                entry.unlink(missing_ok=True)
            except OSError as oe:
                raise SfxError(f"{e}; the cached copy could not be removed ({oe}): delete {entry} by hand") from e
            raise SfxError(f"{e}; the cached copy was discarded, so a rerun will regenerate it") from e
    raise SfxError(f"an entry needs synth, file or gen (got {spec})")


def _sound(spec: dict, pack: Pack, seed: int, value: float | None, gen: Callable, costs_path: Path | None,
           memo: dict[str, np.ndarray]) -> np.ndarray:
    if "synth" in spec:
        ranges = {k: spec[k] for k in ("value_range", "pitch_range") if k in spec}
        return synth(spec["synth"], seed=seed, duration=_num(spec, "duration"), pitch=_num(spec, "pitch"),
                     value=value, **ranges)
    key = json.dumps(spec, sort_keys=True, default=str)
    if key not in memo:  # an entry used by many events is fetched, decoded and logged once per render
        memo[key] = _load(spec, pack, gen, costs_path)
    return memo[key]


def _gain(pack: Pack) -> float:
    gain_db = pack.sfx.get("sfx_gain_db", DEFAULT_GAIN_DB)
    if isinstance(gain_db, bool) or not isinstance(gain_db, (int, float)) or not math.isfinite(gain_db):
        raise SfxError(f"pack {pack.name}: sfx_gain_db must be a number of dB (got {gain_db!r})")
    return 10 ** (gain_db / 20)


def _placed(sound: np.ndarray, t: float) -> tuple[np.ndarray, int]:
    """(samples, first sample) that put the sound's loudest sample on `t`; a sound whose peak is later than `t` allows
    is cut at the start of the stem, not shifted."""
    start = round(t * SR) - int(np.argmax(np.abs(sound)))
    return (sound[-start:], 0) if start < 0 else (sound, start)


def event_clip(ev, pack: Pack, *, gen: Callable = elevenlabs.sound_generation, costs_path: Path | None = None,
               memo: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, int] | None:
    """The sound `render_stem` mixes in for a timed event, as it sits in the stem: (samples with the pack's gain applied,
    the sample it starts on). None when the pack has no sound for the event. Raises SfxError when the sound cannot be
    made. The same sound as the stem's, because the synths are seeded by the event id and a generated sound comes from
    the cache; pass a `gen` that raises to make sure nothing is ever paid for."""
    name = ev.sfx or ev.kind
    spec = pack.sfx.get(name) if isinstance(pack.sfx.get(name), dict) else None
    if spec is None:
        return None
    try:
        sound = _sound(spec, pack, seed=_seed(ev.id), value=ev.value, gen=gen, costs_path=costs_path,
                       memo=memo if memo is not None else {})
    except SfxError as e:
        raise SfxError(f"{ev.id}: sfx '{name}' in pack {pack.name}: {e}") from e
    sound, start = _placed(sound, ev.t)
    return sound.astype(np.float64) * _gain(pack), start


def render_stem(timing: Timing, pack: Pack, out: Path, *, gen: Callable = elevenlabs.sound_generation,
                costs_path: Path | None = None) -> tuple[Path, list[Placement], list[str]]:
    """Mix one sound per event into a stem exactly `timing.duration_s` long, each sound's peak on its event."""
    gain = _gain(pack)
    n = round(timing.duration_s * SR)
    buf = np.zeros(n, dtype=np.float64)
    placements: list[Placement] = []
    warnings: list[str] = []
    memo: dict[str, np.ndarray] = {}
    for ev in timing.events:
        name = ev.sfx or ev.kind
        spec = pack.sfx.get(name) if isinstance(pack.sfx.get(name), dict) else None
        if spec is None:
            if ev.sfx:
                warnings.append(f"{ev.id}: no sfx '{name}' in pack {pack.name}; skipped")
            continue
        try:
            sound = _sound(spec, pack, seed=_seed(ev.id), value=ev.value, gen=gen, costs_path=costs_path, memo=memo)
        except SfxError as e:
            raise SfxError(f"{ev.id}: sfx '{name}' in pack {pack.name}: {e}") from e
        sound, start = _placed(sound, ev.t)
        end = min(n, start + len(sound))
        if end > start:
            buf[start:end] += sound[: end - start] * gain
        placements.append(Placement(event=ev.id, sfx=name, t=ev.t, start_s=round(start / SR, 6)))
    top = float(np.max(np.abs(buf))) if n else 0.0
    if top > 0.99:
        buf *= 0.99 / top
        warnings.append(f"SFX stem peaked at {top:.2f}; scaled down to avoid clipping")
    write_wav(buf.astype(np.float32), out)
    with atomic_path(out.with_name("placements.json")) as tmp:
        tmp.write_text(json.dumps([p.model_dump() for p in placements], indent=2))
    return out, placements, warnings
