"""Voice-over takes: synthesise, trim, normalise, verify by ASR, retake, keep word timings."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import soundfile as sf

from promo import asr as _asr
from promo.models import Caption, LineTake
from promo.packs import Pack, PackError
from promo.proc import atomic_path, sha256_hex
from promo.proc import run as run_cmd
from promo.providers import cache, costs, gemini, supertonic
from promo.run import RunPaths
from promo.shotlist import Shotlist, VOLine, check_aliases, script_text

RETAKE_STYLES = ("", "clear diction", "slightly slower, clear diction")
# Supertonic has no style prompt, so a retake nudges the sampler instead: (steps added, speed factor) per take.
# Take 2 denoises 4 steps more and speaks 3% slower; take 3 denoises 8 more and speaks 3% faster.
SUPERTONIC_RETAKES = ((0, 1.0), (4, 0.97), (8, 1.03))
PROVIDERS = ("gemini", "supertonic")
Params = Mapping[str, int | float]
Aliases = Mapping[str, Sequence[str]]
MAX_WER = 0.1
CLIP_LEVEL = 0.999  # a sample this close to full scale counts as clipped...
CLIP_RUN = 3        # ...when at least this many in a row do; one stray peak is not clipping
MIN_TAKE_S = 0.1    # a take shorter than this once trimmed is silence: ASR would catch it, but not on `verify: false`
SILENT = "every take came back silent"


def spoken_text(text: str, pronounce: dict[str, str]) -> str:
    out = text
    for word, say in pronounce.items():
        out = re.sub(rf"(?<!\w){re.escape(word)}(?!\w)", lambda _m, say=say: say, out)
    return out


def trim_and_normalize(src: Path, dst: Path, lufs: float = -16.0, pad_s: float = 0.06) -> Path:
    trim = f"silenceremove=start_periods=1:start_threshold=-50dB:start_silence={pad_s}"
    af = f"{trim},areverse,{trim},areverse,loudnorm=I={lufs}:TP=-1.5:LRA=11,aresample=48000"
    with atomic_path(dst) as tmp:
        run_cmd(["ffmpeg", "-v", "error", "-y", "-i", src, "-af", af, "-ac", "1", "-ar", "48000",
                 "-c:a", "pcm_s24le", tmp])
    return dst


def _style(*parts: str) -> str:
    return "; ".join(p for p in parts if p)


_script_text = script_text   # the words a listener hears (promo.shotlist.script_text)


def score_take(path: Path, line_text: str, aliases: Aliases, language: str, duration_ms: float,
               transcribe: Callable = _asr.transcribe) -> tuple[float, list[Caption]]:
    """Speech recognition of one take (or of a stretch of the final mix cut to the take's length) against its line:
    (WER, the script's words timed on the recognition's clock).

    The one scoring path. The known mishearings in `aliases` are read as the script's word first, then the WER is taken
    on the normalised words (numbers, currency, US and UK spelling and inline audio tags do not count against the
    line), then each script word is given the time its recognised words start. `synthesize_line` scores a fresh take
    with it, `_rescore` a stored one, and the critic each line's window of the mix, so all three agree.
    """
    script = _script_text(line_text)
    heard = _asr.apply_aliases(transcribe(path, language), aliases or {}, script, language)
    score = _asr.wer(script, " ".join(w.text for w in heard), language)
    return score, _asr.align_to_script(script.split(), heard, duration_ms, language)


def _clipped(path: Path) -> bool:
    """True when the audio clips: CLIP_RUN or more consecutive samples at or above CLIP_LEVEL.

    Run it on the raw TTS output. The normalised take never clips: loudnorm's -1.5 dBTP ceiling caps the peak.
    """
    data, _ = sf.read(str(path), dtype="float32", always_2d=True)
    hot = (np.abs(data) >= CLIP_LEVEL).any(axis=1)
    if hot.sum() < CLIP_RUN:
        return False
    return bool((np.convolve(hot.astype(int), np.ones(CLIP_RUN, dtype=int), mode="valid") >= CLIP_RUN).any())


@dataclass(frozen=True)
class VoiceSettings:
    """Who speaks and how, once the shot list, the pack and the provider's defaults have all been consulted."""

    provider: str
    voice: str
    model: str
    style: str = ""                                  # Gemini only
    options: Params = field(default_factory=dict)    # Supertonic only: steps, speed, silence_s
    pace_wpm: float | None = None                    # what the pace check expects of this voice
    aliases: Aliases = field(default_factory=dict)   # what ASR mishears, from the pack and then the shot list


def voice_settings(sl: Shotlist, pack: Pack) -> VoiceSettings:
    """The shot list's `voice` block, then the pack's voice config, then the provider's defaults."""
    choice, cfg = sl.voice, pack.voice
    try:
        pack_aliases = check_aliases(cfg.get("asr_aliases", {}), f"pack {pack.name}: voice.asr_aliases")
    except ValueError as e:
        raise PackError(str(e)) from e
    aliases = {**pack_aliases, **sl.asr_aliases}
    provider = choice.provider or cfg.get("provider", "gemini")
    if provider == "gemini":
        return VoiceSettings("gemini", voice=choice.name or cfg.get("voice", "Kore"),
                             model=choice.model or cfg.get("model", gemini.DEFAULT_MODEL),
                             style=choice.style or cfg.get("style", ""), pace_wpm=cfg.get("pace_wpm"), aliases=aliases)
    if provider == "supertonic":
        own = cfg.get("supertonic", {})
        options = {}
        for key, default in supertonic.DEFAULT_OPTIONS.items():
            chosen = getattr(choice, key)
            options[key] = chosen if chosen is not None else own.get(key, default)
        return VoiceSettings("supertonic", voice=choice.name or own.get("voice", supertonic.DEFAULT_VOICE),
                             model=choice.model or own.get("model", supertonic.DEFAULT_MODEL), options=options,
                             pace_wpm=own.get("pace_wpm", cfg.get("pace_wpm")), aliases=aliases)
    raise PackError(f"pack {pack.name}: voice.provider '{provider}' is not one of {', '.join(PROVIDERS)}")


def audition_voices(settings: VoiceSettings, pack: Pack) -> list[str]:
    """The voices `promo voice audition` plays when none are named: the chosen provider's voice, then its fallbacks."""
    cfg = pack.voice.get("supertonic", {}) if settings.provider == "supertonic" else pack.voice
    default = supertonic.DEFAULT_VOICE if settings.provider == "supertonic" else "Kore"
    return [cfg.get("voice", default), *cfg.get("fallback_voices", [])]


def _tts_text(provider: str, text: str, pronounce: dict[str, str]) -> str:
    """What the provider is asked to say. Supertonic reads inline audio tags aloud ("short pause"), so they go."""
    spoken = spoken_text(text, pronounce)
    return _script_text(spoken) if provider == "supertonic" else spoken


def _variants(provider: str, base_style: str, line_style: str,
              options: Params) -> list[tuple[str, dict[str, int | float]]]:
    """Take 1, 2, 3 as (style, params). Gemini retakes change the style prompt; Supertonic's change the sampler."""
    if provider == "supertonic":
        o = {**supertonic.DEFAULT_OPTIONS, **options}
        return [("", {"steps": int(o["steps"]) + more,
                      "speed": min(2.0, max(0.7, round(float(o["speed"]) * factor, 4))),
                      "silence_s": o["silence_s"]})
                for more, factor in SUPERTONIC_RETAKES]
    return [(_style(base_style, line_style, extra), {}) for extra in RETAKE_STYLES]


def _generate(provider: str, text: str, *, voice: str, model: str, style: str, language: str,
              params: Params, synth: Callable | None) -> tuple[bytes, bool]:
    """One synthesis through the provider's content-hash cache: (WAV bytes, was it a cache hit)."""
    if provider == "supertonic":
        opts = dict(voice=voice, model=model, lang=language, **params)
        key = supertonic.cache_key(text, **opts)
        return cache.cached_call("supertonic", key, lambda: (synth or supertonic.synthesize)(text, **opts))
    key = sha256_hex("gemini-tts", model, voice, style, text, language)
    return cache.cached_call("gemini", key, lambda: (synth or gemini.synthesize)(text, voice=voice, style=style,
                                                                                 model=model))


def _log_cost(provider: str, costs_path: Path | None, *, model: str, text: str, audio_s: float, cached: bool) -> None:
    if provider == "supertonic":  # on-device: free, which is a price of 0, not an unpriced call
        costs.log_cost(costs_path, provider="supertonic", model=model, units={"chars": len(text), "local": True},
                       usd=0.0, cached=cached)
        return
    costs.log_cost(costs_path, provider="gemini", model=model,
                   units={"audio_s": round(audio_s, 3), "chars": len(text)},
                   usd=gemini.estimate_usd(model, text, audio_s), cached=cached)


def synthesize_line(
    line: VOLine, *, voice: str, base_style: str, model: str, pronounce: dict[str, str], language: str,
    vo_dir: Path, costs_path: Path | None = None, synth: Callable | None = None,
    transcribe: Callable = _asr.transcribe, pace_wpm: float | None = None, max_takes: int = 3,
    provider: str = "gemini", options: Params | None = None, asr_aliases: Aliases | None = None,
) -> LineTake:
    """Up to `max_takes` takes of one line; the first that passes ASR, pace and clip checks, else the best.

    `synth` replaces the provider's own synthesize function (Gemini: `(text, *, voice, style, model)`;
    Supertonic: `(text, *, voice, model, lang, steps, speed, silence_s)`). A line with `verify: false` skips ASR and
    the WER retake, and gets word times from a proportional split of its duration. `asr_aliases` turn known ASR
    mishearings of the line's own words back into the script's word before scoring. The take's words are always the
    script's own, aligned to what ASR heard. A silent take, or a variant whose draws were all silent, falls through to
    the next variant; a line where every one fails comes back with `problem` set instead of raising.
    """
    spoken = _tts_text(provider, line.text, pronounce)
    takes_dir = vo_dir / ".takes"
    takes_dir.mkdir(parents=True, exist_ok=True)
    script = _script_text(line.text)
    variants = _variants(provider, base_style, line.style, options or {})
    best: tuple[LineTake, Path] | None = None
    best_rank: tuple[bool, bool, float] = (True, True, 1.0)
    attempt = 0
    for attempt, (style, params) in enumerate(variants[:max_takes], 1):
        try:
            audio, hit = _generate(provider, spoken, voice=voice, model=model, style=style, language=language,
                                   params=params, synth=synth)
        except supertonic.SupertonicSilence:
            continue                                  # this variant drew nothing but silence: try the next one
        raw = takes_dir / f".{line.id}-{attempt}.raw.wav"
        raw.write_bytes(audio)
        try:
            take_path = trim_and_normalize(raw, takes_dir / f"{line.id}-{attempt}.wav")
            clipped = _clipped(raw)
        finally:
            raw.unlink(missing_ok=True)
        dur = sf.info(str(take_path)).duration
        _log_cost(provider, costs_path, model=model, text=spoken, audio_s=dur, cached=hit)
        silent = dur < MIN_TAKE_S
        if silent:
            words, score = [], 1.0                    # nothing to transcribe
        elif line.verify:
            score, words = score_take(take_path, line.text, asr_aliases or {}, language, dur * 1000, transcribe)
        else:
            words, score = _asr.spread_words(script.split(), 0, dur * 1000), 0.0
        take = LineTake(id=line.id, file=f"audio/vo/{line.id}.wav", duration_s=round(dur, 4), words=words,
                        wer=round(score, 4), takes=attempt, provider=provider, voice=voice, model=model,
                        style=style, params=params, verified=line.verify)
        pace_ok = True
        if pace_wpm:
            expected = max(1, len(script.split())) / pace_wpm * 60
            pace_ok = abs(dur - expected) <= 0.3 * expected + 0.25
        accepted = score <= MAX_WER and pace_ok and not clipped and not silent
        rank = (silent, not accepted, take.wer)  # accepted beats rejected, any sound beats silence; ties keep the earlier
        if best is None or rank < best_rank:
            best, best_rank = (take, take_path), rank
        if accepted:
            break
    if attempt == 0:
        raise ValueError("max_takes must be at least 1")
    wav = vo_dir / f"{line.id}.wav"
    for n in range(attempt + 1, len(variants) + 1):  # takes left over from an earlier, longer attempt
        (takes_dir / f"{line.id}-{n}.wav").unlink(missing_ok=True)
    if best is None:                                  # no variant produced any audio at all
        wav.unlink(missing_ok=True)
        return LineTake(id=line.id, file=f"audio/vo/{line.id}.wav", duration_s=0.0, wer=1.0, takes=attempt,
                        provider=provider, voice=voice, model=model, params=variants[attempt - 1][1],
                        verified=line.verify, problem=SILENT)
    chosen, path = best
    with atomic_path(wav) as tmp:
        shutil.copyfile(path, tmp)
    return chosen.model_copy(update={"takes": attempt, "problem": SILENT if chosen.duration_s < MIN_TAKE_S else ""})


def _fingerprint(line: VOLine, language: str, pronounce: dict[str, str], aliases: Aliases) -> str:
    """What the take was made from besides voice, style and model; the sidecar next to its record.

    The aliases that apply to this line are part of it, because they change its recorded WER and word times. A line
    no alias applies to keeps the fingerprint it always had.
    """
    text = f"{line.text}\n{language}\n{spoken_text(line.text, pronounce)}"
    active = _asr.active_aliases(_script_text(line.text), aliases, language) if line.verify else {}
    return f"{text}\n{json.dumps(active, sort_keys=True, ensure_ascii=False)}" if active else text


def _unchanged(record: Path, line: VOLine, *, settings: VoiceSettings, language: str,
               pronounce: dict[str, str]) -> LineTake | None:
    """The stored take, if it was made from exactly these inputs (any retake variant counts)."""
    meta = record.with_name(f"{line.id}.src.txt")
    if not (record.exists() and record.with_name(f"{line.id}.wav").exists() and meta.exists()):
        return None
    if meta.read_text() != _fingerprint(line, language, pronounce, settings.aliases):
        return None
    try:
        prev = LineTake.model_validate_json(record.read_text())
    except ValueError:
        return None
    if prev.problem:
        return None                                   # unusable: try it again
    if (prev.provider, prev.voice, prev.model, prev.verified) != (settings.provider, settings.voice, settings.model,
                                                                  line.verify):
        return None
    variants = _variants(settings.provider, settings.style, line.style, settings.options)
    return prev if (prev.style, prev.params) in variants else None


def _has_script_words(take: LineTake, line: VOLine) -> bool:
    return [w.text for w in take.words] == _script_text(line.text).split()


def _rescore(prev: LineTake, line: VOLine, *, vo_dir: Path, language: str, aliases: Aliases,
             transcribe: Callable) -> LineTake:
    """The stored take with its words and WER worked out again from the take's own WAV (audio/vo/<id>.wav)."""
    wav = vo_dir / f"{line.id}.wav"
    duration_ms = sf.info(str(wav)).duration * 1000
    script = _script_text(line.text)
    if not line.verify:
        return prev.model_copy(update={"words": _asr.spread_words(script.split(), 0, duration_ms)})
    score, words = score_take(wav, line.text, aliases, language, duration_ms, transcribe)
    return prev.model_copy(update={"words": words, "wer": round(score, 4)})


def _store(record: Path, line: VOLine, take: LineTake, sl: Shotlist, settings: VoiceSettings) -> None:
    with atomic_path(record) as tmp:
        tmp.write_text(take.model_dump_json(indent=2))
    with atomic_path(record.with_name(f"{line.id}.src.txt")) as tmp:
        tmp.write_text(_fingerprint(line, sl.language, sl.pronounce, settings.aliases))


def synthesize_all(
    run: RunPaths, sl: Shotlist, pack: Pack, only: list[str] | None = None, force: bool = False,
    synth: Callable | None = None, transcribe: Callable = _asr.transcribe,
) -> list[LineTake]:
    settings = voice_settings(sl, pack)
    run.vo_dir.mkdir(parents=True, exist_ok=True)
    out: list[LineTake] = []
    for line in sl.all_lines():
        if only and line.id not in only:
            continue
        record = run.vo_dir / f"{line.id}.json"
        if not force:
            prev = _unchanged(record, line, settings=settings, language=sl.language, pronounce=sl.pronounce)
            if prev is not None and not _has_script_words(prev, line):
                # written before a take's words were kept in the script's spelling: score the take that is on disk
                # again. It never re-synthesises, so a missing cache costs nothing (Gemini) and cannot change the voice.
                prev = _rescore(prev, line, vo_dir=run.vo_dir, language=sl.language, aliases=settings.aliases,
                                transcribe=transcribe)
                _store(record, line, prev, sl, settings)
            if prev is not None:
                out.append(prev)
                continue
        take = synthesize_line(line, voice=settings.voice, base_style=settings.style, model=settings.model,
                               pronounce=sl.pronounce, language=sl.language, vo_dir=run.vo_dir, costs_path=run.costs,
                               synth=synth, transcribe=transcribe, pace_wpm=settings.pace_wpm,
                               provider=settings.provider, options=settings.options, asr_aliases=settings.aliases)
        _store(record, line, take, sl, settings)
        out.append(take)
    return out


def audition(text: str, voices: list[str], style: str, out_dir: Path, model: str | None = None,
             synth: Callable | None = None, costs_path: Path | None = None,
             settings: VoiceSettings | None = None, language: str = "en") -> list[Path]:
    """One WAV per voice. Gemini auditions on the lite model; with Supertonic `settings` they use its options."""
    provider = settings.provider if settings else "gemini"
    if provider == "supertonic":
        model = model or settings.model
        params = {**supertonic.DEFAULT_OPTIONS, **settings.options}
        spoken, key_language = _tts_text(provider, text, {}), language
    else:
        model = model or gemini.LITE_MODEL
        params = {}
        spoken, key_language = text, "audition"  # keeps auditions cached before Supertonic existed valid
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for v in voices:
        audio, hit = _generate(provider, spoken, voice=v, model=model, style=style, language=key_language,
                               params=params, synth=synth)
        raw = out_dir / f".{v}.raw.wav"
        raw.write_bytes(audio)
        try:
            paths.append(trim_and_normalize(raw, out_dir / f"{v}.wav"))
        finally:
            raw.unlink(missing_ok=True)
        dur = sf.info(str(paths[-1])).duration
        _log_cost(provider, costs_path, model=model, text=spoken, audio_s=dur, cached=hit)
    return paths
