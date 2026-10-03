"""promo doctor: is this machine ready to make a video?

Keys are reported as present or missing and never shown. Nothing here needs a key unless `--live` is passed, and
a live check only ever calls a provider's free key-validation endpoint.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from promo.config import load_env
from promo.proc import ToolError, run

FFMPEG_FILTERS = ("loudnorm", "sidechaincompress", "alimiter", "ebur128", "blackdetect", "freezedetect", "scdet",
                  "silencedetect", "showspectrumpic")
# The harness never uses these (all text is rendered by HyperFrames), so ffmpeg lacking them is expected.
TEXT_FILTERS = ("drawtext", "subtitles", "ass")
PY_MODULES = ("librosa", "opentimelineio", "parakeet_mlx", "mlx_whisper", "soundfile", "num2words")
SUPERTONIC_INSTALL = "pip install supertonic or uv tool install supertonic"
HF_TIMEOUT_S = 300
# Checks in `hyperframes doctor --json` (0.8.98) that this harness cannot render without, and the ones it never uses.
HF_BLOCKING = ("Node.js", "Memory", "Disk", "Frames cache", "Archive extractor", "Settings lock", "FFmpeg", "FFprobe",
               "Chrome")
HF_NOT_NEEDED = ("Version", "CPU", "Environment", "whisper-cpp", "TTS (Kokoro)", "BGM (MusicGen)", "onnxruntime-node",
                 "@google/genai", "Docker", "Docker running")


class Check(BaseModel):
    name: str
    ok: bool
    level: Literal["error", "warning", "info"] = "error"
    detail: str = ""


class KeyNeeds(BaseModel):
    """What the current run uses each key for; an empty list means the run does not need the key."""

    gemini: list[str] = []
    elevenlabs: list[str] = []


def check_keys(env: dict, needs: KeyNeeds | None = None) -> list[Check]:
    """Presence only: the value of a key is never read into a message.

    Outside a run (`needs` None) a missing Gemini or ElevenLabs key is a warning: what needs it depends on the film.
    Inside a run a missing key is an error only when that run needs it (`needs`), and a warning otherwise.
    """
    out = []
    for name, uses in (("GEMINI_API_KEY", needs.gemini if needs else None),
                       ("ELEVENLABS_API_KEY", needs.elevenlabs if needs else None), ("OPENROUTER_API_KEY", None)):
        present = bool(env.get(name, "").strip())
        if present:
            out.append(Check(name=name, ok=True, detail="set"))
        elif uses:
            out.append(Check(name=name, ok=False, level="error",
                             detail=f"missing, and this run needs it for {' and '.join(uses)}: add it to the harness .env"))
        else:
            why = "this run does not need it" if needs is not None and name != "OPENROUTER_API_KEY" else "optional"
            out.append(Check(name=name, ok=False, level="warning", detail=f"missing ({why}): add it to the harness .env"))
    return out


def check_node(version: str) -> Check:
    m = re.match(r"v?(\d+)", version.strip())
    major = int(m.group(1)) if m else 0
    return Check(name="node", ok=major >= 22, detail=f"{version.strip()} (need 22+)")


def check_ffmpeg(filters_text: str) -> list[Check]:
    have = set(re.findall(r"^\s*\S+\s+(\w+)\s", filters_text, flags=re.M))
    out = [Check(name=f"ffmpeg filter {f}", ok=f in have, detail="present" if f in have else "missing")
           for f in FFMPEG_FILTERS]
    text = [f for f in TEXT_FILTERS if f in have]
    out.append(Check(name="ffmpeg text filters", ok=True, level="info",
                     detail=f"{', '.join(text)} present but never used (text is rendered by HyperFrames)" if text
                     else f"no {'/'.join(TEXT_FILTERS)}, as expected: the harness never uses them"))
    return out


def check_disk(free_bytes: int) -> Check:
    gb = free_bytes / 1e9
    if gb >= 20:
        return Check(name="disk", ok=True, detail=f"{gb:.0f} GB free")
    level = "warning" if gb >= 8 else "error"
    return Check(name="disk", ok=False, level=level,
                 detail=f"{gb:.0f} GB free; per-shot renders need about 3 GB each plus a 5 GB reserve")


def check_module(name: str) -> Check:
    try:
        importlib.import_module(name)
    except Exception as e:      # ImportError, or a native library that fails to load
        why = (str(e).splitlines() or [type(e).__name__])[0]
        return Check(name=f"python {name}", ok=False, detail=f"cannot import ({why}): run `uv sync`")
    return Check(name=f"python {name}", ok=True, detail="imports")


def check_supertonic(version: str | None, *, required: bool) -> Check:
    """The local voice. Optional, unless the run's shot list picks it as its voice provider."""
    if version:
        return Check(name="supertonic", ok=True, detail=version)
    level = "error" if required else "warning"
    why = "this run's shot list uses it as the voice provider" if required else "optional: the local voice"
    return Check(name="supertonic", ok=False, level=level, detail=f"not on PATH ({why}): {SUPERTONIC_INSTALL}")


def check_supertonic_model(path: Path) -> Check:
    if path.is_dir():
        return Check(name="supertonic model", ok=True, level="info", detail=str(path))
    return Check(name="supertonic model", ok=False, level="info",
                 detail=f"{path} does not exist yet; the first Supertonic voice run downloads the model")


def check_asr_model(label: str, repo: str, snapshot: Path | None) -> Check:
    """A speech-recognition model in the Hugging Face cache (its weights are there). A missing one downloads on first use."""
    weights = [p for p in (*snapshot.glob("*.safetensors"), *snapshot.glob("*.npz")) if p.exists()] if snapshot else []
    if weights:
        return Check(name=f"asr {label} model", ok=True, detail=repo)
    return Check(name=f"asr {label} model", ok=False, level="warning",
                 detail=f"{repo} is not downloaded; the first run that needs it fetches it (network)")


def check_hyperframes(payload: dict) -> Check:
    """Gate on the `doctor --json` payload (the command itself always exits 0; its top-level `.ok` is false on a
    healthy Mac that lacks an optional extra such as Docker, Kokoro or MusicGen, so it is read through the checks).

    A failed check in HF_BLOCKING fails this check, whatever its detail says. A failed check the harness knows it
    does not need (HF_NOT_NEEDED) is only noted. Any other failed check is a warning: a name this pin of HyperFrames
    did not have when the lists were written. The version pin is checked separately, against `_meta.version`.
    """
    from promo.hyperframes import HF_VERSION as pin

    if "error" in payload:
        return Check(name="hyperframes", ok=False, detail=str(payload["error"]))
    version = (payload.get("_meta") or {}).get("version")
    if version is not None and version != pin:
        return Check(name="hyperframes", ok=False, detail=f"runs {version}, but the harness pins {pin}")
    failed = [c for c in payload.get("checks") or [] if not c.get("ok")]
    blocking = [c for c in failed if c.get("name") in HF_BLOCKING]
    unknown = [c for c in failed if c.get("name") not in HF_BLOCKING + HF_NOT_NEEDED]
    unneeded = [c for c in failed if c.get("name") in HF_NOT_NEEDED]
    if blocking:
        return Check(name="hyperframes", ok=False,
                     detail="; ".join(f"{c['name']}: {c.get('detail', '')}" for c in blocking)[:300])
    if not payload.get("ok") and not failed:
        return Check(name="hyperframes", ok=False, detail="doctor reported not ok")
    if unknown:
        return Check(name="hyperframes", ok=False, level="warning", detail=(
            "a check this harness does not know failed: "
            + "; ".join(f"{c.get('name')}: {c.get('detail', '')}" for c in unknown))[:300])
    note = f"; not needed here: {', '.join(c['name'] for c in unneeded)}" if unneeded else ""
    return Check(name="hyperframes", ok=True, detail=f"{pin} doctor ok{note}")


def _hf_doctor() -> dict:
    from promo.hyperframes import hf_cmd, hf_env

    try:
        out = run(hf_cmd("doctor", "--json"), env=hf_env(), check=False, timeout=HF_TIMEOUT_S).stdout
    except FileNotFoundError:
        return {"error": "needs Node.js (npx not found)"}
    except subprocess.TimeoutExpired:
        return {"error": f"hyperframes doctor did not finish within {HF_TIMEOUT_S} s"}
    start = out.find("{")
    if start < 0:
        return {"error": "hyperframes doctor printed no JSON"}
    try:
        return json.loads(out[start:])
    except json.JSONDecodeError:
        return {"error": "hyperframes doctor printed invalid JSON"}


def _redact(text: str, secret: str) -> str:
    """`text` without the secret in any form an exception message can carry it in: as it is, as its repr (a quoted
    string or bytes literal, with escapes such as \\n), ASCII-escaped, or as JSON."""
    if not secret:
        return text
    forms = {secret, repr(secret)[1:-1], ascii(secret)[1:-1], repr(secret.encode("utf-8"))[2:-1], json.dumps(secret)[1:-1],
             secret.encode("unicode_escape").decode("ascii")}
    for form in sorted(forms, key=len, reverse=True):
        text = text.replace(form, "***")
    return text


def _pings() -> dict:
    """Per provider: (key variable, URL, headers from the key, level of a rejected key). Each URL is free."""
    from promo.providers import gemini

    return {
        "gemini": ("GEMINI_API_KEY", f"https://generativelanguage.googleapis.com/v1beta/models/{gemini.DEFAULT_MODEL}",
                   lambda k: {"x-goog-api-key": k}, "error"),
        "elevenlabs": ("ELEVENLABS_API_KEY", "https://api.elevenlabs.io/v1/user", lambda k: {"xi-api-key": k}, "error"),
        "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/key",
                       lambda k: {"Authorization": f"Bearer {k}"}, "warning"),
    }


def _ping_one(provider: str) -> Check | None:
    """One free key-validation call (a model lookup, the account, the key's own record); None when the key is not set.

    The key never reaches a message: a key a header cannot carry is refused before any request, an exception other
    than the provider's own error is reported by its type alone, and the provider's message is redacted."""
    from promo.providers.http import ProviderError, request

    env_name, url, headers, level = _pings()[provider]
    key = os.environ.get(env_name, "").strip()
    if not key:
        return None
    name = f"{provider} ping"
    if re.search(r"[\s\x00-\x1f\x7f-\x9f]", key):
        return Check(name=name, ok=False, level=level, detail="key contains whitespace or control characters")
    if not key.isascii():
        return Check(name=name, ok=False, level=level, detail="key contains non-ASCII characters")
    try:
        request(provider, "GET", url, headers=headers(key), retries=1, timeout=20)
    except ProviderError as e:
        return Check(name=name, ok=False, level=level, detail=_redact(e.detail, key))
    except Exception as e:      # whatever it is, its text can hold the header it choked on
        return Check(name=name, ok=False, level=level, detail=f"the request failed ({type(e).__name__})")
    return Check(name=name, ok=True, detail="key accepted")


def _ping(live: bool) -> list[Check]:
    if not live:
        return []
    return [c for provider in _pings() if (c := _ping_one(provider))]


def _run_context():
    """(run, shot list, pack) of the run folder here, or None outside one or when its shot list or pack cannot be
    read: that is another command's error to report."""
    try:
        from promo.packs import load_pack
        from promo.run import RunNotFound, find_run
        from promo.shotlist import load_shotlist

        run = find_run()
        sl = load_shotlist(run.shotlist)
        return run, sl, load_pack(sl.pack)
    except (ImportError, RunNotFound, OSError, ValueError):    # ShotlistError and PackError are ValueErrors
        return None


def _run_voice_provider() -> str | None:
    """The voice provider the current run's shot list selects (shot list, then pack, then the default), or None
    outside a run folder or when its shot list or pack cannot be read."""
    ctx = _run_context()
    if ctx is None:
        return None
    from promo.voice import voice_settings

    _, sl, pack = ctx
    try:
        return voice_settings(sl, pack).provider
    except ValueError:
        return None


def run_key_needs(run, sl, pack, provider: str | None) -> KeyNeeds:
    """What the run uses each paid API for. Gemini: a Gemini voice. A generated music bed (a music-led film whose pack
    plans a bed and that has no bed of its own; a voice-led film can go without music) needs its music provider's key:
    ElevenLabs by default, Gemini for Lyria. ElevenLabs also: pack sounds of the run's events that are generated
    (`gen`) rather than synthesised or read from a file."""
    gemini_uses = ["the voice (Gemini TTS)"] if provider == "gemini" else []
    eleven_uses = []
    own_bed = (run.music_dir / "bed.wav").exists() and not (run.music_dir / "bed.src.mp3").exists()
    if sl.timing_mode == "music" and pack.music.get("mode", "bed") != "none" and not sl.music.source and not own_bed:
        music_provider = getattr(sl.music, "provider", None) or "elevenlabs"
        (gemini_uses if music_provider == "lyria" else eleven_uses).append(
            f"the music bed (`promo music`, {'Lyria' if music_provider == 'lyria' else 'ElevenLabs'})")
    names = {ev.sfx or ev.kind for shot in sl.all_shots() for ev in shot.events}
    generated = sorted(n for n in names if isinstance(pack.sfx.get(n), dict) and "gen" in pack.sfx[n])
    if generated:
        eleven_uses.append(f"generated sound effects ({', '.join(generated)})")
    return KeyNeeds(gemini=gemini_uses, elevenlabs=eleven_uses)


def _run_key_needs() -> KeyNeeds | None:
    ctx = _run_context()
    return run_key_needs(*ctx, _run_voice_provider()) if ctx else None


def _supertonic_checks(required: bool) -> list[Check]:
    from promo.providers import supertonic

    if shutil.which(supertonic.BINARY) is None:
        return [check_supertonic(None, required=required)]
    try:
        version = supertonic.version()
    except (ToolError, OSError, subprocess.TimeoutExpired) as e:   # on PATH, but does not run
        why = (str(e).splitlines() or [type(e).__name__])[0]
        return [Check(name="supertonic", ok=False, level="error" if required else "warning",
                      detail=f"found on PATH but `supertonic version` failed ({why})")]
    return [check_supertonic(version, required=required),
            check_supertonic_model(Path.home() / ".cache" / "supertonic2")]


def _hf_snapshot(repo: str) -> Path | None:
    """The cached snapshot folder of a Hugging Face model, found without any network call."""
    try:
        from huggingface_hub import try_to_load_from_cache

        config = try_to_load_from_cache(repo, "config.json")
    except ImportError:
        return None
    return Path(config).parent if isinstance(config, str) else None


def _asr_checks() -> list[Check]:
    from promo import asr

    return [check_asr_model("parakeet", asr.PARAKEET_MODEL, _hf_snapshot(asr.PARAKEET_MODEL)),
            check_asr_model("whisper", asr.WHISPER_MODEL, _hf_snapshot(asr.WHISPER_MODEL))]


def _stdout_of(cmd: list[str]) -> str | None:
    """What a quick tool prints, or None when it will not run (so `promo doctor` itself never crashes)."""
    try:
        proc = run(cmd, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _node_checks() -> list[Check]:
    out = []
    version = _stdout_of(["node", "--version"]) if shutil.which("node") else None
    out.append(check_node(version) if version else
               Check(name="node", ok=False, detail="not found or does not run: install Node.js 22 or newer"))
    npx = shutil.which("npx")
    out.append(Check(name="npx", ok=bool(npx), detail=npx or "not found: it comes with Node.js, and runs HyperFrames"))
    return out


def _ffmpeg_checks() -> list[Check]:
    filters = _stdout_of(["ffmpeg", "-hide_banner", "-filters"]) if shutil.which("ffmpeg") else None
    if filters is None:
        return [Check(name="ffmpeg", ok=False, detail="not found or does not run: install ffmpeg (for example `brew install ffmpeg`)")]
    out = check_ffmpeg(filters)
    probe = shutil.which("ffprobe")
    out.append(Check(name="ffprobe", ok=bool(probe), detail=probe or "not found: it comes with ffmpeg"))
    return out


def run_checks(live: bool = False) -> list[Check]:
    load_env()
    checks = check_keys(dict(os.environ), _run_key_needs())
    checks += _node_checks()
    checks += _ffmpeg_checks()
    checks += [check_module(m) for m in PY_MODULES]
    checks += _supertonic_checks(required=_run_voice_provider() == "supertonic")
    checks += _asr_checks()
    checks.append(check_hyperframes(_hf_doctor()))
    checks.append(check_disk(shutil.disk_usage(tempfile.gettempdir()).free))
    checks.append(Check(name="manim / revideo", ok=True, level="info", detail="arrive in plan 3"))
    return checks + _ping(live)
