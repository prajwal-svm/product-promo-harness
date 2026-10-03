"""Supertonic: local, on-device text to speech through its `supertonic` CLI. No key, no cost.

The model has no style prompt. Its output is not deterministic (the diffusion sampler starts from unseeded
noise), so every synthesis is cached by content hash: a rerun reuses the first take instead of drawing a new one.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from promo.proc import ToolError, sha256_hex
from promo.proc import run as run_cmd

BINARY = "supertonic"
DEFAULT_MODEL = "supertonic-2"
DEFAULT_VOICE = "F2"
DEFAULT_LANG = "en"
# About 167 wpm on a 22-word line: the Chain-style density. The CLI's own defaults (5 steps, 1.05x) are slower and rougher.
DEFAULT_OPTIONS = {"steps": 12, "speed": 1.25, "silence_s": 0.3}
TIMEOUT_S = 600.0  # the first run of a model downloads it
# Very short lines ("Psst.", "Hm.") now and then come back as digital silence (peak 3e-5; 2 of 30 and 6 of 30 draws
# measured, 0 of 30 for a normal sentence). The sampler is unseeded, so a fresh draw usually speaks.
SILENT_PEAK = 1e-3
SILENT_DRAWS = 5
NOT_INSTALLED = ("Supertonic is not installed (pip install supertonic or `uv tool install supertonic`; "
                 "Homebrew has no formula for it)")


class SupertonicError(ToolError):
    """The supertonic CLI failed."""


class SupertonicSilence(SupertonicError):
    """Every draw of a line came back silent."""


class SupertonicNotInstalled(SupertonicError):
    """No `supertonic` executable on PATH: a missing prerequisite, not a failed take."""


def _binary() -> str:
    path = shutil.which(BINARY)
    if path is None:
        raise SupertonicNotInstalled(NOT_INSTALLED)
    return path


def _failure(proc: subprocess.CompletedProcess) -> str:
    """The CLI's own explanation: its `❌ Error:` line (stdout), else the last line it wrote to stderr."""
    for ln in proc.stdout.splitlines():
        if ln.startswith("❌"):
            return ln.lstrip("❌").strip().removeprefix("Error:").strip()
    lines = [ln.strip() for ln in proc.stderr.splitlines() if ln.strip()]
    return lines[-1] if lines else "no output"


def _run(cmd: list, *, timeout: float) -> subprocess.CompletedProcess:
    try:
        proc = run_cmd(cmd, check=False, timeout=timeout)
    except FileNotFoundError as e:  # it was on PATH a moment ago
        raise SupertonicNotInstalled(NOT_INSTALLED) from e
    except subprocess.TimeoutExpired as e:
        raise SupertonicError(f"supertonic did not finish within {timeout:g} s") from e
    if proc.returncode != 0:
        raise SupertonicError(f"supertonic failed (exit {proc.returncode}): {_failure(proc)}")
    return proc


@lru_cache(maxsize=None)
def _version_of(binary: str) -> str:
    out = _run([binary, "version"], timeout=60.0).stdout.strip()
    return out.splitlines()[-1].strip() if out else "unknown"


def version() -> str:
    """What `supertonic version` prints, e.g. `supertonic 1.1.2`; looked up once per executable."""
    return _version_of(_binary())


def _num(x: float) -> str:
    return f"{x:g}"


def primary_lang(lang: str) -> str:
    """`en-GB` and `pt_BR` are `en` and `pt`: the CLI takes a bare language code and crashes on a region tag."""
    return re.split(r"[-_]", lang.strip().lower())[0]


def command(binary: str, text: str, out: Path, *, voice: str, model: str, lang: str, steps: int, speed: float,
            silence_s: float) -> list[str]:
    """The argument list for one line. `--` keeps text that starts with a dash from being read as an option."""
    return [binary, "tts", "-o", str(out), "--model", model, "--voice", voice, "--lang", primary_lang(lang),
            "--steps", str(int(steps)), "--speed", _num(speed), "--silence-duration", _num(silence_s), "--", text]


def cache_key(text: str, *, voice: str, model: str, lang: str, steps: int, speed: float, silence_s: float) -> str:
    """Everything the audio depends on, including the installed version: a new release may sound different.

    The language is the full tag the caller gave, not the code the CLI gets, so the key is a pure function of the request.
    """
    return sha256_hex("supertonic-tts", version(), text, voice, model, lang, str(int(steps)), _num(speed),
                      _num(silence_s))


def _speak(text: str, **opts) -> bytes:
    """One run of the CLI: the WAV it wrote."""
    binary = _binary()
    with tempfile.TemporaryDirectory(prefix="promo-supertonic-") as tmp:
        out = Path(tmp) / "line.wav"
        _run(command(binary, text, out, **opts), timeout=TIMEOUT_S)
        data = out.read_bytes() if out.is_file() else b""
    if data[:4] != b"RIFF":
        raise SupertonicError("supertonic exited 0 but wrote no WAV file")
    return data


def _peak(wav: bytes) -> float:
    try:
        samples, _ = sf.read(io.BytesIO(wav), dtype="float32")
    except RuntimeError as e:
        raise SupertonicError(f"supertonic wrote a WAV file that cannot be read ({e})") from e
    return float(np.abs(samples).max()) if samples.size else 0.0


def synthesize(text: str, *, voice: str = DEFAULT_VOICE, model: str = DEFAULT_MODEL, lang: str = DEFAULT_LANG,
               steps: int = DEFAULT_OPTIONS["steps"], speed: float = DEFAULT_OPTIONS["speed"],
               silence_s: float = DEFAULT_OPTIONS["silence_s"]) -> bytes:
    """One line of speech as WAV bytes (44.1 kHz mono). A silent draw is thrown away and drawn again."""
    opts = dict(voice=voice, model=model, lang=lang, steps=steps, speed=speed, silence_s=silence_s)
    for _ in range(SILENT_DRAWS):
        wav = _speak(text, **opts)
        if _peak(wav) >= SILENT_PEAK:
            return wav
    raise SupertonicSilence(f"supertonic returned silence {SILENT_DRAWS} times in a row for {text!r}")
