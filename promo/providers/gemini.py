"""Gemini 3.8 TTS over the Gemini API (generateContent with AUDIO output)."""

from __future__ import annotations

import base64
import io
import re
import wave
from pathlib import Path

from promo.config import cache_dir, require_key
from promo.proc import atomic_path
from promo.providers.http import ProviderError, request

TTS_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-3.8-flash-tts"
LITE_MODEL = "gemini-3.8-flash-lite-tts"
SHAPES = ("prebuilt", "voice")
AUDIO_TOKENS_PER_S = 25
# USD per 1M tokens (text in, audio out), verified 2026-09-30. Both double on 2027-01-01.
PRICE = {DEFAULT_MODEL: (0.50, 9.00), LITE_MODEL: (0.50, 6.00)}


def build_body(text: str, voice: str, style: str, shape: str) -> dict:
    part: dict = {"text": text}
    if style:
        part["speech_metadata"] = {"style": style}
    voice_cfg = {"prebuiltVoiceConfig": {"voiceName": voice}} if shape == "prebuilt" else {"voice": voice}
    return {
        "contents": [{"role": "user", "parts": [part]}],
        "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {"voiceConfig": voice_cfg}},
    }


def ensure_wav(data: bytes, mime: str = "") -> bytes:
    if data[:4] == b"RIFF":
        return data
    m = re.search(r"rate=(\d+)", mime or "")
    rate = int(m.group(1)) if m else 24000
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data)
    return buf.getvalue()


def _shape_file() -> Path:
    return cache_dir("gemini") / "voice_shape.txt"


def _shape_order() -> list[str]:
    f = _shape_file()
    first = f.read_text().strip() if f.exists() else ""
    if first in SHAPES:
        return [first] + [s for s in SHAPES if s != first]
    return list(SHAPES)


def _remember_shape(shape: str) -> None:
    f = _shape_file()
    if f.exists() and f.read_text().strip() == shape:
        return
    with atomic_path(f) as tmp:
        tmp.write_text(shape)


def _extract_audio(payload: dict) -> tuple[bytes, str]:
    try:
        parts = payload["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as e:
        raise ProviderError("gemini", 200, f"no audio in response: {str(payload)[:200]}") from e
    for p in parts:
        inline = p.get("inlineData") or p.get("inline_data")
        if inline and inline.get("data"):
            return base64.b64decode(inline["data"]), inline.get("mimeType") or inline.get("mime_type") or ""
    raise ProviderError("gemini", 200, "no inlineData audio part in response")


def synthesize(text: str, *, voice: str, style: str = "", model: str = DEFAULT_MODEL, client=None) -> bytes:
    key = require_key("GEMINI_API_KEY")
    url = TTS_URL.format(model=model)
    last: ProviderError | None = None
    for shape in _shape_order():
        try:
            resp = request("gemini", "POST", url, headers={"x-goog-api-key": key},
                           json=build_body(text, voice, style, shape), client=client)
        except ProviderError as e:
            if e.status == 400 and re.search(r"voice|Unknown name|speech_?config", e.detail, re.I):
                last = e
                continue
            raise
        data, mime = _extract_audio(resp.json())
        wav = ensure_wav(data, mime)
        try:
            _remember_shape(shape)
        except OSError:
            pass  # best-effort state must never lose a paid take
        return wav
    assert last is not None
    raise last


def estimate_usd(model: str, text: str, audio_s: float) -> float:
    tin, tout = PRICE.get(model, PRICE[DEFAULT_MODEL])
    return (len(text) / 4) * tin / 1e6 + audio_s * AUDIO_TOKENS_PER_S * tout / 1e6
