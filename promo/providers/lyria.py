"""Lyria music over the Gemini API (the Interactions API), with the harness's GEMINI_API_KEY.

https://ai.google.dev/gemini-api/docs/music-generation (read 2026-10-01): `POST /v1beta/interactions` with
`{"model", "input"}`; the reply's `model_output` steps hold text blocks (the model's song structure) and an `audio`
block (base64 MP3, 44.1 kHz stereo, SynthID-watermarked). Lyria has no length, tempo, seed or instrumental field:
the prompt asks for all of them, so `compose` takes `length_ms` only to match the ElevenLabs callable.
"""

from __future__ import annotations

import base64

from promo.config import require_key
from promo.providers.http import ProviderError, request

INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MUSIC_MODEL = "lyria-3.5"
CLIP_MODEL = "lyria-3-clip-preview"
# The docs promise "a couple of minutes" a call; asked for 264 s, lyria-3.5 made 177 s (probe, 2026-10-01). Beds
# longer than this come in segments (promo.music.plan_segments).
MAX_CALL_S = 165.0
# USD per call ("per song"), paid tier only; https://ai.google.dev/gemini-api/docs/pricing, read 2026-10-01.
USD_PER_CALL = {MUSIC_MODEL: 0.08, "lyria-3-pro-preview": 0.08, CLIP_MODEL: 0.04}


def call_usd(model: str) -> float:
    return USD_PER_CALL.get(model, USD_PER_CALL[MUSIC_MODEL])


def build_body(prompt: str, model: str = MUSIC_MODEL) -> dict:
    # store=False: the prompt describes an unreleased product; nothing here needs server-side state.
    return {"model": model, "input": prompt, "store": False}


def _blocks(payload: dict) -> list[dict]:
    """Every content block of the reply, in order: `steps[].content[]` of model_output steps, or the older
    top-level `outputs[]`."""
    out: list[dict] = []
    for step in payload.get("steps") or []:
        if isinstance(step, dict) and step.get("type") == "model_output":
            out.extend(b for b in step.get("content") or [] if isinstance(b, dict))
    out.extend(b for b in payload.get("outputs") or [] if isinstance(b, dict))
    return out


def extract_audio(payload: dict) -> tuple[bytes, str]:
    """(audio bytes, mime type) of the reply's last audio block, or a ProviderError that says what came back."""
    blocks = _blocks(payload) if isinstance(payload, dict) else []
    audio = [b for b in blocks if b.get("type") == "audio" and b.get("data")]
    if audio:
        last = audio[-1]
        return base64.b64decode(last["data"]), last.get("mime_type") or last.get("mimeType") or ""
    text = " ".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text").strip()
    status = payload.get("status") if isinstance(payload, dict) else None
    detail = "; ".join(x for x in [f"status {status}" if status else "", text[:300]] if x)
    raise ProviderError("lyria", 200, f"no audio in the reply{': ' + detail if detail else ''}")


def compose(prompt: str, length_ms: float, *, model: str = MUSIC_MODEL, client=None) -> bytes:
    """One generation. The prompt carries the length (build_prompt writes it); `length_ms` is unused."""
    key = require_key("GEMINI_API_KEY")
    resp = request("lyria", "POST", INTERACTIONS_URL, headers={"x-goog-api-key": key},
                   json=build_body(prompt, model), timeout=900.0, client=client)
    audio, _ = extract_audio(resp.json())
    return audio
