"""ElevenLabs music and sound effects (Scribe lives in promo.asr)."""

from __future__ import annotations

from promo.config import require_key
from promo.providers.http import request

BASE = "https://api.elevenlabs.io"
MUSIC_URL = f"{BASE}/v1/music"
SFX_URL = f"{BASE}/v1/sound-generation"
MUSIC_MODEL = "music_v2_5"
MUSIC_USD_PER_MIN = 0.15
MUSIC_MIN_MS = 3_000
MUSIC_MAX_MS = 600_000
SFX_OUTPUT_FORMAT = "mp3_44100_128"
SFX_PROMPT_INFLUENCE = 0.5


def clamp_length_ms(length_ms: float) -> int:
    """The music length the API is actually asked for (and billed on)."""
    return int(min(MUSIC_MAX_MS, max(MUSIC_MIN_MS, round(length_ms))))


def compose(prompt: str, length_ms: float, *, instrumental: bool = True, model: str = MUSIC_MODEL,
            output_format: str = "mp3_44100_192", client=None) -> bytes:
    key = require_key("ELEVENLABS_API_KEY")
    length = clamp_length_ms(length_ms)
    resp = request("elevenlabs", "POST", MUSIC_URL, headers={"xi-api-key": key},
                   params={"output_format": output_format},
                   json={"prompt": prompt, "music_length_ms": length, "model_id": model,
                         "force_instrumental": instrumental},
                   timeout=600.0, client=client)
    return resp.content


def sound_generation(text: str, duration_s: float, *, prompt_influence: float = SFX_PROMPT_INFLUENCE,
                     output_format: str = SFX_OUTPUT_FORMAT, client=None) -> bytes:
    key = require_key("ELEVENLABS_API_KEY")
    resp = request("elevenlabs", "POST", SFX_URL, headers={"xi-api-key": key},
                   params={"output_format": output_format},
                   json={"text": text, "duration_seconds": duration_s, "prompt_influence": prompt_influence},
                   timeout=180.0, client=client)
    return resp.content
