"""The SFX stem for this film: audio/sfx/track.wav, one sound per event that names an `sfx` in the shot list.

    python tools/sfx.py        (run after every `promo timing`; `promo assemble` mixes the stem as it is on disk)

The sounds are short ElevenLabs sound generations, cached by prompt (`promo.sfx.generate`; ELEVENLABS_API_KEY on the
first run only), peak-normalised, each loudest sample placed on its event, at -4 dB plus a per-kind offset.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

RUN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN.parents[2]))
from promo.sfx import _decode_bytes, generate  # noqa: E402

SR, GAIN_DB = 48000, -4
REL_DB = {"click": 0, "whoosh": -2, "impact": -1, "pop": -4, "tick": -8}
GEN = {
    "click": ("a single crisp modern laptop trackpad click, clean user-interface sound, very short, no reverb", 0.5),
    "whoosh": ("a quick smooth airy swoosh for a slide transition in a modern app promo video, clean and short", 0.8),
    "impact": ("a soft punchy whoosh hit for a title appearing in a modern promo video, short", 1.0),
    "pop": ("a soft friendly bubble pop, a light user-interface notification sound, short", 0.5),
    "tick": ("a tiny soft user-interface tick, subtle and clean, very short", 0.5),
}


def main() -> None:
    tj = json.loads((RUN / "timing.json").read_text())
    n = round(tj["duration_s"] * SR)
    buf, memo = np.zeros(n), {}
    for e in tj["events"]:
        k = e.get("sfx")
        if k not in GEN:
            continue
        if k not in memo:
            x = np.asarray(_decode_bytes(generate(*GEN[k], costs_path=RUN / "costs.jsonl"), k), dtype=np.float64)
            x = x.mean(axis=1) if x.ndim > 1 else x
            memo[k] = x / (np.abs(x).max() or 1.0) * 0.9 * 10 ** ((GAIN_DB + REL_DB[k]) / 20)
        s = memo[k]
        start = round(e["t"] * SR) - int(np.argmax(np.abs(s)))
        a0, s0 = max(0, start), max(0, -start)
        end = min(n, a0 + len(s) - s0)
        if end > a0:
            buf[a0:end] += s[s0:s0 + end - a0]
    out = RUN / "audio" / "sfx" / "track.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), np.clip(buf, -1, 1), SR, subtype="PCM_24")
    print(out, sum(1 for e in tj["events"] if e.get("sfx") in GEN), "sounds")


if __name__ == "__main__":
    main()
