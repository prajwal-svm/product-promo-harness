"""Master chain: loudnorm measurement → linear gain → limiter → AAC mux."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from promo.proc import ToolError, atomic_path, run

SR = 48000


def loudnorm_measure(wav: Path, target_i: float, tp: float, lra: float = 11.0) -> dict:
    """loudnorm's first pass: the measured input_i, input_tp, input_lra, input_thresh and target_offset (strings)."""
    err = run(["ffmpeg", "-hide_banner", "-nostats", "-i", wav, "-af",
               f"loudnorm=I={target_i}:TP={tp}:LRA={lra}:print_format=json", "-f", "null", "-"]).stderr
    blocks = re.findall(r"\{[^{}]*\}", err, flags=re.S)
    if not blocks:
        raise ToolError(f"ffmpeg loudnorm printed no measurement for {wav}")
    return json.loads(blocks[-1])


def master_gain_db(measured: dict, target_i: float) -> float | None:
    """The one gain that takes the measured mix to `target_i` LUFS; None when the mix is silent."""
    try:
        input_i = float(measured.get("input_i"))
    except (TypeError, ValueError):
        return None
    return round(target_i - input_i, 2) if math.isfinite(input_i) else None


def master_audio(src: Path, dst: Path, *, target_i: float = -14.0, ceiling_db: float = -2.0, lra: float = 11.0,
                 measured: dict | None = None) -> Path:
    """Bring `src` to `target_i` LUFS with one linear gain (pass 1's measured offset), then limit its peaks to
    `ceiling_db` dBFS. `measured` is loudnorm's pass-1 JSON when the caller already has it.

    No loudnorm pass 2: it stays linear only while the measured true peak plus the gain is at or below its TP, which a
    float pre-master with SFX above full scale rarely is, and its dynamic mode then lifts a planned quiet section
    (-10 dB came out -7.5 dB). The limiter handles the peaks instead. It runs at 4x the sample rate, so it also holds
    the peaks between samples: at 48 kHz a boosted noise burst limited to -2 dBFS still measured +1.8 dBTP on the
    MP4, at 192 kHz -2.0. It compensates its look-ahead, so the sound does not move against the picture. A silent mix
    is only limited.
    """
    m = measured if measured is not None else loudnorm_measure(src, target_i, ceiling_db, lra)
    gain = master_gain_db(m, target_i)
    limit = (f"aresample={4 * SR},alimiter=limit={10 ** (ceiling_db / 20):.4f}:level=false:attack=5:release=50:"
             f"latency=true,aresample={SR}")
    af = limit if gain is None else f"volume={gain:.2f}dB,{limit}"
    with atomic_path(dst) as tmp:
        run(["ffmpeg", "-v", "error", "-y", "-i", src, "-af", af, "-ar", str(SR), "-c:a", "pcm_s24le", tmp])
    return dst


def mux(picture: Path, audio: Path, out: Path, *, draft: bool = False) -> Path:
    """Copy the picture and encode the master to AAC. No -shortest: the picture and the mix are both exactly the
    film's length, and -shortest on a copied H.264 stream also drops the B-frames decoded after its cut."""
    with atomic_path(out) as tmp:
        run(["ffmpeg", "-v", "error", "-y", "-i", picture, "-i", audio, "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k" if draft else "320k", "-ar", str(SR),
             "-movflags", "+faststart", tmp])
    return out
