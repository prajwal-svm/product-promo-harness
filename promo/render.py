"""OpenTimelineIO → one ffmpeg command: a silent picture and the pre-master mix."""

from __future__ import annotations

import math
import re
from pathlib import Path

import opentimelineio as otio

SR = 48000
# Ducking. VO takes are normalised to -16 LUFS (promo.voice) and reach the VO bus at unity. Under sidechaincompress's
# RMS detector (attack 15 ms, release 350 ms) speech then holds an effective key level of -9.4 dBFS, measured on
# recorded speech. With the threshold fixed, the depth during speech is (key - threshold) * (1 - 1/ratio), so the
# pack's depth in dB sets the ratio. Measured: 4.0, 6.0, 7.0, 8.0 and 10.0 dB for 4, 6, 7, 8 and 10 asked. A low
# threshold keeps the depth insensitive to how a particular voice's level moves (about a third of a dB per dB).
DUCK_KEY_DB = -9.4
DUCK_THRESHOLD = 0.03
DUCK_ATTACK_MS = 15
DUCK_RELEASE_MS = 350
MAX_DUCK_DB = 18.0
# Generated frames (gaps) are converted with the BT.709 matrix the shots use, and the picture is tagged BT.709.
# ffmpeg converts a colour source with BT.601 by default, and the encoder options alone leave primaries and
# transfer unset in ffmpeg 8, so the frames are tagged too.
TO_BT709 = "format=gbrp,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"
TAG_BT709 = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
BT709_FLAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
# Every source onto the stereo bus. Mono (the VO takes and the SFX stem) lands on both channels at unity, so the
# mix carries the stem's own samples; stereo passes through untouched. ffmpeg's implicit upmix would put mono at
# -3 dB per channel instead.
TO_STEREO = "pan=stereo|FL=FL+FC|FR=FR+FC"
_HEX = re.compile(r"#?([0-9A-Fa-f]{6}|[0-9A-Fa-f]{3})")


def ffmpeg_colour(css: str) -> str:
    """`#F2F3EC` or `#abc` → `0xF2F3EC`; anything else (rgba(), names) is a ValueError."""
    m = _HEX.fullmatch(str(css).strip())
    if not m:
        raise ValueError(f"background colour {css!r} must be a hex colour like #F2F3EC")
    h = m.group(1)
    return "0x" + (h if len(h) == 6 else "".join(c * 2 for c in h)).upper()


def duck_filter(depth_db: float) -> str | None:
    """sidechaincompress for a VO duck of `depth_db` dB under -16 LUFS speech; None for no duck (0 dB)."""
    if isinstance(depth_db, bool) or not isinstance(depth_db, (int, float)) or not 0 <= depth_db <= MAX_DUCK_DB:
        raise ValueError(f"music.duck_db must be a depth from 0 to {MAX_DUCK_DB:g} dB (got {depth_db!r})")
    if depth_db == 0:
        return None
    ratio = 1 / (1 - depth_db / (DUCK_KEY_DB - 20 * math.log10(DUCK_THRESHOLD)))
    return (f"sidechaincompress=threshold={DUCK_THRESHOLD}:ratio={ratio:.3f}:attack={DUCK_ATTACK_MS}:"
            f"release={DUCK_RELEASE_MS}")


def _items(track):
    for i, item in enumerate(track):
        r = track.range_of_child_at_index(i)
        yield item, r.start_time.to_seconds(), r.duration.to_seconds()


def _is_clip(item) -> bool:
    return isinstance(item, otio.schema.Clip)


def plan_ffmpeg(tl, *, width: int, height: int, fps: int, bg: str, duck_db: float, fade_out_s: float,
                picture_out: Path, mix_out: Path, draft: bool = False) -> list[str]:
    """One ffmpeg command for the whole edit. Every file is an `-i` input addressed by index, never a path in the
    graph. The picture is H.264 at `fps`; the mix is a 48 kHz float WAV exactly `duration_s` long (float, because
    VO + music + SFX summed at unity can pass full scale before the master chain brings it down).

    `duck_db` is how far the music drops under the voice (the pack's music.duck_db); `fade_out_s` is the fade at the
    end of the bed, whether the bed runs out early or the film end cuts it."""
    duck = duck_filter(duck_db)
    if isinstance(fade_out_s, bool) or not isinstance(fade_out_s, (int, float)) or not 0 <= fade_out_s <= 30:
        raise ValueError(f"music.fade_out_s must be 0 to 30 seconds (got {fade_out_s!r})")
    total = float(tl.metadata["promo"]["duration_s"])
    bgc = ffmpeg_colour(bg)
    inputs: list[str] = []
    graph: list[str] = []
    count = 0

    def add(*args: str) -> int:
        nonlocal count
        inputs.extend(args)
        count += 1
        return count - 1

    tracks = {t.name: t for t in tl.tracks}

    # V1. Each item's frame count comes from its absolute start and end at the output rate, so a cut that falls
    # between two output frames never shifts the shots after it. A render shorter than its slot holds its last frame.
    vlabels = []
    for k, (item, start, dur) in enumerate(_items(tracks["V1"])):
        frames = round((start + dur) * fps) - round(start * fps)
        if frames <= 0:
            continue
        if _is_clip(item):
            idx = add("-i", item.media_reference.target_url)
            src = item.source_range.start_time.to_seconds()
            graph.append(
                f"[{idx}:v]trim=start={src:.6f}:duration={dur:.6f},setpts=PTS-STARTPTS,fps={fps},"
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={bgc},setsar=1,format=yuv420p,"
                f"tpad=stop_mode=clone:stop_duration={dur:.6f},trim=end_frame={frames}[v{k}]")
        else:  # a hole or a missing shot: the pack background
            idx = add("-f", "lavfi", "-i", f"color=c={bgc}:s={width}x{height}:r={fps}:d={(frames + 1) / fps:.6f}")
            graph.append(f"[{idx}:v]{TO_BT709},trim=end_frame={frames},setsar=1[v{k}]")
        vlabels.append(f"[v{k}]")
    graph.append(f"{''.join(vlabels)}concat=n={len(vlabels)}:v=1:a=0[vbase]")
    vcur = "vbase"
    if "V2" in tracks:
        for k, (item, start, dur) in enumerate(_items(tracks["V2"])):
            if not _is_clip(item):
                continue
            idx = add("-i", item.media_reference.target_url)
            # At the output rate: after the base ends, overlay keeps emitting frames at the overlay's own times, so
            # a 60 fps overlay on a 30 fps draft would add a frame past the end of the film.
            graph.append(f"[{idx}:v]trim=duration={dur:.6f},setpts=PTS-STARTPTS,fps={fps},setpts=PTS+{start:.6f}/TB,"
                         f"scale={width}:{height}[ov{k}]")
            graph.append(f"[{vcur}][ov{k}]overlay=0:0:enable='between(t,{start:.6f},{start + dur:.6f})':format=auto[vov{k}]")
            vcur = f"vov{k}"
    graph.append(f"[{vcur}]trim=end_frame={round(total * fps)},format=yuv420p,{TAG_BT709}[vout]")

    def audio_track(name: str, prefix: str, *, fade_out: bool = False) -> list[str]:
        labels = []
        if name not in tracks:
            return labels
        for k, (item, start, dur) in enumerate(_items(tracks[name])):
            if not _is_clip(item):
                continue
            idx = add("-i", item.media_reference.target_url)
            src = item.source_range.start_time.to_seconds()
            chain = [f"atrim=start={src:.6f}:duration={dur:.6f}", "asetpts=PTS-STARTPTS", f"aresample={SR}",
                     TO_STEREO, "aformat=sample_fmts=fltp:channel_layouts=stereo"]
            if fade_out and fade_out_s > 0:  # the bed's end never stops dead, whether it ran out or the film cut it
                fade = min(fade_out_s, dur)
                chain.append(f"afade=t=out:st={dur - fade:.6f}:d={fade:.6f}:curve=hsin")
            chain.append(f"adelay=delays={round(start * 1000)}:all=1")
            graph.append(f"[{idx}:a]{','.join(chain)}[{prefix}{k}]")
            labels.append(f"[{prefix}{k}]")
        return labels

    def mixed(labels: list[str]) -> str:
        """The start of a chain that sums `labels` at unity (one label needs no amix)."""
        if len(labels) == 1:
            return labels[0]
        return f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:dropout_transition=0,"

    def bus(labels: list[str], out: str) -> None:
        """Sum one track's clips, then silence without end. sidechaincompress stops at the first end of either input
        and drops what the other still holds, and which one ends first depends on thread timing; a VO key that ends
        with the last line would cut the music there. No bus ends, so only the final atrim ends the mix."""
        graph.append(f"{mixed(labels)}apad[{out}]")

    vo = audio_track("A1 VO", "vo")
    mus = audio_track("A2 Music", "mus", fade_out=True)
    sfx = audio_track("A3 SFX", "sfx")
    if vo:
        bus(vo, "vo")
    else:
        graph.append(f"anullsrc=r={SR}:cl=stereo[vo]")
    parts = []
    if mus and duck:
        bus(mus, "musraw")
        graph.append("[vo]asplit=2[vosc][vomix]")
        graph.append(f"[musraw][vosc]{duck}[duck]")
        parts += ["[duck]", "[vomix]"]
    elif mus:
        bus(mus, "musraw")
        parts += ["[musraw]", "[vo]"]
    else:
        parts.append("[vo]")
    if sfx:
        bus(sfx, "sfxall")
        parts.append("[sfxall]")
    graph.append(f"{mixed(parts)}atrim=0:{total:.6f}[mix]")
    if draft:
        vcodec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
    else:
        vcodec = ["-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-crf", "16"]
    return ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(graph),
            "-map", "[vout]", *vcodec, "-pix_fmt", "yuv420p", *BT709_FLAGS, "-r", str(fps), "-an", str(picture_out),
            "-map", "[mix]", "-c:a", "pcm_f32le", "-ar", str(SR), str(mix_out)]
