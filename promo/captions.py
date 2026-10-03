"""Word-level captions: pages, SRT/VTT sidecars, and a transparent HyperFrames overlay."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from promo.hyperframes import HF_VERSION, TEMPLATE_DIR, render, stamp_root
from promo.models import Caption
from promo.packs import Pack
from promo.proc import atomic_path, sha256_hex
from promo.run import RunPaths
from promo.timing import Timing

OVERLAY_TEMPLATE = TEMPLATE_DIR.parent / "caption-overlay" / "index.html"

# Layout shared with the overlay template (passed to it as captions.json "box"):
# a page is inset BOX_MARGIN of the frame width on each side, and a plate adds PLATE_PAD_X_EM of padding per side.
BOX_MARGIN = 0.08
PLATE_PAD_X_EM = 0.5
# Conservative average glyph width, in em, used to turn the text-box width into a characters-per-line limit.
# Inter at weight 600 averages 0.50 em on lower-case prose, 0.61 on all-caps prose, 0.55 on digits and currency and
# 0.74 on a run of the widest capitals ("WOMEN MAKE MOMMY WOW ..."), with single lines of that run above 0.75.
# Checked in Chrome on 11,000 pages of those corpora plus wide lower-case, mixed and long-word text, at 16:9, 9:16
# and 1:1, 54-90 px, with and without a plate: 0.75 left 4 pages on a third line, 0.80 leaves none. A line that
# `pages` counts as fitting therefore also fits in the browser, and a page never wraps to a third line.
GLYPH_EM = 0.80


class CaptionsError(RuntimeError):
    """The captions cannot be built from this run's files (for example an unreadable design.json)."""


class Page(BaseModel):
    text: str
    startMs: int
    endMs: int
    words: list[Caption]


def words_from_timing(timing: Timing) -> list[Caption]:
    return sorted((w for line in timing.lines for w in line.words), key=lambda w: w.startMs)


def _line_count(words: list[Caption], max_chars: int) -> int:
    lines, cur = 1, 0
    for w in words:
        need = len(w.text) + (1 if cur else 0)
        if cur and cur + need > max_chars:
            lines, cur = lines + 1, len(w.text)
        else:
            cur += need
    return lines


def _page(ws: list[Caption]) -> Page:
    return Page(text=" ".join(w.text for w in ws), startMs=ws[0].startMs, endMs=ws[-1].endMs, words=list(ws))


_CLOSERS = "\"'\u201d\u2019)]\u00bb"
_ABBREVIATIONS = {"fig.", "eq."}
_INITIALISM = re.compile(r"(?:[A-Za-z]\.){2,}")   # e.g.  i.e.  U.S.  a.m.


def _ends_sentence(ws: list[Caption]) -> bool:
    """Does the last word close a sentence? Closing quotes and brackets may follow the stop ('."', '?"', '.)')."""
    core = ws[-1].text.rstrip(_CLOSERS)
    if not core or core[-1] not in ".!?\u2026":
        return False
    if core[-1] == "\u2026":
        return True
    low = core.lower()
    if low in _ABBREVIATIONS or _INITIALISM.fullmatch(core):
        return False
    return not (low == "al." and len(ws) > 1 and ws[-2].text.lower() == "et")


def pages(words: list[Caption], *, max_chars: int, max_lines: int = 2, pause_ms: int = 250, min_ms: int = 700,
          end_ms: int | None = None) -> list[Page]:
    """Break words into pages (after a sentence, after a pause, or when `max_lines` would overflow), then fit times.

    A page shorter than `min_ms` is extended, but every page ends no later than the next page starts and no later
    than `end_ms` (the film length). A page left with no length is dropped.
    """
    raw: list[Page] = []
    cur: list[Caption] = []
    for w in words:
        if cur:
            gap = w.startMs - cur[-1].endMs
            if gap > pause_ms or _ends_sentence(cur) or _line_count(cur + [w], max_chars) > max_lines:
                raw.append(_page(cur))
                cur = []
        cur.append(w)
    if cur:
        raw.append(_page(cur))
    out: list[Page] = []
    for i, p in enumerate(raw):
        end = max(p.endMs, p.startMs + min_ms)
        if i + 1 < len(raw):
            end = min(end, raw[i + 1].startMs)
        if end_ms is not None:
            end = min(end, end_ms)
        if end > p.startMs:
            out.append(p.model_copy(update={"endMs": end}))
    return out


def fit_max_chars(max_chars: int, *, width_px: int, size_px: float | None, plate: bool = False) -> int:
    """The pack's characters-per-line limit, lowered when the text box is too narrow for it at this font size."""
    if not isinstance(size_px, (int, float)) or isinstance(size_px, bool) or size_px <= 0:
        return max_chars
    inner = width_px * (1 - 2 * BOX_MARGIN) - (2 * PLATE_PAD_X_EM * size_px if plate else 0)
    return max(1, min(max_chars, int(inner / (size_px * GLYPH_EM) + 1e-9)))


def _ts(ms: int, sep: str) -> str:
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms_ = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms_:03d}"


def to_srt(ps: list[Page]) -> str:
    return "\n".join(f"{i}\n{_ts(p.startMs, ',')} --> {_ts(p.endMs, ',')}\n{p.text}\n" for i, p in enumerate(ps, 1))


def _vtt_text(text: str) -> str:
    """WebVTT cue text is markup: `&` and `<` must be escaped or a word like "R&D" corrupts the cue."""
    return text.replace("&", "&amp;").replace("<", "&lt;")


def to_vtt(ps: list[Page]) -> str:
    return "WEBVTT\n\n" + "\n".join(
        f"{_ts(p.startMs, '.')} --> {_ts(p.endMs, '.')}\n{_vtt_text(p.text)}\n" for p in ps)


def _colors(design: dict) -> dict:
    tokens = design.get("tokens") if isinstance(design, dict) else None
    colors = tokens.get("colors") if isinstance(tokens, dict) else None
    return colors if isinstance(colors, dict) else {}


def resolve_style(style: dict, design: dict) -> dict:
    """Map the style's colour roles (`ink`, `accent`, ...) to hex from the design's colour tokens.

    `color` and `highlight` are the text colours, `plate` the tint behind a page and `shadow` the shadow colour;
    anything that is not a role name (a literal such as `#fff` or `none`) is kept as written.
    """
    colors = _colors(design)
    out = dict(style)
    for key in ("color", "highlight", "plate", "shadow"):
        if isinstance(out.get(key), str) and out[key] in colors:
            out[key] = colors[out[key]]
    return out


_TRANSPARENT = re.compile(r"^(?:none|transparent|(?:rgba|hsla)\([^)]*[,/]\s*0*(?:\.0*)?\s*\)|#[0-9a-f]{6}00|#[0-9a-f]{3}0)$", re.I)


def _has_plate(value: object) -> bool:
    """A plate is set when it is a colour that can be seen; "none" and a fully transparent colour mean no plate."""
    return isinstance(value, str) and bool(value.strip()) and not _TRANSPARENT.match(value.strip())


def _write(path: Path, text: str) -> Path:
    with atomic_path(path) as tmp:
        tmp.write_text(text)
    return path


def _overlay_data(ps: list[Page], style: dict, band: list[float]) -> dict:
    box = {"margin": BOX_MARGIN, "plate_pad_x_em": PLATE_PAD_X_EM}
    return {"pages": [p.model_dump() for p in ps], "style": style, "band": band, "box": box}


def overlay_digest(ps: list[Page], timing: Timing, style: dict, band: list[float]) -> str:
    """Hash of everything the overlay render depends on: pages, style, band, layout, template, canvas and tool version."""
    return sha256_hex(HF_VERSION, OVERLAY_TEMPLATE.read_text(), json.dumps(_overlay_data(ps, style, band), sort_keys=True),
                      f"{timing.width}x{timing.height}@{timing.fps}", f"{timing.duration_s:.6f}")


def overlay_project(dest: Path, ps: list[Page], timing: Timing, style: dict, band: list[float]) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copytree(TEMPLATE_DIR / "assets", dest / "assets")
    html = stamp_root(OVERLAY_TEMPLATE.read_text(), "captions", duration=timing.duration_s, fps=timing.fps,
                      width=timing.width, height=timing.height)
    _write(dest / "index.html", html)
    _write(dest / "captions.json", json.dumps(_overlay_data(ps, style, band)))
    return dest


def _overlay_sidecar(overlay: Path) -> Path:
    return overlay.with_name(overlay.name + ".render.json")


def _overlay_is_current(overlay: Path, digest: str) -> bool:
    if not overlay.is_file():
        return False
    try:
        meta = json.loads(_overlay_sidecar(overlay).read_text())
    except (OSError, ValueError):
        return False
    return isinstance(meta, dict) and isinstance(meta.get("inputs"), dict) and meta["inputs"].get("hash") == digest


def _discard_overlay(overlay: Path) -> None:
    """Remove an overlay that no longer matches the captions: timeline.collect_media puts any that exists on V2."""
    overlay.unlink(missing_ok=True)
    _overlay_sidecar(overlay).unlink(missing_ok=True)


def _caption_style(run: RunPaths, pack: Pack) -> dict:
    from promo.scenes import design_for

    try:
        design = design_for(run, pack)
    except ValueError as e:   # JSONDecodeError and UnicodeDecodeError are both ValueErrors
        raise CaptionsError(f"design.json cannot be read ({type(e).__name__}); fix or delete it and run again") from e
    style = resolve_style(pack.captions.get("style", {}), design)
    if not _has_plate(style.get("plate")):
        style.pop("plate", None)
    return style


def build(run: RunPaths, timing: Timing, pack: Pack, *, render_fn: Callable = render) -> dict[str, Path]:
    cfg = pack.captions
    aspect = timing.aspect
    words = words_from_timing(timing)
    overlay = run.captions_dir / "overlay" / "captions.mov"
    overlaid = pack.caption_mode(aspect) == "subtitle" and bool(words)
    try:
        style = _caption_style(run, pack) if overlaid else {}
    except Exception:
        # Without the style the overlay's inputs cannot be known, so the one on disk can no longer be vouched for.
        _discard_overlay(overlay)
        raise
    max_chars = cfg.get("max_chars", {}).get(aspect, 32)
    if overlaid:   # the overlay is what the text must fit, so the font size and plate bound the line length
        max_chars = fit_max_chars(max_chars, width_px=timing.width, size_px=style.get("size_px"), plate="plate" in style)
    ps = pages(words, max_chars=max_chars, max_lines=cfg.get("max_lines", 2), pause_ms=cfg.get("pause_ms", 250),
               min_ms=cfg.get("min_ms", 700), end_ms=round(timing.duration_s * 1000))
    # Decide what happens to the overlay on disk before anything else is written, so a failure in the sidecars below
    # cannot leave an overlay that no longer matches the captions (timeline.collect_media puts any that exists on V2).
    # With nothing to caption (a silent film) an empty full-length overlay would cost gigabytes of frame storage, so
    # there is none.
    render_overlay = overlaid and bool(ps)
    keep = False
    if render_overlay:
        band = pack.tokens.get("caption_band", {}).get(aspect, [0.8, 0.94])
        digest = overlay_digest(ps, timing, style, band)
        keep = _overlay_is_current(overlay, digest)   # same inputs as the render on disk: keep it, as promo scenes does
    if not keep:
        _discard_overlay(overlay)
    run.captions_dir.mkdir(parents=True, exist_ok=True)
    out = {"words": run.captions_dir / "words.json", "srt": run.captions_dir / "captions.srt",
           "vtt": run.captions_dir / "captions.vtt"}
    _write(out["words"], json.dumps([w.model_dump() for w in words], indent=2))
    _write(out["srt"], to_srt(ps))
    _write(out["vtt"], to_vtt(ps))
    if not render_overlay:
        return out
    if keep:
        out["overlay"] = overlay
        return out
    project = overlay_project(run.captions_dir / "overlay" / "project", ps, timing, style, band)
    out["overlay"] = render_fn(project, overlay, fps=timing.fps, width=timing.width, height=timing.height,
                               duration=timing.duration_s, quality="high", fmt="mov", inputs={"hash": digest})
    return out
