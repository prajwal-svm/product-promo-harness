"""Determinism of a shot's STATE: the same inline styles and transforms must come back whatever the seek order.

Each capture carries, in its top rows, the stage's digest of the scene (see lib/stage.js, capture-settle): every
element's inline style and transform, and an audit of the timeline for overlapping tweens. `verify` compares the
digests of forward, reverse and one-by-one captures exactly and FAILS on any difference, naming the element, the
property and both values. Pixels are compared after the digest rows are masked out: when the state is identical but
the pixels differ, Chrome reused raster from an earlier frame, which is a WARNING, never a failure."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel

from promo.hyperframes import TEMPLATE_DIR, build_shot_project, snapshot
from promo.packs import Pack
from promo.proc import atomic_path
from promo.run import RunPaths
from promo.scenes import design_for, shot_hash
from promo.timing import Timing

QUALITY = "high"     # the hash is of what `promo scenes` renders for the master


class VerifyError(RuntimeError):
    """The shots cannot be verified (an unreadable design.json)."""


class MissingScenes(VerifyError):
    """Nothing to verify as asked: a shot that is not in timing.json, or one with no scene file or render project."""


class VerifyResult(BaseModel):
    """`mismatches` are the failures, each with a `kind` ("state", "overlap" or "pixels") and a `message`; `warnings` are
    pixel-only differences (and overlaps when allowed), which do not fail."""

    ok: bool
    shots: list[str]
    mismatches: list[dict]
    warnings: list[dict] = []
    fingerprints: dict[str, str] = {}     # shot id -> the hash `promo scenes` uses, so a later edit shows as stale


def verifiable_shots(timing: Timing) -> list[str]:
    """The HyperFrames shots that have at least one frame."""
    return [s.id for s in timing.shots if s.engine == "hyperframes"
            and round(s.end_s * timing.fps) - round(s.start_s * timing.fps) > 0]


def missing_scenes(run: RunPaths, timing: Timing, only: list[str] | None = None) -> list[str]:
    """The shots to verify that have neither a scene file nor a render project to snapshot."""
    return [i for i in verifiable_shots(timing) if (not only or i in only)
            and not (run.shots_dir / f"{i}.html").exists() and not (run.render_dir / i).exists()]


def pick_times(timing: Timing, shot_id: str, n: int = 6) -> list[float]:
    """Up to `n` local times in a shot: its first and last frame, its middle, and each event and 0.1 s after it."""
    shot = timing.shot(shot_id)
    last = round(shot.duration_s - 1 / timing.fps, 3)
    cands = [0.0, last, round(shot.duration_s / 2, 3)]
    for e in timing.events_in(shot_id):
        cands += [round(min(last, e.t_local), 3), round(min(last, e.t_local + 0.1), 3)]
    uniq = sorted({c for c in cands if 0 <= c <= last})
    if n <= 1:
        return uniq[len(uniq) // 2:][:1]
    if len(uniq) > n:
        step = (len(uniq) - 1) / (n - 1)
        uniq = [uniq[round(i * step)] for i in range(n)]
    return uniq


DIGEST_MAGIC = b"PSD1"


def frame_hash(png: Path, mask_rows: int = 0) -> str:
    """The hash of a capture's pixels, its first `mask_rows` rows (the stage's digest) left out."""
    pixels = np.asarray(Image.open(png).convert("RGBA")).copy()
    pixels[:mask_rows] = 0
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def frame_delta(a: Path, b: Path, mask_rows: int = 0) -> dict:
    """How two captures differ below the digest rows: the largest per-channel difference, the number of differing
    pixels, and the box [x0, y0, x1, y1] around them (None when they are the same)."""
    x = np.asarray(Image.open(a).convert("RGBA"), dtype=np.int16).copy()
    y = np.asarray(Image.open(b).convert("RGBA"), dtype=np.int16).copy()
    if x.shape != y.shape:
        return {"max_diff": 255, "pixels": max(x.shape[0] * x.shape[1], y.shape[0] * y.shape[1]), "region": None}
    x[:mask_rows] = 0
    y[:mask_rows] = 0
    d = np.abs(x - y)
    ys, xs = np.nonzero(d.sum(axis=2))
    region = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None
    return {"max_diff": int(d.max()), "pixels": int(len(xs)), "region": region}


def frame_diff(a: Path, b: Path, mask_rows: int = 0) -> tuple[int, int]:
    """(largest per-channel difference, number of differing pixels) between two PNGs."""
    delta = frame_delta(a, b, mask_rows)
    return delta["max_diff"], delta["pixels"]


def decode_digest(png: Path) -> tuple[dict | None, int]:
    """(the stage's payload, the number of top rows it occupies) from a capture; (None, 0) when the capture has none."""
    image = Image.open(png).convert("RGB")
    width, height = image.size
    head = np.asarray(image.crop((0, 0, width, 1)), dtype=np.uint8).reshape(-1).tobytes()
    if head[:4] != DIGEST_MAGIC:
        return None, 0
    size = int.from_bytes(head[4:8], "big")
    rows = math.ceil((8 + size) / (3 * width))
    if size <= 0 or rows > height:
        return None, 0
    flat = np.asarray(image.crop((0, 0, width, rows)), dtype=np.uint8).reshape(-1).tobytes()
    try:
        payload = json.loads(zlib.decompress(flat[8:8 + size]))
    except (zlib.error, ValueError):
        return None, 0
    return (payload, rows) if isinstance(payload, dict) and isinstance(payload.get("state"), list) else (None, 0)


def _by_time(paths: list[Path]) -> dict[float, Path]:
    out = {}
    for p in paths:
        m = re.search(r"at-([\d.]+)s", p.name)
        if m:
            out[round(float(m.group(1)), 3)] = p
    return out


def _style_props(style: str) -> dict[str, str]:
    """`a: 1; b: f(x; y)` -> {"a": "1", "b": "f(x; y)"}: the order GSAP wrote them in does not matter."""
    out, depth, cur = {}, 0, ""
    for ch in style + ";":
        if ch == ";" and depth == 0:
            if ":" in cur:
                name, _, value = cur.partition(":")
                out[name.strip()] = value.strip()
            cur = ""
            continue
        depth += (ch == "(") - (ch == ")")
        cur += ch
    return out


def _state_map(state: list) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for entry in state:
        path, style, transform = entry[0], entry[1], entry[2]
        key, n = path, 1
        while key in out:                       # two elements with one path (no ids): keep both
            n += 1
            key = f"{path}~{n}"
        props = _style_props(style)
        if transform:
            props["transform (attribute)"] = transform
        out[key] = {"props": props, "rect": entry[3] if len(entry) > 3 else None,
                    "matrix": entry[4] if len(entry) > 4 else ""}
    return out


# --- comparing values: canonical, not exact --------------------------------------------------------------------------
#
# Two writings of one value are the same value, whatever GSAP or the browser printed. A value is canonicalised before it
# is compared: whitespace and case are folded; numbers are compared at 4 decimals (two numbers are the same when they
# differ by no more than 0.0001, so 5.64341deg and 5.6434deg are one value and the way the rounding falls cannot make
# them two); a zero has no unit; a transform is written in one notation (`scale(1.11)` is `scale(1.11, 1.11)`,
# `translate(0px, 30px)` is `translate3d(0px, 30px, 0px)`, `rotateZ` is `rotate`); and a transform that still differs
# is the same value when its computed matrix (the browser's own canonical form, 6 significant digits) is the same to
# 1e-5. This is canonicalisation of notation and of print precision, not a tolerance on pixels: 0.0001 of a degree or a
# pixel is below anything a frame can show, and a difference of 0.001 or more, or a different unit, still fails.

NUMBER_TOLERANCE = 1e-4
MATRIX_TOLERANCE = 1e-5
_NUMBER_UNIT = re.compile(r"(?<![\w.])([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)([a-z%]*)")
_FUNCTION = re.compile(r"([a-z0-9]+)\(([^()]*)\)")


def _skeleton(value: str) -> tuple[str, list[float]]:
    """(the value with each number replaced by `#` and its unit kept, the numbers); a zero (to 4 decimals) keeps no unit."""
    numbers: list[float] = []

    def take(m: re.Match) -> str:
        number = float(m.group(1))
        numbers.append(number)
        return "#" if abs(number) <= NUMBER_TOLERANCE else "#" + m.group(2)

    folded = re.sub(r"\s*([,()])\s*", r"\1", re.sub(r"\s+", " ", value.strip().lower()))
    return _NUMBER_UNIT.sub(take, folded), numbers


def _norm_transform(value: str) -> str:
    """A transform list in one notation: translate, scale and rotate in their longest forms, the rest as written."""
    out, pos = [], 0
    text = value.strip().lower()
    for m in _FUNCTION.finditer(text):
        if text[pos:m.start()].strip():
            return text                                   # not a plain list of functions: compare it as written
        pos = m.end()
        name, args = m.group(1), [a.strip() for a in m.group(2).split(",")]
        if name in ("translate", "translate3d", "translatex", "translatey", "translatez"):
            x = y = z = "0px"
            if name == "translatex":
                x = args[0]
            elif name == "translatey":
                y = args[0]
            elif name == "translatez":
                z = args[0]
            else:
                x, y, z = (args + ["0px", "0px"])[:3] if name == "translate3d" else (args[0], (args + ["0px"])[1], "0px")
            out.append(f"translate3d({x}, {y}, {z})")
        elif name in ("scale", "scale3d", "scalex", "scaley", "scalez"):
            x = y = z = "1"
            if name == "scalex":
                x = args[0]
            elif name == "scaley":
                y = args[0]
            elif name == "scalez":
                z = args[0]
            elif name == "scale3d":
                x, y, z = (args + ["1", "1"])[:3]
            else:
                x = args[0]
                y = args[1] if len(args) > 1 else args[0]
            out.append(f"scale3d({x}, {y}, {z})")
        elif name in ("rotate", "rotatez"):
            out.append(f"rotate({args[0]})")
        else:
            out.append(f"{name}({', '.join(args)})")
    return text if text[pos:].strip() or not out else " ".join(out)


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?", text)]


def _same_numbers(a: str, b: str) -> bool:
    (sa, na), (sb, nb) = _skeleton(a), _skeleton(b)
    return sa == sb and len(na) == len(nb) and all(abs(p - q) <= NUMBER_TOLERANCE + 1e-12 for p, q in zip(na, nb))


def _same_value(prop: str, forward: str | None, reverse: str | None, m_forward: str = "", m_reverse: str = "") -> bool:
    """Whether two inline values (None: not set) are one value once canonicalised (see above)."""
    if forward == reverse:
        return True
    if forward is None or reverse is None:
        return False
    if prop in ("transform", "transform (attribute)"):          # not transform-origin, -style or -box: a matrix cannot vouch for those
        if _same_numbers(_norm_transform(forward), _norm_transform(reverse)):
            return True
        a, b = _numbers(m_forward), _numbers(m_reverse)
        return bool(a) and len(a) == len(b) and all(abs(p - q) <= MATRIX_TOLERANCE for p, q in zip(a, b))
    return _same_numbers(forward, reverse)


HINT_UNSET = ("the element is there in both orders but this inline property is set in one only: give every animated "
              "property an explicit start value at build time, and never overlap two tweens on it")
HINT_DIFFER = ("values differ between the orders: a tween that reads its start from live state, or two tweens on one "
               "property; give every animated property an explicit start value (see the overlap audit)")
HINT_ELEMENT = "the element exists in one order only"
HINT_NO_ID = ("it has an inline style or transform in one order only; it has no id, so verify cannot tell whether the "
              "element exists in the other")


def _is_id_path(path: str) -> bool:
    return path.startswith("#") and ">" not in path and "~" not in path


def state_diffs(forward: list, reverse: list) -> list[dict]:
    """Every element and property whose inline value differs between two digests (after canonicalising), in document
    order, each with a `hint` naming the cause. Elements with an id are always in a digest, so one missing from a digest
    is really missing; an element without an id is only listed when it has an inline style or transform."""
    fw, rv = _state_map(forward), _state_map(reverse)
    out = []
    for key in [*fw, *[k for k in rv if k not in fw]]:
        a, b = fw.get(key), rv.get(key)
        if a is None or b is None:
            there = a or b
            if _is_id_path(key):
                out.append({"element": key, "property": "(the element)", "forward": "present" if a else "absent",
                            "reverse": "present" if b else "absent", "hint": HINT_ELEMENT})
            else:
                style = "; ".join(f"{k}: {v}" for k, v in there["props"].items()) or "(empty)"
                out.append({"element": key, "property": "(inline style)", "forward": style if a else None,
                            "reverse": style if b else None, "hint": HINT_NO_ID})
            continue
        for prop in [*a["props"], *[p for p in b["props"] if p not in a["props"]]]:
            x, y = a["props"].get(prop), b["props"].get(prop)
            if not _same_value(prop, x, y, a["matrix"], b["matrix"]):
                out.append({"element": key, "property": prop, "forward": x, "reverse": y,
                            "hint": HINT_UNSET if x is None or y is None else HINT_DIFFER})
    return out


def _layer_at(state: list, region: list[int] | None) -> str | None:
    """The smallest element whose box holds the centre of `region`: the layer Chrome reused raster on."""
    if not region:
        return None
    cx, cy = (region[0] + region[2]) / 2, (region[1] + region[3]) / 2
    best = None
    for entry in state:
        rect = entry[3] if len(entry) > 3 else None
        if rect and rect[0] <= cx <= rect[0] + rect[2] and rect[1] <= cy <= rect[1] + rect[3]:
            area = rect[2] * rect[3]
            if best is None or area < best[0]:
                best = (area, entry[0])
    return best[1] if best else None


def _show(value: str | None) -> str:
    return "(unset)" if value is None else value


SITE_LINE = re.compile(r"^L(\d+): (.*)$")
TRAILING_COMMENT = re.compile(r"\s*//[^\n]*$")


def _site_key(text: str) -> str:
    """A line as the served page may have it: HyperFrames strips trailing `// comments` from inline scripts."""
    return TRAILING_COMMENT.sub("", text.strip())[:160]


def _source_lines(project: Path, shot_id: str) -> dict[str, list[str]]:
    """The text of the files a shot's code can come from, by the name to show: its page (the scene file) and lib/*.js."""
    files: dict[str, list[str]] = {}
    page = project / "index.html"
    if page.exists():
        files[f"scenes/shots/{shot_id}.html"] = page.read_text(errors="replace").splitlines()
    for f in sorted((project / "lib").glob("*.js")) if (project / "lib").is_dir() else []:
        files[f"lib/{f.name}"] = f.read_text(errors="replace").splitlines()
    return files


def resolve_sites(audit: list[dict], sources: dict[str, list[str]]) -> list[dict]:
    """The audit's tween sites (each a list of frames, innermost caller first) with file and line. HyperFrames serves a shot as one document with its scripts inlined,
    so the stage can only say "line 691 of the served page: <that line's text>" (`L691: ...`). That text is looked up in
    the shot's own files; a text found once is exact, and the offset between served and source lines (the same for
    every line of one file) settles a text found in several places. A line that cannot be placed stays "line N"."""
    index: dict[str, list[tuple[str, int]]] = {}
    for label, lines in sources.items():
        for number, line in enumerate(lines, 1):
            text = _site_key(line)
            if len(text) >= 8:
                index.setdefault(text, []).append((label, number))
    parsed = []                                          # per site: its parts as (served line, text) or plain strings
    offsets: dict[str, int] = {}
    for o in audit:
        for tween in (o["a"], o["b"]):
            parts = []
            for part in _site_parts(tween):
                m = SITE_LINE.match(part)
                parts.append((int(m.group(1)), _site_key(m.group(2))) if m else part)
            parsed.append((tween, parts))
            for part in parts:
                if isinstance(part, tuple) and len(index.get(part[1], [])) == 1:
                    label, number = index[part[1]][0]
                    offsets[label] = part[0] - number

    def place(part) -> str:
        if not isinstance(part, tuple):
            return part
        served, text = part
        found = index.get(text, [])
        if len(found) > 1:
            found = [c for c in found if offsets.get(c[0]) == served - c[1]] or found
        return f"{found[0][0]}:{found[0][1]}" if found else f"line {served} of the served page"

    out = []
    for o in audit:
        pair = {}
        for key in ("a", "b"):
            tween, parts = next(x for x in parsed if x[0] is o[key])
            pair[key] = {**tween, "site": [place(p) for p in parts if p]}
        out.append({**o, **pair})
    return out


def _site_parts(tween: dict) -> list[str]:
    site = tween.get("site") or []
    return [site] if isinstance(site, str) and site else list(site)


def _site(tween: dict) -> str:
    return f" ({' < '.join(_site_parts(tween))})" if _site_parts(tween) else ""


def _overlap_record(shot: str, o: dict) -> dict:
    a, b = o["a"], o["b"]
    return {"kind": "overlap", "shot": shot, "target": o["target"], "property": o["property"], "a": a, "b": b,
            "message": (f"{shot}: {o['target']} {o['property']} is tweened twice at once, {a['start']:.3f}-{a['end']:.3f}s"
                        f"{_site(a)} and {b['start']:.3f}-{b['end']:.3f}s{_site(b)}; HyperFrames ends each seek with "
                        f"a backward step, so the earlier one wins in every capture and the other never shows")}


def _design(run: RunPaths, pack: Pack) -> dict:
    try:
        return design_for(run, pack)
    except ValueError as e:  # JSONDecodeError and UnicodeDecodeError are both ValueErrors
        raise VerifyError(f"design.json cannot be read ({type(e).__name__}); fix or delete it and run again") from e


def _fingerprint(run: RunPaths, timing: Timing, shot_id: str, design: dict) -> str | None:
    try:
        return shot_hash(run.scenes, timing, shot_id, design, QUALITY, timing.fps)
    except OSError:  # the scene file is gone
        return None


def stale_shots(run: RunPaths, timing: Timing, pack: Pack, recorded: dict[str, str]) -> list[str]:
    """The shots whose timing, scene or design changed since `verify` recorded their fingerprint."""
    design, current = _design(run, pack), set(verifiable_shots(timing))
    return [i for i, was in recorded.items() if i in current and was != _fingerprint(run, timing, i, design)]


def _capture_project(run: RunPaths, timing: Timing, shot_id: str, design: dict, work: Path) -> Path:
    """The shot's project for capture, in <work>/project (never in scenes/.render, which a running `promo scenes`
    renders from): built fresh from its scene file, else a copy of the project an earlier render left behind.

    The stage layer's capture-settle hook is switched on in it (design.json: capture.settle), and its stage.js is the
    template's, which has the hook: Chrome reuses paint and raster work across seeks, so without the hook a frame's
    pixels depend on the order frames were captured in. The project's lib/ is hard-linked to the scene's, so stage.js
    is replaced, not written through. Renders never get the flag: they capture forward and would pay its timeout on
    every frame.
    """
    project = work / "project"
    scene = run.shots_dir / f"{shot_id}.html"
    if scene.exists():
        built = build_shot_project(run.scenes, work / "build", timing, shot_id, design)
        built.rename(project)
        (work / "build").rmdir()
    elif (run.render_dir / shot_id).exists():
        shutil.copytree(run.render_dir / shot_id, project, copy_function=shutil.copy)
    else:
        return project                                  # nothing to arm: the snapshot will say what is missing
    stage = project / "lib" / "stage.js"
    if stage.exists():
        stage.unlink()
        shutil.copyfile(TEMPLATE_DIR / "lib" / "stage.js", stage)
    design_file = project / "design.json"
    armed = json.loads(design_file.read_text()) if design_file.exists() else {}
    armed["capture"] = {**armed.get("capture", {}), "settle": True}
    design_file.write_text(json.dumps(armed))
    return project


def _compare_shot(shot_id: str, times: list[float], a: dict, b: dict, c: dict,
                  sources: dict[str, list[str]] | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """(failures, warnings, overlaps) for one shot from its forward (a), reverse (b) and one-by-one (c) captures."""
    bad, warn, overlaps = [], [], []
    audit_done = False
    for t in times:
        key = round(t, 3)
        if key not in a or key not in b:
            bad.append({"kind": "pixels", "shot": shot_id, "t_local": key, "reason": "no frame was captured",
                        "message": f"{shot_id} at {key}s: no frame was captured"})
            continue
        captures = [("forward", a[key]), ("reverse", b[key])] + ([("single", c[key])] if key in c else [])
        decoded = [(name, path, *decode_digest(path)) for name, path in captures]
        digests = [d for d in decoded if d[2] is not None]
        mask = max(d[3] for d in decoded)
        if not audit_done and digests:
            audit_done = True
            overlaps = [_overlap_record(shot_id, o) for o in resolve_sites(digests[0][2].get("audit", []), sources or {})]
        if not digests:                                  # no stage in this shot: the pixels are all there is to go on
            hashes = [frame_hash(p) for _, p in captures]
            if len(set(hashes)) != 1:
                other = captures[hashes.index(next(h for h in hashes if h != hashes[0]))][1]
                delta = frame_delta(captures[0][1], other)
                bad.append({"kind": "pixels", "shot": shot_id, "t_local": key, "reason": "differs between seek orders",
                            "message": f"{shot_id} at {key}s: pixels differ between seek orders (this shot has no state "
                                       f"digest: it does not load PromoStage)", **delta})
            continue
        if len(digests) < len(decoded):
            missing = next(d[0] for d in decoded if d[2] is None)
            bad.append({"kind": "state", "shot": shot_id, "t_local": key, "reason": "no state digest",
                        "message": f"{shot_id} at {key}s: the {missing} capture has no state digest"})
            continue
        reference = decoded[0]
        found = None
        for name, path, payload, _ in decoded[1:]:
            diffs = state_diffs(reference[2]["state"], payload["state"])
            if diffs:
                found = (name, path, diffs)
                break
        if found:
            name, path, diffs = found
            first = diffs[0]
            more = f" (and {len(diffs) - 1} more)" if len(diffs) > 1 else ""
            bad.append({
                "kind": "state", "shot": shot_id, "t_local": key, "reason": "state differs between seek orders",
                "element": first["element"], "property": first["property"], "forward": first["forward"],
                "reverse": first["reverse"], "elements_differing": len({d["element"] for d in diffs}),
                "hint": first["hint"],
                "message": (f"{shot_id} at {key}s: {first['element']} {first['property']}: {_show(first['forward'])} "
                            f"forward vs {_show(first['reverse'])} {name}{more} — {first['hint']}"),
                **frame_delta(reference[1], path, mask)})
            continue
        hashes = [frame_hash(p, mask) for _, p in captures]
        if len(set(hashes)) != 1:
            other = captures[hashes.index(next(h for h in hashes if h != hashes[0]))][1]
            delta = frame_delta(captures[0][1], other, mask)
            layer = _layer_at(reference[2]["state"], delta["region"])
            warn.append({"kind": "pixels", "shot": shot_id, "t_local": key,
                         "reason": "DOM identical; Chrome reused raster from an earlier frame",
                         "layer": layer, **delta,
                         "message": (f"{shot_id} at {key}s: DOM identical; Chrome reused raster from an earlier frame"
                                     f"{' on ' + layer if layer else ''} (region {delta['region']}, largest difference "
                                     f"{delta['max_diff']} of 255 on {delta['pixels']} pixels)")})
    return bad, warn, overlaps


def verify(run: RunPaths, timing: Timing, pack: Pack, *, only: list[str] | None = None, n: int = 6,
           snap=snapshot, allow_overlaps: bool = False) -> VerifyResult:
    """Capture `n` frames of each HyperFrames shot three ways, in order, in reverse order and one by one, and compare
    them. The scene's STATE (the stage's digest in each capture) must be identical: any difference fails, naming the
    element, the property and both values. Pixels are compared below the digest; a difference with identical state is a
    warning. The stage also audits the timeline: two tweens or sets of one target and property whose times overlap
    fail too, unless `allow_overlaps` makes them warnings. Writes renders/verify.json; a shot's project and captures
    are kept under renders/verify/<shot>/ only when it fails or warns."""
    design = _design(run, pack)
    available = verifiable_shots(timing)
    unknown = sorted(set(only or []) - set(available))
    if unknown:
        raise MissingScenes(f"no HyperFrames shot with id {', '.join(unknown)} in timing.json")
    shots = [i for i in available if not only or i in only]
    mismatches: list[dict] = []
    warnings: list[dict] = []
    fingerprints: dict[str, str] = {}
    for shot_id in shots:
        work = run.renders / "verify" / shot_id
        shutil.rmtree(work, ignore_errors=True)      # a frame left by an earlier run must not be read as this one's
        project = _capture_project(run, timing, shot_id, design, work)
        times = pick_times(timing, shot_id, n)
        a = _by_time(snap(project, times, work / "a"))
        b = _by_time(snap(project, list(reversed(times)), work / "b"))
        c: dict[float, Path] = {}
        for i, t in enumerate(times[:3]):
            c.update(_by_time(snap(project, [t], work / f"c{i}")))
        bad, warn, overlaps = _compare_shot(shot_id, times, a, b, c, _source_lines(project, shot_id))
        (warnings if allow_overlaps else bad).extend(overlaps)
        mismatches += bad
        warnings += warn
        if not (bad or warn):
            shutil.rmtree(work, ignore_errors=True)
        if (fp := _fingerprint(run, timing, shot_id, design)) is not None:
            fingerprints[shot_id] = fp
    try:
        (run.renders / "verify").rmdir()
    except OSError:
        pass  # absent, or holding the captures of a failure or a warning
    res = VerifyResult(ok=not mismatches, shots=shots, mismatches=mismatches, warnings=warnings,
                       fingerprints=fingerprints)
    with atomic_path(run.renders / "verify.json") as tmp:
        tmp.write_text(res.model_dump_json(indent=2))
    return res
