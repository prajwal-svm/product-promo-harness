import json
import re
import shutil
import subprocess
from html.parser import HTMLParser

import pytest

from promo.config import packs_dir

TPL = packs_dir() / "_core" / "scene-template"


def test_template_obeys_the_render_contract():
    html = (TPL / "shots" / "_shot-basic.html").read_text()
    stage = (TPL / "lib" / "stage.js").read_text()
    # The capture-settle hook is the one place that may pace itself with frames and a timer; nothing animates with them.
    settle = re.search(r"// capture-settle:begin.*?// capture-settle:end", stage, flags=re.S).group(0)
    js = (TPL / "lib" / "motion.js").read_text() + stage.replace(settle, "")
    for banned in ("Math.random", "setTimeout", "setInterval", "requestAnimationFrame", "gsap.from(", "repeat: -1"):
        assert banned not in html + js, banned
    assert "requestAnimationFrame" in settle and "setInterval" not in settle and "Math.random" not in settle
    css = html + (TPL / "lib" / "finish.css").read_text()
    assert not re.search(r"\boutline\s*:", css)
    assert not re.search(r"transition\s*:", css)
    assert html.count("{{SHOT_ID}}") == 2
    assert html.index("window.__timelines[ID] = tl") > html.index("PromoStage.finish(tl, S)")


def test_timeline_is_registered_last_behind_a_guard_and_failures_are_logged():
    html = (TPL / "shots" / "_shot-basic.html").read_text()
    guard = "window.__timelines = window.__timelines || {};"
    register = "window.__timelines[ID] = tl;"
    assert html.index(guard) < html.index(register)
    # Nothing builds or touches the timeline after registration: only the catch block follows.
    after = html[html.index(register) + len(register):]
    assert "tl." not in after.split("} catch")[0] and "PromoStage." not in after.split("} catch")[0]
    assert 'console.error("promo shot " + ID + " failed: " + e.message);' in after
    assert "throw e;" in after
    # The catch block must be able to read ID, so the placeholder sits outside the try.
    assert html.index('const ID = "{{SHOT_ID}}";') < html.index("try {")


class _Tree(HTMLParser):
    """Records each element's parent, so a test can ask who contains whom."""

    VOID = {"meta", "link", "br", "img", "hr", "input"}

    def __init__(self):
        super().__init__()
        self.stack, self.parents = [], {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        key = a.get("id") or a.get("class") or tag
        self.parents[key] = self.stack[-1] if self.stack else None
        if tag not in self.VOID:
            self.stack.append(key)

    def handle_endtag(self, tag):
        if tag not in self.VOID and self.stack:
            self.stack.pop()


def test_grain_is_a_sibling_layer_that_can_blend_with_the_scene():
    html = (TPL / "shots" / "_shot-basic.html").read_text()
    css = (TPL / "lib" / "finish.css").read_text()
    tree = _Tree()
    tree.feed(html)
    # A blend mode only reaches the scene when its element is not inside another stacking context
    # (.finish has z-index, so it is one): the grain must be a direct child of #root.
    assert tree.parents["finish-grain"] == "root"
    assert tree.parents["finish"] == "root"
    grain = re.search(r"\.finish-grain\s*\{([^}]*)\}", css).group(1)
    assert re.search(r"mix-blend-mode\s*:\s*overlay", grain)
    assert re.search(r"z-index\s*:\s*1001", grain)
    vignette = re.search(r"\.finish\s*\{([^}]*)\}", css).group(1)
    assert "mix-blend-mode" not in vignette
    assert "var(--vignette, 0)" in vignette, "a clean frame by default: no vignette unless a pack sets one"
    assert "::after" not in css


def test_vendored_assets_present():
    assert (TPL / "assets" / "vendor" / "gsap.min.js").stat().st_size > 50_000
    assert (TPL / "assets" / "fonts" / "InterVariable.woff2").stat().st_size > 20_000
    assert "SIL OPEN FONT LICENSE" in (TPL / "assets" / "fonts" / "OFL.txt").read_text().upper()


STAGE = """
globalThis.window = globalThis;
const order = [], vars = {};
const face = (ok) => ({ load: async () => { order.push("start"); await Promise.resolve(); order.push("loaded"); if (!ok) throw new Error("x"); } });
const faces = [face(true), face(false)];
globalThis.document = {
  fonts: { [Symbol.iterator]: () => faces[Symbol.iterator](), get ready() { order.push("ready"); return Promise.resolve(); } },
  documentElement: { style: { setProperty: (k, v) => { vars[k] = v; } } },
  querySelector: (sel) => ({ sel }),
};
require(process.argv[1]);
require(process.argv[2]);
const timing = {
  aspect: "16:9", limits: { text_lag_s: 0.08 },
  chapters: [{ id: "c1", headline: { text: "Hi", key: "Hi" } }],
  shots: [{ id: "s1", chapter: "c1", start_s: 2, end_s: 3, text_mode: "paraphrase" }],
  events: [{ shot: "s1", kind: "card", t_local: 0.5 }, { shot: "s2", kind: "card", t_local: 0 }],
  lines: [{ id: "l1", words: [{ text: "a", startMs: 2100, endMs: 2300 }, { text: "b", startMs: 3100, endMs: 3200 }] }],
};
const design = { tokens: { colors: { bg: "#fff", bg_alt: "#eee", accent: "#0f0" }, type_scale: { "16:9": { headline_px: 88 } } } };
const res = (status, body) => ({ ok: status >= 200 && status < 300, status, json: async () => { if (body instanceof Error) throw body; return body; } });
async function attempt(timingRes, designRes) {
  globalThis.fetch = async (url) => (url === "timing.json" ? timingRes : designRes);
  try { return { S: await PromoStage.load("s1") }; } catch (e) { return { err: e.message }; }
}
(async () => {
  const good = await attempt(res(200, timing), res(200, design));
  const loaded = order.lastIndexOf("loaded");
  const out = {
    fontsLoadedBeforeReady: loaded >= 0 && order.indexOf("ready") > loaded,
    vars,
    words: good.S.words, events: good.S.events.length, duration: good.S.duration, chapter: good.S.chapter.id,
    noDesign: (await attempt(res(200, timing), res(404, null))).S.design,
    timing500: (await attempt(res(500, null), res(200, design))).err,
    timingMissingShot: (await attempt(res(200, { ...timing, shots: [] }), res(200, design))).err,
    badDesign: (await attempt(res(200, timing), res(200, new SyntaxError("bad json")))).err,
    design500: (await attempt(res(200, timing), res(500, null))).err,
  };
  const calls = [];
  PromoStage.finish({ set: (el, v, t) => calls.push([el.sel, v, t]) }, good.S, 7);
  const again = [];
  PromoStage.finish({ set: (el, v, t) => again.push([el.sel, v, t]) }, good.S, 7);
  out.finishTargets = [...new Set(calls.map((c) => c[0]))];
  out.finishCount = calls.length;
  out.finishDeterministic = JSON.stringify(calls) === JSON.stringify(again);
  out.finishVars = Object.keys(calls[0][1]);
  console.log(JSON.stringify(out));
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node needed")
def test_stage_loader_and_finish_behaviour():
    args = ["node", "-e", STAGE, str(TPL / "lib" / "motion.js"), str(TPL / "lib" / "stage.js")]
    o = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
    # Fonts are explicitly loaded (a failing face does not abort) before fonts.ready is awaited.
    assert o["fontsLoadedBeforeReady"] is True
    # Colour tokens are set under both spellings; type scale comes from the aspect.
    assert o["vars"]["--bg_alt"] == o["vars"]["--bg-alt"] == "#eee"
    assert o["vars"]["--bg"] == "#fff" and o["vars"]["--accent"] == "#0f0"
    assert o["vars"]["--headline-px"] == "88px"
    # Stage object: shot-local times, only this shot's events.
    assert o["events"] == 1 and o["duration"] == 1 and o["chapter"] == "c1"
    assert [w["text"] for w in o["words"]] == ["a"]
    assert o["words"][0]["t"] == pytest.approx(0.1) and o["words"][0]["end"] == pytest.approx(0.3)
    # A missing design.json is fine; a missing shot, a bad timing.json and a broken design.json are loud.
    assert o["noDesign"] == {}
    assert "timing.json" in o["timing500"] and "500" in o["timing500"]
    assert "s1" in o["timingMissingShot"]
    assert "design.json" in o["badDesign"] and "bad json" in o["badDesign"]
    assert "design.json" in o["design500"]
    # The grain layer is the thing that moves, and its jitter is seeded.
    assert o["finishTargets"] == [".finish-grain"]
    assert o["finishVars"] == ["--gx", "--gy"]
    assert 12 <= o["finishCount"] <= 13
    assert o["finishDeterministic"] is True


SETTLE = """
globalThis.window = globalThis;
const log = [];
const style = { setProperty() {} };
const el = (tag, id, parent, attrs = {}, rect = [0, 0, 10, 10], matrix = "none") => {
  const e = { nodeType: 1, tagName: tag.toUpperCase(), id, parentElement: parent, children: [], getAttribute: (n) => (n in attrs ? attrs[n] : null), matrix,
    getBoundingClientRect: () => ({ x: rect[0], y: rect[1], width: rect[2], height: rect[3] }) };
  if (parent) parent.children.push(e);
  return e;
};
const body = el("body", "", null);
const root = el("div", "root", body, { style: "visibility: visible; width: 1920px;" });
const box = el("div", "box", root, { style: "opacity: 0.5; transform: translate(1px, 2px);" }, [0, 0, 10, 10], "matrix(1, 0, 0, 1, 1, 2)");
const bare = el("span", "", box, { style: "color: red;" });
const svg = el("g", "", root, { transform: "rotate(3)" }, [0, 0, 10, 10], "matrix(0.99863, 0.052336, -0.052336, 0.99863, 0, 0)");
const plain = el("div", "plain", root);                                       // an id but no inline style: listed, empty
el("div", "", root);                                                          // none of the three: the selector never returns it
globalThis.getComputedStyle = (e) => ({ transform: e.matrix });
const canvases = [];
globalThis.ImageData = class { constructor(data, w, h) { this.data = data; this.width = w; this.height = h; } };
globalThis.document = {
  fonts: { [Symbol.iterator]: () => [][Symbol.iterator](), ready: Promise.resolve() },
  body,
  documentElement: { style, nodeType: 1, id: "", clientWidth: 64, get offsetWidth() { log.push("layout zoom=" + (style.zoom || "")); return 1; } },
  querySelectorAll: () => [globalThis.document.documentElement, root, box, bare, svg, plain, ...canvases.filter((c) => c.attached)],
  getElementById: (id) => canvases.find((c) => c.id === id && c.attached) || null,
  createElement: (tag) => { const c = { id: "", style: {}, remove() { c.attached = false; }, getContext: () => ({ putImageData: (img) => { c.img = img; } }) }; canvases.push(c); return c; },
};
body.appendChild = (c) => { c.attached = true; };
let frameCallbacks = 0;
globalThis.requestAnimationFrame = (cb) => { frameCallbacks++; log.push("frame zoom=" + (style.zoom || "")); Promise.resolve().then(cb); };
globalThis.setTimeout = (fn, ms) => { log.push("timer " + ms); return 1; };
globalThis.clearTimeout = () => {};
// a stand-in for GSAP's Timeline (the real one is exercised by the audit test and by the real-browser verify)
const proto = { recent() { return this._recent; } };
for (const m of ["to", "from", "fromTo", "set"]) {
  proto[m] = function (target, vars, at) {
    const tween = { vars, startTime: () => at, totalDuration: () => 1, targets: () => [target] };
    this.kids.push(tween);
    this._recent = tween;
    return this;
  };
}
globalThis.gsap = { core: { Timeline: { prototype: proto } } };
require(process.argv[1]);
require(process.argv[2]);
const timing = { aspect: "16:9", chapters: [{ id: "c" }], shots: [{ id: "s", chapter: "c", start_s: 0, end_s: 1 }], events: [], lines: [] };
const load = async (design, before) => {
  delete globalThis.__hfWaitForSeekCompletion;
  if (before) globalThis.__hfWaitForSeekCompletion = before;
  globalThis.fetch = async (url) => ({ ok: true, status: 200, json: async () => (url === "timing.json" ? timing : design) });
  await PromoStage.load("s");
};
const inflate = require("zlib").inflateSync;
const decode = (c) => {                                      // what verify does with the pixels
  const rgb = Buffer.alloc(c.img.width * c.img.height * 3);
  for (let i = 0, j = 0; i < c.img.data.length; i += 4, j += 3) { rgb[j] = c.img.data[i]; rgb[j + 1] = c.img.data[i + 1]; rgb[j + 2] = c.img.data[i + 2]; }
  const n = rgb.readUInt32BE(4);
  return { magic: rgb.subarray(0, 4).toString(), alpha: [...new Set(c.img.data.filter((_, i) => i % 4 === 3))], payload: JSON.parse(inflate(rgb.subarray(8, 8 + n)).toString()), rows: c.img.height };
};
(async () => {
  const out = {};
  await load({}, null);
  out.offByDefault = Object.getOwnPropertyDescriptor(globalThis, "__hfWaitForSeekCompletion") === undefined;
  out.offWithFlagFalse = (await load({ capture: { settle: false } }, null), globalThis.__hfWaitForSeekCompletion === undefined);

  // the host's hook is already there when the stage loads; a timeline is registered after load, as a shot does
  const calls = [];
  await load({ capture: { settle: true } }, async () => { calls.push("host"); });
  const tl = Object.create(proto);
  tl.kids = [];
  tl.getChildren = () => tl.kids;
  tl.duration = () => 2;
  const target = { name: "target" };
  tl.to(target, { x: 10 }, 0);
  tl.to(target, { x: 20 }, 0.5);                              // overlaps the first on x
  window.__timelines = { s: tl };
  log.length = 0; frameCallbacks = 0;
  await globalThis.__hfWaitForSeekCompletion();
  out.hostFirst = calls.slice();
  out.steps = log.slice();
  out.zoomRestored = style.zoom === undefined || style.zoom === "";
  out.frames = frameCallbacks;
  const first = decode(canvases[canvases.length - 1]);
  out.digest = { magic: first.magic, alpha: first.alpha, rows: first.rows, shot: first.payload.shot, state: first.payload.state, audit: first.payload.audit };
  const painted = canvases[canvases.length - 1].img;
  out.pixels = { width: painted.width, rows: painted.height, rgba: Buffer.from(painted.data).toString("base64") };
  
  // a second capture replaces the first digest; the digest is never part of the state it reports
  await globalThis.__hfWaitForSeekCompletion();
  out.oneDigestCanvas = canvases.filter((c) => c.attached).length;
  out.stateIgnoresDigest = decode(canvases[canvases.length - 1]).payload.state.every((e) => e[0] !== "#__promo_digest");

  // the host assigns its hook after the stage loaded: it is kept and chained, not overwritten
  const late = [];
  await load({ capture: { settle: true } }, null);
  window.__timelines = { s: tl };
  globalThis.__hfWaitForSeekCompletion = async () => { late.push("host"); };
  await globalThis.__hfWaitForSeekCompletion();
  out.lateHostRan = late;
  delete globalThis.__hfWaitForSeekCompletion;
  out.canBeDeleted = globalThis.__hfWaitForSeekCompletion === undefined;

  // two frames that never arrive: the guard fails the capture loudly, it never proceeds silently
  await load({ capture: { settle: true } }, null);
  const attached = canvases.filter((c) => c.attached).length;
  globalThis.requestAnimationFrame = () => {};
  globalThis.setTimeout = (fn) => { Promise.resolve().then(fn); return 1; };
  try { await globalThis.__hfWaitForSeekCompletion(); out.hang = "proceeded"; } catch (e) { out.hang = e.message; }
  out.hangPaintedNothing = canvases.filter((c) => c.attached).length === attached;
  console.log(JSON.stringify(out));
})();
"""


AUDIT = """
globalThis.gsap = require(process.argv[3]).gsap;      // before `window` exists, and no `document` at all: with both, GSAP treats plain objects as elements
globalThis.window = globalThis;
require(process.argv[1]);                     // (no document: with window and document both, GSAP treats plain objects as elements)
require(process.argv[2]);
const audit = PromoStage.capture.auditTimeline;
const T = (n) => ({ name: n });
const run = (build) => { const tl = gsap.timeline({ paused: true }); build(tl); return audit(tl).map((f) => [f.target, f.property, f.a.start, f.a.end, f.b.start, f.b.end]); };
const out = {};
out.none = run((tl) => { const a = T("a"); tl.to(a, { x: 1, duration: 1 }, 0); tl.to(a, { x: 2, duration: 1 }, 1); tl.to(a, { y: 2, duration: 1 }, 0.5); });
out.overlap = run((tl) => { const a = T("a"); tl.to(a, { x: 1, duration: 1 }, 0); tl.to(a, { x: 2, duration: 1 }, 0.5); });
out.otherTargetOrProperty = run((tl) => { tl.to(T("a"), { x: 1, duration: 1 }, 0); tl.to(T("b"), { x: 2, duration: 1 }, 0.5); tl.to(T("a"), { opacity: 0, duration: 1 }, 0.5); });
out.setInsideTween = run((tl) => { const a = T("a"); tl.fromTo(a, { alpha: 1 }, { alpha: 0, duration: 1 }, 0); tl.set(a, { alpha: 0.3 }, 0.5); });
out.setsAtTheEdgesAndApart = run((tl) => { const a = T("a"); tl.fromTo(a, { alpha: 1 }, { alpha: 0, duration: 1 }, 0); tl.set(a, { alpha: 1 }, 1); tl.set(a, { alpha: 0.5 }, 2); tl.set(a, { alpha: 0.6 }, 3); });
out.setsAtOneInstant = run((tl) => { const a = T("a"); tl.set(a, { x: 1 }, 2); tl.set(a, { x: 2 }, 2); });
out.aliases = run((tl) => { const a = T("a"); tl.to(a, { scale: 2, duration: 1 }, 0); tl.to(a, { scale: 3, duration: 1 }, 0.5); tl.to(a, { rotation: 5, duration: 1 }, 4); tl.to(a, { rotate: 9, duration: 1 }, 4.5); });
out.nested = run((tl) => { const a = T("a"); const inner = gsap.timeline(); inner.to(a, { x: 1, duration: 1 }, 0); tl.add(inner, 2); tl.to(a, { x: 5, duration: 1 }, 2.5); });
// a shot plays a window of a film-long timeline: tweens outside the window do not count, the others move to where it plays
out.window = run((tl) => {
  const world = gsap.timeline({ paused: true }); const a = T("a");
  world.to(a, { x: 1, duration: 4 }, 0); world.to(a, { x: 2, duration: 4 }, 2);            // overlap 2-4 in world time
  world.to(a, { y: 1, duration: 1 }, 20); world.to(a, { y: 2, duration: 1 }, 20.5);        // overlap far outside the window
  tl.add(world.tweenFromTo(1, 5, { ease: "none" }), 0);
});
out.attrs = run((tl) => { const a = T("a"); tl.to(a, { attr: { href: "x" }, duration: 1 }, 0); tl.to(a, { attr: { width: 5 }, duration: 1 }, 0.5); });
out.sameAttr = run((tl) => { const a = T("a"); tl.to(a, { attr: { href: "x" }, duration: 1 }, 0); tl.to(a, { attr: { href: "y", width: 5 }, duration: 1 }, 0.5); });
out.css = run((tl) => { const a = T("a"); tl.to(a, { css: { x: 1, scale: 2 }, duration: 1 }, 0); tl.to(a, { x: 5, duration: 1 }, 0.5); });
out.cssClean = run((tl) => { const a = T("a"); tl.to(a, { css: { x: 1 }, duration: 1 }, 0); tl.to(a, { y: 5, duration: 1 }, 0.5); });
// keyframes: an array runs one keyframe after the next, each on its own properties
out.keyframesArray = run((tl) => { const a = T("a");
  tl.to(a, { keyframes: [{ x: 10, duration: 1 }, { y: 5, duration: 1 }] }, 0);
  tl.to(a, { y: 1, duration: 0.5 }, 0.2);        // y is only tweened by the second keyframe, 1-2 s: no clash
  tl.to(a, { x: 1, duration: 0.5 }, 0.2);        // x is tweened by the first keyframe, 0-1 s: clash
  tl.to(a, { y: 2, duration: 0.5 }, 1.5); });    // and y in 1-2 s: clash
out.keyframesPercent = run((tl) => { const a = T("a");
  tl.to(a, { keyframes: { "0%": { x: 1 }, "50%": { y: 2 }, "100%": { x: 3 } }, duration: 2 }, 5);
  tl.to(a, { x: 9, duration: 0.5 }, 6);          // x runs 0%-100%: 5-7 s: clash
  tl.to(a, { y: 9, duration: 0.5 }, 6.5); });    // y is only at 50%: from the start to 6 s, so 6.5 is clear
out.keyframesObject = run((tl) => { const a = T("a");
  tl.to(a, { keyframes: { x: [0, 10, 0], y: [1, 2], easeEach: "none" }, duration: 2 }, 8);
  tl.to(a, { y: 9, duration: 0.5 }, 9); });      // y runs the whole tween: clash
out.ignoresReservedKeys = run((tl) => { const a = T("a"); tl.to(a, { x: 1, duration: 1, ease: "none", delay: 0, onUpdate() {} }, 0); tl.to(a, { y: 1, duration: 1, ease: "power2.out", onComplete() {} }, 0.5); });
console.log(JSON.stringify(out));
"""


GSAP = packs_dir() / "_core" / "scene-template" / "assets" / "vendor" / "gsap.min.js"


def _node(script):
    args = ["node", "-e", script, str(TPL / "lib" / "motion.js"), str(TPL / "lib" / "stage.js"), str(GSAP)]
    return json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node needed")
def test_capture_settle_hook_is_opt_in_chains_to_the_host_hook_waits_for_frames_and_paints_the_state(tmp_path):
    o = _node(SETTLE)
    # Without design.json's capture.settle the stage leaves the host's hook alone (a render must not pay for it).
    assert o["offByDefault"] is True and o["offWithFlagFalse"] is True
    # With it: the host's own hook runs first, then the page is relaid out with the zoom nudged, held for two frames,
    # restored, and held for two more; each wait has only a 2 s hang guard (no timer race); the zoom ends where it began.
    assert o["hostFirst"] == ["host"]
    assert o["steps"] == ["layout zoom=1.001", "timer 2000", "frame zoom=1.001", "frame zoom=1.001",
                          "layout zoom=", "timer 2000", "frame zoom=", "frame zoom="]
    assert o["zoomRestored"] is True and o["frames"] == 4
    # The digest: opaque RGB pixels, "PSD1" first, holding the shot, every element with an inline style, a transform
    # attribute or an id (not the root element, not an element with none of them, not the digest itself), each with its
    # computed transform when it has a transform, and the timeline audit.
    d = o["digest"]
    assert d["magic"] == "PSD1" and d["alpha"] == [255] and d["shot"] == "s" and d["rows"] >= 1
    assert d["state"] == [["#root", "visibility: visible; width: 1920px;", "", [0, 0, 10, 10], ""],
                          ["#box", "opacity: 0.5; transform: translate(1px, 2px);", "", [0, 0, 10, 10], "matrix(1, 0, 0, 1, 1, 2)"],
                          ["#box>span:0", "color: red;", "", [0, 0, 10, 10], ""],
                          ["#root>g:1", "", "rotate(3)", [0, 0, 10, 10], "matrix(0.99863, 0.052336, -0.052336, 0.99863, 0, 0)"],
                          ["#plain", "", "", [0, 0, 10, 10], ""]]
    assert d["audit"] == [{"target": "target", "property": "x", "a": {"start": 0, "end": 1, "site": d["audit"][0]["a"]["site"]},
                           "b": {"start": 0.5, "end": 1.5, "site": d["audit"][0]["b"]["site"]}}]
    assert o["oneDigestCanvas"] == 1 and o["stateIgnoresDigest"] is True
    # What the page painted is what `promo verify` reads back from the screenshot, pixel for pixel.
    import base64

    from PIL import Image

    from promo.verify import decode_digest

    px = o["pixels"]
    image = Image.new("RGBA", (px["width"], px["rows"] + 5), (0, 0, 0, 255))
    image.paste(Image.frombytes("RGBA", (px["width"], px["rows"]), base64.b64decode(px["rgba"])), (0, 0))
    image.save(tmp_path / "capture.png")
    payload, rows = decode_digest(tmp_path / "capture.png")
    assert rows == px["rows"] == d["rows"] and payload["shot"] == "s" and payload["state"] == d["state"]
    assert payload["audit"] == d["audit"]
    # A hook the host assigns later is chained too, and the property can still be deleted by the host.
    assert o["lateHostRan"] == ["host"] and o["canBeDeleted"] is True
    # Two frames that never arrive fail the capture, loudly, and paint nothing.
    assert o["hang"].startswith("capture-settle: no animation frame arrived in 2000 ms") and o["hangPaintedNothing"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="node needed")
def test_the_overlap_audit_finds_tweens_of_one_target_and_property_that_overlap_in_time():
    o = _node(AUDIT)
    assert o["none"] == [] and o["otherTargetOrProperty"] == [] and o["ignoresReservedKeys"] == []
    assert o["overlap"] == [["a", "x", 0, 1, 0.5, 1.5]]
    # a set inside a tween is a pop mid-animation; at the tween's edges or apart it is not
    assert o["setInsideTween"] == [["a", "alpha", 0, 1, 0.5, 0.5]] and o["setsAtTheEdgesAndApart"] == []
    assert o["setsAtOneInstant"] == [["a", "x", 2, 2, 2, 2]]
    # scale is scaleX and scaleY, rotate is rotation: a clash is one finding, not two
    assert sorted(o["aliases"]) == sorted([["a", "scaleX, scaleY", 0, 1, 0.5, 1.5], ["a", "rotation", 4, 5, 4.5, 5.5]])
    assert o["nested"] == [["a", "x", 2, 3, 2.5, 3.5]]
    # attr tweens of different attributes are different properties; the same attribute clashes; css:{} is read for its
    # real properties (x and scale here), so a plain tween of one of them clashes
    assert o["attrs"] == [] and o["sameAttr"] == [["a", "attr:href", 0, 1, 0.5, 1.5]]
    assert o["cssClean"] == [] and o["css"] == [["a", "x", 0, 1, 0.5, 1.5]]
    # keyframes tweens are read into their real properties and the time each one is tweened
    assert o["keyframesArray"] == [["a", "x", 0, 1, 0.2, 0.7], ["a", "y", 1, 2, 1.5, 2]]
    assert o["keyframesPercent"] == [["a", "x", 5, 7, 6, 6.5]]
    assert o["keyframesObject"] == [["a", "y", 8, 10, 9, 9.5]]
    # the film-long timeline's tweens at 0-4 s and 2-6 s overlap; the shot plays its 1-5 s window of it, where they sit
    # at 0-3 s and 1-4 s on the shot's own clock; the pair far outside the window (at 20 s) is not this shot's
    assert o["window"] == [["a", "x", 0, 3, 1, 4]]
