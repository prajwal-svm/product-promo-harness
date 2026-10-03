/* PromoStage: loads timing.json/design.json for one shot and offers small, deterministic helpers. */
(function () {
  // A missing optional file (HTTP 404) is {}; any other failure, including bad JSON, is an error.
  async function fetchJson(url, optional) {
    const res = await fetch(url);
    if (optional && res.status === 404) return {};
    if (!res.ok) throw new Error(url + ": HTTP " + res.status);
    try {
      return await res.json();
    } catch (e) {
      throw new Error(url + " is not valid JSON: " + e.message);
    }
  }

  // capture-settle:begin
  // The one place this file touches requestAnimationFrame and setTimeout. They only pace a capture, and the timer is
  // only a hang guard; no animation reads either, so the no-rAF rule (which is about timelines) does not apply here.
  //
  // `promo verify` writes design.json capture.settle into its own project, never into a render's. With it on, before
  // every capture the hook (HyperFrames 0.8.98 awaits window.__hfWaitForSeekCompletion in `snapshot` and `render`):
  //   1. chains to the host's own hook;
  //   2. forces a full relayout and repaint (a root zoom nudge), waits two frames, restores, waits two more, so Chrome
  //      does not reuse paint and raster work from an earlier frame (without it the pixels depend on capture order);
  //   3. reads the scene's STATE: every element with an inline style or transform attribute and that string, plus an
  //      audit of the registered timeline for tweens that overlap on one target and property;
  //   4. paints that, deflated, into the top rows of the page as raw RGB pixels (a 1:1 canvas), so it travels in the
  //      screenshot; verify reads it back, masks those rows out of every pixel comparison, and compares the STATE of
  //      the forward and reverse captures exactly.
  // If two animation frames do not arrive within SETTLE_HANG_MS the hook throws, and the capture fails loudly.
  const SETTLE_KEY = "__hfWaitForSeekCompletion";
  const SETTLE_HANG_MS = 2000;
  const DIGEST_ID = "__promo_digest";
  const AUDIT_EPS = 1e-6;
  const AUDIT_RESERVED = new Set([
    "duration", "ease", "delay", "overwrite", "parent", "stagger", "repeat", "repeatDelay", "yoyo", "yoyoEase",
    "repeatRefresh", "paused", "reversed", "data", "id", "lazy", "immediateRender", "runBackwards", "startAt", "inherit",
    "callbackScope", "onStart", "onUpdate", "onComplete", "onRepeat", "onReverseComplete", "onStartParams",
    "onUpdateParams", "onCompleteParams", "onRepeatParams", "onReverseCompleteParams", "preserveAlpha", "time", "easeEach",
  ]);
  const AUDIT_ALIASES = { rotate: ["rotation"], rotationZ: ["rotation"], scale: ["scaleX", "scaleY"], autoAlpha: ["opacity", "visibility"] };

  function pathOf(el) {
    const parts = [];
    for (let e = el; e && e.nodeType === 1 && e !== document.body && e !== document.documentElement; e = e.parentElement) {
      if (e.id) { parts.unshift("#" + e.id); break; }
      const p = e.parentElement;
      parts.unshift(e.tagName.toLowerCase() + ":" + (p ? Array.prototype.indexOf.call(p.children, e) : 0));
    }
    return parts.join(">");
  }

  // Every element with an inline style, a transform attribute or an id (what GSAP writes, and what a scene names), in
  // document order: [path, inline style, transform attribute, box, computed transform]. Elements with an id are always
  // listed, so an element that is missing from one order's list is really missing, not just without an inline style.
  function stateOf() {
    const out = [];
    for (const el of document.querySelectorAll("[style], [transform], [id]")) {
      if (el.id === DIGEST_ID || el === document.documentElement) continue;
      const style = el.getAttribute("style") || "";
      const transform = el.getAttribute("transform") || "";
      const r = el.getBoundingClientRect(); // only to name the layer when pixels differ with the state the same
      let matrix = "";
      if (transform || /transform/.test(style)) matrix = (getComputedStyle(el).transform || "").replace(/^none$/, "");
      out.push([pathOf(el), style, transform,
        [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)], matrix]);
    }
    return out;
  }

  // The property names a vars object animates. `css: {x: 1}` is `x`; `attr: {href: ...}` is one property PER attribute
  // (`attr:href`), so tweens of different attributes do not clash; `keyframes` is read by propRanges.
  function addKeys(vars, out) {
    for (const k of Object.keys(vars || {})) {
      if (k === "css") addKeys(vars.css, out);
      else if (k === "attr") { for (const a of Object.keys(vars.attr || {})) out.add("attr:" + a); }
      else if (k !== "keyframes" && !AUDIT_RESERVED.has(k)) { for (const p of AUDIT_ALIASES[k] || [k]) out.add(p); }
    }
  }

  // [property, from, to] for a tween of length `d`, `from` and `to` in seconds into the tween. Most properties last as
  // long as the tween. A keyframes tween is read into its real properties: an array of keyframes runs one after the
  // next (a keyframe's own `duration` and `delay`, the rest of the tween shared out), percentage keyframes run from the
  // first percentage that names the property (from the start, when it is the only one) to the last, and an object of
  // arrays (`{x: [0, 10, 0]}`) runs the whole tween.
  function propRanges(tween, d) {
    const vars = tween.vars || {};
    const names = new Set();
    addKeys(vars, names);
    addKeys(vars.startAt, names);
    const ranges = new Map([...names].map((p) => [p, [0, d]]));
    const kf = vars.keyframes;
    const keyed = new Map();            // what the keyframes say about a property replaces "the whole tween"
    const widen = (p, from, to) => { const r = keyed.get(p); keyed.set(p, r ? [Math.min(r[0], from), Math.max(r[1], to)] : [from, to]); };
    if (Array.isArray(kf)) {
      const explicit = kf.reduce((sum, f) => sum + (typeof f.duration === "number" ? f.duration : 0), 0);
      const open = kf.filter((f) => typeof f.duration !== "number").length;
      const each = open ? Math.max(0, d - explicit) / open : 0;
      let t = 0;
      for (const f of kf) {
        const from = t + (typeof f.delay === "number" ? f.delay : 0);
        const to = from + (typeof f.duration === "number" ? f.duration : each);
        const keys = new Set();
        addKeys(f, keys);
        for (const p of keys) widen(p, from, to);
        t = to;
      }
    } else if (kf && typeof kf === "object") {
      const marks = Object.keys(kf).filter((k) => /%$/.test(k));
      if (marks.length) {
        const where = new Map();
        for (const m of marks) {
          const keys = new Set();
          addKeys(kf[m], keys);
          for (const p of keys) where.set(p, (where.get(p) || []).concat(parseFloat(m)));
        }
        for (const [p, at] of where) widen(p, (at.length > 1 ? Math.min(...at) : 0) / 100 * d, Math.max(...at) / 100 * d);
      } else {
        const keys = new Set();
        addKeys(kf, keys);
        for (const p of keys) widen(p, 0, d);
      }
    }
    for (const [p, r] of keyed) ranges.set(p, r);
    return [...ranges].map(([p, r]) => [p, r[0], r[1]]);
  }

  // Every tween and set under a timeline as {target, property, start, end, site} on one clock (the shot's own, in
  // seconds). A child timeline sits where it was added. A tween that plays another timeline over a window
  // (`world.tweenFromTo(from, to)`, how a shot plays its part of a film-long timeline) brings that timeline's tweens
  // in, clipped to the window and moved to where the window plays.
  function collect(tl, base, scale, out) {
    for (const c of tl.getChildren(false, true, true)) {
      const start = base + c.startTime() * scale;
      if (typeof c.getChildren === "function") {
        collect(c, start, scale / (c.timeScale() || 1), out);
        continue;
      }
      let d = c.totalDuration() * scale;
      if (!isFinite(d)) d = Math.max(0, tl.duration() * scale - (start - base));
      const to = c.vars && typeof c.vars.time === "number" ? c.vars.time : null;
      const from = to === null ? 0 : (c.vars.startAt && typeof c.vars.startAt.time === "number" ? c.vars.startAt.time : 0);
      for (const target of c.targets()) {
        if (target && typeof target.getChildren === "function") {
          if (to === null || to <= from) { collect(target, start, scale, out); continue; }
          const rate = d / (to - from);
          const inner = [];
          collect(target, start - from * rate, rate, inner);
          for (const it of inner) {            // keep what plays inside the window, clipped to it
            const s = Math.max(it.start, start), e = Math.min(it.end, start + d);
            const point = it.end - it.start <= AUDIT_EPS;
            if (point ? (it.start >= start - AUDIT_EPS && it.start <= start + d + AUDIT_EPS) : e - s > AUDIT_EPS) {
              out.push(Object.assign({}, it, { start: point ? it.start : s, end: point ? it.end : e }));
            }
          }
          continue;
        }
        for (const [property, from, to] of propRanges(c, d / scale)) {
          out.push({ target, property, start: start + from * scale, end: start + to * scale, frames: c.__promoFrames || [] });
        }
      }
    }
  }

  // Tweens and sets of one target and property whose time ranges overlap. A set is a point: it overlaps a tween only
  // strictly inside it, and another set only at the same instant.
  function auditTimeline(tl, describe) {
    if (!tl || typeof tl.getChildren !== "function") return [];
    const site = describe || describeFrames(null);
    const items = [];
    collect(tl, 0, 1, items);
    const byTarget = new Map();
    for (const it of items) {
      if (!byTarget.has(it.target)) byTarget.set(it.target, new Map());
      const props = byTarget.get(it.target);
      if (!props.has(it.property)) props.set(it.property, []);
      props.get(it.property).push(it);
    }
    const found = [];
    for (const [target, props] of byTarget) {
      for (const [property, list] of props) {
        list.sort((a, b) => a.start - b.start || a.end - b.end);
        for (let i = 0; i < list.length; i++) {
          for (let j = i + 1; j < list.length && list[j].start < list[i].end + AUDIT_EPS; j++) {
            const a = list[i], b = list[j];
            const pa = a.end - a.start <= AUDIT_EPS, pb = b.end - b.start <= AUDIT_EPS;
            const over = pa && pb ? Math.abs(a.start - b.start) < AUDIT_EPS
              : pa ? a.start > b.start + AUDIT_EPS && a.start < b.end - AUDIT_EPS
              : pb ? b.start > a.start + AUDIT_EPS && b.start < a.end - AUDIT_EPS
              : a.start < b.end - AUDIT_EPS && b.start < a.end - AUDIT_EPS;
            if (over && found.length < 500) {
              const label = target && target.nodeType === 1 ? pathOf(target) : (target && target.name) || "object";
              found.push({ target: label, property, a: { start: a.start, end: a.end, site: site(a.frames) }, b: { start: b.start, end: b.end, site: site(b.frames) } });
            }
          }
        }
      }
    }
    // `scale` is scaleX and scaleY: one finding, not two, when the same two tweens clash on both
    const merged = new Map();
    for (const f of found) {
      const key = [f.target, f.a.start, f.a.end, f.b.start, f.b.end].join("|");
      if (merged.has(key)) merged.get(key).property += ", " + f.property;
      else merged.set(key, f);
    }
    return [...merged.values()];
  }

  // Where each tween was made: the first few stack frames that call into the timeline, as {u: url, l: line}. (The
  // first frame is this wrapper itself.) HyperFrames serves a shot as ONE document with its scripts inlined, so a frame
  // names that document and a line in it; `describeFrames` turns the line into the source text, and `promo verify`
  // finds that text in the shot's files.
  function recordSites() {
    const proto = window.gsap && window.gsap.core && window.gsap.core.Timeline && window.gsap.core.Timeline.prototype;
    if (!proto) return;
    for (const name of ["to", "from", "fromTo", "set"]) {
      const original = proto[name];
      if (typeof original !== "function" || original.__promoSite) continue;
      const wrapped = function (...args) {
        const out = original.apply(this, args);
        try {
          const made = this.recent && this.recent();
          if (made && !made.__promoFrames) {
            const frames = [];
            for (const line of String(new Error().stack).split("\n")) {
              const m = /([^\s()]+?):(\d+):\d+\)?\s*$/.exec(line);
              if (m) frames.push({ u: m[1], l: +m[2] });
              if (frames.length === 4) break;
            }
            made.__promoFrames = frames.slice(1);
          }
        } catch (e) { /* a site is a convenience: never let it break a build */ }
        return out;
      };
      wrapped.__promoSite = true;
      proto[name] = wrapped;
    }
  }

  // ["lib/demo-world.js:306", "lib/demo-world.js:454"], or for a frame in the one served document "L691: <that line's text>".
  function describeFrames(lines) {
    return (frames) => (frames || []).map((f) => {
      let path = f.u;
      try { path = new URL(f.u).pathname; } catch (e) { /* not a URL */ }
      const served = path === "/" || /\/index\.html$/.test(path);
      if (served && lines && lines[f.l - 1] !== undefined) return "L" + f.l + ": " + lines[f.l - 1].trim().slice(0, 160);
      return path.replace(/^\//, "") + ":" + f.l;
    });
  }

  function twoFrames() {
    return new Promise((resolve, reject) => {
      let n = 2;
      const timer = setTimeout(() => reject(new Error(
        "capture-settle: no animation frame arrived in " + SETTLE_HANG_MS + " ms; the page cannot be captured reliably")), SETTLE_HANG_MS);
      const step = () => (n-- > 0 ? requestAnimationFrame(step) : (clearTimeout(timer), resolve()));
      step();
    });
  }

  // payload -> deflate -> "PSD1", a big-endian length, the bytes -> opaque RGB pixels in a 1:1 canvas at the top left.
  async function paintDigest(payload) {
    const bytes = new TextEncoder().encode(JSON.stringify(payload));
    const packed = new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new CompressionStream("deflate"))).arrayBuffer());
    const width = document.documentElement.clientWidth || 1920;
    const rows = Math.ceil((8 + packed.length) / (3 * width));
    const flat = new Uint8Array(width * rows * 3);
    flat.set([0x50, 0x53, 0x44, 0x31], 0);
    new DataView(flat.buffer).setUint32(4, packed.length);
    flat.set(packed, 8);
    const rgba = new Uint8ClampedArray(width * rows * 4);
    for (let i = 0, j = 0; i < flat.length; i += 3, j += 4) {
      rgba[j] = flat[i]; rgba[j + 1] = flat[i + 1]; rgba[j + 2] = flat[i + 2]; rgba[j + 3] = 255;
    }
    const old = document.getElementById(DIGEST_ID);
    if (old) old.remove();
    const canvas = document.createElement("canvas");
    canvas.id = DIGEST_ID;
    canvas.width = width;
    canvas.height = rows;
    canvas.style.cssText = "position:fixed;left:0;top:0;width:" + width + "px;height:" + rows + "px;" +
      "image-rendering:pixelated;z-index:2147483647;pointer-events:none";
    canvas.getContext("2d").putImageData(new ImageData(rgba, width, rows), 0, 0);
    document.body.appendChild(canvas);
  }

  function installCaptureSettle(shotId) {
    recordSites();
    let inner = window[SETTLE_KEY];
    let audit = null;
    async function settle(...args) {
      if (typeof inner === "function") await inner.apply(window, args); // chain: HyperFrames' own hook runs first
      const root = document.documentElement;
      const kept = root.style.zoom;
      root.style.zoom = "1.001";
      void root.offsetWidth;
      await twoFrames();
      root.style.zoom = kept;
      void root.offsetWidth;
      await twoFrames();
      if (audit === null) { // the timeline is registered last, so it is read at the first capture
        let lines = null;
        try { lines = (await (await fetch(location.href)).text()).split("\n"); } catch (e) { /* sites fall back to line numbers */ }
        audit = auditTimeline((window.__timelines || {})[shotId], describeFrames(lines));
      }
      await paintDigest({ v: 1, shot: shotId, state: stateOf(), audit });
    }
    // HyperFrames assigns its own hook when the composition loads, which may be after this runs: take over the
    // property, keep whatever it is given as the inner hook, and always hand out ours.
    Object.defineProperty(window, SETTLE_KEY, { configurable: true, get: () => settle, set: (fn) => { inner = fn; } });
  }
  // capture-settle:end

  async function load(shotId) {
    const [timing, design] = await Promise.all([fetchJson("timing.json", false), fetchJson("design.json", true)]);
    if ((design.capture || {}).settle) installCaptureSettle(shotId);
    const shot = timing.shots.find((s) => s.id === shotId);
    if (!shot) throw new Error("shot not in timing.json: " + shotId);
    const tokens = design.tokens || {};
    const style = document.documentElement.style;
    for (const [name, value] of Object.entries(tokens.colors || {})) {
      style.setProperty("--" + name, value);
      style.setProperty("--" + name.replace(/_/g, "-"), value);
    }
    const scale = (tokens.type_scale || {})[timing.aspect] || {};
    for (const [name, value] of Object.entries(scale)) style.setProperty("--" + name.replace(/_/g, "-"), value + "px");
    const chapter = timing.chapters.find((c) => c.id === shot.chapter) || {};
    const events = timing.events.filter((e) => e.shot === shotId);
    const dur = shot.end_s - shot.start_s;
    const words = [];
    for (const line of timing.lines) {
      for (const w of line.words) {
        const t = w.startMs / 1000 - shot.start_s;
        if (t >= -0.001 && t < dur) words.push({ text: w.text, t, end: w.endMs / 1000 - shot.start_s, line: line.id });
      }
    }
    // fonts.ready resolves at once when no text uses a face yet, so load every declared face explicitly.
    await Promise.all([...document.fonts].map((f) => f.load().catch(() => null)));
    await document.fonts.ready;
    return { timing, design, shot, chapter, events, words, duration: dur, lagS: (timing.limits || {}).text_lag_s || 0.08 };
  }

  function splitWords(el, text, key) {
    const keys = new Set((key || "").toLowerCase().split(/\s+/).filter(Boolean));
    el.textContent = "";
    const parts = text.split(/\s+/).filter(Boolean);
    return parts.map((word, i) => {
      const span = document.createElement("span");
      span.className = "w";
      span.textContent = word + (i < parts.length - 1 ? " " : "");
      if (keys.has(word.toLowerCase().replace(/[^\p{L}\p{N}]/gu, ""))) span.classList.add("key");
      el.appendChild(span);
      return span;
    });
  }

  function eventsOf(S, kind) {
    return S.events.filter((e) => e.kind === kind);
  }

  function finish(tl, S, seed = 7) {
    const el = document.querySelector(".finish-grain");
    if (!el) return;
    const r = PromoMotion.rng(seed);
    for (let t = 0; t < S.duration; t += 1 / 12) {
      tl.set(el, { "--gx": Math.floor(r() * 220) + "px", "--gy": Math.floor(r() * 220) + "px" }, t);
    }
  }

  window.PromoStage = { load, splitWords, eventsOf, finish, capture: { stateOf, auditTimeline } };
})();
