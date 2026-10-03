/* PromoMotion: closed-form springs and seeded noise. Pure functions of time (see doctrine.md). */
(function () {
  const PRESETS = { snappy: [320, 30], default: [170, 26], heavy: [90, 20], playful: [220, 14] };
  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));

  function spring(t, k = 170, d = 26) {
    if (t <= 0) return 0;
    const w0 = Math.sqrt(k);
    const z = d / (2 * w0);
    if (Math.abs(z - 1) < 1e-9) return 1 - Math.exp(-w0 * t) * (1 + w0 * t);
    if (z < 1) {
      const wd = w0 * Math.sqrt(1 - z * z);
      return 1 - Math.exp(-z * w0 * t) * (Math.cos(wd * t) + ((z * w0) / wd) * Math.sin(wd * t));
    }
    // Overdamped: two real roots, so the response is a blend of two decaying exponentials.
    const s = Math.sqrt(z * z - 1);
    const r1 = -w0 * (z - s);
    const r2 = -w0 * (z + s);
    return 1 - (r2 * Math.exp(r1 * t) - r1 * Math.exp(r2 * t)) / (r2 - r1);
  }

  function settle(k, d, eps = 0.001) {
    const dt = 1 / 240;
    let last = 0;
    for (let t = 0; t < 10; t += dt) if (Math.abs(1 - spring(t, k, d)) > eps) last = t;
    return last + dt;
  }

  function preset(name) {
    const p = PRESETS[name];
    if (!p) throw new Error("unknown spring preset: " + name);
    return p;
  }

  function ease(name) {
    const [k, d] = preset(name);
    const dur = settle(k, d);
    const f = (p) => (p >= 1 ? 1 : spring(p * dur, k, d));
    f.duration = dur;
    return f;
  }

  function track(t, keys, k = 170, d = 26) {
    let v = keys[0][1];
    for (let i = 1; i < keys.length; i++) v += (keys[i][1] - keys[i - 1][1]) * spring(t - keys[i][0], k, d);
    return v;
  }

  function indicator(t, stops) {
    const lead = track(t, stops, 320, 30);
    const trail = track(t, stops, 140, 22);
    return { left: Math.min(lead, trail), right: Math.max(lead, trail) };
  }

  function swapAlpha(t, tIn, tOut) {
    return Math.min(clamp((t - tIn - 0.08) / 0.12), clamp((tOut - 0.1 - t) / 0.1));
  }

  function rng(seed) {
    return function () {
      seed |= 0;
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const loopT = (t, dur) => ((t % dur) + dur) % dur;

  window.PromoMotion = { PRESETS, spring, settle, ease, track, indicator, swapAlpha, rng, loopT, clamp };
})();
