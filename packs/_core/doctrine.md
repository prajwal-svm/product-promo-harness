# Motion doctrine (core)

This file is the authority for every pack. HyperFrames' own docs disagree with each other on
easing and ambient motion; where they conflict with this file, this file wins. A pack may override
a rule only in its `pack.md` under "Overrides", with a reason.

## Render contract

- Every visual is a pure function of time. HyperFrames seeks the paused GSAP timeline; nothing may
  depend on wall-clock time.
- No `Math.random`. Use `PromoMotion.rng(seed)` (mulberry32).
- No `setTimeout`, `setInterval`, `requestAnimationFrame`, CSS transitions or CSS animations in shots.
- One paused `gsap.timeline()` per shot, registered last as `window.__timelines["<shot-id>"]`, after
  `await PromoStage.load(...)` (which also awaits fonts).
- Entrances use `fromTo`, never `from`.
- Layout constants are computed before the timeline is built. Never read `getBoundingClientRect()`
  inside a tween.
- No `will-change` on anything the camera scales.
- No blur of anything: no blur filter, no depth of field, no motion blur. Opacity goes on a wrapper, never on a
  `preserve-3d` element itself.
- Never tween `visibility`, `display` or `autoAlpha` on a clip. Never use `repeat: -1`.
- Do not combine a CSS `transform` with a GSAP tween on the same property; centre with flex or
  `xPercent`/`yPercent`.
- Round every value taken from `PromoMotion.rng` to 4 decimals before setting or tweening it, so every path to a
  value prints the same digits (`rotate(4.29049deg)` and `rotate(4.2905deg)` are one value to `promo verify`, which
  compares at 4 decimals; a generated value left long can still differ by more than that between two paths).
- Give every animated property an explicit start value at build time (`fromTo`, or a `set` at the shot's start).
  Never let a tween read its start from the live element: that value depends on what was rendered before.
- Never overlap two sets or tweens on the same element property in time. HyperFrames ends every seek with a
  backward step, so the earlier tween wins in every capture and the later one never shows (a click ring popping
  mid-animation, a filter pulse that never plays), and verify's seek-order check cannot see it. `promo verify` audits
  the registered timeline and fails on the overlap; end the first tween before the second begins.
- No CSS `outline`, focus rings or selection outlines. Show state with a background tint or a
  border colour change.
- A clip is visible for `[start, start+duration)`: land end states at least one frame early.
- Every time comes from `timing.json` through `PromoStage`. Never type a time into a shot.

## Motion

- Springs, not curves: `PromoMotion.ease("snappy" | "default" | "heavy" | "playful")`.
  Snappy k320/d30 for buttons, toggles and leading edges; default k170/d26 for cards, containers and
  camera; heavy k90/d20 for big type, 3D objects and logo lockups; playful k220/d14 for mascots and
  stickers (visible overshoot).
- Tiny overshoot on UI, none on type.
- A value with several targets is `PromoMotion.track(t, keys)`: one spring per change, never a
  restarted tween.
- Tab indicators stretch: leading and trailing edges on different springs (`PromoMotion.indicator`).
- Text inside a morphing container enters 0.08 s after the morph starts and leaves 0.1 s before the
  next one (`PromoMotion.swapAlpha`).
- Snap, then hold. The ease belongs on the settle, not on the change.
- Staggers are 0.05–0.14 s. No twinning: paired elements never share timing or angle.
- Anticipation before big moves; a small take after them.
- Pans and pushes run at 105% scale or more so frame edges never show.
- Cut at peak velocity, matching direction and speed on both sides. Seams are cuts, whips without
  blur, pushes or match moves. Every seam states its `handoff_out` and the next shot's `handoff_in`
  numerically.
- The click causes the morph. Park the cursor on the trigger first; the cursor tip is the hotspot.
- Counters use `font-variant-numeric: tabular-nums` inside a fixed-width box, with no overshoot.
- Hit-stop (a 2–4 frame hold on impact) is allowed.
- A hit is several things on the same frame: the visual change, a sound, and at most one
  secondary accent (flash ≤ 0.6 opacity, a small shake, or a spark).

## Reads and pacing

- Time every moment for what the viewer needs to understand, not for how fast code can move it.
  Fast actions, slow meanings. One read at a time; lead the eye to it.
- The key beat of a shot sits in its middle, not at the end where the cut eats it.
- Something new every 2–4 s; no gap between events longer than the pack's payoff interval.
- Hook in the first 2 s. The hero is visible by 0.5 s.
- Every cut adds information. Every transition carries meaning: its carrier is a visible object
  that becomes the next scene.

## Finish

- One finishing layer on every shot, drawn last and outside the camera: grade and grain (seeded,
  12 fps). The default is a clean frame: no vignette, inner glow, feathered edge or grey scrim.
- Crops land on component edges or in the gaps between lines, never through content, and never
  with a feathered fade.
