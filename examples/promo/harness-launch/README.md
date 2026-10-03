# harness-launch

The harness's own 10 s launch sting, made with the harness. Music-led: 5 bars at 120 BPM, one shot, every move on a
beat event from `timing.json`.

```bash
promo timing                 # shotlist.json -> timing.json
promo music --provider lyria # the 10 s bed (GEMINI_API_KEY), conformed to the 120 BPM grid
promo timing                 # snap to the measured grid
python tools/sfx.py          # crisp generated sounds on the events that name one (ELEVENLABS_API_KEY, cached)
promo scenes && promo assemble && promo verify
python tools/still.py 1.5 4.6 9.6   # key frames to renders/look/
```

- `scenes/shots/s1.html` is the whole film: blob floods (discs grown in a stagger, no blur), kinetic type (three noise
  glyphs per letter, then the letter), a jelly cursor (stretch along the path, squash on the click), filled click
  pulses, a timeline, a render bar and a player.
- Every tweened property has a build-time start value and no two tweens overlap on one property, so
  `promo verify` passes (forward and reverse seeks write the same state).
