---
name: promo-critic
description: "Critique a rendered promo or explainer video from the product promo harness like a harsh motion director (`promo critique` evidence: contact sheets, seam strips, phone sheet, sync and loudness measurements), hold it to the user's hard quality rules, score it, and drive the fix-and-rerender loop up to `promo ship`. Use when reviewing, critiquing, scoring or QA-ing a rendered promo video, or when a promo-video run reaches its critic gate."
---

# promo-critic

## One round

1. `promo verify` first (determinism: the same frame whatever the seek order). Then `promo critique` (or
   `promo critique --video renders/draft/animatic.mp4` for a structure round on the animatic, whose `tech` check
   fails on the half-size, 30 fps draft by design). It writes
   `critic/<n>/evidence/` (contact sheets on the beats or every 0.5 s, a strip at every seam and at the fastest
   action, a phone-size sheet, a master sheet, the spectrogram, `loudness.json`, `sync.json`, `tech.json`),
   `critic/<n>/summary.json` and `critic/<n>/PROMPT.md`. It exits 1 when an automatic check fails.
2. Fix failed automatic checks first (`tech`, `loudness`, `sync_sfx`, `sync_vo`, `vo_wer`, `claims`,
   `determinism`, `hf_lint`); they are facts, not taste. Re-render, re-assemble, critique again.
3. Dispatch a **fresh subagent that has not seen the plan or the code**, on the most capable model. Give it only:
   the path to `critic/<n>/`, the instruction to read `PROMPT.md`, open every image in `evidence/` and check every
   frame against the hard rules below (paste them verbatim), and to write `review.md` and `scores.json` next to
   `PROMPT.md`. Do not summarise the video for it.
4. Read `review.md`. Fix the three worst problems, re-render only the affected shots
   (`promo scenes --shot <id>`), `promo assemble`, and start the next round.

## Hard rules (any frame that breaks one fails the round)

- **Readability.** Key text is 22 px or more on screen at 1080p on its beat. The camera pushes in on the element
  the line is about and frames whole components; no word is cropped. The evidence sheets are scaled down (22 px is
  about 4 px on the 360 px-wide phone sheet), so before dispatching, add full-size frames of the doubtful beats to
  `evidence/` (`ffmpeg -ss <t> -i renders/master.mp4 -frames:v 1 critic/<n>/evidence/full-<t>.png`), or measure with a
  readability script that reads every visible text node's rendered height.
- **No blur, anywhere.** No blur filter, depth of field or motion blur on anything, no blurred background, no
  blur-in entrance, no `backdrop-filter`. Fail any blur you see. Competitor names are kept out of frame by the beat
  or the crop, never blurred.
- **No white inner shadow, glow, feathered edge mask, vignette or grey scrim** on any frame.
- **Cursor.** A pointer (hand) cursor over clickables. Each click lands on the real target (button, row or
  field, never a label or panel title). The cursor arrives 0.3-0.5 s before the click.
- **Zooms.** Smooth and accurate, landing framed on the exact element; nothing important clipped (no half
  dialogs, no cut-off documents or panel names); the app window is big enough to hold the content whole.
- **App truth.** Never invent UI, copy, numbers, citations or identifiers. Values on screen are computed from the
  app; a lookup shows only real metadata. Compare against `assets/capture/` and the app itself.
- **No dead frames.** Something new every 2-4 s; never a hold on a sparse or empty grid.
- **Mascot or characters, when used.** Hero scale when they are the shot, natural motion, no anatomy glitches
  (extra limbs, lost line work, foot slide).
- **Fonts are vendored.** No CDN or network at render; a substituted font is a failure.
- **Public copy** went through the `humanizer` skill: no em dash, no "not X but Y", no hype words.
- No CSS outlines or focus rings.

## Pass bar

Every score is 8 or more, every pack checklist item passes, every automatic check passes and no hard rule is
broken. At least 2 rounds, at most 5. After round 5, stop and report what still fails; never claim success you
did not measure.

## The ship gate

`promo ship` accepts only the latest round, and only when its `summary.json` has `ok: true`, its `scores.json`
has `pass: true`, and the video it judged is this master: the round records the SHA-256 of what it judged, so an
animatic round or a re-assembled master does not count. The last round is always on the final master.
`promo ship --force` ships anyway and writes that it did into the run README; use it only when the user says so.
A film silent by design passes `loudness` and `sync_sfx` as "n/a": its shot list plans no VO lines and no SFX
placements, and its pack's music mode is `none`. Anything planned but not made fails and names the step to run.

## Honesty

Audio is checked by measurement (LUFS, true peak, sounds found at their planned events, ASR on the mix), not by
ear: say so. A critic that praises everything is not working; if round 1 scores all 9s, ask the subagent to look
again at the seam strips and the phone sheet.

Companion skills for the fixes: `hyperframes-agent-render-gotchas` (frames that differ by seek order, fonts,
frozen renders) and `ffmpeg-filtergraph-mix-master-gotchas` (loudness, true peak, ducking, the master chain).
