---
name: promo-video
description: "Cut a promo video for this project. The entry point of the product promo harness (the `promo` CLI, installed by its scripts/install.sh: HyperFrames scenes, Supertonic or Gemini 3.8 TTS voice, procedural SFX, ElevenLabs music, OpenTimelineIO assembly, word-level captions, a critic loop) for a showreel-grade promo, launch, sizzle, product demo or explainer video. Use whenever the user asks to cut, make or produce a promo video, launch video, sizzle reel, product video, showreel or motion-graphics video, or an explainer video for this project or product. Prefer it for every such request."
---

# promo-video

Act as a senior motion director, educational storyteller and animator. Be playful in presentation, rigorous about
clarity and decisive in execution. Preserve accuracy, visual continuity and the supplied identity. The user decides
**what** the video is about; the harness decides **how** it is made. Do not stop at a mood board, style frame or
partial prototype.

## Where things are

- Harness: the repo this skill is linked from (`readlink ~/.claude/skills/promo-video` ends in
  `<harness>/skills/promo-video`). Measured tool facts in `docs/api-notes.md`, packs (design systems) in `packs/`.
- CLI: `promo` is on PATH; every command has `--help`. If it is missing, run
  `<harness>/scripts/install.sh`.
- Worked example, read-only: `examples/promo/harness-launch/` (music-led at 120 BPM, a 10 s motion-graphics sting:
  one shot, beat-anchored events, generated SFX on the beat). Read its `shotlist.json` and `scenes/shots/s1.html`
  before building a film of the same kind.
- Keys live in `<harness>/.env`. Never ask the user to paste a key into chat; never print one.
- Each video is a run folder inside the project, `./promo/<slug>/`. Run every `promo` command from inside it.
  `shotlist.json` is the only authored timing source; `timing.json` is derived by `promo timing` and never edited.

## Engines (name them; never hand-roll a seek(t) page)

- Scenes: HyperFrames (pinned 0.8.98), one composition per shot in `scenes/shots/<shot-id>.html`, rendered only
  through `promo scenes`.
- Voice: Supertonic (local, free, no key) or Gemini 3.8 TTS. Load `promo-voice`.
- Music: ElevenLabs through `promo music`, or a licensed track the user supplies. SFX: procedural synth, no key.
- Clips (later plans): Manim for maths and diagrams, Revideo for Motion Canvas-style vector work.

## Effort

Tell the user once, at the start: `/effort xhigh` for a new film, `max` if the first 3 seconds carry a launch,
`medium` for fixes and re-renders.

## Gates (in order; do not skip)

0. **Machine.** `promo doctor`. Inside a run it fails only on what that run needs: `GEMINI_API_KEY` for a Gemini
   voice, `ELEVENLABS_API_KEY` for a generated music bed or generated sounds, Supertonic for a Supertonic voice.
   A keyless Supertonic film with procedural SFX and no music passes.
1. **Brief.** What, for whom, aspect, length, pack, language, voice, CTA. Ask only for what is missing. Then
   `promo new <slug> --pack <pack> --aspect <a> --length <s> --lang <code>` and fill `brief.md`. To learn from a
   reference video: `promo ref <video>` (pacing, loudness, BPM, VO pace); take its grammar, never its content.
2. **Story, claims, real UI.** Read the repo (README, docs, site, the app's own source). Write `story.md` with a
   claims ledger; every number has a source. Collect real product screens into `assets/capture/`. Confirm brand
   colours, font and logo with the user.
3. **Script and shot list.** Load `promo-script`. Write `shotlist.json`. Loop `promo lint-script` and
   `promo timing` until both are clean, then `promo script` and ask the user to approve `script.md`.
   **This is the last stop for approval**; after it, run to the end with the critic as the checkpoint.
4. **Voice.** First run for a pack or brand: `promo voice audition "<a line from the script>"`; the user picks.
   Then `promo voice --all` until no line says CHECK.
5. **Music (optional).** `promo music` (needs the ElevenLabs key; it measures its own beat grid), or put the
   user's track at `audio/music/bed.wav` and run `promo beats`. A voice-led film can have no music at all.
   A warned seam names its fix: `promo music --fresh --segment N` re-rolls just that segment, passed or not (repeat `--segment` for more).
6. **Timing.** `promo timing` again: real takes and the measured grid replace the estimates. 0 errors.
7. **SFX.** `promo sfx render` (one sound per event from the pack's SFX map, procedural, no key). Audition one
   sound with `promo sfx synth <kind> --out /tmp/x.wav`.
   - **Re-run it after every `promo timing`.** Assemble mixes whatever `audio/sfx/track.wav` is on disk and never
     checks it against the timing. On one film, a track several cuts old shipped: its clicks landed on old times and
     it went silent halfway through.
   - An event whose `sfx` name is missing from the pack's map is skipped, with only a warning. Check the warnings:
     whoosh, pop and impact were all silently dropped once.
   - The default `sfx_gain_db` (-12) can be lost under VO plus a music bed. Measure the mix around a click.
   - Word-anchored click events can land before the on-screen click, because the cursor chain lags its word. Put
     click sounds on the measured press times instead.
8. **Stills.** `promo guide`, `promo scenes --scaffold`, build the shots (see Parallel scene work), then
   `promo scenes --stills`. Look at every still against the hard rules; fix composition before motion.
9. **Animatic.** `promo scenes --draft` then `promo assemble --draft` (`renders/draft/animatic.mp4`). Fix pacing
   before polish. For a structure round, `promo critique --video renders/draft/animatic.mp4`: its `tech` check
   fails on the half-size, 30 fps draft by design; judge everything else.
10. **Hardest sample.** `promo scenes --sample A-B` on the 2-4 s you are least sure of; look at it frame by frame.
11. **Full render.** `promo scenes`, then `promo captions` (SRT/VTT always; an overlay where the pack wants
    subtitles), then `promo assemble`. Assemble stops (exit 2) on shot renders older than their timing, scene or
    design: re-render them with `promo scenes --shot <id>`. `--allow-stale` is for a deliberate preview, never
    for the master you ship.
12. **Determinism.** `promo verify`. A mismatch is usually an order-dependent tween in the scene (see the
    gotchas skill).
13. **Critic loop.** Load `promo-critic`: `promo critique`, a fresh critic subagent, fix, re-render only the
    affected shots, `promo assemble`, critique again. At least 2 rounds, at most 5.
14. **Ship.** `promo ship --reveal`: poster from frame 0, a Renders line in the run README, Finder. It refuses
    unless the latest critic round passed on this master. Hand off: what was made, where, what was measured
    (loudness, true peak, sync), anything still failing, and "narration and mix were checked by measurement,
    not by ear".

## Hard quality rules (the user enforced these on the real films; a frame that breaks one fails)

- **Readability.** Key text is 22 px or more on screen at 1080p on its beat. Push in on the element the line is
  about, framing whole components; never crop mid-word.
- **No blur, anywhere.** It is a product demo. No blur filter, no depth of field, no motion blur on text or UI,
  no blurred backgrounds or blur-in entrances, no `backdrop-filter`. Seams use cuts, whips without blur, pushes or
  match moves. Competitor names are never blurred: choose the beat or the crop that keeps them out of frame.
- **No white inner shadow, glow, feathered edge mask, vignette or grey scrim.** The template's frame is clean by
  default; never add one. Crops land on component edges or in the gaps between lines, never through content.
- **Cursor.** A pointer (hand) cursor on clickables. Clicks land on the real target (the button, row or field,
  never a label). The cursor arrives 0.3-0.5 s before the click.
- **Zooms.** Smooth (spring or ease-in-out, no snaps) and accurate: they land framed on the exact element.
  Nothing important clipped. Use a big enough app window (bigger window, slightly smaller content scale) so
  documents, dialogs, menus and panels fit whole through every zoom.
- **App truth.** Never invent UI, copy, numbers, citations or identifiers. Compute real values (counts, scores,
  totals) from the app's own code or data. A lookup shows only the real metadata the real lookup returns.
- **No dead frames.** Something new every 2-4 s. Never hold on a sparse or empty grid: fill it or reframe.
- **Mascot or characters, when used.** Hero scale when they are the shot; natural motion (a real cycle at
  10-12 poses a second with no foot slide, or a hop); no anatomy glitches (stray limbs, lost line work). Check
  every pose at every size it is used.
- **Fonts are vendored** (woff2 plus licence in `scenes/assets/`). No CDN or network at render. Named system
  families such as Times, Georgia or Palatino can make HyperFrames fetch a web font: use vendored families.
- **Public copy** (VO, on-screen text, end card) goes through the `humanizer` skill. No em dash. No "not X but Y".
- No CSS outlines or focus rings anywhere: show state with a tint or a border colour.

Companion skills: `hyperframes-agent-render-gotchas` (determinism, disk, fonts, overlays, frozen renders) when
rendering, and `ffmpeg-filtergraph-mix-master-gotchas` when touching the mix or master chain.

## Parallel scene work

- `promo guide` first. Then one subagent per chapter. Give each its shot blocks from `shotlist.json`,
  `ANIMATION_GUIDE.md` **verbatim**, the hard rules above, and the instruction to edit only
  `scenes/shots/<its ids>.html` and to render with `promo scenes --shot <id>`. Shared files are read-only: report
  problems, never edit them.
- Long films share one world kit in `scenes/lib/`: put shared helpers (a push-in camera, a cursor, UI replicas)
  there once, and measure the 22 px rule with a script that reads every visible text node's rendered height, not by
  eye.

## Rules that are easy to forget

- The showreel question for every shot: would a motion designer put this on their reel?
- Show, don't tell: find the picture before adding a word. One orientation header per shot is allowed.
- Rebuild real UI as moving parts from the app's real layout and copy; never invent it.
- Claims stay exact and carry their disclosure; no number without a source.
- Judge effects in rendered frames, never in a live preview.
- You cannot listen: trust the measurements in `promo voice`, `promo assemble` and `promo critique`, and say so.
- Everything comes from `timing.json`: never type a time into a shot.
- Thumbnail pre-roll: the first shot starts at t=0 and holds its opening state through `preroll_s`, so frame 0
  is the poster. Make it the product doing its job.
- Each HyperFrames shot render needs frame storage on disk; `promo scenes` refuses before launching Chrome when
  the estimate does not fit, and `promo doctor` reports free space.
