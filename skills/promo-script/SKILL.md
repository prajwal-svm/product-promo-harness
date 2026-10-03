---
name: promo-script
description: "Plan and write the script and shot list for a promo or explainer video in a product promo harness run folder (shotlist.json with anchors on beats, words and shot edges, VO in the pack's register, anti-Claudism rules, a claims ledger, app truth). Use when writing or revising a video script, voiceover, storyboard, beat sheet or shot list for `promo`."
---

# promo-script

Order of work: brief → story (logline, the one cause→effect, the proof) → one visual concept and one everyday
analogy → structure from the pack's arc → VO in the pack's register → on-screen text by text mode → the shot list.
Write everything into `shotlist.json`; `promo script` renders the readable `script.md` from it, so text never
exists in two places. Worked example: `examples/promo/harness-launch/shotlist.json` (music-led: bars, beat-anchored
events, each event naming its sound effect).

## Arcs

- **Promo** (ad packs): hook in 2 s or less (the problem in 5 words or fewer of huge type, or the product doing its
  job) → the product appears, UI assembles → feature chapters, each with the cursor doing a real action → one
  sourced proof → recap or dark beat → logo and CTA. A new visual payoff every 2-4 s.
- **Explainer** (`professor` register): hook, familiar world, disruption, mechanism, discovery, consequence,
  recap. Every scene adds understanding. One central cause→effect.

## Timing modes (`timing_mode`)

- `music`: chapters have `bars`; the grid comes from `music.bpm`, then from the measured bed after `promo music`
  or `promo beats`. Cuts land on beats.
- `voice`: the VO leads. Lines follow each other (`{"after": id, "gap": s}`); shots take `seconds` or share what
  the lines need. No music is required.
- `music.style` overrides the pack's mood, instrumentation, arrangement and negative lines for this film.
- A music prompt carries musical words only: each section's note is its shots' `sound.music_cue` values (else
  "continue the groove"), never the chapter `purpose`. Write cues as musical moves ("riser into the drop"); `promo
  music` strips AI, agent, model and brand names from them, which Lyria's policy filter can block.

## Every shot states

`visual` (`composition`, `objects`, `camera`, and `locked`: what must NOT move), `motion` (`entrance`, `primary`,
`secondary`, `exit`), `reads` (what the viewer must understand, `from` and `to` anchors; the key beat mid-shot),
`events` (every visible change, anchored, with `sfx` where it should sound), `sound` (`music_cue`, `ambience`),
`transition_out` (`kind`, the visible `carrier` object that becomes the next scene, and its `meaning`), and
`handoff_out`/`handoff_in` numbers at seams (element, x, y, scale, opacity, direction, speed).

## Anchors, never stopwatch times

- Chapters: `{"bar": n}` (music mode) with `bars`; VO lines `{"beat": n}` or `{"after": id, "gap": s}`.
- Events and reads: `{"beat": n}`, `{"line": id, "word": "Papers", "offset": -0.03}` (the start of the word that
  names the thing; punctuation and case are ignored, `nth` picks a repeat, a part of a hyphenated word works),
  `{"shot": id, "edge": "start", "offset": 0.4}`. `{"t": s}` only when nothing else fits (it warns).
- Something must happen at least every `payoff_interval_s` (pack), and the hook must land by 2 s after pre-roll.

## Registers

- `ad-sparse` (the saas-kinetic default): 1-2 lines a chapter, 12 words a line at most, question→answer allowed,
  35-45% VO coverage.
- `ad-dense`: second person, conversational, 65-80% coverage, pairs with `verbatim-sync` text. The shot list does
  not carry a register yet: lint a dense film with `promo lint-script --register ad-dense`.
- `professor`: 125-145 wpm, complete sentences explaining cause and effect, concrete words before technical
  terms, analogies before figures, 60-75% coverage. Do not duplicate narration on screen.

## App truth and claims

- Every number cites a claim in `claims` (`{"id", "text", "source"}`); the line lists it in `claims`.
- Copy, counts, scores and identifiers shown on screen are the app's real ones, computed from the app's own code
  or data. Never invent UI, citations or lookup results. When the story needs a state the app cannot reach, change
  the story, not the app.
- Competitor names never appear on screen or in VO; frame them out.

## Anti-Claudism (enforced by `promo lint-script`)

No invented numbers; number density capped by register; no "it's not X, it's Y" or "not just X but Y"; no
filler triplets; no em dashes; no stacked fragment runs outside ad headlines; no hype words; no rhetorical
openers ("Imagine…", "Meet the future of…"); vary sentence length. Then run all public copy (VO, headlines, end
card) through the `humanizer` skill.

## Voice-friendly lines

Write lines a TTS voice can say and speech recognition can score: no bare interjections ("Psst." cannot be
verified; the user replaced it with "Hey there, fellow researcher!"). Brand words that are said differently from
their spelling go in `pronounce` (see `promo-voice`).

## Loop

`promo lint-script` and `promo timing` must both be clean (errors block; read every warning). Before any take
exists, `promo timing` estimates line lengths from the pack's pace. Then `promo script` and ask the user to
approve `script.md` before any audio or picture work.
