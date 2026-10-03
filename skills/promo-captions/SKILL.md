---
name: promo-captions
description: "Add word-level captions or subtitles to a promo or explainer video made with the product promo harness (`promo captions`: the script's words on the speech-recognition clock, caption pages, SRT/VTT sidecars, a transparent HyperFrames caption overlay laid on by `promo assemble`). Use when asked to caption or subtitle a video, add word-level or karaoke captions, or check caption timing."
---

# promo-captions

Captions come from the words the voice actually spoke. `promo voice` transcribes every take (Parakeet for
English, Whisper large-v3-turbo otherwise) and aligns it to the script, so each word keeps the script's spelling
on the time it was heard. `promo timing` places the lines in the film; `promo captions` reads the words from
`timing.json` as `Caption` records (`text`, `startMs`, `endMs`, `timestampMs`, `confidence`, and `parts` for
the timed halves of a hyphenated word).

## Modes (pack `captions.json → mode`, per aspect ratio)

- `subtitle`: pages in the pack's caption band (`tokens.caption_band`), the spoken word highlighted, rendered by
  HyperFrames as a transparent ProRes 4444 overlay (`captions/overlay/captions.mov`) that `promo assemble` lays
  over the picture on track V2.
- Anything else (`none`): sidecars only. When the words are the design, set the shot's `text_mode` to
  `verbatim-sync` and animate `S.words` in the scene instead.
- saas-kinetic: `none` at 16:9, `subtitle` at 9:16 and 1:1.

## Commands

1. `promo voice --all`, then `promo timing`. `promo captions` refuses (exit 2) while any line is still an estimate.
2. `promo captions` writes `captions/words.json`, `captions/captions.srt`, `captions/captions.vtt`, and in
   `subtitle` mode the overlay. An overlay whose inputs have not changed is kept, not re-rendered.
3. `promo assemble` puts the overlay on V2. Ship the SRT and VTT next to the master for players that want them.

## Paging rules

At most 2 lines a page. The per-aspect character limit (32 at 16:9, 22 at 9:16, 26 at 1:1) is tightened further
by the caption font size and plate, so a page never wraps to a third line. A page breaks after sentence punctuation
or a pause over 250 ms, shows for at least 0.7 s, never overlaps the next page and ends by the end of the film.

## Style

Pack `captions.json → style`: `size_px`, `weight`, `color`, `highlight`, `plate`. Legibility comes from a soft text
shadow or the plate, never an outline. Light themes need a text-safe highlight colour (contrast 3:1 or more).

## QA

`promo critique` finds every VO take in the final mix by cross-correlation (within a frame of `timing.json`) and
scores each line's words on the mix (`sync_vo`, `vo_wer`). The critic checks that no key action sits inside the
caption band and that caption text is never covered or cut.

## A video that is not a harness run

Captions need a run folder with real takes. For an outside video, `promo ref <video>` writes word times to
`transcript.json` in its audit folder; to burn captions in, make the video a run first.
