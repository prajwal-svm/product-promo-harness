---
name: promo-voice
description: "Direct and generate voice-over for a product promo harness run with `promo voice`, using Supertonic (local, free, no key) or Gemini 3.8 TTS (cloud, style-directed acting): choosing the provider, auditions, pronunciation respellings, ASR aliases, verify false for interjections, retakes verified by speech recognition, and fixing CHECK lines. Use when creating, directing, auditioning or fixing narration or voice-over for a promo or explainer video."
---

# promo-voice

`promo voice` makes, checks and stores every VO take; this skill is how to choose and direct the voice. Measured
facts about both providers are in the harness's `docs/api-notes.md`.

## Pick the provider

| | Supertonic (`"provider": "supertonic"`) | Gemini 3.8 TTS (`"provider": "gemini"`) |
| --- | --- | --- |
| Runs | on this Mac, `supertonic` CLI 1.1.2 | Google's API |
| Key and cost | no key, free (`usd: 0` rows in `costs.jsonl`) | `GEMINI_API_KEY`; cents a film |
| Voices | F1-F5, M1-M5; **F2** is a friendly default | Kore (default), Aoede, Leda, Autonoe, Callirrhoe |
| Direction | none: no style prompt, inline tags are stripped, punctuation is the only pause control | a style string per film, chapter and line |
| Languages | en, ko, es, pt, fr | detected automatically, 130+ |

- **Supertonic F2 is a good default.** It needs no key and takes about a second a line.
- **Gemini** when a line must be acted (a whisper, a sigh, a smile on one word, a mock-weary aside), for a
  language Supertonic lacks, or when the user asks for it.

The pack's default provider is still Gemini, so set it in each run's `shotlist.json`; unset fields fall back to
the pack's `voice.json`. `promo new` writes Gemini's voice (`"name": "Kore"`) into the scaffold: change the name
too when you switch, or Supertonic is asked for a voice it does not have.

```json
"voice": {"provider": "supertonic", "name": "F2", "steps": 12, "speed": 1.25}
"voice": {"provider": "gemini", "name": "Kore", "style": "warm, confident, conversational, lightly upbeat"}
```

Supertonic's `steps`, `speed` (0.7-2.0) and `silence_s` are its only controls: steps 12 at speed 1.25 is about
167 wpm, and the pack checks pace against 170. Giving them with `"provider": "gemini"` is a shot list error.

## Pronunciation (`pronounce`)

Respellings change only what the voice is sent; captions, WER and word anchors keep the real spelling:

```json
"pronounce": {"Kubernetes": "koo-ber-NET-eez", "LaTeX": "Lay-tek", "BibTeX": "Bib-tek", "repo": "ree-poh"}
```

Ask the user how any brand word is said before the first take.

## What speech recognition mishears (`asr_aliases`)

When the audio is right but ASR writes the word differently, list the ways it writes it; scoring then reads them as
the script's word:

```json
"asr_aliases": {"Kubernetes": ["cooper netties"], "Typst": ["tips", "tipst"]}
```

List only mishearings of correct audio. A garbled name is a wrong take, not an alias. US and UK spellings are
already treated as equal.

## Interjections (`"verify": false`)

A pure interjection such as "Psst." or "Hm." cannot be scored by ASR. Put `"verify": false` on that line: it skips
the WER check (pace, clipping and silence are still checked), `promo lint-script` warns, and `promo voice` lists it
under "Not verified". Tell the user to listen to those lines. Prefer a speakable line where you can.

## Commands

- `promo voice audition "<a real line>" --voices F1,F2,M1` (Supertonic) or
  `promo voice audition "<a real line>" --voices Kore,Aoede,Leda --style "..."` (Gemini). Files land in
  `audio/auditions/`; the user listens and picks; set `voice.name`.
- `promo voice --all`, or `promo voice --line <id>` (repeatable). Each take is trimmed, normalised to -16 LUFS,
  transcribed (Parakeet for English, Whisper otherwise), aligned to the script's words and retaken up to 3 times
  until WER is 0.1 or less and the pace is within 30% of the pack's. Gemini retakes add "clear diction", then
  "slightly slower, clear diction" to the style; Supertonic retakes change the sampler (steps +4 and speed x0.97,
  then steps +8 and speed x1.03).
- A line marked CHECK: fix the text, `pronounce` or `asr_aliases`, then `promo voice --line <id>`. Rescoring
  cached audio is free: `promo voice --all --force`. Supertonic draws differ every time and the cache pins the
  first one, so for a fresh draw change `steps` or `speed`.
- Then `promo timing`: real takes replace the estimates.

## Directing Gemini

- Models: `gemini-3.8-flash-tts` for finals, `gemini-3.8-flash-lite-tts` for auditions and drafts.
- Cost: $0.50 per 1M text tokens in; $9.00 (flash) or $6.00 (lite) per 1M audio tokens out; 25 audio tokens a
  second. **Prices double on 2027-01-01.** `costs.jsonl` logs every take.
- The text is read **verbatim**: "Say cheerfully: hello" speaks the instruction. Direction goes only in the style
  strings (pack `voice.json`, `shotlist.json` voice `style`, each line's `style`), sent as `speech_metadata.style`.
- Style vocabulary, most important first: emotion, pace, energy, intimacy, smile. "warm, confident,
  conversational, lightly upbeat; brisk within the line" (ad-sparse); "friendly, quick, playful; a smile in the
  voice; lean on the key word" (ad-dense); "calm, curious, precise; unhurried; lets important words land"
  (professor). Keep one style per chapter so takes do not drift.
- Inline tags inside the text: `<short pause>`, `<long pause>`, `<breath>`, `<sigh>`, `<laugh>`. Sparingly; never
  sound effects. Supertonic would read them aloud, so the harness strips them for it.

## Pitfalls

- Numbers and currency are normalised before WER ("$10,000" = "ten thousand dollars").
- Short Supertonic lines sometimes come back silent; the harness redraws them, and a line silent in every take is
  reported as CHECK, never stored as a take.
- You cannot hear the voice. Never say it "sounds good": say it was verified by speech recognition and loudness.
