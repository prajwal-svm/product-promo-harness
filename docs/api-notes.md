# API notes

What live runs taught us about the outside tools. Each section is written by the task that first touched the tool.

## HyperFrames

Verified live against `hyperframes@0.8.98` on the Mac (Task 15, `uv run pytest tests/test_scenes.py -m live`).

- **One shot is cheap.** A 4.3 s 1080p60 shot rendered with `--no-browser-gpu --quality high` produced an H.264 `yuv420p`
  MP4 with exactly 258 frames at `60/1`; the whole live test (render, lint, snapshot) took 34 s.
- **`--quality` values** are `draft`, `looks` (the CLI default, CRF 16), `delivery` and `standard`/`high`. `promo scenes`
  sends `high` for final and sample renders and `draft` for animatics.
- **`window.__timelines` needs no guard.** HyperFrames defines it before the shot's script runs, so the base template's
  `window.__timelines[ID] = tl` registers fine after its `await PromoStage.load(ID)`; the rendered frames animate.
- **`lint --json`** prints one JSON object (`ok`, `errorCount`, `warningCount`, `infoCount`, `findings`, `filesScanned`).
  Any notice lines come before it, so the wrapper parses from the first `{`.
- **`check --json`** returns one section per audit: `lint`, `runtime`, `layout`, `motion`, `contrast`, `hdr`, `snapshots`.
  The top-level `ok` is false when any section fails.
  - It picks up the generated `index.motion.json` (the `motion` section reports the `specPath` and its sample count), so the
    `appearsBy` and `keepsMoving` assertions are really evaluated.
  - `layout.samples` and `layout.transitionSamples` hold a hundred or more numbers each; trim them before printing.
  - `contrast` is an audit, not a hint: WCAG AA 3:1 for large text, reported as an error. The base template's accent key word
    (`#C6F135`) on the `saas-kinetic` light background (`#F2F3EC`) fails it.
- **`snapshot`** writes `frame-NN-at-<t>s.png` (the time prints as `0.1s` and `1s`) and also a `contact-sheet.jpg` in the
  same directory. The wrapper globs `frame-*.png`, so the contact sheet is ignored.
- **Project layout.** `lint`, `check` and `snapshot` take the per-shot project directory (`scenes/.render/<id>/`) that holds
  `index.html`, `timing.json`, `design.json` and `index.motion.json`; `render` takes the same directory.
- **ProRes 4444 keeps alpha (Task 16, `uv run pytest tests/test_captions.py -m live`).** `render --format mov` of the caption
  overlay (`packs/_core/caption-overlay/`, unpainted `html`, `body` and root) gave `prores` with `yuva444p12le` at `60/1`: a 3 s
  1080x1920 overlay has exactly 180 frames and is about 23 MB. Decoded with ffmpeg, frames before and after the page are
  fully transparent (alpha 0) and the text frames carry alpha 255 on the glyphs with a soft shadow, so the spec's open
  question is settled. The overlay's timeline holds only `set` calls, so it ends at the last page and needs no padding to the
  root `data-duration`. `lint --json` on the overlay project reports 0 errors, 0 warnings.

## Supertonic

Task 11b. Local, on-device TTS through the `supertonic` CLI (1.1.2), added next to Gemini as `voice.provider: "supertonic"`.
Measured on the Mac with `uv run pytest tests/test_supertonic.py -m live` and a dry run of the 72-line Oleafly mascot script.

- **Install.** `pip install supertonic` or `uv tool install supertonic` (a PEP 668 Python refuses a bare `pip install`;
  here it sits in Homebrew's Python 3.11, so the executable is `/opt/homebrew/bin/supertonic`). `brew info supertonic`
  finds no formula. Models download from Hugging Face on first use:
  `supertonic-2` was already in `~/.cache/supertonic2`; the first `--model supertonic` call fetched 22 files in 12 s.
- **CLI.** `supertonic tts -o out.wav --model M --voice V --lang L --steps N --speed X --silence-duration S -- "<text>"`.
  The `--` keeps a line that starts with a dash from being read as a flag. Output is a 44.1 kHz mono 16-bit WAV.
  `supertonic version` prints `supertonic 1.1.2`. Voices are `F1..F5` and `M1..M5`; `--lang` is a bare code, `en|ko|es|pt|fr` (`en-GB` crashes argparse, so the provider passes the primary subtag and the cache
  key keeps the full tag);
  `--speed` must be 0.7 to 2.0. An unknown voice exits 1 with the voices listed on stdout (`❌ Error: ...`), and a bad
  `--lang` exits 2 from argparse, so the provider reports the `❌ Error:` line, else the last stderr line.
- **Speed.** About 0.3 s to load the model and 0.3 s to generate a four-second line, 0.9 s wall per call. The 72-line
  script, with Parakeet ASR on every take and retakes, took 2 min 21 s.
- **Output is not deterministic.** The same text and options three times gave three different files (three SHA-256s).
  They have the same length (187,392 samples) but the samples are uncorrelated (correlation -0.02, RMS difference larger than
  the signal's own RMS): the diffusion sampler starts from `np.random.randn` with no seed, and the CLI has no `--seed` flag. The
  duration is deterministic, so the pace check does not wobble between draws. Every synthesis is therefore cached by content
  hash (`supertonic-tts`, version, text, voice, model, lang, steps, speed, silence): a rerun reuses the first take instead of
  drawing a new one. A retake with identical settings would come from the cache, so retakes change the settings.
- **Short lines sometimes come back silent.** Peak 3e-5, about -90 dBFS, and the exit code is 0. Measured over 30 draws:
  `Psst.` 2 silent, `Hm.` 6 silent, a normal sentence 0. `supertonic.synthesize` throws a silent draw away (peak under 1e-3)
  and draws again, up to five times, so silence never reaches the cache. `voice.synthesize_line` also rejects a trimmed take
  under 0.1 s, because a `verify: false` line has no ASR to notice (WER 1.0 is what normally catches silence). Five silent
  draws, or a silent take, fall through to the next retake variant; a line silent in every variant is stored with
  `problem: "every take came back silent"`, shown as CHECK, and tried again on the next run (it is never "unchanged").
- **Inline audio tags are read aloud.** `Moving money <short pause> just got safer.` came back from ASR as "Moving money sort
  paws just got safer". The provider is sent the line with its tags stripped; punctuation is the only pause control.
- **No style prompt.** `voice.style` and a line's `style` are ignored. Retakes change the sampler instead: take 2 is
  steps+4 at speed x0.97, take 3 is steps+8 at speed x1.03, and each take's `steps`, `speed` and `silence_s` are stored in the
  line's record (`params`), so `promo voice` still skips a line only when its inputs are unchanged.
- **Pace.** `--steps 12 --speed 1.25` gives about 167 wpm on a 22-word line. The `saas-kinetic` pack's 190 wpm target was
  tuned for Gemini and failed 4 of 71 lines on the pace check (all three takes spent); at 170 it fails 1, so the pack's
  `voice.supertonic.pace_wpm` is 170.
- **Respelling works, ASR still disagrees with the script.** Every spelling tried (`oh-LEEF-lee`, `Oh-leef-lee`, ...) is
  spoken as "oh Leafly", and ASR writes it "Oh Leafly", "O Leafly", "O'Leaf Lee" or "O'Leafly" from one draw to the next.
  Before the fix below, words ASR writes differently from the script (Oleafly, Typst, Preflight, CV, `colour` and `favourite`
  against ASR's `color` and `favorite`, `busywork`) failed WER on 12 of the mascot script's 72 lines though the audio was right.
  `Psst.` and other interjections cannot be scored at all, hence `verify: false` (`promo lint-script` warns on it).
- **Scoring aliases and US/UK spelling (fix round 1).** The shot list's `asr_aliases` (a pack's `voice.asr_aliases` gives
  defaults) maps a script word to the ways ASR is known to write it, e.g. `"Oleafly": ["oh leafly", "o'leafly"]`;
  `promo.asr.apply_aliases` rewrites a match in the ASR words before WER and word alignment. It matches normalised tokens,
  across several ASR words, only whole words side by side ("oh leafly" does not match "oh no leafly"), only for a word the
  line's script contains, and no more often than the script says it. List only mishearings where the audio sounded right:
  "O Lee Flee" and "Oh Lee Flea" (the f moved) and "Tipstore" were left out as real errors. `asr.normalize_words` also maps
  British to American spelling for English (-our, -ise/-ize and -isation, -re/-er, -ogue), by named stems, never by bare
  suffix. An alias is matched on tokens with "and" dropped (as WER drops it), so an alias that contains "and" can never match.
  When a line has more candidate matches than the script allows, the one that lowers the edit distance most is made.
  A line's recorded WER depends on both, so an alias that applies to a line is part of its sidecar fingerprint and
  re-scores that line from the cached audio; a change to the spelling rules does not, so use `promo voice --force` after one.
- **Mascot dry run after the fix** (Supertonic F2, steps 12, speed 1.25, real Parakeet): 72 lines, 0 CHECK, exit 0. Still not
  perfect, and within the 0.1 WER tolerance: "a repo" is spoken "a ripple" (l07), "its id" is "its ed" (n02), and "tools" and
  "tracks" lose their ending (n23, n34).
- **Cost.** A cost row is `provider: "supertonic"`, `usd: 0.0`, `units: {"chars": n, "local": true}`: free, not unpriced.
- **A take's words are the script's words (fix round 2).** Word anchors (`{line, word}`) used to match the raw ASR words, which
  differ from the script in ways WER forgives: "read-only." heard as "read | only.", "ATS-friendly" as "ATS | friendly" or as
  one word, "colour," as "color", "chores:" merged into "chores,22". `asr.align_to_script` now cuts script and ASR into the same
  normalised tokens WER uses (spelling, aliases, hyphen splits, merged tokens; "and" kept so an "and" anchor has its own time),
  pairs them by minimal edit path and writes one `LineTake` word per script word, spelled as the script spells it, from the
  start of its first token to the end of its last (a merged ASR word shares its time among its tokens by letters; a word ASR
  dropped gets a share of the gap). `timing.word_time` also lets an anchor name a part of a hyphenated word ("read", "only"),
  starting where its letters fall in the word. A stored take whose words are not the script's words (written before this) is
  re-scored by the next plain `promo voice --all`, from the cached audio. On the 72-line mascot script, `promo timing` over
  the old records gave 6 `word_not_found` errors (id, read-only, chores, ATS-friendly, colour); after the re-score, 0.
- **Region tags (fix round 3).** `en-GB`, `en_US` and `EN` are English everywhere: Parakeet (not Whisper, which raised
  `Unsupported language: en-gb`), the spelling rules and the Supertonic CLI (primary subtag). Whisper gets only the primary
  subtag, and an unsupported language is a one-line `promo voice` error, not a traceback.
- **Hyphen-part times and clamping.** `align_to_script` keeps the ASR time of each part of a hyphenated script word on the
  word (`Caption.parts`, left out of the JSON when empty) so an anchor on `friendly` fires where it was spoken: on the real F2
  take the old by-letters estimate put it about 0.3 s early. Parts are dropped when one was never heard (the anchor then falls
  back to letters), and every time is clamped to the take.
- **Problem lines and stale records.** A record with `problem` is no take: `promo timing` estimates the line and warns
  `no_take` with the reason, so `promo captions` refuses it and `promo assemble` leaves it off the VO track. A stored take
  whose words predate the script-word format is re-scored from its own WAV in `audio/vo/`, never re-synthesised, so a missing
  cache costs nothing; it is re-synthesised only when neither the cache nor the WAV exists.
