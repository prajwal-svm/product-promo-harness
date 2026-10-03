"""Speech recognition with word timings, and WER with number-aware normalisation."""

from __future__ import annotations

import re
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

from num2words import num2words

from promo.models import Caption, WordPart

PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"
CHAR_LANGS = {"ja", "zh", "th"}
_CURRENCY = {"$": "dollars", "€": "euros", "£": "pounds", "₹": "rupees"}
_TYPOGRAPHIC = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'})
# Same rule as promo.lint_script.numbers_in: the suffix is attached to the number and not followed by another letter.
_MAGNITUDE = {"k": 10**3, "m": 10**6, "mn": 10**6, "b": 10**9, "bn": 10**9}
_MAGNITUDE_RE = re.compile(r"(?<!\w)(\d[\d,]*(?:\.\d+)?)(mn|bn|k|m|b)(?![^\W_])")
# A bare 1100-2099 integer with no comma, not a currency amount, percentage, decimal or part of a word.
_YEAR_RE = re.compile(r"(?<![\w.,$€£₹])(?<![$€£₹]\s)(1[1-9]\d\d|20\d\d)(?![\w%]|[.,]\d)")
_ORDINAL_RE = re.compile(r"(?<![\w.])(\d+)(?:st|nd|rd|th)(?![^\W_])")
# num2words' Japanese "year" is an era name (令和八年), not how a calendar year is read.
_NO_YEAR_FORM = {"ja"}

# British to American spelling, so a script written in one and heard in the other scores as a match. Deliberately
# narrow: each rule names its stems, because bare suffix rules would also rewrite hour, your, advise, promise, surprise.
_OUR_RE = re.compile(r"(colo|favo|hono|behavio|neighbo|labo|flavo|humo|rumo|harbo|vapo|savo|odo|armo|rigo|vigo)ur")
_IZE_RE = re.compile(
    r"^((?:un|re|de|mis|over|under|non)?(?:organ|normal|real|recogn|optim|minim|maxim|custom|summar|priorit|categor|"
    r"standard|visual|util|emphas|personal|synchron|author|special|general|central|final|capital|local|mobil|"
    r"apolog|critic|civil|global|digit|memor|familiar|harmon|legal|modern|popular|stabil|sanit|monet))"
    r"is(e|es|ed|ing|er|ers|ation|ations)$")
_YZE_RE = re.compile(r"^(analy|paraly)s(e|es|ed|ing|er|ers)$")
_RE_ER_RE = re.compile(r"^((?:kilo|milli|centi)?met|cent|lit|theat|fib|calib)re(s|d)?$")
_OGUE_RE = re.compile(r"^(catal|anal|dial)ogue(s|d)?$")
_OGUING_RE = re.compile(r"^(catal|dial)oguing$")


def _caption(text: str, start: float, end: float, confidence: float | None = None) -> Caption:
    return Caption(text=text, startMs=round(start * 1000), endMs=round(end * 1000),
                   timestampMs=round(start * 1000), confidence=confidence)


def merge_tokens(tokens: list[tuple[str, float, float]]) -> list[Caption]:
    words: list[list] = []
    for text, start, end in tokens:
        stripped = text.strip()
        if not stripped:
            continue
        if text.startswith(" ") or not words:
            words.append([stripped, start, end])
        elif re.fullmatch(r"[^\w]+", stripped):
            words[-1][0] += stripped
        else:
            words[-1][0] += stripped
            words[-1][2] = end
    return [_caption(w, s, e) for w, s, e in words]


@lru_cache(maxsize=1)
def _parakeet_model():
    from parakeet_mlx import from_pretrained

    return from_pretrained(PARAKEET_MODEL)


def _transcribe_parakeet(path: Path) -> list[Caption]:
    result = _parakeet_model().transcribe(str(path))
    tokens = [(t.text, float(t.start), float(t.end)) for s in result.sentences for t in s.tokens]
    return merge_tokens(tokens)


def primary_language(language: str) -> str:
    """`en-GB`, `en_US` and `EN` are `en`: ASR engines and the spelling rules take a bare language code."""
    return re.split(r"[-_]", language.strip().lower())[0]


def _transcribe_whisper(path: Path, language: str) -> list[Caption]:
    import mlx_whisper

    primary = primary_language(language)
    try:
        result = mlx_whisper.transcribe(
            str(path), path_or_hf_repo=WHISPER_MODEL, word_timestamps=True,
            condition_on_previous_text=False, hallucination_silence_threshold=2.0, language=primary)
    except ValueError as e:      # an unsupported language is a setting to fix, not a crash
        raise ValueError(f"speech recognition (Whisper) cannot transcribe language '{language}': {e}") from e
    out = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            text = w["word"].strip()
            if text:
                out.append(_caption(text, float(w["start"]), float(w["end"]), w.get("probability")))
    return out


def _transcribe_scribe(path: Path, language: str) -> list[Caption]:
    from promo.config import require_key
    from promo.providers.http import request

    key = require_key("ELEVENLABS_API_KEY")
    resp = request(
        "elevenlabs", "POST", SCRIBE_URL, headers={"xi-api-key": key},
        data={"model_id": "scribe_v2", "timestamps_granularity": "word", "language_code": language},
        files={"file": (Path(path).name, Path(path).read_bytes(), "audio/wav")})
    words = [w for w in resp.json().get("words", []) if w.get("type", "word") == "word" and w["text"].strip()]
    return [_caption(w["text"].strip(), float(w["start"]), float(w["end"])) for w in words]


def transcribe(path: Path, language: str = "en", engine: str | None = None) -> list[Caption]:
    engine = engine or ("parakeet" if primary_language(language) == "en" else "whisper")
    if engine == "parakeet":
        return _transcribe_parakeet(path)
    if engine == "whisper":
        return _transcribe_whisper(path, language)
    if engine == "scribe":
        return _transcribe_scribe(path, language)
    raise ValueError(f"unknown ASR engine '{engine}' (parakeet, whisper, scribe)")


def _spell(number: str, language: str) -> str:
    value: int | float = float(number) if "." in number else int(number)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    try:
        return num2words(value, lang=language)
    except (NotImplementedError, OverflowError):
        return num2words(value, lang="en")


def _ordinal(number: str, language: str) -> str:
    try:
        return num2words(int(number), lang=language, to="ordinal")
    except (NotImplementedError, AttributeError, OverflowError):
        return num2words(int(number), lang="en", to="ordinal")


def _year(number: str, language: str) -> str:
    if primary_language(language) not in _NO_YEAR_FORM:
        try:
            return num2words(int(number), lang=language, to="year")
        except (NotImplementedError, AttributeError, OverflowError):
            pass
    return _spell(number, language)


def _expand_magnitude(m: re.Match) -> str:
    value = Decimal(m.group(1).replace(",", "")) * _MAGNITUDE[m.group(2)]
    return format(value.normalize(), "f")


def us_spelling(word: str) -> str:
    """One lower-case English word in American spelling: colour, behaviour (-our), normalise, organisation (-ise,
    -isation), analyse, centre, metre (-re) and catalogue (-ogue). Anything not named in a rule is returned unchanged."""
    word = _OUR_RE.sub(r"\1r", word)
    word = _IZE_RE.sub(r"\1iz\2", word)
    word = _YZE_RE.sub(r"\1z\2", word)
    word = _RE_ER_RE.sub(lambda m: m.group(1) + "er" + {"s": "s", "d": "ed", None: ""}[m.group(2)], word)
    word = _OGUE_RE.sub(lambda m: m.group(1) + "og" + {"s": "s", "d": "ed", None: ""}[m.group(2)], word)
    return _OGUING_RE.sub(r"\1oging", word)


def normalize_words(text: str, language: str = "en", *, keep_and: bool = False) -> list[str]:
    t = text.translate(_TYPOGRAPHIC).lower()
    t = _YEAR_RE.sub(lambda m: f" {_year(m.group(1), language)} ", t)
    t = _MAGNITUDE_RE.sub(_expand_magnitude, t)
    t = re.sub(r"([$€£₹])\s?(\d[\d,]*(?:\.\d+)?)", lambda m: f"{m.group(2)} {_CURRENCY[m.group(1)]}", t)
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    t = t.replace("%", " percent")
    t = _ORDINAL_RE.sub(lambda m: f" {_ordinal(m.group(1), language)} ", t)
    t = re.sub(r"\d+(?:\.\d+)?", lambda m: f" {_spell(m.group(0), language)} ", t)
    if primary_language(language) in CHAR_LANGS:
        return [c for c in t if c.isalnum()]
    tokens = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)*", t)
    if primary_language(language) == "en":
        tokens = [us_spelling(tok) for tok in tokens]
    return [tok for tok in tokens if keep_and or tok != "and"]


def _edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def _count_runs(tokens: list[str], run: list[str]) -> int:
    count, i = 0, 0
    while i <= len(tokens) - len(run):
        if tokens[i:i + len(run)] == run:
            count, i = count + 1, i + len(run)
        else:
            i += 1
    return count


def active_aliases(script: str, aliases: Mapping[str, Sequence[str]], language: str = "en") -> dict[str, list[str]]:
    """The aliases that can matter to one line: those whose canonical word is in its script."""
    words = normalize_words(script, language)
    return {canonical: list(heard) for canonical, heard in aliases.items()
            if (tokens := normalize_words(canonical, language)) and _count_runs(words, tokens)}


def apply_aliases(words: list[Caption], aliases: Mapping[str, Sequence[str]], script: str,
                  language: str = "en") -> list[Caption]:
    """Replace known mishearings in an ASR result with the word the script says.

    `aliases` maps a canonical word to the ways ASR is known to write it ("Oleafly": ["oh leafly", "o'leafly"]).
    Aliases are compared with normalised tokens (case, punctuation and spelling variants do not matter) and may span
    several ASR words. Limits keep an alias from hiding a real error:
    - a match covers whole ASR words, side by side, so "oh leafly" never matches "oh no leafly" or "pro leafly";
    - a word is only replaced when `script` says it, and no more often than the script says it, counting the times ASR
      already wrote it correctly ("tips" is not rewritten to Typst on a line that never mentions Typst);
    - a replacement is made only if it lowers the line's edit distance to the script, best first, so an alias that is
      also a real word elsewhere in the line ("tips") adds no error;
    - the matched words become one, from the first one's start to the last one's end, keeping their punctuation.

    "and" never takes part in a match: normalisation drops it (so "and" does not count in WER), which means an alias
    containing "and" cannot be matched and one that looks like it ("rock and roll") would be read as "rock roll".
    """
    per_word = [normalize_words(w.text, language) for w in words]
    flat = [(tok, i) for i, toks in enumerate(per_word) for tok in toks]
    heard = [t for t, _ in flat]
    script_tokens = normalize_words(script, language)
    canon: dict[str, list[str]] = {c: toks for c in aliases if (toks := normalize_words(c, language))}
    budget = {c: _count_runs(script_tokens, toks) - _count_runs(heard, toks) for c, toks in canon.items()}
    wanted = sorted(((normalize_words(h, language), c) for c in canon for h in aliases[c]), key=lambda p: -len(p[0]))
    wanted = [(tokens, c) for tokens, c in wanted if tokens]

    candidates, pos = [], 0                      # (token start, token end, first word, last word, canonical)
    while pos < len(flat):
        first = flat[pos][1]
        found = None
        if pos == 0 or flat[pos - 1][1] != first:                    # a match starts at the start of an ASR word
            for tokens, c in wanted:
                end = pos + len(tokens)
                if heard[pos:end] != tokens:
                    continue
                last = flat[end - 1][1]
                whole_words = end == len(flat) or flat[end][1] != last
                if whole_words and all(per_word[k] for k in range(first, last + 1)):   # side by side
                    found = (pos, end, first, last, c)
                    break
        if found:
            candidates.append(found)
        pos = found[1] if found else pos + 1

    def tokens_after(chosen: list) -> list[str]:
        out, cursor = [], 0
        for start, end, _, _, c in sorted(chosen):
            out += heard[cursor:start] + canon[c]
            cursor = end
        return out + heard[cursor:]

    chosen: list = []
    distance = _edit_distance(script_tokens, heard)
    while True:
        better = [(d, c) for c in candidates if c not in chosen and budget[c[4]] > 0
                  and (d := _edit_distance(script_tokens, tokens_after(chosen + [c]))) < distance]
        if not better:
            break
        distance, best = min(better, key=lambda pair: (pair[0], pair[1][0]))
        chosen.append(best)
        budget[best[4]] -= 1

    out: list[Caption] = []
    word_i = 0
    for _, _, first, last, c in sorted(chosen):
        out.extend(words[word_i:first])
        a, b = words[first], words[last]
        lead = re.match(r"[^\w]*", a.text).group(0)
        trail = re.search(r"[^\w]*$", b.text).group(0)
        confidences = [w.confidence for w in words[first:last + 1] if w.confidence is not None]
        out.append(Caption(text=f"{lead}{c}{trail}", startMs=a.startMs, endMs=b.endMs, timestampMs=a.timestampMs,
                           confidence=min(confidences) if confidences else None))
        word_i = last + 1
    out.extend(words[word_i:])
    return out


def spread_words(script_words: list[str], start_ms: float, end_ms: float) -> list[Caption]:
    """Script words with the time from `start_ms` to `end_ms` shared out by letters per word."""
    weights = [max(1, sum(c.isalnum() for c in w)) for w in script_words]
    whole, done, out = sum(weights), 0, []
    for word, weight in zip(script_words, weights):
        a = round(start_ms + (end_ms - start_ms) * done / whole)
        done += weight
        out.append(Caption(text=word, startMs=a, endMs=round(start_ms + (end_ms - start_ms) * done / whole),
                           timestampMs=a))
    return out


def _pairs(script: list[str], heard: list[str]) -> list[tuple[int, int]]:
    """Which heard token stands for which script token: the diagonal of a minimal edit path (equal or substituted)."""
    rows = [list(range(len(heard) + 1))]
    for i, x in enumerate(script, 1):
        row = [i] + [0] * len(heard)
        for j, y in enumerate(heard, 1):
            row[j] = min(rows[-1][j] + 1, row[j - 1] + 1, rows[-1][j - 1] + (x != y))
        rows.append(row)
    pairs, i, j = [], len(script), len(heard)
    while i and j:
        if rows[i][j] == rows[i - 1][j - 1] + (script[i - 1] != heard[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif rows[i][j] == rows[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


_HYPHENS = "-\u2010\u2011"


def _parts(word: str, token_ids: list[int], spans: dict[int, list], language: str) -> list[tuple[str, float, float]]:
    """(text, start, end) of each hyphen-separated part of a script word, from the ASR times of its tokens.

    Empty unless the word has two or more parts and every part has a paired ASR token; the caller then falls back to
    sharing the word's time by letters.
    """
    pieces = [p for p in re.split(f"[{_HYPHENS}]", word) if normalize_words(p, language, keep_and=True)]
    counts = [len(normalize_words(p, language, keep_and=True)) for p in pieces]
    if len(pieces) < 2 or sum(counts) != len(token_ids):
        return []
    out, k = [], 0
    for piece, n in zip(pieces, counts):
        heard = [spans[t] for t in token_ids[k:k + n] if t in spans]
        k += n
        if not heard:
            return []
        out.append((piece, min(h[0] for h in heard), max(h[1] for h in heard)))
    return out


def align_to_script(script_words: list[str], words: list[Caption], duration_ms: float,
                    language: str = "en") -> list[Caption]:
    """One Caption per script word, spelled as the script spells it, on the ASR clock.

    Script and ASR are cut into the same normalised tokens that WER uses (so colour matches color, "read-only" matches
    "read" + "only", "chores:" matches a merged "chores,22"), and the two token lists are paired by minimal edit path.
    A script word starts where its first token's ASR word starts and ends where its last one ends; an ASR word that
    stands for several tokens shares its time among them by letters. A script word ASR never wrote gets a share of the gap
    between its neighbours, and a take with no pairs at all shares the whole duration. A hyphenated script word also
    carries `parts`, each timed from its own tokens. Every time is clamped to the take, [0, duration_ms]. Anchors and
    captions can then look words up by the script's own text, whatever ASR heard.
    """
    if not script_words:
        return []
    s_tok = [(t, i) for i, w in enumerate(script_words) for t in normalize_words(w, language, keep_and=True)]
    a_tok: list[tuple[str, float, float, int]] = []
    for j, w in enumerate(words):
        toks = normalize_words(w.text, language, keep_and=True)
        weights = [max(1, len(t)) for t in toks]
        done, span = 0, w.endMs - w.startMs
        for t, weight in zip(toks, weights):
            a_tok.append((t, w.startMs + span * done / sum(weights), w.startMs + span * (done + weight) / sum(weights), j))
            done += weight
    tok_span: dict[int, list] = {}                 # script token -> [start, end] of the ASR token it pairs with
    spans: dict[int, list] = {}                    # script word -> [start, end, confidences]
    for si, ai in _pairs([t for t, _ in s_tok], [t for t, *_ in a_tok]):
        _, start, end, j = a_tok[ai]
        tok_span[si] = [start, end]
        entry = spans.setdefault(s_tok[si][1], [start, end, []])
        entry[0], entry[1] = min(entry[0], start), max(entry[1], end)
        if words[j].confidence is not None:
            entry[2].append(words[j].confidence)
    if not spans:
        return _clamped(spread_words(script_words, 0, duration_ms), duration_ms)
    out: list[Caption] = []
    i = 0
    while i < len(script_words):
        if i in spans:
            start, end, confs = spans[i]
            token_ids = [si for si, (_, wi) in enumerate(s_tok) if wi == i]
            parts = [WordPart(text=t, startMs=round(a), endMs=round(b))
                     for t, a, b in _parts(script_words[i], token_ids, tok_span, language)]
            out.append(Caption(text=script_words[i], startMs=round(start), endMs=round(end), timestampMs=round(start),
                               confidence=min(confs) if confs else None, parts=parts))
            i += 1
            continue
        j = i
        while j < len(script_words) and j not in spans:
            j += 1
        lo = out[-1].endMs if out else 0
        hi = round(spans[j][0]) if j in spans else round(duration_ms)
        out.extend(spread_words(script_words[i:j], lo, max(lo, hi)))
        i = j
    return _clamped(out, duration_ms)


def _clamped(words: list[Caption], duration_ms: float) -> list[Caption]:
    """Every start and end (and part time) inside the take, [0, duration_ms], and no end before its start."""
    top = max(0, round(duration_ms))

    def span(start: int, end: int) -> tuple[int, int]:
        a = min(max(start, 0), top)
        return a, max(a, min(max(end, 0), top))

    out = []
    for w in words:
        a, b = span(w.startMs, w.endMs)
        parts = [WordPart(text=p.text, startMs=span(p.startMs, p.endMs)[0], endMs=span(p.startMs, p.endMs)[1])
                 for p in w.parts]
        out.append(w.model_copy(update={"startMs": a, "endMs": b, "timestampMs": a, "parts": parts}))
    return out


def wer(reference: str, hypothesis: str, language: str = "en") -> float:
    ref = normalize_words(reference, language)
    hyp = normalize_words(hypothesis, language)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)
