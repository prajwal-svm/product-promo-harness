"""Anti-Claudism and register checks for narration and on-screen text."""

from __future__ import annotations

import re
import statistics
from decimal import Decimal

from promo.models import Finding
from promo.shotlist import Shotlist

HYPE_WORDS = (
    "revolutionize", "revolutionise", "revolutionary", "game-changer", "game-changing", "seamless",
    "seamlessly", "unleash", "supercharge", "elevate", "effortless", "effortlessly", "cutting-edge",
    "next-level", "world-class", "best-in-class", "empower", "delve", "tapestry",
    "testament", "synergy", "paradigm", "reimagine", "unlock the power",
    "harness the power", "in today's fast-paced", "take it to the next level",
)
AI_VOCAB = (
    "pivotal", "crucial", "showcase", "vibrant", "landscape", "groundbreaking",
    "transformative", "intricate", "foster", "underscore",
)
OPENERS = ("imagine", "what if", "picture this", "in a world", "meet the future", "say goodbye to", "introducing the")
NUMBER_CAP = {"ad-sparse": 6.0, "ad-dense": 5.0, "professor": 3.0}
AD_REGISTERS = {"ad-sparse", "ad-dense"}

_TRIPLET = re.compile(r"\b([\w']+(?: [\w']+)?), ([\w']+(?: [\w']+)?),? and ([\w']+(?: [\w']+)?)[.!?]?$", re.I)
_NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?(?:%|(?P<suf>mn|bn|k|K|m|M|b|B)(?![A-Za-z]))?")
_NOT_X_FORMS = (
    re.compile(r"\bnot\s+(?:just|only|merely)\s+[^.!?]{1,50}\bbut\b"),
    re.compile(r"\b(?:isn'?t|aren'?t)\s+(?:just|only|merely)\b"),
    re.compile(r"\bnot\s+(?:\w+\s+){1,4}but\s+(?:a|an|the)\s+\w+"),
)
_RESTATEMENT_OPENERS = frozenset({
    "it's", "it is", "they're", "they are", "we're", "we are", "you're", "you are",
    "he's", "he is", "she's", "she is",
})
_NEGATION_OPENERS = frozenset({"not", "no", "never", "don't", "do", "doesn't", "won't", "can't"})
_PRONOUNS = frozenset({"i", "we", "you", "they", "he", "she", "it"})
_RESTATED_PRONOUNS = _PRONOUNS - {"i"}
_CONTRACTION = re.compile(r"(?:'s|'re|'m|'ve|'ll|'d|n't)$")
_EDGE_PUNCT = ".,;:!?\"'"


def _clean(text: str) -> str:
    """Normalize typographic apostrophes and quotes."""
    return text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')


def numbers_in(text: str) -> list[str]:
    out = []
    for m in _NUMBER.finditer(text):
        # Extract the full match
        full_match = m.group()
        suffix = m.group("suf")

        # Remove suffix from match to get numeric part
        if suffix:
            num_str = full_match[:-len(suffix)].replace("$", "").replace("%", "").replace(",", "")
        else:
            num_str = full_match.replace("$", "").replace("%", "").replace(",", "")

        value = Decimal(num_str)

        # Apply magnitude suffix if present
        if suffix:
            suffix_lower = suffix.lower()
            if suffix_lower == "k":
                value *= Decimal(1000)
            elif suffix_lower == "m" or suffix_lower == "mn":
                value *= Decimal(1000000)
            elif suffix_lower == "b" or suffix_lower == "bn":
                value *= Decimal(1000000000)

        # Convert to string and normalize (strip trailing zeros after decimal)
        result = str(value)
        if "." in result:
            result = result.rstrip("0").rstrip(".")
        out.append(result)
    return out


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _words(text: str) -> list[str]:
    text = _clean(text)
    return re.findall(r"[\w'$%.,-]+", text)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def _build_vocab_pattern(word: str) -> str:
    r"""Build a regex pattern for a vocabulary entry with inflections.

    For single-word entries: handle e-dropping (revolutionize → revolutionizing),
    y → ies (tapestry → tapestries), and common suffixes s|es|d|ed|ing|r|rs|er|ers.

    For multi-word entries: inflect first and last words, join with [-\s].
    """
    # Split on both spaces and hyphens to identify word boundaries
    words_raw = re.split(r'[\s-]+', word)
    escaped_words = [re.escape(w) for w in words_raw]

    if len(escaped_words) == 1:
        # Single word
        base = escaped_words[0]
        if base.endswith("e"):
            # e-dropping: revolutionize → revolutionized, revolutionizing
            base_no_e = base[:-1]
            return f"(?:{base}(?:s|es|d|ed|r|rs|er|ers)?|{base_no_e}(?:ing|d|ed))"
        elif base.endswith("y"):
            # y → ies: tapestry → tapestries, synergy → synergies
            base_no_y = base[:-1]
            return f"(?:{base}(?:s|d|ed|ing|r|rs|er|ers)?|{base_no_y}ies)"
        else:
            return f"{base}(?:s|es|d|ed|ing|r|rs|er|ers)?"
    else:
        # Multi-word: inflect first and last, join with [-\s]
        first = escaped_words[0]
        last = escaped_words[-1]
        middle = r"[-\s]".join(escaped_words[1:-1]) if len(escaped_words) > 2 else ""

        # Build first word pattern with inflections
        if first.endswith("e"):
            first_pat = f"(?:{first}(?:s|d|ed|r|rs)?|{first[:-1]}(?:ing|d|ed))"
        elif first.endswith("y"):
            first_pat = f"(?:{first}(?:s|d|ed|ing|r|rs)?|{first[:-1]}ies)"
        else:
            first_pat = f"{first}(?:s|es|d|ed|ing|r|rs)?"

        # Build last word pattern with inflections
        if last.endswith("e"):
            last_pat = f"(?:{last}(?:s|d|ed|r|rs)?|{last[:-1]}(?:ing|d|ed))"
        elif last.endswith("y"):
            last_pat = f"(?:{last}(?:s|d|ed|ing|r|rs)?|{last[:-1]}ies)"
        else:
            last_pat = f"{last}(?:s|es|d|ed|ing|r|rs|er|ers)?"

        # Join with middle words if present
        if middle:
            return f"{first_pat}[-\\s]{middle}[-\\s]{last_pat}"
        else:
            return f"{first_pat}[-\\s]{last_pat}"


def _pronoun_base(word: str) -> str:
    """Lowercase a word and strip punctuation and one trailing contraction: "It's" -> "it"."""
    return _CONTRACTION.sub("", word.lower().strip(_EDGE_PUNCT))


def _has_not_x_but_y(text: str) -> bool:
    """Check for "not X, it's Y" framing: three regex forms plus an adjacent clause-pair check."""
    text = _clean(text).lower()
    if any(form.search(text) for form in _NOT_X_FORMS):
        return True

    clauses = [c.split() for c in re.split(r"[,;.!?]", text)]
    clauses = [c for c in clauses if c]
    for c1, c2 in zip(clauses, clauses[1:]):
        if not 2 <= len(c1) <= 8 or c1[0] in _NEGATION_OPENERS:
            continue
        if "not" not in c1 and not any(w.endswith("n't") for w in c1):
            continue
        if c2[0] == "but" or not {c2[0], " ".join(c2[:2])} & _RESTATEMENT_OPENERS:
            continue
        c1_base, c2_base = _pronoun_base(c1[0]), _pronoun_base(c2[0])
        if c1_base == c2_base and c1_base in _RESTATED_PRONOUNS:
            return True
        if c1_base not in _PRONOUNS and c2_base in {"it", "they"}:
            return True
    return False


def _line_rules(text: str, register: str, ref: str, check_not_x_but_y: bool = True) -> list[Finding]:
    f: list[Finding] = []
    text = _clean(text)

    def add(level, code, msg):
        f.append(Finding(level=level, code=code, message=f"{ref}: {msg}", ref=ref))

    if "—" in text or " -- " in text:
        add("error", "em_dash", "no em-dashes; use a full stop or a comma")
    if check_not_x_but_y and _has_not_x_but_y(text):
        add("error", "not_x_but_y", "'not X, it's Y' framing; say what it is")
    low = text.lower().strip()
    for opener in OPENERS:
        if low.startswith(opener):
            add("error", "rhetorical_opener", f"rhetorical opener '{opener}'")
            break
    # Check hype words with inflections
    for word in HYPE_WORDS:
        pattern = _build_vocab_pattern(word)
        if re.search(rf"(?<![\w-]){pattern}(?![\w-])", low):
            add("error", "hype_word", f"hype word '{word}'")
    # Check AI vocab (warnings)
    for word in AI_VOCAB:
        pattern = _build_vocab_pattern(word)
        if re.search(rf"(?<![\w-]){pattern}(?![\w-])", low):
            add("warning", "ai_vocab", f"AI vocab '{word}' may sound generic")
    if register == "ad-sparse" and len(_words(text)) > 12:
        add("error", "line_too_long", f"{len(_words(text))} words; ad-sparse lines are at most 12")
    for sentence in _sentences(text):
        m = _TRIPLET.search(sentence)
        if m:
            add("warning", "triplet", f"triplet '{m.group(0)}'; is the third item earning its place?")
    return f


def _run_rules(sentences: list[str], register: str, ref: str) -> list[Finding]:
    f: list[Finding] = []
    run = 0
    for s in sentences:
        run = run + 1 if len(_words(s)) <= 3 else 0
        if run == 3:
            level = "warning" if register in AD_REGISTERS else "error"
            f.append(Finding(level=level, code="fragment_run", message=f"{ref}: three or more fragments in a row", ref=ref))
    words = sum(len(_words(s)) for s in sentences)
    nums = sum(len(numbers_in(s)) for s in sentences)
    cap = NUMBER_CAP.get(register, 5.0)
    if words and nums / words * 100 > cap:
        f.append(Finding(level="warning", code="number_density",
                         message=f"{ref}: {nums} numbers in {words} words (cap {cap:g} per 100)", ref=ref))
    if register == "professor" and len(sentences) >= 5:
        if statistics.pstdev([len(_words(s)) for s in sentences]) < 3:
            f.append(Finding(level="warning", code="sentence_variance",
                             message=f"{ref}: sentence lengths barely vary; mix short and long", ref=ref))
    return f


def lint_text(text: str, register: str, ref: str = "text") -> list[Finding]:
    """Lint arbitrary prose text. Apply line rules per sentence, run rules over all sentences."""
    text = _clean(text)
    sentences = _sentences(text)
    findings: list[Finding] = []

    # Apply line_rules per sentence (without checking not_x_but_y per-sentence)
    for sentence in sentences:
        findings += _line_rules(sentence, register, ref, check_not_x_but_y=False)

    # Apply run rules over all sentences
    findings += _run_rules(sentences, register, ref)

    # Check not_x_but_y over the whole text (for cross-sentence forms)
    if _has_not_x_but_y(text):
        findings.append(Finding(level="error", code="not_x_but_y",
                               message=f"{ref}: 'not X, it's Y' framing; say what it is", ref=ref))

    return findings


def lint_shotlist(sl: Shotlist, register: str, text_mode_default: str) -> list[Finding]:
    findings: list[Finding] = []
    claims = {c.id: c for c in sl.claims}
    all_claim_numbers = {n for c in sl.claims for n in numbers_in(_clean(c.text))}
    for ch in sl.chapters:
        sentences: list[str] = []
        for line in ch.vo:
            clean_text = _clean(line.text)
            findings += _line_rules(clean_text, register, line.id)
            if not line.verify:
                findings.append(Finding(level="warning", code="verify_skipped",
                                        message=f"{line.id}: verify is false, so no speech recognition checks this line; "
                                                "keep it for pure interjections such as 'Psst.'", ref=line.id))
            sentences += _sentences(clean_text)
            allowed = {n for cid in line.claims for n in numbers_in(_clean(claims[cid].text))}
            for n in numbers_in(clean_text):
                if n not in allowed:
                    findings.append(Finding(level="error", code="unsourced_number",
                                            message=f"{line.id}: number {n} is not in any claim this line cites", ref=line.id))
        findings += _run_rules(sentences, register, ch.id)
        if ch.headline:
            hl = _clean(ch.headline.text)
            findings += [x for x in _line_rules(hl, "professor", f"{ch.id} headline") if x.code != "triplet"]
            for n in numbers_in(hl):
                if n not in all_claim_numbers:
                    findings.append(Finding(level="error", code="unsourced_number",
                                            message=f"{ch.id} headline: number {n} is not in any claim", ref=ch.id))
            modes = {s.text_mode or text_mode_default for s in ch.shots}
            if modes & {"paraphrase", "labels-only"} and any(_norm(hl) == _norm(l.text) for l in ch.vo):
                findings.append(Finding(level="warning", code="text_duplicates_vo",
                                        message=f"{ch.id}: headline repeats the VO word for word; paraphrase it",
                                        ref=ch.id))
    return findings
