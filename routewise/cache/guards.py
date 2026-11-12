"""Rule guards: cheap, explainable checks that veto a semantic-cache candidate.

Embeddings capture *topic* very well and *exact meaning* poorly: "capital of India" vs
"capital of Indonesia", "Python 3.10" vs "Python 3.12", "is X safe" vs "is X not safe"
all embed close together. Each guard targets one such failure mode. A guard only ever
rejects (it can't create a hit), so a wrong rejection costs a cache miss, never a wrong
answer — we deliberately favour precision over recall.
"""

from __future__ import annotations

import logging
import re
from collections import Counter  # used by word_order_guard
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)

ALL_GUARDS = ("numbers", "negation", "entities", "word_order")

# ---- numbers / years / versions ---------------------------------------------------------
# "3.10", "2024", "1,000", "v2", "5th", "25k". Version-like strings stay intact so 3.10 != 3.1.
_NUMBER = re.compile(r"(?<![\w.])v?(\d+(?:[.,]\d+)*)(k|st|nd|rd|th)?(?![\w])", re.I)
# "one"/"first"/"second" are deliberately absent: they are far more often pronouns or
# ordinary words ("how can one...", "first element", "wait a second") than quantities.
_NUMBER_WORDS = {
    "zero": "0", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "third": "3", "fourth": "4", "fifth": "5", "hundred": "100",
    "thousand": "1000", "million": "1000000", "billion": "1000000000",
}
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _canon_number(digits: str, suffix: str | None) -> str:
    digits = digits.replace(",", "")
    if suffix and suffix.lower() == "k":  # "25k" == "25000"
        try:
            return str(int(float(digits) * 1000))
        except ValueError:
            return digits
    return digits


def extract_numbers(text: str) -> set[str]:
    # A set, not a count: mentioning the same number twice doesn't change the question.
    nums = {_canon_number(m.group(1), m.group(2)) for m in _NUMBER.finditer(text)}
    # Spelled-out numbers count too, so "two" == "2".
    nums |= {_NUMBER_WORDS[w] for w in _WORD.findall(text.lower()) if w in _NUMBER_WORDS}
    return nums


def numbers_guard(a: str, b: str) -> bool:
    """Pass only if both queries mention the same set of numbers."""
    return extract_numbers(a) == extract_numbers(b)


# ---- negation -------------------------------------------------------------------------
_NEGATIONS = {
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "without", "cannot",
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "wont", "cant", "shouldnt",
    "wouldnt", "couldnt", "hasnt", "havent", "hadnt", "mustnt",
}


def count_negations(text: str) -> int:
    t = text.lower().replace("’", "'")
    t = re.sub(r"n't\b", " not", t)  # "isn't" -> "is not"
    t = t.replace("'", "")
    return sum(1 for w in _WORD.findall(t) if w in _NEGATIONS)


def negation_guard(a: str, b: str) -> bool:
    """Pass only if both queries have the same negation polarity (even/odd count).

    Parity rather than presence: "not uncommon"-style double negations are rare, but
    "is it not true that X is not Y" shouldn't be treated as negated.
    """
    return count_negations(a) % 2 == count_negations(b) % 2


# ---- named entities -------------------------------------------------------------------
_ENTITY_LABELS = {"PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"}
_STOP_CAPS = {"what", "who", "how", "why", "when", "where", "which", "is", "are", "can", "do", "does", "i", "the", "a"}


@lru_cache(maxsize=1)
def _nlp() -> Any | None:
    try:
        import spacy

        # Only NER is needed; parser/lemmatizer off for speed.
        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except Exception as e:  # noqa: BLE001
        log.warning("spaCy en_core_web_sm unavailable (%s); entity guard uses capitalized tokens", e)
        return None


def _norm_text(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"'s\b", "", s)
    return " ".join(re.findall(r"[a-z0-9.+#]+", s))


def _norm_entity(s: str) -> str:
    return re.sub(r"^the\s+", "", _norm_text(s))


def extract_entities(text: str) -> set[str]:
    nlp = _nlp()
    if nlp is None:
        # Fallback: capitalized words that aren't the sentence-initial question word.
        return {_norm_entity(w) for w in re.findall(r"\b[A-Z][a-zA-Z]+\b", text) if w.lower() not in _STOP_CAPS}
    # Named entities only. (An earlier version also added every PROPN token; error
    # analysis showed that mostly added noise like "President" or "New Year".)
    # Spans containing digits are skipped: quantities are the numbers guard's job, and the
    # small NER model mislabels them (it tagged "Convert 5 miles" as a PERSON).
    return {e for e in (_norm_entity(ent.text) for ent in spacy_ents(nlp, text))
            if e and not any(ch.isdigit() for ch in e)}


def spacy_ents(nlp: Any, text: str) -> list[Any]:
    return [e for e in nlp(text).ents if e.label_ in _ENTITY_LABELS]


def _mentioned(entity: str, other_text: str) -> bool:
    # Case-insensitive, whole-word presence in the OTHER query's text, not in its NER
    # output: NER is unreliable on lowercase input ("instagram"), plain text matching isn't.
    return re.search(r"(?<![a-z0-9])" + re.escape(entity) + r"(?![a-z0-9])", other_text) is not None


def entities_guard(a: str, b: str) -> bool:
    """Pass only if every entity found in either query is also mentioned in the other."""
    na, nb = _norm_text(a), _norm_text(b)
    return all(_mentioned(e, nb) for e in extract_entities(a)) and all(_mentioned(e, na) for e in extract_entities(b))


# ---- word order swap ------------------------------------------------------------------
def word_order_guard(a: str, b: str) -> bool:
    """Reject "same words, different order" pairs, e.g. "5 miles to km" vs "5 km to miles".

    Found in Phase 2 probing: the QQP cross-encoder scores that pair 0.97 (duplicate!),
    because it mostly learned lexical overlap. Real paraphrases almost always change at
    least one word, so identical bags of words in a different order usually signal a
    swapped direction/role. Cost of a wrong rejection: one cache miss.
    """
    wa, wb = _WORD.findall(a.lower()), _WORD.findall(b.lower())
    return not (Counter(wa) == Counter(wb) and wa != wb)


_GUARD_FUNCS = {
    "numbers": numbers_guard,
    "negation": negation_guard,
    "entities": entities_guard,
    "word_order": word_order_guard,
}


@dataclass
class GuardReport:
    passed: bool
    results: dict[str, bool] = field(default_factory=dict)  # guard name -> passed?

    @property
    def failed(self) -> list[str]:
        return [k for k, v in self.results.items() if not v]


def run_guards(a: str, b: str, enabled: tuple[str, ...] | list[str] = ALL_GUARDS) -> GuardReport:
    results = {name: _GUARD_FUNCS[name](a, b) for name in enabled}
    return GuardReport(all(results.values()), results)
