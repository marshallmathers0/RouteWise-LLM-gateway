"""Router features, computed from the user's query text ONLY.

Two groups:
* the query embedding (bge-small, 384-d) - captures topic and phrasing;
* ~20 handcrafted features - cheap, interpretable signals of difficulty (length, code,
  math notation, multiple-choice options, question type).

What is deliberately NOT a feature: the dataset `source`. It is almost a proxy for the
label (e.g. "competition math" queries are mostly hard), so a router using it would look
great offline and be useless on real traffic, where nobody tells you the source. The
same goes for the answer-format suffix we add to prompts, which is why features use the
raw `query` and never the `prompt`.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from routewise.cache.models import Embedder

QUESTION_WORDS = ("what", "how", "why", "which", "who", "when", "where", "explain", "write", "prove",
                  "calculate", "find", "solve", "list", "summarize", "classify", "compare")

HANDCRAFTED_NAMES = (
    "log_chars", "log_words", "n_lines", "avg_word_len", "digit_ratio", "n_numbers",
    "n_math_ops", "has_latex", "has_code", "n_code_tokens", "n_options", "has_question_mark",
    *(f"qw_{w}" for w in QUESTION_WORDS),
)

# Words that would let a model recognise our datasets by their templates. Asserted absent.
FORBIDDEN_FEATURES = {"source", "split", "label", "gold", "category", "subject"}

_CODE_TOKENS = re.compile(r"```|\bdef\b|\bclass\b|\breturn\b|\bimport\b|\bassert\b|==|\(\)|;|\{|\}|=>|\bfor\b.*\bin\b")
_MATH_OPS = re.compile(r"[=+\-*/^<>]|\\frac|\\sqrt|\\sum|\\int|\\pi|\\le|\\ge|\$")
_OPTION_LINE = re.compile(r"^\s*[A-J][.)]\s+\S", re.M)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def handcrafted(query: str) -> np.ndarray:
    words = query.split()
    lower = query.lower()
    first = re.findall(r"[a-z]+", lower[:60])
    feats = [
        np.log1p(len(query)),
        np.log1p(len(words)),
        query.count("\n") + 1,
        (sum(len(w) for w in words) / len(words)) if words else 0.0,
        sum(c.isdigit() for c in query) / max(1, len(query)),
        len(_NUMBER.findall(query)),
        len(_MATH_OPS.findall(query)),
        float("\\" in query or "$" in query),
        float(bool(_CODE_TOKENS.search(query))),
        len(_CODE_TOKENS.findall(query)),
        len(_OPTION_LINE.findall(query)),
        float("?" in query),
        # Question type: does one of these words appear near the start?
        *(float(w in first) for w in QUESTION_WORDS),
    ]
    return np.asarray(feats, dtype=np.float32)


class RouterFeaturizer:
    """[embedding | handcrafted] for a batch of queries."""

    def __init__(self, embedder: Embedder | None, use_embedding: bool = True) -> None:
        if use_embedding and embedder is None:
            raise ValueError("use_embedding=True needs an embedder")
        self.embedder = embedder
        self.use_embedding = use_embedding

    @property
    def feature_names(self) -> list[str]:
        names = list(HANDCRAFTED_NAMES)
        if self.use_embedding:
            names = [f"emb_{i}" for i in range(self.embedder.dim)] + names  # type: ignore[union-attr]
        assert not FORBIDDEN_FEATURES & set(names), "label-leaking feature in router features"
        return names

    def transform(self, queries: Sequence[str], embeddings: np.ndarray | None = None) -> np.ndarray:
        hand = np.stack([handcrafted(q) for q in queries]) if queries else np.zeros((0, len(HANDCRAFTED_NAMES)))
        if not self.use_embedding:
            return hand
        if embeddings is None:
            embeddings = self.embedder.embed(list(queries))  # type: ignore[union-attr]
        return np.hstack([embeddings.astype(np.float32), hand])
