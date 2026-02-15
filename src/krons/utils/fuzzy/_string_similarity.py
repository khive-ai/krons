# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""String similarity algorithms with rapidfuzz backend.

Provides multiple similarity metrics (Jaro-Winkler, Levenshtein, Hamming,
cosine, SequenceMatcher) for fuzzy string matching and correction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

import rapidfuzz.distance as _rf_distance

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ("SimilarityAlgo", "string_similarity")


class SimilarityAlgo(Enum):
    """String similarity algorithms with associated scoring functions."""

    JARO_WINKLER = "jaro_winkler"
    LEVENSHTEIN = "levenshtein"
    SEQUENCE_MATCHER = "sequence_matcher"
    HAMMING = "hamming"
    COSINE = "cosine"

    @classmethod
    def from_potential_valid(cls, v: SimilarityAlgo | str) -> SimilarityAlgo:
        """Convert string to enum, passing through existing enum values."""
        if isinstance(v, cls):
            return v
        try:
            return cls(v)
        except ValueError as e:
            raise ValueError(f"Invalid similarity algorithm: {v}") from e

    @staticmethod
    def cosine_similarity(s1: str, s2: str) -> float:
        """Character-set cosine similarity. Returns 0.0 if either string empty."""
        if not s1 or not s2:
            return 0.0
        set1, set2 = set(s1), set(s2)
        return len(set1 & set2) / ((len(set1) * len(set2)) ** 0.5)

    @staticmethod
    def hamming_similarity(s1: str, s2: str) -> float:
        """Positional match ratio. Requires equal-length strings (else 0.0)."""
        if not s1 or not s2 or len(s1) != len(s2):
            return 0.0
        return sum(c1 == c2 for c1, c2 in zip(s1, s2, strict=False)) / len(s1)

    @staticmethod
    def jaro_winkler_similarity(s: str, t: str, scaling: float = 0.1) -> float:
        """Jaro-Winkler similarity with prefix boost (via rapidfuzz).

        Args:
            s: First string.
            t: Second string.
            scaling: Prefix weight, must be in [0, 0.25].

        Returns:
            Similarity score in [0, 1].

        Raises:
            ValueError: If scaling not in [0, 0.25].
        """
        if not 0 <= scaling <= 0.25:
            raise ValueError("Scaling factor must be between 0 and 0.25")
        return _rf_distance.JaroWinkler.similarity(s, t, prefix_weight=scaling)

    @staticmethod
    def sequence_matcher_similarity(s1: str, s2: str) -> float:
        """Similarity via difflib.SequenceMatcher (longest contiguous match)."""
        from difflib import SequenceMatcher

        return SequenceMatcher(None, s1, s2).ratio()


SIMILARITY_TYPE = Literal[
    "jaro_winkler",
    "levenshtein",
    "sequence_matcher",
    "hamming",
    "cosine",
]

SimilarityFunc = Callable[[str, str], float]
"""Type alias: (str, str) -> float similarity score."""


@dataclass(frozen=True, slots=True)
class MatchResult:
    """String match with score and original index."""

    word: str
    score: float
    index: int


def string_similarity(
    word: str,
    correct_words: Sequence[str],
    algorithm: SimilarityAlgo = SimilarityAlgo.JARO_WINKLER,
    threshold: float = 0.0,
    case_sensitive: bool = False,
    return_most_similar: bool = False,
) -> str | list[str] | None:
    """Find similar strings using specified similarity algorithm.

    Args:
        word: The input string to find matches for
        correct_words: List of strings to compare against
        algorithm: Similarity algorithm to use
        threshold: Minimum similarity score (0.0 to 1.0)
        case_sensitive: Whether to consider case when matching
        return_most_similar: Return only the most similar match

    Returns:
        Matching string(s) or None if no matches found

    Raises:
        ValueError: If correct_words is empty or threshold is invalid
    """
    if not correct_words:
        raise ValueError("correct_words must not be empty")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")

    compare_word = str(word)
    original_words = [str(w) for w in correct_words]

    if not case_sensitive:
        compare_word = compare_word.lower()
        compare_words = [w.lower() for w in original_words]
    else:
        compare_words = original_words.copy()

    algo_name: str | None = None
    score_func: Callable[[str, str], float]
    match algorithm:
        case SimilarityAlgo.JARO_WINKLER:
            algo_name = "jaro_winkler"
            score_func = SimilarityAlgo.jaro_winkler_similarity
        case SimilarityAlgo.LEVENSHTEIN:
            algo_name = "levenshtein"
            score_func = _rf_distance.Levenshtein.similarity
        case SimilarityAlgo.SEQUENCE_MATCHER:
            algo_name = "sequence_matcher"
            score_func = SimilarityAlgo.sequence_matcher_similarity
        case SimilarityAlgo.HAMMING:
            algo_name = "hamming"
            score_func = SimilarityAlgo.hamming_similarity
        case SimilarityAlgo.COSINE:
            algo_name = "cosine"
            score_func = SimilarityAlgo.cosine_similarity
        case _:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    results = []
    for idx, (orig_word, comp_word) in enumerate(zip(original_words, compare_words, strict=False)):
        if algo_name == "hamming" and len(comp_word) != len(compare_word):
            continue  # Hamming requires equal length
        score = score_func(compare_word, comp_word)  # type: ignore[operator]
        if score >= threshold:
            results.append(MatchResult(orig_word, score, idx))

    if not results:
        return None

    results.sort(key=lambda x: (-x.score, x.index))  # Desc score, asc index

    if case_sensitive:  # Keep only top-scoring matches
        max_score = results[0].score
        results = [r for r in results if r.score == max_score]

    if return_most_similar:
        return results[0].word

    return [r.word for r in results]
