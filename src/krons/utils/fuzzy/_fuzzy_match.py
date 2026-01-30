from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from krons.types import KeysLike

from krons.types._sentinel import Unset

from ._string_similarity import SimilarityAlgo, string_similarity

__all__ = ("fuzzy_match_keys",)

HandleUnmatched = Literal["ignore", "raise", "remove", "fill", "force"]


class HandleUnmatched(Enum):
    """Strategy for handling unmatched keys in fuzzy_match_keys."""

    IGNORE = "ignore"
    """Keep original keys as-is."""

    RAISE = "raise"
    """Raise ValueError on unmatched keys."""

    REMOVE = "remove"
    """Remove unmatched original keys."""

    FILL = "fill"
    """Fill missing expected keys, keep unmatched original keys."""

    FORCE = "force"
    """Fill missing expected keys, remove unmatched original keys."""


def fuzzy_match_keys(
    d_: dict[str, Any],
    keys: KeysLike,
    /,
    *,
    similarity_algo: SimilarityAlgo = SimilarityAlgo.JARO_WINKLER,
    similarity_threshold: float = 0.85,
    fuzzy_match: bool = True,
    handle_unmatched: HandleUnmatched = HandleUnmatched.IGNORE,
    fill_value: Any = Unset,
    fill_mapping: dict[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate and correct dict keys using fuzzy string matching.

    Args:
        d_: Input dictionary to validate.
        keys: Expected keys (list, tuple, or object with .keys()).
        similarity_algo: Algorithm for string similarity comparison.
        similarity_threshold: Minimum similarity score (0.0-1.0).
        fuzzy_match: Enable fuzzy matching for non-exact matches.
        handle_unmatched: Strategy for unmatched keys (see HandleUnmatched).
        fill_value: Default value for missing expected keys.
        fill_mapping: Per-key values for specific missing keys.
        strict: Raise if any expected keys remain unmatched.

    Returns:
        Dictionary with keys corrected to match expected keys.

    Raises:
        TypeError: If d_ is not dict or keys is None.
        ValueError: If threshold out of range, or unmatched in RAISE mode.
    """
    # Input validation
    if not isinstance(d_, dict):
        raise TypeError("First argument must be a dictionary")
    if keys is None:
        raise TypeError("Keys argument cannot be None")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0.0 and 1.0")

    # Extract expected keys
    if isinstance(keys, (list, tuple)):
        fields_set = set(keys)
    elif hasattr(keys, "keys"):
        fields_set = set(keys.keys())
    else:
        fields_set = set(keys)
    if not fields_set:
        return d_.copy()

    corrected_out = {}
    matched_expected = set()
    matched_input = set()

    # Pass 1: exact matches
    for key in d_:
        if key in fields_set:
            corrected_out[key] = d_[key]
            matched_expected.add(key)
            matched_input.add(key)

    # Pass 2: fuzzy matching
    if fuzzy_match:
        remaining_input = set(d_.keys()) - matched_input
        remaining_expected = fields_set - matched_expected

        for key in remaining_input:
            if not remaining_expected:
                break

            matches = string_similarity(
                key,
                list(remaining_expected),
                algorithm=similarity_algo,
                threshold=similarity_threshold,
                return_most_similar=True,
            )

            if matches:
                match = matches if isinstance(matches, str) else matches[0] if matches else None
                if match:
                    corrected_out[match] = d_[key]
                    matched_expected.add(match)
                    matched_input.add(key)
                    remaining_expected.remove(match)
            elif handle_unmatched == HandleUnmatched.IGNORE:
                corrected_out[key] = d_[key]

    unmatched_input = set(d_.keys()) - matched_input
    unmatched_expected = fields_set - matched_expected

    if handle_unmatched == HandleUnmatched.RAISE and unmatched_input:
        raise ValueError(f"Unmatched keys found: {unmatched_input}")

    elif handle_unmatched == HandleUnmatched.IGNORE:
        for key in unmatched_input:
            corrected_out[key] = d_[key]

    elif handle_unmatched in (HandleUnmatched.FILL, HandleUnmatched.FORCE):
        for key in unmatched_expected:
            corrected_out[key] = (
                fill_mapping[key] if fill_mapping and key in fill_mapping else fill_value
            )

        if handle_unmatched == HandleUnmatched.FILL:
            for key in unmatched_input:
                corrected_out[key] = d_[key]

    if strict and unmatched_expected:
        raise ValueError(f"Missing required keys: {unmatched_expected}")

    return corrected_out
