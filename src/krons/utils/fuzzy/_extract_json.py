# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""JSON extraction utilities for parsing JSON from strings and markdown blocks."""

import re
from typing import Any

import orjson

from ._fuzzy_json import MAX_JSON_INPUT_SIZE, fuzzy_json

__all__ = ("extract_json",)

_JSON_BLOCK_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
"""Regex for extracting content from ```json ... ``` code blocks."""

_JSON_PARSE_ERRORS = (orjson.JSONDecodeError, ValueError, TypeError)
"""Exceptions indicating parse failure (not system errors like MemoryError)."""


def extract_json(
    input_data: str | list[str],
    /,
    *,
    fuzzy_parse: bool = False,
    return_one_if_single: bool = True,
    max_size: int = MAX_JSON_INPUT_SIZE,
) -> Any | list[Any]:
    """Extract and parse JSON from string or markdown code blocks.

    Parsing strategy:
        1. Try direct JSON parsing of entire input
        2. On failure, extract ```json ... ``` blocks and parse each
        3. Return [] if no valid JSON found

    Args:
        input_data: String or list of strings (joined with newlines).
        fuzzy_parse: Use fuzzy_json() for malformed JSON.
        return_one_if_single: Return single result directly (not wrapped in list).
        max_size: Max input size in bytes (default: 10MB). DoS protection.

    Returns:
        - Single parsed object if return_one_if_single and exactly one result
        - List of parsed objects otherwise
        - [] if no valid JSON found

    Raises:
        ValueError: Input exceeds max_size.

    Edge Cases:
        - Multiple ```json blocks: returns list of all successfully parsed
        - Invalid JSON in some blocks: skipped silently
        - Empty input: returns []
    """
    input_str = "\n".join(input_data) if isinstance(input_data, list) else input_data

    if len(input_str) > max_size:
        msg = (
            f"Input size ({len(input_str)} bytes) exceeds maximum "
            f"({max_size} bytes). This limit prevents memory exhaustion."
        )
        raise ValueError(msg)

    # Try direct parsing first
    try:
        parsed = fuzzy_json(input_str) if fuzzy_parse else orjson.loads(input_str)
        return parsed if return_one_if_single else [parsed]
    except _JSON_PARSE_ERRORS:
        pass

    # Extract from markdown code blocks
    matches = _JSON_BLOCK_PATTERN.findall(input_str)
    if not matches:
        return []

    if return_one_if_single and len(matches) == 1:
        try:
            return fuzzy_json(matches[0]) if fuzzy_parse else orjson.loads(matches[0])
        except _JSON_PARSE_ERRORS:
            return []

    results: list[Any] = []
    for m in matches:
        try:
            parsed = fuzzy_json(m) if fuzzy_parse else orjson.loads(m)
            results.append(parsed)
        except _JSON_PARSE_ERRORS:
            continue
    return results
