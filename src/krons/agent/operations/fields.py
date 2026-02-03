# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Structured field models for agent operations.

ActionRequestModel: Pydantic model for parsing LLM tool-call output into
validated (function, arguments) pairs. Handles fuzzy/malformed JSON from
various providers.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from krons.core.types import HashableModel
from krons.utils import extract_json, to_dict, to_list


class ActionRequestModel(HashableModel):
    """Validated tool/action request extracted from LLM output.

    Attributes:
        function: Function name from tool_schemas (never invented).
        arguments: Argument dict matching the function's schema.
    """

    function: str = Field(
        description=(
            "Name of the function to call from the provided `tool_schemas`. "
            "If no `tool_schemas` exist, set to None or leave blank. "
            "Never invent new function names outside what's given."
        ),
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Dictionary of arguments for the chosen function. "
            "Use only argument names/types defined in `tool_schemas`. "
            "Never introduce extra argument names."
        ),
    )

    @field_validator("arguments", mode="before")
    @classmethod
    def _coerce_arguments(cls, value: Any) -> dict[str, Any]:
        """Coerce arguments into a dict, handling JSON strings and nested structures."""
        if isinstance(value, dict):
            return value
        return to_dict(
            value,
            fuzzy_parse=True,
            recursive=True,
            recursive_python_only=False,
        )

    @classmethod
    def create(cls, content: str | dict | BaseModel) -> list[ActionRequestModel]:
        """Parse raw LLM output into validated ActionRequestModel instances.

        Handles:
        - JSON objects/arrays with function/arguments keys
        - Python code blocks containing JSON
        - Aliased key names (name→function, parameters→arguments, etc.)

        Returns empty list on parse failure (never raises).
        """
        try:
            parsed = _parse_action_blocks(content)
            return [cls.model_validate(item) for item in parsed] if parsed else []
        except Exception:
            return []


def _parse_action_blocks(content: str | dict | BaseModel) -> list[dict]:
    """Extract action request dicts from raw content.

    Normalizes provider-specific key names to {function, arguments}.
    """
    json_blocks: list = []

    if isinstance(content, BaseModel):
        json_blocks = [content.model_dump()]
    elif isinstance(content, str):
        json_blocks = extract_json(content, fuzzy_parse=True)
        if not json_blocks:
            # Fallback: try extracting from ```python ... ``` blocks
            matches = re.findall(r"```python\s*(.*?)\s*```", content, re.DOTALL)
            json_blocks = to_list(
                [extract_json(m, fuzzy_parse=True) for m in matches],
                dropna=True,
            )
    elif isinstance(content, dict):
        json_blocks = [content]

    if json_blocks and not isinstance(json_blocks, list):
        json_blocks = [json_blocks]

    out: list[dict] = []
    for block in json_blocks:
        if not isinstance(block, dict):
            continue
        normalized = _normalize_action_keys(block)
        if normalized:
            out.append(normalized)
    return out


def _normalize_action_keys(d: dict) -> dict | None:
    """Map provider-specific key names to canonical {function, arguments}.

    Returns None if required keys are missing.
    """
    result: dict[str, Any] = {}

    # Handle nested function.name pattern
    if "function" in d and isinstance(d["function"], dict) and "name" in d["function"]:
        d = {**d, "function": d["function"]["name"]}

    for k, v in d.items():
        # Strip common prefixes: action_name → name, recipient_name → name
        normalized = k.replace("action_", "").replace("recipient_", "").replace("s", "")
        if normalized in ("name", "function", "recipient"):
            result["function"] = v
        elif normalized in ("parameter", "argument", "arg", "param"):
            result["arguments"] = to_dict(
                v, str_type="json", fuzzy_parse=True, suppress=True
            )

    if "function" in result and "arguments" in result and result["arguments"]:
        return result
    return None
