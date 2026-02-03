# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Parse operation: extract structured JSON from raw LLM text.

Handler signature: parse(params, ctx) → dict[str, Any]

Two-stage pipeline:
  1. _direct_parse: regex/fuzzy extraction (fast, no LLM call)
  2. _llm_reparse: LLM-assisted fallback (up to max_retries)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from krons.agent.message.common import CustomParser, CustomRenderer, StructureFormat
from krons.core.types import MaybeUnset, ModelConfig, Params, Unset, is_sentinel
from krons.errors import ConfigurationError, ExecutionError, KronsError, ValidationError
from krons.utils.fuzzy import HandleUnmatched, extract_json, fuzzy_validate_mapping

if TYPE_CHECKING:
    from krons.resource.imodel import iModel
    from krons.session import Branch, Session
    from krons.work.operations import RequestContext

__all__ = ("ParseParams", "parse")


@dataclass(frozen=True, slots=True)
class ParseParams(Params):
    """Parameters for parse operation.

    Attributes:
        text: Raw text to parse (required).
        target_keys: Expected keys for fuzzy matching (or derived from request_model).
        imodel: Model for LLM reparse fallback.
        structure_format: JSON (default) or custom parser.
        max_retries: LLM reparse attempts (1-5, 0 = direct only).
    """

    _config = ModelConfig(sentinel_additions=frozenset({"none", "empty"}))

    text: str
    target_keys: MaybeUnset[list[str]] = Unset
    imodel: iModel | str | None = None
    imodel_kwargs: dict[str, Any] = field(default_factory=dict)
    custom_parser: CustomParser | None = None
    custom_renderer: MaybeUnset[CustomRenderer] = Unset
    structure_format: StructureFormat = StructureFormat.JSON
    tool_schemas: MaybeUnset[list[str]] = Unset
    request_model: MaybeUnset[type[BaseModel]] = Unset
    similarity_threshold: float = 0.85
    handle_unmatched: HandleUnmatched = HandleUnmatched.FORCE
    max_retries: int = 3
    fill_mapping: dict[str, Any] | None = None
    fill_value: Any = Unset


async def parse(params: ParseParams, ctx: RequestContext) -> dict[str, Any]:
    """Parse operation handler: resolve target_keys and delegate to _parse."""
    target_keys = params.target_keys

    if params.is_sentinel_field("target_keys"):
        if params.is_sentinel_field("request_model"):
            raise ValidationError(
                "Either 'target_keys' or 'request_model' must be provided for parse"
            )
        target_keys = list(params.request_model.model_fields.keys())

    session = ctx.get_session()
    data = params.to_dict(exclude={"target_keys", "imodel_kwargs"})

    return await _parse(
        session=session,
        branch=ctx.branch,
        target_keys=target_keys,
        **data,
        **params.imodel_kwargs,
    )


async def _parse(
    session: Session,
    branch: Branch | str,
    text: str,
    target_keys: list[str],
    structure_format: StructureFormat = StructureFormat.JSON,
    custom_parser: CustomParser | None = None,
    similarity_threshold: float = 0.85,
    handle_unmatched: HandleUnmatched = HandleUnmatched.FORCE,
    fill_mapping: dict[str, Any] | None = None,
    fill_value: Any = Unset,
    max_retries: MaybeUnset[int] = Unset,
    imodel: iModel | str | None = None,
    tool_schemas: MaybeUnset[list[str]] = Unset,
    request_model: MaybeUnset[type[BaseModel]] = Unset,
    custom_renderer: MaybeUnset[CustomRenderer] = Unset,
    **imodel_kwargs: Any,
) -> dict[str, Any]:
    """Two-stage parse: try direct extraction, fall back to LLM reparse.

    Raises:
        ValidationError: Missing required params.
        ExecutionError: All parse attempts failed.
    """
    _sentinel_check = {"none", "empty"}
    if is_sentinel(target_keys, _sentinel_check):
        raise ValidationError("No target_keys provided for parse operation")
    if is_sentinel(text, _sentinel_check):
        raise ValidationError("No text provided for parse operation")

    # Stage 1: direct parse (no LLM call)
    try:
        return _direct_parse(
            text=text,
            target_keys=target_keys,
            structure_format=structure_format,
            custom_parser=custom_parser,
            similarity_threshold=similarity_threshold,
            handle_unmatched=handle_unmatched,
            fill_mapping=fill_mapping,
            fill_value=fill_value,
        )
    except KronsError as e:
        if e.retryable is False:
            raise
    except Exception as e:
        # Stage 2: LLM reparse fallback
        if is_sentinel(max_retries, _sentinel_check) or max_retries < 1:
            raise ExecutionError(
                "Direct parse failed and max_retries not enabled, no reparse attempted",
                retryable=False,
                cause=e,
            )

        from .llm_reparse import _llm_reparse

        for _ in range(max_retries):
            try:
                return await _llm_reparse(
                    session=session,
                    branch=branch,
                    text=text,
                    imodel=imodel,
                    tool_schemas=tool_schemas,
                    request_model=request_model,
                    structure_format=structure_format,
                    custom_renderer=custom_renderer,
                    custom_parser=custom_parser,
                    **imodel_kwargs,
                )
            except KronsError as e:
                if e.retryable is False:
                    raise

    raise ExecutionError(
        "All parse attempts (direct and LLM reparse) failed",
        retryable=False,
    )


def _direct_parse(
    text: str,
    target_keys: list[str],
    structure_format: StructureFormat = StructureFormat.JSON,
    custom_parser: CustomParser | None = None,
    similarity_threshold: float = 0.85,
    handle_unmatched: HandleUnmatched = HandleUnmatched.FORCE,
    fill_mapping: dict[str, Any] | None = None,
    fill_value: Any = Unset,
) -> dict[str, Any]:
    """Extract JSON from text without LLM assistance.

    Routes to custom_parser or built-in JSON extraction + fuzzy matching.
    """
    _sentinel_check = {"none", "empty"}
    if is_sentinel(target_keys, _sentinel_check):
        raise ValidationError("No target_keys provided for direct_parse operation")

    match structure_format:
        case StructureFormat.CUSTOM:
            if not callable(custom_parser):
                raise ConfigurationError(
                    "structure_format='custom' requires a custom_parser to be provided",
                    retryable=False,
                )
            try:
                return custom_parser(text, target_keys)
            except Exception as e:
                raise ExecutionError(
                    "Custom parser failed to extract data from text",
                    retryable=True,
                    cause=e,
                )

        case StructureFormat.JSON:
            pass

        case _:
            raise ValidationError(
                f"Unsupported structure_format '{structure_format}' in direct_parse",
                retryable=False,
            )

    extracted = Unset
    try:
        extracted = extract_json(text, fuzzy_parse=True, return_one_if_single=False)
    except Exception as e:
        raise ExecutionError(
            "Failed to extract JSON from text during parse",
            retryable=True,
            cause=e,
        )

    if is_sentinel(extracted, _sentinel_check):
        raise ExecutionError(
            "No JSON object could be extracted from text during parse",
            retryable=True,
        )

    try:
        return fuzzy_validate_mapping(
            extracted[0],
            target_keys,
            similarity_threshold=similarity_threshold,
            handle_unmatched=handle_unmatched,
            fill_mapping=fill_mapping,
            fill_value=fill_value,
        )
    except Exception as e:
        raise ExecutionError(
            "Failed to validate extracted JSON during parse",
            retryable=True,
            cause=e,
        )
