# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel


from krons.agent.message.common import StructureFormat, CustomParser, CustomRenderer
from krons.core.types import MaybeUnset, Unset, is_sentinel
from krons.errors import ConfigurationError, ExecutionError, KronsError, ValidationError
from krons.resource.imodel import iModel
from krons.utils.fuzzy import HandleUnmatched, extract_json, fuzzy_validate_mapping
from dataclasses import dataclass, field
from krons.core.types import Params, ModelConfig
from krons.work.operations import RequestContext

if TYPE_CHECKING:
    from krons.session import Branch, Session

__all__ = ("parse",)


@dataclass(frozen=True, slots=True)
class ParseParams(Params):
    _config = ModelConfig(sentinel_additions=frozenset({"none", "empty"}))

    text: str
    """Required. Raw text to parse."""

    target_keys: MaybeUnset[list[str]] = Unset
    """Expected keys for fuzzy matching."""

    imodel: iModel | str | None = None
    """Model for LLM reparse fallback."""

    imodel_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional kwargs for imodel."""

    custom_parser: CustomParser | None = None
    """Custom parser for structure_format='custom'. Extracts dict from text."""

    custom_renderer: MaybeUnset[CustomRenderer] = Unset
    """Custom renderer for structure_format='custom'. Formats request_model schema."""

    structure_format: StructureFormat = StructureFormat.JSON
    """Format for parsing output ('json' or 'custom')."""

    tool_schemas: MaybeUnset[list[str]] = Unset
    """Tool schemas for function calling (pass-through to instruction)."""

    request_model: MaybeUnset[type[BaseModel]] = Unset

    similarity_threshold: float = 0.85
    """Fuzzy match threshold."""

    handle_unmatched: HandleUnmatched = HandleUnmatched.FORCE
    """How to handle unmatched keys."""

    max_retries: int = 3
    """Retry attempts for LLM reparse. should be less than 5 to avoid long delays."""

    fill_mapping: dict[str, Any] | None = None

    fill_value: Any = Unset


async def parse(params: ParseParams, ctx: RequestContext) -> dict[str, Any]:
    target_keys = params.target_keys

    if params.is_sentinel_field("target_keys"):
        if params.is_sentinel_field("request_model"):
            raise ValidationError(
                "Either 'target_keys' or 'request_model' must be provided for parse operation"
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
    structure_format: StructureFormat,
    custom_parser: CustomParser,
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
):
    if is_sentinel(target_keys, {"none", "empty"}):
        raise ValidationError("No target_keys provided for parse operation")
    if is_sentinel(text, {"none", "empty"}):
        raise ValidationError("No text provided for parse operation")

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
        if is_sentinel(max_retries, {"none", "empty"}) or max_retries < 1:
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
    structure_format: StructureFormat,
    custom_parser: CustomParser,
    similarity_threshold: float = 0.85,
    handle_unmatched: HandleUnmatched = HandleUnmatched.FORCE,
    fill_mapping: dict[str, Any] | None = None,
    fill_value: Any = Unset,
):
    if is_sentinel(target_keys, {"none", "empty"}):
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

    if is_sentinel(extracted, {"none", "empty"}):
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
