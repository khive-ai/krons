# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Structure operation: generate → parse → validate pipeline.

Handler signature: structure(params, ctx) → validated dict or structure instance

Three-stage pipeline:
  1. generate: LLM call to produce raw text (forced return_as=TEXT)
  2. parse: extract JSON from text (direct + LLM reparse fallback)
  3. validate: enforce operable specs + optional Pydantic structure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from krons.core.types import MaybeUnset, ModelConfig, Params, Unset, is_unset

from .generate import GenerateParams, generate
from .parse import ParseParams, parse
from .utils import ReturnAs

if TYPE_CHECKING:
    from krons.agent.message.common import CustomParser
    from krons.core.specs import Operable
    from krons.resource import iModel
    from krons.work.operations import RequestContext
    from krons.work.rules.validator import Validator

__all__ = ("StructureParams", "structure")


@dataclass(frozen=True, slots=True)
class StructureParams(Params):
    """Parameters for structure operation (generate → parse → validate).

    Attributes:
        generate_params: LLM generation config (return_as is forced to TEXT).
        validator: Rule-based validator for operable spec enforcement.
        operable: Spec definition for field validation.
        structure: Optional Pydantic model to cast the validated dict into.
        capabilities: Allowed field subset (None = all operable fields).
        auto_fix: Auto-coerce validation issues (e.g., wrap scalar → list).
        strict: Raise on validation failure vs. skip.
        parse_imodel: Separate model for LLM reparse (None = same as generate).
        parse_imodel_kwargs: Extra kwargs for parse model invocation.
        custom_parser: Custom parser for non-JSON structure formats.
        similarity_threshold: Fuzzy key matching threshold (0-1).
        max_retries: LLM reparse attempts on direct parse failure.
        fill_mapping: Default values for missing keys during fuzzy match.
        fill_value: Scalar default for any unmatched key.
    """

    _config = ModelConfig(sentinel_additions=frozenset({"none", "empty"}))

    # Generate stage
    generate_params: GenerateParams

    # Validate stage
    validator: Validator
    operable: Operable
    structure: MaybeUnset[type[BaseModel]] = Unset
    capabilities: set[str] | None = None
    auto_fix: bool = True
    strict: bool = True

    # Parse stage overrides
    parse_imodel: MaybeUnset[iModel | str] = Unset
    parse_imodel_kwargs: dict[str, Any] = field(default_factory=dict)
    custom_parser: CustomParser | None = None
    similarity_threshold: float = 0.85
    max_retries: int = 3
    fill_mapping: dict[str, Any] | None = None
    fill_value: Any = Unset


async def structure(params: StructureParams, ctx: RequestContext) -> dict[str, Any]:
    """Structure operation handler: generate → parse → validate."""
    return await _structure(
        generate_params=params.generate_params,
        validator=params.validator,
        ctx=ctx,
        operable=params.operable,
        structure=params.structure,
        capabilities=params.capabilities,
        auto_fix=params.auto_fix,
        strict=params.strict,
        parse_imodel=params.parse_imodel,
        parse_imodel_kwargs=params.parse_imodel_kwargs,
        custom_parser=params.custom_parser,
        similarity_threshold=params.similarity_threshold,
        max_retries=params.max_retries,
        fill_mapping=params.fill_mapping,
        fill_value=params.fill_value,
    )


async def _structure(
    generate_params: GenerateParams,
    validator: Validator,
    ctx: RequestContext,
    operable: Operable,
    structure: MaybeUnset[type[BaseModel]] = Unset,
    capabilities: set[str] | None = None,
    auto_fix: bool = True,
    strict: bool = True,
    parse_imodel: MaybeUnset[iModel | str] = Unset,
    parse_imodel_kwargs: dict[str, Any] | None = None,
    custom_parser: CustomParser | None = None,
    similarity_threshold: float = 0.85,
    max_retries: int = 3,
    fill_mapping: dict[str, Any] | None = None,
    fill_value: Any = Unset,
) -> dict[str, Any]:
    """Core pipeline: generate text → parse JSON → validate against operable.

    The generate step is forced to return_as=TEXT so parse receives a string.
    Parse params inherit request_model/tool_schemas/structure_format from
    generate_params, ensuring schema consistency across stages.
    """
    # Stage 1: Generate (force TEXT output for parse consumption)
    gen_params = generate_params.with_updates(
        copy_containers="deep", return_as=ReturnAs.TEXT
    )
    text = await generate(gen_params, ctx)

    # Stage 2: Parse (inherit schema config from generate params)
    parse_params = ParseParams(
        text=text,
        imodel=parse_imodel if parse_imodel is not Unset else None,
        imodel_kwargs=parse_imodel_kwargs or {},
        custom_parser=custom_parser,
        similarity_threshold=similarity_threshold,
        max_retries=max_retries,
        fill_mapping=fill_mapping,
        fill_value=fill_value,
        request_model=gen_params.request_model,
        tool_schemas=gen_params.tool_schemas,
        structure_format=gen_params.structure_format,
        custom_renderer=gen_params.custom_renderer,
    )
    parsed = await parse(parse_params, ctx)

    # Stage 3: Validate against operable specs + optional structure type
    return await validator.validate(
        parsed,
        operable,
        capabilities=capabilities,
        auto_fix=auto_fix,
        strict=strict,
        structure=structure if not is_unset(structure) else None,
    )
