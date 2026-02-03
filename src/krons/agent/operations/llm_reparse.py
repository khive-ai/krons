# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""LLM-assisted reparse: use a model to reformat malformed text into valid JSON.

Called as a fallback when direct parse (regex/fuzzy) fails.
Constructs an Instruction asking the model to extract structured data,
then fuzzy-validates the result against target keys.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from krons.agent.message import Instruction
from krons.core.types import MaybeUnset, Unset
from krons.utils.fuzzy import HandleUnmatched, fuzzy_validate_mapping

from ..message.common import CustomParser, CustomRenderer
from .utils import ReturnAs

if TYPE_CHECKING:
    from krons.resource import iModel
    from krons.session import Branch, Session

__all__ = ("_llm_reparse",)

PARSE_PROMPT = (
    "Reformat text into specified model or structure, "
    "using the provided schema format as a guide"
)


async def _llm_reparse(
    session: Session,
    branch: Branch,
    text: str,
    imodel: iModel | str,
    tool_schemas: MaybeUnset[list[str]] = Unset,
    request_model: MaybeUnset[type[BaseModel]] = Unset,
    structure_format: MaybeUnset[Literal["json", "custom"]] = Unset,
    custom_renderer: MaybeUnset[CustomRenderer] = Unset,
    custom_parser: CustomParser | None = None,
    fill_mapping: dict[str, Any] | None = None,
    **imodel_kwargs: Any,
) -> dict[str, Any]:
    """Ask LLM to reformat text into structured JSON.

    Builds an Instruction with the text as context and the target
    schema from request_model, then generates and parses the result.

    Returns:
        Dict mapping target keys to extracted values.
    """
    instruction = Instruction.create(
        primary=PARSE_PROMPT,
        context=[{"text_to_format": text}],
        request_model=request_model,
        tool_schemas=tool_schemas,
        structure_format=structure_format,
        custom_renderer=custom_renderer,
    )

    from .generate import _generate

    res = await _generate(
        session=session,
        branch=branch,
        instruction=instruction,
        imodel=imodel,
        return_as=ReturnAs.TEXT,
        **imodel_kwargs,
    )

    target_keys = list(request_model.model_fields.keys())

    if custom_parser is not None:
        return custom_parser(res, target_keys)

    return fuzzy_validate_mapping(
        res,
        target_keys,
        handle_unmatched=HandleUnmatched.FORCE,
        fill_mapping=fill_mapping,
    )
