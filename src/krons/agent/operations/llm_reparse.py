from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from krons.agent.message import Instruction
from krons.core.types import MaybeUnset, Unset
from krons.resource import iModel
from krons.utils.fuzzy import HandleUnmatched, fuzzy_validate_mapping

from ..message.common import CustomParser, CustomRenderer
from .utils import ReturnAs

if TYPE_CHECKING:
    from krons.session import Branch, Session

__all__ = ("_llm_reparse",)


PARSE_PROMPT = "Reformat text into specified model or structure, using the provided schema format as a guide"


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
        structure_format=structure_format,
        custom_renderer=custom_renderer,
        return_as=ReturnAs.TEXT,
        **imodel_kwargs,
    )
    if custom_parser is not None:
        return custom_parser(res, list(request_model.model_fields.keys()))

    return fuzzy_validate_mapping(
        res,
        list(request_model.model_fields.keys()),
        handle_unmatched=HandleUnmatched.FORCE,
        fill_mapping=fill_mapping,
    )
