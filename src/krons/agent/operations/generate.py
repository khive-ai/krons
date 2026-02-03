# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from krons.agent.message import Instruction, prepare_messages_for_chat
from krons.session import Message, resource_must_be_accessible
from .utils import handle_return, ReturnAs
from ..message.common import CustomRenderer

from krons.resource import iModel
from krons.session import Branch, Session
from krons.work.operations import RequestContext

from krons.core.types import ID, MaybeUnset, ModelConfig, Params, Unset
from pydantic import BaseModel, JsonValue
from dataclasses import dataclass, field

__all__ = ("handle_return", "generate")


@dataclass(frozen=True, slots=True)
class GenerateParams(Params):
    _config = ModelConfig(
        sentinel_additions=frozenset({"none", "empty", "dataclass", "pydantic"})
    )

    instruction: MaybeUnset[Instruction | Message] = Unset
    """Instruction content or Message."""

    primary: MaybeUnset[str] = Unset
    """Primary instruction text."""

    context: MaybeUnset[JsonValue] = Unset
    """Additional context for instruction."""

    imodel: MaybeUnset[iModel | str] = Unset
    """Model to use for generation."""

    images: MaybeUnset[list[str]] = Unset
    """Image URLs for multimodal input."""

    image_detail: MaybeUnset[Literal["low", "high", "auto"]] = Unset
    """Image detail level."""

    tool_schemas: MaybeUnset[list[str]] = Unset
    """Tool schemas for function calling (pass-through to instruction)."""

    request_model: MaybeUnset[type[BaseModel]] = Unset
    """Pydantic model for structured output schema."""

    structure_format: Literal["json", "custom"] = "json"
    """Format for structured output rendering ('json' or 'custom')."""

    custom_renderer: MaybeUnset[CustomRenderer] = Unset
    """Custom renderer for structure_format='custom'. Formats request_model schema."""

    return_as: ReturnAs = ReturnAs.CALLING
    """Output format."""

    imodel_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional kwargs for imodel."""

    @property
    def instruction_message(self) -> Message:
        """Get instruction as Message."""

        if not self.is_sentinel_field("instruction"):
            if isinstance(self.instruction, Message):
                return self.instruction
            if isinstance(self.instruction, Instruction):
                return Message(content=self.instruction)

        content = Instruction.create(
            primary=self.primary,
            context=self.context,
            images=self.images,
            image_detail=self.image_detail,
            tool_schemas=self.tool_schemas,
            request_model=self.request_model,
            structure_format=self.structure_format,
            custom_renderer=self.custom_renderer,
        )
        return Message(content=content)


async def generate(params: GenerateParams, ctx: RequestContext) -> Any:
    session = await ctx.get_session()
    return await _generate(
        session=session,
        branch=ctx.branch,
        instruction=params.instruction_message,
        imodel=params.imodel,
        structure_format=params.structure_format,
        custom_renderer=params.custom_renderer,
        return_as=params.return_as,
        **params.imodel_kwargs,
    )


async def _generate(
    session: Session,
    branch: Branch | str,
    instruction: Message | Instruction | ID[Message],
    imodel: iModel | str | None = None,
    return_as: ReturnAs = ReturnAs.CALLING,
    **imodel_kwargs: Any,
):
    if imodel is None:
        imodel = session.default_gen_model
    else:
        imodel = session.resources.get(imodel, None)
    if imodel is None:
        raise ValueError(
            "Provided imodel could not be resolved, or no default model is set."
        )

    branch = session.get_branch(branch)
    resource_must_be_accessible(branch, imodel.name)

    if isinstance(instruction, UUID):
        instruction = session.messages[instruction]

    elif isinstance(instruction, Instruction):
        instruction = Message(content=instruction)

    prepared_msgs = prepare_messages_for_chat(session.messages, branch, instruction)
    calling = await imodel.invoke(messages=prepared_msgs, **imodel_kwargs)
    return handle_return(calling, return_as)
