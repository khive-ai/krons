# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Generate operation: stateless LLM call with message preparation.

Handler signature: generate(params, ctx) → Calling | text | raw | Message
Lowest-level operation — no message persistence, no validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from pydantic import BaseModel, JsonValue

from krons.agent.message import Instruction, prepare_messages_for_chat
from krons.core.types import ID, MaybeUnset, ModelConfig, Params, Unset
from krons.errors import ConfigurationError
from krons.session import Message, resource_must_be_accessible

from ..message.common import CustomRenderer
from .utils import ReturnAs, handle_return

if TYPE_CHECKING:
    from krons.resource import iModel
    from krons.session import Branch, Session
    from krons.work.operations import RequestContext

__all__ = ("GenerateParams", "generate", "handle_return")


@dataclass(frozen=True, slots=True)
class GenerateParams(Params):
    """Parameters for generate operation.

    Provide either `instruction` (pre-built Message/Instruction) or
    `primary` (string) to auto-build an Instruction via Instruction.create().

    Attributes:
        instruction: Pre-built Instruction or Message (takes priority).
        primary: Instruction text (used when instruction is Unset).
        context: Additional context merged into instruction.
        imodel: Model name or iModel instance (Unset = session default).
        return_as: How to unwrap the Calling result.
        imodel_kwargs: Extra kwargs forwarded to imodel.invoke().
    """

    _config = ModelConfig(
        sentinel_additions=frozenset({"none", "empty", "dataclass", "pydantic"})
    )

    instruction: MaybeUnset[Instruction | Message] = Unset
    primary: MaybeUnset[str] = Unset
    context: MaybeUnset[JsonValue] = Unset
    imodel: MaybeUnset[iModel | str] = Unset
    images: MaybeUnset[list[str]] = Unset
    image_detail: MaybeUnset[Literal["low", "high", "auto"]] = Unset
    tool_schemas: MaybeUnset[list[str]] = Unset
    request_model: MaybeUnset[type[BaseModel]] = Unset
    structure_format: Literal["json", "custom"] = "json"
    custom_renderer: MaybeUnset[CustomRenderer] = Unset
    return_as: ReturnAs = ReturnAs.CALLING
    imodel_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def instruction_message(self) -> Message:
        """Resolve to a Message, building from primary if needed."""
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
    """Generate operation handler: resolve context and delegate to _generate."""
    session = await ctx.get_session()
    imodel = params.imodel if not params.is_sentinel_field("imodel") else None

    # Propagate verbose from ctx to imodel_kwargs for streaming pretty-print
    imodel_kwargs = dict(params.imodel_kwargs)
    if ctx.metadata.get("_verbose"):
        imodel_kwargs.setdefault("verbose", True)

    return await _generate(
        session=session,
        branch=ctx.branch,
        instruction=params.instruction_message,
        imodel=imodel,
        return_as=params.return_as,
        **imodel_kwargs,
    )


async def _generate(
    session: Session,
    branch: Branch | str,
    instruction: Message | Instruction | ID[Message],
    imodel: iModel | str | None = None,
    return_as: ReturnAs = ReturnAs.CALLING,
    **imodel_kwargs: Any,
) -> Any:
    """Core generate: resolve model/branch/instruction → invoke → handle_return.

    Args:
        instruction: Message, Instruction, or message UUID to look up.
        imodel: Model name (resolved from session.resources) or iModel instance.
        return_as: Controls output unwrapping (see ReturnAs).
        **imodel_kwargs: Forwarded to imodel.invoke().
    """
    if imodel is None:
        imodel = session.default_gen_model
    elif isinstance(imodel, str):
        imodel = session.resources.get(imodel, None)
    # else: already an iModel instance
    if imodel is None:
        raise ConfigurationError(
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
