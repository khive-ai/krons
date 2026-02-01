# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal

from krons.agent.message import Instruction, prepare_messages_for_chat
from krons.errors import ValidationError
from krons.resource import NormalizedResponse
from krons.session import Message, resource_must_be_accessible

from .constraints import resolve_response_is_normalized, response_must_be_completed
from .types import CustomRenderer, GenerateParams, ReturnAs

if TYPE_CHECKING:
    from krons.core.types import Enum, MaybeUnset, Unset, is_sentinel
    from krons.resource import iModel
    from krons.resource.backend import Calling
    from krons.session import Branch, Session
    from krons.work.operations import RequestContext

__all__ = ("handle_return", "generate")


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
    instruction: Message | Instruction,
    imodel: iModel | str | None = None,
    structure_format: Literal["json", "custom"] = "json",
    custom_renderer: CustomRenderer | None = None,
    return_as: Literal["text", "raw", "response", "message", "calling"] = "calling",
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

    if isinstance(instruction, Instruction):
        instruction = Message(content=instruction)

    prepared_msgs = prepare_messages_for_chat(
        session.messages,
        branch,
        instruction,
        to_chat=True,
        structure_format=structure_format,
        custom_renderer=custom_renderer,
    )
    calling = await imodel.invoke(messages=prepared_msgs, **imodel_kwargs)
    return handle_return(calling, return_as)


class ReturnAs(Enum):
    CALLING = "calling"
    DATA = "data"
    RAW = "raw"
    NORMALIZED = "normalized"
    CUSTOM = "custom"


def parse_to_assistant_response(response: NormalizedResponse) -> Message:
    from krons.agent.message import Assistant

    metadata_dict: dict[str, Any] = {"raw_response": response.raw_response}
    if response.metadata is not None:
        metadata_dict.update(response.metadata)

    return Message(
        content=Assistant.create(assistant_response=response.data),
        metadata=metadata_dict,
    )


def handle_return(
    calling: Calling, return_as: ReturnAs, /, return_parser: Callable = None
):
    if return_as == ReturnAs.CALLING:
        return calling

    calling.assert_is_normalized()
    response = calling.response
    match return_as:
        case ReturnAs.DATA:
            return response.data
        case ReturnAs.RAW:
            return response.raw_response
        case ReturnAs.NORMALIZED:
            return response
        case ReturnAs.MESSAGE:
            from krons.agent.message import Assistant

            metadata_dict: dict[str, Any] = {
                "raw_response": calling.response.raw_response
            }
            if response.metadata is not None:
                metadata_dict.update(response.metadata)

            return Message(
                content=Assistant.create(assistant_response=response.data),
                metadata=metadata_dict,
            )
        case _:
            raise ValidationError(f"Unsupported return_as: {return_as}")
