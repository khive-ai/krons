# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Literal
from krons.core.types import HashableModel, Params
from pydantic import Field, field_validator, BaseModel
from krons.resource import Calling
from krons.utils import alcall, to_dict, to_list, extract_json
import re

from krons.agent.message.common import StructureFormat
from krons.agent.message.action import ActionRequest, ActionResponse

from krons.session import Session, Branch, Message
from krons.session.constraints import resource_must_be_accessible, resource_must_exist
from krons.work.operations import RequestContext


class ActParams(Params):
    action_requests: list[Message]
    """List of Messages containing ActionRequest content."""

    delay_before_start: float = 0
    """Delay before starting execution (seconds)."""

    throttle_period: float | None = None
    """Delay between starting tasks (seconds)."""

    max_concurrent: int | None = None
    """Max concurrent executions (default 10)."""

    strategy: Literal["sequential", "concurrent"] = "concurrent"


async def act(params: ActParams, ctx: RequestContext):
    session = await ctx.get_session()
    return await _act(session=session, branch=ctx.branch, **params.to_dict())


async def _act(
    action_requests: list[Message],
    session: Session,
    branch: Branch,
    delay_before_start: float = 0,
    throttle_period: float | None = None,
    max_concurrent: int | None = None,
    strategy: Literal["sequential", "concurrent"] = "concurrent",
) -> list[ActionResponse]:
    """Execute tool calls from action_requests.

    Args:
        action_requests: Tool calls from LLM structured output.
        session: Session containing registered tools.
        branch: Branch for resource access check and message persistence.
        max_concurrent: Max concurrent executions (default 10).
        retry_timeout: Timeout per tool call.
        retry_attempts: Retry attempts on failure.
        throttle_period: Delay between starting tasks.

    Returns:
        List of ActionResponse objects with execution results.
    """
    if not action_requests:
        return []

    for req in action_requests:
        content: ActionRequest = req.content
        resource_must_exist(session, content.function)
        resource_must_be_accessible(branch, content.function)

    async def _act_one(
        req_msg: Message,
    ):
        """Execute a single action request."""
        action_request: ActionRequest = req_msg.content
        calling: Calling = await session.request(
            action_request.function, branch=branch, **action_request.arguments
        )
        try:
            calling.assert_is_normalized()
        except Exception as e:
            return ActionResponse(
                request_id=str(req_msg.id), error=f"ExecutionError: {e}"
            )

        return ActionResponse(request_id=str(req_msg.id), result=calling.response.data)

    _alcall = partial(
        alcall,
        delay_before_start=delay_before_start,
        throttle_period=throttle_period,
        max_concurrent=max_concurrent,
    )

    if strategy == "sequential":
        results = []
        for req_msg in action_requests:
            res = await _alcall(req_msg, _act_one)
            results.append(res)
        return results

    return await _alcall(action_requests, _act_one)
