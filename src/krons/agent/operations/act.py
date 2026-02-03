# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Act operation: execute tool calls from LLM action requests.

Handler signature: act(params, ctx) → list[ActionResponse]

Supports sequential and concurrent execution strategies with
rate-limiting via alcall (delay, throttle, max_concurrent).
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal

from krons.agent.message.action import ActionRequest, ActionResponse
from krons.core.types import Params
from krons.session.constraints import resource_must_be_accessible, resource_must_exist
from krons.utils import alcall

if TYPE_CHECKING:
    from krons.resource.backend import Calling
    from krons.session import Branch, Message, Session
    from krons.work.operations import RequestContext


class ActParams(Params):
    """Parameters for tool execution.

    Attributes:
        action_requests: Messages with ActionRequest content to execute.
        delay_before_start: Initial delay before first execution (seconds).
        throttle_period: Delay between starting tasks (seconds).
        max_concurrent: Concurrency limit (None = unlimited).
        strategy: "concurrent" (default) or "sequential".
    """

    action_requests: list[Message]
    delay_before_start: float = 0
    throttle_period: float | None = None
    max_concurrent: int | None = None
    strategy: Literal["sequential", "concurrent"] = "concurrent"


async def act(params: ActParams, ctx: RequestContext) -> list[ActionResponse]:
    """Execute tool calls from action_requests."""
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
    """Execute action requests against session-registered tools.

    Validates resource existence and branch access before execution.
    Returns ActionResponse per request (with result or error).
    """
    if not action_requests:
        return []

    # Validate all resources exist and are accessible before execution
    for req in action_requests:
        content: ActionRequest = req.content
        resource_must_exist(session, content.function)
        resource_must_be_accessible(branch, content.function)

    async def _execute_one(req_msg: Message) -> ActionResponse:
        """Execute a single action request and normalize result."""
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

    if strategy == "sequential":
        results: list[ActionResponse] = []
        for req_msg in action_requests:
            results.append(await _execute_one(req_msg))
        return results

    return await partial(
        alcall,
        delay_before_start=delay_before_start,
        throttle_period=throttle_period,
        max_concurrent=max_concurrent,
    )(action_requests, _execute_one)
