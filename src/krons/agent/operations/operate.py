# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Operate: top-level agent operation chain.

Handler signature: operate(params, ctx) -> validated model instance

Full pipeline:
  1. Compose request structure from operable (inject action spec if needed)
  2. Structure: generate -> parse -> validate (produces typed model)
  3. Act: extract and execute action_requests, persist messages
  4. Compose response structure, merge action_results, validate

Action spec injection:
  If invoke_actions=True AND tool_schemas are present AND the branch
  has "action" in its capabilities, the action_requests spec is injected
  into the operable before composing the request structure. This lets
  the LLM produce structured tool calls alongside regular output fields.

Unlike lionagi's operate, krons does NOT allow runtime spec injection
for arbitrary fields. Only the action spec is injected automatically
based on explicit capability declarations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from krons.core.types import MaybeUnset, ModelConfig, Params, Unset
from krons.session import Message

from .act import ActParams, act
from .generate import GenerateParams
from .specs import Action, ActionResult, get_action_result_spec, get_action_spec
from .structure import StructureParams, structure

if TYPE_CHECKING:
    from krons.agent.message.common import CustomParser
    from krons.core.specs import Operable
    from krons.resource import iModel
    from krons.work.operations import RequestContext
    from krons.work.rules.validator import Validator

__all__ = ("OperateParams", "operate")


@dataclass(frozen=True, slots=True)
class OperateParams(Params):
    """Parameters for operate (structure + act pipeline).

    Flat parameter set — no nested StructureParams. The operate handler
    builds StructureParams internally after runtime composition.

    Attributes:
        operable: Spec definition (required). Used to compose structures.
        validator: Rule-based validator (required).
        generate_params: LLM generation config.
        request_model: Base model type for structured output.
            If None, composed entirely from operable specs.
        capabilities: Field subset for runtime composition.
        invoke_actions: Enable action spec injection and execution.
        action_strategy: "concurrent" (default) or "sequential".
        max_concurrent: Concurrency limit for tool execution.
        throttle_period: Delay between starting tool calls (seconds).
        persist: Persist assistant/action messages to branch.
    """

    _config = ModelConfig(sentinel_additions=frozenset({"none", "empty"}))

    # Required
    operable: Operable
    validator: Validator
    generate_params: GenerateParams

    # Structure composition
    capabilities: set[str] | None = None
    persist: bool = True

    # Action stage
    invoke_actions: bool = False
    action_strategy: Literal["sequential", "concurrent"] = "concurrent"
    max_concurrent: int | None = None
    throttle_period: float | None = None

    # Validation
    auto_fix: bool = True
    strict: bool = True

    # Parse overrides
    parse_imodel: MaybeUnset[iModel | str] = Unset
    parse_imodel_kwargs: dict[str, Any] = field(default_factory=dict)
    custom_parser: CustomParser | None = None
    similarity_threshold: float = 0.85
    max_retries: int = 3
    fill_mapping: dict[str, Any] | None = None
    fill_value: Any = Unset


async def operate(params: OperateParams, ctx: RequestContext) -> Any:
    """Operate handler: compose -> structure -> act -> merge.

    Returns a validated model instance. If actions were invoked,
    the response structure includes action_results.
    """
    session = await ctx.get_session()
    branch = await ctx.get_branch()

    operable = params.operable
    gen_params = params.generate_params

    # Determine if action spec should be injected
    has_tools = not gen_params.is_sentinel_field("tool_schemas")
    branch_caps = getattr(branch, "capabilities", set())
    inject_actions = params.invoke_actions and has_tools and "action" in branch_caps

    # --- Stage 1: Compose request structure ---
    if inject_actions:
        request_operable = operable.extend([get_action_spec()])
    else:
        request_operable = operable

    request_structure = request_operable.compose_structure()

    # Update generate params with the composed request model
    use_gen_params = gen_params.with_updates(
        copy_containers="deep", request_model=request_structure
    )

    # --- Stage 2: Structure (generate -> parse -> validate) ---
    structure_params = StructureParams(
        generate_params=use_gen_params,
        validator=params.validator,
        operable=request_operable,
        structure=request_structure,
        persist=params.persist,
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
    structured = await structure(structure_params, ctx)

    # --- Stage 3: Extract and execute actions ---
    if not inject_actions:
        return structured

    act_requests = getattr(structured, "action_requests", None)
    if not act_requests:
        return structured

    # Convert Action models to ActionRequest messages, persist to branch
    action_messages = _actions_to_messages(act_requests)
    if not action_messages:
        return structured

    for msg in action_messages:
        session.add_message(msg, branches=branch)

    act_params = ActParams(
        action_requests=action_messages,
        strategy=params.action_strategy,
        max_concurrent=params.max_concurrent,
        throttle_period=params.throttle_period,
    )
    action_responses = await act(act_params, ctx)

    # Persist action response messages to branch
    for resp in action_responses:
        resp_msg = Message(content=resp)
        session.add_message(resp_msg, branches=branch)

    # Build ActionResult models from ActionResponse messages
    action_results = _responses_to_results(action_responses, action_messages)

    # --- Stage 4: Compose response structure, merge, return ---
    # No re-validation needed: structured output was validated in stage 2,
    # action_results are execution artifacts (not LLM output).
    response_operable = request_operable.extend([get_action_result_spec()])
    response_structure = response_operable.compose_structure()

    data = request_operable.dump_instance(structured)
    data["action_results"] = action_results

    return response_structure(**data)


def _actions_to_messages(act_requests: list) -> list[Message]:
    """Convert Action models from structured output to ActionRequest Messages."""
    from krons.agent.message.action import ActionRequest

    messages: list[Message] = []
    for req in act_requests:
        if isinstance(req, Action):
            content = ActionRequest.create(
                function=req.function, arguments=req.arguments
            )
            messages.append(Message(content=content))
        elif isinstance(req, dict):
            content = ActionRequest.create(
                function=req.get("function", ""),
                arguments=req.get("arguments", {}),
            )
            messages.append(Message(content=content))
    return messages


def _responses_to_results(
    action_responses: list,
    action_messages: list[Message],
) -> list[ActionResult]:
    """Convert ActionResponse messages to ActionResult spec models.

    Maps request_id back to function name via action_messages.
    """
    from krons.agent.message.action import ActionResponse

    # Build request_id → function lookup from action messages
    id_to_func: dict[str, str] = {}
    for msg in action_messages:
        content = msg.content
        if hasattr(content, "function"):
            id_to_func[str(msg.id)] = content.function

    results: list[ActionResult] = []
    for resp in action_responses:
        if isinstance(resp, ActionResponse):
            func = id_to_func.get(
                resp.request_id if not resp._is_sentinel(resp.request_id) else "",
                "",
            )
            results.append(
                ActionResult(
                    function=func,
                    result=resp.result if resp.success else None,
                    error=resp.error if not resp.success else None,
                )
            )
        elif isinstance(resp, dict):
            results.append(ActionResult.model_validate(resp))
    return results
