# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""ReAct: multi-round reason-act loop built on operate.

Handler signature: react(params, ctx) -> final answer instance
Streaming variant: react_stream(params, ctx) -> AsyncGenerator[round analysis]

Each round:
  1. operate() with ReActAnalysis as request model
  2. LLM produces reasoning + planned_actions + extension_needed
  3. If actions present and allowed, operate handles execution
  4. If extension_needed and rounds remain, loop continues
  5. Final round: operate() with user's response model for the answer
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field, field_validator

from krons.core.types import HashableModel, MaybeUnset, ModelConfig, Params, Unset

from .generate import GenerateParams
from .operate import OperateParams, operate

if TYPE_CHECKING:
    from krons.core.specs import Operable
    from krons.resource import iModel
    from krons.work.operations import RequestContext
    from krons.work.rules.validator import Validator

__all__ = (
    "Analysis",
    "PlannedAction",
    "ReActAnalysis",
    "ReActParams",
    "react",
    "react_stream",
)


# ---------------------------------------------------------------------------
# ReAct spec models
# ---------------------------------------------------------------------------


class PlannedAction(HashableModel):
    """Short descriptor for an upcoming tool invocation."""

    action_type: str | None = Field(
        default=None,
        description="Name or type of tool/action to invoke.",
    )
    description: str | None = Field(
        default=None,
        description="Concise summary of what the action entails and why.",
    )


class ReActAnalysis(HashableModel):
    """Structured reasoning output for each ReAct round.

    The LLM fills this to express its chain-of-thought, plan actions,
    and signal whether more rounds are needed.
    """

    FIRST_ROUND_PROMPT: ClassVar[str] = (
        "You can perform multiple reason-action steps for accuracy. "
        "If you are not ready to finalize, set extension_needed to True. "
        "Set extension_needed to True if the overall goal is not yet achieved. "
        "Do not set it to False if you are just providing an interim answer. "
        "You have up to {max_rounds} rounds. Strategize accordingly."
    )
    CONTINUE_PROMPT: ClassVar[str] = (
        "Another round is available. You may do multiple actions if needed. "
        "You have up to {remaining} rounds remaining. Continue."
    )
    ANSWER_PROMPT: ClassVar[str] = (
        "Given your reasoning and actions, provide the final answer "
        "to the user's request:\n\n{instruction}"
    )

    analysis: str = Field(
        ...,
        description=(
            "Free-form reasoning or chain-of-thought summary. "
            "Use for planning, reflection, and progress tracking."
        ),
    )
    planned_actions: list[PlannedAction] = Field(
        default_factory=list,
        description="Tool calls or operations to perform this round.",
    )
    extension_needed: bool = Field(
        False,
        description="True if more rounds are needed. False triggers final answer.",
    )
    milestone: str | None = Field(
        None,
        description="Sub-goal or checkpoint to reach before finalizing.",
    )
    action_strategy: Literal["sequential", "concurrent"] = Field(
        "concurrent",
        description="How to execute planned actions: sequential or concurrent.",
    )


class Analysis(HashableModel):
    """Final answer model (default response_model for react)."""

    answer: str | None = None

    @field_validator("answer", mode="before")
    @classmethod
    def _validate_answer(cls, value: Any) -> str | None:
        if not value:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        if not isinstance(value, str):
            raise ValueError("Answer must be a non-empty string.")
        return value.strip()


# ---------------------------------------------------------------------------
# ReAct params
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReActParams(Params):
    """Parameters for ReAct loop.

    Attributes:
        instruction: The user's task/question.
        operable: Spec definition for structured output composition.
        validator: Rule-based validator.
        generate_params: LLM generation config.
        max_rounds: Maximum reason-act rounds before forcing final answer.
        response_model: Model for the final answer (default: Analysis).
        invoke_actions: Enable tool execution in each round.
        action_strategy: Default strategy (overridden by LLM per-round).
        persist: Persist messages to branch.
    """

    _config = ModelConfig(sentinel_additions=frozenset({"none", "empty"}))

    # Required
    instruction: str
    operable: Operable
    validator: Validator
    generate_params: GenerateParams

    # ReAct loop config
    max_rounds: int = 3
    response_model: type | None = None
    invoke_actions: bool = True
    persist: bool = True

    # Action defaults
    action_strategy: Literal["sequential", "concurrent"] = "concurrent"
    max_concurrent: int | None = None
    throttle_period: float | None = None

    # Validation
    auto_fix: bool = True
    strict: bool = True

    # Parse overrides
    parse_imodel: MaybeUnset[iModel | str] = Unset
    parse_imodel_kwargs: dict[str, Any] = field(default_factory=dict)
    similarity_threshold: float = 0.85
    max_retries: int = 3


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def react(params: ReActParams, ctx: RequestContext) -> Any:
    """ReAct handler: collects all rounds, returns final answer.

    The final answer is extracted from the last round's Analysis.answer
    if response_model is not provided (defaults to Analysis).
    """
    result = None
    async for round_result in react_stream(params, ctx):
        result = round_result

    # Last yield is the final answer
    if result is not None and hasattr(result, "answer"):
        return result.answer
    return result


async def react_stream(
    params: ReActParams, ctx: RequestContext
) -> AsyncGenerator[Any, None]:
    """Streaming ReAct: yields each round's structured analysis.

    Yields:
        Round 1..N: ReActAnalysis instances (reasoning + action results)
        Final: response_model instance (the answer)
    """
    max_rounds = min(params.max_rounds, 100)

    # --- Round 1: Initial analysis ---
    instruction_with_prompt = (
        params.instruction
        + "\n\n"
        + ReActAnalysis.FIRST_ROUND_PROMPT.format(max_rounds=max_rounds)
    )

    analysis = await _run_round(params, ctx, instruction_with_prompt, ReActAnalysis)
    yield analysis

    # --- Extension rounds ---
    remaining = max_rounds - 1
    while remaining > 0 and _needs_extension(analysis):
        prompt = ReActAnalysis.CONTINUE_PROMPT.format(remaining=remaining)
        analysis = await _run_round(params, ctx, prompt, ReActAnalysis)
        yield analysis
        remaining -= 1

    # --- Final answer ---
    answer_model = params.response_model or Analysis
    answer_prompt = ReActAnalysis.ANSWER_PROMPT.format(instruction=params.instruction)
    final = await _run_round(
        params, ctx, answer_prompt, answer_model, invoke_actions=False
    )
    yield final


async def _run_round(
    params: ReActParams,
    ctx: RequestContext,
    instruction: str,
    request_model: type,
    invoke_actions: bool | None = None,
) -> Any:
    """Execute a single ReAct round via operate."""
    from krons.core.specs import Operable

    # Build operable from the round's request model
    round_operable = Operable.from_structure(request_model)

    # Override instruction in generate params
    gen_params = params.generate_params.with_updates(
        copy_containers="deep", primary=instruction
    )

    # Resolve action strategy from previous analysis if available
    should_act = invoke_actions if invoke_actions is not None else params.invoke_actions

    operate_params = OperateParams(
        operable=round_operable,
        validator=params.validator,
        generate_params=gen_params,
        invoke_actions=should_act,
        action_strategy=params.action_strategy,
        max_concurrent=params.max_concurrent,
        throttle_period=params.throttle_period,
        persist=params.persist,
        auto_fix=params.auto_fix,
        strict=params.strict,
        parse_imodel=params.parse_imodel,
        parse_imodel_kwargs=params.parse_imodel_kwargs,
        similarity_threshold=params.similarity_threshold,
        max_retries=params.max_retries,
    )

    return await operate(operate_params, ctx)


def _needs_extension(analysis: Any) -> bool:
    """Check if the analysis signals more rounds are needed."""
    if hasattr(analysis, "extension_needed"):
        return bool(analysis.extension_needed)
    if isinstance(analysis, dict):
        return bool(analysis.get("extension_needed", False))
    return False
