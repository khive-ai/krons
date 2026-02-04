# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Agent operations: composable LLM pipeline stages.

Handlers (async handler(params, ctx) -> result):
    generate: Stateless LLM call with message preparation.
    parse: JSON extraction with LLM reparse fallback.
    structure: generate -> parse -> validate pipeline.
    operate: structure + action execution + response composition.
    act: Tool/action execution from structured output.
    react / react_stream: Multi-round reason-act loop.

Spec models:
    Action, ActionResult: Tool call request/result models.
    Instruct: Task handoff bundle for orchestration.
    ReActAnalysis, PlannedAction, Analysis: ReAct loop models.

Built-in handlers are auto-registered on Session creation:
    generate, structure, operate, react, react_stream

Usage:
    result = await session.conduct("operate", branch, params)
    async for analysis in session.stream_conduct("react_stream", params=...):
        print(analysis)
"""

from __future__ import annotations

from .act import ActParams, act
from .generate import GenerateParams, generate
from .operate import OperateParams, operate
from .parse import ParseParams, parse
from .react import (
    Analysis,
    PlannedAction,
    ReActAnalysis,
    ReActParams,
    react,
    react_stream,
)
from .specs import (
    Action,
    ActionResult,
    Instruct,
    get_action_result_spec,
    get_action_spec,
    get_instruct_spec,
)
from .structure import StructureParams, structure
from .utils import ReturnAs

__all__ = (
    # Handlers
    "act",
    "generate",
    "operate",
    "parse",
    "react",
    "react_stream",
    "structure",
    # Params
    "ActParams",
    "GenerateParams",
    "OperateParams",
    "ParseParams",
    "ReActParams",
    "StructureParams",
    # Spec models
    "Action",
    "ActionResult",
    "Analysis",
    "Instruct",
    "PlannedAction",
    "ReActAnalysis",
    # Spec factories
    "get_action_result_spec",
    "get_action_spec",
    "get_instruct_spec",
    # Utils
    "ReturnAs",
)
