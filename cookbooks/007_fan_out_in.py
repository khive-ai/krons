# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Fan-out / fan-in pattern with Claude Code.

Pattern:
  1. Register resources and operations with session
  2. session.conduct() -> orchestrator produces structured Instruct tasks
  3. Tasks fan out to parallel Claude Code instances via session.conduct()
  4. Results fan in and are synthesized by orchestrator via session.conduct()

Usage:
  uv run python cookbooks/007_fan_out_in.py
  uv run python cookbooks/007_fan_out_in.py --simple   # Quick smoke test
"""

from __future__ import annotations

import sys

import anyio
from pydantic import BaseModel, Field

from krons.agent.operations import GenerateParams, Instruct, ReturnAs
from krons.agent.providers.claude_code import (
    ClaudeCodeEndpoint,
    create_claude_code_config,
)
from krons.resource import iModel
from krons.session import Session, SessionConfig
from krons.utils.display import as_readable, display, phase, status
from krons.utils.fuzzy import extract_json, fuzzy_validate_mapping

CC_WORKSPACE = ".khive/workspace"
VERBOSE = True


# ---------------------------------------------------------------------------
# Claude Code factory
# ---------------------------------------------------------------------------


def create_cc(name: str, subdir: str, **kwargs) -> iModel:
    """Create a Claude Code iModel for a workspace subdirectory."""
    config = create_claude_code_config(name=name)
    config.update({"ws": f"{CC_WORKSPACE}/{subdir}", **kwargs})
    endpoint = ClaudeCodeEndpoint(config=config)
    return iModel(backend=endpoint)


# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------


class InvestigationPlan(BaseModel):
    """Orchestrator's structured plan: analysis + parallel Instruct tasks."""

    analysis: str = Field(description="Initial analysis of the codebase")
    research_tasks: list[Instruct] = Field(
        description="Three parallel research instructions for deeper investigation",
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLAN_PROMPT = (
    "Investigate the codebase in the specified directory. "
    "Glance over the key components, pay attention to architecture, "
    "design patterns, and notable features. "
    "Produce three parallel research instructions for deeper investigation."
)

PLAN_PROMPT_SIMPLE = (
    "List the top 3 things to investigate in a Python project. "
    "For each, provide a short research instruction."
)

SYNTHESIS_PROMPT = """\
Synthesize the information from the researcher branches into a cohesive overview:
1. Key components and their roles
2. Architectural patterns used
3. Design patterns and notable features
"""

SYNTHESIS_PROMPT_SIMPLE = """\
Combine the researcher findings into a brief 2-3 sentence summary.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_plan(text: str) -> InvestigationPlan:
    """Extract InvestigationPlan from LLM text output."""
    target_keys = list(InvestigationPlan.model_fields.keys())
    extracted = extract_json(text, fuzzy_parse=True)
    if not extracted:
        raise ValueError("Failed to extract JSON from orchestrator response")

    block = extracted[0] if isinstance(extracted, list) else extracted
    validated = fuzzy_validate_mapping(block, target_keys)
    return InvestigationPlan.model_validate(validated)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(simple: bool = False):
    plan_prompt = PLAN_PROMPT_SIMPLE if simple else PLAN_PROMPT
    synth_prompt = SYNTHESIS_PROMPT_SIMPLE if simple else SYNTHESIS_PROMPT
    context = ["lionagi"] if not simple else None

    # --- 1. Create resources ---
    orc_model = create_cc("orchestrator", "orchestrator")

    # --- 2. Create session with resources and operations ---
    session = Session(
        config=SessionConfig(
            default_branch_name="orchestrator",
            default_gen_model="orchestrator",
            shared_resources={"orchestrator"},
        )
    )
    session.resources.register(orc_model)

    orc_branch = session.default_branch
    status(f"Session ready: {len(session.resources)} resources, branch={orc_branch.name}")

    # --- 3. Phase 1: Plan (structured output from orchestrator) ---
    phase("Phase 1: Planning")

    plan_op = await session.conduct(
        "generate",
        orc_branch,
        GenerateParams(
            primary=plan_prompt,
            context=context,
            request_model=InvestigationPlan,
            imodel="orchestrator",
            return_as=ReturnAs.TEXT,
        ),
        verbose=VERBOSE,
    )
    plan = parse_plan(plan_op.execution.response)

    status(f"Analysis: {plan.analysis[:120]}...")
    status(f"Research tasks: {len(plan.research_tasks)}")
    for i, task in enumerate(plan.research_tasks):
        status(f"  [{i + 1}] {(task.instruction or '')[:80]}...")

    if VERBOSE:
        display(
            as_readable(plan, format_curly=True),
            title="Investigation Plan",
        )

    # --- 4. Phase 2: Fan-out (parallel research) ---
    phase("Phase 2: Parallel Research")
    results: list[str | None] = [None] * len(plan.research_tasks)

    async def run_research(idx: int, task: Instruct) -> None:
        name = f"researcher_{idx}"
        researcher = create_cc(name, name)
        session.resources.register(researcher, update=True)
        branch = session.create_branch(name=name, resources={name})

        instruction = task.instruction or ""
        if task.guidance:
            instruction += f"\n\nGuidance: {task.guidance}"

        op = await session.conduct(
            "generate",
            branch,
            GenerateParams(
                primary=instruction,
                context=task.context,
                imodel=name,
                return_as=ReturnAs.TEXT,
            ),
            verbose=VERBOSE,
        )
        results[idx] = op.execution.response
        status(
            f"Researcher {idx + 1} done ({len(results[idx] or '')} chars)",
            style="success",
        )

    async with anyio.create_task_group() as tg:
        for i, task in enumerate(plan.research_tasks):
            tg.start_soon(run_research, i, task)

    # --- 5. Phase 3: Fan-in (synthesis) ---
    phase("Phase 3: Synthesis")
    research_context = [
        f"--- Researcher {i + 1} ---\n{r}" for i, r in enumerate(results) if r is not None
    ]

    synth_op = await session.conduct(
        "generate",
        orc_branch,
        GenerateParams(
            primary=synth_prompt,
            context=research_context,
            imodel="orchestrator",
            return_as=ReturnAs.TEXT,
        ),
        verbose=VERBOSE,
    )

    phase("Final Synthesis")
    print(synth_op.execution.response)


if __name__ == "__main__":
    simple = "--simple" in sys.argv
    anyio.run(main, simple)
