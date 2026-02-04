# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Minimal test: one structured output via session.conduct("structure").

Usage:
  uv run python cookbooks/test_conduct.py
"""

from __future__ import annotations

import anyio
from pydantic import BaseModel, Field

from krons.agent.operations import (
    GenerateParams,
    Instruct,
    ReturnAs,
    StructureParams,
    generate,
    structure,
)
from krons.agent.providers.claude_code import ClaudeCodeEndpoint, create_claude_code_config
from krons.core.specs import Operable
from krons.resource import iModel
from krons.session import Session, SessionConfig
from krons.utils.display import as_readable, display, phase, status
from krons.work.rules.validator import Validator


class Plan(BaseModel):
    analysis: str = Field(description="One sentence analysis")
    tasks: list[Instruct] = Field(description="Two short research tasks")


async def main():
    # Setup
    config = create_claude_code_config(name="test")
    endpoint = ClaudeCodeEndpoint(config=config)
    model = iModel(backend=endpoint)

    session = Session(
        config=SessionConfig(
            default_branch_name="main",
            default_gen_model="test",
            shared_resources={"test"},
        )
    )
    session.resources.register(model)
    session.operations.register("structure", structure)

    phase("Structured Output Test (structure operation)")

    operable = Operable.from_structure(Plan)

    op = await session.conduct(
        "structure",
        params=StructureParams(
            generate_params=GenerateParams(
                primary="Name two colors. For each, give a one-sentence research task.",
                request_model=Plan,
                imodel="test",
            ),
            validator=Validator(),
            operable=operable,
            strict=False,
        ),
        verbose=True,
    )

    result = op.execution.response
    status(f"Result type: {type(result).__name__}")
    display(as_readable(result, format_curly=True), title="Structured Output")


if __name__ == "__main__":
    anyio.run(main)
