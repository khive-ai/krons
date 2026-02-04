# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""krons Usage Examples

This module provides practical, runnable examples demonstrating krons patterns
with Claude Code integration.

Multi-Agent Patterns (parallel Claude Code):
    code_review_panel   - 3 specialist reviewers + moderator synthesis
    tech_debate         - Adversarial debate with advocate/skeptic/pragmatist + judge

Worker Patterns (declarative workflows):
    validation_loop     - Self-correcting LLM generation with bounded retries
    codegen_pipeline    - Code generation with error recovery workflow
    research_agent      - Adaptive depth research with confidence-based routing

Exchange Patterns (async message passing):
    pipeline_router     - Multi-stage data processing pipeline
    event_sourcing      - Event persistence, snapshots, and replay

Specs Patterns (dynamic schema composition):
    dynamic_response    - Runtime Pydantic model generation with validators

Combined Patterns:
    multi_agent_orchestration - Supervisor-worker coordination via Exchange + Worker

Run any example:
    uv run python examples/code_review_panel.py
    uv run python examples/tech_debate.py
    uv run python examples/validation_loop.py
    # etc.
"""

__all__ = [
    # Multi-agent patterns (featured)
    "code_review_panel",
    "tech_debate",
    # Worker patterns
    "validation_loop",
    "codegen_pipeline",
    "research_agent",
    # Exchange patterns
    "pipeline_router",
    "event_sourcing",
    # Specs patterns
    "dynamic_response",
    # Combined patterns
    "multi_agent_orchestration",
]
