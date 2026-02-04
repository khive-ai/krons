# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Multi-Agent Technical Debate with Claude Code.

Three agents with different stances debate a technical question,
then a judge evaluates arguments and delivers a verdict.

Architecture:
    Judge (moderator)
      └─> Advocate (pro position)  ─┐
      └─> Skeptic (con position)    ├─> Judge evaluates
      └─> Pragmatist (balanced)    ─┘

Features:
  - Adversarial debate pattern
  - Structured arguments with evidence
  - Multi-round rebuttals
  - Judge scoring and verdict

Usage:
    uv run python examples/tech_debate.py
    uv run python examples/tech_debate.py --topic "Should we use microservices?"
"""

from __future__ import annotations

import sys

import anyio
from pydantic import BaseModel, Field

from krons.agent.operations import GenerateParams, ReturnAs
from krons.agent.providers.claude_code import (
    ClaudeCodeEndpoint,
    create_claude_code_config,
)
from krons.resource import iModel
from krons.session import Session, SessionConfig
from krons.utils.display import Timer, phase, status
from krons.utils.fuzzy import extract_json, fuzzy_validate_mapping

CC_WORKSPACE = ".khive/examples/tech_debate"
VERBOSE = True


# ---------------------------------------------------------------------------
# Structured Output Models
# ---------------------------------------------------------------------------


class Argument(BaseModel):
    """A single argument in the debate."""

    claim: str = Field(description="The main claim being made")
    evidence: list[str] = Field(description="Supporting evidence or examples")
    counterpoint: str | None = Field(
        default=None, description="Preemptive counter to opposing view"
    )


class DebatePosition(BaseModel):
    """A debater's complete position."""

    stance: str = Field(
        description="The debater's stance (advocate/skeptic/pragmatist)"
    )
    thesis: str = Field(description="Core thesis statement")
    arguments: list[Argument] = Field(description="Supporting arguments")
    conclusion: str = Field(description="Final summary")
    confidence: float = Field(ge=0, le=1, description="Confidence in position")


class Rebuttal(BaseModel):
    """A rebuttal to another position."""

    target_stance: str = Field(description="Which stance this rebuts")
    challenges: list[str] = Field(description="Specific challenges to their arguments")
    strongest_counter: str = Field(description="The strongest counter-argument")


class JudgeVerdict(BaseModel):
    """The judge's final verdict."""

    topic_summary: str = Field(description="Summary of the debate topic")
    strongest_position: str = Field(
        description="Which position had strongest arguments"
    )
    key_insights: list[str] = Field(description="Most valuable insights from debate")
    verdict: str = Field(description="Final recommendation")
    reasoning: str = Field(description="Explanation of the verdict")
    scores: dict[str, int] = Field(description="Scores for each position (0-100)")


# ---------------------------------------------------------------------------
# Claude Code Factory
# ---------------------------------------------------------------------------


def create_cc(
    name: str, subdir: str, system_prompt: str | None = None, **kwargs
) -> iModel:
    """Create a Claude Code iModel."""
    config = create_claude_code_config(name=name)
    config.update({"ws": f"{CC_WORKSPACE}/{subdir}", "max_turns": 3, **kwargs})
    if system_prompt:
        config["system_prompt"] = system_prompt
    endpoint = ClaudeCodeEndpoint(config=config)
    return iModel(backend=endpoint)


# ---------------------------------------------------------------------------
# Debater Personas
# ---------------------------------------------------------------------------

ADVOCATE_PERSONA = """\
You are the ADVOCATE in a technical debate. Your role is to argue FOR the proposition.
Be passionate but rational. Use concrete examples and real-world evidence.
Acknowledge limitations but emphasize benefits and opportunities.
Your goal is to make the strongest possible case for adoption."""

SKEPTIC_PERSONA = """\
You are the SKEPTIC in a technical debate. Your role is to argue AGAINST the proposition.
Be critical but fair. Highlight risks, costs, and hidden complexities.
Use case studies of failures and cautionary tales.
Your goal is to expose weaknesses and potential problems."""

PRAGMATIST_PERSONA = """\
You are the PRAGMATIST in a technical debate. Your role is to find the BALANCED middle ground.
Consider context, trade-offs, and nuance. Avoid extremes.
Propose conditional recommendations: "It depends on..."
Your goal is to provide practical, context-aware guidance."""

JUDGE_PERSONA = """\
You are the JUDGE in a technical debate. Evaluate arguments objectively.
Consider strength of evidence, logical coherence, and practical applicability.
Be fair to all positions but decisive in your verdict.
Your goal is to synthesize the debate into actionable insight."""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def make_position_prompt(topic: str, stance: str) -> str:
    """Create an opening argument prompt."""
    return f"""\
Technical Debate Topic: {topic}

Present your opening position as the {stance.upper()}.

Structure your response as JSON:
- stance: "{stance}"
- thesis: Your core thesis (1-2 sentences)
- arguments: List of 3 arguments, each with:
  - claim: Main point
  - evidence: 2-3 pieces of supporting evidence
  - counterpoint: Preemptive counter to likely objections
- conclusion: Summarize your position
- confidence: 0.0 to 1.0

Be specific, cite real examples, and make your strongest case. Return valid JSON only."""


def make_rebuttal_prompt(stance: str, other_positions: str) -> str:
    """Create a rebuttal prompt."""
    return f"""\
You are the {stance.upper()}. Review your opponents' positions and provide rebuttals.

Other positions:
{other_positions}

Structure your rebuttals as JSON:
- target_stance: Which position you're rebutting
- challenges: 2-3 specific challenges to their arguments
- strongest_counter: Your single strongest counter-argument

Provide rebuttals for each opposing position. Return valid JSON array only."""


VERDICT_PROMPT = """\
As the JUDGE, evaluate this technical debate and deliver your verdict.

Topic: {topic}

Opening Positions:
{positions}

Rebuttals:
{rebuttals}

Structure your verdict as JSON:
- topic_summary: Brief summary of what was debated
- strongest_position: "advocate", "skeptic", or "pragmatist"
- key_insights: List of 3-5 valuable insights from the debate
- verdict: Your final recommendation (2-3 sentences)
- reasoning: Why you reached this verdict (3-4 sentences)
- scores: {{"advocate": 0-100, "skeptic": 0-100, "pragmatist": 0-100}}

Be decisive but fair. Return valid JSON only."""


# ---------------------------------------------------------------------------
# Parsing Helpers
# ---------------------------------------------------------------------------


def parse_position(text: str, stance: str) -> DebatePosition:
    """Parse a debate position."""
    extracted = extract_json(text, fuzzy_parse=True)
    if not extracted:
        return DebatePosition(
            stance=stance,
            thesis="Position could not be parsed",
            arguments=[],
            conclusion="N/A",
            confidence=0.5,
        )

    block = extracted[0] if isinstance(extracted, list) else extracted
    target_keys = list(DebatePosition.model_fields.keys())
    validated = fuzzy_validate_mapping(block, target_keys)
    validated["stance"] = stance
    return DebatePosition.model_validate(validated)


def parse_rebuttals(text: str) -> list[Rebuttal]:
    """Parse rebuttals."""
    extracted = extract_json(text, fuzzy_parse=True)
    if not extracted:
        return []

    items = extracted if isinstance(extracted, list) else [extracted]
    rebuttals = []
    for item in items:
        try:
            target_keys = list(Rebuttal.model_fields.keys())
            validated = fuzzy_validate_mapping(item, target_keys)
            rebuttals.append(Rebuttal.model_validate(validated))
        except Exception:
            continue
    return rebuttals


def parse_verdict(text: str) -> JudgeVerdict:
    """Parse the judge's verdict."""
    extracted = extract_json(text, fuzzy_parse=True)
    if not extracted:
        return JudgeVerdict(
            topic_summary="Verdict could not be parsed",
            strongest_position="pragmatist",
            key_insights=[],
            verdict="Unable to determine",
            reasoning="Parsing failed",
            scores={"advocate": 50, "skeptic": 50, "pragmatist": 50},
        )

    block = extracted[0] if isinstance(extracted, list) else extracted
    target_keys = list(JudgeVerdict.model_fields.keys())
    validated = fuzzy_validate_mapping(block, target_keys)
    return JudgeVerdict.model_validate(validated)


# ---------------------------------------------------------------------------
# Default Topic
# ---------------------------------------------------------------------------

DEFAULT_TOPIC = """\
Should engineering teams adopt AI coding assistants (like Copilot/Cursor/Claude Code)
as standard tooling for all developers?"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(topic: str | None = None):
    """Run the technical debate."""
    debate_topic = topic or DEFAULT_TOPIC

    print("=" * 70)
    print("  Multi-Agent Technical Debate")
    print("=" * 70)
    print()
    print(f"Topic: {debate_topic}")
    print()

    # --- Setup Session ---
    phase("Setting up debate panel")

    judge_model = create_cc("judge", "judge", system_prompt=JUDGE_PERSONA)
    session = Session(
        config=SessionConfig(
            default_branch_name="debate",
            default_gen_model="judge",
            shared_resources={"judge"},
        )
    )
    session.resources.register(judge_model)

    judge_branch = session.default_branch
    status(f"Debate ready: {len(session.resources)} participants")

    # --- Phase 1: Opening Arguments (parallel) ---
    phase("Phase 1: Opening Arguments")

    debaters = [
        ("advocate", ADVOCATE_PERSONA),
        ("skeptic", SKEPTIC_PERSONA),
        ("pragmatist", PRAGMATIST_PERSONA),
    ]

    positions: dict[str, DebatePosition] = {}
    branches: dict[str, object] = {}

    async def present_position(stance: str, persona: str) -> None:
        name = f"debater_{stance}"
        debater = create_cc(name, name, system_prompt=persona)
        session.resources.register(debater, update=True)
        branch = session.create_branch(name=name, resources={name})
        branches[stance] = branch

        prompt = make_position_prompt(debate_topic, stance)

        with Timer() as t:
            op = await session.conduct(
                "generate",
                branch,
                GenerateParams(
                    primary=prompt,
                    request_model=DebatePosition,
                    imodel=name,
                    return_as=ReturnAs.TEXT,
                ),
                verbose=VERBOSE,
            )

        position = parse_position(op.execution.response, stance)
        positions[stance] = position

        status(
            f"[{stance.upper()}] Confidence: {position.confidence:.0%}, "
            f"Arguments: {len(position.arguments)} ({t.elapsed:.1f}s)",
            style="success",
        )

    async with anyio.create_task_group() as tg:
        for stance, persona in debaters:
            tg.start_soon(present_position, stance, persona)

    # --- Display Opening Positions ---
    print()
    for stance, pos in positions.items():
        icon = {"advocate": "✅", "skeptic": "❌", "pragmatist": "⚖️"}[stance]
        print(f"┌─ {icon} {stance.upper()} (Confidence: {pos.confidence:.0%})")
        print(f"│  Thesis: {pos.thesis}")
        for i, arg in enumerate(pos.arguments[:2], 1):
            print(f"│  Arg {i}: {arg.claim[:80]}...")
        print("└" + "─" * 60)
        print()

    # --- Phase 2: Rebuttals (parallel) ---
    phase("Phase 2: Rebuttals")

    all_rebuttals: dict[str, list[Rebuttal]] = {}

    async def present_rebuttal(stance: str) -> None:
        name = f"debater_{stance}"

        # Format other positions for context
        other_positions = "\n\n".join(
            f"=== {s.upper()} ===\nThesis: {p.thesis}\nArguments:\n"
            + "\n".join(f"- {a.claim}" for a in p.arguments)
            for s, p in positions.items()
            if s != stance
        )

        prompt = make_rebuttal_prompt(stance, other_positions)
        branch = branches[stance]

        with Timer() as t:
            op = await session.conduct(
                "generate",
                branch,
                GenerateParams(
                    primary=prompt,
                    imodel=name,
                    return_as=ReturnAs.TEXT,
                ),
                verbose=VERBOSE,
            )

        rebuttals = parse_rebuttals(op.execution.response)
        all_rebuttals[stance] = rebuttals

        status(
            f"[{stance.upper()}] Rebuttals: {len(rebuttals)} ({t.elapsed:.1f}s)",
            style="success",
        )

    async with anyio.create_task_group() as tg:
        for stance, _ in debaters:
            tg.start_soon(present_rebuttal, stance)

    # --- Display Rebuttals ---
    print()
    for stance, rebuttals in all_rebuttals.items():
        if rebuttals:
            print(f"[{stance.upper()}] rebuts:")
            for r in rebuttals[:2]:
                print(f"  → {r.target_stance}: {r.strongest_counter[:60]}...")
    print()

    # --- Phase 3: Judge's Verdict ---
    phase("Phase 3: Judge's Verdict")

    positions_text = "\n\n".join(
        f"=== {s.upper()} ===\n"
        f"Thesis: {p.thesis}\n"
        f"Arguments:\n"
        + "\n".join(f"- {a.claim}" for a in p.arguments)
        + f"\nConclusion: {p.conclusion}"
        for s, p in positions.items()
    )

    rebuttals_text = "\n\n".join(
        f"=== {s.upper()} REBUTTALS ===\n"
        + "\n".join(f"- vs {r.target_stance}: {r.strongest_counter}" for r in rs)
        for s, rs in all_rebuttals.items()
        if rs
    )

    verdict_prompt = VERDICT_PROMPT.format(
        topic=debate_topic,
        positions=positions_text,
        rebuttals=rebuttals_text,
    )

    with Timer() as t:
        verdict_op = await session.conduct(
            "generate",
            judge_branch,
            GenerateParams(
                primary=verdict_prompt,
                request_model=JudgeVerdict,
                imodel="judge",
                return_as=ReturnAs.TEXT,
            ),
            verbose=VERBOSE,
        )

    verdict = parse_verdict(verdict_op.execution.response)
    status(f"Verdict delivered ({t.elapsed:.1f}s)")

    # --- Final Verdict ---
    print()
    print("=" * 70)
    print("  ⚖️  JUDGE'S VERDICT")
    print("=" * 70)
    print()

    print(f"Topic: {verdict.topic_summary}")
    print()

    print("Scores:")
    for stance, score in sorted(verdict.scores.items(), key=lambda x: -x[1]):
        bar = "█" * (score // 5) + "░" * (20 - score // 5)
        winner = " 🏆" if stance == verdict.strongest_position else ""
        print(f"  {stance:12} [{bar}] {score}/100{winner}")
    print()

    print("Key Insights:")
    for i, insight in enumerate(verdict.key_insights, 1):
        print(f"  {i}. {insight}")
    print()

    print("Verdict:")
    print(f"  {verdict.verdict}")
    print()

    print("Reasoning:")
    print(f"  {verdict.reasoning}")
    print()
    print("─" * 70)


if __name__ == "__main__":
    topic = None
    if "--topic" in sys.argv:
        idx = sys.argv.index("--topic")
        if idx + 1 < len(sys.argv):
            topic = sys.argv[idx + 1]

    anyio.run(main, topic)
