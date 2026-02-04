# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Multi-Agent Code Review Panel with Claude Code.

A panel of specialized reviewers analyze code from different perspectives,
then a moderator synthesizes their findings into actionable feedback.

Architecture:
    Moderator (orchestrator)
      └─> SecurityReviewer   ─┐
      └─> PerformanceReviewer ├─> Moderator synthesizes
      └─> ArchitectureReviewer┘

Features:
  - Structured outputs with Pydantic models
  - Parallel fan-out to specialist agents
  - Severity-scored findings
  - Synthesized final verdict

Usage:
    uv run python examples/code_review_panel.py
    uv run python examples/code_review_panel.py --file path/to/code.py
"""

from __future__ import annotations

import sys
from enum import Enum

import anyio
from pydantic import BaseModel, Field

from krons.agent.operations import GenerateParams, ReturnAs
from krons.agent.providers.claude_code import (
    ClaudeCodeEndpoint,
    create_claude_code_config,
)
from krons.resource import iModel
from krons.session import Session, SessionConfig
from krons.utils.display import Timer, as_readable, display, phase, status
from krons.utils.fuzzy import extract_json, fuzzy_validate_mapping

CC_WORKSPACE = ".khive/examples/code_review_panel"
VERBOSE = True


# ---------------------------------------------------------------------------
# Structured Output Models
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """Issue severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ReviewFinding(BaseModel):
    """A single finding from a reviewer."""

    title: str = Field(description="Short title of the finding")
    severity: Severity = Field(description="Severity level")
    description: str = Field(description="Detailed description of the issue")
    suggestion: str = Field(description="How to fix or improve")
    line_hint: str | None = Field(
        default=None, description="Relevant code location hint"
    )


class SpecialistReview(BaseModel):
    """Complete review from a specialist."""

    perspective: str = Field(
        description="The review perspective (security/performance/architecture)"
    )
    summary: str = Field(description="Executive summary of findings")
    findings: list[ReviewFinding] = Field(description="List of specific findings")
    score: int = Field(ge=0, le=100, description="Overall score 0-100")


class PanelVerdict(BaseModel):
    """Final synthesized verdict from the panel."""

    overall_assessment: str = Field(description="High-level assessment of code quality")
    critical_issues: list[str] = Field(description="Must-fix issues before merge")
    recommendations: list[str] = Field(description="Suggested improvements")
    approval_status: str = Field(description="APPROVED, NEEDS_CHANGES, or REJECTED")
    confidence: float = Field(ge=0, le=1, description="Panel confidence in verdict")


# ---------------------------------------------------------------------------
# Claude Code Factory
# ---------------------------------------------------------------------------


def create_cc(
    name: str, subdir: str, system_prompt: str | None = None, **kwargs
) -> iModel:
    """Create a Claude Code iModel with optional system prompt."""
    config = create_claude_code_config(name=name)
    config.update({"ws": f"{CC_WORKSPACE}/{subdir}", "max_turns": 3, **kwargs})
    if system_prompt:
        config["system_prompt"] = system_prompt
    endpoint = ClaudeCodeEndpoint(config=config)
    return iModel(backend=endpoint)


# ---------------------------------------------------------------------------
# Reviewer Personas
# ---------------------------------------------------------------------------

SECURITY_PERSONA = """\
You are a senior security engineer conducting a security-focused code review.
Focus on: input validation, injection vulnerabilities, authentication/authorization,
secrets handling, cryptography usage, and OWASP Top 10 issues.
Be thorough but practical - prioritize real risks over theoretical concerns."""

PERFORMANCE_PERSONA = """\
You are a performance engineer reviewing code for efficiency.
Focus on: algorithmic complexity, memory usage, I/O patterns, caching opportunities,
unnecessary allocations, blocking operations, and scalability concerns.
Consider both micro-optimizations and architectural performance."""

ARCHITECTURE_PERSONA = """\
You are a software architect reviewing code structure and design.
Focus on: SOLID principles, separation of concerns, dependency management,
testability, maintainability, API design, and code organization.
Balance pragmatism with clean architecture principles."""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def make_review_prompt(code: str, perspective: str) -> str:
    """Create a review prompt for a specific perspective."""
    return f"""\
Review the following code from a {perspective} perspective.

```python
{code}
```

Provide your review as structured JSON matching this schema:
- perspective: "{perspective}"
- summary: Executive summary (2-3 sentences)
- findings: List of findings, each with:
  - title: Short issue title
  - severity: one of "critical", "high", "medium", "low", "info"
  - description: What's wrong and why it matters
  - suggestion: How to fix it
  - line_hint: Approximate location (optional)
- score: Overall score 0-100 for this perspective

Be specific and actionable. Return valid JSON only."""


SYNTHESIS_PROMPT = """\
You are the panel moderator. Synthesize the specialist reviews into a final verdict.

Reviews from specialists:
{reviews}

Provide your verdict as structured JSON:
- overall_assessment: 2-3 sentence summary of code quality
- critical_issues: List of must-fix issues (empty if none)
- recommendations: List of suggested improvements
- approval_status: "APPROVED", "NEEDS_CHANGES", or "REJECTED"
- confidence: 0.0 to 1.0 confidence in this verdict

Consider all perspectives equally. Be decisive but fair. Return valid JSON only."""


# ---------------------------------------------------------------------------
# Parsing Helpers
# ---------------------------------------------------------------------------


def parse_review(text: str, perspective: str) -> SpecialistReview:
    """Parse a specialist review from LLM output."""
    extracted = extract_json(text, fuzzy_parse=True)
    if not extracted:
        # Fallback for unparseable responses
        return SpecialistReview(
            perspective=perspective,
            summary="Review could not be parsed",
            findings=[],
            score=50,
        )

    block = extracted[0] if isinstance(extracted, list) else extracted
    target_keys = list(SpecialistReview.model_fields.keys())
    validated = fuzzy_validate_mapping(block, target_keys)
    validated["perspective"] = perspective  # Ensure correct perspective
    return SpecialistReview.model_validate(validated)


def parse_verdict(text: str) -> PanelVerdict:
    """Parse the final verdict from moderator."""
    extracted = extract_json(text, fuzzy_parse=True)
    if not extracted:
        return PanelVerdict(
            overall_assessment="Verdict could not be parsed",
            critical_issues=[],
            recommendations=[],
            approval_status="NEEDS_CHANGES",
            confidence=0.5,
        )

    block = extracted[0] if isinstance(extracted, list) else extracted
    target_keys = list(PanelVerdict.model_fields.keys())
    validated = fuzzy_validate_mapping(block, target_keys)
    return PanelVerdict.model_validate(validated)


# ---------------------------------------------------------------------------
# Sample Code for Review
# ---------------------------------------------------------------------------

SAMPLE_CODE = """\
import sqlite3
import hashlib
from flask import Flask, request, jsonify

app = Flask(__name__)
db = sqlite3.connect("users.db", check_same_thread=False)

@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]

    # Hash password
    pw_hash = hashlib.md5(password.encode()).hexdigest()

    # Check credentials
    cursor = db.cursor()
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{pw_hash}'"
    result = cursor.execute(query).fetchone()

    if result:
        return jsonify({"status": "success", "user": result})
    return jsonify({"status": "failed"})

@app.route("/users")
def get_users():
    cursor = db.cursor()
    users = cursor.execute("SELECT * FROM users").fetchall()
    return jsonify(users)

def process_data(items):
    result = []
    for item in items:
        for i in range(len(items)):
            if items[i] == item:
                result.append(item * 2)
    return list(set(result))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(code: str | None = None):
    """Run the code review panel."""
    code_to_review = code or SAMPLE_CODE

    print("=" * 70)
    print("  Multi-Agent Code Review Panel")
    print("=" * 70)
    print()

    # --- Setup Session ---
    phase("Setting up review panel")

    moderator = create_cc("moderator", "moderator")
    session = Session(
        config=SessionConfig(
            default_branch_name="panel",
            default_gen_model="moderator",
            shared_resources={"moderator"},
        )
    )
    session.resources.register(moderator)

    panel_branch = session.default_branch
    status(f"Panel ready: {len(session.resources)} resources")

    # --- Phase 1: Parallel Specialist Reviews ---
    phase("Phase 1: Specialist Reviews (parallel)")

    reviewers = [
        ("security", SECURITY_PERSONA),
        ("performance", PERFORMANCE_PERSONA),
        ("architecture", ARCHITECTURE_PERSONA),
    ]

    reviews: dict[str, SpecialistReview] = {}

    async def run_review(perspective: str, persona: str) -> None:
        name = f"reviewer_{perspective}"
        reviewer = create_cc(name, name, system_prompt=persona)
        session.resources.register(reviewer, update=True)
        branch = session.create_branch(name=name, resources={name})

        prompt = make_review_prompt(code_to_review, perspective)

        with Timer() as t:
            op = await session.conduct(
                "generate",
                branch,
                GenerateParams(
                    primary=prompt,
                    request_model=SpecialistReview,
                    imodel=name,
                    return_as=ReturnAs.TEXT,
                ),
                verbose=VERBOSE,
            )

        review = parse_review(op.execution.response, perspective)
        reviews[perspective] = review

        status(
            f"[{perspective.upper()}] Score: {review.score}/100, "
            f"Findings: {len(review.findings)} ({t.elapsed:.1f}s)",
            style="success" if review.score >= 70 else "warning",
        )

    async with anyio.create_task_group() as tg:
        for perspective, persona in reviewers:
            tg.start_soon(run_review, perspective, persona)

    # --- Display Individual Reviews ---
    print()
    for perspective, review in reviews.items():
        print(f"┌─ {perspective.upper()} REVIEW (Score: {review.score}/100)")
        print(f"│  {review.summary}")
        for finding in review.findings[:3]:  # Show top 3 findings
            severity_icon = {
                Severity.CRITICAL: "🔴",
                Severity.HIGH: "🟠",
                Severity.MEDIUM: "🟡",
                Severity.LOW: "🟢",
                Severity.INFO: "ℹ️",
            }.get(finding.severity, "•")
            print(f"│  {severity_icon} [{finding.severity.value}] {finding.title}")
        if len(review.findings) > 3:
            print(f"│  ... and {len(review.findings) - 3} more findings")
        print("└" + "─" * 60)
        print()

    # --- Phase 2: Moderator Synthesis ---
    phase("Phase 2: Panel Synthesis")

    reviews_text = "\n\n".join(
        f"=== {p.upper()} REVIEWER (Score: {r.score}/100) ===\n"
        f"Summary: {r.summary}\n"
        f"Findings:\n"
        + "\n".join(
            f"- [{f.severity.value}] {f.title}: {f.description}" for f in r.findings
        )
        for p, r in reviews.items()
    )

    synthesis_prompt = SYNTHESIS_PROMPT.format(reviews=reviews_text)

    with Timer() as t:
        synth_op = await session.conduct(
            "generate",
            panel_branch,
            GenerateParams(
                primary=synthesis_prompt,
                request_model=PanelVerdict,
                imodel="moderator",
                return_as=ReturnAs.TEXT,
            ),
            verbose=VERBOSE,
        )

    verdict = parse_verdict(synth_op.execution.response)
    status(f"Synthesis complete ({t.elapsed:.1f}s)")

    # --- Final Verdict ---
    print()
    print("=" * 70)
    print("  PANEL VERDICT")
    print("=" * 70)
    print()

    status_icon = {
        "APPROVED": "✅",
        "NEEDS_CHANGES": "⚠️",
        "REJECTED": "❌",
    }.get(verdict.approval_status, "❓")

    print(f"Status: {status_icon} {verdict.approval_status}")
    print(f"Confidence: {verdict.confidence:.0%}")
    print()
    print("Assessment:")
    print(f"  {verdict.overall_assessment}")
    print()

    if verdict.critical_issues:
        print("Critical Issues (must fix):")
        for issue in verdict.critical_issues:
            print(f"  🔴 {issue}")
        print()

    if verdict.recommendations:
        print("Recommendations:")
        for rec in verdict.recommendations:
            print(f"  💡 {rec}")
        print()

    # --- Summary Stats ---
    avg_score = sum(r.score for r in reviews.values()) / len(reviews)
    total_findings = sum(len(r.findings) for r in reviews.values())
    critical_count = sum(
        1
        for r in reviews.values()
        for f in r.findings
        if f.severity == Severity.CRITICAL
    )

    print("─" * 70)
    print(f"Average Score: {avg_score:.0f}/100")
    print(f"Total Findings: {total_findings}")
    print(f"Critical Issues: {critical_count}")
    print("─" * 70)


if __name__ == "__main__":
    code = None
    if "--file" in sys.argv:
        idx = sys.argv.index("--file")
        if idx + 1 < len(sys.argv):
            with open(sys.argv[idx + 1]) as f:
                code = f.read()

    anyio.run(main, code)
