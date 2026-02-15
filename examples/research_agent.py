# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Research agent with dynamic depth using Claude Code.

Demonstrates adaptive research workflow with confidence-based routing:
1. search: Gather initial information via Claude Code
2. analyze: Analyze findings and assess confidence
3. deep_dive: If confidence < threshold, do deeper research
4. synthesize: Create final report

Usage:
    uv run python examples/research_agent.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import anyio

from krons.agent.third_party.claude_code import (
    ClaudeCodeRequest,
    ClaudeSession,
    stream_claude_code_cli,
)
from krons.work import Worker, WorkerEngine, work, worklink

WORKSPACE = ".khive/examples/research_agent"
MODEL = "sonnet"


async def call_claude(
    prompt: str,
    system_prompt: str | None = None,
    max_turns: int = 1,
) -> ClaudeSession:
    """Call Claude Code and return the session."""
    request = ClaudeCodeRequest(
        prompt=prompt,
        system_prompt=system_prompt,
        model=MODEL,
        ws=WORKSPACE,
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        verbose=False,
    )

    session = ClaudeSession()
    async for chunk in stream_claude_code_cli(request, session):
        if isinstance(chunk, ClaudeSession):
            return chunk

    return session


def extract_confidence(text: str) -> float:
    """Extract confidence score from text."""
    # Look for explicit confidence patterns
    patterns = [
        r"CONFIDENCE:\s*(\d+(?:\.\d+)?)",
        r"confidence[:\s]+(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*%?\s*confidence",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            return value / 100 if value > 1 else value

    # Default based on content quality signals
    if "comprehensive" in text.lower() or "thorough" in text.lower():
        return 0.85
    if "partial" in text.lower() or "incomplete" in text.lower():
        return 0.5
    return 0.7


def extract_gaps(text: str) -> list[str]:
    """Extract knowledge gaps from text."""
    gaps = []

    # Look for explicit gap sections
    if "GAPS:" in text.upper():
        gap_section = text[text.upper().find("GAPS:") + 5 :]
        gap_section = gap_section.split("\n\n")[0]
        for line in gap_section.split("\n"):
            line = line.strip().lstrip("-•*")
            if line and len(line) > 10:
                gaps.append(line.strip())

    # Also look for bullet points after "missing" or "unclear"
    for marker in ["missing", "unclear", "need more", "further research"]:
        if marker in text.lower():
            idx = text.lower().find(marker)
            section = text[idx : idx + 300]
            for line in section.split("\n")[1:4]:
                line = line.strip().lstrip("-•*")
                if line and len(line) > 5:
                    gaps.append(line.strip())
                    break

    return gaps[:5]  # Max 5 gaps


@dataclass
class ResearchResult:
    """Result of research operation."""

    query: str
    findings: str
    confidence: float
    gaps: list[str]
    depth: int = 0
    cost: float = 0.0


class ResearchWorker(Worker):
    """Worker that performs adaptive depth research via Claude Code.

    Workflow:
        search -> analyze -> (if low confidence) -> deep_dive -> synthesize
                          -> (if high confidence) -> synthesize
    """

    name = "research_agent"
    confidence_threshold: float = 0.75

    def __init__(self, confidence_threshold: float = 0.75) -> None:
        super().__init__()
        self.confidence_threshold = confidence_threshold

    @work(assignment="query -> search_results", timeout=120.0)
    async def search(self, query: str, **kwargs) -> dict:
        """Search for information via Claude Code.

        Args:
            query: Research query

        Returns:
            Dict with search results
        """
        print(f"[search] Researching: {query[:50]}...")

        prompt = f"""Research the following topic and provide comprehensive findings:

{query}

Requirements:
- Gather key facts and information
- Include multiple perspectives if relevant
- Note any limitations or gaps in knowledge
- Be specific with examples and details

Provide your findings in a structured format."""

        session = await call_claude(
            prompt=prompt,
            system_prompt="You are an expert researcher. Gather comprehensive, accurate information.",
        )

        findings = session.result or "No findings"
        print(f"[search] Found {len(findings)} chars (cost: ${session.total_cost_usd or 0:.4f})")

        return {
            "query": query,
            "findings": findings,
            "depth": 0,
            "cost": session.total_cost_usd or 0,
        }

    @work(assignment="query, findings, depth -> analysis", timeout=90.0)
    async def analyze(self, query: str, findings: str, depth: int = 0, **kwargs) -> ResearchResult:
        """Analyze findings and assess confidence.

        Args:
            query: Original query
            findings: Research findings
            depth: Current research depth

        Returns:
            ResearchResult with confidence and gaps
        """
        print(f"[analyze] Analyzing findings (depth={depth})...")

        prompt = f"""Analyze these research findings for the query: "{query}"

Findings:
{findings[:3000]}

Provide:
1. A brief summary of key points
2. Your confidence in the completeness (0.0 to 1.0)
3. Any gaps or areas needing more research

Format your response as:
SUMMARY: <brief summary>
CONFIDENCE: <number between 0.0 and 1.0>
GAPS:
- <gap 1>
- <gap 2>
..."""

        session = await call_claude(
            prompt=prompt,
            system_prompt="You are a critical analyst. Honestly assess research quality and gaps.",
        )

        result = session.result or ""
        confidence = extract_confidence(result)
        gaps = extract_gaps(result)

        print(f"[analyze] Confidence: {confidence:.0%}, Gaps: {len(gaps)}")

        return ResearchResult(
            query=query,
            findings=findings,
            confidence=confidence,
            gaps=gaps,
            depth=depth,
            cost=session.total_cost_usd or 0,
        )

    @work(assignment="query, gaps -> deep_findings", timeout=120.0)
    async def deep_dive(self, query: str, gaps: list[str], **kwargs) -> dict:
        """Perform deep research on identified gaps.

        Args:
            query: Original query
            gaps: Knowledge gaps to address

        Returns:
            Dict with additional findings
        """
        print(f"[deep_dive] Researching {len(gaps)} gaps...")

        gaps_text = "\n".join(f"- {gap}" for gap in gaps)
        prompt = f"""The original research query was: "{query}"

The following gaps were identified:
{gaps_text}

Please research these specific gaps and provide detailed findings for each.
Focus on addressing the missing information."""

        session = await call_claude(
            prompt=prompt,
            system_prompt="You are a thorough researcher. Fill in knowledge gaps with specific details.",
        )

        findings = session.result or "No additional findings"
        print(f"[deep_dive] Found {len(findings)} chars (cost: ${session.total_cost_usd or 0:.4f})")

        return {
            "query": query,
            "deep_findings": findings,
            "gaps_addressed": gaps,
            "cost": session.total_cost_usd or 0,
        }

    @work(assignment="query, findings, deep_findings -> report", timeout=90.0)
    async def synthesize(
        self, query: str, findings: str, deep_findings: str | None = None, **kwargs
    ) -> dict:
        """Synthesize all findings into a final report.

        Args:
            query: Original query
            findings: Initial findings
            deep_findings: Optional deep dive findings

        Returns:
            Dict with final report
        """
        print("[synthesize] Creating final report...")

        all_findings = findings
        if deep_findings:
            all_findings += f"\n\nAdditional Research:\n{deep_findings}"

        prompt = f"""Create a comprehensive research report for: "{query}"

Based on the following findings:
{all_findings[:4000]}

Create a well-structured report with:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Conclusions and Recommendations"""

        session = await call_claude(
            prompt=prompt,
            system_prompt="You are an expert report writer. Create clear, actionable reports.",
        )

        report = session.result or "Report generation failed"
        print(
            f"[synthesize] Report: {len(report)} chars (cost: ${session.total_cost_usd or 0:.4f})"
        )

        return {
            "query": query,
            "report": report,
            "had_deep_dive": deep_findings is not None,
            "cost": session.total_cost_usd or 0,
        }

    @worklink(from_="search", to_="analyze")
    async def search_to_analyze(self, result: dict) -> dict:
        """Route search -> analyze."""
        return {
            "query": result["query"],
            "findings": result["findings"],
            "depth": result["depth"],
        }

    @worklink(from_="analyze", to_="deep_dive")
    async def analyze_to_dive(self, result: ResearchResult) -> dict | None:
        """Route to deep_dive if confidence is low."""
        if result.confidence >= self.confidence_threshold:
            print(
                f"[analyze->deep_dive] SKIP: confidence {result.confidence:.0%} >= {self.confidence_threshold:.0%}"
            )
            return None

        if not result.gaps:
            print("[analyze->deep_dive] SKIP: no gaps identified")
            return None

        print(f"[analyze->deep_dive] ROUTE: confidence {result.confidence:.0%} < threshold")
        return {
            "query": result.query,
            "gaps": result.gaps,
        }

    @worklink(from_="analyze", to_="synthesize")
    async def analyze_to_synth(self, result: ResearchResult) -> dict | None:
        """Route directly to synthesize if confidence is high."""
        if result.confidence < self.confidence_threshold and result.gaps:
            return None  # Go via deep_dive

        print(f"[analyze->synthesize] Direct route: confidence {result.confidence:.0%}")
        return {
            "query": result.query,
            "findings": result.findings,
            "deep_findings": None,
        }

    @worklink(from_="deep_dive", to_="synthesize")
    async def dive_to_synth(self, result: dict) -> dict:
        """Route deep_dive -> synthesize."""
        return {
            "query": result["query"],
            "findings": result.get("findings", ""),
            "deep_findings": result["deep_findings"],
        }


async def main():
    """Run the research agent with Claude Code."""
    print("=" * 60)
    print("Research Agent - Claude Code Integration")
    print("=" * 60)
    print()

    worker = ResearchWorker(confidence_threshold=0.75)
    engine = WorkerEngine(worker=worker, refresh_time=0.1)

    task = await engine.add_task(
        task_function="search",
        task_max_steps=10,
        query="What are the best practices for async programming in Python?",
    )

    print(f"Task: {task.id}")
    print(f"Confidence threshold: {worker.confidence_threshold:.0%}")
    print()

    await engine.execute()

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Status: {task.status}")
    print(f"Steps: {task.current_step}")
    print()

    # Calculate total cost
    total_cost = 0
    for _, result in task.history:
        if isinstance(result, dict):
            total_cost += result.get("cost", 0)
        elif hasattr(result, "cost"):
            total_cost += getattr(result, "cost", 0)
    print(f"Total cost: ${total_cost:.4f}")
    print()

    # Show workflow
    print("Workflow path:")
    for i, (func, _) in enumerate(task.history):
        print(f"  [{i + 1}] {func}")

    # Show report excerpt
    if task.result and isinstance(task.result, dict) and "report" in task.result:
        print()
        print("Report excerpt:")
        print("-" * 40)
        print(task.result["report"][:800])
        print("-" * 40)


if __name__ == "__main__":
    anyio.run(main)
