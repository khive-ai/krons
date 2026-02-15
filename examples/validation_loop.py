# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Validation loop pattern with Worker @work and @worklink decorators.

Demonstrates a self-correcting content generation workflow using Claude Code:
1. generate: Create draft content via Claude Code
2. validate: Check if content meets criteria via Claude Code
3. Loop back to generate on failure (with feedback), up to max retries

The @worklink decorators define conditional edges:
- gen_to_validate: Always routes generate -> validate
- validate_to_generate: Conditionally routes back on failure (returns None to stop)

Usage:
    uv run python examples/validation_loop.py
"""

from __future__ import annotations

import anyio

from krons.agent.third_party.claude_code import (
    ClaudeCodeRequest,
    ClaudeSession,
    stream_claude_code_cli,
)
from krons.work import Worker, WorkerEngine, work, worklink

# Configuration
WORKSPACE = ".khive/examples/validation_loop"
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


class ValidatedGenerator(Worker):
    """Worker that generates content with validation loop using Claude Code.

    Workflow:
        generate -> validate -> (if failed) -> generate (with feedback)
                             -> (if passed) -> complete
    """

    name = "validated_generator"
    max_retries: int = 3

    def __init__(self, max_retries: int = 3) -> None:
        super().__init__()
        self.max_retries = max_retries
        self._attempts: dict[str, int] = {}

    @work(assignment="prompt, context -> draft", capacity=2, timeout=120.0)
    async def generate(self, prompt: str, context: str | None = None, **kwargs) -> dict:
        """Generate content from prompt via Claude Code.

        Args:
            prompt: The generation prompt
            context: Optional feedback from previous validation failure

        Returns:
            Dict with prompt, draft content, and attempt number
        """
        attempt = self._attempts.get(prompt, 0)
        self._attempts[prompt] = attempt + 1

        print(f"[generate] Attempt {attempt + 1} for: {prompt[:50]}...")

        # Build the Claude prompt
        if context:
            claude_prompt = f"""Previous attempt received this feedback: {context}

Please improve the content based on this feedback.

Original request: {prompt}

Generate improved content that addresses the feedback."""
        else:
            claude_prompt = f"""Generate high-quality content for the following request:

{prompt}

Requirements:
- Be comprehensive and well-structured
- Include specific details and examples
- Ensure clarity and readability"""

        session = await call_claude(
            prompt=claude_prompt,
            system_prompt="You are an expert content generator. Produce high-quality, well-structured content.",
        )

        draft = session.result or "Generation failed"
        print(f"[generate] Generated {len(draft)} chars (cost: ${session.total_cost_usd or 0:.4f})")

        return {
            "prompt": prompt,
            "draft": draft,
            "attempt": attempt,
            "cost": session.total_cost_usd or 0,
        }

    @work(assignment="prompt, draft, attempt -> validation_result", timeout=60.0)
    async def validate(self, prompt: str, draft: str, attempt: int, **kwargs) -> dict:
        """Validate generated content via Claude Code.

        Args:
            prompt: Original prompt (for context)
            draft: Generated content to validate
            attempt: Current attempt number

        Returns:
            Dict with validation result, feedback, and metadata
        """
        print(f"[validate] Checking draft (attempt {attempt + 1})...")

        validation_prompt = f"""Evaluate the following content against the original request.

Original request: {prompt}

Content to evaluate:
---
{draft[:2000]}
---

Evaluate on:
1. Completeness - Does it fully address the request?
2. Quality - Is it well-written and structured?
3. Accuracy - Is the information correct?

Respond in this exact format:
PASSED: yes/no
FEEDBACK: <specific feedback for improvement or confirmation of quality>"""

        session = await call_claude(
            prompt=validation_prompt,
            system_prompt="You are a strict quality evaluator. Be honest about content quality.",
        )

        result = session.result or "PASSED: no\nFEEDBACK: Validation failed"

        # Parse result - handle various response formats
        result_upper = result.upper()
        passed = False

        # Check for explicit pass indicators
        pass_patterns = [
            "PASSED: YES",
            "PASSED:YES",
            "PASSED = YES",
            "PASS: YES",
            "PASS:YES",
            "**PASSED**: YES",
            "**PASSED:**YES",
        ]
        fail_patterns = [
            "PASSED: NO",
            "PASSED:NO",
            "PASSED = NO",
            "PASS: NO",
            "PASS:NO",
            "**PASSED**: NO",
            "**PASSED:**NO",
        ]

        # Check fail first (if explicitly failed, don't pass)
        explicit_fail = any(p in result_upper for p in fail_patterns)
        if not explicit_fail:
            passed = any(p in result_upper for p in pass_patterns)

        feedback_start = result_upper.find("FEEDBACK:")
        feedback = result[feedback_start + 9 :].strip() if feedback_start > 0 else result

        print(f"[validate] {'PASSED' if passed else 'FAILED'}: {feedback[:60]}...")

        return {
            "prompt": prompt,
            "draft": draft,
            "attempt": attempt,
            "passed": passed,
            "feedback": feedback,
            "cost": session.total_cost_usd or 0,
        }

    @worklink(from_="generate", to_="validate")
    async def gen_to_validate(self, from_result: dict) -> dict:
        """Route generate output to validate."""
        return {
            "prompt": from_result["prompt"],
            "draft": from_result["draft"],
            "attempt": from_result["attempt"],
        }

    @worklink(from_="validate", to_="generate")
    async def validate_to_generate(self, from_result: dict) -> dict | None:
        """Conditionally route validation failure back to generate.

        Returns:
            Dict with retry params if should retry, None if done
        """
        prompt = from_result["prompt"]
        passed = from_result["passed"]
        feedback = from_result["feedback"]
        attempt = from_result["attempt"]

        if passed:
            print(f"[validate->generate] PASSED: {feedback[:60]}...")
            return None

        if attempt + 1 >= self.max_retries:
            print(f"[validate->generate] MAX RETRIES ({self.max_retries}) reached")
            return None

        print("[validate->generate] RETRY with feedback")
        return {
            "prompt": prompt,
            "context": feedback,
        }


async def main():
    """Run the validation loop example with Claude Code."""
    print("=" * 60)
    print("Validation Loop Pattern - Claude Code Integration")
    print("=" * 60)
    print()

    worker = ValidatedGenerator(max_retries=3)
    engine = WorkerEngine(worker=worker, refresh_time=0.1)

    task = await engine.add_task(
        task_function="generate",
        task_max_steps=10,
        prompt="Write a concise explanation of how async/await works in Python",
    )

    print(f"Task created: {task.id}")
    print(f"Starting at: {task.function}")
    print()

    await engine.execute()

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Status: {task.status}")
    print(f"Steps taken: {task.current_step}")
    print()

    # Calculate total cost
    total_cost = sum(
        result.get("cost", 0) for _, result in task.history if isinstance(result, dict)
    )
    print(f"Total cost: ${total_cost:.4f}")
    print()

    # Show workflow
    print("Workflow:")
    for i, (func, result) in enumerate(task.history):
        if func == "generate":
            draft_len = len(result.get("draft", ""))
            print(f"  [{i + 1}] generate: {draft_len} chars")
        elif func == "validate":
            status = "PASS" if result.get("passed") else "FAIL"
            print(f"  [{i + 1}] validate: {status}")

    # Final result
    if task.result and task.result.get("passed"):
        print()
        print("Final content:")
        print("-" * 40)
        print(task.history[-2][1].get("draft", "")[:500])
        print("-" * 40)


if __name__ == "__main__":
    anyio.run(main)
