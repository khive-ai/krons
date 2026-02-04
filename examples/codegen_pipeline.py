# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Code generation pipeline with error recovery using Claude Code.

Demonstrates a code generation workflow with conditional debugging:
1. write_code: Generate code via Claude Code
2. execute_code: Run the code in a sandbox
3. debug_code: If execution fails, debug via Claude Code
4. Loop back to execute until success or max debug attempts

Usage:
    uv run python examples/codegen_pipeline.py
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from dataclasses import dataclass
from io import StringIO

import anyio

from krons.agent.third_party.claude_code import (
    ClaudeCodeRequest,
    ClaudeSession,
    stream_claude_code_cli,
)
from krons.work import Worker, WorkerEngine, work, worklink

WORKSPACE = ".khive/examples/codegen_pipeline"
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


def extract_code(text: str) -> str:
    """Extract Python code from markdown code blocks or raw text."""
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()

    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()

    return text.strip()


@dataclass
class ExecutionResult:
    """Result of code execution."""

    code: str
    success: bool
    output: str | None = None
    error: str | None = None
    attempt: int = 1


class CodeGenWorker(Worker):
    """Worker that generates and executes code with error recovery.

    Workflow:
        write_code -> execute_code -> (if error) -> debug_code -> execute_code
                                   -> (if success) -> complete
    """

    name = "codegen"
    max_debug_attempts: int = 3

    def __init__(self, max_debug_attempts: int = 3) -> None:
        super().__init__()
        self.max_debug_attempts = max_debug_attempts

    @work(assignment="instruction, context -> code", timeout=120.0)
    async def write_code(
        self, instruction: str, context: dict | None = None, **kwargs
    ) -> dict:
        """Generate code via Claude Code.

        Args:
            instruction: What code to generate
            context: Optional context dict

        Returns:
            Dict with generated code
        """
        print(f"[write_code] Generating: {instruction[:50]}...")

        prompt = f"""Generate Python code for the following task:

{instruction}

Requirements:
- Write clean, working Python code
- Include only the code, no explanations
- The code should be executable as-is
- Use print() to show the output/result

Respond with ONLY the Python code."""

        if context:
            prompt += f"\n\nContext: {context}"

        session = await call_claude(
            prompt=prompt,
            system_prompt="You are an expert Python programmer. Write clean, working code.",
        )

        code = extract_code(session.result or "print('Error generating code')")
        print(
            f"[write_code] Generated {len(code)} chars (cost: ${session.total_cost_usd or 0:.4f})"
        )

        return {
            "code": code,
            "instruction": instruction,
            "cost": session.total_cost_usd or 0,
        }

    @work(assignment="code -> execution_result", timeout=30.0)
    async def execute_code(
        self, code: str, attempt: int = 1, **kwargs
    ) -> ExecutionResult:
        """Execute code in a sandboxed environment.

        Args:
            code: Python code to execute
            attempt: Current execution attempt

        Returns:
            ExecutionResult with output or error
        """
        print(f"[execute_code] Attempt {attempt}, running code...")

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()

        try:
            # Execute with timeout
            exec_globals: dict = {}

            def run_code():
                exec(code, exec_globals)

            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, run_code), timeout=10.0
            )

            output = captured.getvalue()
            print(f"[execute_code] Success: {output[:60]}...")

            return ExecutionResult(
                code=code,
                success=True,
                output=output,
                attempt=attempt,
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            tb = traceback.format_exc()
            print(f"[execute_code] Error: {error_msg}")

            return ExecutionResult(
                code=code,
                success=False,
                error=f"{error_msg}\n\n{tb}",
                attempt=attempt,
            )

        finally:
            sys.stdout = old_stdout

    @work(assignment="code, error -> fixed_code", timeout=120.0)
    async def debug_code(
        self, code: str, error: str, attempt: int = 1, **kwargs
    ) -> dict:
        """Debug and fix code via Claude Code.

        Args:
            code: The failing code
            error: Error message/traceback

        Returns:
            Dict with fixed code
        """
        print(f"[debug_code] Attempt {attempt}, fixing error...")

        prompt = f"""The following Python code has an error. Fix it.

Code:
```python
{code}
```

Error:
```
{error}
```

Requirements:
- Fix the error while preserving the original intent
- Return ONLY the fixed Python code
- The code should be executable as-is"""

        session = await call_claude(
            prompt=prompt,
            system_prompt="You are an expert Python debugger. Fix code errors precisely.",
        )

        fixed = extract_code(session.result or code)
        print(
            f"[debug_code] Fixed code: {len(fixed)} chars (cost: ${session.total_cost_usd or 0:.4f})"
        )

        return {
            "code": fixed,
            "attempt": attempt,
            "cost": session.total_cost_usd or 0,
        }

    @worklink(from_="write_code", to_="execute_code")
    async def write_to_execute(self, result: dict) -> dict:
        """Route write_code -> execute_code."""
        return {"code": result["code"], "attempt": 1}

    @worklink(from_="execute_code", to_="debug_code")
    async def execute_to_debug(self, result: ExecutionResult) -> dict | None:
        """Route failed execution to debug_code.

        Returns None if execution succeeded.
        """
        if result.success:
            print("[execute->debug] Success - no debug needed")
            return None

        if result.attempt >= self.max_debug_attempts:
            print(
                f"[execute->debug] Max debug attempts ({self.max_debug_attempts}) reached"
            )
            return None

        print(f"[execute->debug] Routing to debug (attempt {result.attempt})")
        return {
            "code": result.code,
            "error": result.error or "Unknown error",
            "attempt": result.attempt,
        }

    @worklink(from_="debug_code", to_="execute_code")
    async def debug_to_execute(self, result: dict) -> dict:
        """Route fixed code back to execution."""
        return {
            "code": result["code"],
            "attempt": result["attempt"] + 1,
        }


async def main():
    """Run the code generation pipeline with Claude Code."""
    print("=" * 60)
    print("Code Generation Pipeline - Claude Code Integration")
    print("=" * 60)
    print()

    worker = CodeGenWorker(max_debug_attempts=3)
    engine = WorkerEngine(worker=worker, refresh_time=0.1)

    # Test with a simple task
    task = await engine.add_task(
        task_function="write_code",
        task_max_steps=10,
        instruction="Write a function that calculates the factorial of a number and print factorial(5)",
    )

    print(f"Task: {task.id}")
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
    print("Workflow:")
    for i, (func, result) in enumerate(task.history):
        if func == "write_code":
            print(f"  [{i + 1}] write_code")
        elif func == "execute_code":
            status = "SUCCESS" if result.success else "FAILED"
            print(f"  [{i + 1}] execute_code: {status}")
        elif func == "debug_code":
            print(f"  [{i + 1}] debug_code")

    # Show final output
    if task.result:
        if isinstance(task.result, ExecutionResult) and task.result.success:
            print()
            print("Output:")
            print("-" * 40)
            print(task.result.output)
            print("-" * 40)


if __name__ == "__main__":
    anyio.run(main)
