# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Pipeline Router: async message passing between chained processing stages.

Demonstrates Exchange pattern with:
- 3 pipeline stages: parser -> transformer -> writer
- Each stage is an Element with registered mailbox
- Direct messaging (sender/recipient specified)
- Concurrent stage loops processing incoming messages
"""

from __future__ import annotations

import asyncio

from krons.core import Element
from krons.session import Exchange


class PipelineStage(Element):
    """A processing stage in the pipeline."""

    name: str = "stage"

    def process(self, data: dict) -> dict:
        """Process incoming data. Override in subclasses."""
        return data


class Parser(PipelineStage):
    """Stage 1: Parse raw input into structured data."""

    name: str = "parser"

    def process(self, data: dict) -> dict:
        """Parse raw text into tokens."""
        raw = data.get("raw", "")
        tokens = raw.strip().split()
        return {"tokens": tokens, "count": len(tokens)}


class Transformer(PipelineStage):
    """Stage 2: Transform parsed data."""

    name: str = "transformer"

    def process(self, data: dict) -> dict:
        """Transform tokens to uppercase."""
        tokens = data.get("tokens", [])
        transformed = [t.upper() for t in tokens]
        return {"tokens": transformed, "count": data.get("count", 0)}


class Writer(PipelineStage):
    """Stage 3: Write final output."""

    name: str = "writer"

    def process(self, data: dict) -> dict:
        """Join tokens into final output string."""
        tokens = data.get("tokens", [])
        output = " ".join(tokens)
        return {"output": output, "count": data.get("count", 0)}


async def stage_loop(
    exchange: Exchange,
    stage: PipelineStage,
    prev_stage: PipelineStage | None,
    next_stage: PipelineStage | None,
    results: list,
    stop_event: asyncio.Event,
) -> None:
    """Run a pipeline stage's message processing loop.

    Args:
        exchange: The Exchange routing messages.
        stage: This stage's Element.
        prev_stage: Previous stage to receive from (None for first stage).
        next_stage: Next stage to send to (None for last stage).
        results: Shared list to collect final outputs.
        stop_event: Signal to stop processing.
    """
    print(f"[{stage.name}] Started")

    while not stop_event.is_set():
        if prev_stage is not None:
            # Pop message from previous stage's inbox
            msg = exchange.pop_message(stage.id, sender=prev_stage.id)
            if msg is not None:
                print(f"[{stage.name}] Received: {msg.content}")

                # Process the data
                result = stage.process(msg.content)
                print(f"[{stage.name}] Processed: {result}")

                if next_stage is not None:
                    # Forward to next stage
                    exchange.send(stage.id, recipient=next_stage.id, content=result)
                    print(f"[{stage.name}] Sent to {next_stage.name}")
                else:
                    # Final stage - collect result
                    results.append(result)
                    print(f"[{stage.name}] Final output collected")

        await asyncio.sleep(0.05)

    print(f"[{stage.name}] Stopped")


async def run_pipeline(inputs: list[str]) -> list[dict]:
    """Run the pipeline with given inputs.

    Args:
        inputs: List of raw text strings to process.

    Returns:
        List of processed output dicts.
    """
    # Create exchange and stages
    exchange = Exchange()
    parser = Parser()
    transformer = Transformer()
    writer = Writer()
    stages = [parser, transformer, writer]

    # Register all stages with the exchange
    for stage in stages:
        exchange.register(stage.id)

    # Create a source element to inject initial messages
    source = Element()
    exchange.register(source.id)

    # Shared results and stop signal
    results: list[dict] = []
    stop_event = asyncio.Event()

    # Start stage loops
    stage_tasks = [
        asyncio.create_task(
            stage_loop(
                exchange,
                stage=stages[i],
                prev_stage=stages[i - 1] if i > 0 else source,
                next_stage=stages[i + 1] if i < len(stages) - 1 else None,
                results=results,
                stop_event=stop_event,
            )
        )
        for i in range(len(stages))
    ]

    # Start exchange sync loop
    exchange_task = asyncio.create_task(exchange.run(interval=0.02))

    # Inject inputs from source to parser
    for raw_text in inputs:
        exchange.send(source.id, recipient=parser.id, content={"raw": raw_text})
        print(f"[source] Injected: {raw_text!r}")

    # Wait for all results
    while len(results) < len(inputs):
        await asyncio.sleep(0.05)

    # Stop everything
    stop_event.set()
    exchange.stop()

    # Wait for tasks to complete
    await asyncio.gather(*stage_tasks, return_exceptions=True)
    exchange_task.cancel()
    try:
        await exchange_task
    except asyncio.CancelledError:
        pass

    return results


async def main() -> None:
    """Run example pipeline."""
    print("=" * 60)
    print("Pipeline Router Example")
    print("=" * 60)
    print()

    inputs = [
        "hello world from krons",
        "async message passing",
        "pipeline stages rock",
    ]

    print("Inputs:")
    for i, text in enumerate(inputs, 1):
        print(f"  {i}. {text!r}")
    print()

    print("Processing...")
    print("-" * 40)

    results = await run_pipeline(inputs)

    print("-" * 40)
    print()
    print("Results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result}")
    print()
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
