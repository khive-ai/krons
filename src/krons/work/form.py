# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Form - Data binding and scheduling for work units.

A Form represents an instantiated work unit with:
- Data binding (input values)
- Execution state tracking (filled, workable)
- Optional Phrase reference for typed I/O

Forms are the stateful layer between Phrase (definition) and Operation (execution).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import Field

from krons.core import Element

if TYPE_CHECKING:
    from .phrase import Phrase

__all__ = ("Form", "ParsedAssignment", "parse_assignment", "parse_full_assignment")


@dataclass
class ParsedAssignment:
    """Parsed form assignment with all components.

    Attributes:
        branch: Branch/worker name (e.g., "classifier1")
        inputs: Input field names
        outputs: Output field names
        resource: Resource hint (e.g., "api:fast")
        raw: Original assignment string
    """

    branch: str | None
    inputs: list[str]
    outputs: list[str]
    resource: str | None
    raw: str


def parse_assignment(assignment: str) -> tuple[list[str], list[str]]:
    """Parse 'inputs -> outputs' assignment DSL (simple form).

    Args:
        assignment: DSL string like "a, b -> c, d"

    Returns:
        Tuple of (input_fields, output_fields)

    Raises:
        ValueError: If assignment format is invalid
    """
    parsed = parse_full_assignment(assignment)
    return parsed.inputs, parsed.outputs


def parse_full_assignment(assignment: str) -> ParsedAssignment:
    """Parse full assignment DSL with branch and resource hints.

    Format: "branch: inputs -> outputs | resource"

    Examples:
        "a, b -> c"                           # Simple
        "classifier: job -> role | api:fast"  # Full
        "writer: context -> summary"          # Branch, no resource

    Args:
        assignment: DSL string

    Returns:
        ParsedAssignment with all components

    Raises:
        ValueError: If format is invalid
    """
    raw = assignment.strip()
    branch = None
    resource = None

    # Extract resource hint (after |)
    if "|" in raw:
        main_part, resource_part = raw.rsplit("|", 1)
        resource = resource_part.strip()
        raw = main_part.strip()

    # Extract branch name (before :)
    if ":" in raw:
        # Check it's not just inside the field list
        colon_idx = raw.find(":")
        arrow_idx = raw.find("->")
        if arrow_idx == -1 or colon_idx < arrow_idx:
            branch_part, raw = raw.split(":", 1)
            branch = branch_part.strip()
            raw = raw.strip()

    # Parse inputs -> outputs
    if "->" not in raw:
        raise ValueError(f"Invalid assignment syntax (missing '->'): {assignment}")

    parts = raw.split("->")
    if len(parts) != 2:
        raise ValueError(f"Invalid assignment syntax: {assignment}")

    inputs = [f.strip() for f in parts[0].split(",") if f.strip()]
    outputs = [f.strip() for f in parts[1].split(",") if f.strip()]

    return ParsedAssignment(
        branch=branch,
        inputs=inputs,
        outputs=outputs,
        resource=resource,
        raw=assignment,
    )


class Form(Element):
    """Data binding container for work units.

    A Form binds input data and tracks execution state. It can be created:
    1. From a Phrase (typed I/O)
    2. From an assignment string (dynamic fields)

    Assignment DSL supports full format:
        "branch: inputs -> outputs | resource"

    Examples:
        "a, b -> c"                           # Simple
        "classifier: job -> role | api:fast"  # Full with branch and resource
        "writer: context -> summary"          # Branch, no resource

    Attributes:
        assignment: DSL string 'branch: inputs -> outputs | resource'
        branch: Worker/branch name for routing
        resource: Resource hint for capability matching
        input_fields: Fields required as inputs
        output_fields: Fields produced as outputs
        available_data: Current data values
        output: Execution result
        filled: Whether form has been executed
        phrase: Optional Phrase reference for typed execution
    """

    assignment: str = Field(
        default="",
        description="Assignment DSL: 'branch: inputs -> outputs | resource'",
    )
    branch: str | None = Field(
        default=None,
        description="Worker/branch name for routing",
    )
    resource: str | None = Field(
        default=None,
        description="Resource hint (e.g., 'api:fast')",
    )
    input_fields: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)
    available_data: dict[str, Any] = Field(default_factory=dict)
    output: Any = Field(default=None)
    filled: bool = Field(default=False)

    # Optional phrase reference (set via from_phrase())
    _phrase: "Phrase | None" = None

    def model_post_init(self, _: Any) -> None:
        """Parse assignment to derive fields if not already set."""
        if self.assignment and not self.input_fields and not self.output_fields:
            parsed = parse_full_assignment(self.assignment)
            self.input_fields = parsed.inputs
            self.output_fields = parsed.outputs
            if parsed.branch and self.branch is None:
                self.branch = parsed.branch
            if parsed.resource and self.resource is None:
                self.resource = parsed.resource

    @classmethod
    def from_phrase(
        cls,
        phrase: "Phrase",
        **initial_data: Any,
    ) -> "Form":
        """Create Form from a Phrase with optional initial data.

        Args:
            phrase: Phrase defining typed I/O
            **initial_data: Initial input values

        Returns:
            Form bound to the phrase
        """
        form = cls(
            assignment=f"{', '.join(phrase.inputs)} -> {', '.join(phrase.outputs)}",
            input_fields=list(phrase.inputs),
            output_fields=list(phrase.outputs),
            available_data=dict(initial_data),
        )
        form._phrase = phrase
        return form

    @property
    def phrase(self) -> "Phrase | None":
        """Get bound phrase if any."""
        return self._phrase

    def is_workable(self) -> bool:
        """Check if form is ready for execution.

        Returns:
            True if all inputs available and not already filled
        """
        if self.filled:
            return False

        for field in self.input_fields:
            if field not in self.available_data:
                return False
            if self.available_data[field] is None:
                return False

        return True

    def get_inputs(self) -> dict[str, Any]:
        """Extract input data for execution.

        Returns:
            Dict of input field values
        """
        return {
            f: self.available_data[f]
            for f in self.input_fields
            if f in self.available_data
        }

    def fill(self, **data: Any) -> None:
        """Add data to available_data.

        Args:
            **data: Field values to add
        """
        self.available_data.update(data)

    def set_output(self, output: Any) -> None:
        """Mark form as filled with output.

        Args:
            output: Execution result
        """
        self.output = output
        self.filled = True

        # Extract output field values from result
        if output is not None:
            for field in self.output_fields:
                if hasattr(output, field):
                    self.available_data[field] = getattr(output, field)
                elif isinstance(output, dict) and field in output:
                    self.available_data[field] = output[field]

    def get_output_data(self) -> dict[str, Any]:
        """Extract output field values.

        Returns:
            Dict mapping output field names to values
        """
        result = {}
        for field in self.output_fields:
            if field in self.available_data:
                result[field] = self.available_data[field]
        return result

    async def execute(self, ctx: Any = None) -> Any:
        """Execute the form if it has a bound phrase.

        Args:
            ctx: Execution context

        Returns:
            Execution result

        Raises:
            RuntimeError: If no phrase bound or form not workable
        """
        if self._phrase is None:
            raise RuntimeError("Form has no bound phrase - cannot execute")

        if not self.is_workable():
            missing = [f for f in self.input_fields if f not in self.available_data]
            raise RuntimeError(f"Form not workable - missing inputs: {missing}")

        result = await self._phrase(self.get_inputs(), ctx)
        self.set_output(result)
        return result

    def __repr__(self) -> str:
        status = (
            "filled" if self.filled else ("ready" if self.is_workable() else "pending")
        )
        phrase_info = f", phrase={self._phrase.name}" if self._phrase else ""
        return f"Form('{self.assignment}', {status}{phrase_info})"
