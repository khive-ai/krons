# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Report - Multi-step workflow orchestration.

A Report orchestrates multiple Forms based on data availability:
- Schedules forms when their inputs become available
- Groups forms by branch for sequential execution within branch
- Propagates outputs between forms
- Tracks overall workflow completion

This is the scheduling layer that enables data-driven DAG execution.

The Report pattern supports declarative workflow definition:
    - Fields as class attributes (typed outputs)
    - form_assignments DSL with branch/resource hints
    - Implicit dependencies from data flow
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import Field

from krons.core import Element, Pile

from .form import Form, parse_assignment, parse_full_assignment

__all__ = ("Report",)


class Report(Element):
    """Workflow orchestrator - schedules forms based on field availability.

    A Report manages a collection of Forms, executing them as their
    inputs become available. Forms are grouped by branch - forms on the
    same branch execute sequentially, different branches can run in parallel.

    Example (simple):
        report = Report(
            assignment="context -> final_score",
            form_assignments=[
                "context -> analysis",
                "analysis -> score",
                "score -> final_score",
            ],
        )
        report.initialize(context="some input")

        while not report.is_complete():
            for form in report.next_forms():
                await form.execute(ctx)
                report.complete_form(form)

    Example (with branches and resources):
        class HiringBriefReport(Report):
            role_classification: RoleClassification | None = None
            strategic_context: StrategicContext | None = None

            assignment: str = "job_input -> executive_summary"

            form_assignments: list[str] = [
                "classifier: job_input -> role_classification | api:fast",
                "strategist: job_input, role_classification -> strategic_context | api:synthesis",
                "writer: strategic_context -> executive_summary | api:reasoning",
            ]

    Attributes:
        assignment: Overall workflow 'inputs -> final_outputs'
        form_assignments: List of form assignments with optional branch/resource
        input_fields: Workflow input fields
        output_fields: Workflow output fields
        forms: All forms in workflow
        completed_forms: Forms that have finished
        available_data: Current state of all field values
    """

    assignment: str = Field(
        default="",
        description="Overall workflow: 'inputs -> final_outputs'",
    )
    form_assignments: list[str] = Field(
        default_factory=list,
        description="List of form assignments: ['branch: a -> b | resource', ...]",
    )

    input_fields: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)

    forms: Pile[Form] = Field(
        default_factory=lambda: Pile(item_type=Form),
        description="All forms in the workflow",
    )
    completed_forms: Pile[Form] = Field(
        default_factory=lambda: Pile(item_type=Form),
        description="Completed forms",
    )
    available_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Current state of all field values",
    )

    # Branch tracking: branch_name -> list of form IDs in order
    _branch_forms: dict[str, list[Form]] = {}
    # Track last completed form per branch for sequential execution
    _branch_progress: dict[str, int] = {}

    def model_post_init(self, _: Any) -> None:
        """Parse assignment and create forms."""
        self._branch_forms = defaultdict(list)
        self._branch_progress = defaultdict(int)

        if not self.assignment:
            return

        # Parse overall assignment
        self.input_fields, self.output_fields = parse_assignment(self.assignment)

        # Create forms from form_assignments
        for fa in self.form_assignments:
            form = Form(assignment=fa)
            self.forms.include(form)

            # Track by branch
            branch = form.branch or "_default"
            self._branch_forms[branch].append(form)

    def initialize(self, **inputs: Any) -> None:
        """Provide initial input data.

        Args:
            **inputs: Initial field values

        Raises:
            ValueError: If required input is missing
        """
        for field in self.input_fields:
            if field not in inputs:
                raise ValueError(f"Missing required input: '{field}'")
            self.available_data[field] = inputs[field]

    def next_forms(self) -> list[Form]:
        """Get forms that are ready to execute.

        Forms with explicit branches execute sequentially within their branch.
        Forms without branches (None) execute in parallel based on data availability.

        Returns:
            List of forms with all inputs available and not yet filled
        """
        ready = []

        for branch, forms in self._branch_forms.items():
            if branch == "_default":
                # No explicit branch - parallel execution based on data
                for form in forms:
                    if form.filled:
                        continue
                    form.available_data = self.available_data.copy()
                    if form.is_workable():
                        ready.append(form)
            else:
                # Explicit branch - sequential execution
                progress = self._branch_progress[branch]

                # Only consider the next form in this branch
                if progress < len(forms):
                    form = forms[progress]
                    if form.filled:
                        # Already done, advance progress
                        self._branch_progress[branch] += 1
                        continue

                    form.available_data = self.available_data.copy()
                    if form.is_workable():
                        ready.append(form)

        return ready

    def complete_form(self, form: Form) -> None:
        """Mark a form as completed and update available data.

        Args:
            form: The completed form

        Raises:
            ValueError: If form is not filled
        """
        if not form.filled:
            raise ValueError("Form is not filled")

        self.completed_forms.include(form)

        # Advance branch progress
        branch = form.branch or "_default"
        if branch in self._branch_forms:
            forms = self._branch_forms[branch]
            progress = self._branch_progress[branch]
            if progress < len(forms) and forms[progress].id == form.id:
                self._branch_progress[branch] += 1

        # Update available data with form outputs
        output_data = form.get_output_data()
        self.available_data.update(output_data)

    def is_complete(self) -> bool:
        """Check if all output fields are available.

        Returns:
            True if workflow is complete
        """
        return all(field in self.available_data for field in self.output_fields)

    def get_deliverable(self) -> dict[str, Any]:
        """Get final output values.

        Returns:
            Dict of output field values
        """
        return {f: self.available_data.get(f) for f in self.output_fields}

    @property
    def progress(self) -> tuple[int, int]:
        """Get progress as (completed, total).

        Returns:
            Tuple of (completed forms, total forms)
        """
        return len(self.completed_forms), len(self.forms)

    def get_forms_by_branch(self, branch: str) -> list[Form]:
        """Get all forms for a specific branch.

        Args:
            branch: Branch name

        Returns:
            List of forms on that branch (in order)
        """
        return list(self._branch_forms.get(branch, []))

    def get_forms_by_resource(self, resource: str) -> list[Form]:
        """Get all forms requiring a specific resource.

        Args:
            resource: Resource hint (e.g., 'api:fast')

        Returns:
            List of forms with that resource hint
        """
        return [f for f in self.forms if f.resource == resource]

    @property
    def branches(self) -> list[str]:
        """Get all branch names in this report."""
        return list(self._branch_forms.keys())

    @property
    def resources(self) -> set[str]:
        """Get all resource hints used in this report."""
        return {f.resource for f in self.forms if f.resource}

    def __repr__(self) -> str:
        completed, total = self.progress
        branches = len(self._branch_forms)
        return f"Report('{self.assignment}', {completed}/{total} forms, {branches} branches)"
