# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for krons.work.report - multi-step workflow orchestration."""

from __future__ import annotations

from uuid import UUID

import pytest

from krons.work.form import Form
from krons.work.report import Report

# =============================================================================
# Tests: Report Creation
# =============================================================================


class TestReportCreation:
    """Tests for Report instantiation."""

    def test_report_with_assignment(self):
        """Report should parse overall workflow assignment."""
        report = Report(
            assignment="input -> output",
            form_assignments=["input -> step1", "step1 -> output"],
        )

        assert report.input_fields == ["input"]
        assert report.output_fields == ["output"]

    def test_report_creates_forms(self):
        """Report should create forms from form_assignments."""
        report = Report(
            assignment="a -> c",
            form_assignments=["a -> b", "b -> c"],
        )

        assert len(report.forms) == 2

    def test_report_forms_have_correct_fields(self):
        """Report forms should have correct input/output fields."""
        report = Report(
            assignment="context -> final_score",
            form_assignments=[
                "context -> analysis",
                "analysis -> score",
                "score -> final_score",
            ],
        )

        forms = list(report.forms)

        assert forms[0].input_fields == ["context"]
        assert forms[0].output_fields == ["analysis"]

        assert forms[1].input_fields == ["analysis"]
        assert forms[1].output_fields == ["score"]

        assert forms[2].input_fields == ["score"]
        assert forms[2].output_fields == ["final_score"]

    def test_report_empty_assignment(self):
        """Report should handle empty assignment gracefully."""
        report = Report()

        assert report.input_fields == []
        assert report.output_fields == []
        assert len(report.forms) == 0

    def test_report_default_available_data(self):
        """Report should default to empty available_data."""
        report = Report(assignment="a -> b")

        assert report.available_data == {}

    def test_report_default_completed_forms(self):
        """Report should default to empty completed_forms."""
        report = Report(assignment="a -> b")

        assert len(report.completed_forms) == 0


# =============================================================================
# Tests: Report.initialize
# =============================================================================


class TestReportInitialize:
    """Tests for Report.initialize input provisioning."""

    def test_initialize_sets_inputs(self):
        """initialize should set input values in available_data."""
        report = Report(
            assignment="a, b -> output",
            form_assignments=["a, b -> output"],
        )

        report.initialize(a=1, b=2)

        assert report.available_data["a"] == 1
        assert report.available_data["b"] == 2

    def test_initialize_missing_input_raises(self):
        """initialize should raise for missing required inputs."""
        report = Report(
            assignment="a, b -> output",
            form_assignments=["a, b -> output"],
        )

        with pytest.raises(ValueError, match="Missing required input: 'b'"):
            report.initialize(a=1)

    def test_initialize_extra_data_ignored(self):
        """initialize should accept but ignore extra data."""
        report = Report(
            assignment="a -> output",
            form_assignments=["a -> output"],
        )

        report.initialize(a=1, extra="ignored")

        assert report.available_data["a"] == 1
        assert "extra" not in report.available_data


# =============================================================================
# Tests: Report.next_forms
# =============================================================================


class TestReportNextForms:
    """Tests for Report.next_forms scheduling."""

    def test_next_forms_returns_ready_forms(self):
        """next_forms should return forms with all inputs available."""
        report = Report(
            assignment="a -> c",
            form_assignments=["a -> b", "b -> c"],
        )
        report.initialize(a=1)

        ready = report.next_forms()

        assert len(ready) == 1
        assert ready[0].input_fields == ["a"]

    def test_next_forms_excludes_filled(self):
        """next_forms should exclude already filled forms."""
        report = Report(
            assignment="a -> c",
            form_assignments=["a -> b", "b -> c"],
        )
        report.initialize(a=1)

        # Fill the first form manually
        forms = list(report.forms)
        forms[0].filled = True

        ready = report.next_forms()

        # First form is filled, second needs 'b' which isn't available
        assert len(ready) == 0

    def test_next_forms_multiple_ready(self):
        """next_forms should return all ready forms."""
        report = Report(
            assignment="a, b -> d",
            form_assignments=["a -> c", "b -> c"],
        )
        report.initialize(a=1, b=2)

        ready = report.next_forms()

        # Both forms should be ready (they have independent inputs)
        assert len(ready) == 2

    def test_next_forms_updates_form_available_data(self):
        """next_forms should copy available_data to forms."""
        report = Report(
            assignment="a -> b",
            form_assignments=["a -> b"],
        )
        report.initialize(a=1)

        ready = report.next_forms()

        assert ready[0].available_data["a"] == 1

    def test_next_forms_empty_when_none_ready(self):
        """next_forms should return empty list when no forms ready."""
        report = Report(
            assignment="a -> b",
            form_assignments=["a -> b"],
        )
        # Don't initialize - no data available

        ready = report.next_forms()

        assert ready == []


# =============================================================================
# Tests: Report.complete_form
# =============================================================================


class TestReportCompleteForm:
    """Tests for Report.complete_form completion handling."""

    def test_complete_form_adds_to_completed(self):
        """complete_form should add form to completed_forms."""
        report = Report(
            assignment="a -> b",
            form_assignments=["a -> b"],
        )
        report.initialize(a=1)

        forms = list(report.forms)
        forms[0].fill(a=1)
        forms[0].set_output({"b": 2})

        report.complete_form(forms[0])

        assert forms[0] in report.completed_forms

    def test_complete_form_updates_available_data(self):
        """complete_form should update available_data with outputs."""
        report = Report(
            assignment="a -> c",
            form_assignments=["a -> b", "b -> c"],
        )
        report.initialize(a=1)

        forms = list(report.forms)
        forms[0].fill(a=1)
        forms[0].set_output({"b": 42})

        report.complete_form(forms[0])

        assert report.available_data["b"] == 42

    def test_complete_form_not_filled_raises(self):
        """complete_form should raise if form not filled."""
        report = Report(
            assignment="a -> b",
            form_assignments=["a -> b"],
        )
        report.initialize(a=1)

        forms = list(report.forms)
        # Don't fill the form

        with pytest.raises(ValueError, match="not filled"):
            report.complete_form(forms[0])

    def test_complete_form_enables_next_forms(self):
        """complete_form should enable dependent forms."""
        report = Report(
            assignment="a -> c",
            form_assignments=["a -> b", "b -> c"],
        )
        report.initialize(a=1)

        # Complete first form
        forms = list(report.forms)
        forms[0].fill(a=1)
        forms[0].set_output({"b": 2})
        report.complete_form(forms[0])

        # Now second form should be ready
        ready = report.next_forms()

        assert len(ready) == 1
        assert ready[0].input_fields == ["b"]


# =============================================================================
# Tests: Report.is_complete
# =============================================================================


class TestReportIsComplete:
    """Tests for Report.is_complete workflow status."""

    def test_is_complete_when_outputs_available(self):
        """is_complete should return True when all outputs available."""
        report = Report(
            assignment="a -> b",
            form_assignments=["a -> b"],
        )
        report.available_data["b"] = 42

        assert report.is_complete() is True

    def test_is_complete_false_when_outputs_missing(self):
        """is_complete should return False when outputs missing."""
        report = Report(
            assignment="a -> b, c",
            form_assignments=["a -> b", "a -> c"],
        )
        report.available_data["b"] = 42
        # c is missing

        assert report.is_complete() is False

    def test_is_complete_initially_false(self):
        """is_complete should be False initially."""
        report = Report(
            assignment="a -> b",
            form_assignments=["a -> b"],
        )

        assert report.is_complete() is False


# =============================================================================
# Tests: Report.get_deliverable
# =============================================================================


class TestReportGetDeliverable:
    """Tests for Report.get_deliverable output extraction."""

    def test_get_deliverable_returns_outputs(self):
        """get_deliverable should return output field values."""
        report = Report(
            assignment="a -> b, c",
            form_assignments=["a -> b", "a -> c"],
        )
        report.available_data["b"] = 1
        report.available_data["c"] = 2
        report.available_data["extra"] = 3

        deliverable = report.get_deliverable()

        assert deliverable == {"b": 1, "c": 2}
        assert "extra" not in deliverable

    def test_get_deliverable_missing_outputs(self):
        """get_deliverable should return None for missing outputs."""
        report = Report(
            assignment="a -> b, c",
            form_assignments=["a -> b", "a -> c"],
        )
        report.available_data["b"] = 1
        # c is missing

        deliverable = report.get_deliverable()

        assert deliverable["b"] == 1
        assert deliverable["c"] is None


# =============================================================================
# Tests: Report.progress
# =============================================================================


class TestReportProgress:
    """Tests for Report.progress tracking."""

    def test_progress_initial(self):
        """progress should be (0, total) initially."""
        report = Report(
            assignment="a -> c",
            form_assignments=["a -> b", "b -> c"],
        )

        completed, total = report.progress

        assert completed == 0
        assert total == 2

    def test_progress_after_completion(self):
        """progress should reflect completed forms."""
        report = Report(
            assignment="a -> c",
            form_assignments=["a -> b", "b -> c"],
        )
        report.initialize(a=1)

        forms = list(report.forms)
        forms[0].fill(a=1)
        forms[0].set_output({"b": 2})
        report.complete_form(forms[0])

        completed, total = report.progress

        assert completed == 1
        assert total == 2

    def test_progress_all_complete(self):
        """progress should be (total, total) when all complete."""
        report = Report(
            assignment="a -> b",
            form_assignments=["a -> b"],
        )
        report.initialize(a=1)

        forms = list(report.forms)
        forms[0].fill(a=1)
        forms[0].set_output({"b": 2})
        report.complete_form(forms[0])

        completed, total = report.progress

        assert completed == 1
        assert total == 1


# =============================================================================
# Tests: Report.__repr__
# =============================================================================


class TestReportRepr:
    """Tests for Report string representation."""

    def test_repr_includes_assignment(self):
        """repr should include assignment."""
        report = Report(
            assignment="input -> output",
            form_assignments=["input -> output"],
        )

        repr_str = repr(report)

        assert "Report" in repr_str
        assert "input -> output" in repr_str

    def test_repr_includes_progress(self):
        """repr should include progress."""
        report = Report(
            assignment="a -> c",
            form_assignments=["a -> b", "b -> c"],
        )

        repr_str = repr(report)

        assert "0/2" in repr_str


# =============================================================================
# Tests: Report Element inheritance
# =============================================================================


class TestReportElement:
    """Tests for Report Element inheritance."""

    def test_report_has_id(self):
        """Report should have UUID id from Element."""
        report = Report(assignment="a -> b")
        assert isinstance(report.id, UUID)

    def test_report_has_created_at(self):
        """Report should have created_at from Element."""
        report = Report(assignment="a -> b")
        assert report.created_at is not None


# =============================================================================
# Tests: Report Workflow Integration
# =============================================================================


class TestReportWorkflow:
    """Integration tests for Report workflow execution."""

    def test_sequential_workflow(self):
        """Test executing a sequential workflow: a -> b -> c -> d."""
        report = Report(
            assignment="context -> final_score",
            form_assignments=[
                "context -> analysis",
                "analysis -> score",
                "score -> final_score",
            ],
        )
        report.initialize(context="some text")

        # Step 1: Execute first form
        ready = report.next_forms()
        assert len(ready) == 1
        assert ready[0].input_fields == ["context"]

        ready[0].fill(context="some text")
        ready[0].set_output({"analysis": "analyzed text"})
        report.complete_form(ready[0])

        # Step 2: Execute second form
        ready = report.next_forms()
        assert len(ready) == 1
        assert ready[0].input_fields == ["analysis"]

        ready[0].fill(analysis="analyzed text")
        ready[0].set_output({"score": 85})
        report.complete_form(ready[0])

        # Step 3: Execute third form
        ready = report.next_forms()
        assert len(ready) == 1
        assert ready[0].input_fields == ["score"]

        ready[0].fill(score=85)
        ready[0].set_output({"final_score": 90})
        report.complete_form(ready[0])

        # Workflow complete
        assert report.is_complete() is True
        assert report.get_deliverable() == {"final_score": 90}

    def test_parallel_workflow(self):
        """Test executing a parallel workflow: a -> (b, c) -> d."""
        report = Report(
            assignment="input -> combined",
            form_assignments=[
                "input -> part1",
                "input -> part2",
                "part1, part2 -> combined",
            ],
        )
        report.initialize(input="data")

        # Step 1: Both first forms should be ready
        ready = report.next_forms()
        assert len(ready) == 2

        # Complete both in parallel
        for form in ready:
            form.fill(input="data")
            if form.output_fields == ["part1"]:
                form.set_output({"part1": "result1"})
            else:
                form.set_output({"part2": "result2"})
            report.complete_form(form)

        # Step 2: Combined form should now be ready
        ready = report.next_forms()
        assert len(ready) == 1
        assert set(ready[0].input_fields) == {"part1", "part2"}

        ready[0].fill(part1="result1", part2="result2")
        ready[0].set_output({"combined": "final"})
        report.complete_form(ready[0])

        assert report.is_complete() is True
        assert report.get_deliverable() == {"combined": "final"}
