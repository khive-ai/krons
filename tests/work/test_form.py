# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for krons.work.form - data binding and scheduling for work units."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from krons.core.specs import Operable, Spec
from krons.work.form import Form, parse_assignment
from krons.work.phrase import Phrase

# =============================================================================
# Tests: parse_assignment
# =============================================================================


class TestParseAssignment:
    """Tests for parse_assignment DSL parser."""

    def test_simple_assignment(self):
        """Parse simple 'a -> b' assignment."""
        inputs, outputs = parse_assignment("a -> b")
        assert inputs == ["a"]
        assert outputs == ["b"]

    def test_multiple_inputs(self):
        """Parse 'a, b -> c' assignment."""
        inputs, outputs = parse_assignment("a, b -> c")
        assert inputs == ["a", "b"]
        assert outputs == ["c"]

    def test_multiple_outputs(self):
        """Parse 'a -> b, c' assignment."""
        inputs, outputs = parse_assignment("a -> b, c")
        assert inputs == ["a"]
        assert outputs == ["b", "c"]

    def test_multiple_inputs_and_outputs(self):
        """Parse 'a, b, c -> d, e, f' assignment."""
        inputs, outputs = parse_assignment("a, b, c -> d, e, f")
        assert inputs == ["a", "b", "c"]
        assert outputs == ["d", "e", "f"]

    def test_whitespace_trimmed(self):
        """Parse assignment with extra whitespace."""
        inputs, outputs = parse_assignment("  a  ,  b  ->  c  ,  d  ")
        assert inputs == ["a", "b"]
        assert outputs == ["c", "d"]

    def test_no_inputs(self):
        """Parse assignment with no inputs (empty left side)."""
        inputs, outputs = parse_assignment(" -> output")
        assert inputs == []
        assert outputs == ["output"]

    def test_no_outputs(self):
        """Parse assignment with no outputs (empty right side)."""
        inputs, outputs = parse_assignment("input -> ")
        assert inputs == ["input"]
        assert outputs == []

    def test_missing_arrow_raises(self):
        """Assignment without '->' should raise ValueError."""
        with pytest.raises(ValueError, match="missing '->'"):
            parse_assignment("a, b, c")

    def test_multiple_arrows_raises(self):
        """Assignment with multiple '->' should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid assignment syntax"):
            parse_assignment("a -> b -> c")

    def test_empty_string_raises(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="missing '->'"):
            parse_assignment("")

    def test_underscored_field_names(self):
        """Parse assignment with underscore field names."""
        inputs, outputs = parse_assignment(
            "user_id, consent_scope -> has_consent, token_id"
        )
        assert inputs == ["user_id", "consent_scope"]
        assert outputs == ["has_consent", "token_id"]


# =============================================================================
# Tests: Form
# =============================================================================


class TestFormCreation:
    """Tests for Form instantiation."""

    def test_form_with_assignment(self):
        """Form should parse assignment to derive fields."""
        form = Form(assignment="input1, input2 -> output1, output2")

        assert form.input_fields == ["input1", "input2"]
        assert form.output_fields == ["output1", "output2"]

    def test_form_with_explicit_fields(self):
        """Form should accept explicit input/output fields."""
        form = Form(
            input_fields=["a", "b"],
            output_fields=["c", "d"],
        )

        assert form.input_fields == ["a", "b"]
        assert form.output_fields == ["c", "d"]

    def test_form_assignment_not_parsed_if_fields_set(self):
        """Form should not parse assignment if fields already set."""
        form = Form(
            assignment="x -> y",
            input_fields=["a", "b"],
            output_fields=["c"],
        )

        # Explicit fields take precedence
        assert form.input_fields == ["a", "b"]
        assert form.output_fields == ["c"]

    def test_form_default_empty_fields(self):
        """Form should default to empty fields if no assignment."""
        form = Form()

        assert form.input_fields == []
        assert form.output_fields == []

    def test_form_available_data(self):
        """Form should accept initial available_data."""
        form = Form(
            assignment="a -> b",
            available_data={"a": 1, "extra": 2},
        )

        assert form.available_data["a"] == 1
        assert form.available_data["extra"] == 2

    def test_form_default_filled_false(self):
        """Form should default to filled=False."""
        form = Form(assignment="a -> b")
        assert form.filled is False

    def test_form_default_output_none(self):
        """Form should default output to None."""
        form = Form(assignment="a -> b")
        assert form.output is None


# =============================================================================
# Tests: Form.from_phrase
# =============================================================================


class TestFormFromPhrase:
    """Tests for Form.from_phrase factory."""

    def test_from_phrase_basic(self):
        """Form.from_phrase should create form bound to phrase."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="count")
        op = Operable([spec1, spec2])

        async def handler(options, ctx):
            return {"count": len(options.name)}

        p = Phrase(
            name="count_chars",
            operable=op,
            inputs={"name"},
            outputs={"count"},
            handler=handler,
        )

        form = Form.from_phrase(p)

        assert form.phrase is p
        assert "name" in form.input_fields
        assert "count" in form.output_fields
        assert "name -> count" in form.assignment or "name" in form.assignment

    def test_from_phrase_with_initial_data(self):
        """Form.from_phrase should accept initial data."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="count")
        op = Operable([spec1, spec2])

        async def handler(options, ctx):
            return {"count": len(options.name)}

        p = Phrase(
            name="count_chars",
            operable=op,
            inputs={"name"},
            outputs={"count"},
            handler=handler,
        )

        form = Form.from_phrase(p, name="hello")

        assert form.available_data["name"] == "hello"

    def test_from_phrase_phrase_property(self):
        """Form.phrase property should return bound phrase."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="count")
        op = Operable([spec1, spec2])

        async def handler(options, ctx):
            return {"count": 0}

        p = Phrase(
            name="test",
            operable=op,
            inputs={"name"},
            outputs={"count"},
            handler=handler,
        )

        form = Form.from_phrase(p)
        assert form.phrase is p

    def test_form_without_phrase(self):
        """Form without phrase should have phrase=None."""
        form = Form(assignment="a -> b")
        assert form.phrase is None


# =============================================================================
# Tests: Form.is_workable
# =============================================================================


class TestFormIsWorkable:
    """Tests for Form.is_workable scheduling check."""

    def test_workable_when_all_inputs_present(self):
        """Form should be workable when all inputs in available_data."""
        form = Form(
            assignment="a, b -> c",
            available_data={"a": 1, "b": 2},
        )

        assert form.is_workable() is True

    def test_not_workable_missing_input(self):
        """Form should not be workable when input missing."""
        form = Form(
            assignment="a, b -> c",
            available_data={"a": 1},  # missing b
        )

        assert form.is_workable() is False

    def test_not_workable_none_input(self):
        """Form should not be workable when input is None."""
        form = Form(
            assignment="a, b -> c",
            available_data={"a": 1, "b": None},
        )

        assert form.is_workable() is False

    def test_not_workable_when_filled(self):
        """Form should not be workable when already filled."""
        form = Form(
            assignment="a -> b",
            available_data={"a": 1},
            filled=True,
        )

        assert form.is_workable() is False

    def test_workable_with_extra_data(self):
        """Form should be workable with extra data present."""
        form = Form(
            assignment="a -> b",
            available_data={"a": 1, "extra": "ignored"},
        )

        assert form.is_workable() is True

    def test_workable_no_inputs_required(self):
        """Form should be workable when no inputs required."""
        form = Form(assignment=" -> output")

        assert form.is_workable() is True


# =============================================================================
# Tests: Form.get_inputs
# =============================================================================


class TestFormGetInputs:
    """Tests for Form.get_inputs data extraction."""

    def test_get_inputs_extracts_fields(self):
        """get_inputs should return dict of input field values."""
        form = Form(
            assignment="a, b -> c",
            available_data={"a": 1, "b": 2, "extra": 3},
        )

        inputs = form.get_inputs()

        assert inputs == {"a": 1, "b": 2}
        assert "extra" not in inputs

    def test_get_inputs_partial(self):
        """get_inputs should return available inputs only."""
        form = Form(
            assignment="a, b -> c",
            available_data={"a": 1},  # b missing
        )

        inputs = form.get_inputs()

        assert inputs == {"a": 1}
        assert "b" not in inputs

    def test_get_inputs_empty(self):
        """get_inputs should return empty dict when no inputs available."""
        form = Form(
            assignment="a, b -> c",
            available_data={},
        )

        inputs = form.get_inputs()

        assert inputs == {}


# =============================================================================
# Tests: Form.fill
# =============================================================================


class TestFormFill:
    """Tests for Form.fill data population."""

    def test_fill_adds_data(self):
        """fill should add data to available_data."""
        form = Form(assignment="a -> b")
        form.fill(a=1, extra=2)

        assert form.available_data["a"] == 1
        assert form.available_data["extra"] == 2

    def test_fill_updates_existing(self):
        """fill should update existing values."""
        form = Form(
            assignment="a -> b",
            available_data={"a": 1},
        )
        form.fill(a=10)

        assert form.available_data["a"] == 10

    def test_fill_multiple_calls(self):
        """fill should support multiple calls."""
        form = Form(assignment="a, b -> c")
        form.fill(a=1)
        form.fill(b=2)

        assert form.available_data["a"] == 1
        assert form.available_data["b"] == 2


# =============================================================================
# Tests: Form.set_output
# =============================================================================


class TestFormSetOutput:
    """Tests for Form.set_output result handling."""

    def test_set_output_marks_filled(self):
        """set_output should mark form as filled."""
        form = Form(assignment="a -> b")
        form.set_output({"b": 42})

        assert form.filled is True
        assert form.output == {"b": 42}

    def test_set_output_extracts_dict_fields(self):
        """set_output should extract output fields from dict result."""
        form = Form(assignment="a -> b, c")
        form.set_output({"b": 1, "c": 2, "extra": 3})

        assert form.available_data["b"] == 1
        assert form.available_data["c"] == 2
        assert "extra" not in form.available_data

    def test_set_output_extracts_object_fields(self):
        """set_output should extract output fields from object attributes."""

        class Result:
            def __init__(self):
                self.b = 10
                self.c = 20

        form = Form(assignment="a -> b, c")
        form.set_output(Result())

        assert form.available_data["b"] == 10
        assert form.available_data["c"] == 20

    def test_set_output_none_still_fills(self):
        """set_output with None should still mark as filled."""
        form = Form(assignment="a -> b")
        form.set_output(None)

        assert form.filled is True
        assert form.output is None


# =============================================================================
# Tests: Form.get_output_data
# =============================================================================


class TestFormGetOutputData:
    """Tests for Form.get_output_data extraction."""

    def test_get_output_data_returns_outputs(self):
        """get_output_data should return output field values."""
        form = Form(
            assignment="a -> b, c",
            available_data={"a": 1, "b": 2, "c": 3},
        )

        output_data = form.get_output_data()

        assert output_data == {"b": 2, "c": 3}
        assert "a" not in output_data

    def test_get_output_data_partial(self):
        """get_output_data should return available outputs only."""
        form = Form(
            assignment="a -> b, c",
            available_data={"a": 1, "b": 2},  # c missing
        )

        output_data = form.get_output_data()

        assert output_data == {"b": 2}

    def test_get_output_data_empty(self):
        """get_output_data should return empty when no outputs available."""
        form = Form(
            assignment="a -> b, c",
            available_data={"a": 1},
        )

        output_data = form.get_output_data()

        assert output_data == {}


# =============================================================================
# Tests: Form.execute
# =============================================================================


class TestFormExecute:
    """Tests for Form.execute async execution."""

    @pytest.mark.anyio
    async def test_execute_with_phrase(self):
        """execute should invoke bound phrase and set output."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="length")
        op = Operable([spec1, spec2])

        async def handler(options, ctx):
            return {"length": len(options.name)}

        p = Phrase(
            name="get_length",
            operable=op,
            inputs={"name"},
            outputs={"length"},
            handler=handler,
        )

        form = Form.from_phrase(p, name="hello")

        result = await form.execute(ctx=None)

        assert form.filled is True
        assert form.available_data["length"] == 5
        assert result.length == 5

    @pytest.mark.anyio
    async def test_execute_no_phrase_raises(self):
        """execute without bound phrase should raise RuntimeError."""
        form = Form(
            assignment="a -> b",
            available_data={"a": 1},
        )

        with pytest.raises(RuntimeError, match="no bound phrase"):
            await form.execute(ctx=None)

    @pytest.mark.anyio
    async def test_execute_not_workable_raises(self):
        """execute when not workable should raise RuntimeError."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="length")
        op = Operable([spec1, spec2])

        async def handler(options, ctx):
            return {"length": 0}

        p = Phrase(
            name="get_length",
            operable=op,
            inputs={"name"},
            outputs={"length"},
            handler=handler,
        )

        form = Form.from_phrase(p)  # No data provided

        with pytest.raises(RuntimeError, match="not workable"):
            await form.execute(ctx=None)


# =============================================================================
# Tests: Form.__repr__
# =============================================================================


class TestFormRepr:
    """Tests for Form string representation."""

    def test_repr_pending(self):
        """repr should show 'pending' when not workable."""
        form = Form(assignment="a -> b")
        repr_str = repr(form)

        assert "Form" in repr_str
        assert "pending" in repr_str

    def test_repr_ready(self):
        """repr should show 'ready' when workable."""
        form = Form(
            assignment="a -> b",
            available_data={"a": 1},
        )
        repr_str = repr(form)

        assert "ready" in repr_str

    def test_repr_filled(self):
        """repr should show 'filled' when filled."""
        form = Form(
            assignment="a -> b",
            filled=True,
        )
        repr_str = repr(form)

        assert "filled" in repr_str

    def test_repr_with_phrase(self):
        """repr should include phrase name when bound."""
        spec1 = Spec(str, name="a")
        spec2 = Spec(int, name="b")
        op = Operable([spec1, spec2])

        async def handler(options, ctx):
            return {"b": 0}

        p = Phrase(
            name="test_phrase",
            operable=op,
            inputs={"a"},
            outputs={"b"},
            handler=handler,
        )

        form = Form.from_phrase(p)
        repr_str = repr(form)

        assert "test_phrase" in repr_str


# =============================================================================
# Tests: Form Element inheritance
# =============================================================================


class TestFormElement:
    """Tests for Form Element inheritance."""

    def test_form_has_id(self):
        """Form should have UUID id from Element."""
        form = Form(assignment="a -> b")
        assert isinstance(form.id, UUID)

    def test_form_has_created_at(self):
        """Form should have created_at from Element."""
        form = Form(assignment="a -> b")
        assert form.created_at is not None

    def test_form_serializable(self):
        """Form should be serializable via to_dict."""
        form = Form(
            assignment="a -> b",
            available_data={"a": 1},
        )

        data = form.to_dict(mode="json")

        assert "id" in data
        assert "assignment" in data
        assert "input_fields" in data
        assert "output_fields" in data
        assert "available_data" in data

    def test_form_roundtrip(self):
        """Form should support serialization roundtrip."""
        form = Form(
            assignment="a, b -> c, d",
            available_data={"a": 1, "b": 2},
        )

        data = form.to_dict(mode="json")
        restored = Form.from_dict(data)

        assert restored.id == form.id
        assert restored.assignment == form.assignment
        assert restored.input_fields == form.input_fields
        assert restored.available_data == form.available_data
