# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.core.progression - Ordered UUID sequences."""

from uuid import UUID, uuid4

import pytest

from krons.core import Progression
from krons.errors import NotFoundError


class TestProgressionCreation:
    """Test Progression instantiation."""

    def test_empty_progression(self):
        """Empty Progression should have len 0."""
        prog = Progression()
        assert len(prog) == 0

    def test_progression_with_order(self, test_elements):
        """Progression should maintain order list."""
        ids = [e.id for e in test_elements]
        prog = Progression(order=ids)
        assert len(prog) == len(ids)
        assert list(prog) == ids

    def test_progression_name(self):
        """Progression should have optional name."""
        prog = Progression(name="test_progression")
        assert prog.name == "test_progression"

    def test_progression_name_none(self):
        """Progression should allow None name."""
        prog = Progression()
        assert prog.name is None

    def test_progression_coerces_elements(self, test_elements):
        """Progression should coerce Elements to UUIDs."""
        prog = Progression(order=test_elements)
        assert all(isinstance(uid, UUID) for uid in prog)

    def test_progression_coerces_strings(self):
        """Progression should coerce string UUIDs."""
        id1 = str(uuid4())
        id2 = str(uuid4())
        prog = Progression(order=[id1, id2])
        assert all(isinstance(uid, UUID) for uid in prog)


class TestProgressionOrder:
    """Test Progression ordering operations."""

    def test_append(self, test_progression):
        """Progression.append() should add UUID to end."""
        initial_len = len(test_progression)
        new_id = uuid4()
        test_progression.append(new_id)

        assert len(test_progression) == initial_len + 1
        assert test_progression[-1] == new_id

    def test_insert(self, test_progression):
        """Progression.insert() should add at position."""
        new_id = uuid4()
        test_progression.insert(0, new_id)

        assert test_progression[0] == new_id

    def test_insert_middle(self, test_progression):
        """Progression.insert() should work in middle."""
        new_id = uuid4()
        test_progression.insert(2, new_id)

        assert test_progression[2] == new_id

    def test_remove(self, test_progression):
        """Progression.remove() should remove UUID."""
        first_id = test_progression[0]
        initial_len = len(test_progression)

        test_progression.remove(first_id)

        assert len(test_progression) == initial_len - 1
        assert first_id not in test_progression

    def test_remove_raises_if_not_found(self):
        """Progression.remove() should raise ValueError if not found."""
        prog = Progression(order=[uuid4()])
        with pytest.raises(ValueError):
            prog.remove(uuid4())

    def test_pop_default(self, test_progression):
        """Progression.pop() should remove last item."""
        last_id = test_progression[-1]
        initial_len = len(test_progression)

        popped = test_progression.pop()

        assert popped == last_id
        assert len(test_progression) == initial_len - 1

    def test_pop_index(self, test_progression):
        """Progression.pop(index) should remove at position."""
        first_id = test_progression[0]
        initial_len = len(test_progression)

        popped = test_progression.pop(0)

        assert popped == first_id
        assert len(test_progression) == initial_len - 1

    def test_pop_with_default(self):
        """Progression.pop() should return default if index out of range."""
        prog = Progression()
        result = prog.pop(0, default="default")
        assert result == "default"

    def test_popleft(self, test_progression):
        """Progression.popleft() should remove first item."""
        first_id = test_progression[0]
        initial_len = len(test_progression)

        popped = test_progression.popleft()

        assert popped == first_id
        assert len(test_progression) == initial_len - 1

    def test_popleft_empty_raises(self):
        """Progression.popleft() on empty should raise NotFoundError."""
        prog = Progression()
        with pytest.raises(NotFoundError):
            prog.popleft()

    def test_clear(self, test_progression):
        """Progression.clear() should remove all UUIDs."""
        test_progression.clear()
        assert len(test_progression) == 0

    def test_extend(self, test_progression):
        """Progression.extend() should add multiple UUIDs."""
        initial_len = len(test_progression)
        new_ids = [uuid4(), uuid4(), uuid4()]
        test_progression.extend(new_ids)

        assert len(test_progression) == initial_len + 3


class TestProgressionIteration:
    """Test Progression iteration."""

    def test_iter(self, test_progression):
        """iter(progression) should yield UUIDs in order."""
        uuids = list(test_progression)
        assert len(uuids) == len(test_progression)
        assert all(isinstance(uid, UUID) for uid in uuids)

    def test_len(self, test_progression):
        """len(progression) should return count."""
        assert len(test_progression) == 5  # test_progression fixture uses 5 elements

    def test_getitem_index(self, test_progression):
        """Progression[int] should return UUID."""
        first = test_progression[0]
        assert isinstance(first, UUID)

    def test_getitem_slice(self, test_progression):
        """Progression[slice] should return list of UUIDs."""
        sliced = test_progression[1:3]
        assert isinstance(sliced, list)
        assert len(sliced) == 2

    def test_setitem_index(self, test_progression):
        """Progression[int] = uuid should replace."""
        new_id = uuid4()
        test_progression[0] = new_id
        assert test_progression[0] == new_id

    def test_setitem_slice(self, test_progression):
        """Progression[slice] = list should replace range."""
        new_ids = [uuid4(), uuid4()]
        test_progression[1:3] = new_ids
        assert test_progression[1] == new_ids[0]
        assert test_progression[2] == new_ids[1]

    def test_contains(self, test_progression):
        """uuid in progression should check membership."""
        first_id = test_progression[0]
        assert first_id in test_progression

    def test_contains_not_found(self, test_progression):
        """uuid not in progression should return False."""
        assert uuid4() not in test_progression

    def test_reversed(self, test_progression):
        """reversed(progression) should iterate in reverse."""
        forward = list(test_progression)
        backward = list(reversed(test_progression))
        assert backward == list(reversed(forward))

    def test_bool_true(self, test_progression):
        """Non-empty progression should be truthy."""
        assert bool(test_progression) is True

    def test_bool_false(self):
        """Empty progression should be falsy."""
        prog = Progression()
        assert bool(prog) is False


class TestProgressionWorkflow:
    """Test Progression workflow operations."""

    def test_move(self, test_progression):
        """Progression.move() should relocate item."""
        original_first = test_progression[0]
        test_progression.move(0, 2)

        assert test_progression[1] == original_first

    def test_swap(self, test_progression):
        """Progression.swap() should exchange items."""
        first = test_progression[0]
        second = test_progression[1]

        test_progression.swap(0, 1)

        assert test_progression[0] == second
        assert test_progression[1] == first

    def test_reverse(self, test_progression):
        """Progression.reverse() should reverse order."""
        original = list(test_progression)
        test_progression.reverse()

        assert list(test_progression) == list(reversed(original))

    def test_index(self, test_progression):
        """Progression.index() should return position."""
        first_id = test_progression[0]
        assert test_progression.index(first_id) == 0


class TestProgressionSetOperations:
    """Test Progression set-like operations."""

    def test_include_new(self, test_progression):
        """Progression.include() should add new UUID."""
        new_id = uuid4()
        result = test_progression.include(new_id)

        assert result is True
        assert new_id in test_progression

    def test_include_existing(self, test_progression):
        """Progression.include() should be idempotent for existing."""
        first_id = test_progression[0]
        initial_len = len(test_progression)

        result = test_progression.include(first_id)

        assert result is False
        assert len(test_progression) == initial_len

    def test_exclude_existing(self, test_progression):
        """Progression.exclude() should remove existing UUID."""
        first_id = test_progression[0]

        result = test_progression.exclude(first_id)

        assert result is True
        assert first_id not in test_progression

    def test_exclude_missing(self, test_progression):
        """Progression.exclude() should be idempotent for missing."""
        result = test_progression.exclude(uuid4())
        assert result is False


class TestProgressionRepr:
    """Test Progression string representation."""

    def test_repr_with_name(self):
        """Progression repr should show name if set."""
        prog = Progression(name="test", order=[uuid4(), uuid4()])
        repr_str = repr(prog)
        assert "Progression" in repr_str
        assert "len=2" in repr_str
        assert "test" in repr_str

    def test_repr_without_name(self):
        """Progression repr should work without name."""
        prog = Progression(order=[uuid4()])
        repr_str = repr(prog)
        assert "Progression" in repr_str
        assert "len=1" in repr_str
