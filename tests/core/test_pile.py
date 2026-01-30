# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.core.pile - O(1) collection with UUID lookup."""

import pytest

from krons.core import Element, Pile
from krons.errors import ExistsError, NotFoundError


class TestPileCreation:
    """Test Pile instantiation."""

    def test_empty_pile(self):
        """Empty Pile should have len 0."""
        pile = Pile()
        assert len(pile) == 0
        assert pile.is_empty()

    def test_pile_with_items(self, test_elements):
        """Pile should accept list of items."""
        pile = Pile(items=test_elements)
        assert len(pile) == len(test_elements)

    def test_pile_type_constraint(self, test_elements):
        """Pile with item_type should reject wrong types."""
        from tests.conftest import TestElement

        pile = Pile(item_type=TestElement)

        # Should accept TestElement
        pile.add(test_elements[0])

        # Should reject plain Element
        with pytest.raises(TypeError):
            pile.add(Element())

    def test_pile_strict_type(self):
        """Pile with strict_type should reject subclasses."""
        from tests.conftest import TestElement

        pile = Pile(item_type=Element, strict_type=True)

        # Should accept exact Element
        pile.add(Element())

        # Should reject TestElement (subclass)
        with pytest.raises(TypeError):
            pile.add(TestElement())

    def test_pile_with_order(self, test_elements):
        """Pile should accept custom order."""
        # Reverse the order
        reversed_ids = [e.id for e in reversed(test_elements)]
        pile = Pile(items=test_elements, order=reversed_ids)

        # Iteration should follow custom order
        iterated = list(pile)
        assert iterated[0].id == reversed_ids[0]


class TestPileLookup:
    """Test Pile O(1) lookup operations."""

    def test_get_by_uuid(self, test_pile):
        """Pile[uuid] should return item in O(1)."""
        first_item = list(test_pile)[0]
        retrieved = test_pile[first_item.id]
        assert retrieved.id == first_item.id

    def test_get_by_string_uuid(self, test_pile):
        """Pile[str] should return item by string UUID."""
        first_item = list(test_pile)[0]
        retrieved = test_pile[str(first_item.id)]
        assert retrieved.id == first_item.id

    def test_get_by_index(self, test_pile):
        """Pile[int] should return item by position."""
        items = list(test_pile)
        assert test_pile[0].id == items[0].id
        assert test_pile[1].id == items[1].id
        assert test_pile[-1].id == items[-1].id

    def test_get_by_slice(self, test_pile):
        """Pile[slice] should return new Pile."""
        sliced = test_pile[1:3]
        assert isinstance(sliced, Pile)
        assert len(sliced) == 2

    def test_get_by_list_of_indices(self, test_pile):
        """Pile[[int, int]] should return new Pile."""
        selected = test_pile[[0, 2, 4]]
        assert isinstance(selected, Pile)
        assert len(selected) == 3

    def test_get_by_callable(self, test_pile):
        """Pile[callable] should filter items."""
        from tests.conftest import TestElement

        filtered = test_pile[lambda x: isinstance(x, TestElement) and x.value < 3]
        assert isinstance(filtered, Pile)
        assert all(item.value < 3 for item in filtered)

    def test_contains(self, test_pile):
        """item in pile should check by UUID."""
        first_item = list(test_pile)[0]
        assert first_item in test_pile
        assert first_item.id in test_pile

    def test_contains_not_found(self, test_pile):
        """item not in pile should return False."""
        new_element = Element()
        assert new_element not in test_pile
        assert new_element.id not in test_pile

    def test_get_with_default(self, test_pile):
        """Pile.get() should return default if not found."""
        from uuid import uuid4

        result = test_pile.get(uuid4(), default=None)
        assert result is None

    def test_get_raises_not_found(self, test_pile):
        """Pile.get() should raise NotFoundError if no default."""
        from uuid import uuid4

        with pytest.raises(NotFoundError):
            test_pile.get(uuid4())


class TestPileMutation:
    """Test Pile add/remove operations."""

    def test_add_item(self, test_pile):
        """Pile.add() should append item."""
        from tests.conftest import TestElement

        initial_len = len(test_pile)
        new_item = TestElement(value=100)
        test_pile.add(new_item)

        assert len(test_pile) == initial_len + 1
        assert new_item in test_pile

    def test_add_duplicate_raises(self, test_pile):
        """Pile.add() should raise ExistsError for duplicate."""
        first_item = list(test_pile)[0]
        with pytest.raises(ExistsError):
            test_pile.add(first_item)

    def test_remove_item(self, test_pile):
        """Pile.remove() should remove by UUID."""
        first_item = list(test_pile)[0]
        initial_len = len(test_pile)

        removed = test_pile.remove(first_item.id)

        assert removed.id == first_item.id
        assert len(test_pile) == initial_len - 1
        assert first_item not in test_pile

    def test_remove_not_found_raises(self, test_pile):
        """Pile.remove() should raise NotFoundError."""
        from uuid import uuid4

        with pytest.raises(NotFoundError):
            test_pile.remove(uuid4())

    def test_pop_item(self, test_pile):
        """Pile.pop() should remove and return."""
        first_item = list(test_pile)[0]
        initial_len = len(test_pile)

        popped = test_pile.pop(first_item.id)

        assert popped.id == first_item.id
        assert len(test_pile) == initial_len - 1

    def test_pop_with_default(self, test_pile):
        """Pile.pop() should return default if not found."""
        from uuid import uuid4

        result = test_pile.pop(uuid4(), default="not found")
        assert result == "not found"

    def test_update_item(self, test_pile):
        """Pile.update() should replace existing item."""
        from tests.conftest import TestElement

        first_item = list(test_pile)[0]
        updated_item = TestElement(id=first_item.id, value=999, name="updated")

        test_pile.update(updated_item)

        retrieved = test_pile[first_item.id]
        assert retrieved.value == 999
        assert retrieved.name == "updated"

    def test_clear(self, test_pile):
        """Pile.clear() should remove all items."""
        test_pile.clear()
        assert len(test_pile) == 0
        assert test_pile.is_empty()


class TestPileSetOperations:
    """Test Pile set-like include/exclude operations."""

    def test_include_new_item(self, test_pile):
        """Pile.include() should add new item."""
        from tests.conftest import TestElement

        new_item = TestElement(value=100)
        result = test_pile.include(new_item)

        assert result is True
        assert new_item in test_pile

    def test_include_existing_item(self, test_pile):
        """Pile.include() should be idempotent for existing item."""
        first_item = list(test_pile)[0]
        initial_len = len(test_pile)

        result = test_pile.include(first_item)

        assert result is True
        assert len(test_pile) == initial_len

    def test_exclude_existing_item(self, test_pile):
        """Pile.exclude() should remove existing item."""
        first_item = list(test_pile)[0]
        initial_len = len(test_pile)

        result = test_pile.exclude(first_item)

        assert result is True
        assert first_item not in test_pile
        assert len(test_pile) == initial_len - 1

    def test_exclude_missing_item(self, test_pile):
        """Pile.exclude() should be idempotent for missing item."""
        from uuid import uuid4

        result = test_pile.exclude(uuid4())
        assert result is True


class TestPileIteration:
    """Test Pile iteration."""

    def test_iter(self, test_pile):
        """iter(pile) should yield items in order."""
        items = list(test_pile)
        assert len(items) == len(test_pile)
        # All items should be Elements
        assert all(isinstance(item, Element) for item in items)

    def test_len(self, test_pile):
        """len(pile) should return count."""
        assert len(test_pile) == 5  # test_pile fixture has 5 items

    def test_keys(self, test_pile):
        """Pile.keys() should yield UUIDs."""
        keys = list(test_pile.keys())
        assert len(keys) == len(test_pile)
        from uuid import UUID

        assert all(isinstance(k, UUID) for k in keys)

    def test_items(self, test_pile):
        """Pile.items() should yield (UUID, item) pairs."""
        for uid, item in test_pile.items():
            assert uid == item.id

    def test_bool_true(self, test_pile):
        """Non-empty pile should be truthy."""
        assert bool(test_pile) is True

    def test_bool_false(self):
        """Empty pile should be falsy."""
        pile = Pile()
        assert bool(pile) is False


class TestPileSerialization:
    """Test Pile serialization."""

    def test_to_dict(self, test_pile):
        """Pile.to_dict() should serialize items."""
        data = test_pile.to_dict()

        assert "items" in data
        assert len(data["items"]) == len(test_pile)

    def test_roundtrip(self, test_pile):
        """Serialization roundtrip should preserve items."""
        data = test_pile.to_dict(mode="json")
        restored = Pile.from_dict(data)

        assert len(restored) == len(test_pile)

        # Check all items preserved
        for original, restored_item in zip(test_pile, restored):
            assert original.id == restored_item.id

    def test_from_dict_with_item_type(self, test_elements):
        """Pile.from_dict() should validate item types."""
        from tests.conftest import TestElement

        pile = Pile(items=test_elements, item_type=TestElement)
        data = pile.to_dict(mode="json")
        restored = Pile.from_dict(data)

        assert len(restored) == len(test_elements)
