# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.core.element - Element base class with UUID identity."""

import datetime as dt
from uuid import UUID

import pytest

from krons.core import Element


class TestElementCreation:
    """Test Element instantiation and identity."""

    def test_element_has_uuid(self):
        """Element should have a valid UUID id."""
        elem = Element()
        assert isinstance(elem.id, UUID)

    def test_element_has_timestamps(self):
        """Element should have created_at timestamp."""
        elem = Element()
        assert isinstance(elem.created_at, dt.datetime)
        # Should be timezone-aware UTC
        assert elem.created_at.tzinfo is not None

    def test_element_metadata(self):
        """Element should support metadata dict."""
        elem = Element(metadata={"key": "value", "nested": {"a": 1}})
        assert elem.metadata["key"] == "value"
        assert elem.metadata["nested"]["a"] == 1

    def test_element_metadata_defaults_empty(self):
        """Element metadata should default to empty dict."""
        elem = Element()
        assert elem.metadata == {}

    def test_element_id_is_frozen(self):
        """Element id should be immutable after creation."""
        elem = Element()
        with pytest.raises(Exception):  # ValidationError for frozen field
            elem.id = UUID("12345678-1234-5678-1234-567812345678")

    def test_element_created_at_is_frozen(self):
        """Element created_at should be immutable after creation."""
        elem = Element()
        with pytest.raises(Exception):  # ValidationError for frozen field
            elem.created_at = dt.datetime.now(dt.timezone.utc)

    def test_element_with_custom_id(self):
        """Element should accept custom UUID id."""
        custom_id = UUID("12345678-1234-5678-1234-567812345678")
        elem = Element(id=custom_id)
        assert elem.id == custom_id

    def test_element_with_string_id(self):
        """Element should coerce string to UUID."""
        elem = Element(id="12345678-1234-5678-1234-567812345678")
        assert isinstance(elem.id, UUID)
        assert str(elem.id) == "12345678-1234-5678-1234-567812345678"


class TestElementSerialization:
    """Test Element serialization and deserialization."""

    def test_to_dict(self):
        """Element.to_dict() should produce valid dict."""
        elem = Element(metadata={"key": "value"})
        data = elem.to_dict()

        assert "id" in data
        assert "created_at" in data
        assert "metadata" in data
        assert data["metadata"]["key"] == "value"
        # kron_class should be injected
        assert "kron_class" in data["metadata"]

    def test_to_dict_json_mode(self):
        """Element.to_dict(mode='json') should produce JSON-serializable dict."""
        elem = Element()
        data = elem.to_dict(mode="json")

        # created_at should be isoformat string in json mode
        assert isinstance(data["created_at"], str)
        # id should be string in json mode
        assert isinstance(data["id"], str)

    def test_from_dict(self):
        """Element.from_dict() should reconstruct Element."""
        elem = Element(metadata={"key": "value"})
        data = elem.to_dict()
        restored = Element.from_dict(data)

        assert restored.id == elem.id
        assert restored.metadata["key"] == "value"

    def test_roundtrip(self):
        """Serialization roundtrip should preserve identity."""
        elem = Element(metadata={"nested": {"deep": [1, 2, 3]}})
        data = elem.to_dict(mode="json")
        restored = Element.from_dict(data)

        assert restored.id == elem.id
        assert restored.created_at == elem.created_at
        assert restored.metadata["nested"]["deep"] == [1, 2, 3]

    def test_to_dict_db_mode(self):
        """Element.to_dict(mode='db') should rename metadata to node_metadata."""
        elem = Element(metadata={"key": "value"})
        data = elem.to_dict(mode="db")

        assert "node_metadata" in data
        assert "metadata" not in data
        assert data["node_metadata"]["key"] == "value"

    def test_from_dict_with_meta_key(self):
        """Element.from_dict() should handle custom meta_key."""
        elem = Element(metadata={"key": "value"})
        data = elem.to_dict(mode="db")

        restored = Element.from_dict(data, meta_key="node_metadata")
        assert restored.metadata["key"] == "value"


class TestElementEquality:
    """Test Element equality and hashing."""

    def test_equality_by_id(self):
        """Elements with same id should be equal."""
        elem1 = Element()
        elem2 = Element(id=elem1.id)

        assert elem1 == elem2

    def test_inequality_different_id(self):
        """Elements with different ids should not be equal."""
        elem1 = Element()
        elem2 = Element()

        assert elem1 != elem2

    def test_hash_by_id(self):
        """Elements should be hashable by id."""
        elem1 = Element()
        elem2 = Element(id=elem1.id)

        assert hash(elem1) == hash(elem2)

    def test_elements_in_set(self):
        """Elements can be used in sets."""
        elem1 = Element()
        elem2 = Element(id=elem1.id)
        elem3 = Element()

        s = {elem1, elem2, elem3}
        assert len(s) == 2  # elem1 and elem2 are same (same id)

    def test_elements_as_dict_keys(self):
        """Elements can be used as dict keys."""
        elem1 = Element()
        elem2 = Element(id=elem1.id)

        d = {elem1: "value1"}
        d[elem2] = "value2"

        assert len(d) == 1  # Same key (same id)
        assert d[elem1] == "value2"

    def test_element_always_truthy(self):
        """Elements should always be truthy."""
        elem = Element()
        assert bool(elem) is True

    def test_element_repr(self):
        """Element repr should show class name and id."""
        elem = Element()
        repr_str = repr(elem)
        assert "Element" in repr_str
        assert str(elem.id) in repr_str
