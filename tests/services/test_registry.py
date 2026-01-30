# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.services.registry - ServiceRegistry."""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import BaseModel

from krons.services import Endpoint, EndpointConfig, ServiceRegistry
from krons.services.imodel import iModel

# =============================================================================
# Test Request Model
# =============================================================================


class RequestModel(BaseModel):
    """Request model for testing (not a test class)."""

    message: str
    temperature: float = 0.7


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def registry():
    """Create empty ServiceRegistry."""
    return ServiceRegistry()


@pytest.fixture
def mock_endpoint():
    """Create mock endpoint."""
    config = EndpointConfig(
        name="test_service",
        provider="test_provider",
        endpoint="test",
        base_url="https://api.test.com",
        request_options=RequestModel,
    )
    return Endpoint(config=config)


@pytest.fixture
def mock_imodel(mock_endpoint):
    """Create mock iModel."""
    return iModel(backend=mock_endpoint)


def create_mock_imodel(name: str) -> iModel:
    """Create iModel with specified name."""
    config = EndpointConfig(
        name=name,
        provider="test_provider",
        endpoint="test",
        base_url="https://api.test.com",
        request_options=RequestModel,
    )
    endpoint = Endpoint(config=config)
    return iModel(backend=endpoint)


# =============================================================================
# ServiceRegistry Basic Tests
# =============================================================================


class TestServiceRegistry:
    """Test ServiceRegistry operations."""

    def test_register_service(self, registry, mock_imodel):
        """Registry.register() should add service."""
        uid = registry.register(mock_imodel)

        assert isinstance(uid, UUID)
        assert uid == mock_imodel.id
        assert registry.count() == 1

    def test_get_service(self, registry, mock_imodel):
        """Registry.get() should retrieve by name."""
        registry.register(mock_imodel)

        result = registry.get("test_service")

        assert result is mock_imodel
        assert result.name == "test_service"

    def test_get_with_default(self, registry):
        """Registry.get() should return default if not found."""
        default = "default_value"
        result = registry.get("nonexistent", default=default)

        assert result == default

    def test_has_service(self, registry, mock_imodel):
        """Registry.has() should check existence."""
        assert registry.has("test_service") is False

        registry.register(mock_imodel)

        assert registry.has("test_service") is True
        assert registry.has("nonexistent") is False

    def test_list_names(self, registry):
        """Registry.list_names() should return all names."""
        model1 = create_mock_imodel("service_a")
        model2 = create_mock_imodel("service_b")
        model3 = create_mock_imodel("service_c")

        registry.register(model1)
        registry.register(model2)
        registry.register(model3)

        names = registry.list_names()

        assert len(names) == 3
        assert "service_a" in names
        assert "service_b" in names
        assert "service_c" in names


# =============================================================================
# ServiceRegistry Registration Tests
# =============================================================================


class TestServiceRegistryRegistration:
    """Test ServiceRegistry registration behavior."""

    def test_register_duplicate_raises(self, registry, mock_imodel):
        """Register duplicate name without update=True should raise."""
        registry.register(mock_imodel)

        # Create another model with same name
        duplicate = create_mock_imodel("test_service")

        with pytest.raises(ValueError, match="already registered"):
            registry.register(duplicate)

    def test_register_duplicate_with_update(self, registry, mock_imodel):
        """Register duplicate with update=True should replace."""
        registry.register(mock_imodel)
        old_id = mock_imodel.id

        # Create another model with same name
        new_model = create_mock_imodel("test_service")

        uid = registry.register(new_model, update=True)

        assert uid == new_model.id
        assert uid != old_id
        assert registry.count() == 1
        assert registry.get("test_service") is new_model


# =============================================================================
# ServiceRegistry Retrieval Tests
# =============================================================================


class TestServiceRegistryRetrieval:
    """Test ServiceRegistry retrieval operations."""

    def test_get_by_uuid(self, registry, mock_imodel):
        """Registry.get() should accept UUID."""
        registry.register(mock_imodel)

        result = registry.get(mock_imodel.id)

        assert result is mock_imodel

    def test_get_by_imodel(self, registry, mock_imodel):
        """Registry.get() should return iModel if passed directly."""
        # Don't even need to register
        result = registry.get(mock_imodel)

        assert result is mock_imodel

    def test_get_not_found_raises(self, registry):
        """Registry.get() without default should raise KeyError."""
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_unregister(self, registry, mock_imodel):
        """Registry.unregister() should remove and return service."""
        registry.register(mock_imodel)

        result = registry.unregister("test_service")

        assert result is mock_imodel
        assert registry.count() == 0
        assert registry.has("test_service") is False

    def test_unregister_not_found_raises(self, registry):
        """Registry.unregister() should raise if not found."""
        with pytest.raises(KeyError, match="not found"):
            registry.unregister("nonexistent")


# =============================================================================
# ServiceRegistry Tag Filtering Tests
# =============================================================================


class TestServiceRegistryTagFiltering:
    """Test ServiceRegistry tag-based filtering."""

    def test_list_by_tag(self, registry):
        """Registry.list_by_tag() should filter by tag."""
        # Create models with different tags
        config1 = EndpointConfig(
            name="api_service",
            provider="test_provider",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=RequestModel,
            tags=["api", "rest"],
        )
        model1 = iModel(backend=Endpoint(config=config1))

        config2 = EndpointConfig(
            name="llm_service",
            provider="test_provider",
            endpoint="chat",
            base_url="https://api.test.com",
            request_options=RequestModel,
            tags=["api", "llm"],
        )
        model2 = iModel(backend=Endpoint(config=config2))

        config3 = EndpointConfig(
            name="internal_service",
            provider="test_provider",
            endpoint="internal",
            base_url="https://api.test.com",
            request_options=RequestModel,
            tags=["internal"],
        )
        model3 = iModel(backend=Endpoint(config=config3))

        registry.register(model1)
        registry.register(model2)
        registry.register(model3)

        # Filter by "api" tag
        api_services = registry.list_by_tag("api")
        assert len(api_services) == 2

        # Filter by "llm" tag
        llm_services = registry.list_by_tag("llm")
        assert len(llm_services) == 1

        # Filter by "internal" tag
        internal_services = registry.list_by_tag("internal")
        assert len(internal_services) == 1

        # Filter by non-existent tag
        no_services = registry.list_by_tag("nonexistent")
        assert len(no_services) == 0


# =============================================================================
# ServiceRegistry Utility Methods Tests
# =============================================================================


class TestServiceRegistryUtilityMethods:
    """Test ServiceRegistry utility methods."""

    def test_count(self, registry):
        """Registry.count() should return number of services."""
        assert registry.count() == 0

        model1 = create_mock_imodel("service_a")
        model2 = create_mock_imodel("service_b")

        registry.register(model1)
        assert registry.count() == 1

        registry.register(model2)
        assert registry.count() == 2

    def test_clear(self, registry):
        """Registry.clear() should remove all services."""
        model1 = create_mock_imodel("service_a")
        model2 = create_mock_imodel("service_b")

        registry.register(model1)
        registry.register(model2)
        assert registry.count() == 2

        registry.clear()

        assert registry.count() == 0
        assert registry.has("service_a") is False
        assert registry.has("service_b") is False

    def test_len(self, registry):
        """len(registry) should return number of services."""
        assert len(registry) == 0

        model1 = create_mock_imodel("service_a")
        registry.register(model1)

        assert len(registry) == 1

    def test_contains(self, registry, mock_imodel):
        """'name in registry' should check existence."""
        assert "test_service" not in registry

        registry.register(mock_imodel)

        assert "test_service" in registry
        assert "nonexistent" not in registry

    def test_repr(self, registry):
        """Registry repr should show count."""
        repr_str = repr(registry)
        assert "ServiceRegistry" in repr_str
        assert "count=0" in repr_str

        model = create_mock_imodel("service_a")
        registry.register(model)

        repr_str = repr(registry)
        assert "count=1" in repr_str
