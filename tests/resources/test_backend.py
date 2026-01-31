# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.resources.backend - ResourceBackend and Calling."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from krons.core import EventStatus
from krons.core.types import Unset, is_sentinel
from krons.resources import Calling, NormalizedResponse, ResourceBackend, ResourceConfig

# =============================================================================
# Mock Components
# =============================================================================


class SimpleRequestModel(BaseModel):
    """Simple request model for testing."""

    message: str
    temperature: float = 0.7


class MockCalling(Calling):
    """Mock Calling implementation for testing."""

    @property
    def call_args(self) -> dict:
        """Return payload as call arguments."""
        return self.payload

    async def _stream(self):
        """Mock stream implementation."""
        raise NotImplementedError("Stream not implemented")


class MockResourceBackend(ResourceBackend):
    """Mock ResourceBackend for testing properties."""

    result_value: str = "test_result"
    should_fail: bool = False
    should_cancel: bool = False

    @property
    def event_type(self) -> type[Calling]:
        """Return MockCalling type."""
        return MockCalling

    async def call(self, *args, **kw) -> NormalizedResponse:
        """Mock call implementation."""
        if self.should_cancel:
            import asyncio

            raise asyncio.CancelledError("Test cancellation")
        if self.should_fail:
            raise RuntimeError("Test failure")

        return NormalizedResponse(
            status="success",
            data=self.result_value,
            raw_response={"result": self.result_value},
        )


@pytest.fixture
def mock_backend():
    """Create mock service backend."""
    config = ResourceConfig(provider="test_provider", name="test_service")
    return MockResourceBackend(config=config)


@pytest.fixture
def mock_calling(mock_backend):
    """Create mock calling instance."""
    return MockCalling(backend=mock_backend, payload={})


# =============================================================================
# ResourceConfig Tests
# =============================================================================


class TestResourceConfig:
    """Test ResourceConfig validation."""

    def test_config_required_fields(self):
        """ResourceConfig should require provider and name."""
        config = ResourceConfig(provider="test_provider", name="test_service")
        assert config.provider == "test_provider"
        assert config.name == "test_service"

    def test_config_required_fields_validation(self):
        """ResourceConfig should validate min length for provider and name."""
        # Provider too short
        with pytest.raises(
            ValidationError, match="String should have at least 4 characters"
        ):
            ResourceConfig(provider="abc", name="test_service")

        # Name too short
        with pytest.raises(
            ValidationError, match="String should have at least 4 characters"
        ):
            ResourceConfig(provider="test_provider", name="abc")

    def test_config_kwargs_handling(self):
        """ResourceConfig should capture extra kwargs."""
        config = ResourceConfig(
            provider="test_provider",
            name="test_service",
            extra_field="extra_value",
            another_field=123,
        )
        assert config.kwargs["extra_field"] == "extra_value"
        assert config.kwargs["another_field"] == 123

    def test_config_with_request_options(self):
        """ResourceConfig should accept request_options as Pydantic model type."""
        config = ResourceConfig(
            provider="test_provider",
            name="test_service",
            request_options=SimpleRequestModel,
        )
        assert config.request_options == SimpleRequestModel

    def test_config_with_request_options_instance(self):
        """ResourceConfig should accept request_options as Pydantic instance."""
        instance = SimpleRequestModel(message="test")
        config = ResourceConfig(
            provider="test_provider",
            name="test_service",
            request_options=instance,
        )
        # Should extract the class from the instance
        assert config.request_options == SimpleRequestModel

    def test_config_validate_payload(self):
        """ResourceConfig.validate_payload should validate against request_options."""
        config = ResourceConfig(
            provider="test_provider",
            name="test_service",
            request_options=SimpleRequestModel,
        )
        data = {"message": "hello", "temperature": 0.5}
        result = config.validate_payload(data)
        assert result == data

    def test_config_validate_payload_no_request_options(self):
        """ResourceConfig.validate_payload should return data when no request_options."""
        config = ResourceConfig(
            provider="test_provider",
            name="test_service",
            request_options=None,
        )
        data = {"any_field": "any_value"}
        result = config.validate_payload(data)
        assert result == data

    def test_config_validate_payload_invalid_raises(self):
        """ResourceConfig.validate_payload should raise on invalid data."""
        config = ResourceConfig(
            provider="test_provider",
            name="test_service",
            request_options=SimpleRequestModel,
        )
        data = {"temperature": "not_a_float"}  # Missing required 'message'
        with pytest.raises(ValueError, match="Invalid payload"):
            config.validate_payload(data)


# =============================================================================
# NormalizedResponse Tests
# =============================================================================


class TestNormalizedResponse:
    """Test NormalizedResponse structure."""

    def test_success_response(self):
        """NormalizedResponse with status='success'."""
        response = NormalizedResponse(
            status="success",
            data={"key": "value"},
            raw_response={"original": "data"},
        )

        assert response.status == "success"
        assert response.data == {"key": "value"}
        assert response.error is None
        assert response.raw_response == {"original": "data"}
        assert response.metadata is None

    def test_error_response(self):
        """NormalizedResponse with status='error'."""
        response = NormalizedResponse(
            status="error",
            error="Something went wrong",
            raw_response={"error": "details"},
        )

        assert response.status == "error"
        assert response.error == "Something went wrong"
        assert response.data is None

    def test_to_dict_excludes_none(self):
        """_to_dict() should exclude None values."""
        response = NormalizedResponse(
            status="success",
            data="result",
            raw_response={"result": "test"},
        )

        result = response._to_dict()
        assert "error" not in result
        assert "metadata" not in result
        assert result["status"] == "success"
        assert result["data"] == "result"

    def test_to_dict_includes_metadata(self):
        """_to_dict() should include metadata when present."""
        response = NormalizedResponse(
            status="success",
            data="result",
            raw_response={"result": "test"},
            metadata={"usage": {"tokens": 100}},
        )

        result = response._to_dict()
        assert result["metadata"] == {"usage": {"tokens": 100}}


# =============================================================================
# Calling Tests
# =============================================================================


class TestCalling:
    """Test Calling event base class."""

    def test_calling_has_backend(self, mock_backend):
        """Calling should reference a ResourceBackend."""
        calling = MockCalling(backend=mock_backend, payload={"test": "data"})
        assert calling.backend is mock_backend
        assert calling.payload == {"test": "data"}

    def test_calling_response_property_before_invoke(self, mock_calling):
        """Calling.response should return Unset before invoke."""
        # Before invoke(), execution.response should be Unset
        assert is_sentinel(mock_calling.execution.response)
        response = mock_calling.response
        assert response is Unset

    @pytest.mark.anyio
    async def test_calling_invoke_success(self, mock_calling):
        """Calling.invoke() should execute successfully."""
        await mock_calling.invoke()

        assert mock_calling.execution.status == EventStatus.COMPLETED
        assert mock_calling.execution.response.status == "success"
        assert mock_calling.execution.response.data == "test_result"
        assert mock_calling.execution.duration > 0

    @pytest.mark.anyio
    async def test_calling_invoke_failure(self, mock_calling):
        """Calling.invoke() should handle failures."""
        mock_calling.backend.should_fail = True
        await mock_calling.invoke()

        assert mock_calling.execution.status == EventStatus.FAILED
        assert str(mock_calling.execution.error) == "Test failure"

    @pytest.mark.anyio
    async def test_calling_invoke_cancellation(self, mock_calling):
        """Calling.invoke() should handle cancellation."""
        import asyncio

        mock_calling.backend.should_cancel = True

        with pytest.raises(asyncio.CancelledError):
            await mock_calling.invoke()

        assert mock_calling.execution.status == EventStatus.CANCELLED
        assert isinstance(mock_calling.execution.error, asyncio.CancelledError)

    @pytest.mark.anyio
    async def test_calling_response_property_after_invoke(self, mock_calling):
        """Calling.response should return NormalizedResponse after successful invoke."""
        await mock_calling.invoke()

        response = mock_calling.response
        assert response is not None
        assert response.status == "success"
        assert response.data == "test_result"


# =============================================================================
# ResourceBackend Tests
# =============================================================================


class TestResourceBackend:
    """Test ResourceBackend abstract class properties."""

    def test_provider_property(self):
        """Test provider property."""
        config = ResourceConfig(provider="test_provider", name="test_name")
        backend = MockResourceBackend(config=config)
        assert backend.provider == "test_provider"

    def test_name_property(self):
        """Test name property."""
        config = ResourceConfig(provider="test_provider", name="test_name")
        backend = MockResourceBackend(config=config)
        assert backend.name == "test_name"

    def test_version_property(self):
        """Test version property."""
        config = ResourceConfig(
            provider="test_provider", name="test_name", version="1.0.0"
        )
        backend = MockResourceBackend(config=config)
        assert backend.version == "1.0.0"

    def test_version_property_none(self):
        """Test version property when not set."""
        config = ResourceConfig(provider="test_provider", name="test_name")
        backend = MockResourceBackend(config=config)
        assert backend.version is None

    def test_tags_property(self):
        """Test tags property."""
        config = ResourceConfig(
            provider="test_provider", name="test_name", tags=["tag1", "tag2"]
        )
        backend = MockResourceBackend(config=config)
        assert backend.tags == {"tag1", "tag2"}

    def test_tags_property_empty(self):
        """Test tags property when empty."""
        config = ResourceConfig(provider="test_provider", name="test_name")
        backend = MockResourceBackend(config=config)
        assert backend.tags == set()

    def test_request_options_property(self):
        """Test request_options property."""
        config = ResourceConfig(
            provider="test_provider",
            name="test_name",
            request_options=SimpleRequestModel,
        )
        backend = MockResourceBackend(config=config)
        assert backend.request_options == SimpleRequestModel

    def test_request_options_property_none(self):
        """Test request_options property when None."""
        config = ResourceConfig(provider="test_provider", name="test_name")
        backend = MockResourceBackend(config=config)
        assert backend.request_options is None

    def test_normalize_response(self):
        """Test normalize_response default implementation."""
        config = ResourceConfig(provider="test_provider", name="test_name")
        backend = MockResourceBackend(config=config)

        raw_response = {"result": "test_data"}
        normalized = backend.normalize_response(raw_response)

        assert normalized.status == "success"
        assert normalized.data == raw_response
        assert normalized.raw_response == raw_response

    @pytest.mark.anyio
    async def test_stream_not_implemented(self):
        """Test that stream() raises NotImplementedError."""
        config = ResourceConfig(provider="test_provider", name="test_name")
        backend = MockResourceBackend(config=config)

        with pytest.raises(
            NotImplementedError, match="does not support streaming calls"
        ):
            await backend.stream()
