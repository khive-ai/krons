# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.services.imodel - iModel unified service interface."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from krons.core import EventStatus, Executor
from krons.services import APICalling, Endpoint, EndpointConfig, NormalizedResponse
from krons.services.hook import HookRegistry
from krons.services.imodel import iModel
from krons.services.utilities.rate_limiter import RateLimitConfig, TokenBucket

# =============================================================================
# Test Request Models
# =============================================================================


class ChatRequest(BaseModel):
    """Chat request model for testing."""

    model: str = "gpt-4"
    messages: list[dict[str, str]]
    temperature: float = 0.7


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_endpoint():
    """Create mock endpoint with mocked HTTP calls."""
    config = EndpointConfig(
        name="test_model",
        provider="test_provider",
        endpoint="chat/completions",
        base_url="https://api.test.com",
        request_options=ChatRequest,
        api_key="test_key",
    )
    endpoint = Endpoint(config=config)

    # Mock _call_http to return success response
    async def mock_call_http(payload, headers, **kwargs):
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{"message": {"content": "Hello!"}}],
        }

    endpoint._call_http = mock_call_http
    return endpoint


@pytest.fixture
def mock_imodel(mock_endpoint):
    """Create iModel with mock endpoint."""
    return iModel(backend=mock_endpoint)


# =============================================================================
# iModel Basic Tests
# =============================================================================


class TestIModelCreation:
    """Test iModel instantiation."""

    def test_imodel_requires_backend(self):
        """iModel should require a backend."""
        with pytest.raises(TypeError):
            iModel()

    @pytest.mark.anyio
    async def test_imodel_with_rate_limiter(self, mock_endpoint):
        """iModel should accept optional rate limiter."""
        rate_limiter = TokenBucket(RateLimitConfig(capacity=10, refill_rate=1.0))
        model = iModel(backend=mock_endpoint, rate_limiter=rate_limiter)
        assert model.rate_limiter is not None
        assert model.rate_limiter.capacity == 10

    def test_imodel_name_from_backend(self, mock_imodel):
        """iModel.name should come from backend."""
        assert mock_imodel.name == "test_model"

    def test_imodel_version_from_backend(self, mock_imodel):
        """iModel.version should come from backend."""
        assert mock_imodel.version == ""  # Default when not set

    def test_imodel_tags_from_backend(self, mock_endpoint):
        """iModel.tags should come from backend."""
        mock_endpoint.config.tags.append("api")
        model = iModel(backend=mock_endpoint)
        assert "api" in model.tags

    def test_imodel_repr(self, mock_imodel):
        """Test iModel repr."""
        repr_str = repr(mock_imodel)
        assert "iModel" in repr_str
        assert "test_model" in repr_str

    def test_imodel_repr_no_backend(self):
        """Test iModel repr when backend is None (deserialization edge case)."""
        # Create model and forcefully set backend to None for edge case testing
        model = iModel.__new__(iModel)
        object.__setattr__(model, "backend", None)
        repr_str = repr(model)
        assert "backend=None" in repr_str


# =============================================================================
# iModel Property Tests
# =============================================================================


class TestIModelProperties:
    """Test iModel properties that delegate to backend."""

    def test_name_raises_if_no_backend(self):
        """name property should raise if backend not configured."""
        model = iModel.__new__(iModel)
        object.__setattr__(model, "backend", None)
        with pytest.raises(RuntimeError, match="Backend not configured"):
            _ = model.name

    def test_version_raises_if_no_backend(self):
        """version property should raise if backend not configured."""
        model = iModel.__new__(iModel)
        object.__setattr__(model, "backend", None)
        with pytest.raises(RuntimeError, match="Backend not configured"):
            _ = model.version

    def test_tags_raises_if_no_backend(self):
        """tags property should raise if backend not configured."""
        model = iModel.__new__(iModel)
        object.__setattr__(model, "backend", None)
        with pytest.raises(RuntimeError, match="Backend not configured"):
            _ = model.tags


# =============================================================================
# iModel Initialization Tests
# =============================================================================


class TestIModelInit:
    """Test iModel initialization options."""

    @pytest.mark.anyio
    async def test_init_with_rate_limiter(self, mock_endpoint):
        """Test iModel init with rate_limiter."""
        rate_limiter = TokenBucket(RateLimitConfig(capacity=10, refill_rate=1.0))
        model = iModel(backend=mock_endpoint, rate_limiter=rate_limiter)
        assert model.rate_limiter is not None
        assert model.rate_limiter.capacity == 10

    @pytest.mark.anyio
    async def test_init_with_limit_requests_creates_executor(self, mock_endpoint):
        """Test iModel init with limit_requests auto-creates executor."""
        model = iModel(
            backend=mock_endpoint,
            limit_requests=50,
            capacity_refresh_time=60,
        )
        assert model.executor is not None

    @pytest.mark.anyio
    async def test_init_with_explicit_executor(self, mock_endpoint):
        """Test iModel init with explicit executor."""
        executor = MagicMock(spec=Executor)
        model = iModel(backend=mock_endpoint, executor=executor)
        assert model.executor is executor


# =============================================================================
# iModel Invoke Tests
# =============================================================================


class TestIModelInvoke:
    """Test iModel.invoke() method."""

    @pytest.mark.anyio
    async def test_invoke_creates_calling(self, mock_imodel):
        """iModel.invoke() should create Calling event."""
        calling = await mock_imodel.invoke(
            timeout=60.0,
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert isinstance(calling, APICalling)
        assert calling.execution.status == EventStatus.COMPLETED
        assert calling.response.status == "success"

    @pytest.mark.anyio
    async def test_invoke_with_hooks(self, mock_endpoint):
        """iModel.invoke() should support hooks."""
        # This test verifies hooks can be attached (not invoking complex hook logic)
        registry = HookRegistry()
        model = iModel(backend=mock_endpoint, hook_registry=registry)

        calling = await model.invoke(
            timeout=60.0,
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert calling.execution.status == EventStatus.COMPLETED

    @pytest.mark.anyio
    async def test_invoke_direct_success(self, mock_imodel):
        """Test invoke without executor (direct path)."""
        calling = await mock_imodel.invoke(
            timeout=60.0,
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert calling.execution.status == EventStatus.COMPLETED
        assert calling.response.status == "success"

    @pytest.mark.anyio
    async def test_invoke_with_pre_created_calling(self, mock_imodel):
        """Test invoke with pre-created calling instance."""
        calling = await mock_imodel.create_calling(
            timeout=120.0,
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Invoke with pre-created calling - arguments should be ignored
        result = await mock_imodel.invoke(calling=calling)

        assert result is calling
        assert result.execution.status == EventStatus.COMPLETED


# =============================================================================
# iModel Rate Limiting Tests
# =============================================================================


class TestIModelRateLimiting:
    """Test iModel rate limiting."""

    @pytest.mark.anyio
    async def test_rate_limited_invoke(self, mock_endpoint):
        """iModel should respect rate limits."""
        rate_limiter = TokenBucket(
            RateLimitConfig(capacity=10, refill_rate=10.0, initial_tokens=10)
        )
        model = iModel(backend=mock_endpoint, rate_limiter=rate_limiter)

        calling = await model.invoke(
            timeout=60.0,
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert calling.execution.status == EventStatus.COMPLETED

    @pytest.mark.anyio
    async def test_invoke_rate_limit_timeout(self, mock_endpoint):
        """Test invoke raises TimeoutError when rate limit cannot be acquired."""
        # Create rate limiter with 0 initial tokens and slow refill
        rate_limiter = TokenBucket(RateLimitConfig(capacity=10, refill_rate=0.01, initial_tokens=0))
        model = iModel(backend=mock_endpoint, rate_limiter=rate_limiter)

        # Mock acquire to return False (timeout)
        async def mock_acquire(timeout=None):
            return False

        rate_limiter.acquire = mock_acquire

        with pytest.raises(TimeoutError, match="Rate limit acquisition"):
            await model.invoke(
                timeout=60.0,
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )


# =============================================================================
# iModel create_calling Tests
# =============================================================================


class TestIModelCreateCalling:
    """Test iModel.create_calling() method."""

    @pytest.mark.anyio
    async def test_create_calling_basic(self, mock_imodel):
        """Test create_calling creates APICalling instance."""
        calling = await mock_imodel.create_calling(
            timeout=60.0,
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert isinstance(calling, APICalling)
        assert calling.backend is mock_imodel.backend
        assert "messages" in calling.payload

    @pytest.mark.anyio
    async def test_create_calling_with_timeout(self, mock_imodel):
        """Test create_calling with timeout parameter."""
        calling = await mock_imodel.create_calling(
            timeout=60.0,
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert calling.timeout == 60.0

    @pytest.mark.anyio
    async def test_create_calling_with_streaming(self, mock_imodel):
        """Test create_calling with streaming parameter."""
        calling = await mock_imodel.create_calling(
            timeout=60.0,
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            streaming=True,
        )

        assert calling.streaming is True

    @pytest.mark.anyio
    async def test_create_calling_raises_if_no_backend(self):
        """Test create_calling raises if backend not configured."""
        model = iModel.__new__(iModel)
        object.__setattr__(model, "backend", None)
        object.__setattr__(model, "hook_registry", None)
        with pytest.raises(RuntimeError, match="Backend not configured"):
            await model.create_calling(timeout=60.0, model="gpt-4", messages=[])


# =============================================================================
# iModel Serialization Tests
# =============================================================================


class TestIModelSerialization:
    """Test iModel serialization/deserialization."""

    def test_to_dict(self, mock_imodel):
        """Test iModel.to_dict() excludes id/created_at."""
        result = mock_imodel.to_dict()

        # Should exclude identity fields
        assert "id" not in result
        assert "created_at" not in result

        # Should include backend
        assert "backend" in result
        assert result["backend"]["config"]["name"] == "test_model"

    @pytest.mark.anyio
    async def test_serialize_rate_limiter(self, mock_endpoint):
        """Test rate_limiter serialization."""
        rate_limiter = TokenBucket(RateLimitConfig(capacity=10, refill_rate=2.5))
        model = iModel(backend=mock_endpoint, rate_limiter=rate_limiter)

        result = model.to_dict()

        assert result["rate_limiter"] is not None
        assert result["rate_limiter"]["capacity"] == 10
        assert result["rate_limiter"]["refill_rate"] == 2.5

    def test_serialize_rate_limiter_none(self, mock_imodel):
        """Test rate_limiter serialization when None."""
        result = mock_imodel.to_dict()
        assert result["rate_limiter"] is None

    @pytest.mark.anyio
    async def test_deserialize_rate_limiter_from_dict(self, mock_endpoint):
        """Test rate_limiter deserialization from dict."""
        # Serialize
        rate_limiter = TokenBucket(RateLimitConfig(capacity=10, refill_rate=2.5))
        model = iModel(backend=mock_endpoint, rate_limiter=rate_limiter)
        data = model.to_dict()

        # Test the validator directly
        result = iModel._deserialize_rate_limiter(data["rate_limiter"])
        assert result is not None
        assert result.capacity == 10

    @pytest.mark.anyio
    async def test_deserialize_rate_limiter_already_token_bucket(self, mock_endpoint):
        """Test rate_limiter validator with existing TokenBucket."""
        rate_limiter = TokenBucket(RateLimitConfig(capacity=5, refill_rate=1.0))
        result = iModel._deserialize_rate_limiter(rate_limiter)
        assert result is rate_limiter

    def test_deserialize_rate_limiter_invalid_type_raises(self):
        """Test rate_limiter validator with invalid type."""
        with pytest.raises(ValueError, match="must be a dict or TokenBucket"):
            iModel._deserialize_rate_limiter("invalid")

    def test_deserialize_backend_none_raises(self):
        """Test backend validator with None raises."""
        with pytest.raises(ValueError, match="backend is required"):
            iModel._deserialize_backend(None)

    def test_deserialize_backend_invalid_type_raises(self):
        """Test backend validator with invalid type."""
        with pytest.raises(ValueError, match="must be a dict or ServiceBackend"):
            iModel._deserialize_backend("invalid")

    def test_deserialize_executor_already_executor(self, mock_endpoint):
        """Test executor validator with existing Executor."""
        executor = MagicMock(spec=Executor)
        result = iModel._deserialize_executor(executor)
        assert result is executor

    def test_deserialize_executor_invalid_type_raises(self):
        """Test executor validator with invalid type."""
        with pytest.raises(ValueError, match="must be a dict or Executor"):
            iModel._deserialize_executor("invalid")


# =============================================================================
# iModel Context Manager Tests
# =============================================================================


class TestIModelContextManager:
    """Test iModel async context manager."""

    @pytest.mark.anyio
    async def test_context_manager_no_executor(self, mock_imodel):
        """Test context manager when no executor configured."""
        async with mock_imodel as model:
            assert model is mock_imodel

    @pytest.mark.anyio
    async def test_context_manager_with_executor(self, mock_endpoint):
        """Test context manager starts/stops executor."""
        model = iModel(
            backend=mock_endpoint,
            limit_requests=50,
        )

        # Executor should be created but not started
        assert model.executor is not None

        async with model as m:
            assert m is model
            # Executor should be running inside context

    @pytest.mark.anyio
    async def test_context_manager_exception_propagates(self, mock_imodel):
        """Test that exceptions propagate (not suppressed)."""
        with pytest.raises(ValueError, match="test error"):
            async with mock_imodel:
                raise ValueError("test error")


# =============================================================================
# iModel Claude Code Session Tests
# =============================================================================


class TestIModelClaudeCodeSession:
    """Test Claude Code session ID storage."""

    def test_store_claude_code_session_id(self):
        """Test _store_claude_code_session_id stores session from response."""
        config = EndpointConfig(
            name="test_model",
            provider="claude_code",
            endpoint="chat",
            base_url="https://api.test.com",
            request_options=ChatRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)
        model = iModel(backend=endpoint)

        # Create mock calling with response
        calling = MagicMock()
        calling.execution.response = NormalizedResponse(
            status="success",
            data={"content": "Hello"},
            raw_response={"content": "Hello"},
            metadata={"session_id": "session_123"},
        )

        model._store_claude_code_session_id(calling)

        assert model.provider_metadata.get("session_id") == "session_123"

    def test_store_claude_code_session_id_no_metadata(self):
        """Test _store_claude_code_session_id with no metadata."""
        config = EndpointConfig(
            name="test_model",
            provider="claude_code",
            endpoint="chat",
            base_url="https://api.test.com",
            request_options=ChatRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)
        model = iModel(backend=endpoint)

        # Create mock calling with response without metadata
        calling = MagicMock()
        calling.execution.response = NormalizedResponse(
            status="success",
            data={"content": "Hello"},
            raw_response={"content": "Hello"},
            metadata=None,
        )

        model._store_claude_code_session_id(calling)

        assert "session_id" not in model.provider_metadata

    def test_store_claude_code_session_id_wrong_provider(self, mock_imodel):
        """Test _store_claude_code_session_id ignores non-claude_code provider."""
        # mock_imodel has provider="test_provider"
        calling = MagicMock()
        calling.execution.response = NormalizedResponse(
            status="success",
            data={"content": "Hello"},
            raw_response={"content": "Hello"},
            metadata={"session_id": "session_123"},
        )

        mock_imodel._store_claude_code_session_id(calling)

        # Should not store because provider is not "claude_code"
        assert "session_id" not in mock_imodel.provider_metadata
