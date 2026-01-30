# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.services.endpoint - HTTP API endpoint backend."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel, SecretStr, ValidationError

from krons.errors import KronConnectionError
from krons.services import APICalling, Endpoint, EndpointConfig
from krons.services.utilities.resilience import CircuitBreaker, RetryConfig

# =============================================================================
# Test Request Models
# =============================================================================


class SimpleRequest(BaseModel):
    """Simple request model for testing."""

    message: str
    temperature: float = 0.7


# =============================================================================
# EndpointConfig Tests
# =============================================================================


class TestEndpointConfig:
    """Test EndpointConfig validation."""

    def test_config_endpoint_url(self):
        """EndpointConfig should construct full URL correctly."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="api/v1/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
        )
        assert config.full_url == "https://api.test.com/api/v1/test"

    def test_config_method_validation(self):
        """EndpointConfig should accept HTTP method."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            method="POST",
        )
        assert config.method == "POST"

        config2 = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            method="GET",
        )
        assert config2.method == "GET"

    def test_validate_kwargs_moves_extra_fields(self):
        """Test that extra fields are moved to kwargs dict."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            extra_field="extra_value",
            another_field=123,
        )
        assert config.kwargs["extra_field"] == "extra_value"
        assert config.kwargs["another_field"] == 123

    def test_validate_api_key_from_secret_str(self):
        """Test API key validation from SecretStr (via Endpoint)."""
        secret = SecretStr("secret_key_123")
        endpoint = Endpoint(
            config={
                "name": "test_endpoint",
                "provider": "test_provider",
                "endpoint": "/test",
                "api_key": secret,
                "request_options": SimpleRequest,
            }
        )
        assert endpoint.config.api_key is None  # Cleared (was raw credential via SecretStr)
        assert endpoint.config._api_key.get_secret_value() == "secret_key_123"

    def test_validate_api_key_from_string_literal(self):
        """Test API key validation from string (not env var)."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            api_key="literal_key_456",
        )
        # String that's not an env var -> treated as raw credential, cleared
        assert config.api_key is None
        assert config._api_key.get_secret_value() == "literal_key_456"

    def test_validate_api_key_from_env_var(self):
        """Test API key validation from environment variable."""
        os.environ["TEST_API_KEY"] = "env_key_789"
        try:
            config = EndpointConfig(
                name="test_endpoint",
                provider="test_provider",
                endpoint="/test",
                api_key="TEST_API_KEY",
            )
            # Successfully resolved from env -> keep env var name
            assert config.api_key == "TEST_API_KEY"
            assert config._api_key.get_secret_value() == "env_key_789"
        finally:
            del os.environ["TEST_API_KEY"]

    def test_validate_provider_empty_raises(self):
        """Test that empty provider raises ValidationError."""
        with pytest.raises(ValidationError, match="String should have at least 4 characters"):
            EndpointConfig(
                name="test_endpoint",
                provider="",
                endpoint="/test",
            )

    def test_full_url_with_endpoint_params(self):
        """Test full_url property with endpoint params."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="api/{version}/test",
            base_url="https://api.test.com",
            endpoint_params=["version"],
            params={"version": "v1"},
            request_options=SimpleRequest,
        )
        assert config.full_url == "https://api.test.com/api/v1/test"

        endpoint = Endpoint(config=config)
        assert endpoint.full_url == "https://api.test.com/api/v1/test"

    def test_full_url_without_endpoint_params(self):
        """Test full_url property without endpoint params."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
        )
        assert config.endpoint_params is None
        assert config.full_url == "https://api.test.com/test"

    def test_validate_api_key_empty_string(self):
        """Test that empty string api_key raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            EndpointConfig(
                name="test_endpoint",
                provider="test_provider",
                endpoint="/test",
                api_key="",
            )

    def test_validate_api_key_whitespace_only(self):
        """Test that whitespace-only api_key raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            EndpointConfig(
                name="test_endpoint",
                provider="test_provider",
                endpoint="/test",
                api_key="   ",
            )

    def test_validate_api_key_empty_env_var(self):
        """Test that empty env var value raises ValueError."""
        os.environ["EMPTY_VAR"] = ""
        try:
            with pytest.raises(ValueError, match="is empty or whitespace"):
                EndpointConfig(
                    name="test_endpoint",
                    provider="test_provider",
                    endpoint="/test",
                    api_key="EMPTY_VAR",
                )
        finally:
            del os.environ["EMPTY_VAR"]

    def test_system_env_var_blocked(self):
        """Test that system env vars are blocked."""
        with pytest.raises(ValueError, match="is a system environment variable"):
            EndpointConfig(
                name="test_endpoint",
                provider="test_provider",
                endpoint="/test",
                api_key="HOME",
            )

    def test_api_key_is_env_true_for_resolved_env_var(self):
        """Test api_key_is_env is True for resolved env var."""
        os.environ["TEST_KEY"] = "value123"
        try:
            config = EndpointConfig(
                name="test_endpoint",
                provider="test_provider",
                endpoint="/test",
                api_key="TEST_KEY",
            )
            assert config.api_key == "TEST_KEY"
            assert config.api_key_is_env is True
            assert config._api_key.get_secret_value() == "value123"
        finally:
            del os.environ["TEST_KEY"]

    def test_api_key_is_env_false_for_raw_credential(self):
        """Test api_key_is_env is False for raw credential."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            api_key="sk-raw",
        )
        assert config.api_key is None  # Cleared
        assert config.api_key_is_env is False

    def test_endpoint_params_invalid_params_raises(self):
        """Test that invalid params raise ValueError."""
        with pytest.raises(ValueError, match="Invalid params"):
            EndpointConfig(
                name="test_endpoint",
                provider="test_provider",
                endpoint="/api/{version}/{resource}",
                endpoint_params=["version", "resource"],
                params={
                    "version": "v1",
                    "invalid_key": "value",
                },  # invalid_key not in endpoint_params
            )


# =============================================================================
# Endpoint Tests
# =============================================================================


class TestEndpoint:
    """Test Endpoint backend."""

    def test_init_with_dict_config(self):
        """Test Endpoint initialization with dict config."""
        config_dict = {
            "name": "test_endpoint",
            "provider": "test_provider",
            "endpoint": "/test",
            "base_url": "https://api.test.com",
            "request_options": SimpleRequest,
        }
        endpoint = Endpoint(config=config_dict)
        assert endpoint.config.name == "test_endpoint"
        assert endpoint.config.provider == "test_provider"

    def test_init_with_endpoint_config(self):
        """Test Endpoint initialization with EndpointConfig instance."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        endpoint = Endpoint(config=config)
        assert endpoint.config.name == "test_endpoint"

    def test_init_with_invalid_config_type(self):
        """Test Endpoint initialization with invalid config type."""
        with pytest.raises(ValueError, match="Config must be a dict or EndpointConfig"):
            Endpoint(config="invalid")

    def test_create_payload(self):
        """Endpoint.create_payload() should build request payload."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)
        request = {"message": "test", "temperature": 0.8}

        payload = endpoint.create_payload(request)

        assert payload["message"] == "test"
        assert payload["temperature"] == 0.8

    def test_create_payload_with_kwargs(self):
        """Test create_payload merges kwargs."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)
        request = {"message": "test"}

        payload = endpoint.create_payload(request, temperature=0.9)

        assert payload["message"] == "test"
        assert payload["temperature"] == 0.9

    def test_create_payload_no_request_options_raises(self):
        """Test create_payload raises if no request_options."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            request_options=None,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        with pytest.raises(ValueError, match="must define request_options"):
            endpoint.create_payload({"data": "test"})

    def test_create_headers(self):
        """Endpoint.create_headers() should build headers dict."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            request_options=SimpleRequest,
            api_key="test_key",
            auth_type="bearer",
        )
        endpoint = Endpoint(config=config)

        headers = endpoint.create_headers()

        assert "Authorization" in headers
        assert "Content-Type" in headers

    def test_create_headers_with_extra_headers(self):
        """Test create_headers with extra headers."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            request_options=SimpleRequest,
            api_key="test_key",
            auth_type="bearer",
        )
        endpoint = Endpoint(config=config)
        extra_headers = {"X-Custom": "header"}

        headers = endpoint.create_headers(extra_headers=extra_headers)

        assert "X-Custom" in headers
        assert headers["X-Custom"] == "header"
        assert "Authorization" in headers

    @pytest.mark.anyio
    async def test_call(self):
        """Endpoint.call() should make HTTP request."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        # Mock _call_http
        async def mock_call_http(payload, headers, **kwargs):
            return {"result": "success"}

        endpoint._call_http = mock_call_http

        result = await endpoint.call(request={"message": "test", "temperature": 0.7})

        assert result.status == "success"
        assert result.data == {"result": "success"}

    @pytest.mark.anyio
    async def test_call_skip_payload_creation_dict(self):
        """Test call with skip_payload_creation and dict request."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        async def mock_call(payload, headers, **kwargs):
            return {"result": "success"}

        endpoint._call = mock_call

        result = await endpoint.call(
            request={"message": "test", "temperature": 0.5},
            skip_payload_creation=True,
        )

        assert result.status == "success"

    @pytest.mark.anyio
    async def test_call_with_retry_only(self):
        """Test call with retry_config only (no circuit breaker)."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        retry_config = RetryConfig(max_retries=2, initial_delay=0.01)
        endpoint = Endpoint(config=config, retry_config=retry_config)

        call_count = 0

        async def mock_call(payload, headers, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise KronConnectionError("Retry me")
            return {"result": "success"}

        endpoint._call = mock_call

        result = await endpoint.call(request={"message": "test", "temperature": 0.7})

        assert result.status == "success"
        assert call_count == 2

    @pytest.mark.anyio
    async def test_call_with_circuit_breaker_only(self):
        """Test call with circuit_breaker only (no retry)."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_time=1.0)
        endpoint = Endpoint(config=config, circuit_breaker=circuit_breaker)

        async def mock_call(payload, headers, **kwargs):
            return {"result": "success"}

        endpoint._call = mock_call

        result = await endpoint.call(request={"message": "test", "temperature": 0.7})

        assert result.status == "success"

    @pytest.mark.anyio
    async def test_call_http_success(self):
        """Test _call_http with successful response."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        # Mock httpx.AsyncClient
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(endpoint, "_create_http_client", return_value=mock_client):
            result = await endpoint._call_http(
                payload={"message": "test", "temperature": 0.7},
                headers={"Authorization": "Bearer test"},
            )

        assert result == {"result": "success"}

    @pytest.mark.anyio
    async def test_call_http_429_raises(self):
        """Test _call_http with 429 rate limit raises."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Rate limited", request=MagicMock(), response=mock_response
        )

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(endpoint, "_create_http_client", return_value=mock_client),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await endpoint._call_http(
                payload={"message": "test", "temperature": 0.7},
                headers={"Authorization": "Bearer test"},
            )

    @pytest.mark.anyio
    async def test_call_http_500_raises(self):
        """Test _call_http with 500 server error raises."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        )

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(endpoint, "_create_http_client", return_value=mock_client),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await endpoint._call_http(
                payload={"message": "test", "temperature": 0.7},
                headers={"Authorization": "Bearer test"},
            )

    def test_to_dict_with_retry_config(self):
        """Test to_dict with retry_config."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        retry_config = RetryConfig(max_retries=3, initial_delay=0.1)
        endpoint = Endpoint(config=config, retry_config=retry_config)

        result = endpoint.to_dict()

        assert result["retry_config"] is not None
        assert result["retry_config"]["max_retries"] == 3

    def test_to_dict_with_circuit_breaker(self):
        """Test to_dict with circuit_breaker."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_time=2.0)
        endpoint = Endpoint(config=config, circuit_breaker=circuit_breaker)

        result = endpoint.to_dict()

        assert result["circuit_breaker"] is not None
        assert result["circuit_breaker"]["failure_threshold"] == 5

    def test_event_type_property(self):
        """Test event_type property returns APICalling."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="/test",
            request_options=SimpleRequest,
        )
        endpoint = Endpoint(config=config)

        assert endpoint.event_type == APICalling

    def test_create_http_client(self):
        """Test _create_http_client creates httpx.AsyncClient."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            timeout=120,
            client_kwargs={"follow_redirects": True},
        )
        endpoint = Endpoint(config=config)

        client = endpoint._create_http_client()

        # Verify it's an httpx.AsyncClient
        assert client.__class__.__name__ == "AsyncClient"
        # Timeout should be configured
        assert client.timeout.read == 120

    def test_init_with_empty_secretstr_raises(self):
        """Test Endpoint init with empty SecretStr raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            Endpoint(
                config={
                    "name": "test_endpoint",
                    "provider": "test_provider",
                    "endpoint": "/test",
                    "request_options": SimpleRequest,
                    "api_key": SecretStr(""),
                }
            )

    def test_deserialize_circuit_breaker_invalid_type_raises(self):
        """Test circuit_breaker validator with invalid type."""
        with pytest.raises(ValidationError, match="circuit_breaker must be a dict"):
            Endpoint(
                config={
                    "name": "test_endpoint",
                    "provider": "test_provider",
                    "endpoint": "/test",
                    "request_options": SimpleRequest,
                },
                circuit_breaker="invalid_string",
            )

    def test_deserialize_retry_config_invalid_type_raises(self):
        """Test retry_config validator with invalid type."""
        with pytest.raises(ValidationError, match="retry_config must be a dict"):
            Endpoint(
                config={
                    "name": "test_endpoint",
                    "provider": "test_provider",
                    "endpoint": "/test",
                    "request_options": SimpleRequest,
                },
                retry_config="invalid_string",
            )


# =============================================================================
# APICalling Tests
# =============================================================================


class TestAPICalling:
    """Test APICalling event."""

    def test_api_calling_call_args(self):
        """APICalling.call_args should return endpoint-specific args."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        calling = APICalling(
            backend=endpoint,
            payload={"message": "test"},
            extra_headers={"X-Custom": "value"},
        )

        call_args = calling.call_args
        assert call_args["request"] == {"message": "test"}
        assert call_args["extra_headers"] == {"X-Custom": "value"}
        assert call_args["skip_payload_creation"] is True

    def test_required_tokens_with_messages(self):
        """Test required_tokens calculates tokens from messages."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
            requires_tokens=True,
        )
        endpoint = Endpoint(config=config)

        calling = APICalling(
            backend=endpoint,
            payload={
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thank you!"},
                ]
            },
        )

        tokens = calling.required_tokens
        assert tokens is not None
        assert tokens > 0

    def test_required_tokens_with_input_string(self):
        """Test required_tokens calculates tokens from string input."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
            requires_tokens=True,
        )
        endpoint = Endpoint(config=config)

        calling = APICalling(
            backend=endpoint,
            payload={"input": "This is a test string for embeddings"},
        )

        tokens = calling.required_tokens
        assert tokens is not None
        assert tokens > 0

    def test_required_tokens_with_input_list(self):
        """Test required_tokens calculates tokens from list input."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
            requires_tokens=True,
        )
        endpoint = Endpoint(config=config)

        calling = APICalling(
            backend=endpoint,
            payload={"input": ["First text", "Second text", "Third text"]},
        )

        tokens = calling.required_tokens
        assert tokens is not None
        assert tokens > 0

    def test_required_tokens_returns_none_for_unknown_payload(self):
        """Test required_tokens returns None for unknown payload format."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
            requires_tokens=True,
        )
        endpoint = Endpoint(config=config)

        calling = APICalling(
            backend=endpoint,
            payload={"custom_field": "unknown format"},
        )

        tokens = calling.required_tokens
        assert tokens is None

    def test_required_tokens_returns_none_when_not_required(self):
        """Test required_tokens returns None when requires_tokens=False."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
            requires_tokens=False,
        )
        endpoint = Endpoint(config=config)

        calling = APICalling(
            backend=endpoint,
            payload={"messages": [{"role": "user", "content": "test"}]},
        )

        tokens = calling.required_tokens
        assert tokens is None

    def test_request_property(self):
        """Test request property returns required_tokens."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
            requires_tokens=True,
        )
        endpoint = Endpoint(config=config)

        calling = APICalling(
            backend=endpoint,
            payload={"messages": [{"role": "user", "content": "test"}]},
        )

        request = calling.request
        assert "required_tokens" in request
        assert request["required_tokens"] is not None

    @pytest.mark.anyio
    async def test_invoke(self):
        """Test _invoke method calls backend.call."""
        config = EndpointConfig(
            name="test_endpoint",
            provider="test_provider",
            endpoint="test",
            base_url="https://api.test.com",
            request_options=SimpleRequest,
            api_key="test_key",
        )
        endpoint = Endpoint(config=config)

        async def mock_call_http(payload, headers, **kwargs):
            return {"result": "success"}

        endpoint._call_http = mock_call_http

        calling = APICalling(
            backend=endpoint,
            payload={"message": "test", "temperature": 0.7},
            extra_headers={"X-Custom": "header"},
        )

        result = await calling._invoke()

        assert result.status == "success"
        assert result.data == {"result": "success"}
