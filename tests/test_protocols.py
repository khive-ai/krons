# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.protocols - runtime-checkable protocols and @implements decorator."""

import warnings

import pytest

from krons.protocols import (
    Allowable,
    Containable,
    Deserializable,
    Hashable,
    Invocable,
    Observable,
    Serializable,
    SignatureMismatchError,
    implements,
)

# =============================================================================
# Test Protocol isinstance checks
# =============================================================================


class TestProtocolIsinstance:
    """Test runtime protocol checking with isinstance."""

    def test_serializable_isinstance(self):
        """Test Serializable protocol isinstance check."""

        class GoodSerializer:
            def to_dict(self, **kwargs):
                return {}

        class BadSerializer:
            pass

        assert isinstance(GoodSerializer(), Serializable)
        assert not isinstance(BadSerializer(), Serializable)

    def test_deserializable_isinstance(self):
        """Test Deserializable protocol isinstance check."""

        class GoodDeserializer:
            @classmethod
            def from_dict(cls, data, **kwargs):
                return cls()

        assert isinstance(GoodDeserializer(), Deserializable)

    def test_hashable_isinstance(self):
        """Test Hashable protocol isinstance check."""

        class GoodHashable:
            def __hash__(self):
                return 42

        assert isinstance(GoodHashable(), Hashable)

    def test_allowable_isinstance(self):
        """Test Allowable protocol isinstance check."""

        class GoodAllowable:
            def allowed(self):
                return {"a", "b"}

        assert isinstance(GoodAllowable(), Allowable)

    def test_containable_isinstance(self):
        """Test Containable protocol isinstance check."""

        class GoodContainable:
            def __contains__(self, item):
                return True

        assert isinstance(GoodContainable(), Containable)


# =============================================================================
# Test @implements decorator
# =============================================================================


class TestImplementsDecorator:
    """Test @implements decorator validation."""

    def test_implements_validates_members(self):
        """Test that @implements validates required members exist."""

        @implements(Serializable, signature_check="skip")
        class Good:
            def to_dict(self, **kwargs):
                return {}

        assert hasattr(Good, "__protocols__")
        assert Serializable in Good.__protocols__

    def test_implements_raises_on_missing_member(self):
        """Test that @implements raises TypeError for missing members."""
        with pytest.raises(TypeError, match="does not define"):

            @implements(Serializable, signature_check="skip")
            class Bad:
                pass

    def test_implements_with_allow_inherited(self):
        """Test @implements with allow_inherited=True."""

        class Base:
            def to_dict(self, **kwargs):
                return {}

        @implements(Serializable, signature_check="skip", allow_inherited=True)
        class Child(Base):
            pass

        assert Serializable in Child.__protocols__

    def test_implements_raises_without_allow_inherited(self):
        """Test @implements raises when member is inherited but allow_inherited=False."""

        class Base:
            def to_dict(self, **kwargs):
                return {}

        with pytest.raises(TypeError, match="does not define"):

            @implements(Serializable, signature_check="skip", allow_inherited=False)
            class Child(Base):
                pass

    def test_implements_signature_check_warn(self):
        """Test signature_check='warn' emits warning on mismatch."""

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(Serializable, signature_check="warn")
            class Impl:
                def to_dict(self):  # Missing **kwargs
                    return {}

            # Should have warned about missing kwargs
            assert len(w) >= 1
            assert "kwargs" in str(w[0].message).lower() or "signature" in str(w[0].message).lower()

    def test_implements_signature_check_error(self):
        """Test signature_check='error' raises on mismatch."""
        with pytest.raises(SignatureMismatchError):

            @implements(Serializable, signature_check="error")
            class Impl:
                def to_dict(self):  # Missing **kwargs
                    return {}

    def test_implements_signature_check_skip(self):
        """Test signature_check='skip' doesn't check signatures."""

        # Should not raise even with wrong signature
        @implements(Serializable, signature_check="skip")
        class Impl:
            def to_dict(self):  # Missing **kwargs but we skip check
                return {}

        assert Serializable in Impl.__protocols__

    def test_implements_multiple_protocols(self):
        """Test implementing multiple protocols."""

        @implements(Serializable, Hashable, signature_check="skip")
        class Multi:
            def to_dict(self, **kwargs):
                return {}

            def __hash__(self):
                return 42

        assert Serializable in Multi.__protocols__
        assert Hashable in Multi.__protocols__


# =============================================================================
# Test signature compatibility
# =============================================================================


class TestSignatureCompatibility:
    """Test signature checking logic."""

    def test_compatible_with_extra_optional_params(self):
        """Implementation can have extra optional params."""

        @implements(Serializable, signature_check="error")
        class Impl:
            def to_dict(self, extra_param=None, **kwargs):
                return {}

        assert Serializable in Impl.__protocols__

    def test_compatible_var_keyword_satisfies_kwargs(self):
        """**kwargs in impl satisfies protocol's **kwargs."""

        @implements(Serializable, signature_check="error")
        class Impl:
            def to_dict(self, **kw):  # Different name but still **kwargs
                return {}

        assert Serializable in Impl.__protocols__


# =============================================================================
# Test _get_signature_params edge cases
# =============================================================================


class TestGetSignatureParams:
    """Test _get_signature_params helper function."""

    def test_property_returns_none(self):
        """Property methods should return None (not introspectable)."""
        from krons.protocols import _get_signature_params

        class MyClass:
            @property
            def value(self):
                return 42

        # Get the property descriptor
        prop = MyClass.__dict__["value"]
        assert _get_signature_params(prop) is None

    def test_non_callable_returns_none(self):
        """Non-callable objects should return None."""
        from krons.protocols import _get_signature_params

        assert _get_signature_params("not a function") is None
        assert _get_signature_params(123) is None
        assert _get_signature_params(None) is None

    def test_classmethod_unpacking(self):
        """Classmethod functions should be unwrapped properly."""
        from krons.protocols import _get_signature_params

        class MyClass:
            @classmethod
            def create(cls, value: int):
                return cls()

        cm = MyClass.__dict__["create"]
        params = _get_signature_params(cm)
        assert params is not None
        assert "value" in params

    def test_staticmethod_unpacking(self):
        """Staticmethod functions should be unwrapped properly."""
        from krons.protocols import _get_signature_params

        class MyClass:
            @staticmethod
            def helper(x: int, y: int):
                return x + y

        sm = MyClass.__dict__["helper"]
        params = _get_signature_params(sm)
        assert params is not None
        assert "x" in params
        assert "y" in params

    def test_builtin_value_error(self):
        """Builtins that can't be introspected should return None."""
        from krons.protocols import _get_signature_params

        # Some builtins may raise ValueError/TypeError for signature
        result = _get_signature_params(len)
        # len may or may not be introspectable depending on Python version
        # Just verify it doesn't raise an exception
        assert result is None or isinstance(result, dict)


# =============================================================================
# Test signature checking edge cases
# =============================================================================


class TestSignatureCheckingEdgeCases:
    """Test edge cases in signature compatibility checking."""

    def test_protocol_with_args_impl_without(self):
        """Protocol with *args but impl doesn't have it warns."""
        from typing import Protocol, runtime_checkable

        @runtime_checkable
        class ArgsProtocol(Protocol):
            def process(self, *args): ...

        # Implementation without *args should cause warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(ArgsProtocol, signature_check="warn")
            class Impl:
                def process(self, x, y):  # No *args
                    pass

            # Should have warned about missing args
            assert any("args" in str(warning.message).lower() for warning in w)

    def test_protocol_optional_impl_required_warns(self):
        """Impl making optional param required warns."""
        from typing import Protocol, runtime_checkable

        @runtime_checkable
        class OptionalProtocol(Protocol):
            def greet(self, name: str = "World"): ...

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(OptionalProtocol, signature_check="warn")
            class Impl:
                def greet(self, name: str):  # Made required
                    pass

            # Should have warned about tightening contract
            assert len(w) >= 1

    def test_impl_keyword_only_protocol_positional_warns(self):
        """Impl with keyword-only when protocol allows positional warns."""
        from typing import Protocol, runtime_checkable

        @runtime_checkable
        class PositionalProtocol(Protocol):
            def compute(self, x, y):  # Positional allowed
                ...

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(PositionalProtocol, signature_check="warn")
            class Impl:
                def compute(self, *, x, y):  # Keyword-only
                    pass

            # Should warn about requiring keyword-only
            assert len(w) >= 1

    def test_impl_extra_required_param_warns(self):
        """Impl with extra required param that protocol doesn't provide warns."""
        from typing import Protocol, runtime_checkable

        @runtime_checkable
        class SimpleProtocol(Protocol):
            def action(self, x): ...

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(SimpleProtocol, signature_check="warn")
            class Impl:
                def action(self, x, extra_required):  # Extra required param
                    pass

            # Should warn about extra required param
            assert len(w) >= 1

    def test_missing_param_with_var_keyword_ok(self):
        """Missing param is OK if impl has **kwargs."""
        from typing import Protocol, runtime_checkable

        @runtime_checkable
        class ParamProtocol(Protocol):
            def update(self, value: int): ...

        # Should not raise because **kwargs can accept 'value'
        @implements(ParamProtocol, signature_check="error")
        class Impl:
            def update(self, **kwargs):  # **kwargs accepts anything
                pass

        assert ParamProtocol in Impl.__protocols__


# =============================================================================
# Test Observable alias
# =============================================================================


class TestObservable:
    """Test Observable protocol."""

    def test_observable_requires_id(self):
        """Observable requires id property."""
        from uuid import uuid4

        class Good:
            @property
            def id(self):
                return uuid4()

        assert isinstance(Good(), Observable)
