# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.types._sentinel - Sentinel values.

Sentinel Semantics:
    - Undefined: Field never set (missing from namespace)
    - Unset: Key present but value not provided
    - None: Explicitly set to null (valid value, not sentinel)

These tests validate singleton behavior, sentinel checks, and pickle roundtrip
for robustness across multiprocessing and serialization scenarios.
"""

import pickle

import pytest

from krons.types import (
    Undefined,
    UndefinedType,
    Unset,
    UnsetType,
    is_sentinel,
    is_undefined,
    is_unset,
    not_sentinel,
)


class TestSentinelSingletons:
    """Test sentinel singleton behavior."""

    def test_undefined_is_singleton(self):
        """Undefined should be a singleton - same instance on every access."""
        u1 = UndefinedType()
        u2 = UndefinedType()
        assert u1 is u2
        assert u1 is Undefined

    def test_unset_is_singleton(self):
        """Unset should be a singleton - same instance on every access."""
        u1 = UnsetType()
        u2 = UnsetType()
        assert u1 is u2
        assert u1 is Unset

    def test_sentinels_are_falsy(self):
        """Undefined and Unset should be falsy for conditional checks."""
        assert not Undefined
        assert not Unset
        assert bool(Undefined) is False
        assert bool(Unset) is False

    def test_undefined_repr(self):
        """Undefined should have clear repr for debugging."""
        assert repr(Undefined) == "Undefined"
        assert str(Undefined) == "Undefined"

    def test_unset_repr(self):
        """Unset should have clear repr for debugging."""
        assert repr(Unset) == "Unset"
        assert str(Unset) == "Unset"


class TestSentinelChecks:
    """Test sentinel check functions."""

    def test_is_sentinel_undefined(self):
        """is_sentinel(Undefined) should be True."""
        assert is_sentinel(Undefined) is True

    def test_is_sentinel_unset(self):
        """is_sentinel(Unset) should be True."""
        assert is_sentinel(Unset) is True

    def test_is_sentinel_none(self):
        """is_sentinel(None) should be False by default."""
        assert is_sentinel(None) is False

    def test_is_sentinel_none_as_sentinel(self):
        """is_sentinel(None, {"none"}) should be True."""
        assert is_sentinel(None, {"none"}) is True

    def test_is_sentinel_empty_as_sentinel(self):
        """is_sentinel with {"empty"} treats empty collections as sentinel."""
        # Empty collections treated as sentinels
        assert is_sentinel([], {"empty"}) is True
        assert is_sentinel({}, {"empty"}) is True
        assert is_sentinel(set(), {"empty"}) is True
        assert is_sentinel("", {"empty"}) is True
        assert is_sentinel(tuple(), {"empty"}) is True

        # Non-empty collections not sentinels
        assert is_sentinel([1, 2], {"empty"}) is False
        assert is_sentinel({"a": 1}, {"empty"}) is False

    def test_not_sentinel(self):
        """not_sentinel() should negate is_sentinel()."""
        assert not_sentinel(Undefined) is False
        assert not_sentinel(Unset) is False
        assert not_sentinel(None) is True
        assert not_sentinel("value") is True
        assert not_sentinel(42) is True
        assert not_sentinel([1, 2, 3]) is True

    def test_is_undefined(self):
        """is_undefined() should check specifically for Undefined sentinel."""
        assert is_undefined(Undefined) is True
        assert is_undefined(Unset) is False
        assert is_undefined(None) is False
        assert is_undefined("value") is False

    def test_is_unset(self):
        """is_unset() should check specifically for Unset sentinel."""
        assert is_unset(Unset) is True
        assert is_unset(Undefined) is False
        assert is_unset(None) is False
        assert is_unset("value") is False


class TestSentinelPickle:
    """Test sentinel pickle/unpickle behavior.

    Pickle roundtrip is critical for multiprocessing and serialization.
    Sentinels must preserve singleton identity across pickle boundaries.
    """

    def test_pickle_undefined(self):
        """Undefined should survive pickle roundtrip and maintain singleton identity."""
        pickled = pickle.dumps(Undefined)
        unpickled = pickle.loads(pickled)

        # Should be the same singleton instance
        assert unpickled is Undefined
        assert isinstance(unpickled, UndefinedType)
        assert bool(unpickled) is False

    def test_pickle_unset(self):
        """Unset should survive pickle roundtrip and maintain singleton identity."""
        pickled = pickle.dumps(Unset)
        unpickled = pickle.loads(pickled)

        # Should be the same singleton instance
        assert unpickled is Unset
        assert isinstance(unpickled, UnsetType)
        assert bool(unpickled) is False

    def test_pickle_multiple_roundtrips(self):
        """Sentinel identity should persist across multiple pickle roundtrips."""
        for _ in range(3):
            undefined_pickled = pickle.dumps(Undefined)
            undefined_restored = pickle.loads(undefined_pickled)
            assert undefined_restored is Undefined

            unset_pickled = pickle.dumps(Unset)
            unset_restored = pickle.loads(unset_pickled)
            assert unset_restored is Unset


class TestSentinelCopy:
    """Test sentinel copy/deepcopy behavior."""

    def test_copy_undefined(self):
        """copy.copy(Undefined) should return same singleton."""
        import copy

        copied = copy.copy(Undefined)
        assert copied is Undefined

    def test_copy_unset(self):
        """copy.copy(Unset) should return same singleton."""
        import copy

        copied = copy.copy(Unset)
        assert copied is Unset

    def test_deepcopy_undefined(self):
        """copy.deepcopy(Undefined) should return same singleton."""
        import copy

        deep_copied = copy.deepcopy(Undefined)
        assert deep_copied is Undefined

    def test_deepcopy_unset(self):
        """copy.deepcopy(Unset) should return same singleton."""
        import copy

        deep_copied = copy.deepcopy(Unset)
        assert deep_copied is Unset


class TestSentinelUnionSyntax:
    """Test sentinel union syntax support (e.g., str | Undefined)."""

    def test_undefined_or_type(self):
        """Undefined | str should create a Union type."""
        from typing import Union, get_args, get_origin

        union_type = Undefined | str
        assert get_origin(union_type) is Union
        args = get_args(union_type)
        assert UndefinedType in args
        assert str in args

    def test_type_or_undefined(self):
        """str | Undefined should create a Union type."""
        from typing import Union, get_args, get_origin

        union_type = str | Undefined
        assert get_origin(union_type) is Union
        args = get_args(union_type)
        assert UndefinedType in args
        assert str in args

    def test_unset_or_type(self):
        """Unset | str should create a Union type."""
        from typing import Union, get_args, get_origin

        union_type = Unset | str
        assert get_origin(union_type) is Union
        args = get_args(union_type)
        assert UnsetType in args
        assert str in args

    def test_type_or_unset(self):
        """str | Unset should create a Union type."""
        from typing import Union, get_args, get_origin

        union_type = str | Unset
        assert get_origin(union_type) is Union
        args = get_args(union_type)
        assert UnsetType in args
        assert str in args
