# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Dictionary conversion utilities with recursive processing and JSON parsing."""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import Enum as _Enum
from typing import Any, cast

import orjson

from ._fuzzy_json import fuzzy_json

__all__ = ("to_dict",)


def to_dict(
    input_: Any,
    /,
    *,
    prioritize_model_dump: bool = False,
    fuzzy_parse: bool = False,
    suppress: bool = False,
    parser: Callable[[str], Any] | None = None,
    recursive: bool = False,
    max_recursive_depth: int | None = None,
    recursive_python_only: bool = True,
    use_enum_values: bool = False,
    **kwargs: Any,
) -> dict[str | int, Any]:
    """Convert input to dictionary with optional recursive processing.

    Type handling:
        - Mapping: copied to dict
        - str: parsed as JSON (orjson or custom parser)
        - set: {v: v for v in set}
        - Enum class: {name: member} or {name: value} if use_enum_values
        - list/tuple: {0: v0, 1: v1, ...} (enumerated)
        - Pydantic BaseModel: model_dump() or dict-like conversion
        - dataclass: dataclasses.asdict()
        - objects with __dict__: returns __dict__
        - None/Undefined: {}

    Args:
        input_: Object to convert.
        prioritize_model_dump: Call .model_dump() first for Pydantic models.
        fuzzy_parse: Use fuzzy_json() for malformed JSON strings.
        suppress: Return {} on errors instead of raising.
        parser: Custom parser(str, **kwargs) -> Any for string inputs.
        recursive: Recursively process nested structures.
        max_recursive_depth: Max depth (default 5, clamped to 10).
        recursive_python_only: Only recurse into Python builtins (not custom objects).
        use_enum_values: Use .value for Enum members.
        **kwargs: Passed to parser and model_dump().

    Returns:
        Dictionary representation. Keys are str or int (for enumerated iterables).

    Raises:
        ValueError: max_recursive_depth negative or >10.
        Exception: Conversion failure (unless suppress=True).

    Edge Cases:
        - Empty string with suppress=False: raises
        - Empty string with suppress=True: returns {}
        - Circular references: limited by max_recursive_depth
    """
    try:
        if not isinstance(max_recursive_depth, int):
            max_depth = 5
        else:
            if max_recursive_depth < 0:
                raise ValueError("max_recursive_depth must be a non-negative integer")
            if max_recursive_depth > 10:
                raise ValueError("max_recursive_depth must be less than or equal to 10")
            max_depth = max_recursive_depth

        str_parse_opts = {
            "fuzzy_parse": fuzzy_parse,
            "parser": parser,
            "use_enum_values": use_enum_values,
            **kwargs,
        }

        obj = input_
        if recursive:
            obj = _preprocess_recursive(
                obj,
                depth=0,
                max_depth=max_depth,
                recursive_custom_types=not recursive_python_only,
                str_parse_opts=str_parse_opts,
                prioritize_model_dump=prioritize_model_dump,
            )

        return _convert_top_level_to_dict(
            obj,
            fuzzy_parse=fuzzy_parse,
            parser=parser,
            prioritize_model_dump=prioritize_model_dump,
            use_enum_values=use_enum_values,
            **kwargs,
        )

    except Exception as e:
        if suppress or input_ == "":
            return {}
        raise e


def _is_na(obj: Any) -> bool:
    """Check if obj is None or a Pydantic/kron undefined sentinel (by typename)."""
    if obj is None:
        return True
    tname = type(obj).__name__
    return tname in {
        "Undefined",
        "UndefinedType",
        "PydanticUndefined",
        "PydanticUndefinedType",
    }


def _enum_class_to_dict(enum_cls: type[_Enum], use_enum_values: bool) -> dict[str, Any]:
    """Convert Enum class to {name: member} or {name: value} dict."""
    members = dict(enum_cls.__members__)
    if use_enum_values:
        return {k: v.value for k, v in members.items()}
    return {k: v for k, v in members.items()}


def _parse_str(
    s: str,
    *,
    fuzzy_parse: bool,
    parser: Callable[[str], Any] | None,
    **kwargs: Any,
) -> Any:
    """Parse string to Python object via JSON or custom parser.

    Args:
        s: String to parse.
        fuzzy_parse: Use fuzzy_json() for malformed JSON.
        parser: Custom parser(s, **kwargs). If provided, takes precedence.
        **kwargs: Passed to custom parser only (orjson.loads ignores them).

    Returns:
        Parsed Python object.
    """
    if parser is not None:
        return parser(s, **kwargs)
    if fuzzy_parse:
        with contextlib.suppress(NameError):
            return fuzzy_json(s)
    return orjson.loads(s)


def _object_to_mapping_like(
    obj: Any,
    *,
    prioritize_model_dump: bool = False,
    **kwargs: Any,
) -> Mapping | dict | Any:
    """Convert custom object to mapping-like via duck-typing.

    Conversion order:
        1. Pydantic model_dump() (if prioritize_model_dump)
        2. Common methods: to_dict, model_dump, dict, to_json, json
        3. dataclasses.asdict()
        4. __dict__ attribute
        5. dict(obj) fallback

    Args:
        obj: Object to convert.
        prioritize_model_dump: Try model_dump() first.
        **kwargs: Passed to conversion methods.

    Returns:
        Mapping-like object or dict.

    Raises:
        TypeError: If obj is not convertible (from dict() fallback).
    """
    if prioritize_model_dump and hasattr(obj, "model_dump"):
        return obj.model_dump(**kwargs)

    for name in ("to_dict", "model_dump", "dict", "to_json", "json"):
        if hasattr(obj, name):
            res = getattr(obj, name)(**kwargs)
            return orjson.loads(res) if isinstance(res, str) else res

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)

    if hasattr(obj, "__dict__"):
        return obj.__dict__

    return dict(obj)


def _enumerate_iterable(it: Iterable) -> dict[int, Any]:
    """Convert iterable to dict with integer indices as keys."""
    return {i: v for i, v in enumerate(it)}


def _preprocess_recursive(
    obj: Any,
    *,
    depth: int,
    max_depth: int,
    recursive_custom_types: bool,
    str_parse_opts: dict[str, Any],
    prioritize_model_dump: bool,
) -> Any:
    """Recursively preprocess nested structures before final conversion.

    Processing:
        - Strings: parsed as JSON, then recursed
        - Mappings: recurse into values (keys unchanged)
        - list/tuple/set/frozenset: recurse into items, preserve container type
        - Enum classes: convert to dict, then recurse
        - Custom objects (if recursive_custom_types): convert to mapping, then recurse

    Args:
        obj: Object to process.
        depth: Current recursion depth.
        max_depth: Maximum depth (stops recursion when reached).
        recursive_custom_types: Also convert custom objects.
        str_parse_opts: Options for _parse_str().
        prioritize_model_dump: Passed to _object_to_mapping_like().

    Returns:
        Preprocessed object with same container types.
    """
    if depth >= max_depth:
        return obj

    # Fast paths by exact type where possible
    t = type(obj)

    # Strings: try to parse; on failure, keep as-is
    if t is str:
        with contextlib.suppress(Exception):
            return _preprocess_recursive(
                _parse_str(obj, **str_parse_opts),
                depth=depth + 1,
                max_depth=max_depth,
                recursive_custom_types=recursive_custom_types,
                str_parse_opts=str_parse_opts,
                prioritize_model_dump=prioritize_model_dump,
            )
        return obj

    # Dict-like
    if isinstance(obj, Mapping):
        # Recurse only into values (keys kept as-is)
        return {
            k: _preprocess_recursive(
                v,
                depth=depth + 1,
                max_depth=max_depth,
                recursive_custom_types=recursive_custom_types,
                str_parse_opts=str_parse_opts,
                prioritize_model_dump=prioritize_model_dump,
            )
            for k, v in obj.items()
        }

    # Sequence/Set-like (but not str)
    if isinstance(obj, list | tuple | set | frozenset):
        items = [
            _preprocess_recursive(
                v,
                depth=depth + 1,
                max_depth=max_depth,
                recursive_custom_types=recursive_custom_types,
                str_parse_opts=str_parse_opts,
                prioritize_model_dump=prioritize_model_dump,
            )
            for v in obj
        ]
        if t is list:
            return items
        if t is tuple:
            return tuple(items)
        if t is set:
            return set(items)
        if t is frozenset:
            return frozenset(items)

    if isinstance(obj, type) and issubclass(obj, _Enum):
        with contextlib.suppress(Exception):
            enum_map = _enum_class_to_dict(
                obj,
                use_enum_values=str_parse_opts.get("use_enum_values", True),
            )
            return _preprocess_recursive(
                enum_map,
                depth=depth + 1,
                max_depth=max_depth,
                recursive_custom_types=recursive_custom_types,
                str_parse_opts=str_parse_opts,
                prioritize_model_dump=prioritize_model_dump,
            )
        return obj

    if recursive_custom_types:
        with contextlib.suppress(Exception):
            mapped = _object_to_mapping_like(obj, prioritize_model_dump=prioritize_model_dump)
            return _preprocess_recursive(
                mapped,
                depth=depth + 1,
                max_depth=max_depth,
                recursive_custom_types=recursive_custom_types,
                str_parse_opts=str_parse_opts,
                prioritize_model_dump=prioritize_model_dump,
            )

    return obj


def _convert_top_level_to_dict(
    obj: Any,
    *,
    fuzzy_parse: bool,
    parser: Callable[[str], Any] | None,
    prioritize_model_dump: bool,
    use_enum_values: bool,
    **kwargs: Any,
) -> dict[str | int, Any]:
    """Convert single object to dict (final conversion step).

    Conversion order:
        1. set -> {v: v}
        2. Enum class -> {name: member/value}
        3. Mapping -> dict(obj)
        4. None/undefined -> {}
        5. str -> parse as JSON
        6. Non-sequence objects -> _object_to_mapping_like
        7. Iterables -> enumerate to {int: value}
        8. Dataclass fallback
        9. dict(obj) last resort

    Args:
        obj: Object to convert.
        fuzzy_parse: Use fuzzy JSON parsing.
        parser: Custom string parser.
        prioritize_model_dump: Prefer model_dump() for Pydantic.
        use_enum_values: Use .value for Enum members.
        **kwargs: Passed to conversion methods.

    Returns:
        Dictionary with str or int keys.
    """
    if isinstance(obj, set):
        return cast(dict[str | int, Any], {v: v for v in obj})

    if isinstance(obj, type) and issubclass(obj, _Enum):
        return cast(dict[str | int, Any], _enum_class_to_dict(obj, use_enum_values))

    if isinstance(obj, Mapping):
        return cast(dict[str | int, Any], dict(obj))

    if _is_na(obj):
        return cast(dict[str | int, Any], {})

    if isinstance(obj, str):
        return _parse_str(obj, fuzzy_parse=fuzzy_parse, parser=parser, **kwargs)

    with contextlib.suppress(Exception):
        if not isinstance(obj, Sequence):
            converted = _object_to_mapping_like(
                obj, prioritize_model_dump=prioritize_model_dump, **kwargs
            )
            if isinstance(converted, str):
                return _parse_str(converted, fuzzy_parse=fuzzy_parse, parser=None)
            if isinstance(converted, Mapping):
                return dict(converted)
            if isinstance(converted, Iterable) and not isinstance(
                converted, str | bytes | bytearray
            ):
                return cast(dict[str | int, Any], _enumerate_iterable(converted))
            return dict(converted)

    if isinstance(obj, Iterable) and not isinstance(obj, str | bytes | bytearray):
        return cast(dict[str | int, Any], _enumerate_iterable(obj))

    with contextlib.suppress(Exception):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return cast(dict[str | int, Any], dataclasses.asdict(obj))

    return dict(obj)
