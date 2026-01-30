# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Numeric conversion utilities with bounds checking and security limits."""

import math
from decimal import Decimal
from typing import Any

__all__ = ("to_num", "MAX_NUMBER_STRING_LENGTH")

MAX_NUMBER_STRING_LENGTH = 1000
"""Max string length for numeric conversion (DoS protection)."""


def to_num(
    input_: Any,
    /,
    *,
    upper_bound: int | float | None = None,
    lower_bound: int | float | None = None,
    num_type: type[int] | type[float] = float,
    precision: int | None = None,
    allow_inf: bool = False,
) -> int | float:
    """Convert input to numeric type with bounds checking and validation.

    Handles: bool, int, float, Decimal, str. Strings are stripped before parsing.

    Args:
        input_: Value to convert. Bools treated as int (True=1, False=0).
        upper_bound: Maximum allowed value (inclusive).
        lower_bound: Minimum allowed value (inclusive).
        num_type: Target type, must be `int` or `float`.
        precision: Decimal places for float rounding (ignored for int).
        allow_inf: Permit infinity values. Default False.

    Returns:
        Converted numeric value of type `num_type`.

    Raises:
        ValueError: Empty/too-long string, out of bounds, NaN, inf (when disallowed).
        TypeError: Unsupported input type.

    Edge Cases:
        - Empty string: raises ValueError
        - Whitespace-only string: raises ValueError (stripped to empty)
        - "inf"/"-inf": raises unless allow_inf=True
        - "nan": always raises
        - Decimal: converted via float (may lose precision)
    """
    if num_type not in (int, float):
        raise ValueError(f"Invalid number type: {num_type}")

    if isinstance(input_, (bool, int, float, Decimal)):
        value = float(input_)
    elif isinstance(input_, str):
        input_ = input_.strip()
        if not input_:
            raise ValueError("Empty string cannot be converted to number")
        if len(input_) > MAX_NUMBER_STRING_LENGTH:
            msg = f"String length ({len(input_)}) exceeds maximum ({MAX_NUMBER_STRING_LENGTH})"
            raise ValueError(msg)
        try:
            value = float(input_)
        except ValueError as e:
            raise ValueError(f"Cannot convert '{input_}' to number") from e
    else:
        raise TypeError(f"Cannot convert {type(input_).__name__} to number")

    # NaN bypasses all comparisons; always reject
    if math.isnan(value):
        raise ValueError("NaN is not allowed")
    if math.isinf(value) and not allow_inf:
        raise ValueError("Infinity is not allowed (use allow_inf=True to permit)")

    if upper_bound is not None and value > upper_bound:
        raise ValueError(f"Value {value} exceeds upper bound {upper_bound}")
    if lower_bound is not None and value < lower_bound:
        raise ValueError(f"Value {value} below lower bound {lower_bound}")

    if precision is not None and num_type is float:
        value = round(value, precision)

    return num_type(value)
