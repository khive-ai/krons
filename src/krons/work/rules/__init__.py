# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Rules module: validation rules with auto-correction support.

Core exports:
- Rule, RuleParams, RuleQualifier: Base rule classes
- ValidationError: Validation failure exception
- Validator: Spec-aware validation orchestrator
- RuleRegistry: Type-to-rule mapping with inheritance
- Common rules: StringRule, NumberRule, BooleanRule, ChoiceRule, MappingRule, BaseModelRule
"""

from krons.errors import ValidationError

from .common import (
    BaseModelRule,
    BooleanRule,
    ChoiceRule,
    MappingRule,
    NumberRule,
    StringRule,
)
from .registry import RuleRegistry, get_default_registry, reset_default_registry
from .rule import Rule, RuleParams, RuleQualifier
from .validator import Validator

__all__ = (
    # Base classes
    "Rule",
    "RuleParams",
    "RuleQualifier",
    "ValidationError",
    # Validator
    "Validator",
    # Registry
    "RuleRegistry",
    "get_default_registry",
    "reset_default_registry",
    # Common rules
    "BaseModelRule",
    "BooleanRule",
    "ChoiceRule",
    "MappingRule",
    "NumberRule",
    "StringRule",
)
