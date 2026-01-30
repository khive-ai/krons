# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Enforcement module - Validation and policy protocols.

Provides:
- Rule: Base validation rule with auto-correction support
- Validator: Validates data against Spec/Operable using rules
- RuleRegistry: Maps types to validation rules
- Policy protocols: Abstract contracts for policy evaluation

Mental model:
    Rule = validation (is data valid?) - with optional auto-fix
    Policy = external evaluation protocol (implementations in domain libs)

Usage:
    from krons.enforcement import Rule, Validator, RuleRegistry

    # Register rules
    registry = RuleRegistry()
    registry.register(str, StringRule(min_length=1))

    # Validate
    validator = Validator(registry=registry)
    result = await validator.validate_spec(spec, value)
"""

from .context import QueryFn, RequestContext
from .policy import EnforcementLevel, PolicyEngine, PolicyResolver, ResolvedPolicy
from .registry import RuleRegistry, get_default_registry
from .rule import Rule, RuleParams, RuleQualifier, ValidationError
from .service import ActionMeta, KronConfig, KronService, action, get_action_meta
from .validator import Validator

__all__ = (
    # Rule system
    "Rule",
    "RuleParams",
    "RuleQualifier",
    "RuleRegistry",
    "ValidationError",
    "Validator",
    "get_default_registry",
    # Policy protocols
    "EnforcementLevel",
    "PolicyEngine",
    "PolicyResolver",
    "ResolvedPolicy",
    # Service
    "ActionMeta",
    "KronConfig",
    "KronService",
    "QueryFn",
    "RequestContext",
    "action",
    "get_action_meta",
)
