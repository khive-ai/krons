# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Operations: executable graph nodes with dependency-aware execution.

Core types:
    Operation: Node + Event hybrid for graph-based execution.
    OperationRegistry: Per-session factory mapping.
    OperationGraphBuilder (Builder): Fluent DAG construction.

Execution:
    flow(): Execute DAG, return results dict.
    flow_stream(): Execute DAG, yield results incrementally.
"""

from __future__ import annotations

from .builder import Builder, OperationGraphBuilder
from .context import QueryFn, RequestContext
from .flow import DependencyAwareExecutor, flow, flow_stream
from .node import Operation
from .registry import OperationRegistry

__all__ = (
    "Builder",
    "DependencyAwareExecutor",
    "Operation",
    "OperationGraphBuilder",
    "OperationRegistry",
    "QueryFn",
    "RequestContext",
    "flow",
    "flow_stream",
)
