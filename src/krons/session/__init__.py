# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Session: conversation orchestration with messages, branches, and services.

Core types:
    Session: Central orchestrator owning branches, messages, services.
    Branch: Message progression with access control (capabilities, resources).
    Message: Inter-entity communication container.
    Exchange: Async message router between entity mailboxes.

Validators (raise on failure):
    resource_must_exist
    resource_must_be_accessible
    capabilities_must_be_granted
"""

from .constraints import (
    capabilities_must_be_granted,
    resource_must_be_accessible,
    resource_must_exist,
)
from .exchange import Exchange
from .message import Message
from .session import Branch, Session, SessionConfig

__all__ = (
    "Branch",
    "Exchange",
    "Message",
    "Session",
    "SessionConfig",
    "capabilities_must_be_granted",
    "resource_must_be_accessible",
    "resource_must_exist",
)
