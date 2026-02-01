# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol, cast, runtime_checkable
from typing_extensions import Self

from pydantic import BaseModel, JsonValue

from krons.core.types import DataClass, Enum, MaybeUnset, ModelConfig, Unset, is_unset
from krons.protocols import Deserializable, implements
from krons.utils import now_utc
from krons.utils.schemas import is_pydantic_model, minimal_yaml
from krons.resources import NormalizedResponse

from ._utils import _format_json_response_structure, _format_model_schema, _format_task

logger = logging.getLogger(__name__)



class MessageRole(Enum):
    """Roles for message sender/recipient in chat interactions."""

    SYSTEM = "system"
    AGENT = "agent"
    USER = "user"
    EVENT = "event"






    SYSTEM = "system"
    """System/Developer instructions defining model behavior"""



    
    """Assistant response (model-generated)"""

    TOOL = "tool"
    """Tool result returned after tool_call execution"""

    UNSET = "unset"
    """No role specified (fallback/unknown)"""
