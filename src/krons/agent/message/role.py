# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from krons.core.types import DataClass, Enum, ModelConfig
from krons.protocols import Deserializable, implements

logger = logging.getLogger(__name__)


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    ACTION = "action"
    UNSET = "unset"


@implements(Deserializable)
@dataclass(slots=True)
class RoledContent(DataClass):
    _config: ClassVar[ModelConfig] = ModelConfig(
        sentinel_additions=frozenset({"none", "empty"}),
        use_enum_values=True,
    )

    role: Role = Role.UNSET

    @classmethod
    @abstractmethod
    def create(cls, **kwargs) -> RoledContent:
        raise NotImplementedError("Subclasses must implement create method")

    @abstractmethod
    def render(self, *args, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement render method")

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> RoledContent:
        return cls.create(
            **{
                k: v
                for k in cls.allowed()
                if (k in data and not cls._is_sentinel(v := data[k]))
            }
        )
