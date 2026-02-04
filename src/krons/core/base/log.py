# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Logging configuration for Session message persistence.

Provides DataLoggerConfig for configuring automatic message dumps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from krons.core.types import HashableModel

__all__ = ("DataLoggerConfig",)


class DataLoggerConfig(HashableModel):
    """Configuration for Session message persistence.

    Attributes:
        persist_dir: Directory for dump files.
        extension: Output format (.json array or .jsonl newline-delimited).
        auto_save_on_exit: Register atexit handler on Session creation.
    """

    persist_dir: str | Path = Field(default="./logs")
    extension: Literal[".json", ".jsonl"] = Field(default=".jsonl")
    auto_save_on_exit: bool = Field(default=True)
