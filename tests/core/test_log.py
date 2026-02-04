# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for krons.core.base.log - DataLoggerConfig."""

from __future__ import annotations

from pathlib import Path

import pytest

from krons.core.base.log import DataLoggerConfig


class TestDataLoggerConfig:
    """Tests for DataLoggerConfig."""

    def test_defaults(self):
        """Config should have sensible defaults."""
        config = DataLoggerConfig()
        assert config.persist_dir == "./logs"
        assert config.extension == ".jsonl"
        assert config.auto_save_on_exit is True

    def test_custom_values(self):
        """Config should accept custom values."""
        config = DataLoggerConfig(
            persist_dir="/tmp/custom_logs",
            extension=".json",
            auto_save_on_exit=False,
        )
        assert config.persist_dir == "/tmp/custom_logs"
        assert config.extension == ".json"
        assert config.auto_save_on_exit is False

    def test_extension_validation(self):
        """Config should only accept .json or .jsonl."""
        with pytest.raises(Exception):
            DataLoggerConfig(extension=".csv")

    def test_persist_dir_as_path(self):
        """Config should accept Path object for persist_dir."""
        config = DataLoggerConfig(persist_dir=Path("/tmp/logs"))
        assert config.persist_dir == Path("/tmp/logs")
