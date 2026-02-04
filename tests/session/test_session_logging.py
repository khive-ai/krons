# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Session message persistence."""

from __future__ import annotations

import json

import pytest

from krons.core.base.log import DataLoggerConfig
from krons.session import Session, SessionConfig
from krons.session.message import Message


class TestSessionMessageDump:
    """Tests for Session.dump_messages and adump_messages."""

    def test_dump_messages_no_config(self):
        """dump_messages should return None when log_config is None."""
        session = Session()
        assert session.dump_messages() is None

    def test_dump_messages_no_messages(self, tmp_path):
        """dump_messages should return None when no messages."""
        config = SessionConfig(
            log_config=DataLoggerConfig(
                persist_dir=str(tmp_path),
                auto_save_on_exit=False,
            )
        )
        session = Session(config=config)
        # Clear default branch messages if any
        session.communications.items.clear()
        assert session.dump_messages() is None

    def test_dump_messages_jsonl(self, tmp_path):
        """dump_messages should write JSONL file."""
        config = SessionConfig(
            log_config=DataLoggerConfig(
                persist_dir=str(tmp_path),
                extension=".jsonl",
                auto_save_on_exit=False,
            )
        )
        session = Session(config=config)

        msg1 = Message(content={"role": "user", "text": "Hello"})
        msg2 = Message(content={"role": "assistant", "text": "Hi there"})
        session.add_message(msg1)
        session.add_message(msg2)

        filepath = session.dump_messages()

        assert filepath is not None
        assert filepath.exists()
        assert filepath.suffix == ".jsonl"

        lines = filepath.read_text().strip().split("\n")
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first["content"]["role"] == "user"

    def test_dump_messages_json(self, tmp_path):
        """dump_messages should write JSON array file."""
        config = SessionConfig(
            log_config=DataLoggerConfig(
                persist_dir=str(tmp_path),
                extension=".json",
                auto_save_on_exit=False,
            )
        )
        session = Session(config=config)

        msg = Message(content={"test": "data"})
        session.add_message(msg)

        filepath = session.dump_messages()

        assert filepath is not None
        assert filepath.suffix == ".json"

        data = json.loads(filepath.read_bytes())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["content"]["test"] == "data"

    def test_dump_messages_no_clear_by_default(self, tmp_path):
        """dump_messages should NOT clear messages by default."""
        config = SessionConfig(
            log_config=DataLoggerConfig(
                persist_dir=str(tmp_path),
                auto_save_on_exit=False,
            )
        )
        session = Session(config=config)

        msg = Message(content={"keep": "me"})
        session.add_message(msg)

        session.dump_messages()
        assert len(session.messages) == 1

    def test_dump_messages_with_clear(self, tmp_path):
        """dump_messages(clear=True) should clear messages."""
        config = SessionConfig(
            log_config=DataLoggerConfig(
                persist_dir=str(tmp_path),
                auto_save_on_exit=False,
            )
        )
        session = Session(config=config)

        msg = Message(content={"delete": "me"})
        session.add_message(msg)

        session.dump_messages(clear=True)
        assert len(session.messages) == 0

    def test_dump_creates_directory(self, tmp_path):
        """dump_messages should create persist_dir if needed."""
        nested = tmp_path / "sub" / "dir"
        config = SessionConfig(
            log_config=DataLoggerConfig(
                persist_dir=str(nested),
                auto_save_on_exit=False,
            )
        )
        session = Session(config=config)

        msg = Message(content={"test": 1})
        session.add_message(msg)

        filepath = session.dump_messages()
        assert filepath is not None
        assert nested.exists()

    def test_atexit_registration(self):
        """Session should register atexit when configured."""
        config = SessionConfig(log_config=DataLoggerConfig(auto_save_on_exit=True))
        session = Session(config=config)
        assert session._registered_atexit is True

    def test_no_atexit_when_disabled(self):
        """Session should not register atexit when disabled."""
        config = SessionConfig(log_config=DataLoggerConfig(auto_save_on_exit=False))
        session = Session(config=config)
        assert session._registered_atexit is False

    def test_no_atexit_when_no_config(self):
        """Session should not register atexit when no log_config."""
        session = Session()
        assert session._registered_atexit is False

    def test_multiple_dumps(self, tmp_path):
        """Multiple dumps should create separate files."""
        config = SessionConfig(
            log_config=DataLoggerConfig(
                persist_dir=str(tmp_path),
                auto_save_on_exit=False,
            )
        )
        session = Session(config=config)

        msg1 = Message(content={"batch": 1})
        session.add_message(msg1)
        p1 = session.dump_messages()

        msg2 = Message(content={"batch": 2})
        session.add_message(msg2)
        p2 = session.dump_messages()

        assert p1 != p2
        assert p1.exists()
        assert p2.exists()

    def test_save_at_exit_with_messages(self, tmp_path):
        """_save_at_exit should dump messages when present."""
        config = SessionConfig(
            log_config=DataLoggerConfig(
                persist_dir=str(tmp_path),
                auto_save_on_exit=False,
            )
        )
        session = Session(config=config)

        msg = Message(content={"atexit": "test"})
        session.add_message(msg)

        # Call directly to test the method
        session._save_at_exit()

        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1

    def test_save_at_exit_no_messages(self, tmp_path):
        """_save_at_exit should not create file when no messages."""
        config = SessionConfig(
            log_config=DataLoggerConfig(
                persist_dir=str(tmp_path),
                auto_save_on_exit=False,
            )
        )
        session = Session(config=config)
        session.communications.items.clear()

        session._save_at_exit()

        files = list(tmp_path.glob("*"))
        assert len(files) == 0

    def test_save_at_exit_suppresses_errors(self, tmp_path):
        """_save_at_exit should suppress exceptions silently."""
        config = SessionConfig(
            log_config=DataLoggerConfig(
                persist_dir="/nonexistent/readonly/path",
                auto_save_on_exit=False,
            )
        )
        session = Session(config=config)

        msg = Message(content={"test": 1})
        session.add_message(msg)

        # Should not raise even with invalid path
        session._save_at_exit()  # No exception = pass


class TestSessionMessageDumpAsync:
    """Tests for Session.adump_messages."""

    @pytest.mark.anyio
    async def test_adump_messages_jsonl(self, tmp_path):
        """adump_messages should write JSONL file."""
        config = SessionConfig(
            log_config=DataLoggerConfig(
                persist_dir=str(tmp_path),
                extension=".jsonl",
                auto_save_on_exit=False,
            )
        )
        session = Session(config=config)

        msg = Message(content={"async": True})
        session.add_message(msg)

        filepath = await session.adump_messages()

        assert filepath is not None
        assert filepath.exists()

        lines = filepath.read_text().strip().split("\n")
        first = json.loads(lines[0])
        assert first["content"]["async"] is True

    @pytest.mark.anyio
    async def test_adump_messages_json(self, tmp_path):
        """adump_messages should write JSON array file."""
        config = SessionConfig(
            log_config=DataLoggerConfig(
                persist_dir=str(tmp_path),
                extension=".json",
                auto_save_on_exit=False,
            )
        )
        session = Session(config=config)

        msg = Message(content={"format": "json"})
        session.add_message(msg)

        filepath = await session.adump_messages()

        assert filepath is not None
        data = json.loads(filepath.read_bytes())
        assert data[0]["content"]["format"] == "json"

    @pytest.mark.anyio
    async def test_adump_no_config(self):
        """adump_messages should return None when no log_config."""
        session = Session()
        assert await session.adump_messages() is None

    @pytest.mark.anyio
    async def test_adump_no_messages(self, tmp_path):
        """adump_messages should return None when no messages."""
        config = SessionConfig(
            log_config=DataLoggerConfig(
                persist_dir=str(tmp_path),
                auto_save_on_exit=False,
            )
        )
        session = Session(config=config)
        session.communications.items.clear()

        assert await session.adump_messages() is None

    @pytest.mark.anyio
    async def test_adump_with_clear(self, tmp_path):
        """adump_messages(clear=True) should clear messages."""
        config = SessionConfig(
            log_config=DataLoggerConfig(
                persist_dir=str(tmp_path),
                auto_save_on_exit=False,
            )
        )
        session = Session(config=config)

        msg = Message(content={"clear": "me"})
        session.add_message(msg)

        await session.adump_messages(clear=True)
        assert len(session.messages) == 0

    @pytest.mark.anyio
    async def test_adump_creates_directory(self, tmp_path):
        """adump_messages should create persist_dir if needed."""
        nested = tmp_path / "async" / "logs"
        config = SessionConfig(
            log_config=DataLoggerConfig(
                persist_dir=str(nested),
                auto_save_on_exit=False,
            )
        )
        session = Session(config=config)

        msg = Message(content={"test": 1})
        session.add_message(msg)

        filepath = await session.adump_messages()
        assert filepath is not None
        assert nested.exists()
