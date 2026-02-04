# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Session persistence (dump/adump)."""

from __future__ import annotations

import json

import pytest

from krons.session import Session, SessionConfig
from krons.session.message import Message


class TestSessionDump:
    """Tests for Session.dump."""

    def test_dump_no_config(self):
        """dump should return None when log_persist_dir is None."""
        session = Session()
        assert session.dump() is None

    def test_dump_no_messages(self, tmp_path):
        """dump should return None when no messages."""
        config = SessionConfig(
            log_persist_dir=str(tmp_path),
            log_auto_save_on_exit=False,
        )
        session = Session(config=config)
        session.communications.clear()
        assert session.dump() is None

    def test_dump_writes_json(self, tmp_path):
        """dump should write full session as JSON."""
        config = SessionConfig(
            log_persist_dir=str(tmp_path),
            log_auto_save_on_exit=False,
        )
        session = Session(config=config)

        msg1 = Message(content={"role": "user", "text": "Hello"})
        msg2 = Message(content={"role": "assistant", "text": "Hi"})
        session.add_message(msg1)
        session.add_message(msg2)

        filepath = session.dump()

        assert filepath is not None
        assert filepath.exists()
        assert filepath.suffix == ".json"

        data = json.loads(filepath.read_bytes())
        assert data["id"] == str(session.id)
        assert "communications" in data

    def test_dump_no_clear_by_default(self, tmp_path):
        """dump should NOT clear by default."""
        config = SessionConfig(
            log_persist_dir=str(tmp_path),
            log_auto_save_on_exit=False,
        )
        session = Session(config=config)

        msg = Message(content={"keep": "me"})
        session.add_message(msg)

        session.dump()
        assert len(session.messages) == 1

    def test_dump_with_clear(self, tmp_path):
        """dump(clear=True) should clear communications."""
        config = SessionConfig(
            log_persist_dir=str(tmp_path),
            log_auto_save_on_exit=False,
        )
        session = Session(config=config)

        msg = Message(content={"delete": "me"})
        session.add_message(msg)

        session.dump(clear=True)
        assert len(session.messages) == 0
        assert len(session.branches) == 0

    def test_dump_creates_directory(self, tmp_path):
        """dump should create persist_dir if needed."""
        nested = tmp_path / "sub" / "dir"
        config = SessionConfig(
            log_persist_dir=str(nested),
            log_auto_save_on_exit=False,
        )
        session = Session(config=config)

        msg = Message(content={"test": 1})
        session.add_message(msg)

        filepath = session.dump()
        assert filepath is not None
        assert nested.exists()

    def test_atexit_registration(self, tmp_path):
        """Session should register atexit when configured."""
        config = SessionConfig(
            log_persist_dir=str(tmp_path),
            log_auto_save_on_exit=True,
        )
        session = Session(config=config)
        assert session._registered_atexit is True

    def test_no_atexit_when_disabled(self, tmp_path):
        """Session should not register atexit when disabled."""
        config = SessionConfig(
            log_persist_dir=str(tmp_path),
            log_auto_save_on_exit=False,
        )
        session = Session(config=config)
        assert session._registered_atexit is False

    def test_no_atexit_when_no_config(self):
        """Session should not register atexit when no log_persist_dir."""
        session = Session()
        assert session._registered_atexit is False

    def test_multiple_dumps(self, tmp_path):
        """Multiple dumps should create separate files."""
        config = SessionConfig(
            log_persist_dir=str(tmp_path),
            log_auto_save_on_exit=False,
        )
        session = Session(config=config)

        msg1 = Message(content={"batch": 1})
        session.add_message(msg1)
        p1 = session.dump()

        msg2 = Message(content={"batch": 2})
        session.add_message(msg2)
        p2 = session.dump()

        assert p1 != p2
        assert p1.exists()
        assert p2.exists()

    def test_save_at_exit_with_messages(self, tmp_path):
        """_save_at_exit should dump when messages present."""
        config = SessionConfig(
            log_persist_dir=str(tmp_path),
            log_auto_save_on_exit=False,
        )
        session = Session(config=config)

        msg = Message(content={"atexit": "test"})
        session.add_message(msg)

        session._save_at_exit()

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

    def test_save_at_exit_no_messages(self, tmp_path):
        """_save_at_exit should not create file when no messages."""
        config = SessionConfig(
            log_persist_dir=str(tmp_path),
            log_auto_save_on_exit=False,
        )
        session = Session(config=config)
        session.communications.clear()

        session._save_at_exit()

        files = list(tmp_path.glob("*"))
        assert len(files) == 0

    def test_dump_contains_session_structure(self, tmp_path):
        """Dump should contain full session structure for restoration."""
        config = SessionConfig(
            log_persist_dir=str(tmp_path),
            log_auto_save_on_exit=False,
        )
        session = Session(config=config)

        msg = Message(content={"role": "user", "text": "Hello"})
        session.add_message(msg, session.default_branch)

        filepath = session.dump()
        data = json.loads(filepath.read_bytes())

        # Verify structure is present for restoration
        assert data["id"] == str(session.id)
        assert "communications" in data
        assert "items" in data["communications"]
        assert "progressions" in data["communications"]
        assert len(data["communications"]["items"]["items"]) == 1


class TestSessionDumpAsync:
    """Tests for Session.adump."""

    @pytest.mark.anyio
    async def test_adump_writes_json(self, tmp_path):
        """adump should write full session as JSON."""
        config = SessionConfig(
            log_persist_dir=str(tmp_path),
            log_auto_save_on_exit=False,
        )
        session = Session(config=config)

        msg = Message(content={"async": True})
        session.add_message(msg)

        filepath = await session.adump()

        assert filepath is not None
        assert filepath.exists()

        data = json.loads(filepath.read_bytes())
        assert data["id"] == str(session.id)

    @pytest.mark.anyio
    async def test_adump_no_config(self):
        """adump should return None when no log_persist_dir."""
        session = Session()
        assert await session.adump() is None

    @pytest.mark.anyio
    async def test_adump_no_messages(self, tmp_path):
        """adump should return None when no messages."""
        config = SessionConfig(
            log_persist_dir=str(tmp_path),
            log_auto_save_on_exit=False,
        )
        session = Session(config=config)
        session.communications.clear()

        assert await session.adump() is None

    @pytest.mark.anyio
    async def test_adump_with_clear(self, tmp_path):
        """adump(clear=True) should clear communications."""
        config = SessionConfig(
            log_persist_dir=str(tmp_path),
            log_auto_save_on_exit=False,
        )
        session = Session(config=config)

        msg = Message(content={"clear": "me"})
        session.add_message(msg)

        await session.adump(clear=True)
        assert len(session.messages) == 0

    @pytest.mark.anyio
    async def test_adump_creates_directory(self, tmp_path):
        """adump should create persist_dir if needed."""
        nested = tmp_path / "async" / "logs"
        config = SessionConfig(
            log_persist_dir=str(nested),
            log_auto_save_on_exit=False,
        )
        session = Session(config=config)

        msg = Message(content={"test": 1})
        session.add_message(msg)

        filepath = await session.adump()
        assert filepath is not None
        assert nested.exists()
