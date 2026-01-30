# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.session.message - Message container."""

from uuid import UUID, uuid4

import pytest

from krons.core import Element
from krons.session import Message


class TestMessageCreation:
    """Test Message instantiation."""

    def test_message_with_content(self):
        """Message should accept structured content (dict, Serializable, BaseModel, None)."""
        # Dict content
        msg = Message(content={"text": "Hello, world!"})
        assert msg.content == {"text": "Hello, world!"}

        # Nested dict content
        msg = Message(content={"key": "value", "nested": {"a": 1}})
        assert msg.content == {"key": "value", "nested": {"a": 1}}

        # None content
        msg = Message(content=None)
        assert msg.content is None

    def test_message_sender_recipient(self):
        """Message should accept sender and recipient UUIDs."""
        sender_id = uuid4()
        recipient_id = uuid4()

        msg = Message(content={"text": "test"}, sender=sender_id, recipient=recipient_id)

        assert msg.sender == sender_id
        assert msg.recipient == recipient_id

    def test_message_channel(self):
        """Message should accept optional channel."""
        msg = Message(content={"text": "test"}, channel="my-channel")
        assert msg.channel == "my-channel"

        # Channel is optional
        msg_no_channel = Message(content={"text": "test"})
        assert msg_no_channel.channel is None


class TestMessageProperties:
    """Test Message properties."""

    def test_is_broadcast(self):
        """is_broadcast should be True when no recipient."""
        msg = Message(content={"text": "broadcast message"}, sender=uuid4())
        assert msg.is_broadcast is True
        assert msg.recipient is None

    def test_is_direct(self):
        """is_direct should be True when recipient set."""
        sender_id = uuid4()
        recipient_id = uuid4()
        msg = Message(content={"text": "direct message"}, sender=sender_id, recipient=recipient_id)

        assert msg.is_direct is True
        assert msg.is_broadcast is False


class TestMessageValidation:
    """Test Message UUID validation."""

    def test_uuid_from_string(self):
        """Message should accept UUID as string."""
        sender_str = "12345678-1234-5678-1234-567812345678"
        recipient_str = "87654321-4321-8765-4321-876543218765"

        msg = Message(content={"text": "test"}, sender=sender_str, recipient=recipient_str)

        assert msg.sender == UUID(sender_str)
        assert msg.recipient == UUID(recipient_str)
        assert isinstance(msg.sender, UUID)
        assert isinstance(msg.recipient, UUID)

    def test_uuid_from_element(self):
        """Message should extract UUID from Element-like objects."""
        sender_element = Element()
        recipient_element = Element()

        msg = Message(content={"text": "test"}, sender=sender_element, recipient=recipient_element)

        assert msg.sender == sender_element.id
        assert msg.recipient == recipient_element.id

    def test_invalid_uuid_raises(self):
        """Message should raise ValueError for invalid UUID inputs."""
        with pytest.raises(ValueError):
            Message(content={"text": "test"}, sender="not-a-uuid")

        with pytest.raises(ValueError):
            Message(content={"text": "test"}, recipient="invalid")

    def test_none_sender_and_recipient(self):
        """Message should accept None for sender and recipient."""
        msg = Message(content={"text": "anonymous message"})
        assert msg.sender is None
        assert msg.recipient is None

    def test_content_must_be_structured(self):
        """Message content must be dict, Serializable, BaseModel, or None."""
        # Raw strings are not allowed
        with pytest.raises(TypeError):
            Message(content="raw string not allowed")

        # Lists are not allowed
        with pytest.raises(TypeError):
            Message(content=[1, 2, 3])


class TestMessageIdentity:
    """Test Message identity (inherited from Element)."""

    def test_message_has_uuid(self):
        """Message should auto-generate UUID on creation."""
        msg = Message(content={"text": "test"})
        assert isinstance(msg.id, UUID)

    def test_message_has_created_at(self):
        """Message should have created_at timestamp."""
        msg = Message(content={"text": "test"})
        assert msg.created_at is not None

    def test_message_repr(self):
        """Message repr should show sender -> target."""
        sender_id = uuid4()
        recipient_id = uuid4()

        # Direct message
        msg = Message(content={"text": "test"}, sender=sender_id, recipient=recipient_id)
        repr_str = repr(msg)
        assert str(sender_id)[:8] in repr_str
        assert str(recipient_id)[:8] in repr_str

        # Broadcast message
        broadcast = Message(content={"text": "test"}, sender=sender_id)
        repr_str = repr(broadcast)
        assert "broadcast" in repr_str

        # With channel
        channeled = Message(content={"text": "test"}, sender=sender_id, channel="updates")
        repr_str = repr(channeled)
        assert "channel=updates" in repr_str


class TestMessageMetadata:
    """Test Message metadata (inherited from Element)."""

    def test_message_metadata(self):
        """Message should support metadata dict."""
        msg = Message(content={"text": "test"}, metadata={"priority": "high", "tags": ["urgent"]})
        assert msg.metadata["priority"] == "high"
        assert msg.metadata["tags"] == ["urgent"]

    def test_message_metadata_default_empty(self):
        """Message metadata should default to empty dict."""
        msg = Message(content={"text": "test"})
        assert msg.metadata == {} or "kron_class" in msg.metadata
