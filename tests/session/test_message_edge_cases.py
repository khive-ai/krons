# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Edge case tests for Message class.

Migrated from lionpride and adapted to kron's simpler Message structure.

Focus areas:
    1. Sender/recipient UUID validation edge cases
    2. Content type edge cases
    3. is_broadcast/is_direct property edge cases
    4. Element-like sender/recipient handling
    5. repr edge cases
    6. None/null handling
"""

from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from krons.core import Element
from krons.session import Message


class TestMessageUUIDValidation:
    """Test Message sender/recipient UUID validation edge cases."""

    def test_sender_from_valid_uuid_string(self):
        """Message should convert valid UUID string to UUID object."""
        uid_str = "12345678-1234-5678-1234-567812345678"
        msg = Message(content={"text": "test"}, sender=uid_str)

        assert isinstance(msg.sender, UUID)
        assert str(msg.sender) == uid_str

    def test_recipient_from_valid_uuid_string(self):
        """Message should convert valid UUID string to UUID object."""
        uid_str = "87654321-4321-8765-4321-876543218765"
        msg = Message(content={"text": "test"}, recipient=uid_str)

        assert isinstance(msg.recipient, UUID)
        assert str(msg.recipient) == uid_str

    def test_both_sender_and_recipient_as_strings(self):
        """Message should convert both sender and recipient UUID strings."""
        sender_str = "12345678-1234-5678-1234-567812345678"
        recipient_str = "87654321-4321-8765-4321-876543218765"

        msg = Message(content={"text": "test"}, sender=sender_str, recipient=recipient_str)

        assert isinstance(msg.sender, UUID)
        assert isinstance(msg.recipient, UUID)
        assert str(msg.sender) == sender_str
        assert str(msg.recipient) == recipient_str

    def test_sender_from_uuid_object(self):
        """Message should accept UUID object directly."""
        uid = uuid4()
        msg = Message(content={"text": "test"}, sender=uid)

        assert msg.sender == uid
        assert isinstance(msg.sender, UUID)

    def test_sender_from_element_like_object(self):
        """Message should extract UUID from Element-like objects with .id attribute."""
        element = Element()  # Element has auto-generated UUID
        msg = Message(content={"text": "test"}, sender=element)

        assert msg.sender == element.id
        assert isinstance(msg.sender, UUID)

    def test_recipient_from_element_like_object(self):
        """Message should extract UUID from Element-like objects for recipient."""
        element = Element()
        msg = Message(content={"text": "test"}, recipient=element)

        assert msg.recipient == element.id
        assert isinstance(msg.recipient, UUID)

    def test_invalid_uuid_string_raises_valueerror(self):
        """Message should raise ValueError for invalid UUID string."""
        with pytest.raises(ValueError):
            Message(content={"text": "test"}, sender="not-a-valid-uuid")

    def test_invalid_recipient_string_raises_valueerror(self):
        """Message should raise ValueError for invalid recipient UUID string."""
        with pytest.raises(ValueError):
            Message(content={"text": "test"}, recipient="invalid-uuid")

    def test_partial_uuid_string_raises(self):
        """Message should raise ValueError for partial UUID strings."""
        with pytest.raises(ValueError):
            Message(content={"text": "test"}, sender="12345678")

    def test_uuid_with_extra_chars_raises(self):
        """Message should raise ValueError for UUID with extra characters."""
        with pytest.raises(ValueError):
            Message(
                content={"text": "test"},
                sender="12345678-1234-5678-1234-567812345678-extra",
            )

    def test_empty_string_sender_raises(self):
        """Message should raise ValueError for empty string sender."""
        with pytest.raises(ValueError):
            Message(content={"text": "test"}, sender="")


class TestMessageNoneHandling:
    """Test Message with None values."""

    def test_none_sender_preserved(self):
        """Message should allow None sender (anonymous message)."""
        msg = Message(content={"text": "test"}, sender=None)
        assert msg.sender is None

    def test_none_recipient_preserved(self):
        """Message should allow None recipient (broadcast message)."""
        msg = Message(content={"text": "test"}, recipient=None)
        assert msg.recipient is None

    def test_both_none_creates_anonymous_broadcast(self):
        """Message with both sender and recipient None is valid."""
        msg = Message(content={"text": "test"})
        assert msg.sender is None
        assert msg.recipient is None
        assert msg.is_broadcast is True

    def test_none_content_is_allowed(self):
        """Message should allow None content."""
        msg = Message(content=None)
        assert msg.content is None

    def test_none_channel_is_default(self):
        """Message channel should default to None."""
        msg = Message(content={"text": "test"})
        assert msg.channel is None


class TestMessageBroadcastDirectProperties:
    """Test is_broadcast and is_direct properties edge cases."""

    def test_is_broadcast_when_recipient_none(self):
        """is_broadcast should be True when recipient is None."""
        msg = Message(content={"text": "test"}, sender=uuid4(), recipient=None)
        assert msg.is_broadcast is True
        assert msg.is_direct is False

    def test_is_direct_when_recipient_set(self):
        """is_direct should be True when recipient is set."""
        msg = Message(content={"text": "test"}, sender=uuid4(), recipient=uuid4())
        assert msg.is_direct is True
        assert msg.is_broadcast is False

    def test_broadcast_without_sender(self):
        """Broadcast without sender is still valid broadcast."""
        msg = Message(content={"text": "test"}, sender=None, recipient=None)
        assert msg.is_broadcast is True
        assert msg.is_direct is False

    def test_direct_without_sender(self):
        """Direct message without sender is still valid direct message."""
        msg = Message(content={"text": "test"}, sender=None, recipient=uuid4())
        assert msg.is_direct is True
        assert msg.is_broadcast is False


class TestMessageContentValidation:
    """Test Message content type edge cases."""

    def test_dict_content_accepted(self):
        """Message should accept dict content."""
        msg = Message(content={"key": "value"})
        assert msg.content == {"key": "value"}

    def test_nested_dict_content_accepted(self):
        """Message should accept deeply nested dict content."""
        content = {"level1": {"level2": {"level3": {"value": [1, 2, 3]}}}}
        msg = Message(content=content)
        assert msg.content == content

    def test_empty_dict_content_accepted(self):
        """Message should accept empty dict content."""
        msg = Message(content={})
        assert msg.content == {}

    def test_string_content_rejected(self):
        """Message should reject raw string content."""
        with pytest.raises(TypeError):
            Message(content="raw string")

    def test_list_content_rejected(self):
        """Message should reject list content."""
        with pytest.raises(TypeError):
            Message(content=[1, 2, 3])

    def test_int_content_rejected(self):
        """Message should reject int content."""
        with pytest.raises(TypeError):
            Message(content=42)

    def test_float_content_rejected(self):
        """Message should reject float content."""
        with pytest.raises(TypeError):
            Message(content=3.14)

    def test_basemodel_content_accepted(self):
        """Message should accept BaseModel instance content."""

        class MyModel(BaseModel):
            field: str = "value"

        model = MyModel()
        msg = Message(content=model)
        # Content stored as BaseModel instance
        assert hasattr(msg.content, "field")


class TestMessageReprEdgeCases:
    """Test Message __repr__ edge cases."""

    def test_repr_with_broadcast_shows_broadcast(self):
        """Message repr should show 'broadcast' for broadcast messages."""
        msg = Message(content={"text": "test"}, sender=uuid4())
        repr_str = repr(msg)
        assert "broadcast" in repr_str

    def test_repr_with_unknown_sender_shows_unknown(self):
        """Message repr should show 'unknown' for unknown sender."""
        msg = Message(content={"text": "test"})
        repr_str = repr(msg)
        assert "unknown" in repr_str

    def test_repr_with_channel_shows_channel(self):
        """Message repr should show channel when set."""
        msg = Message(content={"text": "test"}, channel="updates")
        repr_str = repr(msg)
        assert "channel=updates" in repr_str

    def test_repr_without_channel_no_channel_in_output(self):
        """Message repr should not show channel when not set."""
        msg = Message(content={"text": "test"})
        repr_str = repr(msg)
        assert "channel=" not in repr_str

    def test_repr_shows_truncated_uuids(self):
        """Message repr should show first 8 chars of UUIDs."""
        sender_id = uuid4()
        recipient_id = uuid4()
        msg = Message(content={"text": "test"}, sender=sender_id, recipient=recipient_id)

        repr_str = repr(msg)
        assert str(sender_id)[:8] in repr_str
        assert str(recipient_id)[:8] in repr_str


class TestMessageChannelEdgeCases:
    """Test Message channel field edge cases."""

    def test_channel_with_empty_string(self):
        """Message should accept empty string channel."""
        msg = Message(content={"text": "test"}, channel="")
        assert msg.channel == ""

    def test_channel_with_special_characters(self):
        """Message should accept channel with special characters."""
        msg = Message(content={"text": "test"}, channel="user/updates#main")
        assert msg.channel == "user/updates#main"

    def test_channel_with_unicode(self):
        """Message should accept channel with unicode characters."""
        msg = Message(content={"text": "test"}, channel="updates_\u00e9")
        assert msg.channel == "updates_\u00e9"

    def test_channel_with_spaces(self):
        """Message should accept channel with spaces."""
        msg = Message(content={"text": "test"}, channel="my channel name")
        assert msg.channel == "my channel name"


class TestMessageMetadataEdgeCases:
    """Test Message metadata edge cases (inherited from Element)."""

    def test_metadata_with_nested_dict(self):
        """Message metadata should support nested dicts."""
        msg = Message(
            content={"text": "test"},
            metadata={
                "tags": ["urgent", "important"],
                "config": {"nested": {"deep": {"value": 42}}},
            },
        )
        assert msg.metadata["tags"] == ["urgent", "important"]
        assert msg.metadata["config"]["nested"]["deep"]["value"] == 42

    def test_metadata_modification_after_creation(self):
        """Message metadata should be modifiable after creation."""
        msg = Message(content={"text": "test"})
        msg.metadata["new_key"] = "new_value"
        assert msg.metadata["new_key"] == "new_value"


class TestMessageIdentityEdgeCases:
    """Test Message identity edge cases (inherited from Element)."""

    def test_multiple_messages_have_unique_ids(self):
        """Multiple messages should have unique UUIDs."""
        messages = [Message(content={"n": i}) for i in range(100)]
        ids = [m.id for m in messages]
        assert len(set(ids)) == 100

    def test_message_created_at_is_set(self):
        """Message should have created_at timestamp."""
        msg = Message(content={"text": "test"})
        assert msg.created_at is not None

    def test_message_id_is_uuid(self):
        """Message id should be UUID type."""
        msg = Message(content={"text": "test"})
        assert isinstance(msg.id, UUID)
