# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.session.session - Session orchestrator."""

from uuid import UUID

import pytest

from krons.errors import AccessError, NotFoundError
from krons.session import (
    Message,
    Session,
    SessionConfig,
    capabilities_must_be_subset_of_branch,
    resolve_branch_exists_in_session,
    resource_must_be_accessible_by_branch,
)


class TestSessionConfig:
    """Test SessionConfig options."""

    def test_config_defaults(self):
        """SessionConfig should have sensible defaults."""
        config = SessionConfig()

        assert config.default_branch_name is None
        assert config.default_capabilities == set()
        assert config.default_resources == set()
        assert config.auto_create_default_branch is True

    def test_auto_create_default_branch(self):
        """SessionConfig.auto_create_default_branch should work."""
        # True by default - session creates a branch
        session_auto = Session(config=SessionConfig(auto_create_default_branch=True))
        assert session_auto.default_branch is not None
        assert len(session_auto.branches) == 1

        # False - no default branch created
        session_no_auto = Session(config=SessionConfig(auto_create_default_branch=False))
        assert session_no_auto.default_branch is None
        assert len(session_no_auto.branches) == 0

    def test_config_default_branch_name(self):
        """SessionConfig.default_branch_name should set branch name."""
        config = SessionConfig(default_branch_name="my-main-branch")
        session = Session(config=config)

        assert session.default_branch.name == "my-main-branch"

    def test_config_default_capabilities_and_resources(self):
        """SessionConfig should propagate defaults to created branch."""
        config = SessionConfig(
            default_capabilities={"tool1", "tool2"},
            default_resources={"service1", "service2"},
        )
        session = Session(config=config)

        assert session.default_branch.capabilities == {"tool1", "tool2"}
        assert session.default_branch.resources == {"service1", "service2"}


class TestSessionCreation:
    """Test Session instantiation."""

    def test_session_creates_default_branch(self):
        """Session should auto-create default branch."""
        session = Session()

        assert session.default_branch is not None
        assert session.default_branch.name == "main"
        assert session.default_branch_id is not None
        assert len(session.branches) == 1

    def test_session_has_registries(self):
        """Session should have services and operations registries."""
        session = Session()

        assert session.services is not None
        assert session.operations is not None
        assert session.communications is not None

    def test_session_has_uuid(self):
        """Session should have auto-generated UUID."""
        session = Session()
        assert isinstance(session.id, UUID)

    def test_session_user(self):
        """Session should support optional user field."""
        session = Session(user="test-user")
        assert session.user == "test-user"

        session_no_user = Session()
        assert session_no_user.user is None


class TestBranchManagement:
    """Test Session branch operations."""

    def test_create_branch(self):
        """Session.create_branch() should create new branch."""
        session = Session(config=SessionConfig(auto_create_default_branch=False))

        branch = session.create_branch(name="test-branch")

        assert branch.name == "test-branch"
        assert branch.session_id == session.id
        assert branch in session.branches
        assert len(session.branches) == 1

    def test_create_branch_with_capabilities(self):
        """Session.create_branch() should accept capabilities."""
        session = Session(config=SessionConfig(auto_create_default_branch=False))

        branch = session.create_branch(name="limited", capabilities={"tool1", "tool2"})

        assert branch.capabilities == {"tool1", "tool2"}

    def test_create_branch_with_resources(self):
        """Session.create_branch() should accept resources."""
        session = Session(config=SessionConfig(auto_create_default_branch=False))

        branch = session.create_branch(name="resource-branch", resources={"service1", "service2"})

        assert branch.resources == {"service1", "service2"}

    def test_get_branch_by_name(self):
        """Session.get_branch() should find by name."""
        session = Session()
        branch = session.create_branch(name="named-branch")

        retrieved = session.get_branch("named-branch")

        assert retrieved.id == branch.id
        assert retrieved.name == "named-branch"

    def test_get_branch_by_uuid(self):
        """Session.get_branch() should find by UUID."""
        session = Session()
        branch = session.create_branch(name="uuid-branch")

        retrieved = session.get_branch(branch.id)

        assert retrieved.id == branch.id
        assert retrieved.name == "uuid-branch"

    def test_get_branch_by_instance(self):
        """Session.get_branch() should accept Branch instance."""
        session = Session()
        branch = session.create_branch(name="instance-branch")

        retrieved = session.get_branch(branch)

        assert retrieved is branch

    def test_get_branch_not_found(self):
        """Session.get_branch() should raise NotFoundError if not found."""
        session = Session()

        with pytest.raises(NotFoundError):
            session.get_branch("nonexistent-branch")

    def test_get_branch_with_default(self):
        """Session.get_branch() should return default if not found."""
        session = Session()
        fallback = session.create_branch(name="fallback")

        result = session.get_branch("nonexistent", fallback)

        assert result is fallback

    def test_fork_branch(self):
        """Session.fork() should create copy of branch."""
        session = Session()
        original = session.create_branch(name="original")

        # Add a message to original
        msg = Message(content={"text": "test"})
        session.add_message(msg, branches=original)

        # Fork
        forked = session.fork(original, name="forked")

        assert forked.id != original.id
        assert forked.name == "forked"
        assert len(forked) == len(original)
        assert "forked_from" in forked.metadata

    def test_fork_branch_copies_messages_order(self):
        """Session.fork() should copy message UUIDs from source."""
        session = Session()
        original = session.create_branch(name="original")

        msg1 = Message(content={"text": "first"})
        msg2 = Message(content={"text": "second"})
        session.add_message(msg1, branches=original)
        session.add_message(msg2, branches=original)

        forked = session.fork(original, name="forked")

        assert list(forked.order) == list(original.order)

    def test_fork_branch_capabilities_copy(self):
        """Session.fork() with capabilities=True should copy capabilities."""
        session = Session()
        original = session.create_branch(name="original", capabilities={"cap1", "cap2"})

        forked = session.fork(original, name="forked", capabilities=True)

        assert forked.capabilities == {"cap1", "cap2"}

    def test_fork_branch_resources_copy(self):
        """Session.fork() with resources=True should copy resources."""
        session = Session()
        original = session.create_branch(name="original", resources={"res1", "res2"})

        forked = session.fork(original, name="forked", resources=True)

        assert forked.resources == {"res1", "res2"}

    def test_set_default_branch(self):
        """Session.set_default_branch() should change default."""
        session = Session()
        new_default = session.create_branch(name="new-default")

        session.set_default_branch(new_default)

        assert session.default_branch_id == new_default.id
        assert session.default_branch.name == "new-default"


class TestMessageManagement:
    """Test Session message operations."""

    def test_add_message(self):
        """Session.add_message() should add to session."""
        session = Session()
        msg = Message(content={"text": "test message"})

        session.add_message(msg)

        assert msg.id in session.messages
        assert len(session.messages) == 1

    def test_add_message_to_branch(self):
        """Session.add_message() should add to branches."""
        session = Session()
        branch = session.default_branch
        msg = Message(content={"text": "test message"})

        session.add_message(msg, branches=branch)

        # Message in session storage
        assert msg.id in session.messages

        # Message in branch order
        assert msg.id in branch.order

    def test_add_message_to_multiple_branches(self):
        """Session.add_message() should support multiple branches."""
        session = Session()
        branch1 = session.create_branch(name="branch1")
        branch2 = session.create_branch(name="branch2")
        msg = Message(content={"text": "shared message"})

        session.add_message(msg, branches=[branch1, branch2])

        # Message in both branches
        assert msg.id in branch1.order
        assert msg.id in branch2.order

        # But only stored once
        assert len(session.messages) == 1

    def test_messages_property(self):
        """Session.messages should be a Pile view."""
        session = Session()
        msg = Message(content={"text": "test"})
        session.add_message(msg)

        assert len(session.messages) == 1
        assert msg.id in session.messages

    def test_branches_property(self):
        """Session.branches should be a Pile view."""
        session = Session()

        # Default branch exists
        assert len(session.branches) >= 1

        session.create_branch(name="another")
        assert len(session.branches) >= 2


class TestBranch:
    """Test Branch class."""

    def test_branch_capabilities(self):
        """Branch should have capabilities set."""
        session = Session()
        branch = session.create_branch(name="capable", capabilities={"tool1", "tool2", "tool3"})

        assert "tool1" in branch.capabilities
        assert "tool2" in branch.capabilities
        assert "tool3" in branch.capabilities
        assert "tool4" not in branch.capabilities

    def test_branch_resources(self):
        """Branch should have resources set."""
        session = Session()
        branch = session.create_branch(name="resourceful", resources={"service1", "service2"})

        assert "service1" in branch.resources
        assert "service2" in branch.resources
        assert "service3" not in branch.resources

    def test_branch_session_id(self):
        """Branch should store session_id."""
        session = Session()
        branch = session.create_branch(name="test")

        assert branch.session_id == session.id

    def test_branch_len(self):
        """Branch len should reflect message count."""
        session = Session()
        branch = session.create_branch(name="test")

        assert len(branch) == 0

        msg = Message(content={"text": "test"})
        session.add_message(msg, branches=branch)

        assert len(branch) == 1

    def test_branch_repr(self):
        """Branch repr should show message count and session."""
        session = Session()
        branch = session.create_branch(name="my-branch")

        repr_str = repr(branch)

        assert "messages=" in repr_str
        assert "session=" in repr_str
        assert "my-branch" in repr_str


class TestSessionRepr:
    """Test Session string representation."""

    def test_session_repr(self):
        """Session repr should show counts."""
        session = Session()
        msg = Message(content={"text": "test"})
        session.add_message(msg, branches=session.default_branch)

        repr_str = repr(session)

        assert "messages=" in repr_str
        assert "branches=" in repr_str
        assert "services=" in repr_str


class TestAccessControl:
    """Test access control helper functions."""

    def test_resource_must_be_accessible_by_branch(self):
        """resource_must_be_accessible_by_branch should enforce access."""
        session = Session()
        branch = session.create_branch(name="limited", resources={"allowed_service"})

        # Should pass for allowed resource
        resource_must_be_accessible_by_branch(branch, "allowed_service")

        # Should raise for disallowed resource
        with pytest.raises(AccessError):
            resource_must_be_accessible_by_branch(branch, "forbidden_service")

    def test_capabilities_must_be_subset_of_branch(self):
        """capabilities_must_be_subset_of_branch should enforce subset."""
        session = Session()
        branch = session.create_branch(name="capable", capabilities={"cap1", "cap2"})

        # Should pass for subset
        capabilities_must_be_subset_of_branch(branch, {"cap1"})
        capabilities_must_be_subset_of_branch(branch, {"cap1", "cap2"})

        # Should raise for superset
        with pytest.raises(AccessError):
            capabilities_must_be_subset_of_branch(branch, {"cap1", "cap2", "cap3"})

    def test_resolve_branch_exists_in_session(self):
        """resolve_branch_exists_in_session should validate branch."""
        session = Session()
        branch = session.create_branch(name="existing")

        # Should return branch for valid
        result = resolve_branch_exists_in_session(session, branch)
        assert result is branch

        result_by_name = resolve_branch_exists_in_session(session, "existing")
        assert result_by_name.id == branch.id

        # Should raise for invalid
        with pytest.raises(NotFoundError):
            resolve_branch_exists_in_session(session, "nonexistent")
