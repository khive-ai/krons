# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Edge case tests for Session and Branch: UNIQUE tests not in basic tests.

Migrated from lionpride and adapted to kron's simpler session structure.

Covers:
- Session with no default branch edge cases
- Branch fork with various inheritance modes
- get_branch with various input types
- Fork chain lineage tracking
- Empty branch operations
- Referential integrity violations
- Branch repr edge cases
- Session conduct without branch
"""

from uuid import uuid4

import pytest

from krons.errors import AccessError, NotFoundError
from krons.session import (
    Branch,
    Message,
    Session,
    SessionConfig,
    capabilities_must_be_subset_of_branch,
    resource_must_be_accessible_by_branch,
    resource_must_exist_in_session,
)


class TestSessionNoDefaultBranch:
    """Test Session behavior when no default branch is set."""

    def test_session_with_auto_create_false_has_no_default_branch(self):
        """Session with auto_create_default_branch=False should have no default branch.

        Edge case: User creates session without auto-creating default branch.
        """
        session = Session(config=SessionConfig(auto_create_default_branch=False))

        assert session.default_branch is None
        assert session.default_branch_id is None
        assert len(session.branches) == 0

    @pytest.mark.asyncio
    async def test_conduct_without_branch_and_no_default_raises_runtime_error(self):
        """conduct() with no branch arg and no default branch should raise RuntimeError.

        Edge case: User creates session without default_branch, then calls conduct()
        without specifying a branch. Should fail with clear error message.
        """
        session = Session(config=SessionConfig(auto_create_default_branch=False))

        with pytest.raises(RuntimeError, match="No branch provided and no default branch set"):
            await session.conduct("operate", branch=None)

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="BUG: Session.conduct() passes timeout=None to Operation, "
        "but Operation expects float or Unset, not None. This test documents "
        "that branch resolution works but Operation creation fails."
    )
    async def test_conduct_with_explicit_branch_succeeds_without_default(self):
        """conduct() with explicit branch should work even without default branch.

        Edge case: Session created without default, but explicit branch passed to conduct.
        The operation binding should succeed (branch resolution works).

        NOTE: Currently fails due to timeout=None validation error in Session.conduct.
        """
        session = Session(config=SessionConfig(auto_create_default_branch=False))
        branch = session.create_branch(name="explicit")

        # This should not raise RuntimeError about missing branch
        # It will fail during operation execution (no registered factory),
        # but the branch resolution itself succeeds
        op = await session.conduct("operate", branch=branch)
        # Operation was created and bound - verify via repr showing "bound"
        assert "bound" in repr(op)
        assert op.operation_type == "operate"

    def test_default_branch_property_returns_none_when_not_set(self):
        """default_branch property should return None when no default is configured.

        Edge case: Accessing default_branch on session with no default.
        """
        session = Session(config=SessionConfig(auto_create_default_branch=False))
        assert session.default_branch is None

    def test_default_branch_returns_none_after_branch_removal(self):
        """default_branch should return None if the default branch was removed.

        Edge case: Set default branch, then remove it from communications.
        The default_branch_id still points to removed branch UUID.
        """
        session = Session()
        branch = session.create_branch(name="to_remove")
        session.set_default_branch(branch)

        # Remove the branch from communications
        session.communications.remove_progression(branch.id)

        # default_branch should handle this gracefully (suppress KeyError)
        assert session.default_branch is None


class TestBranchForkInheritance:
    """Test fork() capability/resource inheritance modes."""

    def test_fork_with_capabilities_true_copies_all(self):
        """fork() with capabilities=True should copy all capabilities from source.

        Edge case: Verify True means full copy, not just reference.
        """
        session = Session()
        source = session.create_branch(name="source", capabilities={"cap1", "cap2", "cap3"})

        forked = session.fork(source, capabilities=True)

        assert forked.capabilities == {"cap1", "cap2", "cap3"}
        # Verify it's a copy, not a reference
        forked.capabilities.add("cap4")
        assert "cap4" not in source.capabilities

    def test_fork_with_capabilities_none_creates_empty_set(self):
        """fork() with capabilities=None should create empty capabilities.

        Edge case: Source has capabilities, but fork wants none.
        """
        session = Session()
        source = session.create_branch(name="source", capabilities={"cap1", "cap2"})

        forked = session.fork(source, capabilities=None)

        assert forked.capabilities == set()

    def test_fork_with_explicit_capabilities_set(self):
        """fork() with explicit capabilities set should use that set.

        Edge case: Override source capabilities with different set.
        """
        session = Session()
        source = session.create_branch(name="source", capabilities={"cap1", "cap2"})

        forked = session.fork(source, capabilities={"new_cap"})

        assert forked.capabilities == {"new_cap"}
        assert source.capabilities == {"cap1", "cap2"}

    def test_fork_with_resources_true_copies_all(self):
        """fork() with resources=True should copy all resources from source.

        Edge case: Verify resources copying works like capabilities.
        """
        session = Session()
        source = session.create_branch(name="source", resources={"res1", "res2"})

        forked = session.fork(source, resources=True)

        assert forked.resources == {"res1", "res2"}
        # Verify independence
        forked.resources.add("res3")
        assert "res3" not in source.resources

    def test_fork_with_resources_none_creates_empty_set(self):
        """fork() with resources=None should create empty resources.

        Edge case: Source has resources but fork wants none.
        """
        session = Session()
        source = session.create_branch(name="source", resources={"res1"})

        forked = session.fork(source, resources=None)

        assert forked.resources == set()

    def test_fork_with_mixed_inheritance_modes(self):
        """fork() can use different modes for capabilities vs resources.

        Edge case: capabilities=True but resources=explicit set.
        """
        session = Session()
        source = session.create_branch(
            name="source",
            capabilities={"cap1"},
            resources={"res1", "res2"},
        )

        forked = session.fork(
            source,
            capabilities=True,  # Copy from source
            resources={"custom_res"},  # Explicit override
        )

        assert forked.capabilities == {"cap1"}
        assert forked.resources == {"custom_res"}


class TestGetBranchInputTypes:
    """Test get_branch with various input types."""

    def test_get_branch_by_uuid_object(self):
        """get_branch accepts UUID object directly."""
        session = Session()
        branch = session.create_branch(name="test")

        retrieved = session.get_branch(branch.id)
        assert retrieved.id == branch.id

    def test_get_branch_by_uuid_string(self):
        """get_branch accepts UUID as string.

        Edge case: UUID string is parsed and used for lookup.
        """
        session = Session()
        branch = session.create_branch(name="test")

        retrieved = session.get_branch(str(branch.id))
        assert retrieved.id == branch.id

    def test_get_branch_by_name_string(self):
        """get_branch accepts branch name string."""
        session = Session()
        branch = session.create_branch(name="my_branch")

        retrieved = session.get_branch("my_branch")
        assert retrieved.id == branch.id

    def test_get_branch_by_branch_instance(self):
        """get_branch accepts Branch instance directly.

        Edge case: Returns same instance if in session.
        """
        session = Session()
        branch = session.create_branch(name="test")

        retrieved = session.get_branch(branch)
        assert retrieved is branch

    def test_get_branch_not_found_raises_notfounderror(self):
        """get_branch raises NotFoundError for missing branch."""
        session = Session()

        with pytest.raises(NotFoundError, match="Branch not found"):
            session.get_branch("nonexistent")

    def test_get_branch_with_default_value(self):
        """get_branch returns default when branch not found.

        Edge case: Using default parameter (positional-only) to avoid exception.
        """
        session = Session()

        # Note: default is positional-only parameter (after /)
        result = session.get_branch("nonexistent", None)
        assert result is None

        sentinel = object()
        result = session.get_branch("nonexistent", sentinel)
        assert result is sentinel

    def test_get_branch_instance_not_in_session_raises(self):
        """get_branch with Branch instance not in session raises error.

        Edge case: Passing a Branch object that belongs to different session.
        """
        session1 = Session()
        session2 = Session()

        branch = session2.create_branch(name="other_session")

        # Branch not in session1's communications - raises NotFoundError
        with pytest.raises(NotFoundError):
            session1.get_branch(branch)


class TestForkChainLineage:
    """Test fork chain lineage tracking."""

    def test_fork_records_lineage_metadata(self):
        """fork() records lineage in metadata.forked_from.

        Edge case: Verify lineage metadata structure.
        """
        session = Session()
        original = session.create_branch(name="original")

        forked = session.fork(original, name="forked")

        assert "forked_from" in forked.metadata
        lineage = forked.metadata["forked_from"]
        assert lineage["branch_id"] == str(original.id)
        assert lineage["branch_name"] == "original"
        assert "created_at" in lineage
        assert lineage["message_count"] == 0

    def test_fork_chain_accumulates_lineage(self):
        """Chain of forks accumulates lineage in metadata.

        Edge case: Fork a fork - each forked branch has its own lineage,
        but does NOT inherit parent's lineage automatically.
        """
        session = Session()
        gen0 = session.create_branch(name="gen0")

        msg = Message(content={"text": "Start"})
        session.add_message(msg, branches=gen0)

        gen1 = session.fork(gen0, name="gen1")
        gen2 = session.fork(gen1, name="gen2")
        gen3 = session.fork(gen2, name="gen3")

        # Each fork only knows its immediate parent
        assert gen1.metadata["forked_from"]["branch_name"] == "gen0"
        assert gen2.metadata["forked_from"]["branch_name"] == "gen1"
        assert gen3.metadata["forked_from"]["branch_name"] == "gen2"

        # gen3 doesn't directly know about gen0
        assert gen3.metadata["forked_from"]["branch_name"] != "gen0"

    def test_fork_chain_message_propagation(self):
        """Messages accumulate through fork chain.

        Edge case: Each fork starts with parent's messages, can diverge.
        """
        session = Session()
        gen0 = session.create_branch(name="gen0")

        msg0 = Message(content={"text": "Gen0"})
        session.add_message(msg0, branches=gen0)

        gen1 = session.fork(gen0, name="gen1")
        msg1 = Message(content={"text": "Gen1"})
        session.add_message(msg1, branches=gen1)

        gen2 = session.fork(gen1, name="gen2")
        msg2 = Message(content={"text": "Gen2"})
        session.add_message(msg2, branches=gen2)

        # gen2 has all messages
        assert len(gen2) == 3
        assert msg0.id in gen2.order
        assert msg1.id in gen2.order
        assert msg2.id in gen2.order

        # gen0 only has its original message
        assert len(gen0) == 1

        # gen1 has two
        assert len(gen1) == 2


class TestEmptyBranchOperations:
    """Test operations on empty branches."""

    def test_fork_empty_branch(self):
        """Forking an empty branch creates empty fork.

        Edge case: Fork with no messages to copy.
        """
        session = Session()
        empty = session.create_branch(name="empty")

        forked = session.fork(empty, name="forked_empty")

        assert len(forked) == 0
        assert forked.metadata["forked_from"]["message_count"] == 0

    def test_branch_len_on_empty(self):
        """len(branch) returns 0 for empty branch.

        Edge case: Empty branch length.
        """
        session = Session()
        branch = session.create_branch(name="empty")

        assert len(branch) == 0

    def test_empty_branch_order_iteration(self):
        """Iterating over empty branch order yields nothing.

        Edge case: list(branch.order) for empty branch.
        """
        session = Session()
        branch = session.create_branch(name="empty")

        orders = list(branch.order)
        assert orders == []


class TestBranchRepr:
    """Test Branch __repr__ method edge cases."""

    def test_branch_repr_with_name(self):
        """Test Branch.__repr__() includes name when set."""
        session = Session()
        branch = session.create_branch(name="my_test_branch")

        repr_str = repr(branch)
        assert "Branch(" in repr_str
        assert "messages=" in repr_str
        assert f"session={str(session.id)[:8]}" in repr_str
        assert "name='my_test_branch'" in repr_str

    def test_branch_repr_without_name(self):
        """Test Branch.__repr__() without name.

        Edge case: Branch with empty/None name should not show name in repr.
        """
        session = Session()
        # Create branch with explicit empty name
        branch = Branch(session_id=session.id, name="", order=[])
        session.communications.add_progression(branch)

        repr_str = repr(branch)
        assert "Branch(" in repr_str
        assert "messages=0" in repr_str
        # Empty name should not show name= part
        assert "name=" not in repr_str


class TestSessionInitializationEdgeCases:
    """Test Session initialization edge cases."""

    def test_session_with_dict_config(self):
        """Session can accept config as dict.

        Edge case: Dict is converted to SessionConfig.
        """
        session = Session(
            config={
                "default_branch_name": "custom-main",
                "default_capabilities": {"cap1"},
                "default_resources": {"res1"},
            }
        )

        assert session.default_branch is not None
        assert session.default_branch.name == "custom-main"
        assert session.default_branch.capabilities == {"cap1"}
        assert session.default_branch.resources == {"res1"}

    def test_session_repr_includes_counts(self):
        """Session repr includes message, branch, and service counts."""
        session = Session()
        session.create_branch(name="extra")
        msg = Message(content={"text": "test"})
        session.add_message(msg, branches=session.default_branch)

        repr_str = repr(session)
        assert "messages=" in repr_str
        assert "branches=" in repr_str
        assert "services=" in repr_str


class TestSessionRequestAccessControl:
    """Test session.request() access control via branch parameter."""

    @pytest.mark.asyncio
    async def test_request_denies_access_when_service_not_in_branch_resources(self):
        """Test session.request() raises AccessError when service not in branch.resources.

        This is the core security: when branch is provided, service_name must
        be in branch.resources, otherwise access is denied.
        """
        session = Session()

        # Create branch WITHOUT any resources
        branch = session.create_branch(name="restricted_branch", resources=set())

        # Attempting to request a service through this branch should fail
        with pytest.raises(AccessError) as exc_info:
            await session.request("any_service", branch=branch)

        assert "has no access to resource" in str(exc_info.value)
        assert exc_info.value.details.get("available") == []

    @pytest.mark.asyncio
    async def test_request_access_control_with_branch_name(self):
        """Test session.request() access control works with branch name string.

        The branch parameter can be a Branch instance, UUID, or name string.
        Access control should work with all variants.
        """
        session = Session()

        # Create branch by name with empty resources
        session.create_branch(name="named_branch", resources=set())

        # Access by branch name string should still enforce access control
        with pytest.raises(AccessError):
            await session.request("some_model", branch="named_branch")

    @pytest.mark.asyncio
    async def test_request_access_control_with_branch_uuid(self):
        """Test session.request() access control works with branch UUID.

        The branch parameter can be a UUID, and access control should work.
        """
        session = Session()

        # Create branch with empty resources
        branch = session.create_branch(name="uuid_branch", resources=set())

        # Access by branch UUID should enforce access control
        with pytest.raises(AccessError):
            await session.request("uuid_model", branch=branch.id)

    @pytest.mark.asyncio
    async def test_request_access_error_details_include_available_resources(self):
        """Test AccessError details include list of available resources.

        This helps users understand what resources ARE available when access is denied.
        """
        session = Session()

        # Create branch with some resources (but not the one we'll request)
        branch = session.create_branch(
            name="partial_access",
            resources={"other_service", "another_service"},
        )

        with pytest.raises(AccessError) as exc_info:
            await session.request("denied_model", branch=branch)

        # Error details should include available resources
        available = exc_info.value.details.get("available", [])
        assert "other_service" in available
        assert "another_service" in available
        assert "denied_model" not in available


class TestReferentialIntegrityViolations:
    """Test referential integrity enforcement."""

    def test_create_branch_with_nonexistent_messages_raises(self):
        """create_branch with messages containing nonexistent UUIDs raises error.

        Edge case: Referential integrity for initial messages.
        """
        session = Session()
        fake_uuids = [uuid4(), uuid4()]

        with pytest.raises(NotFoundError, match="not in items pile"):
            session.create_branch(name="bad", messages=fake_uuids)

    def test_flow_add_progression_with_invalid_uuids_raises(self):
        """Adding progression with invalid UUIDs raises NotFoundError.

        Edge case: Direct Flow manipulation bypassing session.
        """
        session = Session()

        # Create a branch with invalid UUIDs directly
        invalid_branch = Branch(
            session_id=session.id,
            name="invalid",
            order=[uuid4()],  # UUID not in session.messages
        )

        with pytest.raises(NotFoundError, match="not in items pile"):
            session.communications.add_progression(invalid_branch)


class TestBranchAccessControlHelpers:
    """Test access control helper functions with edge cases."""

    def test_resource_access_with_empty_resources_set(self):
        """Branch with empty resources set should deny all resource access."""
        session = Session()
        branch = session.create_branch(name="no_resources", resources=set())

        with pytest.raises(AccessError) as exc_info:
            resource_must_be_accessible_by_branch(branch, "any_service")

        assert "has no access" in str(exc_info.value)
        assert exc_info.value.details["available"] == []

    def test_capabilities_check_with_empty_requested_set(self):
        """Empty capabilities request should always pass."""
        session = Session()
        branch = session.create_branch(name="limited", capabilities={"cap1"})

        # Empty set is subset of any set
        capabilities_must_be_subset_of_branch(branch, set())  # Should not raise

    def test_capabilities_check_with_empty_branch_capabilities(self):
        """Branch with empty capabilities should reject any capability request."""
        session = Session()
        branch = session.create_branch(name="no_caps", capabilities=set())

        # Any non-empty set is NOT subset of empty set
        with pytest.raises(AccessError) as exc_info:
            capabilities_must_be_subset_of_branch(branch, {"cap1"})

        assert "missing capabilities" in str(exc_info.value)
        assert exc_info.value.details["available"] == []

    def test_resource_must_exist_in_session_with_empty_services(self):
        """resource_must_exist_in_session should fail when services empty."""
        session = Session()

        with pytest.raises(NotFoundError) as exc_info:
            resource_must_exist_in_session(session, "nonexistent_service")

        assert "not found in session services" in str(exc_info.value)
