# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.core.node - Node with polymorphic content."""

import pytest
from pydantic import BaseModel

from krons.core import Node
from krons.core.node import NODE_REGISTRY, create_node

# Create Node subclass with embedding support for embedding tests
EmbeddableNode = create_node("EmbeddableNode", embedding_enabled=True, embedding_dim=4)


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    value: int


class TestNodeCreation:
    """Test Node instantiation with various content types."""

    def test_node_with_dict(self):
        """Node should accept dict content."""
        node = Node(content={"key": "value", "nested": {"a": 1}})
        assert node.content["key"] == "value"
        assert node.content["nested"]["a"] == 1

    def test_node_with_basemodel(self):
        """Node should accept BaseModel content."""
        model = SampleModel(name="test", value=42)
        node = Node(content=model)
        assert node.content.name == "test"
        assert node.content.value == 42

    def test_node_content_none(self):
        """Node should accept None content."""
        node = Node(content=None)
        assert node.content is None

    def test_node_content_empty_dict(self):
        """Node should accept empty dict content."""
        node = Node(content={})
        assert node.content == {}

    def test_node_invalid_content_type(self):
        """Node should reject invalid content types."""
        with pytest.raises(TypeError):
            Node(content="string is not allowed")

        with pytest.raises(TypeError):
            Node(content=123)

        with pytest.raises(TypeError):
            Node(content=[1, 2, 3])


class TestNodeEmbedding:
    """Test Node embedding support via create_node factory."""

    def test_node_with_embedding(self):
        """EmbeddableNode should accept embedding list."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        node = EmbeddableNode(content={"value": "test"}, embedding=embedding)
        assert node.embedding == embedding

    def test_node_embedding_none(self):
        """EmbeddableNode should default embedding to None."""
        node = EmbeddableNode(content={"value": "test"})
        assert node.embedding is None

    def test_node_embedding_rejects_string(self):
        """EmbeddableNode should reject string embedding (must be list)."""
        Node3D = create_node("Node3D", embedding_enabled=True, embedding_dim=3)
        with pytest.raises(Exception):  # Pydantic ValidationError
            Node3D(content={"value": "test"}, embedding="[0.1, 0.2, 0.3]")

    def test_node_embedding_coerces_int_to_float(self):
        """EmbeddableNode should coerce int values to float in embedding."""
        # Create node with dim=3 for this test
        Node3D = create_node("Node3DInt", embedding_enabled=True, embedding_dim=3)
        node = Node3D(content={"value": "test"}, embedding=[1, 2, 3])
        assert node.embedding == [1.0, 2.0, 3.0]
        assert all(isinstance(v, float) for v in node.embedding)

    def test_node_embedding_accepts_empty_list(self):
        """EmbeddableNode accepts empty embedding (no dimension validation at runtime)."""
        # Empty list is valid from Pydantic's perspective (list[float])
        # Dimension validation is DDL-time, not runtime
        node = EmbeddableNode(content={"value": "test"}, embedding=[])
        assert node.embedding == []


class TestNodeSerialization:
    """Test Node serialization."""

    def test_node_to_dict(self):
        """Node.to_dict() should serialize content."""
        node = Node(content={"key": "value"})
        data = node.to_dict()

        assert "content" in data
        assert data["content"]["key"] == "value"
        assert "id" in data
        assert "created_at" in data

    def test_node_to_dict_with_embedding(self):
        """EmbeddableNode.to_dict() should serialize embedding."""
        Node3D = create_node("Node3DSer", embedding_enabled=True, embedding_dim=3)
        node = Node3D(content={"key": "value"}, embedding=[0.1, 0.2, 0.3])
        data = node.to_dict()

        assert data["embedding"] == [0.1, 0.2, 0.3]

    def test_node_to_dict_json_mode(self):
        """Node.to_dict(mode='json') should produce JSON-serializable dict."""
        node = Node(content={"key": "value"})
        data = node.to_dict(mode="json")

        assert isinstance(data["id"], str)
        assert isinstance(data["created_at"], str)

    def test_node_roundtrip(self):
        """Serialization roundtrip should preserve content."""
        Node3DRT = create_node("Node3DRT", embedding_enabled=True, embedding_dim=3)
        node = Node3DRT(
            content={"key": "value", "nested": {"deep": [1, 2, 3]}},
            embedding=[0.1, 0.2, 0.3],
        )
        data = node.to_dict(mode="json")
        restored = Node3DRT.from_dict(data)

        assert restored.id == node.id
        assert restored.content["key"] == "value"
        assert restored.content["nested"]["deep"] == [1, 2, 3]
        assert restored.embedding == [0.1, 0.2, 0.3]

    def test_node_roundtrip_with_basemodel(self):
        """Serialization roundtrip should handle BaseModel content."""
        model = SampleModel(name="test", value=42)
        node = Node(content=model)
        data = node.to_dict(mode="json")
        restored = Node.from_dict(data)

        assert restored.id == node.id
        # Content is serialized as dict
        assert restored.content["name"] == "test"
        assert restored.content["value"] == 42

    def test_node_to_dict_db_mode(self):
        """Node.to_dict(mode='db') should rename metadata."""
        node = Node(content={"key": "value"})
        data = node.to_dict(mode="db")

        assert "node_metadata" in data
        assert "metadata" not in data


class TestNodeDBSerialization:
    """Test Node DB mode serialization with content flattening."""

    def test_to_dict_db_mode_flattens_typed_content(self):
        """to_dict(mode='db') should flatten typed BaseModel content."""

        class JobContent(BaseModel):
            title: str
            salary: int

        JobNode = create_node(
            "JobNode",
            content=JobContent,
            flatten_content=True,
        )

        job = JobNode(content=JobContent(title="Engineer", salary=100000))
        data = job.to_dict(mode="db")

        # Content fields should be spread into result
        assert "title" in data
        assert data["title"] == "Engineer"
        assert "salary" in data
        assert data["salary"] == 100000
        # "content" key should NOT be present
        assert "content" not in data

    def test_to_dict_db_mode_no_flatten_for_dict_content(self):
        """to_dict(mode='db') should NOT flatten generic dict content."""
        # Node without typed content model
        node = Node(content={"key": "value", "nested": {"a": 1}})
        data = node.to_dict(mode="db")

        # Content should remain as "content" field, not flattened
        assert "content" in data
        assert data["content"]["key"] == "value"
        # Individual keys should NOT be at top level
        assert "key" not in data

    def test_from_dict_from_row_reconstructs_content(self):
        """from_dict(from_row=True) should reconstruct content from flattened data."""

        class TaskContent(BaseModel):
            name: str
            priority: int

        TaskNode = create_node(
            "TaskNode",
            content=TaskContent,
            flatten_content=True,
        )

        # Create and serialize to DB format
        task = TaskNode(content=TaskContent(name="Review PR", priority=1))
        db_data = task.to_dict(mode="db")

        # Verify flattening worked
        assert "name" in db_data
        assert "content" not in db_data

        # Reconstruct from row
        restored = TaskNode.from_dict(db_data, from_row=True)

        assert restored.content is not None
        assert isinstance(restored.content, TaskContent)
        assert restored.content.name == "Review PR"
        assert restored.content.priority == 1

    def test_from_dict_from_row_roundtrip(self):
        """Complete roundtrip: to_dict(mode='db') -> from_dict(from_row=True)."""

        class ArticleContent(BaseModel):
            headline: str
            body: str
            views: int = 0

        ArticleNode = create_node(
            "ArticleNode",
            content=ArticleContent,
            flatten_content=True,
            embedding_enabled=True,
            embedding_dim=3,
        )

        original = ArticleNode(
            content=ArticleContent(headline="Breaking News", body="Details here", views=42),
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test"},
        )

        # Serialize to DB format
        db_data = original.to_dict(mode="db")

        # Reconstruct
        restored = ArticleNode.from_dict(db_data, from_row=True)

        assert restored.id == original.id
        assert restored.content.headline == "Breaking News"
        assert restored.content.body == "Details here"
        assert restored.content.views == 42
        assert restored.embedding == [0.1, 0.2, 0.3]


class TestGetFKDependencies:
    """Test get_fk_dependencies function."""

    def test_get_fk_dependencies_no_content(self):
        """Nodes without content should return empty set."""
        from krons.core.node import get_fk_dependencies

        SimpleNode = create_node("SimpleNode", flatten_content=False)
        deps = get_fk_dependencies(SimpleNode)
        assert deps == set()

    def test_get_fk_dependencies_no_fk_fields(self):
        """Nodes with content but no FK fields should return empty set."""
        from krons.core.node import get_fk_dependencies

        class PlainContent(BaseModel):
            name: str
            value: int

        PlainNode = create_node("PlainNode", content=PlainContent)
        deps = get_fk_dependencies(PlainNode)
        assert deps == set()

    def test_get_fk_dependencies_with_fk_fields(self):
        """Nodes with FK fields should return referenced table names."""
        from typing import Annotated

        from krons.core.node import get_fk_dependencies
        from krons.types.db_types import FK

        # Create a target node type
        class UserContent(BaseModel):
            username: str

        UserNode = create_node("UserNode", content=UserContent, table_name="users")

        # Create a node that references UserNode
        class PostContent(BaseModel):
            title: str
            author_id: Annotated[str, FK[UserNode]]

        PostNode = create_node("PostNode", content=PostContent, table_name="posts")

        deps = get_fk_dependencies(PostNode)
        assert "users" in deps


class TestNodeLifecycle:
    """Test Node lifecycle methods."""

    def test_soft_delete(self):
        """Test soft_delete() method."""
        from uuid import uuid4

        SoftDeleteNode = create_node(
            "SoftDeleteNode",
            soft_delete=True,
            track_deleted_by=True,
        )

        node = SoftDeleteNode(content={"value": "test"})
        assert node.is_deleted is False
        assert node.deleted_at is None

        # Soft delete
        user_id = uuid4()
        node.soft_delete(by=user_id)
        assert node.is_deleted is True
        assert node.deleted_at is not None
        assert node.deleted_by == user_id

    def test_soft_delete_requires_config(self):
        """soft_delete() should fail if not enabled in config."""
        node = Node(content={"value": "test"})
        with pytest.raises(RuntimeError, match="does not support soft_delete"):
            node.soft_delete()

    def test_activate_and_deactivate(self):
        """Test activate() and deactivate() methods."""
        from uuid import uuid4

        ActiveNode = create_node(
            "ActiveNode",
            track_is_active=True,
        )

        node = ActiveNode(content={"value": "test"})
        assert node.is_active is True  # Default

        # Deactivate
        user_id = uuid4()
        node.deactivate(by=user_id)
        assert node.is_active is False

        # Activate
        another_id = uuid4()
        node.activate(by=another_id)
        assert node.is_active is True

    def test_activate_requires_config(self):
        """activate() should fail if not enabled in config."""
        node = Node(content={"value": "test"})
        with pytest.raises(RuntimeError, match="does not support activate"):
            node.activate()

    def test_versioning(self):
        """Test version tracking."""
        VersionedNode = create_node(
            "VersionedNode",
            versioning=True,
        )

        node = VersionedNode(content={"value": "test"})
        assert node.version == 1

        # touch() increments version
        node.touch()
        assert node.version == 2

        node.touch()
        assert node.version == 3

    def test_content_hashing(self):
        """Test content hash computation."""
        HashedNode = create_node(
            "HashedNode",
            content_hashing=True,
        )

        node = HashedNode(content={"value": "test"})
        # rehash() computes and stores hash
        hash1 = node.rehash()
        assert hash1 is not None
        assert node.content_hash == hash1

        # Same content = same hash
        hash2 = node.rehash()
        assert hash1 == hash2


class TestNodeRegistry:
    """Test Node polymorphic registry."""

    def test_node_in_registry(self):
        """Node should be registered in NODE_REGISTRY."""
        assert "Node" in NODE_REGISTRY
        assert NODE_REGISTRY["Node"] is Node

    def test_subclass_auto_registered(self):
        """Node subclasses with polymorphic=True should auto-register."""
        from krons.core.node import NodeConfig

        class CustomNode(Node):
            node_config = NodeConfig(polymorphic=True)
            custom_field: str = "default"

        # Registry uses full class name (kron_class pattern)
        full_name = CustomNode.class_name(full=True)
        assert full_name in NODE_REGISTRY
        assert NODE_REGISTRY[full_name] is CustomNode


class TestNodeIntegrityHashing:
    """Tests for Node.update_integrity_hash()."""

    def test_integrity_hash_chain(self):
        """Test integrity hash creates tamper-evident chain."""
        IntegrityNode = create_node(
            "IntegrityNode",
            content=dict,
            integrity_hashing=True,
        )
        node = IntegrityNode(content={"data": "test"})

        # First hash (no previous)
        hash1 = node.update_integrity_hash()
        assert hash1 is not None

        # Chain hash (with previous)
        hash2 = node.update_integrity_hash(previous_hash=hash1)
        assert hash2 is not None
        assert hash1 != hash2  # Chain progression

    def test_integrity_hash_disabled_returns_none(self):
        """Test update_integrity_hash returns None when disabled."""
        node = Node(content={"data": "test"})
        result = node.update_integrity_hash()
        assert result is None


class TestNodeRestore:
    """Tests for Node.restore() method."""

    def test_restore_after_soft_delete(self):
        """Test restore() reverses soft_delete()."""
        from uuid import uuid4

        SoftDeleteRestoreNode = create_node(
            "SoftDeleteRestoreNode",
            content=dict,
            soft_delete=True,
        )
        node = SoftDeleteRestoreNode(content={"data": "test"})
        actor_id = uuid4()

        # Soft delete
        node.soft_delete(by=actor_id)
        assert node.is_deleted is True
        assert node.deleted_at is not None

        # Restore
        node.restore(by=actor_id)
        assert node.is_deleted is False
        assert node.deleted_at is None


class TestNodeDDLGeneration:
    """Tests for Node DDL generation."""

    def test_generate_ddl_basic(self):
        """Test generate_ddl produces valid SQL."""
        from krons.core.node import generate_ddl

        class JobContent(BaseModel):
            title: str
            salary: int

        JobDDLNode = create_node(
            "JobDDLNode",
            content=JobContent,
            flatten_content=True,
            table_name="jobs",
        )

        ddl = generate_ddl(JobDDLNode)
        assert "CREATE TABLE" in ddl
        assert "jobs" in ddl
        assert "title" in ddl  # Flattened field
        assert "salary" in ddl  # Flattened field

    def test_generate_ddl_with_fk(self):
        """Test generate_ddl includes FK constraints."""
        from krons.core.node import generate_ddl
        from krons.types.db_types import FK

        # Create a target node type for FK reference
        class UserContent(BaseModel):
            username: str

        UserDDLNode = create_node(
            "UserDDLNode",
            content=UserContent,
            table_name="users",
        )

        class PostContent(BaseModel):
            title: str
            # FK[Model] expands to Annotated[UUID, FKMeta(Model)]
            author_id: FK[UserDDLNode]

        PostDDLNode = create_node(
            "PostDDLNode",
            content=PostContent,
            flatten_content=True,
            table_name="posts",
        )

        ddl = generate_ddl(PostDDLNode)
        assert "FOREIGN KEY" in ddl or "REFERENCES" in ddl
