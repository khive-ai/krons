# krons - Spec-Based Composable Framework

## Overview

**krons** is a Python framework for building spec-based, composable systems. It provides:

- **Spec/Operable**: Type-safe field definitions with validation, defaults, and DB metadata
- **Node**: Polymorphic content containers with DB serialization
- **Services**: Unified service interfaces (iModel, KronService) with hooks and rate limiting
- **Enforcement**: Policy evaluation and action handlers with typed I/O

## Architecture

```
krons/
├── core/           # Foundation: Element, Node, Event, Flow, Graph, Pile
├── specs/          # Spec definitions, Operable composition, adapters
│   ├── catalog/    # Pre-built specs (Content, Audit, Common, Enforcement)
│   └── adapters/   # Pydantic, SQL DDL, Dataclass adapters
├── services/       # Service backends, iModel, hooks, rate limiting
├── enforcement/    # Policy protocols, KronService, action decorators
├── types/          # Base types, sentinels, DB types (FK, Vector)
├── operations/     # Operation builders and registry
├── protocols.py    # Runtime-checkable protocols with @implements
└── utils/          # Fuzzy matching, SQL utilities, helpers
```

## Key Patterns

### 1. Spec & Operable

**Spec** defines a single field with type, name, default, validation, and DB metadata:

```python
from krons.specs import Spec, Operable
from krons.types.db_types import FK, VectorMeta

# Basic specs
name_spec = Spec(str, name="name")
count_spec = Spec(int, name="count", default=0, ge=0)

# With DB metadata
user_id = Spec(UUID, name="user_id", as_fk=FK[User])
embedding = Spec(list[float], name="embedding", embedding=VectorMeta(1536))

# Nullable
email = Spec(str, name="email").as_nullable()
```

**Operable** composes multiple Specs into structures:

```python
# From specs
operable = Operable([name_spec, count_spec, email])

# From Pydantic BaseModel
operable = Operable.from_structure(MyModel)

# Generate typed structures
MyDataclass = operable.compose_structure("MyDataclass")
specs_list = operable.get_specs()
```

### 2. Catalog Specs (BaseModel Pattern)

Pre-built specs use BaseModel for field definitions:

```python
from krons.specs.catalog import ContentSpecs, AuditSpecs, CommonSpecs

# Get specs with customization
content_specs = ContentSpecs.get_specs(dim=1536)  # With vector dimension
audit_specs = AuditSpecs.get_specs(use_uuid=True)  # UUID actor IDs
common_specs = CommonSpecs.get_specs(status_default="pending")
```

**Pattern for catalog specs:**

```python
class MySpecs(BaseModel):
    field1: str
    field2: int = 0
    field3: datetime = Field(default_factory=now_utc)

    @classmethod
    def get_specs(cls, **overrides) -> list[Spec]:
        operable = Operable.from_structure(cls)
        specs = {s.name: s for s in operable.get_specs()}
        # Apply overrides...
        return list(specs.values())
```

### 3. Node (Polymorphic Content)

**Node** stores polymorphic content with DB serialization:

```python
from krons.core import Node
from krons.core.node import create_node, NodeConfig

# Basic usage
node = Node(content={"key": "value"})
data = node.to_dict(mode="json")  # For JSON serialization
db_data = node.to_dict(mode="db")  # For database (renames metadata)

# Custom node with typed content
class JobContent(BaseModel):
    title: str
    salary: int

JobNode = create_node(
    "JobNode",
    content=JobContent,
    flatten_content=True,  # Spreads content fields in DB mode
    embedding_enabled=True,
    embedding_dim=1536,
    soft_delete=True,
    versioning=True,
)

# DB roundtrip
job = JobNode(content=JobContent(title="Engineer", salary=100000))
db_data = job.to_dict(mode="db")  # {"title": "Engineer", "salary": 100000, ...}
restored = JobNode.from_dict(db_data, from_row=True)  # Reconstructs content
```

### 4. Services (iModel, KronService)

**iModel** - Unified service interface with rate limiting:

```python
from krons.services import Endpoint, EndpointConfig, iModel

config = EndpointConfig(
    name="gpt-4",
    provider="openai",
    endpoint="chat/completions",
    base_url="https://api.openai.com/v1",
    api_key="...",
)
endpoint = Endpoint(config=config)
model = iModel(backend=endpoint)

response = await model.invoke({"messages": [...]})
```

**KronService** - Action handlers with policy evaluation:

```python
from krons.enforcement import KronService, KronConfig, action, RequestContext

class MyService(KronService):
    @property
    def event_type(self):
        return Calling  # Required abstract property

    @action(name="user.create", inputs={"name", "email"}, outputs={"user_id"})
    async def _handle_create(self, options, ctx):
        return {"user_id": uuid4()}

service = MyService(config=KronConfig(provider="my", name="service"))
result = await service.call("user.create", {"name": "John"}, RequestContext(name="user.create"))
```

### 5. Protocols with @implements

Runtime-checkable protocols with signature validation:

```python
from krons.protocols import implements, Serializable, SignatureMismatchError

@implements(Serializable, signature_check="error")  # "error", "warn", "skip"
class MyClass:
    def to_dict(self, **kwargs):  # Must match protocol signature
        return {"data": ...}
```

### 6. DB Types (FK, Vector)

Foreign keys and vector embeddings for SQL DDL:

```python
from krons.types.db_types import FK, Vector, FKMeta, VectorMeta, extract_kron_db_meta

# In type annotations
class Post(BaseModel):
    author_id: FK[User]  # Expands to Annotated[UUID, FKMeta(User)]
    embedding: Vector[1536]  # Expands to Annotated[list[float], VectorMeta(1536)]

# Extract metadata
fk_meta = extract_kron_db_meta(field_info, metas="FK")
vec_meta = extract_kron_db_meta(field_info, metas="Vector")
```

## Testing Patterns

### Test Structure

```
tests/
├── core/           # Node, Element, Event tests
├── specs/          # Spec, Operable, Catalog tests
├── services/       # iModel, hook tests
├── enforcement/    # KronService, policy tests
└── utils/          # Utility function tests
```

### Key Test Utilities

```python
import pytest

# Async tests
@pytest.mark.anyio
async def test_async_operation():
    result = await some_async_call()
    assert result == expected

# Testing abstract classes (provide required implementations)
class TestService(KronService):
    @property
    def event_type(self):
        return Calling  # Satisfy abstract property

# Mock policy engine/resolver (they're Protocols)
class MockPolicyEngine:
    async def evaluate(self, policy_id, input_data, **options):
        return {}
    async def evaluate_batch(self, policy_ids, input_data, **options):
        return []
```

## Common Gotchas

1. **Circular imports in catalog**: Use direct imports from submodules:
   ```python
   # Wrong
   from krons.specs import Operable, Spec

   # Right (in catalog files)
   from krons.specs.operable import Operable
   from krons.specs.spec import Spec
   ```

2. **PolicyEngine/PolicyResolver are Protocols**: Can't instantiate directly, create mock classes.

3. **Node content flattening**: Only works with typed BaseModel content, not generic dicts.

4. **Spec base_type for lists**: `list[float]` becomes `float` with `is_listable=True`.

5. **compose_structure frozen param**: Currently broken in PydanticSpecAdapter (doesn't accept
   `frozen` kwarg).

## Running Tests

```bash
cd libs/krons

# All tests
uv run pytest tests/ -q

# With coverage
uv run pytest tests/ --cov=krons --cov-report=term-missing

# Specific module
uv run pytest tests/specs/test_catalog.py -v

# Single test
uv run pytest tests/core/test_node.py::TestNodeCreation::test_node_with_dict -v
```

## Code Style

- Python 3.11+ with type hints
- Pydantic v2 for models
- anyio for async (not asyncio directly)
- ruff for linting (line-length=100)
- pytest with anyio plugin for async tests

## File Naming Conventions

- `_internal.py` - Private module internals
- `catalog/_*.py` - Catalog spec definitions
- `adapters/*.py` - Framework adapters (Pydantic, SQL, etc.)
- `test_*.py` - Test files mirror source structure
