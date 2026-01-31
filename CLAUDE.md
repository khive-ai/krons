# krons - Spec-Based Composable Framework

## Overview

**krons** is a Python framework for building spec-based, composable systems. It provides:

- **Spec/Operable**: Type-safe field definitions with validation, defaults, and DB metadata
- **Node**: Polymorphic content containers with DB serialization
- **Services**: Unified service interfaces (iModel, KronService) with hooks and rate limiting
- **Work**: Declarative workflow orchestration with Report/Worker pattern
- **Rules**: Validation rules and enforcement

## Architecture

```
krons/
├── core/           # Foundation: Element, Node, Event, Flow, Graph, Pile
├── specs/          # Spec definitions, Operable composition, adapters
│   ├── catalog/    # Pre-built specs (Content, Audit, Common, Enforcement)
│   └── adapters/   # Pydantic, SQL DDL, Dataclass adapters
├── work/           # Workflow orchestration (Report, Worker, Form, Phrase)
├── services/       # Service backends, iModel, hooks, rate limiting
├── operations/     # Operation builders, context, registry
├── rules/          # Validation rules, registry, common validators
├── types/          # Base types, sentinels, DB types (FK, Vector)
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

### 7. Work (Workflow Orchestration)

Two complementary patterns at different abstraction levels:

**Report** (artifact state) - Declarative workflow definition:

```python
from krons.work import Report

class HiringBriefReport(Report):
    """Multi-step workflow with typed outputs as class attributes."""

    # Typed output fields
    role_classification: RoleClassification | None = None
    strategic_context: StrategicContext | None = None
    executive_summary: ExecutiveSummary | None = None

    # Overall contract
    assignment: str = "job_input, market_context -> executive_summary"

    # Form assignments with branch/resource hints
    # DSL: "branch: inputs -> outputs | resource"
    form_assignments: list[str] = [
        # Same branch = sequential execution
        "classifier: job_input -> role_classification | api:fast",
        "classifier: role_classification -> extracted_skills | api:fast",

        # Different branches = parallel execution
        "strategist: job_input, role_classification -> strategic_context | api:synthesis",
        "writer: strategic_context -> executive_summary | api:reasoning",
    ]

# Execute workflow
report = HiringBriefReport()
report.initialize(job_input="...", market_context="...")

while not report.is_complete():
    for form in report.next_forms():  # Data-driven scheduling
        result = await execute_form(form)  # Route to Worker via form.resource
        form.set_output(result)
        report.complete_form(form)

output = report.get_deliverable()
```

**Worker** (execution capability) - Functional station with @work methods:

```python
from krons.work import Worker, WorkerEngine, work, worklink

class ClassifierWorker(Worker):
    """Execution capability with internal DAG for retries."""

    name = "classifier"

    @work(assignment="job_input -> role_classification", capacity=2)
    async def classify_role(self, job_input, **kwargs):
        result = await self.llm.chat(**kwargs)
        return result.role_classification

    @work(assignment="code -> execution_result")
    async def execute_code(self, code):
        error = run_code(code)
        return code, error

    # Conditional edge for retry loops
    @worklink(from_="execute_code", to_="debug_code")
    async def maybe_debug(self, result):
        code, error = result
        if error:  # Only follow if error
            return {"code": code, "error": error}
        return None  # Skip edge

# Execute with engine
engine = WorkerEngine(worker=ClassifierWorker())
task = await engine.add_task(task_function="classify_role", job_input="...")
await engine.execute()
```

**Key concepts:**

| Component | Role | State |
|-----------|------|-------|
| **Report** | Work order / artifact | Stateful (tracks one job) |
| **Worker** | Station / capability | Stateless (handles many jobs) |
| **Form** | Unit of work | Stateful (one assignment) |
| **Phrase** | Typed I/O signature | Definition only |

**Form assignment DSL:**

```
"branch: inputs -> outputs | resource"

Examples:
  "a, b -> c"                           # Simple (no branch, no resource)
  "classifier: job -> role | api:fast"  # Full (branch + resource)
  "writer: context -> summary"          # Branch only
```

- **branch**: Groups forms for sequential execution (same branch = sequential)
- **resource**: Hint for routing to Worker capabilities (e.g., `api:fast`, `api:reasoning`)
- Forms without branch execute in parallel based on data availability

## Testing Patterns

### Test Structure

```
tests/
├── core/           # Node, Element, Event tests
├── specs/          # Spec, Operable, Catalog tests
├── work/           # Report, Worker, Form, Phrase, Engine tests
├── services/       # iModel, hook tests
├── operations/     # Operation, context tests
├── rules/          # Validation rule tests
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

6. **Report vs Worker**: They operate at different levels:
   - Report = declarative workflow (WHAT to do) - subclass and define `form_assignments`
   - Worker = execution capability (HOW to do it) - has `@work` methods
   - Forms without explicit branch run in parallel; same branch = sequential

7. **Form assignment DSL**: The full format is `"branch: inputs -> outputs | resource"`.
   All parts except `inputs -> outputs` are optional.

## Running Tests

```bash
# All tests
uv run pytest tests/ -q

# With coverage
uv run pytest tests/ --cov=krons --cov-report=term-missing

# Specific module
uv run pytest tests/work/test_report.py -v
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
