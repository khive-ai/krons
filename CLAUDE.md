# krons - Spec-Based Composable Framework

## Quick Reference

```text
krons/
├── core/           # Element, Node, Flow, Graph, Pile, Event
├── session/        # Session, Branch, Message (conversation orchestration)
├── resource/       # iModel, Endpoint, HookRegistry (API backends)
├── agent/          # Operations pipeline (generate → parse → structure → operate → react)
├── work/           # Report, Worker, Form (workflow orchestration)
└── utils/          # JSON, concurrency, display helpers
```

## Core Primitives

### Element — Base identity class

```python
from krons.core import Element

class MyEntity(Element):
    name: str

e = MyEntity(name="test")
e.id          # UUID (frozen)
e.created_at  # datetime (frozen)
e.to_dict(mode="json")  # Serialization with kron_class for polymorphic restore
Element.from_dict(data)  # Polymorphic deserialization
```

### Pile — Thread-safe typed collection

```python
from krons.core import Pile

pile = Pile[MyEntity](item_type={MyEntity})
pile.add(entity)
pile.get(uuid_or_str)
pile[0]                    # By index
pile[lambda x: x.name]     # Filter
async with pile:           # Async lock
    ...
```

### Node — Polymorphic content container

```python
from krons.core import Node, create_node

# Dynamic node
node = Node(content={"key": "value"})

# Typed node with DB features
MyNode = create_node("MyNode", content=MyContent, flatten_content=True)
```

## Session — Conversation Orchestration

```python
from krons.session import Session, SessionConfig

session = Session(config=SessionConfig(
    default_branch_name="main",
    default_gen_model="gpt-4",
    log_persist_dir="./logs",      # Enable persistence (None = disabled)
    log_auto_save_on_exit=True,    # atexit handler
))

# Register model
session.resources.register(imodel)

# Execute operations
op = await session.conduct("generate", params=GenerateParams(...))
op = await session.conduct("structure", params=StructureParams(...))

# Access messages
session.messages              # Pile[Message]
session.dump()                # Sync dump full session to JSON
await session.adump()         # Async dump

# Restore from dump
restored = Session.from_dict(data)  # Then re-register resources
```

## Resource — API Backends

```python
from krons.resource import iModel, Endpoint, EndpointConfig

config = EndpointConfig(
    name="gpt-4",
    provider="openai",
    endpoint="chat/completions",
    base_url="https://api.openai.com/v1",
    api_key="...",
)
model = iModel(backend=Endpoint(config=config))
calling = await model.invoke(messages=[...])
```

## Agent Operations

Pipeline: `generate → parse → structure → operate → react`

```python
from krons.agent.operations import GenerateParams, StructureParams

# Generate raw LLM response
op = await session.conduct("generate", params=GenerateParams(
    primary="Your prompt",
    imodel="gpt-4",
))

# Structured output with validation
op = await session.conduct("structure", params=StructureParams(
    generate_params=GenerateParams(primary="...", request_model=MyModel),
    validator=Validator(),
    operable=Operable.from_structure(MyModel),
))
```

## Work — Workflow Orchestration

### Report (declarative workflow)

```python
from krons.work import Report

class MyReport(Report):
    result: str | None = None
    assignment: str = "input -> result"
    form_assignments: list[str] = [
        "step1: input -> intermediate",
        "step2: intermediate -> result",
    ]
```

### Worker (execution capability)

```python
from krons.work import Worker, work

class MyWorker(Worker):
    name = "processor"

    @work(assignment="input -> output")
    async def process(self, input, **kwargs):
        return transform(input)
```

## Protocols

```python
from krons.protocols import implements, Serializable

@implements(Serializable, signature_check="error")
class MyClass:
    def to_dict(self, **kwargs): ...
```

## Testing

```bash
uv run pytest tests/ -q              # All tests
uv run pytest tests/core/ -xvs       # Core module verbose
uv run pytest -k "test_session" -v   # Pattern match
```

```python
import pytest

@pytest.mark.anyio
async def test_async():
    result = await some_call()
    assert result
```

## Code Style

- Python 3.11+, Pydantic v2, anyio
- `uv run ruff check src/ --fix && uv run ruff format src/`
- Type hints required on public APIs
