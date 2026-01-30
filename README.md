# krons

Spec-based composable framework for building type-safe systems.

## Installation

```bash
pip install krons
```

## Features

- **Spec/Operable**: Type-safe field definitions with validation, defaults, and DB metadata
- **Node**: Polymorphic content containers with DB serialization
- **Services**: Unified service interfaces (iModel) with hooks and rate limiting
- **Enforcement**: Policy evaluation and action handlers with typed I/O
- **Protocols**: Runtime-checkable protocols with `@implements` decorator

## Quick Start

```python
from kronos.specs import Spec, Operable

# Define specs
name_spec = Spec(str, name="name")
count_spec = Spec(int, name="count", default=0, ge=0)

# Compose into structure
operable = Operable([name_spec, count_spec])
MyModel = operable.compose_structure("MyModel")
```

## Requirements

- Python 3.11+
- pydantic 2.x
- anyio
- httpx
- orjson

## License

Apache-2.0
