# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Work system - Declarative workflow orchestration.

Two complementary patterns at different abstraction levels:

**Report** (artifact state):
    Declarative workflow definition via form_assignments DSL.
    Tracks one specific job's progress through the workflow.
    Dependencies implicit from field names.

    class HiringBriefReport(Report):
        role_classification: RoleClassification | None = None
        strategic_context: StrategicContext | None = None

        assignment: str = "job_input -> executive_summary"

        form_assignments: list[str] = [
            "classifier: job_input -> role_classification | api:fast",
            "strategist: job_input, role_classification -> strategic_context | api:synthesis",
        ]

**Worker** (execution capability):
    Functional station that can execute forms.
    Has internal DAG for retries/error handling.
    Matches to forms via resource hints.

    class ClassifierWorker(Worker):
        @work(assignment="job_input -> role_classification")
        async def classify(self, job_input, **kwargs):
            return await self.llm.chat(**kwargs)

Core concepts:
- Phrase: Typed operation signature (inputs -> outputs)
- Form: Data binding + scheduling (stateful artifact)
- Report: Multi-step workflow declaration (stateful artifact)
- Worker: Execution capability (stateless station)
- WorkerEngine: Execution driver
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # engine
    "WorkerEngine": ("krons.work.engine", "WorkerEngine"),
    "WorkerTask": ("krons.work.engine", "WorkerTask"),
    # form
    "Form": ("krons.work.form", "Form"),
    "ParsedAssignment": ("krons.work.form", "ParsedAssignment"),
    "parse_assignment": ("krons.work.form", "parse_assignment"),
    "parse_full_assignment": ("krons.work.form", "parse_full_assignment"),
    # phrase
    "CrudOperation": ("krons.work.phrase", "CrudOperation"),
    "CrudPattern": ("krons.work.phrase", "CrudPattern"),
    "Phrase": ("krons.work.phrase", "Phrase"),
    "phrase": ("krons.work.phrase", "phrase"),
    # report
    "Report": ("krons.work.report", "Report"),
    # worker
    "Worker": ("krons.work.worker", "Worker"),
    "WorkConfig": ("krons.work.worker", "WorkConfig"),
    "WorkLink": ("krons.work.worker", "WorkLink"),
    "work": ("krons.work.worker", "work"),
    "worklink": ("krons.work.worker", "worklink"),
}

_LOADED: dict[str, object] = {}


def __getattr__(name: str) -> object:
    """Lazy import attributes on first access."""
    if name in _LOADED:
        return _LOADED[name]

    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        _LOADED[name] = value
        return value

    raise AttributeError(f"module 'krons.work' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return all available attributes for autocomplete."""
    return list(__all__)


# TYPE_CHECKING block for static analysis
if TYPE_CHECKING:
    from krons.work.engine import WorkerEngine, WorkerTask
    from krons.work.form import (
        Form,
        ParsedAssignment,
        parse_assignment,
        parse_full_assignment,
    )
    from krons.work.phrase import CrudOperation, CrudPattern, Phrase, phrase
    from krons.work.report import Report
    from krons.work.worker import WorkConfig, Worker, WorkLink, work, worklink

__all__ = (
    "CrudOperation",
    "CrudPattern",
    "Form",
    "ParsedAssignment",
    "Phrase",
    "Report",
    "WorkConfig",
    "WorkLink",
    "Worker",
    "WorkerEngine",
    "WorkerTask",
    "parse_assignment",
    "parse_full_assignment",
    "phrase",
    "work",
    "worklink",
)
