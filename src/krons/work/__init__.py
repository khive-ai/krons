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

from .engine import WorkerEngine, WorkerTask
from .form import Form, ParsedAssignment, parse_assignment, parse_full_assignment
from .phrase import CrudOperation, CrudPattern, Phrase, phrase
from .report import Report
from .worker import Worker, WorkConfig, WorkLink, work, worklink

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
