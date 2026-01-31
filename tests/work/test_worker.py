# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for krons.work.worker - declarative workflow definition via decorators."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from krons.work.form import Form
from krons.work.worker import WorkConfig, Worker, WorkLink, work, worklink

# =============================================================================
# Tests: WorkConfig
# =============================================================================


class TestWorkConfig:
    """Tests for WorkConfig dataclass."""

    def test_default_values(self):
        """WorkConfig should have sensible defaults."""
        config = WorkConfig()

        assert config.assignment == ""
        assert config.form_param_key == ""
        assert config.capacity == 1
        assert config.refresh_time == 0.1
        assert config.timeout is None

    def test_custom_values(self):
        """WorkConfig should accept custom values."""
        config = WorkConfig(
            assignment="a -> b",
            form_param_key="form_id",
            capacity=5,
            refresh_time=0.5,
            timeout=30.0,
        )

        assert config.assignment == "a -> b"
        assert config.form_param_key == "form_id"
        assert config.capacity == 5
        assert config.refresh_time == 0.5
        assert config.timeout == 30.0


# =============================================================================
# Tests: WorkLink
# =============================================================================


class TestWorkLink:
    """Tests for WorkLink dataclass."""

    def test_worklink_fields(self):
        """WorkLink should store from_, to_, handler_name."""
        link = WorkLink(
            from_="source_method",
            to_="target_method",
            handler_name="link_handler",
        )

        assert link.from_ == "source_method"
        assert link.to_ == "target_method"
        assert link.handler_name == "link_handler"


# =============================================================================
# Tests: @work decorator
# =============================================================================


class TestWorkDecorator:
    """Tests for @work decorator."""

    def test_work_attaches_config(self):
        """@work should attach WorkConfig to method."""

        @work(assignment="a -> b", capacity=3)
        async def my_method(self, **kwargs):
            return "result"

        assert hasattr(my_method, "_work_config")
        assert isinstance(my_method._work_config, WorkConfig)
        assert my_method._work_config.assignment == "a -> b"
        assert my_method._work_config.capacity == 3

    def test_work_default_config(self):
        """@work with no args should use default config."""

        @work()
        async def my_method(self, **kwargs):
            return "result"

        assert my_method._work_config.assignment == ""
        assert my_method._work_config.capacity == 1

    def test_work_preserves_function_name(self):
        """@work should preserve function __name__."""

        @work(assignment="a -> b")
        async def my_custom_method(self, **kwargs):
            return "result"

        assert my_custom_method.__name__ == "my_custom_method"

    def test_work_with_form_param_key(self):
        """@work should accept form_param_key."""

        @work(assignment="context -> code", form_param_key="form_id")
        async def write_code(self, form_id, **kwargs):
            return form_id

        assert write_code._work_config.form_param_key == "form_id"

    def test_work_with_timeout(self):
        """@work should accept timeout."""

        @work(timeout=60.0)
        async def slow_method(self, **kwargs):
            return "done"

        assert slow_method._work_config.timeout == 60.0

    @pytest.mark.anyio
    async def test_work_callable(self):
        """@work decorated method should be callable."""

        @work(assignment="a -> b")
        async def my_method(**kwargs):
            return kwargs.get("input", "default")

        result = await my_method(input="test")
        assert result == "test"


# =============================================================================
# Tests: @worklink decorator
# =============================================================================


class TestWorklinkDecorator:
    """Tests for @worklink decorator."""

    def test_worklink_attaches_info(self):
        """@worklink should attach from/to info to method."""

        @worklink(from_="source", to_="target")
        async def my_link(self, from_result):
            return {"next": from_result}

        assert hasattr(my_link, "_worklink_from")
        assert hasattr(my_link, "_worklink_to")
        assert my_link._worklink_from == "source"
        assert my_link._worklink_to == "target"

    def test_worklink_preserves_function_name(self):
        """@worklink should preserve function __name__."""

        @worklink(from_="a", to_="b")
        async def my_custom_link(self, from_result):
            return {}

        assert my_custom_link.__name__ == "my_custom_link"

    @pytest.mark.anyio
    async def test_worklink_callable(self):
        """@worklink decorated method should be callable."""

        @worklink(from_="a", to_="b")
        async def my_link(from_result):
            return {"transformed": from_result * 2}

        result = await my_link(10)
        assert result == {"transformed": 20}


# =============================================================================
# Tests: Worker Base Class
# =============================================================================


class TestWorkerBase:
    """Tests for Worker base class."""

    def test_worker_default_name(self):
        """Worker should have default name 'worker'."""

        class MyWorker(Worker):
            pass

        w = MyWorker()
        assert w.name == "worker"

    def test_worker_custom_name(self):
        """Worker should support custom name."""

        class MyWorker(Worker):
            name = "my_custom_worker"

        w = MyWorker()
        assert w.name == "my_custom_worker"

    def test_worker_forms_storage(self):
        """Worker should initialize empty forms dict."""

        class MyWorker(Worker):
            pass

        w = MyWorker()
        assert w.forms == {}
        assert isinstance(w.forms, dict)

    def test_worker_stopped_initially_false(self):
        """Worker should start with _stopped=False."""

        class MyWorker(Worker):
            pass

        w = MyWorker()
        assert w._stopped is False
        assert w.is_stopped() is False


# =============================================================================
# Tests: Worker Metadata Collection
# =============================================================================


class TestWorkerMetadataCollection:
    """Tests for Worker @work/@worklink metadata collection."""

    def test_collects_work_methods(self):
        """Worker should collect @work decorated methods."""

        class MyWorker(Worker):
            @work(assignment="a -> b", capacity=2)
            async def process(self, **kwargs):
                return "processed"

            @work(assignment="b -> c")
            async def transform(self, **kwargs):
                return "transformed"

        w = MyWorker()

        assert "process" in w._work_methods
        assert "transform" in w._work_methods
        assert len(w._work_methods) == 2

        # Verify config attached
        method, config = w._work_methods["process"]
        assert config.assignment == "a -> b"
        assert config.capacity == 2

    def test_collects_work_links(self):
        """Worker should collect @worklink decorated methods."""

        class MyWorker(Worker):
            @work()
            async def step1(self, **kwargs):
                return "done"

            @work()
            async def step2(self, **kwargs):
                return "done"

            @worklink(from_="step1", to_="step2")
            async def step1_to_step2(self, from_result):
                return {}

        w = MyWorker()

        assert len(w._work_links) == 1
        link = w._work_links[0]
        assert link.from_ == "step1"
        assert link.to_ == "step2"
        assert link.handler_name == "step1_to_step2"

    def test_collects_multiple_links(self):
        """Worker should collect multiple @worklink methods."""

        class MyWorker(Worker):
            @work()
            async def start(self, **kwargs):
                return "started"

            @work()
            async def branch_a(self, **kwargs):
                return "a"

            @work()
            async def branch_b(self, **kwargs):
                return "b"

            @worklink(from_="start", to_="branch_a")
            async def to_branch_a(self, from_result):
                return {"input": from_result}

            @worklink(from_="start", to_="branch_b")
            async def to_branch_b(self, from_result):
                return {"input": from_result}

        w = MyWorker()

        assert len(w._work_links) == 2

    def test_ignores_private_methods(self):
        """Worker should ignore private methods starting with _."""

        class MyWorker(Worker):
            @work()
            async def public_method(self, **kwargs):
                return "public"

            async def _private_method(self, **kwargs):
                return "private"

        w = MyWorker()

        assert "public_method" in w._work_methods
        assert "_private_method" not in w._work_methods


# =============================================================================
# Tests: Worker.get_links_from / get_links_to
# =============================================================================


class TestWorkerLinkQueries:
    """Tests for Worker link query methods."""

    def test_get_links_from(self):
        """get_links_from should return outgoing links."""

        class MyWorker(Worker):
            @work()
            async def step1(self, **kwargs):
                return 1

            @work()
            async def step2(self, **kwargs):
                return 2

            @work()
            async def step3(self, **kwargs):
                return 3

            @worklink(from_="step1", to_="step2")
            async def link_1_2(self, from_result):
                return {}

            @worklink(from_="step1", to_="step3")
            async def link_1_3(self, from_result):
                return {}

        w = MyWorker()

        links = w.get_links_from("step1")
        assert len(links) == 2

        targets = {link.to_ for link in links}
        assert targets == {"step2", "step3"}

    def test_get_links_from_empty(self):
        """get_links_from should return empty for terminal methods."""

        class MyWorker(Worker):
            @work()
            async def terminal(self, **kwargs):
                return "end"

        w = MyWorker()

        links = w.get_links_from("terminal")
        assert links == []

    def test_get_links_to(self):
        """get_links_to should return incoming links."""

        class MyWorker(Worker):
            @work()
            async def step1(self, **kwargs):
                return 1

            @work()
            async def step2(self, **kwargs):
                return 2

            @work()
            async def step3(self, **kwargs):
                return 3

            @worklink(from_="step1", to_="step3")
            async def link_1_3(self, from_result):
                return {}

            @worklink(from_="step2", to_="step3")
            async def link_2_3(self, from_result):
                return {}

        w = MyWorker()

        links = w.get_links_to("step3")
        assert len(links) == 2

        sources = {link.from_ for link in links}
        assert sources == {"step1", "step2"}

    def test_get_links_to_empty(self):
        """get_links_to should return empty for entry methods."""

        class MyWorker(Worker):
            @work()
            async def entry(self, **kwargs):
                return "start"

        w = MyWorker()

        links = w.get_links_to("entry")
        assert links == []


# =============================================================================
# Tests: Worker Start/Stop
# =============================================================================


class TestWorkerStartStop:
    """Tests for Worker start/stop control."""

    @pytest.mark.anyio
    async def test_stop_sets_flag(self):
        """stop should set _stopped to True."""

        class MyWorker(Worker):
            pass

        w = MyWorker()
        assert w.is_stopped() is False

        await w.stop()
        assert w.is_stopped() is True

    @pytest.mark.anyio
    async def test_start_clears_flag(self):
        """start should clear _stopped flag."""

        class MyWorker(Worker):
            pass

        w = MyWorker()
        w._stopped = True

        await w.start()
        assert w.is_stopped() is False


# =============================================================================
# Tests: Worker.__repr__
# =============================================================================


class TestWorkerRepr:
    """Tests for Worker string representation."""

    def test_repr_includes_methods(self):
        """repr should include work method names."""

        class MyWorker(Worker):
            @work()
            async def method1(self, **kwargs):
                return 1

            @work()
            async def method2(self, **kwargs):
                return 2

        w = MyWorker()
        repr_str = repr(w)

        assert "MyWorker" in repr_str
        assert "method1" in repr_str or "method2" in repr_str

    def test_repr_includes_link_count(self):
        """repr should include link count."""

        class MyWorker(Worker):
            @work()
            async def a(self, **kwargs):
                return 1

            @work()
            async def b(self, **kwargs):
                return 2

            @worklink(from_="a", to_="b")
            async def a_to_b(self, result):
                return {}

        w = MyWorker()
        repr_str = repr(w)

        assert "links=1" in repr_str

    def test_repr_includes_form_count(self):
        """repr should include form count."""

        class MyWorker(Worker):
            pass

        w = MyWorker()
        w.forms["form1"] = Form(assignment="a -> b")
        w.forms["form2"] = Form(assignment="c -> d")

        repr_str = repr(w)

        assert "forms=2" in repr_str


# =============================================================================
# Tests: Worker Form Storage
# =============================================================================


class TestWorkerFormStorage:
    """Tests for Worker form storage."""

    def test_store_form_by_string_key(self):
        """Worker should store forms by string key."""

        class MyWorker(Worker):
            pass

        w = MyWorker()
        form = Form(assignment="a -> b")

        w.forms["task1"] = form

        assert w.forms["task1"] is form

    def test_store_form_by_uuid_key(self):
        """Worker should store forms by UUID key."""

        class MyWorker(Worker):
            pass

        w = MyWorker()
        form = Form(assignment="a -> b")
        key = uuid4()

        w.forms[key] = form

        assert w.forms[key] is form

    def test_multiple_forms(self):
        """Worker should store multiple forms."""

        class MyWorker(Worker):
            pass

        w = MyWorker()

        for i in range(10):
            w.forms[f"task_{i}"] = Form(assignment=f"input_{i} -> output_{i}")

        assert len(w.forms) == 10


# =============================================================================
# Tests: Complete Worker Example
# =============================================================================


class TestCompleteWorkerExample:
    """Integration test with a complete Worker definition."""

    def test_complete_worker_definition(self):
        """Test a complete worker with multiple methods and links."""

        class FileCoder(Worker):
            name = "file_coder"

            @work(assignment="instruction, context -> code", capacity=2)
            async def write_code(self, form_id, **kwargs):
                return form_id, kwargs.get("code", "generated_code")

            @work(assignment="code -> result", form_param_key="form_id")
            async def execute_code(self, form_id, **kwargs):
                return form_id, None  # No error

            @work(assignment="code, error -> fixed_code", form_param_key="form_id")
            async def debug_code(self, form_id, **kwargs):
                return form_id, kwargs.get("fixed_code", "debugged_code")

            @worklink(from_="write_code", to_="execute_code")
            async def write_to_execute(self, from_result):
                form_id, code = from_result
                return {"form_id": form_id, "code": code}

            @worklink(from_="execute_code", to_="debug_code")
            async def execute_to_debug(self, from_result):
                form_id, error = from_result
                if error is not None:
                    return {"form_id": form_id, "error": error}
                return None  # No error, don't follow this link

        coder = FileCoder()

        # Verify structure
        assert coder.name == "file_coder"
        assert len(coder._work_methods) == 3
        assert len(coder._work_links) == 2

        # Verify method configs
        _, write_config = coder._work_methods["write_code"]
        assert write_config.capacity == 2

        _, exec_config = coder._work_methods["execute_code"]
        assert exec_config.form_param_key == "form_id"

        # Verify links
        from_write = coder.get_links_from("write_code")
        assert len(from_write) == 1
        assert from_write[0].to_ == "execute_code"

        from_execute = coder.get_links_from("execute_code")
        assert len(from_execute) == 1
        assert from_execute[0].to_ == "debug_code"

    @pytest.mark.anyio
    async def test_worker_method_invocation(self):
        """Test invoking worker methods directly."""

        class SimpleWorker(Worker):
            @work(assignment="input -> output")
            async def process(self, value, **kwargs):
                return value * 2

            @worklink(from_="process", to_="process")
            async def loop_link(self, from_result):
                if from_result < 100:
                    return {"value": from_result}
                return None  # Stop looping

        w = SimpleWorker()

        # Direct method invocation
        method, config = w._work_methods["process"]
        result = await method(value=5)

        assert result == 10

        # Link handler invocation
        handler = getattr(w, "loop_link")
        next_kwargs = await handler(10)

        assert next_kwargs == {"value": 10}

        # Link returns None when condition not met
        next_kwargs = await handler(150)
        assert next_kwargs is None
