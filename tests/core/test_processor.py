# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.core.processor - Event queue consumer."""

import pytest

from krons.core import Event, Pile, Processor
from krons.core.event import EventStatus


@pytest.fixture
def configured_processor():
    """Create a TestProcessor with proper configuration."""
    from tests.conftest import TestProcessor

    pile = Pile[Event]()
    return TestProcessor(
        pile=pile,
        queue_capacity=10,
        capacity_refresh_time=0.1,
    )


class TestProcessorCreation:
    """Test Processor instantiation."""

    def test_processor_with_pile(self, configured_processor):
        """Processor should accept event pile."""
        assert configured_processor.pile is not None
        assert isinstance(configured_processor.pile, Pile)

    def test_processor_event_type(self, configured_processor):
        """Processor should have event_type property."""
        from tests.conftest import SimpleTestEvent

        assert configured_processor.event_type is SimpleTestEvent

    def test_processor_queue_capacity(self):
        """Processor should accept queue_capacity parameter."""
        from tests.conftest import SimpleTestEvent, TestProcessor

        pile = Pile[Event]()
        processor = TestProcessor(
            pile=pile,
            queue_capacity=10,
            capacity_refresh_time=1.0,
        )
        assert processor.queue_capacity == 10

    def test_processor_invalid_queue_capacity_zero(self):
        """Processor should reject queue_capacity < 1."""
        from tests.conftest import TestProcessor

        pile = Pile[Event]()
        with pytest.raises(ValueError):
            TestProcessor(
                pile=pile,
                queue_capacity=0,
                capacity_refresh_time=1.0,
            )

    def test_processor_invalid_queue_capacity_too_large(self):
        """Processor should reject queue_capacity > 10000."""
        from tests.conftest import TestProcessor

        pile = Pile[Event]()
        with pytest.raises(ValueError):
            TestProcessor(
                pile=pile,
                queue_capacity=10001,
                capacity_refresh_time=1.0,
            )

    def test_processor_invalid_refresh_time_too_small(self):
        """Processor should reject capacity_refresh_time < 0.01."""
        from tests.conftest import TestProcessor

        pile = Pile[Event]()
        with pytest.raises(ValueError):
            TestProcessor(
                pile=pile,
                queue_capacity=10,
                capacity_refresh_time=0.001,
            )

    def test_processor_invalid_refresh_time_too_large(self):
        """Processor should reject capacity_refresh_time > 3600."""
        from tests.conftest import TestProcessor

        pile = Pile[Event]()
        with pytest.raises(ValueError):
            TestProcessor(
                pile=pile,
                queue_capacity=10,
                capacity_refresh_time=3601,
            )


class TestProcessorLifecycle:
    """Test Processor start/stop lifecycle."""

    @pytest.mark.anyio
    async def test_start_stop(self, configured_processor):
        """Processor should start and stop cleanly."""
        assert configured_processor.is_stopped() is False

        await configured_processor.stop()
        assert configured_processor.is_stopped() is True

        await configured_processor.start()
        assert configured_processor.is_stopped() is False

    @pytest.mark.anyio
    async def test_process_event(self, configured_processor, simple_event):
        """Processor should process events from queue."""
        # Add event to pile and queue
        configured_processor.pile.add(simple_event)
        await configured_processor.enqueue(simple_event.id)

        # Process
        await configured_processor.process()

        # Event should be completed
        assert simple_event.execution.status == EventStatus.COMPLETED


class TestProcessorQueue:
    """Test Processor queue operations."""

    @pytest.mark.anyio
    async def test_enqueue(self, configured_processor, simple_event):
        """Processor.enqueue() should add event to queue."""
        configured_processor.pile.add(simple_event)
        await configured_processor.enqueue(simple_event.id)

        assert configured_processor.queue.qsize() == 1

    @pytest.mark.anyio
    async def test_enqueue_with_priority(self, configured_processor):
        """Processor.enqueue() should respect priority."""
        from tests.conftest import SimpleTestEvent

        event1 = SimpleTestEvent(return_value="first")
        event2 = SimpleTestEvent(return_value="second")

        configured_processor.pile.add(event1)
        configured_processor.pile.add(event2)

        # Enqueue with priorities (lower = processed first)
        await configured_processor.enqueue(event2.id, priority=10.0)
        await configured_processor.enqueue(event1.id, priority=1.0)

        # Dequeue should return event1 first (lower priority)
        dequeued = await configured_processor.dequeue()
        assert dequeued.id == event1.id

    @pytest.mark.anyio
    async def test_dequeue(self, configured_processor, simple_event):
        """Processor.dequeue() should return event from queue."""
        configured_processor.pile.add(simple_event)
        await configured_processor.enqueue(simple_event.id)

        dequeued = await configured_processor.dequeue()
        assert dequeued.id == simple_event.id


class TestProcessorCapacity:
    """Test Processor capacity management."""

    @pytest.mark.anyio
    async def test_available_capacity(self):
        """Processor should track available capacity."""
        from tests.conftest import TestProcessor

        pile = Pile[Event]()
        processor = TestProcessor(
            pile=pile,
            queue_capacity=5,
            capacity_refresh_time=1.0,
        )

        assert processor.available_capacity == 5

    @pytest.mark.anyio
    async def test_capacity_consumed_on_process(self):
        """Processor should consume capacity when processing."""
        from tests.conftest import SimpleTestEvent, TestProcessor

        pile = Pile[Event]()
        processor = TestProcessor(
            pile=pile,
            queue_capacity=5,
            capacity_refresh_time=1.0,
        )

        event = SimpleTestEvent(return_value="test")
        pile.add(event)
        await processor.enqueue(event.id)

        initial_capacity = processor.available_capacity
        await processor.process()

        # Capacity should be reset after processing
        assert processor.available_capacity == processor.queue_capacity


class TestProcessorConcurrency:
    """Test Processor concurrency limits."""

    @pytest.mark.anyio
    async def test_concurrency_limit(self):
        """Processor should respect concurrency_limit."""
        from tests.conftest import TestProcessor

        pile = Pile[Event]()
        processor = TestProcessor(
            pile=pile,
            queue_capacity=10,
            capacity_refresh_time=1.0,
            concurrency_limit=5,
        )

        assert processor.concurrency_limit == 5


class TestProcessorRequestPermission:
    """Test Processor permission checking."""

    @pytest.mark.anyio
    async def test_request_permission_default(self, configured_processor):
        """Processor.request_permission() should return True by default."""
        result = await configured_processor.request_permission()
        assert result is True
