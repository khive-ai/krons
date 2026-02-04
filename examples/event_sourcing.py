# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Event Sourcing: audit trails and state reconstruction via Flow persistence.

Demonstrates:
- Recording events (commands, state changes) via Exchange message passing
- Serializing Flow state to JSON for persistence (snapshots)
- Restoring Flow from JSON (recovery)
- Replaying events from progressions in order
- Querying and auditing event history

Architecture:
- Event Producer: sends events to Event Store
- Event Store: dedicated entity that collects all events in its inbox
- After sync(), events are preserved in the Event Store's Flow
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any
from uuid import UUID

from krons.core import Element, Flow
from krons.session import Exchange


class EventEntity(Element):
    """An entity that emits or stores events."""

    name: str = "entity"


def replay_event(event_content: dict[str, Any], event_time: datetime) -> None:
    """Process a replayed event. In real systems, this rebuilds state."""
    event_type = event_content.get("type", "unknown")
    print(f"  [REPLAY] {event_time.isoformat()[:19]} | {event_type}: {event_content}")


def audit_event(
    msg_id: UUID, event_content: dict[str, Any], event_time: datetime
) -> dict[str, Any]:
    """Create an audit record for an event."""
    return {
        "event_id": str(msg_id),
        "timestamp": event_time.isoformat(),
        "type": event_content.get("type", "unknown"),
        "data": event_content,
    }


async def run_event_sourcing_demo() -> None:
    """Demonstrate event sourcing pattern with Flow persistence."""

    print("=" * 70)
    print("Event Sourcing with krons Exchange and Flow")
    print("=" * 70)
    print()

    # ========================================
    # PHASE 1: Record Events
    # ========================================
    print("[Phase 1] Recording Events")
    print("-" * 40)

    exchange = Exchange()

    # Event producer (e.g., order service)
    order_service = EventEntity(name="order_service")
    exchange.register(order_service.id)

    # Event store - dedicated entity that collects all events
    event_store = EventEntity(name="event_store")
    exchange.register(event_store.id)

    # Simulate a sequence of domain events for an order lifecycle
    events = [
        {"type": "OrderCreated", "order_id": "ORD-001", "customer": "Alice"},
        {"type": "ItemAdded", "order_id": "ORD-001", "item": "Widget", "qty": 2},
        {"type": "ItemAdded", "order_id": "ORD-001", "item": "Gadget", "qty": 1},
        {"type": "PaymentReceived", "order_id": "ORD-001", "amount": 150.00},
        {"type": "OrderShipped", "order_id": "ORD-001", "tracking": "TRK-12345"},
        {"type": "OrderDelivered", "order_id": "ORD-001", "signature": "A. Smith"},
    ]

    # Send events from order_service to event_store (direct messaging)
    for event in events:
        msg = exchange.send(
            sender=order_service.id,
            recipient=event_store.id,  # Direct to event store
            content=event,
            channel="orders",
        )
        print(f"  Recorded: {event['type']} (msg_id: {str(msg.id)[:8]})")

    # Sync routes messages from outbox to event_store's inbox
    await exchange.sync()

    print(f"\nTotal events recorded: {len(events)}")

    # ========================================
    # PHASE 2: Snapshot (Serialize Flow)
    # ========================================
    print()
    print("[Phase 2] Creating Snapshot")
    print("-" * 40)

    # Get the event store's flow (contains all events in inbox)
    flow = exchange.get(event_store.id)
    if flow is None:
        raise RuntimeError("Flow not found")

    # Serialize to JSON-compatible dict
    snapshot = flow.to_dict(mode="json")
    snapshot_json = json.dumps(snapshot, indent=2)

    print(f"Snapshot size: {len(snapshot_json)} bytes")
    print(f"Flow state: {len(flow.items)} items, {len(flow.progressions)} progressions")

    # In production: save to file/database
    # with open("flow_snapshot.json", "w") as f:
    #     json.dump(snapshot, f)
    print("Snapshot created (ready for persistence)")

    # ========================================
    # PHASE 3: Restore from Snapshot
    # ========================================
    print()
    print("[Phase 3] Restoring from Snapshot")
    print("-" * 40)

    # Simulate loading from persistence
    restored_snapshot = json.loads(snapshot_json)

    # Restore Flow from dict
    restored_flow = Flow.from_dict(restored_snapshot)

    print(f"Restored flow: {len(restored_flow.items)} items")
    print(f"Restored progressions: {len(restored_flow.progressions)}")
    print(f"Flow ID match: {restored_flow.id == flow.id}")

    # ========================================
    # PHASE 4: Replay Events
    # ========================================
    print()
    print("[Phase 4] Replaying Events (State Reconstruction)")
    print("-" * 40)

    # Iterate through all progressions and replay events
    for progression in restored_flow.progressions:
        prog_name = progression.name or "unnamed"
        print(f"\nProgression '{prog_name}' ({len(progression)} events):")

        # Events are stored in order within the progression
        for msg_id in progression:
            msg = restored_flow.items.get(msg_id)
            if msg is not None:
                replay_event(msg.content, msg.created_at)

    # ========================================
    # PHASE 5: Audit / Query Event History
    # ========================================
    print()
    print("[Phase 5] Audit Trail Query")
    print("-" * 40)

    # Build audit log from all events
    audit_log: list[dict[str, Any]] = []

    for progression in restored_flow.progressions:
        for msg_id in progression:
            msg = restored_flow.items.get(msg_id)
            if msg is not None:
                audit_log.append(audit_event(msg.id, msg.content, msg.created_at))

    # Sort by timestamp (events should already be in order, but good practice)
    audit_log.sort(key=lambda x: x["timestamp"])

    print(f"Audit log contains {len(audit_log)} events:")
    for record in audit_log:
        print(f"  {record['timestamp'][:19]} | {record['type']}")

    # Query specific events
    print()
    print("Query: Find all 'ItemAdded' events")
    item_events = [e for e in audit_log if e["type"] == "ItemAdded"]
    for event in item_events:
        item = event["data"].get("item", "unknown")
        qty = event["data"].get("qty", 0)
        print(f"  - {item} x{qty}")

    # ========================================
    # PHASE 6: Advanced - Time Travel Query
    # ========================================
    print()
    print("[Phase 6] Time Travel Query")
    print("-" * 40)

    # Find state at a specific point in time
    # (replay events up to cutoff timestamp)
    if len(audit_log) >= 3:
        cutoff_time = audit_log[2]["timestamp"]
        print(f"Reconstructing state as of: {cutoff_time[:19]}")

        events_before_cutoff = [e for e in audit_log if e["timestamp"] <= cutoff_time]
        print(f"Events up to cutoff: {len(events_before_cutoff)}")
        for event in events_before_cutoff:
            print(f"  {event['type']}")

    print()
    print("=" * 70)
    print("Event Sourcing Demo Complete")
    print("=" * 70)


async def main() -> None:
    """Entry point."""
    await run_event_sourcing_demo()


if __name__ == "__main__":
    asyncio.run(main())
