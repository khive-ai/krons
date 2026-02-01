from krons.core import Pile
from krons.core.types import ID

from .session import Session

SESSION_REGISTRY: Pile[Session] = Pile(item_type=Session, strict_type=True)


async def get_session(session_id: ID[Session]) -> Session:
    if session_id not in SESSION_REGISTRY:
        raise ValueError(f"Session with id {session_id} not found in registry.")
    async with SESSION_REGISTRY:
        return SESSION_REGISTRY[session_id]


async def create_session():
    session = Session()
    async with SESSION_REGISTRY:
        SESSION_REGISTRY.add(session)


async def delete_session(session_id: ID[Session]):
    if session_id not in SESSION_REGISTRY:
        raise ValueError(f"Session with id {session_id} not found in registry.")
    async with SESSION_REGISTRY:
        SESSION_REGISTRY.remove(session_id)


async def list_sessions_ids() -> list[ID[Session]]:
    return list(SESSION_REGISTRY.keys())


async def clear_sessions():
    async with SESSION_REGISTRY:
        SESSION_REGISTRY.clear()
