from .content import (
    ActionRequest,
    ActionResponse,
    Assistant,
    CustomParser,
    CustomRenderer,
    Instruction,
    MessageContent,
    MessageRole,
    System,
)
from .prepare_msg import prepare_messages_for_chat

# Aliases for backward compatibility with *Content naming
ActionRequestContent = ActionRequest
ActionResponseContent = ActionResponse
AssistantResponseContent = Assistant
InstructionContent = Instruction
SystemContent = System

__all__ = (
    "ActionRequest",
    "ActionRequestContent",
    "ActionResponse",
    "ActionResponseContent",
    "Assistant",
    "AssistantResponseContent",
    "CustomParser",
    "CustomRenderer",
    "Instruction",
    "InstructionContent",
    "MessageContent",
    "MessageRole",
    "System",
    "SystemContent",
    "prepare_messages_for_chat",
)

