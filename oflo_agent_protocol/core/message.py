"""Canonical message format — normalized across OpenAI, Anthropic, A2A, and LangGraph."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from .types import MessageRole


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]

    def to_openai(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }

    def to_anthropic(self) -> Dict[str, Any]:
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.arguments}


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    content: Any
    is_error: bool = False


@dataclass
class CanonicalMessage:
    """Single message type that all runtimes translate to/from."""

    role: MessageRole
    content: str
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation to provider formats
    # ------------------------------------------------------------------

    def to_openai(self) -> Dict[str, Any]:
        if self.role == MessageRole.TOOL:
            return {
                "role": "tool",
                "tool_call_id": self.tool_results[0].tool_call_id if self.tool_results else "",
                "content": str(self.tool_results[0].content) if self.tool_results else self.content,
            }
        msg: Dict[str, Any] = {"role": self.role.value, "content": self.content or ""}
        if self.name:
            msg["name"] = self.name
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
            msg["content"] = None
        return msg

    def to_anthropic(self) -> Dict[str, Any]:
        if self.role == MessageRole.TOOL:
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tr.tool_call_id,
                        "content": str(tr.content),
                        "is_error": tr.is_error,
                    }
                    for tr in self.tool_results
                ],
            }
        if self.tool_calls and self.role == MessageRole.ASSISTANT:
            blocks: List[Dict[str, Any]] = []
            if self.content:
                blocks.append({"type": "text", "text": self.content})
            blocks.extend(tc.to_anthropic() for tc in self.tool_calls)
            return {"role": "assistant", "content": blocks}
        return {"role": self.role.value, "content": self.content or ""}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.message_id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in self.tool_calls
            ],
            "tool_results": [
                {"id": tr.tool_call_id, "name": tr.name, "content": tr.content}
                for tr in self.tool_results
            ],
        }

    # ------------------------------------------------------------------
    # Deserialisation from provider formats
    # ------------------------------------------------------------------

    @classmethod
    def from_openai(cls, data: Dict[str, Any]) -> "CanonicalMessage":
        role = MessageRole(data.get("role", "user"))
        content = data.get("content") or ""
        tool_calls: List[ToolCall] = []
        for tc in data.get("tool_calls") or []:
            args_raw = tc["function"].get("arguments", "{}")
            tool_calls.append(
                ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=json.loads(args_raw) if isinstance(args_raw, str) else args_raw,
                )
            )
        return cls(role=role, content=content, tool_calls=tool_calls)

    @classmethod
    def from_anthropic_response(cls, response: Any) -> "CanonicalMessage":
        """Parse an Anthropic SDK Message object."""
        content_text = ""
        tool_calls: List[ToolCall] = []
        for block in response.content:
            if hasattr(block, "text"):
                content_text += block.text
            elif hasattr(block, "type") and block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input or {})
                )
        return cls(role=MessageRole.ASSISTANT, content=content_text, tool_calls=tool_calls)

    @classmethod
    def user(cls, text: str, **meta: Any) -> "CanonicalMessage":
        return cls(role=MessageRole.USER, content=text, metadata=meta)

    @classmethod
    def system(cls, text: str) -> "CanonicalMessage":
        return cls(role=MessageRole.SYSTEM, content=text)

    @classmethod
    def assistant(cls, text: str) -> "CanonicalMessage":
        return cls(role=MessageRole.ASSISTANT, content=text)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalMessage":
        role = MessageRole(data.get("role", "user"))
        tool_calls = [
            ToolCall(id=tc["id"], name=tc["name"], arguments=tc.get("arguments", {}))
            for tc in data.get("tool_calls", [])
        ]
        tool_results = [
            ToolResult(tool_call_id=tr["id"], name=tr["name"], content=tr.get("content"))
            for tr in data.get("tool_results", [])
        ]
        return cls(
            role=role,
            content=data.get("content", ""),
            message_id=data.get("id", str(uuid.uuid4())),
            tool_calls=tool_calls,
            tool_results=tool_results,
        )


def normalise_history(
    messages: List[Union[CanonicalMessage, Dict[str, Any]]]
) -> List[CanonicalMessage]:
    """Ensure all messages are CanonicalMessage instances."""
    out = []
    for m in messages:
        if isinstance(m, CanonicalMessage):
            out.append(m)
        elif isinstance(m, dict):
            out.append(CanonicalMessage.from_dict(m))
    return out
