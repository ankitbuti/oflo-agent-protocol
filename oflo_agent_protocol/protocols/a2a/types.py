"""Google Agent-to-Agent (A2A) protocol types — June 2025 spec.

Spec: https://google.github.io/A2A/specification/

Key concepts:
  AgentCard    — JSON published at /.well-known/agent.json describing capabilities
  A2ATask      — Unit of work sent to a remote agent
  Artifact     — Output produced by a task
  Message      — Message within a task conversation
  TaskStatus   — submitted → working → (input-required |) completed | failed | canceled
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# AgentCard
# ---------------------------------------------------------------------------

@dataclass
class AgentSkill:
    id: str
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    input_modes: List[str] = field(default_factory=lambda: ["text"])
    output_modes: List[str] = field(default_factory=lambda: ["text"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "examples": self.examples,
            "inputModes": self.input_modes,
            "outputModes": self.output_modes,
        }


@dataclass
class AgentCard:
    """Describes an agent's identity and capabilities for A2A discovery."""
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    skills: List[AgentSkill] = field(default_factory=list)
    default_input_modes: List[str] = field(default_factory=lambda: ["text"])
    default_output_modes: List[str] = field(default_factory=lambda: ["text"])
    documentation_url: Optional[str] = None
    provider: Optional[Dict[str, str]] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "skills": [s.to_dict() for s in self.skills],
            "defaultInputModes": self.default_input_modes,
            "defaultOutputModes": self.default_output_modes,
            "capabilities": {
                "streaming": self.capabilities.get("streaming", True),
                "pushNotifications": self.capabilities.get("pushNotifications", False),
                "stateTransitionHistory": self.capabilities.get("stateTransitionHistory", True),
                **{k: v for k, v in self.capabilities.items()
                   if k not in ("streaming", "pushNotifications", "stateTransitionHistory")},
            },
        }
        if self.documentation_url:
            d["documentationUrl"] = self.documentation_url
        if self.provider:
            d["provider"] = self.provider
        return d


# ---------------------------------------------------------------------------
# Task & Artifact
# ---------------------------------------------------------------------------

@dataclass
class TextPart:
    text: str
    mime_type: str = "text/plain"

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "text", "text": self.text, "mimeType": self.mime_type}


@dataclass
class DataPart:
    data: Dict[str, Any]
    mime_type: str = "application/json"

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "data", "data": self.data, "mimeType": self.mime_type}


@dataclass
class A2AMessage:
    role: str  # "user" | "agent"
    parts: List[Any]
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(
            p.text if hasattr(p, "text") else str(p.get("text", ""))
            for p in self.parts
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "parts": [p.to_dict() if hasattr(p, "to_dict") else p for p in self.parts],
            "messageId": self.message_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def text_message(cls, role: str, text: str) -> "A2AMessage":
        return cls(role=role, parts=[TextPart(text=text)])


@dataclass
class Artifact:
    name: str
    parts: List[Any]
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: Optional[str] = None
    index: int = 0
    append: bool = False
    last_chunk: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifactId": self.artifact_id,
            "name": self.name,
            "description": self.description,
            "parts": [p.to_dict() if hasattr(p, "to_dict") else p for p in self.parts],
            "index": self.index,
            "append": self.append,
            "lastChunk": self.last_chunk,
        }


@dataclass
class TaskStatusUpdate:
    state: str  # submitted | working | input-required | completed | failed | canceled
    message: Optional[A2AMessage] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"state": self.state, "timestamp": self.timestamp}
        if self.message:
            d["message"] = self.message.to_dict()
        return d


@dataclass
class A2ATask:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    status: TaskStatusUpdate = field(
        default_factory=lambda: TaskStatusUpdate(state="submitted")
    )
    history: List[A2AMessage] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.task_id,
            "sessionId": self.session_id,
            "status": self.status.to_dict(),
            "history": [m.to_dict() for m in self.history],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 envelopes
# ---------------------------------------------------------------------------

def jsonrpc_request(method: str, params: Any, req_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id or str(uuid.uuid4()),
        "method": method,
        "params": params,
    }


def jsonrpc_response(result: Any, req_id: Optional[str] = None) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def jsonrpc_error(
    code: int, message: str, data: Any = None, req_id: Optional[str] = None
) -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


# Standard A2A error codes
A2A_ERRORS = {
    "task_not_found": (-32001, "Task not found"),
    "task_not_cancelable": (-32002, "Task not cancelable"),
    "push_not_supported": (-32003, "Push notifications not supported"),
    "unsupported_operation": (-32004, "Unsupported operation"),
    "internal_error": (-32603, "Internal error"),
    "invalid_params": (-32602, "Invalid params"),
    "method_not_found": (-32601, "Method not found"),
}
