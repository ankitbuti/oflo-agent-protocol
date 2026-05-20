"""Output guardrails — content safety, length, JSON validation, PII scrubbing."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern

from oflo_agent_protocol.core.message import CanonicalMessage

logger = logging.getLogger(__name__)

# Simple PII patterns — extend as needed
_PII_PATTERNS: List[tuple[str, Pattern]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
    ("phone", re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")),
]

_TOXICITY_WORDS = {
    "hate", "violence", "self-harm",  # extend from a real list in production
}


@dataclass
class GuardrailResult:
    passed: bool = True
    flags: List[str] = field(default_factory=list)
    scrubbed_content: Optional[str] = None
    blocked: bool = False

    def add_flag(self, flag: str) -> None:
        self.flags.append(flag)
        self.passed = False


@dataclass
class GuardrailConfig:
    max_output_tokens: int = 8192
    block_pii: bool = True
    scrub_pii: bool = True
    toxicity_check: bool = True
    require_json: bool = False
    json_schema: Optional[Dict[str, Any]] = None
    custom_blocks: List[str] = field(default_factory=list)
    max_length_chars: Optional[int] = None


class Guardrails:
    """
    Lightweight, synchronous guardrail engine.

    Run with `check(message, config)` — returns GuardrailResult.
    Blocking a message means the agent should not send the response.
    """

    def check(self, message: CanonicalMessage, config: GuardrailConfig) -> GuardrailResult:
        result = GuardrailResult()
        content = message.content or ""

        # Length
        if config.max_length_chars and len(content) > config.max_length_chars:
            result.add_flag(f"output_too_long:{len(content)}")

        # PII detection
        if config.block_pii or config.scrub_pii:
            scrubbed = content
            for pii_type, pattern in _PII_PATTERNS:
                if pattern.search(content):
                    result.add_flag(f"pii:{pii_type}")
                    if config.scrub_pii:
                        scrubbed = pattern.sub(f"[{pii_type.upper()}_REDACTED]", scrubbed)
                    elif config.block_pii:
                        result.blocked = True
            if scrubbed != content:
                result.scrubbed_content = scrubbed

        # Toxicity
        if config.toxicity_check:
            lower = content.lower()
            for word in _TOXICITY_WORDS:
                if word in lower:
                    result.add_flag(f"toxicity:{word}")
                    result.blocked = True

        # Custom block strings
        for block in config.custom_blocks:
            if block.lower() in content.lower():
                result.add_flag(f"custom_block:{block}")
                result.blocked = True

        # JSON validation
        if config.require_json:
            try:
                parsed = json.loads(content)
                if config.json_schema:
                    self._validate_schema(parsed, config.json_schema, result)
            except json.JSONDecodeError:
                result.add_flag("invalid_json")

        return result

    @staticmethod
    def _validate_schema(
        data: Any, schema: Dict[str, Any], result: GuardrailResult
    ) -> None:
        required = schema.get("required", [])
        if not isinstance(data, dict):
            result.add_flag("json_not_object")
            return
        for key in required:
            if key not in data:
                result.add_flag(f"missing_required_key:{key}")


# Module-level default instance
_default_guardrails = Guardrails()


def check_output(
    message: CanonicalMessage,
    config: Optional[GuardrailConfig] = None,
) -> GuardrailResult:
    return _default_guardrails.check(message, config or GuardrailConfig())
