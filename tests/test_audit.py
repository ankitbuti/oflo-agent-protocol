"""Tests for audit logger, telemetry, and guardrails."""
from __future__ import annotations

import tempfile
import uuid

import pytest

from oflo_agent_protocol.audit.audit_logger import AuditLogger
from oflo_agent_protocol.audit.guardrails import GuardrailConfig, Guardrails, GuardrailResult
from oflo_agent_protocol.audit.telemetry import Telemetry
from oflo_agent_protocol.core.message import CanonicalMessage
from oflo_agent_protocol.core.types import AuditRecord, ModelProvider, TokenUsage


# ── AuditLogger ───────────────────────────────────────────────────────────────

class TestAuditLogger:
    @pytest.fixture
    def logger(self, tmp_path):
        return AuditLogger(project_id="test-proj", log_dir=str(tmp_path))

    @pytest.mark.asyncio
    async def test_log_and_query(self, logger):
        record = AuditRecord(
            agent_id="agent-1",
            agent_name="TestAgent",
            project_id="test-proj",
            provider="anthropic",
            model="claude-sonnet-4-6",
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
            latency_ms=250.0,
            cost_usd=0.00045,
            success=True,
        )
        await logger.log(record)
        results = await logger.query(limit=10)
        assert len(results) == 1
        assert results[0]["agent_name"] == "TestAgent"

    @pytest.mark.asyncio
    async def test_summary(self, logger):
        for i in range(3):
            rec = AuditRecord(
                agent_id=f"agent-{i}",
                agent_name="Agent",
                project_id="test-proj",
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                token_usage=TokenUsage(prompt_tokens=50, completion_tokens=25),
                cost_usd=0.0001,
                success=True,
            )
            await logger.log(rec)

        summary = logger.get_summary()
        assert summary["total_calls"] == 3
        assert summary["total_cost_usd"] == pytest.approx(0.0003, rel=1e-3)

    @pytest.mark.asyncio
    async def test_error_tracking(self, logger):
        rec = AuditRecord(
            agent_id="agent-err",
            agent_name="ErrAgent",
            project_id="test-proj",
            provider="openai",
            model="gpt-4o",
            success=False,
            error="Rate limit exceeded",
        )
        await logger.log(rec)
        summary = logger.get_summary()
        assert summary["error_count"] >= 1

    def test_hash_prompt(self):
        h1 = AuditLogger.hash_prompt("hello world")
        h2 = AuditLogger.hash_prompt("hello world")
        h3 = AuditLogger.hash_prompt("different text")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 16  # first 16 hex chars of SHA-256


# ── Telemetry ─────────────────────────────────────────────────────────────────

class TestTelemetry:
    @pytest.mark.asyncio
    async def test_record_and_summary(self):
        tel = Telemetry(cost_budget_usd=1.0, token_budget=10000)
        rec = AuditRecord(
            agent_id="a1",
            agent_name="Bot",
            project_id="proj",
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
            cost_usd=0.0002,
            latency_ms=300.0,
            success=True,
        )
        await tel.record(rec)
        summary = tel.summary()
        # Summary has project_cost_usd and per-agent breakdown
        assert "project_cost_usd" in summary
        assert summary["project_cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_cost_alert_fires(self):
        alerts = []
        tel = Telemetry(cost_budget_usd=0.001)
        # on_alert callback receives (alert_type: str, data: dict)
        tel.on_alert(lambda alert_type, data: alerts.append(alert_type))

        # Exceed the budget
        for _ in range(5):
            rec = AuditRecord(
                agent_id="a1", agent_name="Bot", project_id="proj",
                provider="anthropic", model="claude-sonnet-4-6",
                token_usage=TokenUsage(prompt_tokens=1000, completion_tokens=500),
                cost_usd=0.001,
                success=True,
            )
            await tel.record(rec)

        assert len(alerts) >= 1
        assert "cost_budget_exceeded" in alerts


# ── Guardrails ────────────────────────────────────────────────────────────────

class TestGuardrails:
    @pytest.fixture
    def guardrails(self):
        return Guardrails()

    def test_clean_message_passes(self, guardrails):
        msg = CanonicalMessage.assistant("The weather today is sunny and warm.")
        config = GuardrailConfig()
        result = guardrails.check(msg, config)
        assert result.passed is True
        assert result.blocked is False
        assert not result.flags

    def test_email_pii_scrubbed(self, guardrails):
        msg = CanonicalMessage.assistant("Contact john.doe@example.com for details.")
        config = GuardrailConfig(scrub_pii=True, block_pii=False)
        result = guardrails.check(msg, config)
        assert result.scrubbed_content is not None
        assert "[EMAIL_REDACTED]" in result.scrubbed_content
        assert "john.doe@example.com" not in result.scrubbed_content
        assert any("email" in f for f in result.flags)

    def test_phone_pii_scrubbed(self, guardrails):
        msg = CanonicalMessage.assistant("Call us at 555-123-4567.")
        config = GuardrailConfig(scrub_pii=True, block_pii=False)
        result = guardrails.check(msg, config)
        assert result.scrubbed_content is not None
        assert "[PHONE_REDACTED]" in result.scrubbed_content

    def test_ssn_pii_scrubbed(self, guardrails):
        msg = CanonicalMessage.assistant("My SSN is 123-45-6789.")
        config = GuardrailConfig(scrub_pii=True, block_pii=False)
        result = guardrails.check(msg, config)
        assert result.scrubbed_content is not None
        assert "[SSN_REDACTED]" in result.scrubbed_content

    def test_pii_block_mode(self, guardrails):
        msg = CanonicalMessage.assistant("Email: user@test.com")
        config = GuardrailConfig(block_pii=True, scrub_pii=False)
        result = guardrails.check(msg, config)
        assert result.blocked is True

    def test_custom_block_phrase(self, guardrails):
        msg = CanonicalMessage.assistant("This is totally forbidden content here.")
        config = GuardrailConfig(custom_blocks=["forbidden content"], block_pii=False, scrub_pii=False)
        result = guardrails.check(msg, config)
        assert result.blocked is True

    def test_max_length_flag(self, guardrails):
        long_text = "x" * 5000
        msg = CanonicalMessage.assistant(long_text)
        config = GuardrailConfig(max_length_chars=100, block_pii=False, scrub_pii=False)
        result = guardrails.check(msg, config)
        # Max length violation is flagged (may or may not block depending on impl)
        assert any("output_too_long" in f for f in result.flags)

    def test_no_scrub_when_disabled(self, guardrails):
        msg = CanonicalMessage.assistant("Email: admin@oflo.ai")
        config = GuardrailConfig(block_pii=False, scrub_pii=False)
        result = guardrails.check(msg, config)
        # No scrubbing — content unchanged
        assert result.scrubbed_content is None
