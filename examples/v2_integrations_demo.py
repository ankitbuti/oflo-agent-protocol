"""
v2 Integrations Demo
====================
Demonstrates all four Part-2 integrations working together:

  1. OpenRouter  — multi-model fallback routing
  2. ElevenLabs  — voice AI session via NullAudioInterface (CI-safe)
  3. Daytona     — sandboxed code execution (requires DAYTONA_API_KEY)
  4. Redis Memory — shared session memory across agents

Run:
    python examples/v2_integrations_demo.py

Required env vars (set what you have; each section degrades gracefully):
    ANTHROPIC_API_KEY      — for primary LLM calls
    OPENROUTER_API_KEY     — for multi-model fallback demo
    ELEVENLABS_API_KEY     — for voice agent demo
    ELEVENLABS_AGENT_ID    — ElevenLabs conversational agent ID
    DAYTONA_API_KEY        — for sandbox demo
    REDIS_MEMORY_URL       — e.g. http://localhost:8000 (defaults to that)
"""
from __future__ import annotations

import asyncio
import os
import sys


# ─────────────────────────────────────────────────────────────────────────────
# 1. OpenRouter — multi-model fallback
# ─────────────────────────────────────────────────────────────────────────────

async def demo_openrouter() -> None:
    print("\n" + "=" * 60)
    print("1. OPENROUTER — Multi-model fallback routing")
    print("=" * 60)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("  [skip] OPENROUTER_API_KEY not set")
        return

    try:
        from oflo_agent_protocol.runtimes.openrouter_runtime import OpenRouterRuntime
        from oflo_agent_protocol.core.agent import BaseAgentV2

        # Create a runtime with a primary model + two fallbacks
        runtime = OpenRouterRuntime.with_fallbacks(
            primary="claude-sonnet",
            fallbacks=["gpt-4o", "llama-70b"],
            strategy="quality",
            max_price_per_m=20.0,
        )

        agent = BaseAgentV2(
            name="Researcher",
            system_prompt="You are a concise research assistant.",
            runtime=runtime,
        )

        reply = await agent.chat("Name three recent breakthroughs in AI agent protocols.")
        print(f"\n  Agent reply:\n  {reply[:300]}{'...' if len(reply) > 300 else ''}")
        print(f"\n  Runtime: {runtime.provider_name}/{runtime.model_id}")

    except Exception as exc:
        print(f"  [error] {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ElevenLabs — voice AI agent (NullAudioInterface for CI)
# ─────────────────────────────────────────────────────────────────────────────

async def demo_voice() -> None:
    print("\n" + "=" * 60)
    print("2. ELEVENLABS — Voice AI Agent")
    print("=" * 60)

    api_key = os.getenv("ELEVENLABS_API_KEY")
    agent_id = os.getenv("ELEVENLABS_AGENT_ID")
    if not api_key or not agent_id:
        print("  [skip] ELEVENLABS_API_KEY or ELEVENLABS_AGENT_ID not set")
        return

    try:
        from oflo_agent_protocol.voice.voice_agent import VoiceAgent
        from oflo_agent_protocol.voice.audio_interface import NullAudioInterface

        voice_agent = VoiceAgent(
            name="VoiceAssistant",
            system_prompt="You are a friendly voice assistant.",
            elevenlabs_agent_id=agent_id,
            elevenlabs_api_key=api_key,
        )

        # Register a tool to demo client-tool injection into voice session
        @voice_agent.tool(description="Get current time")
        async def get_time() -> dict:
            import datetime
            return {"time": datetime.datetime.now().isoformat()}

        print("  Starting voice session with NullAudioInterface (silent, CI-safe)...")

        # Patch DefaultAudioInterface with NullAudioInterface to avoid mic access
        import oflo_agent_protocol.voice.voice_agent as va_module
        original = va_module.__dict__.get("DefaultAudioInterface")
        va_module.DefaultAudioInterface = NullAudioInterface  # type: ignore

        try:
            # NullAudioInterface yields 1s of silence then ends — session auto-closes
            stats = await asyncio.wait_for(
                voice_agent.start_voice_session(
                    on_user_transcript=lambda t: print(f"  User: {t}"),
                    on_agent_response=lambda r: print(f"  Agent: {r}"),
                ),
                timeout=5.0,
            )
            print(f"  Session stats: {stats}")
        except asyncio.TimeoutError:
            print("  Session ended (timeout — expected with NullAudioInterface)")
        finally:
            if original:
                va_module.DefaultAudioInterface = original  # type: ignore

    except Exception as exc:
        print(f"  [error] {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Daytona — sandboxed code execution
# ─────────────────────────────────────────────────────────────────────────────

async def demo_daytona() -> None:
    print("\n" + "=" * 60)
    print("3. DAYTONA — Sandboxed Code Execution")
    print("=" * 60)

    api_key = os.getenv("DAYTONA_API_KEY")
    if not api_key:
        print("  [skip] DAYTONA_API_KEY not set")
        return

    try:
        from oflo_agent_protocol.runtimes.daytona_runtime import (
            DaytonaSandboxRuntime,
            daytona_session,
        )
        from oflo_agent_protocol.runtimes.claude_runtime import ClaudeRuntime
        from oflo_agent_protocol.core.agent import BaseAgentV2

        # Wrap Claude with Daytona sandboxing
        runtime = DaytonaSandboxRuntime(
            llm_runtime=ClaudeRuntime(),
            api_key=api_key,
            auto_destroy=True,
        )

        agent = BaseAgentV2(
            name="CodeAgent",
            system_prompt="You are a coding assistant that executes Python safely.",
            runtime=runtime,
        )

        # Register tools that delegate to the sandbox
        @agent.tool(description="Execute Python code in an isolated sandbox")
        async def run_python(code: str) -> dict:
            return await runtime.exec_code(code)

        @agent.tool(description="Run a shell command in the sandbox")
        async def run_shell(command: str) -> str:
            return await runtime.exec_command(command)

        print("  Creating sandbox and running a computation...")
        reply = await agent.chat(
            "Use run_python to compute the first 10 fibonacci numbers and return them."
        )
        print(f"\n  Agent reply:\n  {reply[:400]}")
        print(f"\n  Runtime: {runtime.provider_name}")

        await runtime.close()

    except Exception as exc:
        print(f"  [error] {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Redis Memory — shared session memory
# ─────────────────────────────────────────────────────────────────────────────

async def demo_redis_memory() -> None:
    print("\n" + "=" * 60)
    print("4. REDIS MEMORY — Shared Session Memory")
    print("=" * 60)

    redis_url = os.getenv("REDIS_MEMORY_URL", "http://localhost:8000")

    try:
        from oflo_agent_protocol.memory.redis_memory import RedisMemoryManager, SharedSessionMemory

        mem = RedisMemoryManager(
            session_id="demo-session-001",
            project_id="demo",
            base_url=redis_url,
        )

        available = await mem._is_available()
        if not available:
            print(f"  [skip] Redis Memory Server not reachable at {redis_url}")
            print("         Start it: docker run -p 8000:8000 redis/agent-memory-server")
            return

        print(f"  Connected to Redis Memory Server at {redis_url}")

        # Working memory: shared message window
        await mem.clear_working_memory()
        await mem.append_to_working_memory("user", "What is our Q1 revenue target?")
        await mem.append_to_working_memory("assistant", "Q1 target is $2.5M based on last year's growth.")
        await mem.append_to_working_memory("user", "And Q2?")

        wm = await mem.get_working_memory()
        msgs = (wm or {}).get("messages", [])
        print(f"  Working memory has {len(msgs)} messages")

        # Long-term memory: persist facts
        await mem.add_long_term(
            text="Revenue targets: Q1=$2.5M, Q2=$3.2M, Q3=$4.1M, Q4=$5.0M",
            agent_id="finance-agent",
            memory_type="fact",
            topics=["revenue", "targets"],
        )
        print("  Stored revenue targets in long-term memory")

        # Search
        results = await mem.search("revenue Q2", limit=3)
        print(f"  Search 'revenue Q2' returned {len(results)} results")
        if results:
            print(f"  Top result: {str(results[0])[:120]}")

        # Prompt hydration
        enriched = await mem.enrich_system_prompt(
            "You are a financial advisor.",
            "What are our targets?",
        )
        extra_lines = len(enriched.splitlines()) - 1
        print(f"  System prompt enriched: +{extra_lines} lines of memory context")

        # SharedSessionMemory context manager
        async with SharedSessionMemory("team-session", "demo", clear_on_exit=True) as shared:
            await shared.append_to_working_memory("user", "Brief the whole team.")
            wm2 = await shared.get_working_memory()
            print(f"  SharedSessionMemory: {len((wm2 or {}).get('messages', []))} messages (auto-cleared on exit)")

    except Exception as exc:
        print(f"  [error] {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Combined: AgentManager with Redis memory wired in
# ─────────────────────────────────────────────────────────────────────────────

async def demo_agent_manager_with_redis() -> None:
    print("\n" + "=" * 60)
    print("5. AGENT MANAGER — Redis memory + multi-agent routing")
    print("=" * 60)

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("  [skip] ANTHROPIC_API_KEY not set")
        return

    redis_url = os.getenv("REDIS_MEMORY_URL", "http://localhost:8000")

    try:
        from oflo_agent_protocol.managers.agent_manager import AgentManager
        from oflo_agent_protocol.runtimes.claude_runtime import ClaudeRuntime

        mgr = AgentManager(
            project_id="integrations-demo",
            cost_budget_usd=1.0,
            redis_base_url=redis_url,
            session_id="demo-session-002",
        )

        analyst = mgr.create_agent(
            "Analyst",
            system_prompt="You are a data analyst. Be concise.",
            runtime=ClaudeRuntime(model_id="claude-haiku-4-5-20251001"),
        )
        writer = mgr.create_agent(
            "Writer",
            system_prompt="You write clear summaries. Be brief.",
            runtime=ClaudeRuntime(model_id="claude-haiku-4-5-20251001"),
        )

        # Route through manager — memory automatically enriches context
        analysis = await mgr.route_message(
            "Analyst",
            "Summarise the main trends in AI agent protocols in two sentences.",
        )
        print(f"\n  Analyst: {analysis[:200]}")

        summary = await mgr.route_message(
            "Writer",
            f"Make this punchier: {analysis[:150]}",
        )
        print(f"\n  Writer: {summary[:200]}")

        # Pipeline (chain)
        chain_results = await mgr.chain([
            {"agent": "Analyst", "message": "List three key AI trends in one sentence each."},
            {"agent": "Writer", "message": "Turn this into a tweet thread.", "use_previous": True},
        ])
        print(f"\n  Pipeline result (Writer):\n  {chain_results[-1][:300]}")

        mem_results = await mgr.memory_search("AI trends", limit=3)
        print(f"\n  Memory search returned {len(mem_results)} past exchanges")
        print(f"\n  Manager: {mgr!r}")

    except Exception as exc:
        print(f"  [error] {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("Oflo Agent Protocol v2 — Integration Demo")
    print("Showing: OpenRouter · ElevenLabs Voice · Daytona Sandbox · Redis Memory")

    await demo_openrouter()
    await demo_voice()
    await demo_daytona()
    await demo_redis_memory()
    await demo_agent_manager_with_redis()

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
