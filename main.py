"""
Oflo Agent Protocol v2 — entry point / smoke test.

Run:  python main.py
"""
import asyncio
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")


async def main() -> None:
    from oflo_agent_protocol import Project, RoutingStrategy
    from oflo_agent_protocol.routing.llm_router import route

    print("Oflo Agent Protocol v2")
    print("─" * 40)

    # ── Routing demo (no LLM key needed) ─────────────────────────────
    decision = route(strategy=RoutingStrategy.BALANCED)
    print(f"BALANCED route  → {decision.provider.value}/{decision.model_id}")

    decision_fast = route(strategy=RoutingStrategy.FASTEST)
    print(f"FASTEST route   → {decision_fast.provider.value}/{decision_fast.model_id}")

    decision_cheap = route(strategy=RoutingStrategy.CHEAPEST)
    print(f"CHEAPEST route  → {decision_cheap.provider.value}/{decision_cheap.model_id}")

    # ── Project + agent (requires ANTHROPIC_API_KEY) ──────────────────
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n[skip] Set ANTHROPIC_API_KEY to run the full agent demo.")
        return

    from oflo_agent_protocol.runtimes.claude_runtime import ClaudeRuntime

    project = Project("demo", cost_budget_usd=0.10)
    agent = project.add_agent(
        "Assistant",
        system_prompt="You are a concise and helpful assistant.",
        runtime=ClaudeRuntime(model_id="claude-haiku-4-5-20251001"),
    )

    @agent.tool(description="Return the current UTC timestamp")
    async def get_time() -> dict:
        import datetime
        return {"utc": datetime.datetime.utcnow().isoformat() + "Z"}

    print("\nSending: 'What time is it right now?'")
    reply = await project.ask("Assistant", "What time is it right now?")
    print(f"Reply: {reply}")
    print(f"Stats: {project.stats()}")


if __name__ == "__main__":
    asyncio.run(main())
