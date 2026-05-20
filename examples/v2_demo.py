"""
Oflo Agent Protocol v2 — Complete Demonstration
================================================

Showcases:
  1. Multi-LLM smart routing (auto-selects cheapest available provider)
  2. Tool use with full agentic loop
  3. Agent-to-Agent delegation within a project
  4. MCP server exposure
  5. Google A2A cross-project call
  6. Audit trail + telemetry
  7. LangGraph multi-agent pipeline
  8. Guardrails (PII scrubbing)

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...
    python examples/v2_demo.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("demo")

# ---------------------------------------------------------------------------
# Lazy imports — only load what's available
# ---------------------------------------------------------------------------
from oflo_agent_protocol import (
    ClaudeRuntime,
    GuardrailConfig,
    MemoryManager,
    ModelProvider,
    Project,
    RoutingStrategy,
    route,
)


# ---------------------------------------------------------------------------
# Demo 1: Simple multi-LLM routing
# ---------------------------------------------------------------------------
def demo_routing() -> None:
    print("\n" + "=" * 60)
    print("DEMO 1: Smart LLM Routing")
    print("=" * 60)
    from oflo_agent_protocol import RoutingRequest, SmartRouter
    router = SmartRouter()
    print("Available providers:", router.describe())

    for strategy in ("cheapest", "fastest", "balanced", "smartest"):
        from oflo_agent_protocol.core.types import RoutingStrategy as RS
        decision = router.route(RoutingRequest(strategy=RS(strategy)))
        print(f"  [{strategy:12s}] → {decision.provider.value}/{decision.model_id}")


# ---------------------------------------------------------------------------
# Demo 2: Project with tools + delegation
# ---------------------------------------------------------------------------
async def demo_project() -> None:
    print("\n" + "=" * 60)
    print("DEMO 2: Project with Tools & Delegation")
    print("=" * 60)

    # Create a project with a $2 budget
    marketing = Project(
        "marketing_demo",
        strategy=RoutingStrategy.CHEAPEST,
        cost_budget_usd=2.0,
    )

    # Cost alert
    def on_alert(alert_type: str, data: dict) -> None:
        print(f"  ⚠ ALERT: {alert_type} — {data}")

    marketing.on_cost_alert(on_alert)

    # Create analyst agent
    analyst = marketing.add_agent(
        "Analyst",
        system_prompt=(
            "You are a marketing analyst. Use tools to fetch data, then provide "
            "a concise, insight-driven analysis. Always cite the data."
        ),
    )

    # Register tools
    @analyst.tool(description="Fetch campaign performance metrics for a campaign ID")
    async def get_campaign_metrics(campaign_id: str) -> dict:
        # Simulated — replace with real API call
        return {
            "campaign_id": campaign_id,
            "clicks": 12_400,
            "impressions": 180_000,
            "conversions": 620,
            "ctr": 0.069,
            "cpa": 8.50,
        }

    @analyst.tool(description="Get competitor benchmark data for a category")
    async def get_benchmark(category: str) -> dict:
        return {
            "category": category,
            "avg_ctr": 0.045,
            "avg_cpa": 12.00,
            "top_performer_ctr": 0.12,
        }

    # Create writer agent
    writer = marketing.add_agent(
        "Writer",
        system_prompt=(
            "You are a marketing copywriter. Given analysis data, write compelling, "
            "concise marketing copy. Match the brand voice: friendly, data-driven, bold."
        ),
    )

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("  ⚠ No API keys found — skipping live LLM calls.")
        print("  Agents created:", marketing.agents)
        return

    # Route a message to Analyst
    print("\n  [Analyst] Processing campaign report request...")
    try:
        analysis = await marketing.ask(
            "Analyst",
            "Analyse campaign C-2025-Q2 performance vs benchmarks and give me the top 2 insights.",
        )
        print(f"\n  Analyst reply:\n  {analysis[:500]}")

        # Delegate analysis → Writer
        print("\n  [Delegation] Analyst → Writer...")
        copy = await marketing.delegate(
            "Analyst",
            "Writer",
            "Turn the analysis into a one-paragraph social media post.",
        )
        print(f"\n  Writer reply:\n  {copy[:400]}")

    except Exception as exc:
        print(f"  LLM call failed (expected if no API key): {exc}")

    # Telemetry
    print("\n  Telemetry:", json.dumps(marketing.stats()["telemetry"], indent=2))


# ---------------------------------------------------------------------------
# Demo 3: MCP server
# ---------------------------------------------------------------------------
async def demo_mcp() -> None:
    print("\n" + "=" * 60)
    print("DEMO 3: MCP Server")
    print("=" * 60)

    from oflo_agent_protocol import MCPServer, MCPClient

    server = MCPServer(name="demo-mcp", port=18080)

    # Create a simple agent and register it
    project = Project("mcp_demo")
    helper = project.add_agent("Helper", system_prompt="You answer questions concisely.")

    @helper.tool(description="Get current timestamp")
    async def now() -> str:
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"

    server.register_agent(helper)
    await server.start()
    print(f"  MCP server started — tools: {await _list_tools_local(server)}")

    # Test via client
    async with MCPClient("http://localhost:18080") as client:
        tools = await client.list_tools()
        print(f"  Client sees {len(tools)} tools: {[t['name'] for t in tools]}")
        if tools:
            result = await client.call_tool("Helper.now", {})
            print(f"  Tool call result: {result}")


async def _list_tools_local(server: any) -> list:
    """Call list tools directly on the server app without HTTP."""
    fake_req = {"jsonrpc": "2.0", "id": "x", "method": "tools/list", "params": {}}
    resp = await server._dispatch(fake_req)
    data = json.loads(resp.body)
    return [t["name"] for t in data.get("result", {}).get("tools", [])]


# ---------------------------------------------------------------------------
# Demo 4: Google A2A Server + Client
# ---------------------------------------------------------------------------
async def demo_a2a() -> None:
    print("\n" + "=" * 60)
    print("DEMO 4: Google A2A Protocol")
    print("=" * 60)

    from oflo_agent_protocol import A2AClient, A2AServer, AgentCard, AgentSkill

    # Build a mock agent that echoes
    class EchoAgent:
        name = "EchoAgent"
        async def chat(self, text: str) -> str:
            return f"A2A echo: {text}"

    card = AgentCard(
        name="Echo Service",
        description="Echoes messages for demo purposes",
        url="http://localhost:19000",
        skills=[AgentSkill(id="echo", name="Echo", description="Echoes any input")],
    )
    server = A2AServer(card=card, agent=EchoAgent(), port=19000)
    await server.start()

    # Client call
    async with A2AClient("http://localhost:19000") as client:
        discovered = await client.discover()
        print(f"  Discovered: {discovered.name} @ {discovered.url}")
        task = await client.send("Hello from A2A client!")
        print(f"  Task state: {task.status.state}")
        if task.artifacts:
            print(f"  Artifact: {task.artifacts[0].parts[0].text}")


# ---------------------------------------------------------------------------
# Demo 5: LangGraph multi-agent pipeline
# ---------------------------------------------------------------------------
async def demo_langgraph() -> None:
    print("\n" + "=" * 60)
    print("DEMO 5: LangGraph Multi-Agent Pipeline")
    print("=" * 60)

    from oflo_agent_protocol import LangGraphOrchestrator, ModelProvider

    orch = LangGraphOrchestrator()

    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage

        llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

        def researcher_node(state: dict) -> dict:
            msgs = state.get("messages", [])
            prompt = msgs[-1].content if msgs else ""
            resp = llm.invoke([HumanMessage(content=f"Research this briefly: {prompt}")])
            return {"messages": msgs + [resp]}

        def writer_node(state: dict) -> dict:
            msgs = state.get("messages", [])
            last = msgs[-1].content if msgs else ""
            resp = llm.invoke([HumanMessage(content=f"Write a tweet about: {last}")])
            return {"messages": msgs + [resp]}

        orch.add_node("researcher", researcher_node)
        orch.add_node("writer", writer_node)
        orch.add_edge("researcher", "writer")

        result = await orch.run("Latest advances in multi-agent AI systems")
        final = result["messages"][-1].content
        print(f"  LangGraph result: {final[:300]}")

    except ImportError:
        print("  langchain-anthropic not installed — skipping live LangGraph demo")
    except Exception as exc:
        print(f"  LangGraph demo error (expected without API key): {exc}")


# ---------------------------------------------------------------------------
# Demo 6: Memory Manager
# ---------------------------------------------------------------------------
async def demo_memory() -> None:
    print("\n" + "=" * 60)
    print("DEMO 6: Memory Manager")
    print("=" * 60)

    mem = MemoryManager(project_id="demo")
    await mem.store("agent-1", "User prefers concise answers", memory_type="semantic")
    await mem.store("agent-1", "Campaign C-2025-Q2 had CTR of 6.9%", memory_type="episodic")
    await mem.store("agent-2", "Pricing strategy: freemium", memory_type="semantic")

    results = await mem.search("CTR campaign", agent_id="agent-1")
    print(f"  Search 'CTR campaign' → {len(results)} results")
    for r in results:
        print(f"    [{r.memory_type}] {r.content}")

    recent = await mem.get_recent("agent-1", limit=5)
    print(f"  Recent memories for agent-1: {len(recent)}")


# ---------------------------------------------------------------------------
# Demo 7: Guardrails
# ---------------------------------------------------------------------------
def demo_guardrails() -> None:
    print("\n" + "=" * 60)
    print("DEMO 7: Guardrails (PII + Content Safety)")
    print("=" * 60)

    from oflo_agent_protocol import Guardrails
    from oflo_agent_protocol.core.message import CanonicalMessage

    gr = Guardrails()
    config = GuardrailConfig(block_pii=False, scrub_pii=True, toxicity_check=True)

    test_cases = [
        "The customer email is john.doe@example.com and phone 555-123-4567.",
        "Our revenue grew 23% QoQ to $4.2M. Strong momentum in enterprise.",
        "Here is a clean response with no issues.",
    ]

    for text in test_cases:
        msg = CanonicalMessage.assistant(text)
        result = gr.check(msg, config)
        status = "FLAGGED" if result.flags else "OK"
        print(f"  [{status}] {text[:60]}...")
        if result.flags:
            print(f"         Flags: {result.flags}")
        if result.scrubbed_content:
            print(f"         Scrubbed: {result.scrubbed_content[:80]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    print("\n" + "█" * 60)
    print("  Oflo Agent Protocol v2 — Demo Suite")
    print("█" * 60)

    demo_routing()
    await demo_project()
    await demo_a2a()
    await demo_memory()
    demo_guardrails()

    # These require network servers — uncomment to test:
    # await demo_mcp()
    # await demo_langgraph()

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
