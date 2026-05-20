# Oflo AI Agent Protocol v2

A modular, auditable, multi-LLM agent protocol for building production-grade AI agents.  
Smart routing · MCP 2024-11 · Google A2A · Voice · Daytona Sandbox · Redis Memory · Composio (300+ app connectors)

---

## What it is

Oflo Agent Protocol is a Python framework for building AI agents that are:

- **Provider-agnostic** — route across Claude, GPT-4o, Gemini, Llama (via Groq/Ollama/OpenRouter) with a smart cost/latency router
- **Tool-first** — any async Python function becomes a tool with one decorator
- **Fully auditable** — every call produces an `AuditRecord` with token usage, cost, latency, and PII-scrubbed content
- **Project-isolated** — each business domain gets its own `Project` with budget limits, agents, and audit log
- **Protocol-compliant** — exposes agents as MCP servers (2024-11 spec) and Google A2A endpoints simultaneously

---

## Installation

```bash
# Core framework only
pip install oflo-ai-agent-protocol

# With a specific provider
pip install "oflo-ai-agent-protocol[anthropic]"
pip install "oflo-ai-agent-protocol[openai]"

# With Composio connectors (300+ apps)
pip install "oflo-ai-agent-protocol[composio]"

# With voice (ElevenLabs)
pip install "oflo-ai-agent-protocol[voice]"

# Everything
pip install "oflo-ai-agent-protocol[all]"
```

---

## Quick start

```python
import asyncio
from oflo_agent_protocol import Project
from oflo_agent_protocol.runtimes.claude_runtime import ClaudeRuntime

async def main():
    project = Project("marketing", cost_budget_usd=5.0)

    analyst = project.add_agent(
        "Analyst",
        system_prompt="You are a data analyst.",
        runtime=ClaudeRuntime(),
    )

    @analyst.tool(description="Look up campaign performance")
    async def get_campaign_metrics(campaign_id: str) -> dict:
        return {"clicks": 1200, "conversions": 48, "ctr": 0.04}

    reply = await project.ask("Analyst", "How did campaign A1 perform?")
    print(reply)

asyncio.run(main())
```

---

## Architecture

```
oflo_agent_protocol/
├── core/
│   ├── agent.py          # BaseAgentV2 — agentic loop, tool registry, guardrails
│   ├── message.py        # CanonicalMessage — normalises OpenAI ↔ Anthropic formats
│   ├── registry.py       # AgentRegistry — async-safe per-project agent store
│   └── types.py          # All shared enums and dataclasses
├── routing/
│   ├── llm_router.py     # SmartRouter — CHEAPEST / FASTEST / SMARTEST / BALANCED
│   └── providers.py      # Model catalogue with pricing, latency, capability flags
├── runtimes/
│   ├── claude_runtime.py     # Anthropic (prompt caching, tool use, streaming)
│   ├── openai_runtime.py     # OpenAI + Groq + Ollama (OpenAI-compatible)
│   ├── openrouter_runtime.py # OpenRouter — multi-model fallback
│   ├── daytona_runtime.py    # Daytona — sandboxed code execution
│   └── langgraph_runtime.py  # LangGraph orchestration
├── protocols/
│   ├── mcp/              # MCP 2024-11 server + client
│   └── a2a/              # Google A2A server + client (JSON-RPC, SSE)
├── managers/
│   └── agent_manager.py  # Per-project orchestrator (delegate, chain, broadcast)
├── projects/
│   └── base_project.py   # Project — top-level isolation unit
├── connectors/
│   └── composio_connector.py # 300+ app connectors via Composio
├── voice/
│   ├── voice_agent.py    # VoiceAgent — ElevenLabs conversational AI
│   └── audio_interface.py    # Audio I/O adapters (mic, file, WebSocket, null)
├── memory/
│   ├── memory_manager.py # In-process + Weaviate v4 vector memory
│   └── redis_memory.py   # Redis Agent Memory Server (working + long-term)
└── audit/
    ├── audit_logger.py   # JSONL audit log per project
    ├── telemetry.py      # In-process metrics, budget alerts
    └── guardrails.py     # PII scrubbing, toxicity blocking, custom rules
```

---

## LLM Routing

```python
from oflo_agent_protocol.routing.llm_router import route, RoutingStrategy

# Auto-selects best available model based on strategy
decision = route(strategy=RoutingStrategy.BALANCED)
print(decision.provider.value, decision.model_id)

# Require specific capabilities
decision = route(
    strategy=RoutingStrategy.CHEAPEST,
    need_vision=True,
    need_long_context=True,
)
```

Supported strategies: `CHEAPEST`, `FASTEST`, `SMARTEST`, `BALANCED`, `CAPABILITY_MATCH`  
Supported providers (env-var driven): Anthropic, OpenAI, Google, Groq, Ollama, OpenRouter

---

## Multi-agent patterns

```python
mgr = AgentManager("sales", cost_budget_usd=10.0)

researcher = mgr.create_agent("Researcher", system_prompt="Research leads.")
writer     = mgr.create_agent("Writer",     system_prompt="Write outreach emails.")

# Delegate: Researcher → Writer
result = await mgr.delegate("Researcher", "Writer", "Top 3 fintech leads this week")

# Broadcast to all agents
replies = await mgr.broadcast("What is your current status?")

# Pipeline
chain = await mgr.chain([
    {"agent": "Researcher", "message": "Find 5 AI startups"},
    {"agent": "Writer",     "message": "Draft cold emails", "use_previous": True},
])
```

---

## Composio — 300+ app connectors

```python
from oflo_agent_protocol.connectors import ComposioConnector, ComposioToolKit

connector = ComposioConnector(api_key=os.getenv("COMPOSIO_API_KEY"))

# Inject GitHub + Slack tools into an agent
await connector.inject_into_agent(agent, toolkits=["github", "slack"])

# Pre-built toolkit groups
await connector.inject_into_agent(agent, toolkits=ComposioToolKit.DEVOPS)

# OAuth connect flow
url = await connector.connect_app("github", callback_url="https://myapp.com/callback")

# Direct execution (no LLM)
result = await connector.execute_action(
    "GITHUB_CREATE_ISSUE",
    {"owner": "my-org", "repo": "backend", "title": "Bug: login fails"},
)
```

`ComposioToolKit` groups: `DEVOPS`, `COMMUNICATION`, `PRODUCTIVITY`, `DATA`, `CLOUD`, `ECOMMERCE`

Or wire Composio directly into `AgentManager`:

```python
mgr = AgentManager(
    "engineering",
    composio_api_key=os.getenv("COMPOSIO_API_KEY"),
)
dev_agent = mgr.create_agent(
    "DevBot",
    system_prompt="You manage GitHub issues and Slack notifications.",
    composio_toolkits=["github", "slack"],   # auto-injected on creation
)
```

---

## Voice AI (ElevenLabs)

```python
from oflo_agent_protocol.voice.voice_agent import VoiceAgent

agent = VoiceAgent(
    name="SalesRep",
    system_prompt="You are a helpful sales representative.",
    elevenlabs_agent_id=os.getenv("ELEVENLABS_AGENT_ID"),
)

@agent.tool(description="Look up a customer account")
async def get_account(customer_id: str) -> dict:
    return {"name": "Acme Corp", "tier": "enterprise"}

stats = await agent.start_voice_session(
    on_user_transcript=lambda t: print(f"User: {t}"),
    on_agent_response=lambda r: print(f"Agent: {r}"),
)
```

---

## Daytona — Sandboxed code execution

```python
from oflo_agent_protocol.runtimes.daytona_runtime import DaytonaSandboxRuntime
from oflo_agent_protocol.runtimes.claude_runtime import ClaudeRuntime

runtime = DaytonaSandboxRuntime(llm_runtime=ClaudeRuntime(), auto_destroy=True)
agent = BaseAgentV2("CodeAgent", runtime=runtime)

@agent.tool(description="Execute Python in an isolated sandbox")
async def run_python(code: str) -> dict:
    return await runtime.exec_code(code)

reply = await agent.chat("Compute the first 20 Fibonacci numbers.")
await runtime.close()
```

---

## Redis shared memory

```python
from oflo_agent_protocol.memory.redis_memory import SharedSessionMemory

async with SharedSessionMemory("session-abc", "my-project") as mem:
    await mem.append_to_working_memory("user", "What are our Q2 targets?")
    enriched = await mem.enrich_system_prompt(agent.system_prompt, "Q2 targets")
    results  = await mem.search("revenue targets", limit=5)
```

Or via `AgentManager`:
```python
mgr = AgentManager("finance", redis_base_url="http://localhost:8000")
# route_message() automatically enriches context and persists exchanges
reply = await mgr.route_message("FinanceBot", "What is our Q2 runway?")
```

---

## MCP & A2A servers

```python
project = Project("analytics")

# Expose as MCP (tool-calling protocol)
mcp = project.mcp_server(port=8080)

# Expose as Google A2A (agent-to-agent)
a2a = project.a2a_server(port=9000)

# Start both
await project.serve(mcp_port=8080, a2a_port=9000)
```

---

## Audit & guardrails

```python
from oflo_agent_protocol.audit.guardrails import GuardrailConfig

config = GuardrailConfig(
    scrub_pii=True,          # redact email, phone, SSN, credit card
    block_pii=False,         # scrub instead of block
    custom_blocks=["confidential project X"],
    max_length_chars=4000,
)

agent = BaseAgentV2("SafeBot", guardrail_config=config)
```

Every `agent.chat()` call emits an `AuditRecord` to `AuditLogger` (JSONL) and `Telemetry` (in-memory metrics with budget alerts).

---

## Environment variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude models |
| `OPENAI_API_KEY` | GPT-4o, o3 |
| `GOOGLE_API_KEY` | Gemini |
| `GROQ_API_KEY` | Llama via Groq |
| `OPENROUTER_API_KEY` | 300+ models via OpenRouter |
| `OLLAMA_HOST` | Local Ollama server |
| `COMPOSIO_API_KEY` | Composio app connectors |
| `ELEVENLABS_API_KEY` | ElevenLabs voice AI |
| `ELEVENLABS_AGENT_ID` | Conversational agent ID |
| `DAYTONA_API_KEY` | Daytona sandbox |
| `REDIS_MEMORY_URL` | Redis Agent Memory Server |

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

---

## License

MIT © Ankit Buti / Oflo AI
