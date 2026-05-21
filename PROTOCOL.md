# Oflo Agent Protocol — Inter-Agent Alignment Spec

> This document is the contract. If your project follows it, oflo becomes the
> circuit board between your agents and every other agent in the network.

---

## 0. Core principle

Every agent is a node. Every node speaks the same three things:
- **A2A** — receives tasks from other agents
- **MCP** — exposes its tools to other agents
- **CanonicalMessage** — the single message format across all LLM providers

Do not build point-to-point integrations between agents. Align to this spec and
the routing, memory, and audit layers work for free.

---

## 1. Install

```bash
pip install oflo-ai-agent-protocol
# or with extras you need:
pip install "oflo-ai-agent-protocol[anthropic,composio]"
```

---

## 2. Declare your agent

Every agent must be a `BaseAgentV2`. No exceptions.

```python
from oflo_agent_protocol import BaseAgentV2
from oflo_agent_protocol.runtimes.claude_runtime import ClaudeRuntime

agent = BaseAgentV2(
    name="YourAgentName",           # unique within your project
    system_prompt="...",
    runtime=ClaudeRuntime(),        # or any BaseRuntime subclass
    project_id="your-project-id",  # namespace for audit + routing
)
```

**Required fields:** `name`, `system_prompt`, `project_id`  
**Runtime is optional** — omit it and the SmartRouter selects automatically.

---

## 3. Declare your tools

Use the decorator. No other registration method is recognised by the protocol.

```python
@agent.tool(description="What this tool does — be specific")
async def your_tool(param_one: str, param_two: int) -> dict:
    """Docstring becomes the description fallback."""
    ...
    return {"result": ...}
```

Rules:
- Handler **must be async**.
- Return type **must be JSON-serialisable** (dict, list, str, int, bool).
- Parameter types **must be annotated** — they become the JSON schema automatically.
- Tool name is the function name. Make it unambiguous.

---

## 4. Expose via A2A (receive tasks from other agents)

```python
from oflo_agent_protocol.protocols.a2a.server import A2AServer
from oflo_agent_protocol.protocols.a2a.types import AgentCard, AgentSkill

card = AgentCard(
    name="YourAgentName",
    description="One sentence: what this agent does and for whom.",
    url="https://your-service.com",           # public URL of this A2A server
    skills=[
        AgentSkill(
            id="primary-skill",
            name="Primary capability",
            description="What you can ask this agent to do.",
            input_modes=["text"],
            output_modes=["text"],
        )
    ],
    version="1.0.0",
)

server = A2AServer(card=card, agent=agent, host="0.0.0.0", port=9000)
await server.start()
```

Your agent is now discoverable at `GET https://your-service.com/.well-known/agent.json`  
and receives tasks at `POST https://your-service.com/`.

**The oflo router will call you here.** Keep this endpoint alive.

---

## 5. Expose via MCP (share tools with other agents)

```python
from oflo_agent_protocol.protocols.mcp.server import MCPServer

mcp = MCPServer(name="your-project-id", version="1.0.0", host="0.0.0.0", port=8080)
mcp.register_agent(agent)
await mcp.start()
```

Your tools are now callable by any MCP-compatible client at:
- `POST /tools/call` with `{"name": "YourAgentName.tool_name", "arguments": {...}}`
- `GET /tools/list` to enumerate all tools

**Expose both ports.** A2A = agent-level tasks. MCP = tool-level calls.

---

## 6. Call other agents (send tasks outbound)

```python
from oflo_agent_protocol.protocols.a2a.client import A2AClient

async with A2AClient("https://target-agent.com") as client:
    card = await client.discover()          # inspect capabilities first
    task = await client.send_and_wait(
        message="Your instruction here",
        timeout=60.0,
    )
    print(task.artifacts[0].parts[0].text)
```

Or call an MCP tool on another agent:

```python
from oflo_agent_protocol.protocols.mcp.client import MCPClient

async with MCPClient("https://target-agent.com:8080") as client:
    result = await client.call_tool("AgentName.tool_name", {"param": "value"})
```

---

## 7. Join a project (multi-agent coordination)

```python
from oflo_agent_protocol.managers.agent_manager import AgentManager

mgr = AgentManager(
    project_id="shared-project-id",     # same across all agents in this project
    cost_budget_usd=10.0,
    composio_api_key=os.getenv("COMPOSIO_API_KEY"),  # optional: shared app connectors
    redis_base_url=os.getenv("REDIS_MEMORY_URL"),    # optional: shared memory
)

agent = mgr.create_agent("YourAgent", system_prompt="...")

# Delegate to another agent in the same project
result = await mgr.delegate("YourAgent", "OtherAgent", "task description")

# Pipeline across agents
outputs = await mgr.chain([
    {"agent": "Researcher", "message": "Find X"},
    {"agent": "Writer",     "message": "Draft from this", "use_previous": True},
])
```

All agents registered to the same `project_id` share:
- Audit log (JSONL, append-only)
- Telemetry and budget alerts
- Guardrail config
- Redis working memory (if configured)

---

## 8. Use shared memory

If the network has a Redis memory server running, every agent in a project can
read and write the same working context.

```python
# Read context before responding
enriched_prompt = await mgr.memory_search("topic of current task", limit=5)

# Store after a significant exchange
await mgr._redis.add_long_term(
    text="Key insight: ...",
    agent_id=agent.id,
    memory_type="fact",
    topics=["relevant", "tags"],
)
```

The `route_message()` path does this automatically. If you call `agent.chat()`
directly, hydrate the system prompt yourself:

```python
from oflo_agent_protocol.memory.redis_memory import RedisMemoryManager

mem = RedisMemoryManager(session_id="session-id", project_id="project-id")
agent._system_prompt = await mem.enrich_system_prompt(agent.system_prompt, user_query)
```

Start the memory server:
```bash
docker run -p 8000:8000 redis/agent-memory-server
```

---

## 9. Connect external apps via Composio

Do not write custom integrations for GitHub, Slack, Gmail, Notion, Jira, etc.
Use Composio. One line per app.

```python
from oflo_agent_protocol.connectors import ComposioConnector, ComposioToolKit

connector = ComposioConnector(api_key=os.getenv("COMPOSIO_API_KEY"))

# Inject tools at agent creation
await connector.inject_into_agent(agent, toolkits=["github", "slack"])

# Or use named groups
await connector.inject_into_agent(agent, toolkits=ComposioToolKit.DEVOPS)

# Initiate OAuth for a user
url = await connector.connect_app("github", callback_url="https://your-app.com/oauth")
```

Or via `AgentManager`:

```python
mgr = AgentManager("project", composio_api_key="...")
agent = mgr.create_agent("DevBot", composio_toolkits=["github", "linear"])
```

---

## 10. Implement the CanonicalMessage contract

If you write a custom runtime (not using one of the built-in runtimes), it
**must** satisfy `BaseRuntime`:

```python
from oflo_agent_protocol.runtimes.base_runtime import BaseRuntime
from oflo_agent_protocol.core.message import CanonicalMessage
from oflo_agent_protocol.core.types import TokenUsage

class YourRuntime(BaseRuntime):

    @property
    def provider_name(self) -> str:
        return "your-provider"            # used in audit records

    @property
    def model_id(self) -> str:
        return "your-model-id"

    async def complete(
        self, messages, system=None, tools=None, max_tokens=4096, temperature=0.7, **kwargs
    ) -> tuple[CanonicalMessage, TokenUsage]:
        # Call your LLM here
        # Return a CanonicalMessage with role=ASSISTANT and correct TokenUsage
        ...

    async def stream(self, messages, ...) -> AsyncIterator[str]:
        ...

    async def health_check(self) -> bool:
        return True
```

All message translation to/from your provider's format happens inside your runtime.
The agent never sees provider-specific objects.

---

## 11. Message format reference

```python
from oflo_agent_protocol.core.message import CanonicalMessage, ToolCall, ToolResult
from oflo_agent_protocol.core.types import MessageRole

# Build messages
user_msg  = CanonicalMessage.user("text")
sys_msg   = CanonicalMessage.system("You are...")
asst_msg  = CanonicalMessage.assistant("text")

# Convert to provider format
openai_dict     = msg.to_openai()      # {"role": "user", "content": "..."}
anthropic_dict  = msg.to_anthropic()   # {"role": "user", "content": [...]}

# Parse from provider format
msg = CanonicalMessage.from_openai({"role": "assistant", "content": "..."})

# Tool call (LLM wants to run a tool)
tc = ToolCall(id="tc1", name="tool_name", arguments={"key": "val"})

# Tool result (your code ran the tool)
tr = ToolResult(tool_call_id="tc1", name="tool_name", content={"result": ...})
```

---

## 12. Routing contract

Declare what your agent needs from the LLM at the routing layer, not at
the agent layer. The router selects the best available model automatically.

```python
from oflo_agent_protocol.routing.llm_router import route, RoutingStrategy

decision = route(
    strategy=RoutingStrategy.BALANCED,    # CHEAPEST | FASTEST | SMARTEST | BALANCED | CAPABILITY_MATCH
    need_vision=False,
    need_long_context=True,               # >128k tokens
    need_function_calling=True,
    task_complexity=0.8,                  # 0.0=trivial, 1.0=expert
)
# decision.provider, decision.model_id, decision.fallback_chain
```

Env vars drive availability — set the keys for the providers you want available:

```
ANTHROPIC_API_KEY   OPENAI_API_KEY   GOOGLE_API_KEY
GROQ_API_KEY        OPENROUTER_API_KEY   OLLAMA_HOST
```

---

## 13. Audit contract

Every call through `agent.chat()` or `agent.process()` automatically produces
an `AuditRecord`. You do not write audit code — you read it.

```python
records = await mgr.audit_report(limit=100)
# Each record: agent_id, agent_name, provider, model, prompt_hash,
#              token_usage, cost_usd, latency_ms, success, error,
#              guardrail_flags, timestamp
```

To attach a cost alert:
```python
mgr.on_cost_alert(lambda alert_type, data: notify_slack(data))
```

---

## 14. Guardrail contract

Set once at the project level. Applies to every agent in the project.

```python
from oflo_agent_protocol.audit.guardrails import GuardrailConfig

config = GuardrailConfig(
    scrub_pii=True,                          # redact email, phone, SSN, card
    block_pii=False,                         # scrub instead of block
    custom_blocks=["internal codename X"],   # exact-phrase blocklist
    max_length_chars=8000,
)

mgr = AgentManager("project", guardrail_config=config)
```

---

## 15. Environment variables — complete reference

| Variable | Required for |
|---|---|
| `ANTHROPIC_API_KEY` | Claude models |
| `OPENAI_API_KEY` | GPT-4o, o3 |
| `GOOGLE_API_KEY` | Gemini 2.x |
| `GROQ_API_KEY` | Llama via Groq |
| `OPENROUTER_API_KEY` | 300+ models via OpenRouter |
| `OLLAMA_HOST` | Local Ollama (e.g. `http://localhost:11434`) |
| `COMPOSIO_API_KEY` | App connectors (GitHub, Slack, Gmail…) |
| `ELEVENLABS_API_KEY` | Voice AI sessions |
| `ELEVENLABS_AGENT_ID` | Specific ElevenLabs conversational agent |
| `DAYTONA_API_KEY` | Sandboxed code execution |
| `DAYTONA_API_URL` | Daytona server URL (default: `https://app.daytona.io/api`) |
| `REDIS_MEMORY_URL` | Redis Agent Memory Server (default: `http://localhost:8000`) |

---

## 16. Verification checklist

Before connecting your agent to the network, confirm:

- [ ] `agent.name` is unique within your `project_id`
- [ ] Every tool handler is `async` and returns JSON-serialisable data
- [ ] `GET /.well-known/agent.json` returns a valid `AgentCard`
- [ ] `POST /` (A2A) responds to `tasks/send` and returns a task with status
- [ ] `GET /tools/list` (MCP) lists all agent tools
- [ ] `agent.process()` produces an `AuditRecord` on every call
- [ ] Guardrail config is set (even if permissive — `GuardrailConfig()` is fine)
- [ ] `health_check()` on your runtime returns `True`
- [ ] `python3 -m pytest` passes in your project

---

## 17. Minimal working example

Copy this. Replace the tool and system prompt. You are on the network.

```python
import asyncio, os
from oflo_agent_protocol import BaseAgentV2
from oflo_agent_protocol.protocols.a2a.server import A2AServer
from oflo_agent_protocol.protocols.a2a.types import AgentCard, AgentSkill
from oflo_agent_protocol.protocols.mcp.server import MCPServer

agent = BaseAgentV2(
    name="MyAgent",
    system_prompt="You are a specialist in X.",
    project_id="my-project",
)

@agent.tool(description="Describe what this tool does")
async def my_tool(input: str) -> dict:
    return {"output": input.upper()}

card = AgentCard(
    name="MyAgent",
    description="Specialist in X.",
    url=os.getenv("AGENT_PUBLIC_URL", "http://localhost:9000"),
    skills=[AgentSkill(id="x", name="X skill", description="Does X.")],
)

async def main():
    a2a = A2AServer(card=card, agent=agent, port=9000)
    mcp = MCPServer(name="my-project", port=8080)
    mcp.register_agent(agent)
    await asyncio.gather(a2a.start(), mcp.start())

asyncio.run(main())
```

---

*Oflo Agent Protocol v2 · MIT License · https://github.com/ankitbuti/oflo-agent-protocol*
