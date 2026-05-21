# Oflo Agent Protocol — TypeScript & C# Integration Spec

> These agents are fully independent. They do not use the Python package.
> They implement the same wire protocol so oflo routes to them, they call
> back, and shared memory works across all three languages.

---

## Wire contracts (language-agnostic)

Everything below is plain JSON over HTTP. Three endpoints make you a node.

| Endpoint | Purpose |
|---|---|
| `GET  /.well-known/agent.json` | Discovery — who you are, what you can do |
| `POST /` | A2A JSON-RPC — receive tasks from other agents |
| `POST /` (MCP port) | MCP JSON-RPC — expose your tools to other agents |

Both protocols use JSON-RPC 2.0: `{"jsonrpc":"2.0","id":"...","method":"...","params":{...}}`

---

## Part 1 — TypeScript / Node.js

### Install

```bash
npm install express uuid
npm install @anthropic-ai/sdk          # or openai, whichever you use
npm install composio-core               # Composio app connectors
npm install -D @types/express @types/uuid typescript ts-node
```

### 1.1 Types (copy this file verbatim)

```typescript
// oflo-protocol.ts
export interface AgentSkill {
  id: string;
  name: string;
  description: string;
  tags?: string[];
  examples?: string[];
  inputModes: string[];
  outputModes: string[];
}

export interface AgentCard {
  name: string;
  description: string;
  url: string;
  version: string;
  skills: AgentSkill[];
  defaultInputModes: string[];
  defaultOutputModes: string[];
  capabilities: {
    streaming: boolean;
    pushNotifications: boolean;
    stateTransitionHistory: boolean;
  };
  provider?: { organization: string };
}

export interface TextPart   { type: "text"; text: string; mimeType?: string }
export interface DataPart   { type: "data"; data: unknown; mimeType?: string }
export type Part = TextPart | DataPart;

export interface A2AMessage {
  role: "user" | "agent";
  parts: Part[];
  messageId: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export interface Artifact {
  artifactId: string;
  name: string;
  parts: Part[];
  description?: string;
  index: number;
  append: boolean;
  lastChunk: boolean;
}

export interface TaskStatus {
  state: "submitted" | "working" | "input-required" | "completed" | "failed" | "canceled";
  timestamp: string;
  message?: A2AMessage;
}

export interface A2ATask {
  id: string;
  sessionId?: string;
  status: TaskStatus;
  history: A2AMessage[];
  artifacts: Artifact[];
  metadata?: Record<string, unknown>;
}

export interface JsonRpcRequest {
  jsonrpc: "2.0";
  id: string | number;
  method: string;
  params: unknown;
}

export interface ToolDefinition {
  name: string;                        // "AgentName.tool_name"
  description: string;
  inputSchema: {
    type: "object";
    properties: Record<string, unknown>;
    required?: string[];
  };
  handler: (args: Record<string, unknown>) => Promise<unknown>;
}

// JSON-RPC helpers
export function rpcOk(id: unknown, result: unknown) {
  return { jsonrpc: "2.0", id, result };
}
export function rpcErr(id: unknown, code: number, message: string) {
  return { jsonrpc: "2.0", id, error: { code, message } };
}
```

### 1.2 A2A Server

```typescript
// a2a-server.ts
import express, { Request, Response } from "express";
import { v4 as uuid } from "uuid";
import {
  AgentCard, A2ATask, A2AMessage, Artifact,
  JsonRpcRequest, rpcOk, rpcErr
} from "./oflo-protocol";

// ── Your LLM call (swap for any provider) ──────────────────────────────────
async function callLLM(systemPrompt: string, userText: string): Promise<string> {
  const Anthropic = require("@anthropic-ai/sdk");
  const client = new Anthropic.Anthropic();
  const resp = await client.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 1024,
    system: systemPrompt,
    messages: [{ role: "user", content: userText }],
  });
  return resp.content[0].type === "text" ? resp.content[0].text : "";
}

// ── Task store ─────────────────────────────────────────────────────────────
const tasks = new Map<string, A2ATask>();

// ── AgentCard ──────────────────────────────────────────────────────────────
const AGENT_CARD: AgentCard = {
  name: "MyTsAgent",
  description: "TypeScript agent — describe what you do here.",
  url: process.env.AGENT_URL ?? "http://localhost:9000",
  version: "1.0.0",
  skills: [
    {
      id: "primary",
      name: "Primary skill",
      description: "What this agent does.",
      inputModes: ["text"],
      outputModes: ["text"],
    },
  ],
  defaultInputModes: ["text"],
  defaultOutputModes: ["text"],
  capabilities: { streaming: false, pushNotifications: false, stateTransitionHistory: true },
  provider: { organization: "YourOrg" },
};

const SYSTEM_PROMPT = "You are a TypeScript-based assistant. Be concise.";

// ── Express app ────────────────────────────────────────────────────────────
export function createA2AServer() {
  const app = express();
  app.use(express.json());

  // Discovery
  app.get("/.well-known/agent.json", (_req, res) => {
    res.json(AGENT_CARD);
  });

  app.get("/health", (_req, res) => {
    res.json({ status: "healthy", agent: AGENT_CARD.name });
  });

  // JSON-RPC dispatch
  app.post("/", async (req: Request, res: Response) => {
    const body: JsonRpcRequest = req.body;
    const { id, method, params } = body;

    try {
      switch (method) {
        case "tasks/send":      return res.json(await handleTaskSend(id, params as any));
        case "tasks/get":       return res.json(handleTaskGet(id, params as any));
        case "tasks/cancel":    return res.json(handleTaskCancel(id, params as any));
        default:                return res.json(rpcErr(id, -32601, `Method not found: ${method}`));
      }
    } catch (err: any) {
      return res.json(rpcErr(id, -32603, err.message ?? "Internal error"));
    }
  });

  // SSE streaming (simple poll-based; swap for real SSE if needed)
  app.get("/tasks/:taskId/stream", (req, res) => {
    const task = tasks.get(req.params.taskId);
    if (!task) { res.status(404).end(); return; }
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.write(`data: ${JSON.stringify(task)}\n\n`);
    res.end();
  });

  return app;
}

// ── Handlers ───────────────────────────────────────────────────────────────
async function handleTaskSend(id: unknown, params: any) {
  const taskId: string = params.id ?? uuid();
  const sessionId: string = params.sessionId ?? uuid();
  const userText: string = params.message?.parts?.[0]?.text ?? "";

  // Create task in "working" state
  const task: A2ATask = {
    id: taskId,
    sessionId,
    status: { state: "working", timestamp: new Date().toISOString() },
    history: [params.message],
    artifacts: [],
  };
  tasks.set(taskId, task);

  // Call LLM
  const reply = await callLLM(SYSTEM_PROMPT, userText);

  // Complete
  task.status = { state: "completed", timestamp: new Date().toISOString() };
  task.artifacts = [
    {
      artifactId: uuid(),
      name: "response",
      parts: [{ type: "text", text: reply }],
      index: 0,
      append: false,
      lastChunk: true,
    },
  ];
  task.history.push({
    role: "agent",
    parts: [{ type: "text", text: reply }],
    messageId: uuid(),
    timestamp: new Date().toISOString(),
  });

  return rpcOk(id, task);
}

function handleTaskGet(id: unknown, params: { id: string }) {
  const task = tasks.get(params.id);
  if (!task) return rpcErr(id, -32001, "Task not found");
  return rpcOk(id, task);
}

function handleTaskCancel(id: unknown, params: { id: string }) {
  const task = tasks.get(params.id);
  if (!task) return rpcErr(id, -32001, "Task not found");
  if (task.status.state === "completed" || task.status.state === "failed") {
    return rpcErr(id, -32002, "Task not cancelable");
  }
  task.status = { state: "canceled", timestamp: new Date().toISOString() };
  return rpcOk(id, task);
}

// ── Start ──────────────────────────────────────────────────────────────────
if (require.main === module) {
  const app = createA2AServer();
  app.listen(9000, () => console.log("A2A server: http://localhost:9000"));
}
```

### 1.3 MCP Server

```typescript
// mcp-server.ts
import express from "express";
import { rpcOk, rpcErr, ToolDefinition } from "./oflo-protocol";

const MCP_VERSION = "2024-11-05";
const AGENT_NAME  = "MyTsAgent";

// ── Register your tools here ───────────────────────────────────────────────
const tools: ToolDefinition[] = [
  {
    name: `${AGENT_NAME}.get_data`,
    description: "Fetch data by ID",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Record ID" },
      },
      required: ["id"],
    },
    handler: async ({ id }) => ({ id, value: "some result", timestamp: Date.now() }),
  },
  {
    name: `${AGENT_NAME}.process`,
    description: "Process text and return structured result",
    inputSchema: {
      type: "object",
      properties: {
        input: { type: "string", description: "Input text" },
        mode:  { type: "string", description: "Processing mode" },
      },
      required: ["input"],
    },
    handler: async ({ input, mode }) => ({
      processed: String(input).toUpperCase(),
      mode: mode ?? "default",
    }),
  },
];

const toolIndex = Object.fromEntries(tools.map(t => [t.name, t]));

export function createMCPServer() {
  const app = express();
  app.use(express.json());

  app.get("/health", (_req, res) => {
    res.json({ status: "healthy", protocol: MCP_VERSION, tools: tools.length });
  });

  app.get("/agents", (_req, res) => {
    res.json({ agents: [{ name: AGENT_NAME, tools: tools.map(t => t.name) }] });
  });

  app.post("/", async (req, res) => {
    const { id, method, params = {} } = req.body;

    try {
      switch (method) {
        case "initialize":
          return res.json(rpcOk(id, {
            protocolVersion: MCP_VERSION,
            capabilities: { tools: { listChanged: false }, resources: {}, prompts: {} },
            serverInfo: { name: AGENT_NAME, version: "1.0.0" },
          }));

        case "ping":
          return res.json(rpcOk(id, { pong: true }));

        case "tools/list":
          return res.json(rpcOk(id, {
            tools: tools.map(t => ({
              name: t.name,
              description: t.description,
              inputSchema: t.inputSchema,
            })),
          }));

        case "tools/call": {
          const { name, arguments: args = {} } = params as any;
          const tool = toolIndex[name];
          if (!tool) return res.json(rpcErr(id, -32602, `Tool '${name}' not found`));
          try {
            const result = await tool.handler(args);
            const text = typeof result === "string" ? result : JSON.stringify(result);
            return res.json(rpcOk(id, { content: [{ type: "text", text }], isError: false }));
          } catch (err: any) {
            return res.json(rpcOk(id, { content: [{ type: "text", text: err.message }], isError: true }));
          }
        }

        case "messages/create": {
          // Simple passthrough to LLM
          const { messages: msgs = [] } = params as any;
          const last = [...msgs].reverse().find((m: any) => m.role === "user");
          const userText = last?.content ?? "";
          // Replace with your LLM call
          const reply = `Echo: ${userText}`;
          return res.json(rpcOk(id, {
            id: `msg_${Date.now()}`,
            type: "message",
            role: "assistant",
            content: [{ type: "text", text: reply }],
            stop_reason: "end_turn",
          }));
        }

        default:
          return res.json(rpcErr(id, -32601, `Method not found: ${method}`));
      }
    } catch (err: any) {
      return res.json(rpcErr(id, -32603, err.message));
    }
  });

  return app;
}

if (require.main === module) {
  const app = createMCPServer();
  app.listen(8080, () => console.log("MCP server: http://localhost:8080"));
}
```

### 1.4 Call a Python/oflo agent from TypeScript

```typescript
// oflo-client.ts
import { v4 as uuid } from "uuid";

export async function discoverAgent(baseUrl: string) {
  const r = await fetch(`${baseUrl}/.well-known/agent.json`);
  return r.json();
}

export async function sendTask(baseUrl: string, message: string): Promise<string> {
  const taskId = uuid();
  const body = {
    jsonrpc: "2.0",
    id: uuid(),
    method: "tasks/send",
    params: {
      id: taskId,
      sessionId: uuid(),
      message: {
        role: "user",
        parts: [{ type: "text", text: message }],
        messageId: uuid(),
        timestamp: new Date().toISOString(),
      },
    },
  };

  const r = await fetch(baseUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  const task = data.result;

  // If still working, poll until complete
  if (task?.status?.state === "working") {
    return pollTask(baseUrl, taskId);
  }

  return task?.artifacts?.[0]?.parts?.[0]?.text ?? "";
}

async function pollTask(baseUrl: string, taskId: string, maxMs = 60_000): Promise<string> {
  const deadline = Date.now() + maxMs;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 1000));
    const r = await fetch(baseUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        jsonrpc: "2.0", id: uuid(), method: "tasks/get",
        params: { id: taskId },
      }),
    });
    const data = await r.json();
    const task = data.result;
    const state = task?.status?.state;
    if (state === "completed") return task?.artifacts?.[0]?.parts?.[0]?.text ?? "";
    if (state === "failed" || state === "canceled") throw new Error(`Task ${state}`);
  }
  throw new Error("Task timed out");
}

export async function callMcpTool(
  mcpBaseUrl: string,
  toolName: string,               // "AgentName.tool_name"
  args: Record<string, unknown>
): Promise<unknown> {
  const r = await fetch(mcpBaseUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0", id: uuid(), method: "tools/call",
      params: { name: toolName, arguments: args },
    }),
  });
  const data = await r.json();
  const text = data?.result?.content?.[0]?.text ?? "";
  try { return JSON.parse(text); } catch { return text; }
}
```

### 1.5 Redis shared memory from TypeScript

```typescript
// redis-memory.ts
const REDIS_URL = process.env.REDIS_MEMORY_URL ?? "http://localhost:8000";

export async function appendToWorkingMemory(
  sessionId: string, role: "user" | "assistant", content: string
) {
  const r = await fetch(`${REDIS_URL}/v1/working-memory/${sessionId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role, content }),
  });
  return r.ok;
}

export async function getWorkingMemory(sessionId: string) {
  const r = await fetch(`${REDIS_URL}/v1/working-memory/${sessionId}`);
  if (!r.ok) return null;
  return r.json();
}

export async function searchLongTermMemory(
  query: string, sessionId: string, limit = 5
): Promise<any[]> {
  const r = await fetch(`${REDIS_URL}/v1/long-term-memory/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, session_id: sessionId, limit }),
  });
  if (!r.ok) return [];
  const data = await r.json();
  return data.results ?? [];
}

export async function addLongTermMemory(
  text: string, agentId: string, sessionId: string, topics?: string[]
) {
  await fetch(`${REDIS_URL}/v1/long-term-memory/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      agent_id: agentId,
      session_id: sessionId,
      memory_type: "conversation",
      topics: topics ?? [],
    }),
  });
}
```

### 1.6 Composio from TypeScript

```typescript
// composio-tools.ts
import { OpenAIToolSet } from "composio-core";

const toolset = new OpenAIToolSet({ apiKey: process.env.COMPOSIO_API_KEY });

// Get tools as OpenAI-format function specs
export async function getComposioTools(apps: string[]) {
  return toolset.getTools({ apps });
}

// Execute after LLM returns a tool_call
export async function executeComposioAction(
  actionName: string,
  args: Record<string, unknown>
) {
  return toolset.executeToolCall(
    { name: actionName, arguments: JSON.stringify(args) },
    process.env.COMPOSIO_USER_ID ?? "default"
  );
}

// Initiate OAuth for a user
export async function connectApp(app: string, userId: string) {
  const entity = await toolset.client.getEntity(userId);
  const req    = await entity.initiateConnection({ appName: app });
  return req.redirectUrl;
}
```

### 1.7 Entry point — run both servers

```typescript
// index.ts
import { createA2AServer } from "./a2a-server";
import { createMCPServer }  from "./mcp-server";

const A2A_PORT = parseInt(process.env.A2A_PORT ?? "9000");
const MCP_PORT = parseInt(process.env.MCP_PORT ?? "8080");

createA2AServer().listen(A2A_PORT, () =>
  console.log(`A2A  http://localhost:${A2A_PORT}`)
);

createMCPServer().listen(MCP_PORT, () =>
  console.log(`MCP  http://localhost:${MCP_PORT}`)
);
```

---

## Part 2 — C# / ASP.NET Core

### NuGet packages

```xml
<ItemGroup>
  <PackageReference Include="Microsoft.AspNetCore.App" />
  <PackageReference Include="System.Text.Json"         Version="8.*" />
  <PackageReference Include="Anthropic.SDK"             Version="3.*" />  <!-- or OpenAI -->
  <PackageReference Include="Composio.SDK"              Version="*"   />  <!-- if available -->
</ItemGroup>
```

### 2.1 Protocol types

```csharp
// OfloProtocol.cs
using System.Text.Json;
using System.Text.Json.Serialization;

namespace OfloProtocol;

public record AgentSkill(
    string Id, string Name, string Description,
    string[] InputModes, string[] OutputModes,
    string[]? Tags = null
);

public record AgentCapabilities(
    bool Streaming = false,
    bool PushNotifications = false,
    bool StateTransitionHistory = true
);

public record AgentCard(
    string Name, string Description, string Url, string Version,
    AgentSkill[] Skills,
    string[] DefaultInputModes, string[] DefaultOutputModes,
    AgentCapabilities Capabilities
);

public record TextPart(string Text, string MimeType = "text/plain")
{
    [JsonPropertyName("type")] public string Type => "text";
}

public record A2AMessage(
    string Role,                          // "user" | "agent"
    TextPart[] Parts,
    string MessageId,
    string Timestamp,
    Dictionary<string, object>? Metadata = null
);

public record Artifact(
    string ArtifactId, string Name, TextPart[] Parts,
    int Index = 0, bool Append = false, bool LastChunk = true
);

public record TaskStatus(
    string State,                         // submitted|working|completed|failed|canceled
    string Timestamp,
    A2AMessage? Message = null
);

public record A2ATask(
    string Id, string? SessionId,
    TaskStatus Status,
    List<A2AMessage> History,
    List<Artifact> Artifacts,
    Dictionary<string, object>? Metadata = null
);

public record JsonRpcRequest(
    [property: JsonPropertyName("jsonrpc")] string Jsonrpc,
    [property: JsonPropertyName("id")]      object? Id,
    [property: JsonPropertyName("method")]  string Method,
    [property: JsonPropertyName("params")]  JsonElement Params
);

public static class Rpc
{
    public static object Ok(object? id, object result)
        => new { jsonrpc = "2.0", id, result };

    public static object Err(object? id, int code, string message)
        => new { jsonrpc = "2.0", id, error = new { code, message } };
}
```

### 2.2 A2A Server

```csharp
// A2AServer.cs
using OfloProtocol;
using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddSingleton<TaskStore>();
builder.Services.AddSingleton<LlmService>();
var app = builder.Build();

// ── AgentCard ──────────────────────────────────────────────────────────────
var card = new AgentCard(
    Name:               "MyCSharpAgent",
    Description:        "C# agent — describe what you do here.",
    Url:                Environment.GetEnvironmentVariable("AGENT_URL") ?? "http://localhost:9000",
    Version:            "1.0.0",
    Skills:             [
        new AgentSkill(
            Id: "primary", Name: "Primary skill",
            Description: "What this agent does.",
            InputModes: ["text"], OutputModes: ["text"]
        )
    ],
    DefaultInputModes:  ["text"],
    DefaultOutputModes: ["text"],
    Capabilities:       new AgentCapabilities()
);

// Discovery
app.MapGet("/.well-known/agent.json", () => Results.Json(card));
app.MapGet("/health", () => Results.Json(new { status = "healthy", agent = card.Name }));

// JSON-RPC dispatch
app.MapPost("/", async (JsonRpcRequest req, TaskStore store, LlmService llm) =>
{
    var id = req.Id;
    try
    {
        return req.Method switch
        {
            "tasks/send"   => Results.Json(await HandleTaskSend(id, req.Params, store, llm)),
            "tasks/get"    => Results.Json(HandleTaskGet(id, req.Params, store)),
            "tasks/cancel" => Results.Json(HandleTaskCancel(id, req.Params, store)),
            _              => Results.Json(Rpc.Err(id, -32601, $"Method not found: {req.Method}"))
        };
    }
    catch (Exception ex)
    {
        return Results.Json(Rpc.Err(id, -32603, ex.Message));
    }
});

app.Run("http://0.0.0.0:9000");

// ── Handlers ───────────────────────────────────────────────────────────────
static async Task<object> HandleTaskSend(
    object? id, JsonElement p, TaskStore store, LlmService llm)
{
    var taskId    = p.TryGetString("id")        ?? Guid.NewGuid().ToString();
    var sessionId = p.TryGetString("sessionId") ?? Guid.NewGuid().ToString();
    var userText  = p.GetProperty("message")
                     .GetProperty("parts")[0]
                     .GetProperty("text").GetString() ?? "";

    var task = new A2ATask(taskId, sessionId,
        new TaskStatus("working", DateTime.UtcNow.ToString("O")),
        [], []);
    store.Set(taskId, task);

    var reply = await llm.CallAsync(userText);

    var completed = task with
    {
        Status    = new TaskStatus("completed", DateTime.UtcNow.ToString("O")),
        Artifacts = [ new Artifact(
            Guid.NewGuid().ToString(), "response",
            [new TextPart(reply)]) ],
        History   = [
            new A2AMessage("user",  [new TextPart(userText)],
                Guid.NewGuid().ToString(), DateTime.UtcNow.ToString("O")),
            new A2AMessage("agent", [new TextPart(reply)],
                Guid.NewGuid().ToString(), DateTime.UtcNow.ToString("O")),
        ]
    };
    store.Set(taskId, completed);
    return Rpc.Ok(id, completed);
}

static object HandleTaskGet(object? id, JsonElement p, TaskStore store)
{
    var taskId = p.GetProperty("id").GetString() ?? "";
    var task   = store.Get(taskId);
    return task is null ? Rpc.Err(id, -32001, "Task not found") : Rpc.Ok(id, task);
}

static object HandleTaskCancel(object? id, JsonElement p, TaskStore store)
{
    var taskId = p.GetProperty("id").GetString() ?? "";
    var task   = store.Get(taskId);
    if (task is null) return Rpc.Err(id, -32001, "Task not found");
    if (task.Status.State is "completed" or "failed")
        return Rpc.Err(id, -32002, "Task not cancelable");
    var canceled = task with { Status = new TaskStatus("canceled", DateTime.UtcNow.ToString("O")) };
    store.Set(taskId, canceled);
    return Rpc.Ok(id, canceled);
}

// ── Support classes ────────────────────────────────────────────────────────
public class TaskStore
{
    private readonly Dictionary<string, A2ATask> _tasks = new();
    public void     Set(string id, A2ATask task) => _tasks[id] = task;
    public A2ATask? Get(string id) => _tasks.GetValueOrDefault(id);
}

public class LlmService
{
    private readonly string _systemPrompt = "You are a C# assistant. Be concise.";

    public async Task<string> CallAsync(string userText)
    {
        // Replace with Anthropic.SDK, Azure OpenAI, or any provider
        // Example using Anthropic.SDK:
        //
        // var client   = new AnthropicClient(Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY"));
        // var messages = new List<Message> { new(RoleType.User, userText) };
        // var req      = new MessageParameters {
        //     Model    = AnthropicModels.Claude3Haiku,
        //     MaxTokens = 1024,
        //     System   = _systemPrompt,
        //     Messages = messages,
        // };
        // var resp = await client.Messages.GetClaudeMessageAsync(req);
        // return resp.Content.First().Text;

        await Task.Delay(10); // placeholder
        return $"[C# agent echo] {userText}";
    }
}

// Extension helper
public static class JsonElementExtensions
{
    public static string? TryGetString(this JsonElement el, string key)
        => el.TryGetProperty(key, out var v) ? v.GetString() : null;
}
```

### 2.3 MCP Server (separate port)

```csharp
// McpServer.cs
using OfloProtocol;
using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

const string AgentName  = "MyCSharpAgent";
const string McpVersion = "2024-11-05";

// ── Tool registry ──────────────────────────────────────────────────────────
var toolRegistry = new Dictionary<string, Func<JsonElement, Task<object>>>
{
    [$"{AgentName}.get_data"] = async args =>
    {
        var id = args.TryGetString("id") ?? "unknown";
        await Task.Delay(1);
        return new { id, value = "result from C# agent", ts = DateTimeOffset.UtcNow };
    },
    [$"{AgentName}.process"] = async args =>
    {
        var input = args.TryGetString("input") ?? "";
        var mode  = args.TryGetString("mode")  ?? "default";
        await Task.Delay(1);
        return new { processed = input.ToUpper(), mode };
    },
};

var toolSchemas = new[]
{
    new
    {
        name        = $"{AgentName}.get_data",
        description = "Fetch a record by ID",
        inputSchema = new {
            type       = "object",
            properties = new { id = new { type = "string", description = "Record ID" } },
            required   = new[] { "id" }
        }
    },
    new
    {
        name        = $"{AgentName}.process",
        description = "Process text and return structured result",
        inputSchema = new {
            type       = "object",
            properties = new {
                input = new { type = "string", description = "Input text" },
                mode  = new { type = "string", description = "Processing mode" }
            },
            required = new[] { "input" }
        }
    },
};

// ── Endpoints ──────────────────────────────────────────────────────────────
app.MapGet("/health",  () => Results.Json(new { status = "healthy", protocol = McpVersion }));
app.MapGet("/agents",  () => Results.Json(new { agents = new[] {
    new { name = AgentName, tools = toolSchemas.Select(t => t.name) }
}}));

app.MapPost("/", async (JsonRpcRequest req) =>
{
    var id = req.Id;
    switch (req.Method)
    {
        case "initialize":
            return Results.Json(Rpc.Ok(id, new {
                protocolVersion = McpVersion,
                capabilities    = new { tools = new { listChanged = false } },
                serverInfo      = new { name = AgentName, version = "1.0.0" },
            }));

        case "ping":
            return Results.Json(Rpc.Ok(id, new { pong = true }));

        case "tools/list":
            return Results.Json(Rpc.Ok(id, new { tools = toolSchemas }));

        case "tools/call":
        {
            var name = req.Params.TryGetString("name") ?? "";
            var args = req.Params.TryGetProperty("arguments", out var a) ? a : default;

            if (!toolRegistry.TryGetValue(name, out var handler))
                return Results.Json(Rpc.Err(id, -32602, $"Tool '{name}' not found"));

            try
            {
                var result = await handler(args);
                var text   = System.Text.Json.JsonSerializer.Serialize(result);
                return Results.Json(Rpc.Ok(id, new {
                    content = new[] { new { type = "text", text } },
                    isError = false
                }));
            }
            catch (Exception ex)
            {
                return Results.Json(Rpc.Ok(id, new {
                    content = new[] { new { type = "text", text = ex.Message } },
                    isError = true
                }));
            }
        }

        default:
            return Results.Json(Rpc.Err(id, -32601, $"Method not found: {req.Method}"));
    }
});

app.Run("http://0.0.0.0:8080");
```

### 2.4 Call a Python/oflo agent from C#

```csharp
// OfloClient.cs
using System.Net.Http.Json;

public static class OfloClient
{
    private static readonly HttpClient Http = new();

    public static async Task<string> DiscoverAgentAsync(string baseUrl)
    {
        var card = await Http.GetStringAsync($"{baseUrl}/.well-known/agent.json");
        return card;
    }

    public static async Task<string> SendTaskAsync(string baseUrl, string message)
    {
        var taskId    = Guid.NewGuid().ToString();
        var sessionId = Guid.NewGuid().ToString();

        var body = new
        {
            jsonrpc = "2.0",
            id      = Guid.NewGuid().ToString(),
            method  = "tasks/send",
            @params = new
            {
                id        = taskId,
                sessionId = sessionId,
                message   = new
                {
                    role    = "user",
                    parts   = new[] { new { type = "text", text = message } },
                    messageId  = Guid.NewGuid().ToString(),
                    timestamp  = DateTime.UtcNow.ToString("O"),
                }
            }
        };

        var resp = await Http.PostAsJsonAsync(baseUrl, body);
        var data = await resp.Content.ReadFromJsonAsync<JsonElement>();
        var state = data.GetProperty("result").GetProperty("status")
                        .GetProperty("state").GetString();

        if (state == "working")
            return await PollTaskAsync(baseUrl, taskId);

        return ExtractArtifactText(data.GetProperty("result"));
    }

    private static async Task<string> PollTaskAsync(
        string baseUrl, string taskId, int maxSeconds = 60)
    {
        var deadline = DateTime.UtcNow.AddSeconds(maxSeconds);
        while (DateTime.UtcNow < deadline)
        {
            await Task.Delay(1000);
            var body = new
            {
                jsonrpc = "2.0",
                id      = Guid.NewGuid().ToString(),
                method  = "tasks/get",
                @params = new { id = taskId }
            };
            var resp  = await Http.PostAsJsonAsync(baseUrl, body);
            var data  = await resp.Content.ReadFromJsonAsync<JsonElement>();
            var task  = data.GetProperty("result");
            var state = task.GetProperty("status").GetProperty("state").GetString();
            if (state == "completed") return ExtractArtifactText(task);
            if (state is "failed" or "canceled") throw new Exception($"Task {state}");
        }
        throw new TimeoutException("Task did not complete in time");
    }

    public static async Task<object?> CallMcpToolAsync(
        string mcpBaseUrl, string toolName, object arguments)
    {
        var body = new
        {
            jsonrpc = "2.0",
            id      = Guid.NewGuid().ToString(),
            method  = "tools/call",
            @params = new { name = toolName, arguments }
        };
        var resp = await Http.PostAsJsonAsync(mcpBaseUrl, body);
        var data = await resp.Content.ReadFromJsonAsync<JsonElement>();
        var text = data.GetProperty("result")
                       .GetProperty("content")[0]
                       .GetProperty("text").GetString() ?? "";
        try { return System.Text.Json.JsonSerializer.Deserialize<object>(text); }
        catch { return text; }
    }

    private static string ExtractArtifactText(JsonElement task)
    {
        try
        {
            return task.GetProperty("artifacts")[0]
                       .GetProperty("parts")[0]
                       .GetProperty("text").GetString() ?? "";
        }
        catch { return ""; }
    }
}
```

### 2.5 Redis shared memory from C#

```csharp
// RedisMemory.cs
using System.Net.Http.Json;

public class RedisMemoryClient(string sessionId, string baseUrl = "http://localhost:8000")
{
    private static readonly HttpClient Http = new();

    public async Task AppendAsync(string role, string content)
    {
        await Http.PostAsJsonAsync(
            $"{baseUrl}/v1/working-memory/{sessionId}/messages",
            new { role, content }
        );
    }

    public async Task<JsonElement?> GetWorkingMemoryAsync()
    {
        var r = await Http.GetAsync($"{baseUrl}/v1/working-memory/{sessionId}");
        if (!r.IsSuccessStatusCode) return null;
        return await r.Content.ReadFromJsonAsync<JsonElement>();
    }

    public async Task<JsonElement[]> SearchLongTermAsync(string query, int limit = 5)
    {
        var r = await Http.PostAsJsonAsync(
            $"{baseUrl}/v1/long-term-memory/search",
            new { query, session_id = sessionId, limit }
        );
        if (!r.IsSuccessStatusCode) return [];
        var data = await r.Content.ReadFromJsonAsync<JsonElement>();
        return data.TryGetProperty("results", out var res)
            ? res.EnumerateArray().ToArray()
            : [];
    }

    public async Task AddLongTermAsync(
        string text, string agentId, string[] topics)
    {
        await Http.PostAsJsonAsync($"{baseUrl}/v1/long-term-memory/", new {
            text,
            agent_id    = agentId,
            session_id  = sessionId,
            memory_type = "conversation",
            topics,
        });
    }
}
```

---

## Part 3 — Verification checklist (both languages)

Run these checks before connecting to the oflo network.

```bash
# 1. AgentCard is valid JSON at the right URL
curl http://localhost:9000/.well-known/agent.json | jq .name

# 2. A2A: send a task
curl -s -X POST http://localhost:9000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0","id":"1","method":"tasks/send",
    "params":{
      "id":"test-task","sessionId":"s1",
      "message":{"role":"user","parts":[{"type":"text","text":"ping"}],
                 "messageId":"m1","timestamp":"2025-01-01T00:00:00Z"}
    }
  }' | jq .result.status.state
# expected: "completed"

# 3. MCP: initialize handshake
curl -s -X POST http://localhost:8080/ \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"1","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{}}}' \
  | jq .result.protocolVersion
# expected: "2024-11-05"

# 4. MCP: list tools
curl -s -X POST http://localhost:8080/ \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"2","method":"tools/list","params":{}}' \
  | jq '.result.tools[0].name'
# expected: "AgentName.tool_name"

# 5. MCP: call a tool
curl -s -X POST http://localhost:8080/ \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"3","method":"tools/call","params":{"name":"AgentName.get_data","arguments":{"id":"abc"}}}' \
  | jq .result.isError
# expected: false

# 6. Health checks
curl http://localhost:9000/health | jq .status   # "healthy"
curl http://localhost:8080/health | jq .status   # "healthy"
```

---

## Part 4 — What oflo does for you automatically

Once your A2A and MCP servers pass the checks above, the oflo network provides:

| Feature | How you get it |
|---|---|
| **Routing** | Python agents call your A2A endpoint via `A2AClient` |
| **Tool discovery** | Python agents call your MCP `/tools/list` and execute via `/tools/call` |
| **Shared memory** | All agents on the same Redis session see each other's working memory |
| **Audit** | The oflo `AgentManager` logs every cross-agent call in its JSONL audit |
| **Composio** | Use the JS SDK (`composio-core`) or C# HTTP client — same REST API |
| **Multi-agent chain** | A Python `AgentManager.chain()` can include your agent by URL |

You do not need to register. Agents find each other via:
- The `AgentCard` URL you publish
- Direct `A2AClient("http://your-agent:9000")` calls from Python agents
- Direct `MCPClient("http://your-agent:8080")` calls for tool access

---

*Oflo Agent Protocol v2 — https://github.com/ankitbuti/oflo-agent-protocol*
