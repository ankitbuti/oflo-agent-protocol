/// <reference types="@cloudflare/workers-types" />
import type { ExecutionContext, KVNamespace } from '@cloudflare/workers-types';
import { spawn } from 'child_process';

interface Env {
  AGENT_STATE: KVNamespace;
  ENVIRONMENT: string;
}

interface AgentResponse {
  role: string;
  content: string;
  timestamp: string;
}

async function runPythonAgent(message: string): Promise<AgentResponse> {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python3', [
      '-c',
      `
import asyncio
import json
from oflo_agent_protocol.examples.ping_pong import PingPongAgent

async def run():
    agent = PingPongAgent()
    await agent.initialize()
    response = await agent.process_message("${message}")
    print(json.dumps(response.to_dict()))
    await agent.terminate()

asyncio.run(run())
      `
    ]);

    let output = '';
    let error = '';

    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          const response = JSON.parse(output);
          resolve(response);
        } catch (e) {
          reject(new Error(`Failed to parse agent response: ${e instanceof Error ? e.message : String(e)}`));
        }
      } else {
        reject(new Error(`Agent process failed: ${error}`));
      }
    });
  });
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    // Handle CORS
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
      });
    }

    try {
      if (request.method === 'POST') {
        const body = await request.json() as { message?: string };
        const message = body.message;

        if (!message) {
          return new Response(JSON.stringify({ error: 'Message is required' }), {
            status: 400,
            headers: { 'Content-Type': 'application/json' },
          });
        }

        // Process message using Python agent
        const response = await runPythonAgent(message);

        // Store conversation state if needed
        if (env.ENVIRONMENT === 'production') {
          await env.AGENT_STATE.put(
            `conversation:${Date.now()}`,
            JSON.stringify({
              message,
              response: response.content,
              timestamp: new Date().toISOString(),
            })
          );
        }

        return new Response(JSON.stringify(response), {
          headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
          },
        });
      }

      // Handle GET request - return agent status
      return new Response(
        JSON.stringify({
          status: 'active',
          name: 'PingPongAgent',
          version: '0.1.0',
        }),
        {
          headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
          },
        }
      );
    } catch (error: unknown) {
      console.error('Error:', error);
      return new Response(
        JSON.stringify({
          error: 'Internal Server Error',
          message: error instanceof Error ? error.message : 'Unknown error',
        }),
        {
          status: 500,
          headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
          },
        }
      );
    }
  },

  // Handle scheduled tasks
  async scheduled(event: ScheduledEvent, env: Env, ctx: ExecutionContext): Promise<void> {
    // Clean up old conversation states
    const now = Date.now();
    const oneWeekAgo = now - 7 * 24 * 60 * 60 * 1000;

    const list = await env.AGENT_STATE.list({ prefix: 'conversation:' });
    for (const key of list.keys) {
      const timestamp = parseInt(key.name.split(':')[1]);
      if (timestamp < oneWeekAgo) {
        await env.AGENT_STATE.delete(key.name);
      }
    }
  },
}; 