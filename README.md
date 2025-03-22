# Oflo AI Agent Protocol

A powerful protocol for building business AI agents with seamless integration with MCP (Multi-Cloud Protocol), supporting multiple LLMs and multi-modal capabilities.

## Features

- 🤖 **Flexible Agent Architecture**: Build custom AI agents with a standardized interface
- 🔌 **MCP Integration**: Seamless integration with Multi-Cloud Protocol
- 🧠 **Multi-LLM Support**: Works with OpenAI, Anthropic, and other LLM providers
- 📱 **Multi-Modal**: Handle text, images, and other data types
- 🏭 **Agent Factory**: Easy agent creation and management
- 🔄 **State Management**: Built-in memory and conversation history
- 🛠️ **Tool Integration**: Easy integration with external tools and APIs
- 🔒 **Security**: Built-in security features and best practices

## Installation

```bash
pip install oflo-ai-agent-protocol
```

For development:
```bash
pip install oflo-ai-agent-protocol[dev]
```

For documentation:
```bash
pip install oflo-ai-agent-protocol[docs]
```

## Quick Start

```python
from oflo_agent_protocol import BaseAgent, Message

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="MyAgent", purpose="A custom business agent")
        
    async def process_message(self, message: Message) -> Message:
        # Your custom message processing logic here
        return Message(role="assistant", content="Hello from MyAgent!")

# Create and initialize the agent
agent = MyAgent()
await agent.initialize()

# Process a message
response = await agent.process_message("Hello!")
print(response.content)  # Output: Hello from MyAgent!
```

## Examples

Check out our example implementations:

- [Ping Pong Agent](examples/ping-pong-oflo/): Simple example of agent communication
- [Marketing Agent](examples/marketing-oflo/): AI agent for marketing tasks
- [Sales Agent](examples/sales-oflo/): AI agent for sales automation
- [Revenue Agent](examples/revenue-oflo/): AI agent for revenue optimization
- [Trader Agent](examples/trader-oflo/): AI agent for trading strategies

## Cloudflare Worker Deployment

Deploy your Oflo Agent as a Cloudflare Worker:

1. Install Wrangler:
```bash
npm install -g wrangler
```

2. Configure your worker:
```bash
cd examples/marketing-oflo/worker
wrangler dev  # For local development
wrangler publish  # For production deployment
```

## Documentation

Full documentation is available at [https://docs.oflo.ai/agent-protocol](https://docs.oflo.ai/agent-protocol)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📚 [Documentation](https://docs.oflo.ai/agent-protocol)
- 💬 [Discord Community](https://discord.gg/oflo)
- 🐦 [Twitter](https://twitter.com/ofloai)
- 📧 [Email Support](mailto:support@oflo.ai) 
