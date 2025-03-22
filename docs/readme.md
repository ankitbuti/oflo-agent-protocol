# oflo-agent-protocol Documentation

## Overview

The `oflo-agent-protocol` is a framework designed for building and managing intelligent agents that can communicate and perform tasks based on a defined protocol. This project provides a structured way to create agents that can process messages, call functions, and maintain their state.

## Key Components

1. **Resource**: A wrapper around the MCP Resource that adds additional functionality, allowing agents to manage their metadata and descriptions.

2. **OfloAgentFactory**: A factory class responsible for creating agents of different types. It abstracts the instantiation logic and allows for easy addition of new agent types.

3. **BaseOfloAgent**: The base class for all agents, defining the essential properties and methods that any agent must implement, such as `initialize`, `process_message`, and `call_function`.

4. **OfloAgentStatus**: An enumeration that defines the possible statuses of an agent, such as ACTIVE, INACTIVE, and ERROR.

5. **OfloMessage**: A class representing messages exchanged between agents, including methods for converting messages to and from dictionary format.

## Getting Started

### Installation

To get started with the `oflo-agent-protocol`, clone the repository and install the required dependencies:

# Start Generation Here

### Example Usage

To use the `oflo-agent-protocol`, follow these steps:

1. **Create an Agent**: Use the `OfloAgentFactory` to create an agent of your desired type.

