# Bitext Customer Service Data Analyst Agent

This repository contains a multi-agent, asynchronous command-line interface (CLI) application built with LangGraph and the Model Context Protocol (MCP). It functions as a specialized data analyst capable of querying customer service datasets, maintaining persistent conversational state, and extracting long-term user profiles in the background.

## Architecture Overview

The system architecture is divided into three primary components:

1. **Stateful Graph Routing (LangGraph):** The core intelligence routes queries between three distinct nodes:
    * **Structured Node:** Executes strict data filtering and calculation tools sequentially.
    * **Unstructured Node:** Handles open-ended data exploration and summarization.
    * **Out-of-Scope Node:** A strict guardrail that deflects non-analytical queries.
2. **Persistent Memory Layer:** Utilizes asynchronous SQLite (`aiosqlite`) to maintain conversational state (`checkpoints.sqlite`). A secondary background process utilizes `asyncio.Lock()` to concurrently extract and save a dynamic user profile to a local JSON file without blocking the main event loop.
3. **Model Context Protocol (MCP) Server:** Tools are entirely decoupled from the main agent logic. A local FastMCP server exposes database operations, which the LangGraph agent connects to via standard I/O (`stdio`).

### Defined Tools
The MCP server exposes the following analytical tools to the agent:
* `mcp_filter_dataset`: Filters the dataset down to rows that match a specific criteria (e.g., intent or category).
* `mcp_count_rows`: Counts the total number of rows in the currently active dataset.
* `mcp_get_distribution`: Returns a breakdown of all unique values and their exact counts for a specific column.
* `mcp_get_samples`: Retrieves random sample user instructions from a specific category.
* `mcp_get_unique_values`: Returns a list of all valid categories and intents to guide filtering.

## Model Selection

**Primary Model:** `meta-llama/Llama-3.3-70B-Instruct` (via Nebius Token Factory)

**Justification:** This 70B parameter model was selected as the unified driver for both the main conversational agent and the background memory extractor. 
* **Tool Calling Reliability:** The Llama-3.3-70B model exhibits exceptional adherence to strict tool-calling sequences (such as the ReAct paradigm) and reliably follows negative constraints (e.g., avoiding parallel tool execution). 
* **Structured Output:** The background memory layer requires strict adherence to Pydantic JSON schemas to update the user profile. The 70B model provides the necessary reasoning depth to accurately deduce analytical interests from chat history without breaking the required schema structure.

## Setup Instructions

Follow these steps to clone the repository and run the agent.

**1. Clone the repository**
git clone https://github.com/SeanGl76/customer-service-agent.git
cd customer-service-agent

**2. Install dependencies**
Ensure you have Python 3.10 or higher installed.
pip install -r requirements.txt

**3. Configure Environment Variables**
Create a `.env` file in the root directory of the project and define your Nebius Token Factory credentials:
NEBIUS_API_KEY=your_nebius_api_key_here
DEFAULT_SESSION_ID=default_test_session

## Running the Application

Because the architecture utilizes the Model Context Protocol, the tool server and the agent client run as separate processes. You will need two terminal windows.

**Terminal 1: Start the MCP Server**
Run this command and leave the terminal running in the background. This establishes the local tool registry.
python -m server.mcp_server

**Terminal 2: Start the Agent Client**
In a new terminal, launch the main application. This will connect to the MCP server and initiate the interactive CLI.
python main.py

*Note: To resume a previous conversation and load an existing memory profile, pass a custom session ID:*
python main.py --session my_custom_session

## Connecting an External Client via MCP

The tools in this project are exposed using the standard Model Context Protocol over `stdio`. Any external MCP-compatible client can connect to the server and utilize its tools independently of the LangGraph agent.

To connect a custom Python client to the tools, initialize a subprocess connection to the server script using the `mcp` library:

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_client():
    # Define the command to boot the server as a subprocess
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "server.mcp_server"] 
    )
    
    # Establish the stdio transport connection
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # The client can now list and execute tools
            tools = await session.list_tools()
            print("Available Tools:", tools)

if __name__ == "__main__":
    asyncio.run(run_client())