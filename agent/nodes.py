import os
import asyncio
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from agent.state import AgentState
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from contextlib import AsyncExitStack

load_dotenv()

exit_stack = AsyncExitStack()
cached_mcp_tools = None

llm = ChatOpenAI(
    model="meta-llama/Llama-3.3-70B-Instruct"   ,
    api_key=os.environ.get("NEBIUS_API_KEY", "YOUR_KEY_HERE"),
    base_url="https://api.studio.nebius.ai/v1/",
    temperature=0.1
)

async def get_mcp_tools():
    """
    Connects to the FastMCP server, initializes the session, and caches the tools.
    The connection stays alive in the background for the duration of the chat loop.
    """
    global cached_mcp_tools, exit_stack
    
    # If tools are already loaded, return them instantly to save network calls
    if cached_mcp_tools is not None:
        return cached_mcp_tools
        
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "server.mcp_server"] 
    )
    
    # 1. Establish the I/O connection and keep it open
    stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
    read, write = stdio_transport
    
    # 2. Initialize the MCP protocol session
    session = await exit_stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    
    # 3. Load the tools using the LIVE session (This fixes the list_tools error)
    cached_mcp_tools = await load_mcp_tools(session)
    
    return cached_mcp_tools

# 2. The Router Logic
class RouteOutput(BaseModel):
    
    destination: Literal["structured", "unstructured", "out_of_scope"] = Field(
        description="The target pipeline based on the user query."
    )

def route_query(state: AgentState) -> str:
    
    latest_user_message = state["messages"][-1].content
    
    structured_llm = llm.with_structured_output(RouteOutput)
    
    system_prompt = """You are a classification router for a customer service data analyst.
    - If the query asks for counts, distributions, specific rows, or numbers, output 'structured'.
    - If the query asks for summaries, themes, or text extraction, output 'unstructured'.
    - If the query asks about sports, poetry, or general world knowledge outside of a customer service dataset, output 'out_of_scope'."""
    
    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content =latest_user_message)
    ])
    
    if result.destination == "structured":
        return "structured_node"
    elif result.destination == "unstructured":
        return "unstructured_node"
    else:
        return "out_of_scope_node"

async def structured_agent(state: dict):
    """
    Fetches tools from the MCP server, binds them to the LLM, and evaluates the user's request.
    """
    mcp_tools = await get_mcp_tools()
    
    agent_llm = llm.bind_tools(mcp_tools, parallel_tool_calls=False)
    
    system_prompt = SystemMessage(content=(
        "You are a strict data analyst. You have access to a remote database via tools. "
        "You must filter the dataset before counting rows. Do not guess or hallucinate numbers. "
        "CRITICAL TOOL INSTRUCTIONS: You must execute tools strictly one at a time. "
        "DO NOT nest tool calls. DO NOT pass a function name as an argument to another function. "
        "Step 1: Call 'mcp_filter_dataset' and wait for the observation. "
        "Step 2: Only after filtering is successful, call 'mcp_count_rows'."
        "CRITICAL FORMATTING RULE: If the user explicitly asks to 'show examples', "
        "'list rows', 'see the data', and etc, you MUST quote the actual text of those "
        "examples in your final response. Do not just summarize or say you found them. "
        "Explicitly print the raw text of the examples as bullet points."
    ))

    # Adding a message trimmer to handle cases where there are already active 50 messages
    # that were sent by the user, therefore only sending 10 to not bloat the LLM context window.
    trimmed_history = trim_messages(
        state["messages"],
        max_tokens=10, 
        strategy="last",
        token_counter=len,
        include_system=False
    )
    
    messages = [system_prompt] + trimmed_history
    response = await agent_llm.ainvoke(messages)
    
    return {"messages": [response]}


async def unstructured_agent(state: dict):
    """
    The qualitative worker. Operates similarly but with a prompt tuned for text summarization.
    """
    mcp_tools = await get_mcp_tools()
    agent_llm = llm.bind_tools(mcp_tools, parallel_tool_calls=False)
    
    system_prompt = SystemMessage(content=(
        "You are an exploratory data analyst. The user has asked an open-ended question. "
        "Your goal is to provide a brief, high-level overview. "
        "CRITICAL INSTRUCTIONS: "
        "1. Call a MAXIMUM of 2 exploratory tools (e.g., get_samples or get_distribution). "
        "2. DO NOT attempt to map the entire database or call tools endlessly. "
        "3. Once you have a general feel for the data, STOP calling tools immediately. "
        "4. Summarize your brief findings directly to the user."
        "CRITICAL FORMATTING RULE: If the user explicitly asks to 'show examples', "
        "'list rows', or 'see the data', you MUST quote the actual text of those "
        "examples in your final response. Do not just summarize or say you found them. "
        "Explicitly print the raw text of the examples as bullet points."
    ))
    
    trimmed_history = trim_messages(
        state["messages"],
        max_tokens=10, 
        strategy="last",
        token_counter=len,
        include_system=False
    )
    
    messages = [system_prompt] + trimmed_history
    response = await agent_llm.ainvoke(messages)
    
    return {"messages": [response]}


def out_of_scope_worker(state: dict):
    """Handles off-topic questions gracefully without wasting API tokens."""
    fallback_message = (
        "I am a specialized Customer Service Data Analyst. "
        "I can only answer questions related to the Bitext dataset. How can I help you analyze our data?"
    )
    return {"messages": [AIMessage(content=fallback_message)]}