import argparse
import asyncio
import uuid
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from agent.graph import build_graph
from agent.nodes import get_mcp_tools
from memory.profile import get_profile_context_str, update_profile_from_history
from langgraph.errors import GraphRecursionError

load_dotenv()

async def run_chat_loop():
    # 1. Parse Command Line Arguments for Session ID
    parser = argparse.ArgumentParser(description="Bitext Customer Service Data Analyst Agent")
    parser.add_argument(
        "--session", 
        type=str, 
        default=str(uuid.uuid4())[:8], 
        help="Session ID to restore previous conversations and profiles."
    )
    args = parser.parse_args()
    session_id = args.session

    print(f"--- Starting Agent Session: {session_id} ---")
    print("Connecting to FastMCP Server... (Ensure server/mcp.py is running)")
    
    # 2. Initialize the MCP Tools and the Graph
    try:
        mcp_tools = await get_mcp_tools()
    except Exception as e:
        print(f"Failed to connect to MCP server. {e}")
        sys.exit(1)
        
    print("Agent is ready! Type 'exit' or 'quit' to end the conversation.\n")

    # 3. The Interactive Conversation Loop
    async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as memory:
        
        # Pass the memory into the graph builder
        app = build_graph(mcp_tools, memory)
        
        # Indent your entire while loop inside the async context manager
        while True:
            try:
                user_input = await asyncio.to_thread(input, "\nUser: ")
                if user_input.lower() in ['exit', 'quit']:
                    print("Ending session. Goodbye!")
                    break
                if not user_input.strip():
                    continue

                profile_context = get_profile_context_str(session_id)
                enriched_prompt = f"{profile_context}\n\nUser Query: {user_input}"
                config = {
                    "configurable": {"thread_id": session_id},
                    "recursion_limit": 15 
                }
                
                print("\n--- Agent Reasoning Process ---")
                
                async for event in app.astream({"messages": [HumanMessage(content=enriched_prompt)]}, config=config):
                    
                    for node_name, node_state in event.items():
                        if node_name in ["structured_node", "unstructured_node", "out_of_scope_node"]:
                            latest_msg = node_state["messages"][-1]
                            if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                                for tc in latest_msg.tool_calls:
                                    print(f"[\u2699\ufe0f Tool Called] {tc['name']} with args: {tc['args']}")
                                
                    for node_name, node_state in event.items():
                        if node_name in ["structured_tools", "unstructured_tools"]:
                            latest_msg = node_state["messages"][-1]
                            print(f"[\U0001f4dc Observation] {latest_msg.content}")
                
                print("-------------------------------\n")
                
                final_state = await app.aget_state(config)
                final_message = final_state.values["messages"][-1].content
                print(f"Agent: {final_message}")
                
                asyncio.create_task(update_profile_from_history(session_id, final_state.values["messages"]))

            except KeyboardInterrupt:
                print("\nSession interrupted by user. Exiting...")
                break
            except GraphRecursionError:
                print("\nAgent: I've reached my exploration limit. Based on what I've seen so far, the database contains various intents and categories. Can you be more specific about what you are looking for?")
            except Exception as e:
                print(f"\nAn error occurred during execution: {e}")

if __name__ == "__main__":
    asyncio.run(run_chat_loop())