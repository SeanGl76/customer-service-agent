from langgraph.graph import StateGraph, START, END 
from langgraph.prebuilt import ToolNode, tools_condition
from agent.state import AgentState
from agent.nodes import route_query, structured_agent, unstructured_agent, out_of_scope_worker

def build_graph(mcp_tools: list, memory):
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("structured_node", structured_agent)
    workflow.add_node("unstructured_node", unstructured_agent)
    workflow.add_node("out_of_scope_node", out_of_scope_worker)
    
    workflow.add_node("structured_tools", ToolNode(mcp_tools))
    workflow.add_node("unstructured_tools", ToolNode(mcp_tools))
    
    workflow.add_conditional_edges(
        START,
        route_query,
        {
            "structured_node": "structured_node",
            "unstructured_node": "unstructured_node",
            "out_of_scope_node": "out_of_scope_node"
        }
    )
    
    # ReAct Loop for the Structured Agent
    workflow.add_conditional_edges(
        "structured_node", 
        tools_condition, 
        {"tools": "structured_tools", END: END}
    )
    workflow.add_edge("structured_tools", "structured_node")
    
    # ReAct Loop for the Unstructured Agent
    workflow.add_conditional_edges(
        "unstructured_node", 
        tools_condition, 
        {"tools": "unstructured_tools", END: END}
    )
    workflow.add_edge("unstructured_tools", "unstructured_node")
    
    workflow.add_edge("out_of_scope_node", END)
     
    app = workflow.compile(checkpointer=memory)
    
    return app