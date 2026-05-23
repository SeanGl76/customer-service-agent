from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Tracks the conversation
    messages: Annotated[list[AnyMessage], add_messages]
    
    # Tracks the user profile
    user_profile: str