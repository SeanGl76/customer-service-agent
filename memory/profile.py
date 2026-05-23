import os
import json
import asyncio
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

_session_locks = {}

def get_session_lock(session_id: str) -> asyncio.Lock:
    """Retrieves or creates a unique async lock for a specific session."""
    if session_id not in _session_locks:
        _session_locks[session_id] = asyncio.Lock()
    return _session_locks[session_id]


# 1. Initialize the Nebius LLM with a temperature of 0 for deterministic data extraction
llm = ChatOpenAI(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_key=os.environ.get("NEBIUS_API_KEY", "YOUR_KEY_HERE"),
    base_url="https://api.studio.nebius.ai/v1/",
    temperature=0
)

# 2. Define the strict Pydantic schema for the profile
class UserProfile(BaseModel):
    name: str = Field(
        description="The user's name if explicitly stated. Default to 'Unknown' if not provided.",
        default="Unknown"
    )
    interests: List[str] = Field(
        description="A list of dataset categories or intents the user frequently asks about.",
        default_factory=list
    )
    facts: List[str] = Field(
        description="A list of explicit declarative facts or preferences learned about the user.",
        default_factory=list
    )

def get_profile_path(session_id: str) -> Path:
    """Calculates the absolute path to a session's persistent JSON profile file."""
    current_dir = Path(__file__).parent
    profiles_dir = current_dir.parent / "data" / "profiles"
    
    profiles_dir.mkdir(parents=True, exist_ok=True)
    return profiles_dir / f"{session_id}_profile.json"

def load_profile(session_id: str) -> dict:
    """Loads a user profile summary from disk, returning a default schema if empty."""
    path = get_profile_path(session_id)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("User profile failed to load")

    return {"name": "Unknown", "interests": [], "facts": []}

def save_profile(session_id: str, profile_data: dict):
    """Persists the distilled user profile information to disk across restarts."""
    path = get_profile_path(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=4, ensure_ascii=False)

def get_profile_context_str(session_id: str) -> str:
    """
    Transforms structural profile data into a clean, readable text block.
    """
    profile = load_profile(session_id)
    
    facts_str = "\n".join([f"- {fact}" for fact in profile.get("facts", [])]) or "- No explicit facts recorded yet."
    interests_str = ", ".join(profile.get("interests", [])) or "None recorded yet."
    
    return (
        f"\n=== USER PROFILE ===\n"
        f"User Name: {profile.get('name', 'Unknown')}\n"
        f"Frequent Topics/Interests: {interests_str}\n"
        f"Distilled Facts Learned From Conversations:\n{facts_str}\n"
        f"================================\n"
    )

async def update_profile_from_history(session_id: str, messages: list):
    """
    A non-blocking text extraction process that tracks new entries in chat history,
    distills recurring analytical preferences, and merges updates silently.
    """
    if not messages:
        return
    
    lock = get_session_lock(session_id)

    async with lock:
        
        profile_context = get_profile_context_str(session_id)
        
        recent_turns = "\n".join([f"{msg.type}: {msg.content}" for msg in messages[-6:]])
        
        extraction_prompt = f"""
        You are a background profile intelligence system. Your sole objective is to discover user names, 
        stated user preferences, or topics they frequently inquire about within a customer service dataset.
        
        Do not replicate past logs. Distill long-term facts.
        
        Current Profile State:
        {profile_context}
        
        Recent Interactions:
        {recent_turns}
        
        Analyze the recent interactions. If you discover new facts or interests, merge them with the 
        Current Profile State and output the updated profile. If nothing new is discovered, return 
        the Current Profile State exactly as it is.
        """
        
        try:
            extractor_llm = llm.with_structured_output(UserProfile)
            
            # Invoke the model
            updated_profile_obj = await extractor_llm.ainvoke([
                SystemMessage(content=extraction_prompt)
            ])
            
            updated_data = updated_profile_obj.model_dump()
            save_profile(session_id, updated_data)

            
        except Exception as e:
            
            print(f"\nBackground profile update suspended: {e}")