"""
Pydantic Models for Type Safety and Validation
"""

from typing import Optional, Literal, Annotated
from operator import add
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


# ===== Request/Response Models =====

class ChatRequest(BaseModel):
    """Structured chat request model"""
    user_message: str = Field(..., description="User's message content")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation continuity")
    user_id: str = Field(..., description="User identifier for memory persistence")
    user_role: Optional[str] = Field(None, description="User's role/function for personalized responses")


class ChatMetadata(BaseModel):
    """Metadata about agent execution"""
    routing_decision: Optional[str] = Field(None, description="Which agent was selected and why")
    memories_retrieved: int = Field(0, description="Number of long-term facts retrieved")
    execution_time: Optional[float] = Field(None, description="Total execution time in seconds")
    agent_name: Optional[str] = Field(None, description="Name of agent that handled request")


class ChatResponse(BaseModel):
    """Structured chat response model"""
    response: str = Field(..., description="Agent's response content")
    metadata: ChatMetadata = Field(..., description="Execution metadata")
    session_info: dict = Field(default_factory=dict, description="Session tracking info")


class RouteDecision(BaseModel):
    """Routing decision model for supervisor agent"""
    agent: Literal["Analyst", "Search"]
    reasoning: str


# ===== Agent State Model =====

def merge_dicts(existing: dict, new: dict) -> dict:
    """Merge dictionaries for execution time tracking"""
    return {**existing, **new}


class State(MessagesState):
    """Agent state with HITL tracking, routing, and memory"""
    next: Optional[Literal["Analyst", "Search", "FINISH"]] = None
    user_edits: Annotated[list[dict], add] = []
    execution_times: Annotated[dict[str, float], merge_dicts] = {}
    routing_reasoning: Annotated[list[str], add] = []
    query_intent: Optional[str] = None
    retrieved_memories: list[dict] = []
    pending_review: Optional[dict] = None  # For Chainlit HITL
