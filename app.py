"""
Chainlit UI Application - HITL Multi-Agent Chat Interface
Handles all Chainlit-specific UI interactions and event handlers
"""

import uuid
import logging
import chainlit as cl
from chainlit.types import ThreadDict
from chainlit.user import User
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime

from src.sobeyes_agent.config import AzureConfig
from src.sobeyes_agent.graph import GraphBuilder
from src.sobeyes_agent.storage.chainlit_cosmos import CosmosDataLayer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    datefmt='%H:%M:%S'
#     handlers=[
#         logging.FileHandler("log.txt", mode='a'),
#         logging.StreamHandler()
#     ]
)
logger = logging.getLogger(__name__)


# ===== Initialize Services =====
azure_config = AzureConfig()
graph_builder = GraphBuilder(azure_config)
graph = graph_builder.get_graph()
facts_store = azure_config.get_facts_store()
conversation_store = azure_config.get_conversation_store()
agent_nodes = graph_builder.agent_nodes

# Initialize Chainlit Data Layer for History Sidebar
cl.data_layer = CosmosDataLayer(azure_config)


# ===== Chainlit Event Handlers =====

@cl.password_auth_callback
def auth_callback(username, password):
    """
    Simple authentication callback to enable the History Sidebar.
    """
    # Static credentials for demonstration
    if username == "admin" and password == "admin":
        return User(identifier=username)
    return None

@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session"""
    logger.info("="*60)
    logger.info("NEW CHAT SESSION STARTED")
    logger.info("="*60)
    
    # Use consistent user_id for memory persistence across sessions
    user_obj = cl.user_session.get("user")
    logger.info(f"DEBUG: Authenticated User: {user_obj}")
    user_id = user_obj.identifier if user_obj else "default_user"
    
    # Attempt to find the most recent thread for "Resume" option
    latest_thread_id = None
    
    try:
        # Create filter and pagination for listing threads
        from chainlit.types import Pagination, ThreadFilter
        pagination = Pagination(first=1)
        # Note: ThreadFilter fields may vary by version, using kwargs to be safe or standard field
        # Try passing both to cover different Chainlit versions
        try:
            thread_filter = ThreadFilter(userIdentifier=user_id, userId=user_id)
        except:
            thread_filter = ThreadFilter(userIdentifier=user_id)
        
        # List threads to find the most recent one
        response = await cl.data_layer.list_threads(pagination, thread_filter)
        if response.data:
            latest_thread = response.data[0]
            latest_thread_id = latest_thread["id"]
            logger.info(f"Found existing thread for resume option: {latest_thread_id}")
    except Exception as e:
        logger.error(f"Failed to fetch existing threads: {e}")
    
    # Always create a new thread for the current session
    thread_id = f"thread_{uuid.uuid4().hex}"
    logger.info(f"Created new thread: {thread_id}")
    
    logger.info(f"Session Configuration:")
    logger.info(f"  - User ID: {user_id}")
    logger.info(f"  - Thread ID: {thread_id}")
    logger.info(f"  - Facts Store: {'Connected' if facts_store else 'In-memory only'}")
    logger.info(f"  - Conversation Store: {'Connected' if conversation_store else 'In-memory only'}")
    
    print(f"Session started with user_id: {user_id}, thread_id: {thread_id}")
    
    # Store in session
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("user_role", None)  # Will be set when user shares their role
    
    # Prepare welcome message actions
    actions = []
    if latest_thread_id:
        actions.append(
            cl.Action(
                name="resume_thread", 
                value=latest_thread_id, 
                label="Resume Last Conversation",
                description="Continue from your most recent chat",
                payload={"value": latest_thread_id}
            )
        )

    await cl.Message(
        content=f"Hello! I'm your assistant with memory.\n\n"
                f"How can I help you today?",
        actions=actions
    ).send()


@cl.action_callback("resume_thread")
async def on_resume_thread(action: cl.Action):
    """Action to resume a specific thread"""
    # Safely retrieve value from payload or attribute
    thread_id = action.payload.get("value") if action.payload else None
    if not thread_id:
        thread_id = getattr(action, "value", None)
        
    user_id = cl.user_session.get("user_id")
    
    logger.info(f"User {user_id} requested to resume thread {thread_id}")
    
    # Update session
    cl.user_session.set("thread_id", thread_id)
    
    # Clear UI
    await cl.Message(content="Resuming conversation...").send()
    
    # Fetch and display history
    try:
        steps = await cl.data_layer.get_steps(thread_id)
        if steps:
            for step in steps:
                if step["type"] == "user_message":
                    await cl.Message(
                        content=step["content"], 
                        author="You", 
                        type="user_message"
                    ).send()
                elif step["type"] == "assistant_message":
                    await cl.Message(
                        content=step["content"], 
                        author="Assistant", 
                        type="assistant_message"
                    ).send()
            
            await cl.Message(content="--- Conversation Resumed ---").send()
    except Exception as e:
        logger.error(f"Failed to restore steps: {e}")
        await cl.Message(content=f"Error resuming conversation: {e}").send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Resume a chat session from history"""
    logger.info("="*60)
    logger.info(f"RESUMING CHAT SESSION: {thread['id']}")
    logger.info("="*60)
    
    user_id = thread.get("userIdentifier") or "default_user"
    thread_id = thread["id"]
    
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("user_role", None) # Reset or load from metadata if available
    
    await cl.Message(
        content=f"Welcome back! I've restored your conversation context."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages with conversational HITL support and semantic search"""
    user_id = cl.user_session.get("user_id")
    thread_id = cl.user_session.get("thread_id")
    user_role = cl.user_session.get("user_role")
    
    # Check for semantic search command
    if message.content.startswith("/search "):
        query = message.content[8:].strip()
        if conversation_store and query:
            await cl.Message(content=f"Searching your conversation history for: *{query}*...").send()
            
            try:
                results = await agent_nodes.search_conversation_history(user_id, query, top_k=5)
                
                if results:
                    search_results = "## Relevant Past Conversations\n\n"
                    for conv in results:
                        search_results += f"### {conv['datetime']} (Session: {conv['thread_id'][:8]}...)\n"
                        search_results += f"**You:** {conv['user_message']}\n\n"
                        search_results += f"**Assistant:** {conv['assistant_message'][:200]}...\n\n"
                        search_results += f"*Relevance: {conv['similarity']}*\n\n---\n\n"
                    
                    await cl.Message(content=search_results).send()
                else:
                    await cl.Message(content="No relevant conversations found.").send()
            except Exception as e:
                logger.error(f"Semantic search error: {e}")
                await cl.Message(content=f"Search failed: {str(e)}").send()
        else:
            await cl.Message(content="Usage: `/search <your query>` - Search your conversation history semantically").send()
        return
    
    # Simple role detection (can be enhanced with NER or explicit command)
    if not user_role and any(keyword in message.content.lower() for keyword in ["i am", "my role is", "i work as"]):
        # Extract potential role (simple heuristic)
        for role_keyword in ["analyst", "manager", "developer", "marketing", "loyalty", "data scientist"]:
            if role_keyword in message.content.lower():
                user_role = role_keyword.title()
                cl.user_session.set("user_role", user_role)
                await cl.Message(content=f"I'll remember you're a {user_role}!").send()
                break
    
    # Prepare config for agent with user role context
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "user_role": user_role
        }
    }
    
    try:
        # Stream multi-agent graph responses
        # Note: Conversation history automatically managed by checkpointer
        
        # Initialize message for streaming
        msg = cl.Message(content="")
        await msg.send()
        
        # Track state across stream chunks
        result = None
        routing = []
        exec_times = {}
        memories_used = 0
        pending_review = None
        
        # Try token-level streaming with astream_events (improved UX)
        # Fallback to update-level streaming if not supported
        streaming_supported = True
        logger.info("\n[STREAMING] Starting token-level streaming (astream_events v2)...")
        
        try:
            # Token-level streaming for real-time response display
            event_stream = graph.astream_events(
                {"messages": [HumanMessage(content=message.content)]},
                config=config,
                version="v2"
            )
            
            event_count = 0
            token_count = 0
            
            async for event in event_stream:
                event_count += 1
                kind = event.get("event")
                
                # Stream tokens from LLM
                if kind == "on_chat_model_stream":
                    content = event.get("data", {}).get("chunk", {}).content
                    if content:
                        token_count += 1
                        await msg.stream_token(content)
                
                # Capture metadata from node completions
                elif kind == "on_chain_end":
                    node_name = event.get("name", "unknown")
                    logger.info(f"[STREAMING] Chain end event from: {node_name}")
                    
                    outputs = event.get("data", {}).get("output", {})
                    if outputs and isinstance(outputs, dict):
                        if "routing_reasoning" in outputs:
                            routing = outputs["routing_reasoning"]
                            logger.info(f"[STREAMING] Routing captured: {routing}")
                        if "execution_times" in outputs:
                            logger.info(f"[STREAMING] Execution times captured")
                            exec_times.update(outputs.get("execution_times", {}))
                        if "retrieved_memories" in outputs:
                            memories_used = len(outputs["retrieved_memories"])
                        if "pending_review" in outputs:
                            pending_review = outputs["pending_review"]
        
            logger.info(f"[STREAMING] Token-level streaming completed:")
            logger.info(f"  - Total events: {event_count}")
            logger.info(f"  - Tokens streamed: {token_count}")
        
        except (Exception, GeneratorExit) as stream_error:
            # Fallback to update-level streaming
            if not isinstance(stream_error, GeneratorExit):
                logger.warning(f"[STREAMING] Token streaming failed, falling back to update streaming: {stream_error}")
                print(f"Token streaming not supported, using update streaming: {stream_error}")
            else:
                logger.info("[STREAMING] Token streaming ended with GeneratorExit")
            streaming_supported = False
            
            logger.info("[STREAMING] Starting update-level streaming (fallback)...")
            async for chunk in graph.astream(
                {"messages": [HumanMessage(content=message.content)]},
                config=config,
                stream_mode="updates"
            ):
                if chunk:
                    logger.info(f"[STREAMING] Update chunk received from: {list(chunk.keys())}")
                    for node_name, node_output in chunk.items():
                        if node_output:
                            # Update metadata
                            if "routing_reasoning" in node_output:
                                routing = node_output["routing_reasoning"]
                            if "execution_times" in node_output:
                                exec_times.update(node_output.get("execution_times", {}))
                            if "retrieved_memories" in node_output:
                                memories_used = len(node_output["retrieved_memories"])
                            if "pending_review" in node_output:
                                pending_review = node_output["pending_review"]
                            
                            # Stream message updates if available
                            if "messages" in node_output and node_output["messages"]:
                                # Extract last AI message correctly
                                ai_messages = [m for m in node_output["messages"] if isinstance(m, AIMessage)]
                                if ai_messages and hasattr(ai_messages[-1], "content"):
                                    await msg.stream_token(ai_messages[-1].content)
        
        # Get final result
        final_state = await graph.aget_state(config)
        result = final_state.values if final_state else {}
        
        # Display metadata after streaming completes
        metadata_msgs = []
        
        if routing:
            metadata_msgs.append(f"**Routing**: {routing[-1]}")
        
        if memories_used > 0:
            metadata_msgs.append(f"Retrieved {memories_used} relevant facts from long-term memory")
        
        if exec_times:
            agent_name = list(exec_times.keys())[-1]
            duration = exec_times[agent_name]
            metadata_msgs.append(f"{agent_name} execution: {duration:.2f}s")
        
        # Send metadata as a single message
        if metadata_msgs:
            await cl.Message(content="\n".join(metadata_msgs)).send()
        
        # If streaming didn't capture the response, send it now
        if pending_review:
            assistant_response = pending_review["response"]
            agent_name = pending_review.get("agent", "Agent")
            
            if not msg.content:  # If no streaming occurred
                await msg.update(content=f"**{agent_name} Response:**\n\n{assistant_response}\n\n"
                                        "*Feel free to provide feedback, ask follow-up questions, or request changes!*")
            else:
                # Append annotation
                await cl.Message(content="*Feel free to provide feedback, ask follow-up questions, or request changes!*").send()
        elif result and "messages" in result and not msg.content:
            # Fallback: if streaming didn't work, send final message
            # Extract AI messages correctly - filter by type, then get content
            ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
            if ai_messages:
                last_ai_content = ai_messages[-1].content
                await msg.update(content=last_ai_content)
        
        # Note: All conversation automatically stored by checkpointer
        # Human can naturally continue the conversation with feedback/corrections
        
    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()
        print(f"Error details: {e}")


@cl.action_callback("store_fact")
async def on_store_fact(action: cl.Action):
    """Action to store a fact about the user"""
    user_id = cl.user_session.get("user_id")
    namespace = ("user_facts", user_id)
    
    # Simple fact extraction (in production, use NER or LLM extraction)
    fact_text = action.payload.get("value") if action.payload else None
    if not fact_text:
        # Fallback if value is available directly (older versions)
        fact_text = getattr(action, "value", None)
        
    if not fact_text:
        await cl.Message(content="Error: Could not retrieve fact text").send()
        return

    fact_key = f"fact_{datetime.utcnow().timestamp()}"
    
    await facts_store.aput(namespace, fact_key, {"fact": fact_text})
    
    await cl.Message(content=f"Stored fact: {fact_text}").send()





if __name__ == "__main__":
    # Run with: chainlit run app.py -w
    pass
