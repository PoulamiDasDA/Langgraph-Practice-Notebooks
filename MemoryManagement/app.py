"""
Chainlit UI Application - HITL Multi-Agent Chat Interface
Handles all Chainlit-specific UI interactions and event handlers
"""

import uuid
import logging
import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime

from config import AzureConfig
from graph_builder import GraphBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ===== Initialize Services =====
azure_config = AzureConfig()
graph_builder = GraphBuilder(azure_config)
graph = graph_builder.get_graph()
facts_store = azure_config.get_facts_store()
conversation_store = azure_config.get_conversation_store()
agent_nodes = graph_builder.agent_nodes
tavily_available = azure_config.get_tavily_tool() is not None


# ===== Chainlit Event Handlers =====

@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session with previous conversations sidebar"""
    logger.info("="*60)
    logger.info("NEW CHAT SESSION STARTED")
    logger.info("="*60)
    
    # Use consistent user_id for memory persistence across sessions
    user_id = cl.user_session.get("user").identifier if cl.user_session.get("user") else "default_user"
    thread_id = f"thread_{uuid.uuid4().hex}"
    
    logger.info(f"Session Configuration:")
    logger.info(f"  - User ID: {user_id}")
    logger.info(f"  - Thread ID: {thread_id}")
    logger.info(f"  - Tavily Available: {tavily_available}")
    logger.info(f"  - Facts Store: {'Connected' if facts_store else 'In-memory only'}")
    logger.info(f"  - Conversation Store: {'Connected' if conversation_store else 'In-memory only'}")
    
    print(f"🔑 Session started with user_id: {user_id}, thread_id: {thread_id}")
    
    # Store in session
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("user_role", None)  # Will be set when user shares their role
    
    # Load previous sessions for this user and display in sidebar
    if conversation_store:
        try:
            sessions = await agent_nodes.get_user_sessions(user_id)
            if sessions:
                logger.info(f"[CHAT HISTORY] Found {len(sessions)} previous sessions for user {user_id}")
                
                # Display previous sessions as actions in the chat
                sessions_text = "## 📜 Your Previous Conversations\n\n"
                actions = []
                
                for idx, session in enumerate(sessions[:10]):  # Show last 10 sessions
                    sessions_text += f"{idx+1}. **{session['datetime']}**\n   _{session['preview']}_\n\n"
                    actions.append(
                        cl.Action(
                            name="load_session",
                            value=session['thread_id'],
                            label=f"Load Session {idx+1}"
                        )
                    )
                
                # Send as a message with actions
                await cl.Message(
                    content=sessions_text,
                    actions=actions
                ).send()
        except Exception as e:
            logger.error(f"[CHAT HISTORY] Failed to load previous sessions: {e}")
    
    await cl.Message(
        content=f"Hello! I'm your assistant with memory.\n\n"
                f"How can I help you today?"
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
            await cl.Message(content=f"🔍 Searching your conversation history for: *{query}*...").send()
            
            try:
                results = await agent_nodes.search_conversation_history(user_id, query, top_k=5)
                
                if results:
                    search_results = "## 🔍 Relevant Past Conversations\n\n"
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
                await cl.Message(content=f"❌ Search failed: {str(e)}").send()
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
                await cl.Message(content=f"✓ I'll remember you're a {user_role}!").send()
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
            metadata_msgs.append(f"📚 Retrieved {memories_used} relevant facts from long-term memory")
        
        if exec_times:
            agent_name = list(exec_times.keys())[-1]
            duration = exec_times[agent_name]
            metadata_msgs.append(f"⏱️ {agent_name} execution: {duration:.2f}s")
        
        # Send metadata as a single message
        if metadata_msgs:
            await cl.Message(content="\n".join(metadata_msgs)).send()
        
        # If streaming didn't capture the response, send it now
        if pending_review:
            assistant_response = pending_review["response"]
            agent_name = pending_review.get("agent", "Agent")
            
            if not msg.content:  # If no streaming occurred
                await msg.update(content=f"**{agent_name} Response:**\n\n{assistant_response}\n\n"
                                        "💬 *Feel free to provide feedback, ask follow-up questions, or request changes!*")
            else:
                # Append annotation
                await cl.Message(content="💬 *Feel free to provide feedback, ask follow-up questions, or request changes!*").send()
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
        await cl.Message(content=f"❌ Error: {str(e)}").send()
        print(f"Error details: {e}")


@cl.action_callback("store_fact")
async def on_store_fact(action: cl.Action):
    """Action to store a fact about the user"""
    user_id = cl.user_session.get("user_id")
    namespace = ("user_facts", user_id)
    
    # Simple fact extraction (in production, use NER or LLM extraction)
    fact_text = action.value
    fact_key = f"fact_{datetime.utcnow().timestamp()}"
    
    await facts_store.aput(namespace, fact_key, {"fact": fact_text})
    
    await cl.Message(content=f"✓ Stored fact: {fact_text}").send()


@cl.action_callback("load_session")
async def on_load_session(action: cl.Action):
    """Action to load a previous chat session"""
    user_id = cl.user_session.get("user_id")
    thread_id = action.value
    
    if not conversation_store:
        await cl.Message(content="❌ Conversation store not available").send()
        return
    
    try:
        logger.info(f"[CHAT HISTORY] Loading session: {thread_id}")
        
        # Get the conversation history for this session
        history = await agent_nodes.get_session_history(user_id, thread_id)
        
        if not history:
            await cl.Message(content="No conversation history found for this session.").send()
            return
        
        # Display the conversation history
        history_text = f"## 📜 Previous Conversation (Session: {thread_id[:8]}...)\n\n"
        for turn in history:
            history_text += f"**You ({turn['datetime']}):** {turn['user_message']}\n\n"
            history_text += f"**Assistant:** {turn['assistant_message']}\n\n"
            history_text += "---\n\n"
        
        await cl.Message(content=history_text).send()
        
        # Option to continue this session
        await cl.Message(
            content="💡 You're viewing a previous conversation. You can:\n"
                   "- Continue chatting in your current session\n"
                   "- Or type `/resume` to continue this previous conversation"
        ).send()
        
        # Store the loaded thread_id for potential resume
        cl.user_session.set("loaded_thread_id", thread_id)
        
    except Exception as e:
        logger.error(f"[CHAT HISTORY] Failed to load session: {e}")
        await cl.Message(content=f"❌ Failed to load conversation: {str(e)}").send()


@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings panel interactions"""
    user_id = cl.user_session.get("user_id")
    
    # Check if a session button was clicked
    for key, value in settings.items():
        if key.startswith("session_") and value:
            thread_id = key.replace("session_", "")
            await on_load_session(cl.Action(name="load_session", value=thread_id))
            break


if __name__ == "__main__":
    # Run with: chainlit run app.py -w
    pass
