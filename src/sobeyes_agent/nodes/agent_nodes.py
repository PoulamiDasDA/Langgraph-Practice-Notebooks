"""
Agent Nodes for Multi-Agent HITL System (LangGraph Pattern)
- Summarization Node: Summarizes user input before routing (uses langmem for token management)
- Tool Node: Executes additional tools
- Analyst Node: Data analysis with HITL review
- Search Node: Web search capabilities
- Supervisor Node: Orchestrates routing and long-term memory retrieval

Memory Architecture:
- Short-term: Conversation history managed by LangGraph checkpointer (automatic)
- Long-term: User facts stored in Cosmos DB with semantic search

Security Notes:
- SQL queries are encapsulated in store methods to prevent SQL injection
- Never expose raw SQL query construction to user input
- All Cosmos DB queries use parameterized operations through the CosmosDBStore interface
"""

import time
import logging
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.types import Command
from ..models import FactExtraction

logger = logging.getLogger(__name__)

# Optional: langmem for advanced summarization (install with: uv pip install langmem)
try:
    from langmem import create_manage_memory_tool, create_search_memory_tool
    LANGMEM_AVAILABLE = True
except ImportError:
    LANGMEM_AVAILABLE = False


class AgentNodes:
    """Collection of agent nodes for the multi-agent system"""
    
    def __init__(self, model, facts_store, routing_chain, conversation_store=None):
        """Initialize agent nodes with shared resources
        
        Args:
            model: LLM model for generating responses
            facts_store: CosmosDBStore for long-term user facts
            routing_chain: LLM chain for routing decisions
            conversation_store: CosmosDBStore for persistent conversation history
        
        Note: Chat history managed by checkpointer AND persisted to Cosmos DB
        """
        self.model = model
        self.facts_store = facts_store
        self.conversation_store = conversation_store
        self.routing_chain = routing_chain
    
    async def _extract_and_store_facts(self, user_query: str, user_id: str):
        """Extract personal facts from user query and store in Cosmos DB
        
        Args:
            user_query: User's message
            user_id: User identifier for namespace
        
        Returns:
            Number of facts stored
        """
        if not self.facts_store:
            return 0
        
        logger.info("\n[FACT EXTRACTION] Analyzing query for personal information...")
        
        # Use LLM to extract facts with structured output
        extraction_prompt = f"""Analyze the following user message and extract ONLY personal facts that should be remembered long-term.
Categorize each fact into one of the following categories:
- marketing_insights: Facts related to marketing strategies, campaigns, or market data
- optimization: Facts related to process or system optimization preferences
- inference: Facts derived from analysis or logical deduction
- user_preference: Explicit user preferences or settings
- general: Other personal facts

Extract facts like:
- Name (e.g., "User's name is Maria")
- Preferences (e.g., "Prefers beach vacations")
- Personal details (e.g., "Travels with family of 4")
- Role/occupation (e.g., "Works as a data analyst")
- Interests (e.g., "Interested in photography")

DO NOT extract:
- Temporary trip details (destinations, dates, prices)
- Questions or requests
- General conversation

User message: "{user_query}"
"""
        
        try:
            structured_llm = self.model.with_structured_output(FactExtraction)
            result = structured_llm.invoke([SystemMessage(content=extraction_prompt)])
            
            if not result or not result.facts:
                logger.info("[FACT EXTRACTION] No personal facts detected")
                return 0
            
            logger.info(f"[FACT EXTRACTION] Extracted {len(result.facts)} fact(s):")
            for idx, fact in enumerate(result.facts, 1):
                logger.info(f"  {idx}. [{fact.category}] {fact.text}")
            
            # Store facts in Cosmos DB
            logger.info("\n[FACT STORAGE] Storing facts in Cosmos DB...")
            namespace = ("user_facts", user_id)
            stored_count = 0
            
            for idx, fact in enumerate(result.facts):
                # Deduplication check: Search for semantically similar facts
                try:
                    existing = await self.facts_store.asearch(namespace, query=fact.text, limit=1)
                    if existing:
                        # Check similarity score (Cosmos DB VectorDistance returns cosine similarity)
                        similarity = getattr(existing[0], 'similarity_score', None)
                        if similarity is None and isinstance(existing[0].value, dict):
                            similarity = existing[0].value.get("_similarity_score", 0)
                        
                        # If > 0.85, it's likely a duplicate or very close variation
                        if similarity and similarity > 0.85:
                            logger.info(f"  Skipping duplicate fact (similarity {similarity:.2f}): {fact.text}")
                            continue
                except Exception as e:
                    logger.warning(f"  Deduplication check failed: {e}")

                key = f"fact_{int(time.time())}_{idx}"
                try:
                    # Store structured fact
                    fact_data = {
                        "text": fact.text,
                        "fact": fact.text,
                        "category": fact.category.value,
                        "source": "user_interaction",
                        "created_at": datetime.utcnow().isoformat()
                    }
                    
                    # Use multi-field embedding: Embed both text and category
                    # This allows searching for "marketing facts" or specific content
                    await self.facts_store.aput(
                        namespace, 
                        key, 
                        fact_data, 
                        index=["text", "category"]
                    )
                    stored_count += 1
                    logger.info(f"  Stored: {fact.text}")
                except Exception as e:
                    logger.error(f"  Failed to store fact: {e}")
            
            logger.info(f"\n[FACT STORAGE] Successfully stored {stored_count}/{len(result.facts)} facts in Cosmos DB")
            print(f"Stored {stored_count} new fact(s) in Cosmos DB")
            
            return stored_count
            
        except Exception as e:
            logger.error(f"[FACT EXTRACTION] Error: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    async def _save_conversation_turn(self, user_message: str, assistant_message: str, user_id: str, thread_id: str):
        """Save a conversation turn (user + assistant) to Cosmos DB for persistent history with embeddings
        
        Stores conversation with embeddings for semantic search capabilities.
        The embedding is created from the combined user message and assistant response,
        allowing semantic retrieval of relevant past conversations.
        
        Args:
            user_message: User's message
            assistant_message: Assistant's response
            user_id: User identifier
            thread_id: Thread/session identifier
        """
        if not self.conversation_store:
            return
        
        logger.info("\n[CONVERSATION STORAGE] Saving conversation turn to Cosmos DB with embeddings...")
        
        try:
            namespace = ("chat_history", user_id, thread_id)
            timestamp = int(time.time())
            
            # Create a text representation for embedding - combine user and assistant messages
            # This allows semantic search over the entire conversation context
            conversation_text = f"User: {user_message}\nAssistant: {assistant_message}"
            
            # Store the conversation turn with text for embedding
            # CosmosDBStore will automatically create embeddings from the text
            turn_data = {
                "text": conversation_text,  # This field will be embedded
                "timestamp": timestamp,
                "user_message": user_message,
                "assistant_message": assistant_message,
                "user_id": user_id,
                "thread_id": thread_id
            }
            
            key = f"turn_{timestamp}"
            await self.conversation_store.aput(namespace, key, turn_data)
            
            logger.info(f"[CONVERSATION STORAGE] Saved conversation turn with embeddings")
            logger.info(f"  - User: {user_message[:80]}..." if len(user_message) > 80 else f"  - User: {user_message}")
            logger.info(f"  - Assistant: {assistant_message[:80]}..." if len(assistant_message) > 80 else f"  - Assistant: {assistant_message}")
            logger.info(f"  - Embedding text length: {len(conversation_text)} chars")
            print(f"Saved conversation turn to Cosmos DB with embeddings")
            
        except Exception as e:
            logger.error(f"[CONVERSATION STORAGE] Failed to save conversation: {e}")
            import traceback
            traceback.print_exc()
    
    async def get_user_sessions(self, user_id: str):
        """Get all chat sessions for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            List of session information with thread_id, timestamp, and preview
        """
        if not self.conversation_store:
            logger.warning("[CONVERSATION RETRIEVAL] No conversation store configured")
            return []
        
        try:
            logger.info(f"\n[CONVERSATION RETRIEVAL] Fetching sessions for user: {user_id}")
            
            # Use asearch with query=None to get all items (filtered by namespace, no vector search)
            # This retrieves all chat history for this user without semantic ranking
            namespace_prefix = ("chat_history", user_id)
            all_items = await self.conversation_store.asearch(
                namespace_prefix,  # Positional argument
                query=None,  # No semantic search, just namespace filtering
                limit=1000  # High limit to get all sessions
            )
            
            logger.info(f"[CONVERSATION RETRIEVAL] Retrieved {len(all_items)} conversation turns")
            
            # Group by thread_id and get first message of each session
            sessions = {}
            for item in all_items:
                value = item.value
                thread_id = value.get("thread_id")
                timestamp = value.get("timestamp", 0)
                user_message = value.get("user_message", "")
                
                if thread_id and (thread_id not in sessions or timestamp < sessions[thread_id]["timestamp"]):
                    sessions[thread_id] = {
                        "thread_id": thread_id,
                        "timestamp": timestamp,
                        "preview": user_message[:100] if user_message else "No preview",
                        "datetime": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    }
            
            # Sort by timestamp (newest first)
            sorted_sessions = sorted(sessions.values(), key=lambda x: x["timestamp"], reverse=True)
            
            logger.info(f"[CONVERSATION RETRIEVAL] Found {len(sorted_sessions)} unique sessions")
            return sorted_sessions
            
        except Exception as e:
            logger.error(f"[CONVERSATION RETRIEVAL] Failed to fetch sessions: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def get_session_history(self, user_id: str, thread_id: str):
        """Get complete conversation history for a specific session
        
        Args:
            user_id: User identifier
            thread_id: Thread/session identifier
            
        Returns:
            List of conversation turns in chronological order
        """
        if not self.conversation_store:
            logger.warning("[CONVERSATION RETRIEVAL] No conversation store configured")
            return []
        
        try:
            logger.info(f"\n[CONVERSATION RETRIEVAL] Fetching history for session: {thread_id}")
            
            namespace = ("chat_history", user_id, thread_id)
            # Use asearch with query=None to get all messages (filtered by namespace, no vector search)
            items = await self.conversation_store.asearch(
                namespace,  # Positional argument
                query=None,  # No semantic search, just namespace filtering
                limit=1000  # High limit to get all turns
            )
            
            logger.info(f"[CONVERSATION RETRIEVAL] Retrieved {len(items)} items from store")
            
            # Sort by timestamp
            turns = []
            for item in items:
                value = item.value
                turns.append({
                    "timestamp": value.get("timestamp", 0),
                    "user_message": value.get("user_message", ""),
                    "assistant_message": value.get("assistant_message", ""),
                    "datetime": datetime.fromtimestamp(value.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M:%S")
                })
            
            turns.sort(key=lambda x: x["timestamp"])
            
            logger.info(f"[CONVERSATION RETRIEVAL] Retrieved {len(turns)} conversation turns")
            return turns
            
        except Exception as e:
            logger.error(f"[CONVERSATION RETRIEVAL] Failed to fetch history: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def search_conversation_history(self, user_id: str, query: str, top_k: int = 5):
        """Semantic search across user's conversation history using embeddings
        
        Searches all past conversations for semantically relevant content.
        Uses vector embeddings to find conversations similar to the query.
        
        Args:
            user_id: User identifier
            query: Search query (natural language)
            top_k: Number of most relevant conversations to return (default: 5)
            
        Returns:
            List of relevant conversation turns with similarity scores
        """
        if not self.conversation_store:
            logger.warning("[CONVERSATION SEARCH] No conversation store configured")
            return []
        
        try:
            logger.info(f"\n[CONVERSATION SEARCH] Searching conversations for: '{query[:100]}...'")
            
            # Search across all user's conversations using vector similarity
            namespace_prefix = ("chat_history", user_id)
            items = await self.conversation_store.asearch(
                namespace_prefix,  # Positional argument
                query=query,
                limit=top_k
            )
            
            logger.info(f"[CONVERSATION SEARCH] Retrieved {len(items)} items from vector search")
            
            # Format results
            relevant_conversations = []
            for idx, item in enumerate(items):
                value = item.value
                similarity = getattr(item, 'similarity_score', None) or getattr(item, 'score', None)
                if similarity is None and isinstance(value, dict):
                    similarity = value.get("_similarity_score", 'N/A')
                
                relevant_conversations.append({
                    "rank": idx + 1,
                    "similarity": similarity,
                    "user_message": value.get("user_message", ""),
                    "assistant_message": value.get("assistant_message", ""),
                    "thread_id": value.get("thread_id", ""),
                    "timestamp": value.get("timestamp", 0),
                    "datetime": datetime.fromtimestamp(value.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M:%S")
                })
            
            logger.info(f"[CONVERSATION SEARCH] Found {len(relevant_conversations)} relevant conversations")
            for conv in relevant_conversations:
                logger.info(f"  #{conv['rank']} (similarity: {conv['similarity']}) - {conv['datetime']}")
                logger.info(f"    User: {conv['user_message'][:60]}...")
            
            return relevant_conversations
            
        except Exception as e:
            logger.error(f"[CONVERSATION SEARCH] Failed to search conversations: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def custom_summarization_node(self, state, config: RunnableConfig):
        """Summarize user input before passing to agents (Legacy Custom Implementation)
        
        Creates a concise summary of the user's query to improve:
        - Memory retrieval accuracy
        - Routing decision quality
        - Agent response relevance
        """
        logger.info("\n[SUMMARIZATION NODE] Starting (Custom)...")
        start_time = time.time()
        
        # Get user query - extract HumanMessage correctly
        human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not human_messages:
            logger.info("[SUMMARIZATION NODE] No messages found, skipping")
            return {"query_intent": "", "execution_times": {"Summarization": 0}}
        user_query = human_messages[-1].content
        
        logger.info(f"[SUMMARIZATION NODE] Query length: {len(user_query)} chars")
        logger.info(f"[SUMMARIZATION NODE] Query word count: {len(user_query.split())} words")
        
        # Check if query is already concise (< 100 chars)
        if len(user_query) < 100:
            summary = user_query
            logger.info("[SUMMARIZATION NODE] Query is concise, no summarization needed")
            print("Query is concise, no summarization needed")
        else:
            # Generate summary using LLM
            logger.info("[SUMMARIZATION NODE] Query is long, generating summary...")
            
            summary_prompt = f"""Summarize the following user query into a concise search query (1-2 sentences max).
Focus on key intent, entities, and action.

User Query: {user_query}

Summary:"""
            
            response = self.model.invoke([SystemMessage(content=summary_prompt)])
            summary = response.content.strip()
            
            logger.info(f"[SUMMARIZATION NODE] Summarized query:")
            logger.info(f"  Original ({len(user_query)} chars): {user_query[:80]}...")
            logger.info(f"  Summary ({len(summary)} chars): {summary}")
            print(f"Summarized: '{user_query[:50]}...' -> '{summary}'")
        
        execution_time = time.time() - start_time
        
        return {
            "query_intent": summary,
            "execution_times": {"Summarization": execution_time}
        }
    
    def tool_node(self, state, config: RunnableConfig):
        """Execute additional tools based on agent requirements
        
        Supports:
        - Custom calculations
        - Data transformations
        - External API calls
        - File operations
        """
        start_time = time.time()
        
        # Check if tools are requested
        tool_calls = state.get("tool_calls", [])
        
        if not tool_calls:
            print("No tool calls requested")
            return {"messages": [], "execution_times": {"Tools": time.time() - start_time}}
        
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            
            print(f"Executing tool: {tool_name}")
            
            try:
                if tool_name == "calculator":
                    # Simple calculator tool
                    expression = tool_args.get("expression")
                    result = eval(expression)  # Note: Use safe evaluation in production
                    results.append(f"Calculator result: {result}")
                
                elif tool_name == "fact_lookup":
                    # Look up user facts (async)
                    user_id = config.get("configurable", {}).get("user_id", "default_user")
                    query = tool_args.get("query")
                    namespace = ("user_facts", user_id)
                    
                    # Note: This needs to be called from async context
                    # For now, skip async call in sync tool node
                    results.append(f"Fact lookup not supported in sync tool node. Use supervisor memory retrieval instead.")
                
                elif tool_name == "chat_history_lookup":
                    # Chat history is available in state["messages"] - use that instead
                    results.append("Chat history is available in conversation state. Use state['messages'] to access it.")
                
                else:
                    results.append(f"Unknown tool: {tool_name}")
            
            except Exception as e:
                results.append(f"Tool error ({tool_name}): {str(e)}")
        
        execution_time = time.time() - start_time
        
        return {
            "messages": [AIMessage(content="\n\n".join(results), name="Tools")],
            "execution_times": {"Tools": execution_time}
        }
    
    def analyst_node(self, state, config: RunnableConfig):
        """Analyst agent with HITL review - handles analysis, calculations, reasoning"""
        logger.info("\n[ANALYST NODE] Starting analysis...")
        start_time = time.time()
        
        # Get user_id, user_role, and user query
        user_id = config.get("configurable", {}).get("user_id", "default_user")
        user_role = config.get("configurable", {}).get("user_role")
        
        logger.info(f"[ANALYST NODE] User ID: {user_id}")
        logger.info(f"[ANALYST NODE] User Role: {user_role if user_role else 'Not set'}")
        
        # Extract user query correctly - filter HumanMessages first
        human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not human_messages:
            return Command(update={"messages": [AIMessage(content="No user query found.", name="Analyst")]})
        user_query = human_messages[-1].content
        
        # Get retrieved long-term memories (facts) from state
        memories = state.get("retrieved_memories", [])
        
        # Build context from long-term facts
        if memories:
            fact_context = "\n".join([f"- {m.get('text', '')}" for m in memories])
            memory_context = f"User Facts (from previous sessions):\n{fact_context}"
        else:
            memory_context = "No relevant user facts from previous sessions."
        
        # Get recent conversation context from state (short-term memory)
        # Checkpointer automatically maintains this - use last 6 messages (3 turns)
        recent_messages = state["messages"][-6:] if len(state["messages"]) > 1 else []
        conversation_context = ""
        if len(recent_messages) > 1:  # More than just current message
            conv_parts = []
            for msg in recent_messages[:-1]:  # Exclude current query
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                conv_parts.append(f"{role}: {content}")
            conversation_context = f"\n\nRecent Conversation:\n" + "\n".join(conv_parts)
        
        # Build role context for personalized responses
        role_context = ""
        if user_role:
            role_context = f"\n\nUser Role: {user_role}\nTailor your response to a {user_role}'s perspective and needs."
        
        # Build analyst prompt with both long-term facts and recent conversation
        analyst_prompt = f"""You are a data analyst assistant.

{memory_context}{conversation_context}{role_context}

Current Query: {user_query}

Provide a detailed, analytical response. Include calculations, insights, and reasoning. Use the context from previous facts and recent conversation to give personalized responses."""
        
        # Generate response
        response = self.model.invoke([SystemMessage(content=analyst_prompt)])
        analyst_response = response.content
        
        execution_time = time.time() - start_time
        
        # Store for HITL review in Chainlit
        return Command(
            update={
                "messages": [AIMessage(content=analyst_response, name="Analyst")],
                "execution_times": {"Analyst": execution_time},
                "pending_review": {
                    "agent": "Analyst",
                    "response": analyst_response,
                    "memories_used": len(memories)
                }
            }
        )
    

    
    async def supervisor_node(self, state, config: RunnableConfig):
        """Supervisor orchestrates routing and memory retrieval (async)"""
        
        if state["messages"]:
            last_msg = state["messages"][-1]
            sender = getattr(last_msg, "name", None)
            
            # If agent responded, save conversation and extract facts, then finish
            if sender in ["Analyst", "Tools"]:
                human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
                ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
                
                if human_messages and ai_messages:
                    last_user_query = human_messages[-1].content
                    last_assistant_response = ai_messages[-1].content
                    user_id = config["configurable"]["user_id"]
                    thread_id = config["configurable"]["thread_id"]
                    
                    # Save conversation turn to Cosmos DB (persistent chat history)
                    if self.conversation_store:
                        await self._save_conversation_turn(
                            last_user_query, 
                            last_assistant_response, 
                            user_id, 
                            thread_id
                        )
                    
                    # Extract and store user facts asynchronously
                    if self.facts_store:
                        await self._extract_and_store_facts(last_user_query, user_id)
                
                return {"next": "FINISH"}
        
        # Get user query - extract HumanMessage correctly
        human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not human_messages:
            return {"next": "FINISH"}
        user_query = human_messages[-1].content
        user_id = config["configurable"]["user_id"]
        
        # Use summarized query if available for better retrieval
        search_query = state.get("query_intent", user_query)
        
        # Retrieve relevant long-term memories from Cosmos DB (async)
        # Try chat history first (since that's where data is stored), then facts
        # Short-term conversation history is in state["messages"] (managed by checkpointer)
        logger.info("\n[SUPERVISOR NODE] Memory Retrieval:")
        logger.info(f"  - Conversation Store: {'Available' if self.conversation_store else 'Not configured (in-memory only)'}")
        logger.info(f"  - Facts Store: {'Available' if self.facts_store else 'Not configured (in-memory only)'}")
        
        memories = []
        
        # Search chat history first (semantic search across all user's conversations)
        if self.conversation_store:
            namespace_prefix = ("chat_history", user_id)
            logger.info(f"  - Searching Chat History...")
            logger.info(f"  - Namespace Prefix: {namespace_prefix}")
            logger.info(f"  - Search Query: {search_query[:100]}..." if len(search_query) > 100 else f"  - Search Query: {search_query}")
            
            try:
                # Semantic search across all conversations for this user
                history_items = await self.conversation_store.asearch(namespace_prefix, query=search_query, limit=5)
                if history_items:
                    logger.info(f"  - Retrieved {len(history_items)} relevant conversations from chat history")
                    for idx, item in enumerate(history_items, 1):
                        conv_data = item.value
                        text = f"Previous conversation: User: {conv_data.get('user_message', '')} | Assistant: {conv_data.get('assistant_message', '')}"
                        memories.append({"text": text})
                        logger.info(f"    {idx}. {text[:100]}..." if len(text) > 100 else f"    {idx}. {text}")
                    print(f"Retrieved {len(history_items)} relevant conversations from chat history")
                else:
                    logger.info(f"  - No relevant conversations found in chat history")
            except Exception as e:
                logger.error(f"  - Chat history retrieval error: {e}")
                print(f"Chat history retrieval error: {e}")
                import traceback
                traceback.print_exc()
        
        # Also search facts if available
        if self.facts_store and not memories:  # Only if no chat history found
            namespace = ("user_facts", user_id)
            logger.info(f"  - Searching User Facts...")
            logger.info(f"  - Namespace: {namespace}")
            
            try:
                facts_items = await self.facts_store.asearch(namespace, query=search_query, limit=5)
                if facts_items:
                    memories = [{"text": m.value.get("text", m.value.get("fact", str(m.value)))} for m in facts_items]
                    logger.info(f"  - Retrieved {len(memories)} facts")
                    print(f"Retrieved {len(memories)} facts from user facts store")
                else:
                    logger.info(f"  - No facts found")
            except Exception as e:
                logger.error(f"  - Fact retrieval error: {e}")
        
        if not self.conversation_store and not self.facts_store:
            logger.info(f"  - Skipping long-term memory (running in-memory only mode)")
        
        # Route using LLM
        logger.info("\n[SUPERVISOR NODE] Routing Decision:")
        logger.info(f"  - Invoking routing LLM...")
        
        decision = self.routing_chain.invoke({"query": user_query})
        
        logger.info(f"  - Agent Selected: {decision.agent}")
        logger.info(f"  - Reasoning: {decision.reasoning}")
        
        print(f"Routing to {decision.agent}: {decision.reasoning}")
        
        logger.info("[SUPERVISOR NODE] Completed\n")
        
        return Command(
            update={
                "next": decision.agent,
                "routing_reasoning": [decision.reasoning],
                "retrieved_memories": memories
            }
        )
