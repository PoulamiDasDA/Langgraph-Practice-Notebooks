# Codebase Flow & Architecture

This document explains the flow of the Sobeyes Agent application, detailing the purpose and location of key components.

## 1. Entry Point: `agent_application/app.py`
This is the main entry point for the Chainlit UI application.

*   **Purpose**: Handles the UI lifecycle, user sessions, and real-time interaction.
*   **Key Functions**:
    *   `on_chat_start`: Initializes a new chat session, sets up user/thread IDs, and loads previous conversation history from Cosmos DB.
    *   `on_message`: The core event handler. It:
        1.  Receives user input.
        2.  Checks for commands (e.g., `/search`).
        3.  Invokes the LangGraph agent (`graph.astream_events`).
        4.  Streams tokens and metadata (routing decisions, execution times) back to the UI.
    *   `on_store_fact` / `on_load_session`: UI actions for manual fact storage or loading past chats.

## 2. Graph Construction: `src/sobeyes_agent/graph.py`
This module builds the LangGraph state machine that drives the agent's behavior.

*   **Purpose**: Defines the nodes (agents) and edges (flow) of the system.
*   **Key Components**:
    *   `GraphBuilder`: Class that orchestrates the graph setup.
    *   `_build_graph()`: Defines the workflow:
        *   `START` -> `summarization` -> `supervisor`
        *   `supervisor` -> `Analyst` OR `Tools` OR `END` (based on routing)
    *   **Nodes**:
        *   `summarization`: Summarizes long user queries (uses `langmem` or custom fallback).
        *   `supervisor`: The "brain" that decides which agent to call next.
        *   `Analyst`: The worker agent for data analysis.
        *   `Tools`: Executes specific tools.

## 3. Agent Logic: `src/sobeyes_agent/nodes/agent_nodes.py`
Contains the actual implementation logic for each node in the graph.

*   **Purpose**: Defines what each agent actually *does*.
*   **Key Functions**:
    *   `supervisor_node`:
        *   Retrieves memory (Chat History & User Facts) from Cosmos DB.
        *   Calls the `routing_chain` to decide the next step.
        *   Triggers background tasks like saving conversation turns and extracting facts.
    *   `analyst_node`:
        *   Receives the user query and retrieved memories.
        *   Generates an analytical response using the LLM.
        *   Returns a `Command` to update the state.
    *   `_extract_and_store_facts`: Background task that uses an LLM to extract personal facts (e.g., "User is a Marketing Manager") and stores them in Cosmos DB.
    *   `_save_conversation_turn`: Saves the interaction to Cosmos DB with vector embeddings for future semantic search.

## 4. Storage Layer: `src/sobeyes_agent/storage/cosmos.py`
Handles all interactions with Azure Cosmos DB. This layer has been generalized to support flexible configurations.

*   **Purpose**: Provides a persistent memory store for the agent, adaptable to different database schemas.
*   **Key Features**:
    *   **Generalized Configuration**: Accepts database name, container names, and partition key field via environment variables.
    *   **Flexible Partitioning**: Can be configured to use any field as the partition key (default: `user_id`).
    *   **Multi-Field Embedding**: The `aput` method supports an `index` parameter that accepts a list of fields (e.g., `["text", "summary"]`). These fields are concatenated and embedded into a single vector, allowing semantic search across multiple data points.
*   **Key Components**:
    *   `CosmosDBStore`: An async implementation of LangGraph's `BaseStore`.
    *   `aput(namespace, key, value, index=...)`: Stores items.
        *   **Embedding Logic**: If `index` is a list (e.g., `["text", "summary"]`), it combines the values of those fields into a single string for embedding.
    *   `asearch`: Performs searches.
        *   If `query` is provided: Uses `VectorDistance` in SQL for semantic search.
        *   If no `query`: Performs a standard metadata filter search.
    *   `_create_document_id` / `_get_partition_key`: Manages how data is organized in the database (partitioning by the configured `COSMOS_PARTITION_KEY`).

## 5. Configuration: `src/sobeyes_agent/config.py`
*   **Purpose**: Centralizes Azure configuration and client initialization.
*   **Environment Variables**:
    *   `COSMOS_DATABASE`: Name of the database (default: `agent_memory`).
    *   `COSMOS_CONTAINER_FACTS`: Container for user facts.
    *   `COSMOS_PARTITION_KEY`: The field name to use for partitioning (default: `user_id`).
    *   `COSMOS_EMBEDDING_DIMENSIONS`: Dimensions for the vector (default: `3072`).
*   **Key Functions**:
    *   `get_facts_store()`: Returns the configured `CosmosDBStore` for user facts.
    *   `get_conversation_store()`: Returns the `CosmosDBStore` for chat history.
    *   `get_memory_saver()`: Returns the checkpointer for short-term thread state.

## Data Flow Summary
1.  **User** sends a message in Chainlit (`app.py`).
2.  **Graph** starts (`graph.py`).
3.  **Summarization** node runs to condense the query.
4.  **Supervisor** node runs (`agent_nodes.py`):
    *   Queries `CosmosDBStore` (`cosmos.py`) for relevant past conversations and facts.
    *   Decides to route to **Analyst**.
5.  **Analyst** node runs:
    *   Generates a response using the context.
6.  **Supervisor** runs again:
    *   Sees the Analyst has responded.
    *   Saves the turn to Cosmos DB (embedding generated in `cosmos.py`).
    *   Extracts new facts (if any) and saves them.
    *   Ends the turn.
7.  **App** streams the response back to the user.
