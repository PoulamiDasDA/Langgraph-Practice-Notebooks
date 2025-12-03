import logging
import time
from typing import List, Optional, Dict, Any, Union
import chainlit as cl
import chainlit.data as cl_data
from chainlit.types import (
    ThreadDict, Pagination, PaginatedResponse, ThreadFilter
)
from chainlit.user import User, PersistedUser
import chainlit.context as cl_context
from src.sobeyes_agent.config import AzureConfig
from datetime import datetime

logger = logging.getLogger(__name__)

class CosmosDataLayer(cl_data.BaseDataLayer):
    """
    Chainlit Data Layer implementation using Azure Cosmos DB.
    Maps Chainlit's Thread/Step model to our Cosmos DB 'turns' model.
    """
    
    def __init__(self, azure_config: AzureConfig):
        self.azure_config = azure_config
        self.store = azure_config.get_conversation_store()
        logger.info("CosmosDataLayer initialized")

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        """Get user by identifier"""
        # We don't strictly manage users in Cosmos yet, so we return a mock user
        # or we could store users in a "users" container if needed.
        # For now, just return a PersistedUser with the identifier.
        return PersistedUser(
            id=identifier,
            identifier=identifier,
            createdAt=datetime.utcnow().isoformat()
        )

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        """Create a new user"""
        # We accept any user
        return PersistedUser(
            id=user.identifier,
            identifier=user.identifier,
            createdAt=datetime.utcnow().isoformat(),
            metadata=user.metadata
        )

    async def list_threads(
        self, pagination: Pagination, filter: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        """List threads for a user"""
        if not self.store:
            return PaginatedResponse(data=[], hasMore=False)

        logger.info(f"DEBUG: list_threads filter: {filter}")
        user_id = getattr(filter, "userIdentifier", None) or getattr(filter, "userId", None) or "default_user"
        logger.info(f"Listing threads for user: {user_id}")

        try:
            # Reuse logic from AgentNodes.get_user_sessions
            # We fetch all items for the user to group them by thread
            # This is inefficient for large histories but fits our current data model
            namespace_prefix = ("chat_history", user_id)
            all_items = await self.store.asearch(
                namespace_prefix,
                query=None,
                limit=1000 # Fetch enough to group
            )
            logger.info(f"Found {len(all_items)} items for user {user_id} in list_threads")

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
                        "preview": user_message[:100] if user_message else "New Conversation",
                        "datetime": datetime.fromtimestamp(timestamp).isoformat()
                    }
            
            # Sort by timestamp (newest first)
            sorted_sessions = sorted(sessions.values(), key=lambda x: x["timestamp"], reverse=True)
            
            # Apply pagination
            start = 0
            if pagination.cursor:
                # Simple cursor implementation (index based for now)
                try:
                    start = int(pagination.cursor)
                except:
                    start = 0
            
            limit = pagination.first or 20
            end = start + limit
            page_items = sorted_sessions[start:end]
            has_more = end < len(sorted_sessions)
            next_cursor = str(end) if has_more else None

            threads: List[ThreadDict] = []
            for session in page_items:
                threads.append(ThreadDict(
                    id=session["thread_id"],
                    createdAt=session["datetime"],
                    name=session["preview"],
                    userId=user_id,
                    userIdentifier=user_id,
                    tags=[],
                    metadata={}
                ))

            return PaginatedResponse(data=threads, hasMore=has_more, cursor=next_cursor, pageInfo={"hasNextPage": has_more, "endCursor": next_cursor, "startCursor": str(start)})

        except Exception as e:
            logger.error(f"Failed to list threads: {e}")
            return PaginatedResponse(data=[], hasMore=False, pageInfo={"hasNextPage": False, "endCursor": None, "startCursor": None})

    def _get_current_user_id(self) -> str:
        try:
            user = cl.user_session.get("user")
            if user and hasattr(user, "identifier"):
                return user.identifier
        except Exception as e:
            logger.error(f"Error getting user from session: {e}")
        return "default_user"

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        """Get a specific thread"""
        user_id = self._get_current_user_id()
        logger.info(f"Getting thread {thread_id} for user {user_id}")
        
        # Fetch thread items to construct real metadata
        namespace = ("chat_history", user_id, thread_id)
        try:
            items = await self.store.asearch(namespace, query=None, limit=1000)
            if not items:
                logger.warning(f"Thread {thread_id} not found for user {user_id}")
                return None
                
            # Sort by timestamp to find start time
            items.sort(key=lambda x: x.value.get("timestamp", 0))
            
            first_item = items[0].value
            timestamp = first_item.get("timestamp", 0)
            dt = datetime.fromtimestamp(timestamp).isoformat()
            
            # Find a name (first user message)
            name = "Conversation"
            for item in items:
                val = item.value
                if val.get("user_message"):
                    name = val.get("user_message")[:100]
                    break
            
            return ThreadDict(
                id=thread_id,
                createdAt=dt,
                name=name,
                userId=user_id,
                userIdentifier=user_id,
                tags=[],
                metadata={}
            )
        except Exception as e:
            logger.error(f"Failed to get thread {thread_id}: {e}")
            return None

    async def get_steps(self, thread_id: str) -> List[Dict]:
        """Get steps (messages) for a thread"""
        if not self.store:
            return []

        user_id = self._get_current_user_id()
        logger.info(f"Getting steps for thread {thread_id} user {user_id}")
        return await self._get_steps_for_user(user_id, thread_id)

    async def _get_steps_for_user(self, user_id: str, thread_id: str) -> List[Dict]:
        try:
            namespace = ("chat_history", user_id, thread_id)
            items = await self.store.asearch(namespace, query=None, limit=1000)
            
            # Sort by timestamp
            items.sort(key=lambda x: x.value.get("timestamp", 0))
            
            steps = []
            for item in items:
                val = item.value
                timestamp = val.get("timestamp", 0)
                dt = datetime.fromtimestamp(timestamp).isoformat()
                
                # User message step
                if val.get("user_message"):
                    steps.append({
                        "id": f"{item.key}_user",
                        "threadId": thread_id,
                        "parentId": None,
                        "type": "user_message",
                        "name": "User",
                        "content": val["user_message"],
                        "output": val["user_message"],
                        "createdAt": dt,
                        "start": dt,
                        "end": dt,
                        "input": None,
                        "metadata": {},
                        "generation": None
                    })
                
                # Assistant message step
                if val.get("assistant_message"):
                    steps.append({
                        "id": f"{item.key}_assistant",
                        "threadId": thread_id,
                        "parentId": None,
                        "type": "assistant_message",
                        "name": "Assistant",
                        "content": val["assistant_message"],
                        "output": val["assistant_message"],
                        "createdAt": dt,
                        "start": dt,
                        "end": dt,
                        "input": None,
                        "metadata": {},
                        "generation": None
                    })
            
            return steps
        except Exception as e:
            logger.error(f"Failed to get steps: {e}")
            return []

    async def update_thread(self, thread_id: str, name: Optional[str] = None, user_id: Optional[str] = None, metadata: Optional[Dict] = None, tags: Optional[List[str]] = None):
        """Update a thread"""
        # We don't support updating threads in this simple implementation yet
        # But we must return something to avoid errors if Chainlit calls this
        pass

    async def delete_thread(self, thread_id: str):
        """Delete a thread"""
        # We don't support deleting threads yet
        pass

    async def create_step(self, step_dict: Dict):
        pass

    async def update_step(self, step_dict: Dict):
        pass

    async def delete_step(self, step_id: str):
        pass

    async def build_debug_url(self) -> str:
        return ""

    async def close(self):
        pass

    async def create_element(self, element_dict: Dict):
        pass

    async def delete_element(self, element_id: str):
        pass

    async def delete_feedback(self, feedback_id: str):
        pass

    async def get_element(self, element_id: str):
        pass

    async def get_thread_author(self, thread_id: str) -> str:
        return "unknown"

    async def upsert_feedback(self, feedback: Any) -> str:
        return ""
