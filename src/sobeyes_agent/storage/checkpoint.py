"""
Async Cosmos DB Checkpoint Saver for LangGraph
Provides persistent checkpoint storage in Azure Cosmos DB

Features:
- Parameterized SQL queries for security
- Cross-partition query support
- Base64 serialization for checkpoint data
- Async operations for efficiency
"""

import asyncio
import base64
import json
from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple, Iterable

from azure.cosmos.aio import CosmosClient, DatabaseProxy
from langchain_core.runnables import RunnableConfig
from azure.identity import DefaultAzureCredential
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from pydantic import BaseModel


class AsyncCosmosDBCheckpointSaverConfig(BaseModel):
    """Configuration for Cosmos DB Checkpoint Saver"""
    DATABASE: str
    ENDPOINT: str
    CHECKPOINTS_CONTAINER: str
    CHECKPOINT_WRITES_CONTAINER: str


class AsyncCosmosDBCheckpointSaver(BaseCheckpointSaver):
    """A checkpoint saver that stores checkpoints in a CosmosDB database."""

    client: CosmosClient
    db: DatabaseProxy

    def __init__(self, credential: DefaultAzureCredential, config: AsyncCosmosDBCheckpointSaverConfig) -> None:
        super().__init__()

        # Initialize Cosmos DB client
        self.client = CosmosClient(url=config.ENDPOINT, credential=credential)
        self.db = self.client.get_database_client(config.DATABASE)

        # Get containers: checkpoints and checkpoint_writes
        self.checkpoints_container = self.db.get_container_client(
            config.CHECKPOINTS_CONTAINER
        )
        self.writes_container = self.db.get_container_client(
            config.CHECKPOINT_WRITES_CONTAINER
        )

    def dumps_typed(self, obj: Any) -> Tuple[str, str]:
        """Serializes an object and encodes the serialized data in base64 format."""
        type_, serialized_ = self.serde.dumps_typed(obj)
        return type_, base64.b64encode(serialized_).decode("utf-8")

    def loads_typed(self, data: Tuple[str, str]) -> Any:
        """Deserialize a tuple containing a string and a base64 encoded string."""
        return self.serde.loads_typed(
            (data[0], base64.b64decode(data[1].encode("utf-8")))
        )

    def dumps(self, obj: Any) -> str:
        """Serializes an object to a JSON string."""
        return json.dumps(obj, default=str)

    def loads(self, data: str) -> Any:
        """Deserialize a JSON string into a Python object."""
        return json.loads(data)

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database."""
        assert "configurable" in config
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        user_id = config["configurable"].get("user_id")
        
        parameters = [
            {"name": "@thread_id", "value": thread_id},
            {"name": "@checkpoint_ns", "value": checkpoint_ns},
        ]
        
        where_clause = "c.thread_id = @thread_id AND c.checkpoint_ns = @checkpoint_ns"
        if user_id:
            parameters.append({"name": "@user_id", "value": user_id})
            where_clause += " AND c.user_id = @user_id"

        if checkpoint_id := get_checkpoint_id(config):
            query = f"SELECT * FROM c WHERE {where_clause} AND c.checkpoint_id = @checkpoint_id"
            parameters.append({"name": "@checkpoint_id", "value": checkpoint_id})
        else:
            query = f"SELECT * FROM c WHERE {where_clause} ORDER BY c.checkpoint_id DESC"

        result = [item async for item in self.checkpoints_container.query_items(query, parameters=parameters)]
        if result:
            doc = result[0]
            config_values = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": doc["checkpoint_id"],
            }
            if user_id:
                config_values["user_id"] = user_id

            checkpoint = self.loads_typed((doc["type"], doc["checkpoint"]))
            
            writes_query = f"SELECT * FROM c WHERE {where_clause} AND c.checkpoint_id = @checkpoint_id"
            writes_parameters = [
                {"name": "@thread_id", "value": thread_id},
                {"name": "@checkpoint_ns", "value": checkpoint_ns},
                {"name": "@checkpoint_id", "value": doc["checkpoint_id"]},
            ]
            if user_id:
                writes_parameters.append({"name": "@user_id", "value": user_id})

            _serialized_writes = self.writes_container.query_items(
                writes_query, parameters=writes_parameters
            )
            serialized_writes = [writes async for writes in _serialized_writes]

            pending_writes = [
                (
                    write_doc["task_id"],
                    write_doc["channel"],
                    self.loads_typed((write_doc["type"], write_doc["value"])),
                )
                for write_doc in serialized_writes
            ]
            return CheckpointTuple(
                {"configurable": config_values},
                checkpoint,
                self.loads(doc["metadata"]),
                (
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": doc["parent_checkpoint_id"],
                        }
                    }
                    if doc.get("parent_checkpoint_id")
                    else None
                ),
                pending_writes,
            )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database."""
        where_clauses = []
        parameters = []
        
        if config is not None:
            assert "configurable" in config
            where_clauses.append("c.thread_id = @thread_id AND c.checkpoint_ns = @checkpoint_ns")
            parameters.extend([
                {"name": "@thread_id", "value": config['configurable']['thread_id']},
                {"name": "@checkpoint_ns", "value": config['configurable'].get('checkpoint_ns', '')}
            ])
            if user_id := config['configurable'].get('user_id'):
                where_clauses.append("c.user_id = @user_id")
                parameters.append({"name": "@user_id", "value": user_id})

        if filter:
            for key, value in filter.items():
                param_name = f"@metadata_{key}"
                where_clauses.append(f"c.metadata.{key} = {param_name}")
                parameters.append({"name": param_name, "value": value})

        if before is not None:
            assert "configurable" in before
            where_clauses.append("c.checkpoint_id < @before_checkpoint_id")
            parameters.append({"name": "@before_checkpoint_id", "value": before['configurable']['checkpoint_id']})

        query = "SELECT * FROM c"
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
            
        query += " ORDER BY c.checkpoint_id DESC"

        if limit is not None:
            query = query.replace("SELECT *", f"SELECT TOP {int(limit)} *")

        result = self.checkpoints_container.query_items(query, parameters=parameters)

        async for doc in result:
            checkpoint = self.loads_typed((doc["type"], doc["checkpoint"]))
            yield CheckpointTuple(
                {
                    "configurable": {
                        "thread_id": doc["thread_id"],
                        "checkpoint_ns": doc["checkpoint_ns"],
                        "checkpoint_id": doc["checkpoint_id"],
                    }
                },
                checkpoint,
                self.loads(doc["metadata"]),
                (
                    {
                        "configurable": {
                            "thread_id": doc["thread_id"],
                            "checkpoint_ns": doc["checkpoint_ns"],
                            "checkpoint_id": doc["parent_checkpoint_id"],
                        }
                    }
                    if doc.get("parent_checkpoint_id")
                    else None
                ),
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database."""
        assert "configurable" in config
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        user_id = config["configurable"].get("user_id")
        type_, serialized_checkpoint = self.dumps_typed(checkpoint)
        doc = {
            "id": f"{thread_id}_{checkpoint_ns}_{checkpoint_id}",
            "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
            "type": type_,
            "checkpoint": serialized_checkpoint,
            "metadata": self.dumps(metadata),
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "user_id": user_id,
        }
        await self.checkpoints_container.upsert_item(doc)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "user_id": user_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint."""
        assert "configurable" in config
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]
        user_id = config["configurable"].get("user_id")
        for idx, (channel, value) in enumerate(writes):
            type_, serialized_value = self.dumps_typed(value)
            doc = {
                "id": f"{thread_id}_{checkpoint_ns}_{checkpoint_id}_{task_id}_{idx}",
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "task_path": task_path,
                "idx": idx,
                "channel": channel,
                "type": type_,
                "value": serialized_value,
                "user_id": user_id,
            }
            await self.writes_container.upsert_item(doc)

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.
        
        This method is a synchronous wrapper for aget_tuple.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Cannot use sync methods in async context. Use aget_tuple() instead.")
            return loop.run_until_complete(self.aget_tuple(config))
        except RuntimeError as e:
            if "no running event loop" in str(e).lower() or "no current event loop" in str(e).lower():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.aget_tuple(config))
                finally:
                    loop.close()
            raise

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterable[CheckpointTuple]:
        """List checkpoints from the database.
        
        This method is a synchronous wrapper for alist.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Cannot use sync methods in async context. Use alist() instead.")
            
            # Helper to collect async generator results
            async def _collect_list():
                results = []
                async for item in self.alist(config, filter=filter, before=before, limit=limit):
                    results.append(item)
                return results
                
            return loop.run_until_complete(_collect_list())
        except RuntimeError as e:
            if "no running event loop" in str(e).lower() or "no current event loop" in str(e).lower():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Helper to collect async generator results
                    async def _collect_list():
                        results = []
                        async for item in self.alist(config, filter=filter, before=before, limit=limit):
                            results.append(item)
                        return results
                    return loop.run_until_complete(_collect_list())
                finally:
                    loop.close()
            raise

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.
        
        This method is a synchronous wrapper for aput.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Cannot use sync methods in async context. Use aput() instead.")
            return loop.run_until_complete(self.aput(config, checkpoint, metadata, new_versions))
        except RuntimeError as e:
            if "no running event loop" in str(e).lower() or "no current event loop" in str(e).lower():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.aput(config, checkpoint, metadata, new_versions))
                finally:
                    loop.close()
            raise

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.
        
        This method is a synchronous wrapper for aput_writes.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Cannot use sync methods in async context. Use aput_writes() instead.")
            return loop.run_until_complete(self.aput_writes(config, writes, task_id, task_path))
        except RuntimeError as e:
            if "no running event loop" in str(e).lower() or "no current event loop" in str(e).lower():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.aput_writes(config, writes, task_id, task_path))
                finally:
                    loop.close()
            raise
            
    async def adelete_thread(self, config: RunnableConfig) -> None:
        """Delete a thread from the database."""
        # Not implemented yet, but required by base class
        pass
        
    def delete_thread(self, config: RunnableConfig) -> None:
        """Delete a thread from the database."""
        # Not implemented yet, but required by base class
        pass
