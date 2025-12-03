"""
Cosmos DB Store for LangGraph Memory Management
Follows LangGraph PostgresStore pattern with Azure Cosmos DB optimizations

Features:
- Async-first design with sync wrapper methods
- Semantic search with vector embeddings
- TTL (Time-To-Live) support for automatic expiration
- Batch operations for efficiency
- Filter-based search with operators ($gt, $lt, $eq, $ne, etc.)
- Namespace management with proper isolation
- Parameterized queries for security
- Version tracking and audit trails
- Proper error handling with specific exceptions
"""

import logging
from typing import Optional, Iterable, List, Tuple, Dict, Any, AsyncIterator, Callable
from langgraph.store.base import BaseStore, Item
from datetime import datetime, timedelta
from pydantic import BaseModel
from azure.cosmos.aio import CosmosClient, ContainerProxy, DatabaseProxy
from azure.cosmos.exceptions import CosmosResourceNotFoundError, CosmosHttpResponseError
from azure.identity.aio import DefaultAzureCredential
import asyncio

logger = logging.getLogger(__name__)


class CosmosDBStoreConfig(BaseModel):
    """Configuration for Cosmos DB Store"""
    endpoint: str
    database: str
    container: str
    partition_key: str = "user_id"
    enable_embeddings: bool = True
    embedding_dimensions: int = 3072  # Default to text-embedding-3-large


class CosmosDBStore(BaseStore):
    """
    Async-first Cosmos DB store with full BaseStore interface implementation.
    Follows LangGraph PostgresStore pattern with Cosmos DB optimizations.
    Supports both sync and async methods.
    """
    
    client: CosmosClient
    db: DatabaseProxy
    container: ContainerProxy
    
    def __init__(
        self, 
        container: ContainerProxy, 
        embeddings_model=None,
        partition_key_field: str = "user_id",
        partition_key_func: Optional[Callable[[Tuple[str, ...]], str]] = None,
        embedding_dimensions: int = 3072
    ):
        """Initialize Async Cosmos DB store
        
        Args:
            container: Azure Cosmos DB async container client (azure.cosmos.aio.ContainerProxy)
            embeddings_model: Optional embeddings model for semantic search
            partition_key_field: Name of the partition key field in Cosmos DB (default: "user_id")
            partition_key_func: Optional function to extract partition key from namespace
            embedding_dimensions: Dimensions of the embedding model (default: 3072)
        """
        super().__init__()
        self.container = container
        self.embeddings = embeddings_model
        self.index_config = {"embed": embeddings_model, "dims": embedding_dimensions} if embeddings_model else None
        self.partition_key_field = partition_key_field
        self.partition_key_func = partition_key_func or self._default_partition_key_extractor
    
    @classmethod
    async def create(
        cls,
        credential: DefaultAzureCredential,
        config: CosmosDBStoreConfig,
        embeddings_model=None
    ):
        """Factory method to create store with async client initialization
        
        Args:
            credential: Azure credential for authentication
            config: Store configuration
            embeddings_model: Optional embeddings model
            
        Returns:
            Configured CosmosDBStore instance
        """
        client = CosmosClient(url=config.endpoint, credential=credential)
        db = client.get_database_client(config.database)
        container = db.get_container_client(config.container)
        
        store = cls(
            container, 
            embeddings_model,
            partition_key_field=config.partition_key,
            embedding_dimensions=config.embedding_dimensions
        )
        store.client = client
        store.db = db
        logger.info(f"CosmosDBStore created: database={config.database}, container={config.container}")
        return store
    
    def _default_partition_key_extractor(self, namespace: Tuple[str, ...]) -> str:
        """Default partition key extractor: uses namespace[1] as user_id"""
        return namespace[1] if len(namespace) > 1 else "default"

    def _create_document_id(self, namespace: Tuple[str, ...], key: str) -> str:
        """Create consistent document ID with namespace hierarchy
        
        Args:
            namespace: Namespace tuple
            key: Item key
            
        Returns:
            Document ID string
        """
        # Use partition key as prefix for ID to ensure uniqueness within partition if needed
        # But ID must be unique within partition.
        # Legacy logic: user_id + namespace_suffix + key
        
        pk = self._get_partition_key(namespace)
        
        # Construct a suffix from the rest of the namespace
        # If default extractor is used, namespace[1] is pk.
        # We can just join the whole namespace to be safe and generic.
        ns_str = "_".join(str(ns) for ns in namespace)
        return f"{ns_str}_{key}"
    
    def _get_partition_key(self, namespace: Tuple[str, ...]) -> str:
        """Extract partition key from namespace using configured strategy
        
        Args:
            namespace: Namespace tuple
            
        Returns:
            Partition key value
        """
        return self.partition_key_func(namespace)
    
    async def aput(
        self, 
        namespace: Tuple[str, ...], 
        key: str, 
        value: dict, 
        index: bool | List[str] | None = None,
        ttl_minutes: int | None = None
    ) -> None:
        """Store an item asynchronously with optional embedding and TTL
        
        Args:
            namespace: Namespace tuple for item organization
            key: Unique key for the item
            value: Dictionary value to store
            index: Fields to index for search (True for default, list for specific fields)
            ttl_minutes: Optional TTL in minutes for auto-expiration
        """
        partition_key_val = self._get_partition_key(namespace)
        doc_id = self._create_document_id(namespace, key)
        
        # Check for existing item to track versions
        existing_item = None
        try:
            existing_item = await self.container.read_item(item=doc_id, partition_key=partition_key_val)
        except CosmosResourceNotFoundError:
            pass
        
        item = {
            "id": doc_id,
            self.partition_key_field: partition_key_val,
            "namespace": list(namespace),
            "key": key,
            "value": value,
            "created_at": existing_item["created_at"] if existing_item else datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "version": (existing_item.get("version", 0) + 1) if existing_item else 1,
            "previous_version_id": existing_item.get("id") if existing_item else None
        }
        
        # Add TTL fields
        if ttl_minutes is not None:
            item["expires_at"] = (datetime.utcnow() + timedelta(minutes=ttl_minutes)).isoformat()
            item["ttl_minutes"] = ttl_minutes
        else:
            item["expires_at"] = None
            item["ttl_minutes"] = None
        
        # Generate embedding if requested
        if self.embeddings and index is not False:
            if isinstance(value, dict):
                text_to_embed = None
                
                if index is None or index is True:
                    text_to_embed = value.get('text') or value.get('fact')
                elif isinstance(index, list):
                    # Extract fields, ensure they exist, and convert to string safely
                    texts = []
                    for field in index:
                        val = value.get(field)
                        if val is not None and val != "":
                            texts.append(str(val))
                    text_to_embed = " ".join(texts) if texts else None
                
                if text_to_embed:
                    try:
                        embeddings = await self.embeddings.aembed_documents([text_to_embed])
                        item['embedding'] = embeddings[0] if embeddings else None
                        logger.debug(f"Generated embedding for item: {doc_id}")
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for {doc_id}: {e}")
                        item['embedding'] = None
        
        try:
            await self.container.upsert_item(item)
            logger.debug(f"Stored item: namespace={namespace}, key={key}, version={item['version']}")
        except CosmosHttpResponseError as e:
            logger.error(f"Failed to store item {doc_id}: {e}")
            raise
    
    async def aget(self, namespace: Tuple[str, ...], key: str, refresh_ttl: bool = False) -> Optional[Item]:
        """Retrieve an item asynchronously from Cosmos DB
        
        Args:
            namespace: Namespace tuple
            key: Item key
            refresh_ttl: If True, refresh the TTL on access
            
        Returns:
            Item if found and not expired, None otherwise
        """
        partition_key_val = self._get_partition_key(namespace)
        doc_id = self._create_document_id(namespace, key)
        
        try:
            result = await self.container.read_item(item=doc_id, partition_key=partition_key_val)
            
            # Check if expired
            if result.get("expires_at"):
                expires_at = datetime.fromisoformat(result["expires_at"])
                if datetime.utcnow() > expires_at:
                    logger.debug(f"Item expired: {doc_id}")
                    return None
            
            # Refresh TTL if requested
            if refresh_ttl and result.get("ttl_minutes"):
                result["expires_at"] = (datetime.utcnow() + timedelta(minutes=result["ttl_minutes"])).isoformat()
                result["updated_at"] = datetime.utcnow().isoformat()
                await self.container.upsert_item(result)
                logger.debug(f"Refreshed TTL for item: {doc_id}")
            
            return Item(
                value=result["value"],
                key=result["key"],
                namespace=tuple(result["namespace"]),
                created_at=result.get("created_at"),
                updated_at=result.get("updated_at")
            )
        except CosmosResourceNotFoundError:
            logger.debug(f"Item not found: {doc_id}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving item {doc_id}: {e}")
            raise
    
    async def asearch(
        self, 
        namespace_prefix: Tuple[str, ...], 
        *, 
        query: str | None = None, 
        limit: int = 10,
        offset: int = 0,
        filter: Dict[str, Any] | None = None
    ) -> List[Item]:
        """Search for items asynchronously with optional semantic search and filters
        
        Args:
            namespace_prefix: Namespace prefix to search within
            query: Optional semantic search query
            limit: Maximum number of results
            offset: Number of results to skip
            filter: Optional filter conditions
            
        Returns:
            List of matching items
        """
        partition_key_val = self._get_partition_key(namespace_prefix)
        
        if query and self.embeddings:
            return await self._asemantic_search(partition_key_val, query, limit, offset, filter, namespace_prefix)
        
        return await self._afiltered_search(partition_key_val, limit, offset, filter, namespace_prefix)
    
    async def _asemantic_search(
        self, 
        partition_key_val: str, 
        query: str, 
        limit: int = 10,
        offset: int = 0,
        filter: Dict[str, Any] | None = None,
        namespace_prefix: Tuple[str, ...] | None = None
    ) -> List[Item]:
        """Internal async semantic search with vector similarity
        
        Uses Cosmos DB's VectorDistance function for semantic ranking.
        """
        if not self.embeddings:
            raise ValueError("Embeddings model not configured for semantic search")
        
        try:
            query_embeddings = await self.embeddings.aembed_documents([query])
            query_embedding = query_embeddings[0] if query_embeddings else None
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []
        
        if not query_embedding:
            logger.warning("Empty query embedding generated")
            return []
        
        parameters = [
            {"name": "@partition_key", "value": partition_key_val},
            {"name": "@embedding", "value": query_embedding},
            {"name": "@limit", "value": limit},
            {"name": "@offset", "value": offset}
        ]
        
        filter_clause = ""
        
        # Add namespace filtering
        if namespace_prefix and len(namespace_prefix) > 0:
            namespace_conditions = []
            for idx, ns_part in enumerate(namespace_prefix):
                param_name = f"@namespace_{idx}"
                parameters.append({"name": param_name, "value": ns_part})
                namespace_conditions.append(f"c.namespace[{idx}] = {param_name}")
            
            if namespace_conditions:
                filter_clause += f" AND {' AND '.join(namespace_conditions)}"
        
        # Add custom filters
        if filter:
            filter_conditions = self._build_filter_conditions(filter, parameters)
            if filter_conditions:
                filter_clause += f" AND {filter_conditions}"
        
        sql_query = f"""
            SELECT c.id, c.key, c["value"], c.namespace, c.created_at, c.updated_at, c.version,
                   VectorDistance(c.embedding, @embedding) AS similarity_score
            FROM c
            WHERE c.{self.partition_key_field} = @partition_key 
            AND IS_DEFINED(c.embedding)
            AND (NOT IS_DEFINED(c.expires_at) OR c.expires_at = null OR c.expires_at > GetCurrentDateTime())
            {filter_clause}
            ORDER BY VectorDistance(c.embedding, @embedding)
            OFFSET @offset LIMIT @limit
        """
        
        logger.debug(f"Semantic search query: {sql_query}")
        logger.debug(f"Parameters: {parameters}")
        
        try:
            results_raw = self.container.query_items(
                query=sql_query,
                parameters=parameters,
                partition_key=partition_key_val
            )
            
            items = []
            async for result in results_raw:
                item = Item(
                    value=result["value"],
                    key=result["key"],
                    namespace=tuple(result["namespace"]),
                    created_at=result.get("created_at"),
                    updated_at=result.get("updated_at")
                )
                # Attach similarity score to value dict since Item might be immutable
                if isinstance(item.value, dict):
                    item.value["_similarity_score"] = result.get("similarity_score")
                items.append(item)
            
            logger.debug(f"Semantic search returned {len(items)} results")
            return items
        except CosmosHttpResponseError as e:
            logger.error(f"Semantic search query failed: {e}")
            raise
    
    async def _afiltered_search(
        self,
        partition_key_val: str,
        limit: int = 10,
        offset: int = 0,
        filter: Dict[str, Any] | None = None,
        namespace_prefix: Tuple[str, ...] | None = None
    ) -> List[Item]:
        """Internal async filtered search without semantic ranking"""
        parameters = [
            {"name": "@partition_key", "value": partition_key_val},
            {"name": "@limit", "value": limit},
            {"name": "@offset", "value": offset}
        ]
        
        filter_clause = ""
        
        # Add namespace filtering
        if namespace_prefix and len(namespace_prefix) > 0:
            namespace_conditions = []
            for idx, ns_part in enumerate(namespace_prefix):
                param_name = f"@namespace_{idx}"
                parameters.append({"name": param_name, "value": ns_part})
                namespace_conditions.append(f"c.namespace[{idx}] = {param_name}")
            
            if namespace_conditions:
                filter_clause += f" AND {' AND '.join(namespace_conditions)}"
        
        # Add custom filters
        if filter:
            filter_conditions = self._build_filter_conditions(filter, parameters)
            if filter_conditions:
                filter_clause += f" AND {filter_conditions}"
        
        sql_query = f"""
            SELECT c.id, c.key, c["value"], c.namespace, c.created_at, c.updated_at, c.version
            FROM c
            WHERE c.{self.partition_key_field} = @partition_key
            AND (NOT IS_DEFINED(c.expires_at) OR c.expires_at = null OR c.expires_at > GetCurrentDateTime())
            {filter_clause}
            ORDER BY c.updated_at DESC
            OFFSET @offset LIMIT @limit
        """
        
        logger.debug(f"Filtered search query: {sql_query}")
        
        try:
            items_raw = self.container.query_items(
                query=sql_query,
                parameters=parameters,
                partition_key=partition_key_val
            )
            
            results = []
            async for item in items_raw:
                results.append(Item(
                    value=item["value"],
                    key=item["key"],
                    namespace=tuple(item["namespace"]),
                    created_at=item.get("created_at"),
                    updated_at=item.get("updated_at")
                ))
            
            logger.debug(f"Filtered search returned {len(results)} results")
            return results
        except CosmosHttpResponseError as e:
            logger.error(f"Filtered search query failed: {e}")
            raise
    
    def _build_filter_conditions(self, filter: Dict[str, Any], parameters: List[Dict]) -> str:
        """Build parameterized SQL filter conditions from filter dict
        
        Supports operators: $eq, $ne, $gt, $gte, $lt, $lte
        """
        conditions = []
        param_counter = len(parameters)
        
        for field, value in filter.items():
            if isinstance(value, dict):
                # Handle operator-based filters
                for op, val in value.items():
                    param_name = f"@filter_{param_counter}"
                    param_counter += 1
                    
                    if op == "$eq":
                        conditions.append(f'c["value"]["{field}"] = {param_name}')
                    elif op == "$ne":
                        conditions.append(f'c["value"]["{field}"] != {param_name}')
                    elif op == "$gt":
                        conditions.append(f'c["value"]["{field}"] > {param_name}')
                    elif op == "$gte":
                        conditions.append(f'c["value"]["{field}"] >= {param_name}')
                    elif op == "$lt":
                        conditions.append(f'c["value"]["{field}"] < {param_name}')
                    elif op == "$lte":
                        conditions.append(f'c["value"]["{field}"] <= {param_name}')
                    else:
                        logger.warning(f"Unsupported filter operator: {op}")
                        continue
                    
                    parameters.append({"name": param_name, "value": val})
            else:
                # Simple equality filter
                param_name = f"@filter_{param_counter}"
                param_counter += 1
                conditions.append(f'c["value"]["{field}"] = {param_name}')
                parameters.append({"name": param_name, "value": value})
        
        return " AND ".join(conditions)
    
    async def adelete(self, namespace: Tuple[str, ...], key: str) -> None:
        """Delete a single item asynchronously
        
        Args:
            namespace: Namespace tuple
            key: Item key
        """
        partition_key_val = self._get_partition_key(namespace)
        doc_id = self._create_document_id(namespace, key)
        
        try:
            await self.container.delete_item(item=doc_id, partition_key=partition_key_val)
            logger.debug(f"Deleted item: {doc_id}")
        except CosmosResourceNotFoundError:
            logger.debug(f"Item not found for deletion: {doc_id}")
        except Exception as e:
            logger.error(f"Failed to delete item {doc_id}: {e}")
            raise
    
    async def abatch(self, ops: List[tuple]) -> List:
        """Execute a batch of operations asynchronously
        
        Args:
            ops: List of operation tuples: (op_type, *args)
            
        Returns:
            List of results for each operation
        """
        results = []
        for op in ops:
            try:
                op_type = op[0]
                if op_type == "put":
                    namespace, key, value = op[1], op[2], op[3]
                    await self.aput(namespace, key, value)
                    results.append(None)
                elif op_type == "get":
                    namespace, key = op[1], op[2]
                    item = await self.aget(namespace, key)
                    results.append(item)
                elif op_type == "search":
                    namespace_prefix = op[1]
                    kwargs = op[2] if len(op) > 2 else {}
                    items = await self.asearch(namespace_prefix, **kwargs)
                    results.append(items)
                elif op_type == "delete":
                    namespace, key = op[1], op[2]
                    await self.adelete(namespace, key)
                    results.append(None)
                else:
                    logger.warning(f"Unsupported batch operation: {op_type}")
                    results.append(ValueError(f"Unsupported operation: {op_type}"))
            except Exception as e:
                logger.error(f"Batch operation failed: {op_type}, error: {e}")
                results.append(e)
        
        return results
    
    # Sync wrapper methods for compatibility
    def put(self, namespace: Tuple[str, ...], key: str, value: dict, **kwargs) -> None:
        """Sync wrapper for aput - runs async operation in event loop"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Cannot use sync methods in async context. Use aput() instead.")
            return loop.run_until_complete(self.aput(namespace, key, value, **kwargs))
        except RuntimeError as e:
            if "no running event loop" in str(e).lower() or "no current event loop" in str(e).lower():
                # Create new event loop if none exists
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.aput(namespace, key, value, **kwargs))
                finally:
                    loop.close()
            raise
    
    def get(self, namespace: Tuple[str, ...], key: str, **kwargs) -> Optional[Item]:
        """Sync wrapper for aget"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Cannot use sync methods in async context. Use aget() instead.")
            return loop.run_until_complete(self.aget(namespace, key, **kwargs))
        except RuntimeError as e:
            if "no running event loop" in str(e).lower() or "no current event loop" in str(e).lower():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.aget(namespace, key, **kwargs))
                finally:
                    loop.close()
            raise
    
    def search(self, namespace_prefix: Tuple[str, ...], **kwargs) -> List[Item]:
        """Sync wrapper for asearch"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Cannot use sync methods in async context. Use asearch() instead.")
            return loop.run_until_complete(self.asearch(namespace_prefix, **kwargs))
        except RuntimeError as e:
            if "no running event loop" in str(e).lower() or "no current event loop" in str(e).lower():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.asearch(namespace_prefix, **kwargs))
                finally:
                    loop.close()
            raise
    
    def delete(self, namespace: Tuple[str, ...], key: str) -> None:
        """Sync wrapper for adelete"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Cannot use sync methods in async context. Use adelete() instead.")
            return loop.run_until_complete(self.adelete(namespace, key))
        except RuntimeError as e:
            if "no running event loop" in str(e).lower() or "no current event loop" in str(e).lower():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.adelete(namespace, key))
                finally:
                    loop.close()
            raise
    
    def batch(self, ops: List[tuple]) -> List:
        """Sync wrapper for abatch"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Cannot use sync methods in async context. Use abatch() instead.")
            return loop.run_until_complete(self.abatch(ops))
        except RuntimeError as e:
            if "no running event loop" in str(e).lower() or "no current event loop" in str(e).lower():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.abatch(ops))
                finally:
                    loop.close()
            raise
