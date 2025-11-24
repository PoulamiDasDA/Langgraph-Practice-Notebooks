"""
Enhanced Cosmos DB Store for LangGraph memory management
Based on LangGraph PostgresStore pattern with Cosmos DB optimizations

Features:
- Semantic search with vector embeddings
- TTL (Time-To-Live) support for automatic expiration
- Batch operations for efficiency
- Filter-based search with operators ($gt, $lt, $eq, $ne, etc.)
- Namespace management and listing
- Update with merge support
"""

from typing import Optional, Iterable, Tuple, List, Dict, Any, Literal
from langgraph.store.base import BaseStore, Item
from datetime import datetime, timedelta
import uuid as uuid_lib


class CosmosDBStore(BaseStore):
    """Enhanced Cosmos DB store with full BaseStore interface implementation"""
    
    def __init__(self, container, embeddings_model=None):
        """Initialize Cosmos DB store
        
        Args:
            container: Azure Cosmos DB container client
            embeddings_model: Optional embeddings model for semantic search
        """
        self.container = container
        self.embeddings = embeddings_model
        self.index_config = {"embed": embeddings_model, "dims": 3072} if embeddings_model else None
    
    def put(
        self, 
        namespace: Tuple[str, ...], 
        key: str, 
        value: dict, 
        index: bool | List[str] | None = None,
        ttl_minutes: int | None = None
    ) -> None:
        """Store an item with optional embedding and TTL
        
        Args:
            namespace: Tuple of namespace parts (e.g., ("user_facts", "user_123"))
            key: Unique key within namespace
            value: Dictionary value to store
            index: True to embed all text fields, False to skip, or list of field paths
            ttl_minutes: Optional expiration time in minutes
        
        Example:
            store.put(("user_facts", "user_123"), "note1", {"text": "User likes Paris"})
            store.put(("user_facts", "user_123"), "temp", {"note": "temp"}, ttl_minutes=60)
        """
        user_id = namespace[1] if len(namespace) > 1 else "default"
        
        item = {
            "id": f"{user_id}_{key}",
            "user_id": user_id,
            "namespace": list(namespace),
            "key": key,
            "value": value,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
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
                # Extract text for embedding
                if index is None or index is True:
                    # Default: embed 'text' or 'fact' fields
                    text_to_embed = value.get('text') or value.get('fact')
                elif isinstance(index, list):
                    # Embed specific fields
                    texts = [value.get(field) for field in index if value.get(field)]
                    text_to_embed = " ".join(texts) if texts else None
                else:
                    text_to_embed = None
                
                if text_to_embed:
                    embedding = self.embeddings.embed_query(text_to_embed)
                    item['embedding'] = embedding
        
        self.container.upsert_item(item)
    
    def get(
        self, 
        namespace: Tuple[str, ...], 
        key: str, 
        refresh_ttl: bool = False
    ) -> Optional[Item]:
        """Retrieve an item from Cosmos DB
        
        Args:
            namespace: Namespace tuple
            key: Item key
            refresh_ttl: If True, refresh TTL on access (extend expiration)
        
        Returns:
            Item object or None if not found/expired
        """
        user_id = namespace[1] if len(namespace) > 1 else "default"
        item_id = f"{user_id}_{key}"
        
        try:
            result = self.container.read_item(item=item_id, partition_key=user_id)
            
            # Check if expired
            if result.get("expires_at"):
                expires_at = datetime.fromisoformat(result["expires_at"])
                if datetime.utcnow() > expires_at:
                    return None  # Item expired
            
            # Refresh TTL if requested
            if refresh_ttl and result.get("ttl_minutes"):
                result["expires_at"] = (datetime.utcnow() + timedelta(minutes=result["ttl_minutes"])).isoformat()
                result["updated_at"] = datetime.utcnow().isoformat()
                self.container.upsert_item(result)
            
            return Item(
                value=result["value"],
                key=result["key"],
                namespace=tuple(result["namespace"]),
                created_at=result.get("created_at"),
                updated_at=result.get("updated_at")
            )
        except Exception:
            return None
    
    def search(
        self, 
        namespace_prefix: Tuple[str, ...], 
        *, 
        query: str | None = None, 
        limit: int = 10,
        offset: int = 0,
        filter: Dict[str, Any] | None = None
    ) -> Iterable[Item]:
        """Search for items with optional semantic search and filters
        
        Args:
            namespace_prefix: Namespace to search in
            query: Optional semantic search query (uses embeddings)
            limit: Maximum number of results
            offset: Number of results to skip (pagination)
            filter: Filter conditions (e.g., {"age": {"$gt": 18}, "city": "Paris"})
        
        Returns:
            Iterable of Item objects
        
        Example:
            # Semantic search
            results = store.search(("user_facts", "user_123"), query="Paris travel")
            
            # Filtered search
            results = store.search(
                ("user_facts", "user_123"),
                filter={"age": {"$gt": 25}, "city": "Paris"}
            )
        """
        user_id = namespace_prefix[1] if len(namespace_prefix) > 1 else "default"
        
        # Semantic search if query provided
        if query and self.embeddings:
            return self._semantic_search(user_id, query, limit, offset, filter)
        
        # Regular search with filters
        return self._filtered_search(user_id, limit, offset, filter)
    
    def _semantic_search(
        self, 
        user_id: str, 
        query: str, 
        limit: int = 10,
        offset: int = 0,
        filter: Dict[str, Any] | None = None
    ):
        """Internal semantic search using vector embeddings"""
        if not self.embeddings:
            raise ValueError("Embeddings model not configured")
        
        query_embedding = self.embeddings.embed_query(query)
        
        # Build filter conditions
        filter_clause = ""
        parameters = [
            {"name": "@user_id", "value": user_id},
            {"name": "@embedding", "value": query_embedding},
            {"name": "@limit", "value": limit},
            {"name": "@offset", "value": offset}
        ]
        
        if filter:
            filter_conditions = self._build_filter_conditions(filter, parameters)
            if filter_conditions:
                filter_clause = f"AND {filter_conditions}"
        
        # Exclude expired items
        sql_query = f"""
            SELECT c.id, c.key, c["value"], c.namespace, c.created_at, c.updated_at,
                   VectorDistance(c.embedding, @embedding) AS similarity_score
            FROM c
            WHERE c.user_id = @user_id 
            AND IS_DEFINED(c.embedding)
            AND (c.expires_at IS NULL OR c.expires_at > GetCurrentDateTime())
            {filter_clause}
            ORDER BY VectorDistance(c.embedding, @embedding)
            OFFSET @offset LIMIT @limit
        """
        
        results_raw = self.container.query_items(
            query=sql_query,
            parameters=parameters,
            partition_key=user_id
        )
        
        items = []
        for result in results_raw:
            items.append(Item(
                value=result["value"],
                key=result["key"],
                namespace=tuple(result["namespace"]),
                created_at=result.get("created_at"),
                updated_at=result.get("updated_at")
            ))
        return items
    
    def _filtered_search(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0,
        filter: Dict[str, Any] | None = None
    ):
        """Internal regular search with filter conditions"""
        parameters = [
            {"name": "@user_id", "value": user_id},
            {"name": "@limit", "value": limit},
            {"name": "@offset", "value": offset}
        ]
        
        filter_clause = ""
        if filter:
            filter_conditions = self._build_filter_conditions(filter, parameters)
            if filter_conditions:
                filter_clause = f"AND {filter_conditions}"
        
        sql_query = f"""
            SELECT c.id, c.key, c["value"], c.namespace, c.created_at, c.updated_at
            FROM c
            WHERE c.user_id = @user_id
            AND (c.expires_at IS NULL OR c.expires_at > GetCurrentDateTime())
            {filter_clause}
            ORDER BY c.updated_at DESC
            OFFSET @offset LIMIT @limit
        """
        
        items_raw = self.container.query_items(
            query=sql_query,
            parameters=parameters,
            partition_key=user_id
        )
        
        results = []
        for item in items_raw:
            results.append(Item(
                value=item["value"],
                key=item["key"],
                namespace=tuple(item["namespace"]),
                created_at=item.get("created_at"),
                updated_at=item.get("updated_at")
            ))
        return results
    
    def _build_filter_conditions(self, filter: Dict[str, Any], parameters: List[Dict]) -> str:
        """Build SQL filter conditions from filter dict
        
        Supports operators: $eq, $ne, $gt, $gte, $lt, $lte
        Example: {"age": {"$gt": 18}, "city": "Paris"}
        """
        conditions = []
        param_counter = len(parameters)
        
        for field, value in filter.items():
            if isinstance(value, dict):
                # Operator-based filter
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
                        continue
                    
                    parameters.append({"name": param_name, "value": val})
            else:
                # Simple equality
                param_name = f"@filter_{param_counter}"
                param_counter += 1
                conditions.append(f'c["value"]["{field}"] = {param_name}')
                parameters.append({"name": param_name, "value": value})
        
        return " AND ".join(conditions)
    
    def delete(self, namespace: Tuple[str, ...], key: str) -> None:
        """Delete a single item
        
        Args:
            namespace: Namespace tuple
            key: Item key
        
        Example:
            store.delete(("user_facts", "user_123"), "old_note")
        """
        user_id = namespace[1] if len(namespace) > 1 else "default"
        item_id = f"{user_id}_{key}"
        
        try:
            self.container.delete_item(item=item_id, partition_key=user_id)
        except Exception:
            pass
    
    def update(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: dict,
        merge: bool = False
    ) -> None:
        """Update an existing item
        
        Args:
            namespace: Namespace tuple
            key: Item key
            value: New value
            merge: If True, merge with existing value; if False, replace entirely
        
        Example:
            # Replace
            store.update(("user_facts", "user_123"), "profile", {"name": "John"})
            
            # Merge
            store.update(("user_facts", "user_123"), "profile", 
                        {"last_active": "2025-01-20"}, merge=True)
        """
        if merge:
            # Get existing item and merge
            existing = self.get(namespace, key)
            if existing:
                merged_value = {**existing.value, **value}
                self.put(namespace, key, merged_value)
            else:
                self.put(namespace, key, value)
        else:
            # Replace entirely
            self.put(namespace, key, value)
    
    def exists(self, namespace: Tuple[str, ...], key: str) -> bool:
        """Check if item exists without retrieving full data
        
        Args:
            namespace: Namespace tuple
            key: Item key
        
        Returns:
            True if exists, False otherwise
        
        Example:
            if store.exists(("user_facts", "user_123"), "preferences"):
                print("User has preferences")
        """
        user_id = namespace[1] if len(namespace) > 1 else "default"
        item_id = f"{user_id}_{key}"
        
        try:
            query = "SELECT VALUE 1 FROM c WHERE c.id = @id AND c.user_id = @user_id"
            parameters = [
                {"name": "@id", "value": item_id},
                {"name": "@user_id", "value": user_id}
            ]
            
            results = list(self.container.query_items(
                query=query,
                parameters=parameters,
                partition_key=user_id,
                max_item_count=1
            ))
            
            return len(results) > 0
        except Exception:
            return False
    
    def count(self, namespace: Tuple[str, ...] | None = None) -> int:
        """Count items in namespace or entire store
        
        Args:
            namespace: If provided, count items in this namespace; if None, count all
        
        Returns:
            Number of items
        
        Example:
            total = store.count()
            user_facts = store.count(("user_facts", "user_123"))
        """
        if namespace:
            user_id = namespace[1] if len(namespace) > 1 else "default"
            query = "SELECT VALUE COUNT(1) FROM c WHERE c.user_id = @user_id"
            parameters = [{"name": "@user_id", "value": user_id}]
            partition_key = user_id
        else:
            query = "SELECT VALUE COUNT(1) FROM c"
            parameters = []
            partition_key = None
        
        results = list(self.container.query_items(
            query=query,
            parameters=parameters,
            partition_key=partition_key,
            enable_cross_partition_query=(partition_key is None)
        ))
        
        return results[0] if results else 0
    
    def list_namespaces(
        self,
        prefix: Tuple[str, ...] = (),
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Tuple[str, ...]]:
        """List all unique namespaces with optional prefix filter
        
        Args:
            prefix: Only return namespaces starting with this prefix
            max_depth: Limit namespace depth (e.g., max_depth=1 returns only top level)
            limit: Maximum number of namespaces to return
            offset: Number of results to skip
        
        Returns:
            List of namespace tuples
        
        Example:
            # All namespaces
            all_ns = store.list_namespaces()
            
            # User facts only
            user_ns = store.list_namespaces(prefix=("user_facts",))
            
            # Top level only
            top_level = store.list_namespaces(max_depth=1)
        """
        # Query for distinct namespaces
        query = "SELECT DISTINCT c.namespace FROM c"
        
        results = self.container.query_items(
            query=query,
            enable_cross_partition_query=True
        )
        
        # Collect and filter namespaces
        namespaces = set()
        for item in results:
            ns = tuple(item["namespace"])
            
            # Apply prefix filter
            if prefix and not self._namespace_matches_prefix(ns, prefix):
                continue
            
            # Apply depth limit
            if max_depth is not None:
                ns = ns[:max_depth]
            
            namespaces.add(ns)
        
        # Convert to sorted list with pagination
        sorted_namespaces = sorted(list(namespaces))
        return sorted_namespaces[offset:offset + limit]
    
    def _namespace_matches_prefix(self, namespace: Tuple[str, ...], prefix: Tuple[str, ...]) -> bool:
        """Check if namespace starts with prefix"""
        if len(namespace) < len(prefix):
            return False
        return namespace[:len(prefix)] == prefix
    
    def sweep_expired(self) -> int:
        """Remove expired items based on TTL
        
        Returns:
            Number of items deleted
        
        Example:
            deleted = store.sweep_expired()
            print(f"Removed {deleted} expired items")
        """
        query = """
            SELECT c.id, c.user_id 
            FROM c 
            WHERE c.expires_at IS NOT NULL 
            AND c.expires_at < GetCurrentDateTime()
        """
        
        expired_items = list(self.container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        
        deleted_count = 0
        for item in expired_items:
            try:
                self.container.delete_item(item=item["id"], partition_key=item["user_id"])
                deleted_count += 1
            except Exception:
                pass
        
        return deleted_count
    
    def batch(self, ops: List[tuple]) -> List:
        """Execute a batch of operations efficiently
        
        Args:
            ops: List of operation tuples
        
        Operations format:
            - ("put", namespace, key, value)
            - ("get", namespace, key)
            - ("delete", namespace, key)
            - ("search", namespace_prefix)
        
        Returns:
            List of results (None for put/delete, Item for get, list for search)
        
        Example:
            results = store.batch([
                ("put", ("user_facts", "user_123"), "key1", {"text": "value1"}),
                ("get", ("user_facts", "user_123"), "key2"),
                ("delete", ("user_facts", "user_123"), "old_key")
            ])
        """
        results = []
        for op in ops:
            try:
                op_type = op[0]
                if op_type == "put":
                    namespace, key, value = op[1], op[2], op[3]
                    self.put(namespace, key, value)
                    results.append(None)
                elif op_type == "get":
                    namespace, key = op[1], op[2]
                    item = self.get(namespace, key)
                    results.append(item)
                elif op_type == "search":
                    namespace_prefix = op[1]
                    items = list(self.search(namespace_prefix))
                    results.append(items)
                elif op_type == "delete":
                    namespace, key = op[1], op[2]
                    self.delete(namespace, key)
                    results.append(None)
            except Exception as e:
                results.append(e)
        return results
    
    async def abatch(self, ops: List[tuple]) -> List:
        """Execute a batch of operations asynchronously
        
        Note: Currently falls back to synchronous execution
        """
        return self.batch(ops)
