
import asyncio
import os
from dotenv import load_dotenv
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import DefaultAzureCredential

async def inspect_container():
    load_dotenv()
    
    endpoint = os.getenv('COSMOS_ENDPOINT')
    database_name = os.getenv('COSMOS_DATABASE', "agent_memory")
    container_name = os.getenv('COSMOS_CONTAINER_CONVERSATIONS', "chat_history")
    partition_key_field = os.getenv('COSMOS_PARTITION_KEY', "user_id")
    
    print(f"Connecting to {endpoint}...")
    print(f"Database: {database_name}")
    print(f"Container: {container_name}")
    
    credential = DefaultAzureCredential()
    client = CosmosClient(url=endpoint, credential=credential)
    
    try:
        database = client.get_database_client(database_name)
        container = database.get_container_client(container_name)
        
        print("\n--- Querying with filters ---")
        
        query = """
            SELECT c.id, c.key, c["value"], c.namespace, c.created_at, c.updated_at, c.version
            FROM c
            WHERE c.user_id = @partition_key
            AND c.namespace[0] = @namespace_0 AND c.namespace[1] = @namespace_1
            ORDER BY c.updated_at DESC
            OFFSET 0 LIMIT 1000
        """
        
        parameters = [
            {"name": "@partition_key", "value": "default_user"},
            {"name": "@namespace_0", "value": "chat_history"},
            {"name": "@namespace_1", "value": "default_user"}
        ]
        
        items = [item async for item in container.query_items(
            query=query, 
            parameters=parameters, 
            partition_key="default_user"
        )]
        
        print(f"Found {len(items)} items.")
        for item in items:
            print("\nItem:")
            print(f"  id: {item.get('id')}")
            print(f"  {partition_key_field}: {item.get(partition_key_field)}")
            print(f"  namespace: {item.get('namespace')}")
            print(f"  key: {item.get('key')}")
            print(f"  updated_at: {item.get('updated_at')}")
            # print(f"  value: {item.get('value')}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()
        await credential.close()

if __name__ == "__main__":
    asyncio.run(inspect_container())
