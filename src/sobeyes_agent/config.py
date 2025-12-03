"""
Azure Configuration and Service Initialization
"""

import os
import logging
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import DefaultAzureCredential
from .storage.cosmos import CosmosDBStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from .storage.checkpoint import (
    AsyncCosmosDBCheckpointSaver,
    AsyncCosmosDBCheckpointSaverConfig
)

logger = logging.getLogger(__name__)


class AzureConfig:
    """Azure service configuration and initialization"""
    
    def __init__(self):
        """Load environment variables and initialize services"""
        logger.info("\n" + "="*60)
        logger.info("AZURE CONFIGURATION INITIALIZATION")
        logger.info("="*60)
        
        load_dotenv()
        logger.info("Environment variables loaded")
        
        # Azure OpenAI Configuration
        self.openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.openai_version = os.getenv('AZURE_OPENAI_VERSION')
        
        # Cosmos DB Configuration
        self.cosmos_endpoint = os.getenv('COSMOS_ENDPOINT')
        self.cosmos_key = os.getenv('COSMOS_KEY')
        self.cosmos_database = os.getenv('COSMOS_DATABASE', "agent_memory")
        self.cosmos_container_facts = os.getenv('COSMOS_CONTAINER_FACTS', "user_facts")
        self.cosmos_container_conversations = os.getenv('COSMOS_CONTAINER_CONVERSATIONS', "chat_history")
        self.cosmos_container_checkpoints = os.getenv('COSMOS_CONTAINER_CHECKPOINTS', "checkpoints")
        self.cosmos_container_checkpoint_writes = os.getenv('COSMOS_CONTAINER_CHECKPOINT_WRITES', "checkpoint_writes")
        self.cosmos_partition_key = os.getenv('COSMOS_PARTITION_KEY', "user_id")
        
        # Initialize services
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all Azure services"""
        logger.info("\n[SERVICES INITIALIZATION]")
        
        # Azure OpenAI Model
        logger.info("Initializing Azure OpenAI Model (gpt-4o)...")
        self.model = AzureChatOpenAI(
            model="gpt-4o",
            azure_endpoint=self.openai_endpoint,
            api_version=self.openai_version,
            api_key=self.openai_api_key,
            temperature=0
        )
        logger.info("Azure OpenAI Model initialized")
        
        # Azure OpenAI Embeddings
        logger.info("Initializing Azure OpenAI Embeddings (text-embedding-3-large)...")
        self.embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-large",
            azure_endpoint=self.openai_endpoint,
            api_key=self.openai_api_key,
            api_version=self.openai_version
        )
        logger.info("Azure OpenAI Embeddings initialized")
        
        # Cosmos DB Client (using DefaultAzureCredential)
        logger.info("\n[COSMOS DB CONNECTION]")
        logger.info(f"Cosmos Endpoint: {self.cosmos_endpoint}")
        logger.info(f"Database: {self.cosmos_database}")
        logger.info(f"Container: {self.cosmos_container_facts}")
        logger.info(f"Partition Key Field: {self.cosmos_partition_key}")
        logger.info("Connecting to Cosmos DB with DefaultAzureCredential...")
        print("Connecting to Cosmos DB...")
        
        try:
            credential = DefaultAzureCredential(
                process_timeout=10
            )
            
            logger.info("  - Creating async Cosmos client...")
            cosmos_client = CosmosClient(url=self.cosmos_endpoint, credential=credential)
            
            logger.info("  - Getting database client...")
            database = cosmos_client.get_database_client(self.cosmos_database)
            
            logger.info("  - Getting async container clients...")
            container_facts = database.get_container_client(self.cosmos_container_facts)
            container_conversations = database.get_container_client(self.cosmos_container_conversations)
            
            logger.info("Cosmos DB connected successfully")
            print("Cosmos DB connected")
            
            # Initialize stores with async containers
            logger.info("Initializing async memory stores...")
            self.facts_store = CosmosDBStore(
                container_facts, 
                self.embeddings,
                partition_key_field=self.cosmos_partition_key
            )
            self.conversation_store = CosmosDBStore(
                container_conversations, 
                self.embeddings,
                partition_key_field=self.cosmos_partition_key
            )
            
            # Store client references for proper lifecycle management
            self.cosmos_client = cosmos_client
            self.cosmos_database_client = database
            
            # Initialize Cosmos DB Checkpoint Saver (replaces MemorySaver)
            logger.info("Initializing AsyncCosmosDBCheckpointSaver...")
            checkpoint_config = AsyncCosmosDBCheckpointSaverConfig(
                DATABASE=self.cosmos_database,
                ENDPOINT=self.cosmos_endpoint,
                CHECKPOINTS_CONTAINER=self.cosmos_container_checkpoints,
                CHECKPOINT_WRITES_CONTAINER=self.cosmos_container_checkpoint_writes
            )
            self.memory_saver = AsyncCosmosDBCheckpointSaver(credential, checkpoint_config)
            
            logger.info("Async Cosmos DB facts store initialized (long-term user facts)")
            logger.info("Async Cosmos DB conversation store initialized (persistent chat history)")
            logger.info("AsyncCosmosDBCheckpointSaver initialized (persistent checkpoints in Cosmos DB)")
            print("Memory stores initialized")
            print("Async Cosmos DB facts store initialized (long-term user facts)")
            print("Async Cosmos DB conversation store initialized (persistent chat history)")
            print("AsyncCosmosDBCheckpointSaver initialized (persistent checkpoints)")
            
        except Exception as e:
            logger.error(f"Cosmos DB connection failed: {e}", exc_info=True)
            logger.warning("Continuing without long-term memory (in-memory only mode)")
            print(f"Cosmos DB connection failed: {e}")
            print("Continuing without long-term memory...")
            self.facts_store = None
            self.conversation_store = None
            self.memory_saver = MemorySaver()
            print("MemorySaver initialized (short-term conversation history - fallback)")
    
    def get_model(self):
        """Get Azure OpenAI model"""
        return self.model
    
    def get_embeddings(self):
        """Get Azure OpenAI embeddings"""
        return self.embeddings
    
    def get_facts_store(self):
        """Get Cosmos DB facts store"""
        return self.facts_store
    
    def get_conversation_store(self):
        """Get Cosmos DB conversation store"""
        return self.conversation_store
    
    def get_memory_saver(self):
        """Get MemorySaver checkpointer"""
        return self.memory_saver
