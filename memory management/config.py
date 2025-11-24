"""
Azure Configuration and Service Initialization
"""

import os
import logging
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
from cosmosdb_store_async import AsyncCosmosDBStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore

logger = logging.getLogger(__name__)


class AzureConfig:
    """Azure service configuration and initialization"""
    
    def __init__(self):
        """Load environment variables and initialize services"""
        logger.info("\n" + "="*60)
        logger.info("AZURE CONFIGURATION INITIALIZATION")
        logger.info("="*60)
        
        load_dotenv()
        logger.info("✓ Environment variables loaded")
        
        # Azure OpenAI Configuration
        self.openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.openai_version = os.getenv('AZURE_OPENAI_VERSION')
        
        # Cosmos DB Configuration
        self.cosmos_endpoint = os.getenv('COSMOS_ENDPOINT')
        self.cosmos_key = os.getenv('COSMOS_KEY')
        self.cosmos_database = "agent_memory"
        self.cosmos_container_facts = "user_facts"
        self.cosmos_container_conversations = "chat_history"
        
        # Tavily Configuration
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        
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
        logger.info("✓ Azure OpenAI Model initialized")
        
        # Azure OpenAI Embeddings
        logger.info("Initializing Azure OpenAI Embeddings (text-embedding-3-large)...")
        self.embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-large",
            azure_endpoint=self.openai_endpoint,
            api_key=self.openai_api_key,
            api_version=self.openai_version
        )
        logger.info("✓ Azure OpenAI Embeddings initialized")
        
        # Tavily Search Tool
        logger.info("Checking Tavily Search Tool...")
        if self.tavily_api_key:
            os.environ["TAVILY_API_KEY"] = self.tavily_api_key
            self.tavily_tool = TavilySearchResults(max_results=3)
            logger.info("✓ Tavily search initialized")
            print("✓ Tavily search initialized")
        else:
            self.tavily_tool = None
            logger.warning("⚠ Tavily API key not found - search agent disabled")
            print("⚠ Tavily API key not found - search agent disabled")
        
        # Cosmos DB Client (using DefaultAzureCredential with optimized settings)
        logger.info("\n[COSMOS DB CONNECTION]")
        logger.info(f"Cosmos Endpoint: {self.cosmos_endpoint}")
        logger.info(f"Database: {self.cosmos_database}")
        logger.info(f"Container: {self.cosmos_container_facts}")
        logger.info("Connecting to Cosmos DB with DefaultAzureCredential...")
        print("🔄 Connecting to Cosmos DB...")
        
        try:
            # Use DefaultAzureCredential with shorter timeouts to avoid hanging
            # Exclude slow/problematic credential types
            logger.info("  - Excluded credentials: VS Code, PowerShell, Azure Dev CLI")
            logger.info("  - Process timeout: 5 seconds")
            
            credential = DefaultAzureCredential(
                exclude_visual_studio_code_credential=True,
                exclude_powershell_credential=True,
                exclude_developer_cli_credential=True,
                process_timeout=5  # Faster timeout
            )
            
            logger.info("  - Creating Cosmos client...")
            cosmos_client = CosmosClient(self.cosmos_endpoint, credential=credential)
            
            logger.info("  - Getting database client...")
            database = cosmos_client.get_database_client(self.cosmos_database)
            
            logger.info("  - Getting container clients...")
            container_facts = database.get_container_client(self.cosmos_container_facts)
            container_conversations = database.get_container_client(self.cosmos_container_conversations)
            
            logger.info("✓ Cosmos DB connected successfully")
            print("✓ Cosmos DB connected")
            
            # Initialize stores
            logger.info("Initializing memory stores...")
            self.facts_store = AsyncCosmosDBStore(container_facts, self.embeddings)
            self.conversation_store = AsyncCosmosDBStore(container_conversations, self.embeddings)
            
            # Use MemorySaver but we'll manually save to Cosmos DB
            self.memory_saver = MemorySaver()
            
            logger.info("✓ Async Cosmos DB facts store initialized (long-term user facts)")
            logger.info("✓ Async Cosmos DB conversation store initialized (persistent chat history)")
            logger.info("✓ MemorySaver initialized (in-memory checkpointer)")
            print("✓ Memory stores initialized")
            print("✓ Async Cosmos DB facts store initialized (long-term user facts)")
            print("✓ Async Cosmos DB conversation store initialized (persistent chat history)")
            print("✓ MemorySaver initialized (in-memory checkpointer)")
            
        except Exception as e:
            logger.error(f"❌ Cosmos DB connection failed: {e}", exc_info=True)
            logger.warning("⚠ Continuing without long-term memory (in-memory only mode)")
            print(f"❌ Cosmos DB connection failed: {e}")
            print("⚠ Continuing without long-term memory...")
            self.facts_store = None
            self.conversation_store = None
            self.memory_saver = MemorySaver()
        
        print("✓ Async Cosmos DB facts store initialized (long-term memory)")
        print("✓ MemorySaver initialized (short-term conversation history)")
    
    def get_model(self):
        """Get Azure OpenAI model"""
        return self.model
    
    def get_embeddings(self):
        """Get Azure OpenAI embeddings"""
        return self.embeddings
    
    def get_tavily_tool(self):
        """Get Tavily search tool (or None)"""
        return self.tavily_tool
    
    def get_facts_store(self):
        """Get Cosmos DB facts store"""
        return self.facts_store
    
    def get_conversation_store(self):
        """Get Cosmos DB conversation store"""
        return self.conversation_store
    
    def get_memory_saver(self):
        """Get MemorySaver checkpointer"""
        return self.memory_saver
