"""
LangGraph Multi-Agent Graph Builder
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from models import State, RouteDecision
from agent_nodes import AgentNodes

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds and compiles the LangGraph multi-agent system"""
    
    def __init__(self, config):
        """
        Initialize graph builder with Azure config
        
        Args:
            config: AzureConfig instance with initialized services
        """
        logger.info("\n[GRAPH BUILDER] Initializing...")
        
        self.config = config
        self.model = config.get_model()
        self.facts_store = config.get_facts_store()
        self.conversation_store = config.get_conversation_store()
        self.memory_saver = config.get_memory_saver()
        self.tavily_tool = config.get_tavily_tool()
        
        logger.info(f"[GRAPH BUILDER] Configuration:")
        logger.info(f"  - Model: Configured")
        logger.info(f"  - Facts Store: {'Cosmos DB' if self.facts_store else 'In-memory'}")
        logger.info(f"  - Conversation Store: {'Cosmos DB' if self.conversation_store else 'In-memory'}")
        logger.info(f"  - Memory Saver: Configured")
        logger.info(f"  - Tavily Tool: {'Available' if self.tavily_tool else 'Not configured'}")
        
        # Build routing chain
        logger.info("[GRAPH BUILDER] Building routing chain...")
        self.routing_chain = self._build_routing_chain()
        
        # Initialize agent nodes
        logger.info("[GRAPH BUILDER] Initializing agent nodes...")
        self.agent_nodes = AgentNodes(
            model=self.model,
            facts_store=self.facts_store,
            conversation_store=self.conversation_store,
            routing_chain=self.routing_chain,
            tavily_tool=self.tavily_tool
        )
        
        # Build and compile graph
        logger.info("[GRAPH BUILDER] Building and compiling graph...")
        self.graph = self._build_graph()
        logger.info("[GRAPH BUILDER] ✓ Graph compiled successfully")
    
    def _build_routing_chain(self):
        """Build routing decision chain"""
        # Note: Search agent disabled if Tavily not configured
        available_agents = "- Analyst: Data analysis, calculations, business insights, reasoning tasks, technical explanations, data-related questions, general questions"
        
        if self.tavily_tool:
            available_agents += "\n- Search: Web searches, current events, general knowledge, recent news, research, real-time information"
        
        routing_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a routing assistant. Analyze the query and decide which agent should handle it.

Available Agents:
{available_agents}

{"Note: Search agent is currently unavailable. Route all queries to Analyst." if not self.tavily_tool else "Choose the most appropriate agent based on the query type."}"""),
            ("user", "{query}")
        ])
        
        return routing_prompt | self.model.with_structured_output(RouteDecision)
    
    def _select_next_node(self, state: State) -> str:
        """Select next node based on state"""
        return state.get("next", "FINISH")
    
    def _build_graph(self):
        """Build the multi-agent graph"""
        builder = StateGraph(State)
        
        # Add nodes
        builder.add_node("summarization", self.agent_nodes.summarization_node)
        builder.add_node("supervisor", self.agent_nodes.supervisor_node)
        builder.add_node("Analyst", self.agent_nodes.analyst_node)
        builder.add_node("Search", self.agent_nodes.search_node)
        builder.add_node("Tools", self.agent_nodes.tool_node)
        
        # Add edges
        builder.add_edge(START, "summarization")
        builder.add_edge("summarization", "supervisor")
        builder.add_conditional_edges(
            "supervisor",
            self._select_next_node,
            {"Analyst": "Analyst", "Search": "Search", "Tools": "Tools", "FINISH": END}
        )
        builder.add_edge("Analyst", "supervisor")
        builder.add_edge("Search", "supervisor")
        builder.add_edge("Tools", "supervisor")
        
        # Compile graph with memory (LangGraph pattern)
        graph = builder.compile(
            checkpointer=self.memory_saver,
            store=self.facts_store
        )
        
        print("✓ Multi-agent graph compiled with HITL support")
        print("✓ Memory: Checkpointer (short-term) + CosmosDB Store (long-term facts)")
        
        return graph
    
    def get_graph(self):
        """Get compiled graph"""
        return self.graph
    
    def get_agent_nodes(self):
        """Get agent nodes instance"""
        return self.agent_nodes
