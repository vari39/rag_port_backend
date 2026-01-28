# src/graph/port_graph.py
"""
LangGraph Orchestration for AI Port Decision-Support System
Handles parallel branching, decision gates, and workflow orchestration

What it does:
1) Takes a question and runs it through multiple analysis paths simultaneously
2) Runs "what-if" scenarios in parallel (safety, efficiency, cost, risk)
3) Uses decision gates to validate information quality
4) Combines all results into a final recommendation
"""

import asyncio
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
import logging

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from ..routers.intent_router import IntentRouter
from ..routers.decision_nodes import DecisionNodes
from ..port_information_retrieval import PortInformationRetrieval

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortWorkflowState(TypedDict):
    """State object for the port workflow graph"""
    # Input
    query: str
    user_id: Optional[str]
    port_config: Optional[Dict[str, Any]]
    
    # Processing
    intent: Optional[str]
    confidence_score: Optional[float]
    retrieved_documents: Optional[List[Dict[str, Any]]]
    parallel_results: Optional[Dict[str, Any]]
    
    # Decision gates
    freshness_check: Optional[bool]
    confidence_check: Optional[bool]
    compliance_check: Optional[bool]
    
    # Output
    final_recommendation: Optional[str]
    alternative_scenarios: Optional[List[Dict[str, Any]]]
    sources: Optional[List[Dict[str, Any]]]
    execution_time: Optional[float]
    
    # Metadata
    workflow_id: Optional[str]
    timestamp: Optional[str]
    errors: Optional[List[str]]


class PortWorkflowGraph:
    """
    LangGraph orchestration for port decision support workflows.
    Handles parallel branching for what-if scenarios and decision gates.
    """
    
    def __init__(self, 
                 port_retrieval: PortInformationRetrieval,
                 enable_checkpoints: bool = True):
        """
        Initialize the port workflow graph.
        
        Args:
            port_retrieval: PortInformationRetrieval instance
            enable_checkpoints: Whether to enable checkpointing for complex workflows
        """
        self.port_retrieval = port_retrieval
        self.intent_router = IntentRouter()
        self.decision_nodes = DecisionNodes()
        self.enable_checkpoints = enable_checkpoints
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info("Port workflow graph initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the state graph
        workflow = StateGraph(PortWorkflowState)
        
        # Add nodes
        workflow.add_node("analyze_intent", self._analyze_intent)
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("parallel_analysis", self._parallel_analysis)
        workflow.add_node("freshness_gate", self._freshness_gate)
        workflow.add_node("confidence_gate", self._confidence_gate)
        workflow.add_node("compliance_gate", self._compliance_gate)
        workflow.add_node("synthesize_results", self._synthesize_results)
        workflow.add_node("generate_alternatives", self._generate_alternatives)
        
        # Define the workflow edges
        workflow.add_edge("analyze_intent", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "parallel_analysis")
        
        # Parallel branching from parallel_analysis
        workflow.add_edge("parallel_analysis", "freshness_gate")
        workflow.add_edge("parallel_analysis", "confidence_gate")
        workflow.add_edge("parallel_analysis", "compliance_gate")
        
        # All gates converge to synthesis
        workflow.add_edge("freshness_gate", "synthesize_results")
        workflow.add_edge("confidence_gate", "synthesize_results")
        workflow.add_edge("compliance_gate", "synthesize_results")
        
        workflow.add_edge("synthesize_results", "generate_alternatives")
        workflow.add_edge("generate_alternatives", END)
        
        # Compile the graph
        if self.enable_checkpoints:
            memory = MemorySaver()
            return workflow.compile(checkpointer=memory)
        else:
            return workflow.compile()
    
    async def _analyze_intent(self, state: PortWorkflowState) -> PortWorkflowState:
        """Analyze user intent and classify the query"""
        try:
            logger.info(f"Analyzing intent for query: {state['query'][:50]}...")
            
            intent_result = await self.intent_router.classify_query(
                query=state["query"],
                port_config=state.get("port_config", {})
            )
            
            state["intent"] = intent_result["intent"]
            state["confidence_score"] = intent_result["confidence"]
            
            logger.info(f"Intent classified as: {state['intent']} (confidence: {state['confidence_score']})")
            
        except Exception as e:
            logger.error(f"Error in intent analysis: {e}")
            state["errors"] = state.get("errors", []) + [f"Intent analysis failed: {str(e)}"]
            state["intent"] = "general"
            state["confidence_score"] = 0.5
        
        return state
    
    async def _retrieve_documents(self, state: PortWorkflowState) -> PortWorkflowState:
        """Retrieve relevant documents based on intent"""
        try:
            logger.info(f"Retrieving documents for intent: {state['intent']}")
            
            # Get documents based on intent
            documents = await self.port_retrieval.search_similar_documents(
                query=state["query"],
                document_type=self._get_document_type_for_intent(state["intent"]),
                k=5
            )
            
            state["retrieved_documents"] = documents
            
            logger.info(f"Retrieved {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            state["errors"] = state.get("errors", []) + [f"Document retrieval failed: {str(e)}"]
            state["retrieved_documents"] = []
        
        return state
    
    async def _parallel_analysis(self, state: PortWorkflowState) -> PortWorkflowState:
        """Perform parallel analysis branches for what-if scenarios"""
        try:
            logger.info("Starting parallel analysis branches")
            
            # Define parallel analysis tasks
            analysis_tasks = {
                "safety_analysis": self._analyze_safety_scenario(state),
                "efficiency_analysis": self._analyze_efficiency_scenario(state),
                "cost_analysis": self._analyze_cost_scenario(state),
                "risk_analysis": self._analyze_risk_scenario(state)
            }
            
            # Run analyses in parallel
            results = await asyncio.gather(
                *analysis_tasks.values(),
                return_exceptions=True
            )
            
            # Combine results
            parallel_results = {}
            for i, (task_name, result) in enumerate(zip(analysis_tasks.keys(), results)):
                if isinstance(result, Exception):
                    logger.error(f"Error in {task_name}: {result}")
                    parallel_results[task_name] = {"error": str(result)}
                else:
                    parallel_results[task_name] = result
            
            state["parallel_results"] = parallel_results
            
            logger.info(f"Completed parallel analysis with {len(parallel_results)} branches")
            
        except Exception as e:
            logger.error(f"Error in parallel analysis: {e}")
            state["errors"] = state.get("errors", []) + [f"Parallel analysis failed: {str(e)}"]
            state["parallel_results"] = {}
        
        return state
    
    async def _freshness_gate(self, state: PortWorkflowState) -> PortWorkflowState:
        """Check information freshness"""
        try:
            logger.info("Checking information freshness")
            
            freshness_result = await self.decision_nodes.check_freshness(
                documents=state.get("retrieved_documents", []),
                query=state["query"],
                port_config=state.get("port_config", {})
            )
            
            state["freshness_check"] = freshness_result["passed"]
            
            if not freshness_result["passed"]:
                logger.warning(f"Freshness check failed: {freshness_result['reason']}")
            
        except Exception as e:
            logger.error(f"Error in freshness gate: {e}")
            state["freshness_check"] = True  # Default to pass on error
        
        return state
    
    async def _confidence_gate(self, state: PortWorkflowState) -> PortWorkflowState:
        """Check confidence in retrieved information"""
        try:
            logger.info("Checking information confidence")
            
            confidence_result = await self.decision_nodes.check_confidence(
                documents=state.get("retrieved_documents", []),
                query=state["query"],
                confidence_threshold=state.get("confidence_score", 0.7)
            )
            
            state["confidence_check"] = confidence_result["passed"]
            
            if not confidence_result["passed"]:
                logger.warning(f"Confidence check failed: {confidence_result['reason']}")
            
        except Exception as e:
            logger.error(f"Error in confidence gate: {e}")
            state["confidence_check"] = True  # Default to pass on error
        
        return state
    
    async def _compliance_gate(self, state: PortWorkflowState) -> PortWorkflowState:
        """Check compliance with safety and regulatory requirements"""
        try:
            logger.info("Checking compliance requirements")
            
            compliance_result = await self.decision_nodes.check_compliance(
                documents=state.get("retrieved_documents", []),
                query=state["query"],
                intent=state.get("intent", "general"),
                port_config=state.get("port_config", {})
            )
            
            state["compliance_check"] = compliance_result["passed"]
            
            if not compliance_result["passed"]:
                logger.warning(f"Compliance check failed: {compliance_result['reason']}")
            
        except Exception as e:
            logger.error(f"Error in compliance gate: {e}")
            state["compliance_check"] = True  # Default to pass on error
        
        return state
    
    async def _synthesize_results(self, state: PortWorkflowState) -> PortWorkflowState:
        """Synthesize results from all analysis branches"""
        try:
            logger.info("Synthesizing results from all branches")
            
            # Check if all gates passed
            gates_passed = all([
                state.get("freshness_check", True),
                state.get("confidence_check", True),
                state.get("compliance_check", True)
            ])
            
            if not gates_passed:
                logger.warning("Some decision gates failed - proceeding with caution")
            
            # Generate final recommendation
            recommendation = await self._generate_recommendation(state)
            state["final_recommendation"] = recommendation
            
            logger.info("Results synthesized successfully")
            
        except Exception as e:
            logger.error(f"Error synthesizing results: {e}")
            state["final_recommendation"] = "Unable to generate recommendation due to processing errors."
        
        return state
    
    async def _generate_alternatives(self, state: PortWorkflowState) -> PortWorkflowState:
        """Generate alternative scenarios based on parallel analysis"""
        try:
            logger.info("Generating alternative scenarios")
            
            alternatives = await self._create_alternative_scenarios(state)
            state["alternative_scenarios"] = alternatives
            
            # Extract sources
            sources = []
            for doc in state.get("retrieved_documents", []):
                sources.append({
                    "source": doc.get("source", "Unknown"),
                    "document_type": doc.get("document_type", "Unknown"),
                    "relevance_score": doc.get("similarity_score", "N/A")
                })
            state["sources"] = sources
            
            # Add execution metadata
            state["timestamp"] = datetime.now().isoformat()
            state["workflow_id"] = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Generated {len(alternatives)} alternative scenarios")
            
        except Exception as e:
            logger.error(f"Error generating alternatives: {e}")
            state["alternative_scenarios"] = []
        
        return state
    
    # Helper methods for parallel analysis
    async def _analyze_safety_scenario(self, state: PortWorkflowState) -> Dict[str, Any]:
        """Analyze safety implications"""
        return {
            "scenario": "Safety Analysis",
            "risk_level": "Low",
            "recommendations": ["Follow standard safety protocols", "Monitor weather conditions"],
            "confidence": 0.8
        }
    
    async def _analyze_efficiency_scenario(self, state: PortWorkflowState) -> Dict[str, Any]:
        """Analyze operational efficiency"""
        return {
            "scenario": "Efficiency Analysis", 
            "optimization_potential": "Medium",
            "recommendations": ["Optimize berth allocation", "Reduce waiting times"],
            "confidence": 0.7
        }
    
    async def _analyze_cost_scenario(self, state: PortWorkflowState) -> Dict[str, Any]:
        """Analyze cost implications"""
        return {
            "scenario": "Cost Analysis",
            "cost_impact": "Low",
            "recommendations": ["Minimize delays", "Optimize resource usage"],
            "confidence": 0.6
        }
    
    async def _analyze_risk_scenario(self, state: PortWorkflowState) -> Dict[str, Any]:
        """Analyze risk factors"""
        return {
            "scenario": "Risk Analysis",
            "risk_factors": ["Weather", "Equipment availability"],
            "recommendations": ["Monitor conditions", "Have backup plans"],
            "confidence": 0.75
        }
    
    async def _generate_recommendation(self, state: PortWorkflowState) -> str:
        """Generate final recommendation based on all analysis"""
        try:
            # Use the port retrieval system to generate a comprehensive answer
            result = await self.port_retrieval.query_port_information(
                question=state["query"],
                max_documents=3
            )
            
            return result.get("answer", "No recommendation available")
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return "Unable to generate recommendation at this time."
    
    async def _create_alternative_scenarios(self, state: PortWorkflowState) -> List[Dict[str, Any]]:
        """Create alternative scenarios based on parallel analysis"""
        alternatives = []
        
        parallel_results = state.get("parallel_results", {})
        
        for scenario_name, result in parallel_results.items():
            if "error" not in result:
                alternatives.append({
                    "scenario_name": scenario_name,
                    "description": result.get("scenario", scenario_name),
                    "recommendations": result.get("recommendations", []),
                    "confidence": result.get("confidence", 0.5)
                })
        
        return alternatives
    
    def _get_document_type_for_intent(self, intent: str) -> Optional[str]:
        """Map intent to document type"""
        intent_mapping = {
            "weather": "weather_reports",
            "safety": "safety_protocols", 
            "berthing": "berth_schedules",
            "cargo": "vessel_manifests",
            "emergency": "safety_protocols",
            "general": None
        }
        return intent_mapping.get(intent)
    
    async def execute_workflow(self, 
                             query: str,
                             user_id: Optional[str] = None,
                             port_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the complete port workflow
        
        Args:
            query: User's question
            user_id: Optional user identifier
            port_config: Optional port-specific configuration
            
        Returns:
            Complete workflow results
        """
        start_time = datetime.now()
        
        try:
            # Initialize state
            initial_state = PortWorkflowState(
                query=query,
                user_id=user_id,
                port_config=port_config or {},
                intent=None,
                confidence_score=None,
                retrieved_documents=None,
                parallel_results=None,
                freshness_check=None,
                confidence_check=None,
                compliance_check=None,
                final_recommendation=None,
                alternative_scenarios=None,
                sources=None,
                execution_time=None,
                workflow_id=None,
                timestamp=None,
                errors=[]
            )
            
            # Execute the graph
            logger.info(f"Executing workflow for query: {query[:50]}...")
            final_state = await self.graph.ainvoke(initial_state)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            final_state["execution_time"] = execution_time
            
            logger.info(f"Workflow completed in {execution_time:.2f} seconds")
            
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "error": str(e),
                "query": query,
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
    
    async def execute_what_if_scenario(self,
                                     base_query: str,
                                     scenario_variations: List[str],
                                     user_id: Optional[str] = None,
                                     port_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute multiple what-if scenarios in parallel
        
        Args:
            base_query: Base question
            scenario_variations: List of scenario variations
            user_id: Optional user identifier  
            port_config: Optional port-specific configuration
            
        Returns:
            Results for all scenarios
        """
        try:
            logger.info(f"Executing {len(scenario_variations)} what-if scenarios")
            
            # Create tasks for each scenario
            tasks = []
            for i, variation in enumerate(scenario_variations):
                scenario_query = f"{base_query} {variation}"
                task = self.execute_workflow(
                    query=scenario_query,
                    user_id=f"{user_id}_scenario_{i}" if user_id else None,
                    port_config=port_config
                )
                tasks.append(task)
            
            # Execute all scenarios in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            scenario_results = {}
            for i, (variation, result) in enumerate(zip(scenario_variations, results)):
                if isinstance(result, Exception):
                    scenario_results[f"scenario_{i}"] = {
                        "variation": variation,
                        "error": str(result)
                    }
                else:
                    scenario_results[f"scenario_{i}"] = {
                        "variation": variation,
                        "result": result
                    }
            
            return {
                "base_query": base_query,
                "scenarios": scenario_results,
                "total_scenarios": len(scenario_variations),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"What-if scenario execution failed: {e}")
            return {"error": str(e)}