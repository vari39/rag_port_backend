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
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from ..routers.intent_router import IntentRouter
from ..routers.decision_nodes import DecisionNodes
from ..port_information_retrieval import PortInformationRetrieval

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reduce_decision_gates(left: Optional[bool], right: Optional[bool]) -> Optional[bool]:
    """Reducer for decision gate results"""
    if right is not None:
        return right
    return left

def reduce_errors(left: Optional[List[str]], right: Optional[List[str]]) -> Optional[List[str]]:
    """Reducer for error lists - combine them"""
    if left is None:
        left = []
    if right is None:
        right = []
    return left + right

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
    
    # Decision gates (using Annotated to handle parallel updates)
    freshness_check: Annotated[Optional[bool], reduce_decision_gates]
    confidence_check: Annotated[Optional[bool], reduce_decision_gates]
    compliance_check: Annotated[Optional[bool], reduce_decision_gates]
    
    # Output
    final_recommendation: Optional[str]
    alternative_scenarios: Optional[List[Dict[str, Any]]]
    sources: Optional[List[Dict[str, Any]]]
    execution_time: Optional[float]
    
    # Metadata
    workflow_id: Optional[str]
    timestamp: Optional[str]
    errors: Annotated[Optional[List[str]], reduce_errors]


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
        
        # Define the workflow edges - START entry point
        workflow.add_edge(START, "analyze_intent")
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
    
    async def _analyze_intent(self, state: PortWorkflowState) -> Dict[str, Any]:
        """Analyze user intent and classify the query"""
        try:
            logger.info(f"Analyzing intent for query: {state['query'][:50]}...")
            
            intent_result = await self.intent_router.classify_query(
                query=state["query"],
                port_config=state.get("port_config", {})
            )
            
            intent = intent_result["intent"]
            confidence = intent_result["confidence"]
            
            logger.info(f"Intent classified as: {intent} (confidence: {confidence})")
            
            # Return only the fields we're updating (partial state)
            return {
                "intent": intent,
                "confidence_score": confidence
            }
            
        except Exception as e:
            logger.error(f"Error in intent analysis: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return defaults on error
            return {
                "intent": "general",
                "confidence_score": 0.5,
                "errors": (state.get("errors", []) + [f"Intent analysis failed: {str(e)}"])
            }
    
    async def _retrieve_documents(self, state: PortWorkflowState) -> Dict[str, Any]:
        """Retrieve relevant documents based on intent"""
        try:
            logger.info(f"Retrieving documents for intent: {state.get('intent', 'general')}")
            
            # Get documents based on intent
            documents = await self.port_retrieval.search_similar_documents(
                query=state["query"],
                document_type=self._get_document_type_for_intent(state.get("intent", "general")),
                k=5
            )
            
            logger.info(f"Retrieved {len(documents)} documents")
            
            # Return only the fields we're updating (partial state)
            return {"retrieved_documents": documents}
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty documents on error
            return {
                "retrieved_documents": [],
                "errors": (state.get("errors", []) + [f"Document retrieval failed: {str(e)}"])
            }
    
    async def _parallel_analysis(self, state: PortWorkflowState) -> Dict[str, Any]:
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
            
            logger.info(f"Completed parallel analysis with {len(parallel_results)} branches")
            
            # Return only the fields we're updating (partial state)
            return {"parallel_results": parallel_results}
            
        except Exception as e:
            logger.error(f"Error in parallel analysis: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty parallel_results on error
            return {
                "parallel_results": {},
                "errors": (state.get("errors", []) + [f"Parallel analysis failed: {str(e)}"])
            }
    
    async def _freshness_gate(self, state: PortWorkflowState) -> Dict[str, Any]:
        """Check information freshness"""
        try:
            logger.info("Checking information freshness")
            
            freshness_result = await self.decision_nodes.check_freshness(
                documents=state.get("retrieved_documents", []),
                query=state["query"],
                port_config=state.get("port_config", {})
            )
            
            # DecisionResult is a dataclass, use dot notation
            passed = freshness_result.passed
            
            if not passed:
                logger.warning(f"Freshness check failed: {freshness_result.reason}")
            
            # Only return fields we're updating
            return {"freshness_check": passed}
            
        except Exception as e:
            logger.error(f"Error in freshness gate: {e}")
            errors = state.get("errors", []) + [f"Freshness check failed: {str(e)}"]
            return {"freshness_check": False, "errors": errors}
    
    async def _confidence_gate(self, state: PortWorkflowState) -> Dict[str, Any]:
        """Check confidence in retrieved information"""
        try:
            logger.info("Checking information confidence")
            
            confidence_result = await self.decision_nodes.check_confidence(
                documents=state.get("retrieved_documents", []),
                query=state["query"],
                confidence_threshold=state.get("confidence_score", 0.7)
            )
            
            # DecisionResult is a dataclass, use dot notation
            passed = confidence_result.passed
            
            if not passed:
                logger.warning(f"Confidence check failed: {confidence_result.reason}")
            
            # Only return fields we're updating
            return {"confidence_check": passed}
            
        except Exception as e:
            logger.error(f"Error in confidence gate: {e}")
            errors = state.get("errors", []) + [f"Confidence check failed: {str(e)}"]
            return {"confidence_check": False, "errors": errors}
    
    async def _compliance_gate(self, state: PortWorkflowState) -> Dict[str, Any]:
        """Check compliance with safety and regulatory requirements"""
        try:
            logger.info("Checking compliance requirements")
            
            compliance_result = await self.decision_nodes.check_compliance(
                documents=state.get("retrieved_documents", []),
                query=state["query"],
                intent=state.get("intent", "general"),
                port_config=state.get("port_config", {})
            )
            
            # DecisionResult is a dataclass, use dot notation
            passed = compliance_result.passed
            
            if not passed:
                logger.warning(f"Compliance check failed: {compliance_result.reason}")
            
            # Only return fields we're updating
            return {"compliance_check": passed}
            
        except Exception as e:
            logger.error(f"Error in compliance gate: {e}")
            errors = state.get("errors", []) + [f"Compliance check failed: {str(e)}"]
            return {"compliance_check": False, "errors": errors}
    
    async def _synthesize_results(self, state: PortWorkflowState) -> Dict[str, Any]:
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
            
            # Generate final recommendation AND get sources from the query
            recommendation_result = await self.port_retrieval.query_port_information(
                question=state["query"],
                max_documents=3
            )
            
            recommendation = recommendation_result.get("answer", "")
            if not recommendation or recommendation.strip() == "":
                raise ValueError("Recommendation generation returned empty result")
            
            # Get sources from the recommendation query result
            sources = recommendation_result.get("sources", [])
            
            logger.info(f"Results synthesized successfully with {len(sources)} sources")
            
            # Return recommendation and sources
            return {
                "final_recommendation": recommendation,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error synthesizing results: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            error_msg = f"Unable to generate recommendation due to processing errors: {str(e)}"
            raise ValueError(error_msg)
    
    async def _generate_alternatives(self, state: PortWorkflowState) -> Dict[str, Any]:
        """Generate alternative scenarios based on parallel analysis"""
        try:
            logger.info("Generating alternative scenarios")
            
            alternatives_result = await self._create_alternative_scenarios(state)
            alternatives = alternatives_result.get("alternative_scenarios", [])
            
            # Use sources from state (already set by _synthesize_results)
            # Don't overwrite them - they're already correct
            sources = state.get("sources", [])
            
            # Add execution metadata
            timestamp = datetime.now().isoformat()
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Generated {len(alternatives)} alternative scenarios")
            
            # Only return fields we're updating (don't overwrite sources if they exist)
            result = {
                "alternative_scenarios": alternatives,
                "timestamp": timestamp,
                "workflow_id": workflow_id
            }
            
            # Only add sources if they weren't already set
            if not sources and state.get("retrieved_documents"):
                # Fallback: extract from retrieved_documents if sources not set
                sources = []
                for doc in state.get("retrieved_documents", []):
                    sources.append({
                        "source": doc.get("source", "Unknown"),
                        "document_type": doc.get("document_type", "Unknown"),
                        "relevance_score": doc.get("similarity_score", "N/A")
                    })
                result["sources"] = sources
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating alternatives: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty alternatives but don't fail completely
            return {
                "alternative_scenarios": [],
                "sources": [],
                "timestamp": datetime.now().isoformat(),
                "workflow_id": f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
    
    # Helper methods for parallel analysis
    async def _analyze_safety_scenario(self, state: PortWorkflowState) -> Dict[str, Any]:
        """Analyze safety implications using actual database queries"""
        try:
            query = f"{state['query']} safety protocols emergency procedures"
            result = await self.port_retrieval.query_port_information(
                question=query,
                max_documents=2
            )
            return {
                "scenario": "Safety Analysis",
                "analysis": result.get("answer", ""),
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.0),
                "risk_level": "Assessed from database"
            }
        except Exception as e:
            logger.error(f"Safety analysis failed: {e}")
            raise ValueError(f"Safety analysis could not be completed: {str(e)}")
    
    async def _analyze_efficiency_scenario(self, state: PortWorkflowState) -> Dict[str, Any]:
        """Analyze operational efficiency using actual database queries"""
        try:
            query = f"{state['query']} efficiency optimization productivity"
            result = await self.port_retrieval.query_port_information(
                question=query,
                max_documents=2
            )
            return {
                "scenario": "Efficiency Analysis",
                "analysis": result.get("answer", ""),
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.0),
                "optimization_potential": "Assessed from database"
            }
        except Exception as e:
            logger.error(f"Efficiency analysis failed: {e}")
            raise ValueError(f"Efficiency analysis could not be completed: {str(e)}")
    
    async def _analyze_cost_scenario(self, state: PortWorkflowState) -> Dict[str, Any]:
        """Analyze cost implications using actual database queries"""
        try:
            query = f"{state['query']} cost expenses budget financial"
            result = await self.port_retrieval.query_port_information(
                question=query,
                max_documents=2
            )
            return {
                "scenario": "Cost Analysis",
                "analysis": result.get("answer", ""),
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.0),
                "cost_impact": "Assessed from database"
            }
        except Exception as e:
            logger.error(f"Cost analysis failed: {e}")
            raise ValueError(f"Cost analysis could not be completed: {str(e)}")
    
    async def _analyze_risk_scenario(self, state: PortWorkflowState) -> Dict[str, Any]:
        """Analyze risk factors using actual database queries"""
        try:
            query = f"{state['query']} risk factors hazards threats"
            result = await self.port_retrieval.query_port_information(
                question=query,
                max_documents=2
            )
            return {
                "scenario": "Risk Analysis",
                "analysis": result.get("answer", ""),
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.0),
                "risk_factors": "Assessed from database"
            }
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            raise ValueError(f"Risk analysis could not be completed: {str(e)}")
    
    async def _generate_recommendation(self, state: PortWorkflowState) -> str:
        """Generate final recommendation based on all analysis"""
        # Note: This method is now called from _synthesize_results which handles the query
        # This is kept for backward compatibility but shouldn't be called directly
        try:
            # Use the port retrieval system to generate a comprehensive answer
            result = await self.port_retrieval.query_port_information(
                question=state["query"],
                max_documents=3
            )
            
            answer = result.get("answer")
            if not answer or answer.strip() == "":
                raise ValueError("Port retrieval system returned an empty answer")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            raise ValueError(f"Unable to generate recommendation: {str(e)}")
    
    async def _create_alternative_scenarios(self, state: PortWorkflowState) -> Dict[str, Any]:
        """Create alternative scenarios based on parallel analysis"""
        alternatives = []
        
        parallel_results = state.get("parallel_results", {})
        
        if not parallel_results:
            logger.warning("No parallel results available for alternative scenarios")
            return {"alternative_scenarios": []}
        
        for scenario_name, result in parallel_results.items():
            if isinstance(result, dict) and "error" not in result and result:
                # Extract analysis from database query results
                analysis = result.get("analysis", "")
                sources = result.get("sources", [])
                confidence = result.get("confidence", 0.0)
                
                if analysis and analysis.strip():
                    # Schema requires recommendations as List[str], not analysis string
                    # Split analysis into sentences or use as single recommendation
                    # For now, use the analysis as a single recommendation
                    recommendations = [analysis.strip()]
                    
                    alternatives.append({
                        "scenario_name": scenario_name,
                        "description": result.get("scenario", scenario_name.replace("_", " ").title()),
                        "recommendations": recommendations,  # Required by schema: List[str]
                        "confidence": confidence
                    })
        
        if not alternatives:
            logger.warning("No valid alternative scenarios could be generated from database queries")
            return {"alternative_scenarios": []}
        
        # Return as dict for partial state update
        return {"alternative_scenarios": alternatives}
    
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
            
            # If checkpoints are enabled, provide thread_id configuration
            if self.enable_checkpoints:
                thread_id = user_id or f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                config = {"configurable": {"thread_id": thread_id}}
                final_state = await self.graph.ainvoke(initial_state, config=config)
            else:
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
                # Ensure each scenario has a unique user_id for thread_id
                scenario_user_id = f"{user_id}_scenario_{i}" if user_id else f"scenario_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                task = self.execute_workflow(
                    query=scenario_query,
                    user_id=scenario_user_id,
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
