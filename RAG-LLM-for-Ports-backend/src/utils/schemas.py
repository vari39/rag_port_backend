# src/utils/schemas.py
"""
Pydantic Models for AI Port Decision-Support System
Type-safe data models for API requests and responses
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class IntentType(str, Enum):
    """Enumeration of supported intent types"""
    WEATHER = "weather"
    SAFETY = "safety"
    BERTHING = "berthing"
    CARGO = "cargo"
    OPERATIONS = "operations"
    EQUIPMENT = "equipment"
    REGULATORY = "regulatory"
    EMERGENCY = "emergency"
    GENERAL = "general"


class DocumentType(str, Enum):
    """Enumeration of supported document types"""
    SOP = "sop"
    TOS_LOGS = "tos_logs"
    EDI_MESSAGES = "edi_messages"
    AIS_DATA = "ais_data"
    WEATHER_REPORTS = "weather_reports"
    BERTH_SCHEDULES = "berth_schedules"
    VESSEL_MANIFESTS = "vessel_manifests"
    SAFETY_PROTOCOLS = "safety_protocols"
    REGULATORY_DOCS = "regulatory_docs"
    MAINTENANCE_LOGS = "maintenance_logs"
    GENERAL = "general"


class SystemStatus(str, Enum):
    """System status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# Request Models
class QueryRequest(BaseModel):
    """Request model for simple RAG queries"""
    query: str = Field(..., description="User's question", min_length=1, max_length=1000)
    document_types: Optional[List[DocumentType]] = Field(None, description="Specific document types to search")
    max_documents: Optional[int] = Field(5, description="Maximum number of documents to retrieve", ge=1, le=20)
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


class WorkflowRequest(BaseModel):
    """Request model for LangGraph workflow queries"""
    query: str = Field(..., description="User's question", min_length=1, max_length=1000)
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    port_config: Optional[Dict[str, Any]] = Field(None, description="Port-specific configuration")
    enable_parallel_analysis: bool = Field(True, description="Enable parallel analysis branches")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


class WhatIfRequest(BaseModel):
    """Request model for what-if scenario analysis"""
    base_query: str = Field(..., description="Base question for scenarios", min_length=1, max_length=1000)
    scenarios: List[str] = Field(..., description="List of scenario variations", min_items=1, max_items=10)
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    port_config: Optional[Dict[str, Any]] = Field(None, description="Port-specific configuration")
    
    @validator('scenarios')
    def validate_scenarios(cls, v):
        if not all(scenario.strip() for scenario in v):
            raise ValueError('All scenarios must be non-empty')
        return [scenario.strip() for scenario in v]


class SearchRequest(BaseModel):
    """Request model for document similarity search"""
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    document_type: Optional[DocumentType] = Field(None, description="Specific document type to search")
    max_results: Optional[int] = Field(10, description="Maximum number of results", ge=1, le=50)
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


# Response Models
class SourceInfo(BaseModel):
    """Information about a document source"""
    source: str = Field(..., description="Source document name")
    document_type: str = Field(..., description="Type of document")
    page: Optional[Union[int, str]] = Field(None, description="Page number or section")
    relevance_score: Optional[Union[float, str]] = Field(None, description="Relevance score")
    timestamp: Optional[str] = Field(None, description="Document timestamp")


class QueryResponse(BaseModel):
    """Response model for simple RAG queries"""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceInfo] = Field(..., description="Source documents used")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    timestamp: str = Field(..., description="Response timestamp")
    processing_time: float = Field(..., description="Processing time in seconds", ge=0.0)


class AlternativeScenario(BaseModel):
    """Alternative scenario analysis result"""
    scenario_name: str = Field(..., description="Name of the scenario")
    description: str = Field(..., description="Scenario description")
    recommendations: List[str] = Field(..., description="Scenario-specific recommendations")
    confidence: float = Field(..., description="Confidence in scenario", ge=0.0, le=1.0)


class DecisionGates(BaseModel):
    """Decision gate results"""
    freshness: bool = Field(..., description="Freshness gate result")
    confidence: bool = Field(..., description="Confidence gate result")
    compliance: bool = Field(..., description="Compliance gate result")


class WorkflowResponse(BaseModel):
    """Response model for LangGraph workflow queries"""
    query: str = Field(..., description="Original query")
    intent: str = Field(..., description="Classified intent")
    confidence: float = Field(..., description="Intent confidence", ge=0.0, le=1.0)
    recommendation: str = Field(..., description="Final recommendation")
    alternative_scenarios: List[AlternativeScenario] = Field(..., description="Alternative scenarios")
    sources: List[SourceInfo] = Field(..., description="Source documents used")
    decision_gates: DecisionGates = Field(..., description="Decision gate results")
    execution_time: float = Field(..., description="Total execution time", ge=0.0)
    timestamp: str = Field(..., description="Response timestamp")
    workflow_id: str = Field(..., description="Unique workflow identifier")


class WhatIfResponse(BaseModel):
    """Response model for what-if scenario analysis"""
    base_query: str = Field(..., description="Base query")
    scenarios: Dict[str, Any] = Field(..., description="Results for each scenario")
    total_scenarios: int = Field(..., description="Total number of scenarios processed")
    timestamp: str = Field(..., description="Response timestamp")
    execution_time: float = Field(..., description="Total execution time", ge=0.0)


class DocumentInfo(BaseModel):
    """Information about a retrieved document"""
    content: str = Field(..., description="Document content preview")
    source: str = Field(..., description="Source document name")
    document_type: str = Field(..., description="Type of document")
    page: Optional[Union[int, str]] = Field(None, description="Page number or section")
    similarity_score: Optional[Union[float, str]] = Field(None, description="Similarity score")


class SearchResponse(BaseModel):
    """Response model for document similarity search"""
    query: str = Field(..., description="Search query")
    documents: List[DocumentInfo] = Field(..., description="Retrieved documents")
    total_found: int = Field(..., description="Total number of documents found")
    timestamp: str = Field(..., description="Response timestamp")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: SystemStatus = Field(..., description="System health status")
    timestamp: str = Field(..., description="Health check timestamp")
    components: Dict[str, bool] = Field(..., description="Component health status")
    version: str = Field(..., description="System version")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class SystemStatusResponse(BaseModel):
    """Comprehensive system status response"""
    status: SystemStatus = Field(..., description="Overall system status")
    components: Dict[str, bool] = Field(..., description="Component status")
    document_summary: Optional[Dict[str, Any]] = Field(None, description="Document summary")
    intent_statistics: Optional[Dict[str, Any]] = Field(None, description="Intent classification statistics")
    timestamp: str = Field(..., description="Status timestamp")
    version: str = Field(..., description="System version")


# Internal Models
class PortConfig(BaseModel):
    """Port-specific configuration"""
    port_name: str = Field(..., description="Name of the port")
    port_type: str = Field(..., description="Type of port (container, bulk, multi-purpose)")
    port_code: str = Field(..., description="Port code")
    timezone: str = Field("UTC", description="Port timezone")
    coordinates: Optional[Dict[str, float]] = Field(None, description="Port coordinates")
    operational_hours: str = Field("24/7", description="Operational hours")
    has_weather_station: bool = Field(False, description="Has weather station")
    has_ais_system: bool = Field(True, description="Has AIS system")
    custom_rules: Optional[Dict[str, Any]] = Field(None, description="Custom operational rules")


class WorkflowState(BaseModel):
    """Internal workflow state model"""
    query: str = Field(..., description="User query")
    user_id: Optional[str] = Field(None, description="User identifier")
    port_config: Optional[PortConfig] = Field(None, description="Port configuration")
    intent: Optional[str] = Field(None, description="Classified intent")
    confidence_score: Optional[float] = Field(None, description="Intent confidence")
    retrieved_documents: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved documents")
    parallel_results: Optional[Dict[str, Any]] = Field(None, description="Parallel analysis results")
    freshness_check: Optional[bool] = Field(None, description="Freshness gate result")
    confidence_check: Optional[bool] = Field(None, description="Confidence gate result")
    compliance_check: Optional[bool] = Field(None, description="Compliance gate result")
    final_recommendation: Optional[str] = Field(None, description="Final recommendation")
    alternative_scenarios: Optional[List[AlternativeScenario]] = Field(None, description="Alternative scenarios")
    sources: Optional[List[SourceInfo]] = Field(None, description="Source information")
    execution_time: Optional[float] = Field(None, description="Execution time")
    workflow_id: Optional[str] = Field(None, description="Workflow identifier")
    timestamp: Optional[str] = Field(None, description="Timestamp")
    errors: Optional[List[str]] = Field(None, description="Error messages")


class DecisionResult(BaseModel):
    """Decision gate result"""
    passed: bool = Field(..., description="Whether the gate passed")
    score: float = Field(..., description="Gate score", ge=0.0, le=1.0)
    reason: str = Field(..., description="Reason for the result")
    details: Dict[str, Any] = Field(..., description="Additional details")


class IntentResult(BaseModel):
    """Intent classification result"""
    intent: str = Field(..., description="Classified intent")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    sub_intent: Optional[str] = Field(None, description="Sub-intent classification")
    keywords: List[str] = Field(..., description="Matched keywords")
    routing_path: str = Field(..., description="Routing path")
    priority: int = Field(..., description="Priority level", ge=0, le=4)
