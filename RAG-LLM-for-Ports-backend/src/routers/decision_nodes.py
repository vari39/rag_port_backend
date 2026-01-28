# src/routers/decision_nodes.py
"""
Decision Nodes for AI Port Decision-Support System
Implements freshness, confidence, and compliance gates for information validation

Validates that the information we're using is good quality

1) Freshness Gate: Checks if information is recent enough (24-hour threshold)
2) Confidence Gate: Checks if we're confident in the retrieved documents (0.7 threshold)
3) Compliance Gate: Checks if recommendations follow safety/regulatory rules

Prevents bad decisions based on outdated or unreliable information
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DecisionResult:
    """Result of a decision gate check"""
    passed: bool
    score: float
    reason: str
    details: Dict[str, Any]


class DecisionNodes:
    """
    Decision gates for information validation in port operations.
    Implements freshness, confidence, and compliance checks.
    """
    
    def __init__(self, 
                 freshness_threshold_hours: int = 24,
                 confidence_threshold: float = 0.7,
                 compliance_strict_mode: bool = True):
        """
        Initialize decision nodes with configurable thresholds.
        
        Args:
            freshness_threshold_hours: Maximum age of information in hours
            confidence_threshold: Minimum confidence score (0.0-1.0)
            compliance_strict_mode: Whether to enforce strict compliance checks
        """
        self.freshness_threshold_hours = freshness_threshold_hours
        self.confidence_threshold = confidence_threshold
        self.compliance_strict_mode = compliance_strict_mode
        
        # Compliance rules and patterns
        self.compliance_patterns = self._initialize_compliance_patterns()
        
        # Document type priorities for confidence scoring
        self.document_priorities = {
            "safety_protocols": 1.0,
            "regulatory_docs": 0.9,
            "sop": 0.8,
            "weather_reports": 0.7,
            "berth_schedules": 0.6,
            "tos_logs": 0.5,
            "edi_messages": 0.4,
            "general": 0.3
        }
        
        logger.info("Decision nodes initialized with configurable thresholds")
    
    def _initialize_compliance_patterns(self) -> Dict[str, List[str]]:
        """Initialize compliance patterns for different document types"""
        return {
            "safety_keywords": [
                "safety", "emergency", "evacuation", "fire", "hazard", 
                "danger", "caution", "warning", "protocol", "procedure"
            ],
            "regulatory_keywords": [
                "regulation", "compliance", "standard", "requirement",
                "mandatory", "prohibited", "authorized", "certified"
            ],
            "operational_keywords": [
                "operation", "procedure", "process", "workflow", "step",
                "instruction", "guideline", "policy", "rule"
            ],
            "time_sensitive_keywords": [
                "immediate", "urgent", "critical", "asap", "emergency",
                "time-sensitive", "deadline", "expires"
            ]
        }
    
    async def check_freshness(self, 
                            documents: List[Dict[str, Any]], 
                            query: str,
                            port_config: Optional[Dict[str, Any]] = None) -> DecisionResult:
        """
        Check the freshness of retrieved information.
        
        Args:
            documents: List of retrieved documents
            query: User query
            port_config: Optional port configuration
            
        Returns:
            DecisionResult with freshness check results
        """
        try:
            logger.info("Checking information freshness")
            
            if not documents:
                return DecisionResult(
                    passed=False,
                    score=0.0,
                    reason="No documents available for freshness check",
                    details={"documents_count": 0}
                )
            
            # Calculate freshness scores for each document
            freshness_scores = []
            fresh_documents = 0
            
            for doc in documents:
                freshness_score = self._calculate_document_freshness(doc, query)
                freshness_scores.append(freshness_score)
                
                if freshness_score >= 0.7:  # Consider fresh if score >= 0.7
                    fresh_documents += 1
            
            # Overall freshness score
            overall_freshness = sum(freshness_scores) / len(freshness_scores)
            
            # Determine if check passes
            freshness_threshold = self._get_freshness_threshold(query, port_config)
            passed = overall_freshness >= freshness_threshold
            
            reason = f"Freshness score: {overall_freshness:.2f} (threshold: {freshness_threshold:.2f})"
            if not passed:
                reason += f" - {fresh_documents}/{len(documents)} documents are fresh"
            
            return DecisionResult(
                passed=passed,
                score=overall_freshness,
                reason=reason,
                details={
                    "documents_count": len(documents),
                    "fresh_documents": fresh_documents,
                    "individual_scores": freshness_scores,
                    "threshold": freshness_threshold
                }
            )
            
        except Exception as e:
            logger.error(f"Error in freshness check: {e}")
            return DecisionResult(
                passed=False,
                score=0.0,
                reason=f"Freshness check encountered an error: {str(e)}",
                details={"error": str(e), "error_type": "freshness_check_failure"}
            )
    
    async def check_confidence(self, 
                             documents: List[Dict[str, Any]], 
                             query: str,
                             confidence_threshold: Optional[float] = None) -> DecisionResult:
        """
        Check confidence in retrieved information.
        
        Args:
            documents: List of retrieved documents
            query: User query
            confidence_threshold: Optional custom confidence threshold
            
        Returns:
            DecisionResult with confidence check results
        """
        try:
            logger.info("Checking information confidence")
            
            if not documents:
                return DecisionResult(
                    passed=False,
                    score=0.0,
                    reason="No documents available for confidence check",
                    details={"documents_count": 0}
                )
            
            # Use provided threshold or default
            threshold = confidence_threshold or self.confidence_threshold
            
            # Calculate confidence scores for each document
            confidence_scores = []
            high_confidence_docs = 0
            
            for doc in documents:
                confidence_score = self._calculate_document_confidence(doc, query)
                confidence_scores.append(confidence_score)
                
                if confidence_score >= threshold:
                    high_confidence_docs += 1
            
            # Overall confidence score
            overall_confidence = sum(confidence_scores) / len(confidence_scores)
            
            # Determine if check passes
            passed = overall_confidence >= threshold
            
            reason = f"Confidence score: {overall_confidence:.2f} (threshold: {threshold:.2f})"
            if not passed:
                reason += f" - {high_confidence_docs}/{len(documents)} documents meet confidence threshold"
            
            return DecisionResult(
                passed=passed,
                score=overall_confidence,
                reason=reason,
                details={
                    "documents_count": len(documents),
                    "high_confidence_docs": high_confidence_docs,
                    "individual_scores": confidence_scores,
                    "threshold": threshold
                }
            )
            
        except Exception as e:
            logger.error(f"Error in confidence check: {e}")
            return DecisionResult(
                passed=False,
                score=0.0,
                reason=f"Confidence check encountered an error: {str(e)}",
                details={"error": str(e), "error_type": "confidence_check_failure"}
            )
    
    async def check_compliance(self, 
                             documents: List[Dict[str, Any]], 
                             query: str,
                             intent: str = "general",
                             port_config: Optional[Dict[str, Any]] = None) -> DecisionResult:
        """
        Check compliance with safety and regulatory requirements.
        
        Args:
            documents: List of retrieved documents
            query: User query
            intent: Query intent classification
            port_config: Optional port configuration
            
        Returns:
            DecisionResult with compliance check results
        """
        try:
            logger.info("Checking compliance requirements")
            
            if not documents:
                return DecisionResult(
                    passed=True,  # No documents to check
                    score=1.0,
                    reason="No documents to check for compliance",
                    details={"documents_count": 0}
                )
            
            # Calculate compliance scores for each document
            compliance_scores = []
            compliant_docs = 0
            
            for doc in documents:
                compliance_score = self._calculate_document_compliance(doc, query, intent)
                compliance_scores.append(compliance_score)
                
                if compliance_score >= 0.8:  # Consider compliant if score >= 0.8
                    compliant_docs += 1
            
            # Overall compliance score
            overall_compliance = sum(compliance_scores) / len(compliance_scores)
            
            # Determine if check passes
            compliance_threshold = self._get_compliance_threshold(intent, port_config)
            passed = overall_compliance >= compliance_threshold
            
            reason = f"Compliance score: {overall_compliance:.2f} (threshold: {compliance_threshold:.2f})"
            if not passed:
                reason += f" - {compliant_docs}/{len(documents)} documents meet compliance standards"
            
            return DecisionResult(
                passed=passed,
                score=overall_compliance,
                reason=reason,
                details={
                    "documents_count": len(documents),
                    "compliant_docs": compliant_docs,
                    "individual_scores": compliance_scores,
                    "threshold": compliance_threshold,
                    "intent": intent
                }
            )
            
        except Exception as e:
            logger.error(f"Error in compliance check: {e}")
            return DecisionResult(
                passed=False,
                score=0.0,
                reason=f"Compliance check encountered an error: {str(e)}",
                details={"error": str(e), "error_type": "compliance_check_failure"}
            )
    
    def _calculate_document_freshness(self, doc: Dict[str, Any], query: str) -> float:
        """Calculate freshness score for a single document"""
        try:
            # Check if document has timestamp metadata
            doc_timestamp = doc.get("timestamp") or doc.get("processed_at")
            
            if doc_timestamp:
                try:
                    # Parse timestamp
                    if isinstance(doc_timestamp, str):
                        doc_time = datetime.fromisoformat(doc_timestamp.replace('Z', '+00:00'))
                    else:
                        doc_time = doc_timestamp
                    
                    # Calculate age
                    age_hours = (datetime.now() - doc_time).total_seconds() / 3600
                    
                    # Calculate freshness score (exponential decay)
                    freshness_score = max(0.0, 1.0 - (age_hours / self.freshness_threshold_hours))
                    
                    return min(1.0, freshness_score)
                    
                except Exception as e:
                    logger.warning(f"Error parsing document timestamp: {e}")
            
            # Check for time-sensitive keywords in query
            time_sensitive_keywords = self.compliance_patterns["time_sensitive_keywords"]
            query_lower = query.lower()
            
            if any(keyword in query_lower for keyword in time_sensitive_keywords):
                # For time-sensitive queries, require fresher information
                return 0.3  # Lower score for unknown age with time-sensitive query
            
            # Default score for documents without timestamp
            return 0.6
            
        except Exception as e:
            logger.warning(f"Error calculating document freshness: {e}")
            return 0.5
    
    def _calculate_document_confidence(self, doc: Dict[str, Any], query: str) -> float:
        """Calculate confidence score for a single document"""
        try:
            confidence_score = 0.5  # Base score
            
            # Document type priority
            doc_type = doc.get("document_type", "general")
            type_priority = self.document_priorities.get(doc_type, 0.3)
            confidence_score += type_priority * 0.3
            
            # Similarity score (if available)
            similarity_score = doc.get("similarity_score", 0.5)
            if isinstance(similarity_score, (int, float)):
                confidence_score += similarity_score * 0.4
            
            # Source reliability
            source = doc.get("source", "").lower()
            if any(reliable_source in source for reliable_source in ["official", "regulation", "sop", "protocol"]):
                confidence_score += 0.2
            
            # Content quality indicators
            content = doc.get("content", "")
            if len(content) > 100:  # Substantial content
                confidence_score += 0.1
            
            return min(1.0, confidence_score)
            
        except Exception as e:
            logger.warning(f"Error calculating document confidence: {e}")
            return 0.5
    
    def _calculate_document_compliance(self, doc: Dict[str, Any], query: str, intent: str) -> float:
        """Calculate compliance score for a single document"""
        try:
            compliance_score = 0.5  # Base score
            
            # Document type compliance
            doc_type = doc.get("document_type", "general")
            if doc_type in ["safety_protocols", "regulatory_docs", "sop"]:
                compliance_score += 0.3
            
            # Content compliance keywords
            content = doc.get("content", "").lower()
            query_lower = query.lower()
            
            # Check for safety keywords
            safety_keywords = self.compliance_patterns["safety_keywords"]
            safety_matches = sum(1 for keyword in safety_keywords if keyword in content)
            compliance_score += min(0.2, safety_matches * 0.05)
            
            # Check for regulatory keywords
            regulatory_keywords = self.compliance_patterns["regulatory_keywords"]
            regulatory_matches = sum(1 for keyword in regulatory_keywords if keyword in content)
            compliance_score += min(0.2, regulatory_matches * 0.05)
            
            # Intent-specific compliance
            if intent in ["safety", "emergency"] and any(keyword in content for keyword in safety_keywords):
                compliance_score += 0.2
            
            # Query compliance keywords
            if any(keyword in query_lower for keyword in safety_keywords + regulatory_keywords):
                compliance_score += 0.1
            
            return min(1.0, compliance_score)
            
        except Exception as e:
            logger.warning(f"Error calculating document compliance: {e}")
            return 0.5
    
    def _get_freshness_threshold(self, query: str, port_config: Optional[Dict[str, Any]]) -> float:
        """Get freshness threshold based on query and port configuration"""
        query_lower = query.lower()
        
        # Time-sensitive queries require fresher information
        time_sensitive_keywords = self.compliance_patterns["time_sensitive_keywords"]
        if any(keyword in query_lower for keyword in time_sensitive_keywords):
            return 0.8  # Higher threshold for time-sensitive queries
        
        # Weather queries need recent information
        if any(keyword in query_lower for keyword in ["weather", "forecast", "tide", "wind"]):
            return 0.7
        
        # Default threshold
        return 0.6
    
    def _get_compliance_threshold(self, intent: str, port_config: Optional[Dict[str, Any]]) -> float:
        """Get compliance threshold based on intent and port configuration"""
        
        # Safety-critical intents require higher compliance
        if intent in ["safety", "emergency"]:
            return 0.9 if self.compliance_strict_mode else 0.7
        
        # Operational intents need moderate compliance
        if intent in ["berthing", "cargo", "operations"]:
            return 0.8 if self.compliance_strict_mode else 0.6
        
        # General queries have lower compliance requirements
        return 0.6 if self.compliance_strict_mode else 0.4
    
    def update_thresholds(self, 
                         freshness_threshold_hours: Optional[int] = None,
                         confidence_threshold: Optional[float] = None,
                         compliance_strict_mode: Optional[bool] = None):
        """
        Update decision thresholds dynamically.
        
        Args:
            freshness_threshold_hours: New freshness threshold in hours
            confidence_threshold: New confidence threshold (0.0-1.0)
            compliance_strict_mode: New compliance strict mode setting
        """
        if freshness_threshold_hours is not None:
            self.freshness_threshold_hours = freshness_threshold_hours
            logger.info(f"Updated freshness threshold to {freshness_threshold_hours} hours")
        
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
            logger.info(f"Updated confidence threshold to {confidence_threshold}")
        
        if compliance_strict_mode is not None:
            self.compliance_strict_mode = compliance_strict_mode
            logger.info(f"Updated compliance strict mode to {compliance_strict_mode}")
    
    def get_threshold_status(self) -> Dict[str, Any]:
        """Get current threshold settings"""
        return {
            "freshness_threshold_hours": self.freshness_threshold_hours,
            "confidence_threshold": self.confidence_threshold,
            "compliance_strict_mode": self.compliance_strict_mode,
            "document_priorities": self.document_priorities
        }
