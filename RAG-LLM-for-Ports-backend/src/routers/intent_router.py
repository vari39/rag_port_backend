# src/routers/intent_router.py
"""
Intent Router for AI Port Decision-Support System
Rule-based question classification and routing to appropriate analysis pipelines

What it does:

Understands what type of question you're asking

1)Analyzes your question for keywords and patterns
2)Classifies it into categories: weather, safety, berthing, cargo, etc.
3)Assigns priority levels (emergency = highest, general = lowest)
4)Routes the question to the appropriate analysis pipeline

Key features:
1) 9 different intent types
2) Rule-based classification (fast and reliable)
3) Port-specific adjustments
4) Custom pattern support

"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    """Result of intent classification"""
    intent: str
    confidence: float
    sub_intent: Optional[str]
    keywords: List[str]
    routing_path: str
    priority: int


class IntentRouter:
    """
    Rule-based intent router for port operations queries.
    Classifies questions and routes them to appropriate analysis pipelines.
    """
    
    def __init__(self):
        """Initialize the intent router with classification rules"""
        self.intent_patterns = self._initialize_intent_patterns()
        self.priority_rules = self._initialize_priority_rules()
        self.routing_paths = self._initialize_routing_paths()
        
        logger.info("Intent router initialized with classification rules")
    
    def _initialize_intent_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize intent classification patterns"""
        return {
            "weather": {
                "keywords": [
                    "weather", "forecast", "wind", "storm", "rain", "fog", "visibility",
                    "tide", "current", "wave", "hurricane", "typhoon", "gale", "squall",
                    "meteorological", "conditions", "climate"
                ],
                "patterns": [
                    r"weather.*impact", r"wind.*speed", r"storm.*warning",
                    r"tide.*table", r"forecast.*port", r"visibility.*poor"
                ],
                "priority": 1
            },
            "safety": {
                "keywords": [
                    "safety", "emergency", "evacuation", "fire", "hazard", "danger",
                    "caution", "warning", "protocol", "procedure", "incident", "accident",
                    "rescue", "medical", "first aid", "emergency response"
                ],
                "patterns": [
                    r"safety.*protocol", r"emergency.*procedure", r"fire.*safety",
                    r"hazard.*assessment", r"incident.*response", r"evacuation.*plan"
                ],
                "priority": 1
            },
            "berthing": {
                "keywords": [
                    "berth", "berthing", "dock", "pier", "quay", "mooring", "anchoring",
                    "allocation", "assignment", "schedule", "conflict", "availability",
                    "vessel.*position", "arrival", "departure", "turnaround"
                ],
                "patterns": [
                    r"berth.*conflict", r"vessel.*berth", r"berth.*allocation",
                    r"dock.*schedule", r"mooring.*procedure", r"arrival.*time"
                ],
                "priority": 2
            },
            "cargo": {
                "keywords": [
                    "cargo", "container", "bulk", "liquid", "loading", "unloading",
                    "handling", "storage", "manifest", "cargo.*plan", "stowage",
                    "crane", "equipment", "operation", "sequence"
                ],
                "patterns": [
                    r"cargo.*handling", r"container.*operation", r"loading.*sequence",
                    r"cargo.*plan", r"bulk.*cargo", r"liquid.*cargo"
                ],
                "priority": 2
            },
            "operations": {
                "keywords": [
                    "operation", "operational", "procedure", "process", "workflow",
                    "routine", "standard", "normal", "daily", "shift", "crew",
                    "personnel", "staff", "coordination", "management"
                ],
                "patterns": [
                    r"operational.*procedure", r"standard.*operation", r"daily.*routine",
                    r"shift.*handover", r"crew.*coordination", r"operation.*plan"
                ],
                "priority": 3
            },
            "equipment": {
                "keywords": [
                    "equipment", "crane", "conveyor", "pump", "generator", "machinery",
                    "maintenance", "repair", "breakdown", "fault", "malfunction",
                    "inspection", "testing", "calibration", "service"
                ],
                "patterns": [
                    r"equipment.*maintenance", r"crane.*operation", r"equipment.*failure",
                    r"machinery.*repair", r"equipment.*inspection", r"breakdown.*procedure"
                ],
                "priority": 3
            },
            "regulatory": {
                "keywords": [
                    "regulation", "compliance", "legal", "authority", "permit", "license",
                    "customs", "immigration", "quarantine", "inspection", "clearance",
                    "documentation", "certificate", "standard", "requirement"
                ],
                "patterns": [
                    r"regulatory.*requirement", r"compliance.*check", r"customs.*clearance",
                    r"permit.*application", r"legal.*requirement", r"authority.*approval"
                ],
                "priority": 2
            },
            "emergency": {
                "keywords": [
                    "emergency", "urgent", "critical", "immediate", "asap", "crisis",
                    "disaster", "accident", "incident", "alert", "alarm", "distress",
                    "mayday", "pan-pan", "emergency.*response"
                ],
                "patterns": [
                    r"emergency.*situation", r"urgent.*action", r"critical.*incident",
                    r"emergency.*response", r"distress.*call", r"mayday.*signal"
                ],
                "priority": 0  # Highest priority
            },
            "general": {
                "keywords": [
                    "what", "how", "when", "where", "why", "explain", "describe",
                    "information", "about", "tell", "help", "question", "query"
                ],
                "patterns": [
                    r"what.*is", r"how.*does", r"when.*should", r"where.*can",
                    r"explain.*to", r"tell.*me.*about"
                ],
                "priority": 4
            }
        }
    
    def _initialize_priority_rules(self) -> Dict[str, int]:
        """Initialize priority rules for intent classification"""
        return {
            "emergency": 0,    # Highest priority
            "safety": 1,       # High priority
            "weather": 1,      # High priority
            "berthing": 2,     # Medium-high priority
            "cargo": 2,        # Medium-high priority
            "regulatory": 2,   # Medium-high priority
            "operations": 3,   # Medium priority
            "equipment": 3,    # Medium priority
            "general": 4       # Lowest priority
        }
    
    def _initialize_routing_paths(self) -> Dict[str, str]:
        """Initialize routing paths for different intents"""
        return {
            "emergency": "emergency_response_pipeline",
            "safety": "safety_analysis_pipeline",
            "weather": "weather_impact_pipeline",
            "berthing": "berth_optimization_pipeline",
            "cargo": "cargo_handling_pipeline",
            "operations": "operational_analysis_pipeline",
            "equipment": "equipment_maintenance_pipeline",
            "regulatory": "compliance_check_pipeline",
            "general": "general_information_pipeline"
        }
    
    async def classify_query(self, 
                           query: str, 
                           port_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Classify a user query and determine routing.
        
        Args:
            query: User's question
            port_config: Optional port-specific configuration
            
        Returns:
            Intent classification result
        """
        try:
            logger.info(f"Classifying query: {query[:50]}...")
            
            # Clean and normalize query
            normalized_query = self._normalize_query(query)
            
            # Find matching intents
            intent_matches = self._find_intent_matches(normalized_query)
            
            # Select best intent
            best_intent = self._select_best_intent(intent_matches, normalized_query)
            
            # Determine sub-intent if applicable
            sub_intent = self._determine_sub_intent(best_intent, normalized_query)
            
            # Get routing path
            routing_path = self.routing_paths.get(best_intent.intent, "general_information_pipeline")
            
            # Apply port-specific adjustments
            if port_config:
                best_intent = self._apply_port_specific_adjustments(best_intent, port_config)
            
            result = {
                "intent": best_intent.intent,
                "confidence": best_intent.confidence,
                "sub_intent": sub_intent,
                "keywords": best_intent.keywords,
                "routing_path": routing_path,
                "priority": best_intent.priority,
                "original_query": query,
                "normalized_query": normalized_query
            }
            
            logger.info(f"Query classified as: {best_intent.intent} (confidence: {best_intent.confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return {
                "intent": "general",
                "confidence": 0.5,
                "sub_intent": None,
                "keywords": [],
                "routing_path": "general_information_pipeline",
                "priority": 4,
                "original_query": query,
                "error": str(e)
            }
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for better matching"""
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove punctuation at the end
        normalized = re.sub(r'[.!?]+$', '', normalized)
        
        return normalized
    
    def _find_intent_matches(self, query: str) -> List[IntentResult]:
        """Find all matching intents for a query"""
        matches = []
        
        for intent_name, intent_config in self.intent_patterns.items():
            score = 0.0
            matched_keywords = []
            
            # Check keyword matches
            keywords = intent_config["keywords"]
            for keyword in keywords:
                if keyword in query:
                    score += 1.0
                    matched_keywords.append(keyword)
            
            # Check pattern matches
            patterns = intent_config["patterns"]
            for pattern in patterns:
                if re.search(pattern, query):
                    score += 2.0  # Patterns are weighted higher
            
            # Normalize score
            total_possible = len(keywords) + len(patterns) * 2
            normalized_score = min(1.0, score / total_possible) if total_possible > 0 else 0.0
            
            # Only include matches with some confidence
            if normalized_score > 0.1:
                matches.append(IntentResult(
                    intent=intent_name,
                    confidence=normalized_score,
                    sub_intent=None,
                    keywords=matched_keywords,
                    routing_path=self.routing_paths.get(intent_name, "general_information_pipeline"),
                    priority=intent_config["priority"]
                ))
        
        return matches
    
    def _select_best_intent(self, matches: List[IntentResult], query: str) -> IntentResult:
        """Select the best intent from matches"""
        if not matches:
            return IntentResult(
                intent="general",
                confidence=0.5,
                sub_intent=None,
                keywords=[],
                routing_path="general_information_pipeline",
                priority=4
            )
        
        # Sort by priority first, then by confidence
        matches.sort(key=lambda x: (x.priority, -x.confidence))
        
        best_match = matches[0]
        
        # Boost confidence for exact matches
        if best_match.confidence > 0.8:
            best_match.confidence = min(1.0, best_match.confidence + 0.1)
        
        return best_match
    
    def _determine_sub_intent(self, intent: IntentResult, query: str) -> Optional[str]:
        """Determine sub-intent for more specific routing"""
        sub_intent_patterns = {
            "weather": {
                "forecast": ["forecast", "prediction", "outlook"],
                "current": ["current", "now", "present", "today"],
                "warning": ["warning", "alert", "advisory", "watch"]
            },
            "safety": {
                "protocol": ["protocol", "procedure", "guideline"],
                "emergency": ["emergency", "urgent", "critical"],
                "training": ["training", "education", "course"]
            },
            "berthing": {
                "allocation": ["allocation", "assignment", "booking"],
                "conflict": ["conflict", "dispute", "problem"],
                "schedule": ["schedule", "timing", "arrival", "departure"]
            },
            "cargo": {
                "handling": ["handling", "operation", "process"],
                "planning": ["plan", "planning", "strategy"],
                "equipment": ["equipment", "crane", "machinery"]
            }
        }
        
        intent_sub_patterns = sub_intent_patterns.get(intent.intent, {})
        
        for sub_intent_name, sub_keywords in intent_sub_patterns.items():
            if any(keyword in query for keyword in sub_keywords):
                return sub_intent_name
        
        return None
    
    def _apply_port_specific_adjustments(self, intent: IntentResult, port_config: Dict[str, Any]) -> IntentResult:
        """Apply port-specific adjustments to intent classification"""
        port_type = port_config.get("port_type", "general")
        
        # Adjust confidence based on port type
        if port_type == "container" and intent.intent == "cargo":
            intent.confidence = min(1.0, intent.confidence + 0.1)
        elif port_type == "bulk" and intent.intent == "cargo":
            intent.confidence = min(1.0, intent.confidence + 0.1)
        elif port_type == "multi_purpose" and intent.intent in ["berthing", "cargo", "operations"]:
            intent.confidence = min(1.0, intent.confidence + 0.05)
        
        # Adjust priority based on port operations
        if port_config.get("has_weather_station", False) and intent.intent == "weather":
            intent.priority = max(0, intent.priority - 1)
        
        return intent
    
    def get_intent_statistics(self) -> Dict[str, Any]:
        """Get statistics about intent classification patterns"""
        return {
            "total_intents": len(self.intent_patterns),
            "intent_names": list(self.intent_patterns.keys()),
            "priority_levels": len(set(self.priority_rules.values())),
            "routing_paths": len(self.routing_paths),
            "patterns_per_intent": {
                intent: len(config["keywords"]) + len(config["patterns"])
                for intent, config in self.intent_patterns.items()
            }
        }
    
    def add_custom_intent(self, 
                         intent_name: str, 
                         keywords: List[str], 
                         patterns: List[str], 
                         priority: int = 3,
                         routing_path: Optional[str] = None):
        """
        Add a custom intent classification pattern.
        
        Args:
            intent_name: Name of the new intent
            keywords: List of keywords for this intent
            patterns: List of regex patterns for this intent
            priority: Priority level (0=highest, 4=lowest)
            routing_path: Optional custom routing path
        """
        self.intent_patterns[intent_name] = {
            "keywords": keywords,
            "patterns": patterns,
            "priority": priority
        }
        
        self.priority_rules[intent_name] = priority
        
        if routing_path:
            self.routing_paths[intent_name] = routing_path
        else:
            self.routing_paths[intent_name] = f"{intent_name}_pipeline"
        
        logger.info(f"Added custom intent: {intent_name}")
    
    def update_intent_patterns(self, intent_name: str, **updates):
        """
        Update patterns for an existing intent.
        
        Args:
            intent_name: Name of the intent to update
            **updates: Keyword arguments for updates (keywords, patterns, priority)
        """
        if intent_name not in self.intent_patterns:
            logger.warning(f"Intent {intent_name} not found for update")
            return
        
        if "keywords" in updates:
            self.intent_patterns[intent_name]["keywords"] = updates["keywords"]
        
        if "patterns" in updates:
            self.intent_patterns[intent_name]["patterns"] = updates["patterns"]
        
        if "priority" in updates:
            self.intent_patterns[intent_name]["priority"] = updates["priority"]
            self.priority_rules[intent_name] = updates["priority"]
        
        logger.info(f"Updated intent patterns for: {intent_name}")
