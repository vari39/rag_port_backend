# src/utils/redact.py
"""
Advanced PII Redaction for AI Port Decision-Support System
Implements comprehensive data redaction for sensitive maritime information across multiple languages and scripts
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Unicode and script detection
try:
    import unicodedata
    UNICODE_AVAILABLE = True
except ImportError:
    UNICODE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RedactionRule:
    """Redaction rule configuration"""
    name: str
    pattern: str
    replacement: str
    description: str
    enabled: bool = True
    case_sensitive: bool = False
    language: Optional[str] = None  # Language code if rule is language-specific
    script: Optional[str] = None    # Script type (Latin, Cyrillic, Arabic, etc.)


class AdvancedRedactor:
    """
    Advanced PII and sensitive data redactor for port operations.
    Handles vessel information, cargo values, personal data, and more.
    """
    
    def __init__(self, 
                 custom_rules: Optional[List[RedactionRule]] = None,
                 enable_maritime_redaction: bool = True,
                 enable_pii_redaction: bool = True,
                 enable_financial_redaction: bool = True,
                 enable_multilingual_redaction: bool = True):
        """
        Initialize the advanced redactor.
        
        Args:
            custom_rules: Custom redaction rules
            enable_maritime_redaction: Enable maritime-specific redaction
            enable_pii_redaction: Enable PII redaction
            enable_financial_redaction: Enable financial data redaction
            enable_multilingual_redaction: Enable multilingual redaction patterns
        """
        self.enable_maritime_redaction = enable_maritime_redaction
        self.enable_pii_redaction = enable_pii_redaction
        self.enable_financial_redaction = enable_financial_redaction
        self.enable_multilingual_redaction = enable_multilingual_redaction
        
        # Initialize redaction rules
        self.redaction_rules = self._initialize_redaction_rules()
        
        # Add custom rules if provided
        if custom_rules:
            for rule in custom_rules:
                self.redaction_rules[rule.name] = rule
        
        logger.info(f"Advanced redactor initialized with {len(self.redaction_rules)} rules")
    
    def _initialize_redaction_rules(self) -> Dict[str, RedactionRule]:
        """Initialize comprehensive redaction rules"""
        rules = {}
        
        if self.enable_maritime_redaction:
            # Maritime-specific redaction rules
            maritime_rules = {
                "vessel_imo": RedactionRule(
                    name="vessel_imo",
                    pattern=r'IMO\s*:?\s*(\d{7})',
                    replacement=r'IMO: [REDACTED_VESSEL_ID]',
                    description="Redact vessel IMO numbers",
                    enabled=True
                ),
                "vessel_mmsi": RedactionRule(
                    name="vessel_mmsi",
                    pattern=r'MMSI\s*:?\s*(\d{9})',
                    replacement=r'MMSI: [REDACTED_VESSEL_ID]',
                    description="Redact vessel MMSI numbers",
                    enabled=True
                ),
                "vessel_call_sign": RedactionRule(
                    name="vessel_call_sign",
                    pattern=r'Call\s*Sign\s*:?\s*([A-Z0-9]{3,10})',
                    replacement=r'Call Sign: [REDACTED_CALL_SIGN]',
                    description="Redact vessel call signs",
                    enabled=True
                ),
                "vessel_name": RedactionRule(
                    name="vessel_name",
                    pattern=r'Vessel\s*Name\s*:?\s*([A-Za-z\s\-\']+)',
                    replacement=r'Vessel Name: [REDACTED_VESSEL_NAME]',
                    description="Redact vessel names",
                    enabled=True
                ),
                "berth_number": RedactionRule(
                    name="berth_number",
                    pattern=r'Berth\s*:?\s*([A-Z0-9]+)',
                    replacement=r'Berth: [REDACTED_BERTH]',
                    description="Redact specific berth numbers",
                    enabled=True
                ),
                "terminal_operator": RedactionRule(
                    name="terminal_operator",
                    pattern=r'Terminal\s*Operator\s*:?\s*([A-Za-z\s&]+)',
                    replacement=r'Terminal Operator: [REDACTED_OPERATOR]',
                    description="Redact terminal operator names",
                    enabled=True
                )
            }
            rules.update(maritime_rules)
        
        if self.enable_financial_redaction:
            # Financial data redaction rules
            financial_rules = {
                "cargo_value": RedactionRule(
                    name="cargo_value",
                    pattern=r'\$[\d,]+(?:\.\d{2})?',
                    replacement='[REDACTED_VALUE]',
                    description="Redact monetary values",
                    enabled=True
                ),
                "currency_amount": RedactionRule(
                    name="currency_amount",
                    pattern=r'[€£¥]\s*[\d,]+(?:\.\d{2})?',
                    replacement='[REDACTED_VALUE]',
                    description="Redact currency amounts",
                    enabled=True
                ),
                "bank_account": RedactionRule(
                    name="bank_account",
                    pattern=r'Account\s*:?\s*(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4})',
                    replacement=r'Account: [REDACTED_ACCOUNT]',
                    description="Redact bank account numbers",
                    enabled=True
                ),
                "invoice_number": RedactionRule(
                    name="invoice_number",
                    pattern=r'Invoice\s*#?\s*:?\s*([A-Z0-9\-]+)',
                    replacement=r'Invoice: [REDACTED_INVOICE]',
                    description="Redact invoice numbers",
                    enabled=True
                )
            }
            rules.update(financial_rules)
        
        if self.enable_pii_redaction:
            # Personal information redaction rules
            pii_rules = {
                "email_address": RedactionRule(
                    name="email_address",
                    pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    replacement='[REDACTED_EMAIL]',
                    description="Redact email addresses",
                    enabled=True
                ),
                "phone_number": RedactionRule(
                    name="phone_number",
                    pattern=r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                    replacement='[REDACTED_PHONE]',
                    description="Redact phone numbers",
                    enabled=True
                ),
                "personal_name": RedactionRule(
                    name="personal_name",
                    pattern=r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
                    replacement='[REDACTED_NAME]',
                    description="Redact personal names",
                    enabled=True
                ),
                "ssn": RedactionRule(
                    name="ssn",
                    pattern=r'\b\d{3}-\d{2}-\d{4}\b',
                    replacement='[REDACTED_SSN]',
                    description="Redact Social Security Numbers",
                    enabled=True
                ),
                "passport_number": RedactionRule(
                    name="passport_number",
                    pattern=r'Passport\s*:?\s*([A-Z0-9]{6,12})',
                    replacement=r'Passport: [REDACTED_PASSPORT]',
                    description="Redact passport numbers",
                    enabled=True
                ),
                "driver_license": RedactionRule(
                    name="driver_license",
                    pattern=r'License\s*:?\s*([A-Z0-9]{6,12})',
                    replacement=r'License: [REDACTED_LICENSE]',
                    description="Redact driver license numbers",
                    enabled=True
                )
            }
            rules.update(pii_rules)
        
        # Additional port-specific rules
        port_rules = {
            "container_number": RedactionRule(
                name="container_number",
                pattern=r'Container\s*:?\s*([A-Z]{4}\d{7})',
                replacement=r'Container: [REDACTED_CONTAINER]',
                description="Redact container numbers",
                enabled=True
            ),
            "booking_reference": RedactionRule(
                name="booking_reference",
                pattern=r'Booking\s*Ref\s*:?\s*([A-Z0-9]{6,12})',
                replacement=r'Booking Ref: [REDACTED_BOOKING]',
                description="Redact booking references",
                enabled=True
            ),
            "customs_declaration": RedactionRule(
                name="customs_declaration",
                pattern=r'Customs\s*Decl\s*:?\s*([A-Z0-9]{6,12})',
                replacement=r'Customs Decl: [REDACTED_CUSTOMS]',
                description="Redact customs declaration numbers",
                enabled=True
            ),
            "port_authority_code": RedactionRule(
                name="port_authority_code",
                pattern=r'Port\s*Auth\s*:?\s*([A-Z0-9]{3,8})',
                replacement=r'Port Auth: [REDACTED_AUTHORITY]',
                description="Redact port authority codes",
                enabled=True
            )
        }
        rules.update(port_rules)
        
        # Add multilingual redaction rules if enabled
        if self.enable_multilingual_redaction:
            multilingual_rules = self._initialize_multilingual_rules()
            rules.update(multilingual_rules)
        
        return rules
    
    def _initialize_multilingual_rules(self) -> Dict[str, RedactionRule]:
        """Initialize multilingual redaction rules for different scripts"""
        rules = {}
        
        # Arabic script patterns
        arabic_rules = {
            "arabic_phone": RedactionRule(
                name="arabic_phone",
                pattern=r'[\u0600-\u06FF]*[\d]{3}[\u0600-\u06FF]*[\d]{3}[\u0600-\u06FF]*[\d]{4}',
                replacement='[REDACTED_PHONE_AR]',
                description="Redact Arabic phone numbers",
                enabled=True,
                script="Arabic"
            ),
            "arabic_email": RedactionRule(
                name="arabic_email",
                pattern=r'[\u0600-\u06FF\w._%+-]+@[\u0600-\u06FF\w.-]+\.[\u0600-\u06FF\w]{2,}',
                replacement='[REDACTED_EMAIL_AR]',
                description="Redact Arabic email addresses",
                enabled=True,
                script="Arabic"
            ),
            "arabic_name": RedactionRule(
                name="arabic_name",
                pattern=r'[\u0600-\u06FF]+\s+[\u0600-\u06FF]+',
                replacement='[REDACTED_NAME_AR]',
                description="Redact Arabic names",
                enabled=True,
                script="Arabic"
            )
        }
        rules.update(arabic_rules)
        
        # Cyrillic script patterns
        cyrillic_rules = {
            "cyrillic_phone": RedactionRule(
                name="cyrillic_phone",
                pattern=r'[\u0400-\u04FF]*[\d]{3}[\u0400-\u04FF]*[\d]{3}[\u0400-\u04FF]*[\d]{4}',
                replacement='[REDACTED_PHONE_CY]',
                description="Redact Cyrillic phone numbers",
                enabled=True,
                script="Cyrillic"
            ),
            "cyrillic_email": RedactionRule(
                name="cyrillic_email",
                pattern=r'[\u0400-\u04FF\w._%+-]+@[\u0400-\u04FF\w.-]+\.[\u0400-\u04FF\w]{2,}',
                replacement='[REDACTED_EMAIL_CY]',
                description="Redact Cyrillic email addresses",
                enabled=True,
                script="Cyrillic"
            ),
            "cyrillic_name": RedactionRule(
                name="cyrillic_name",
                pattern=r'[\u0400-\u04FF]+\s+[\u0400-\u04FF]+',
                replacement='[REDACTED_NAME_CY]',
                description="Redact Cyrillic names",
                enabled=True,
                script="Cyrillic"
            )
        }
        rules.update(cyrillic_rules)
        
        # Chinese script patterns
        chinese_rules = {
            "chinese_phone": RedactionRule(
                name="chinese_phone",
                pattern=r'[\u4e00-\u9fff]*[\d]{3}[\u4e00-\u9fff]*[\d]{4}[\u4e00-\u9fff]*[\d]{4}',
                replacement='[REDACTED_PHONE_ZH]',
                description="Redact Chinese phone numbers",
                enabled=True,
                script="Chinese"
            ),
            "chinese_email": RedactionRule(
                name="chinese_email",
                pattern=r'[\u4e00-\u9fff\w._%+-]+@[\u4e00-\u9fff\w.-]+\.[\u4e00-\u9fff\w]{2,}',
                replacement='[REDACTED_EMAIL_ZH]',
                description="Redact Chinese email addresses",
                enabled=True,
                script="Chinese"
            ),
            "chinese_name": RedactionRule(
                name="chinese_name",
                pattern=r'[\u4e00-\u9fff]{2,4}',
                replacement='[REDACTED_NAME_ZH]',
                description="Redact Chinese names",
                enabled=True,
                script="Chinese"
            )
        }
        rules.update(chinese_rules)
        
        # Japanese script patterns
        japanese_rules = {
            "japanese_phone": RedactionRule(
                name="japanese_phone",
                pattern=r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]*[\d]{3}[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]*[\d]{4}[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]*[\d]{4}',
                replacement='[REDACTED_PHONE_JA]',
                description="Redact Japanese phone numbers",
                enabled=True,
                script="Japanese"
            ),
            "japanese_email": RedactionRule(
                name="japanese_email",
                pattern=r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\w._%+-]+@[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\w.-]+\.[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\w]{2,}',
                replacement='[REDACTED_EMAIL_JA]',
                description="Redact Japanese email addresses",
                enabled=True,
                script="Japanese"
            ),
            "japanese_name": RedactionRule(
                name="japanese_name",
                pattern=r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]{2,6}',
                replacement='[REDACTED_NAME_JA]',
                description="Redact Japanese names",
                enabled=True,
                script="Japanese"
            )
        }
        rules.update(japanese_rules)
        
        # Korean script patterns
        korean_rules = {
            "korean_phone": RedactionRule(
                name="korean_phone",
                pattern=r'[\uac00-\ud7af]*[\d]{3}[\uac00-\ud7af]*[\d]{4}[\uac00-\ud7af]*[\d]{4}',
                replacement='[REDACTED_PHONE_KO]',
                description="Redact Korean phone numbers",
                enabled=True,
                script="Korean"
            ),
            "korean_email": RedactionRule(
                name="korean_email",
                pattern=r'[\uac00-\ud7af\w._%+-]+@[\uac00-\ud7af\w.-]+\.[\uac00-\ud7af\w]{2,}',
                replacement='[REDACTED_EMAIL_KO]',
                description="Redact Korean email addresses",
                enabled=True,
                script="Korean"
            ),
            "korean_name": RedactionRule(
                name="korean_name",
                pattern=r'[\uac00-\ud7af]{2,4}',
                replacement='[REDACTED_NAME_KO]',
                description="Redact Korean names",
                enabled=True,
                script="Korean"
            )
        }
        rules.update(korean_rules)
        
        return rules
    
    def detect_script(self, text: str) -> str:
        """
        Detect the primary script used in the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Script name (Latin, Arabic, Cyrillic, Chinese, Japanese, Korean, etc.)
        """
        if not UNICODE_AVAILABLE or not text.strip():
            return "Latin"
        
        script_counts = {}
        
        for char in text:
            if char.isalpha():
                script = unicodedata.name(char).split()[0]
                script_counts[script] = script_counts.get(script, 0) + 1
        
        if not script_counts:
            return "Latin"
        
        # Return the most common script
        return max(script_counts.items(), key=lambda x: x[1])[0]
    
    def get_script_specific_rules(self, script: str) -> List[str]:
        """
        Get redaction rules specific to a script.
        
        Args:
            script: Script name
            
        Returns:
            List of rule names for the script
        """
        script_rules = []
        
        for rule_name, rule in self.redaction_rules.items():
            if rule.script and rule.script.lower() == script.lower():
                script_rules.append(rule_name)
        
        return script_rules
    
    def redact_text(self, text: str, custom_rules: Optional[List[str]] = None, 
                   auto_detect_script: bool = True) -> Tuple[str, Dict[str, int]]:
        """
        Redact sensitive information from text with multilingual support.
        
        Args:
            text: Text to redact
            custom_rules: Optional list of specific rules to apply
            auto_detect_script: Whether to automatically detect script and apply script-specific rules
            
        Returns:
            Tuple of (redacted_text, redaction_counts)
        """
        try:
            redacted_text = text
            redaction_counts = {}
            
            # Detect script if enabled and no custom rules specified
            if auto_detect_script and custom_rules is None:
                detected_script = self.detect_script(text)
                script_specific_rules = self.get_script_specific_rules(detected_script)
                logger.info(f"Detected script: {detected_script}, applying {len(script_specific_rules)} script-specific rules")
                
                # Combine general rules with script-specific rules
                general_rules = [name for name, rule in self.redaction_rules.items() if rule.script is None]
                rules_to_apply = general_rules + script_specific_rules
            else:
                rules_to_apply = custom_rules or list(self.redaction_rules.keys())
            
            for rule_name in rules_to_apply:
                if rule_name not in self.redaction_rules:
                    logger.warning(f"Redaction rule '{rule_name}' not found")
                    continue
                
                rule = self.redaction_rules[rule_name]
                if not rule.enabled:
                    continue
                
                # Apply redaction rule
                flags = 0 if rule.case_sensitive else re.IGNORECASE
                matches = re.findall(rule.pattern, redacted_text, flags)
                
                if matches:
                    redacted_text = re.sub(rule.pattern, rule.replacement, redacted_text, flags=flags)
                    redaction_counts[rule_name] = len(matches)
                    logger.debug(f"Applied rule '{rule_name}': {len(matches)} matches")
            
            return redacted_text, redaction_counts
            
        except Exception as e:
            logger.error(f"Error in text redaction: {e}")
            return text, {}
    
    def redact_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redact sensitive information from a document.
        
        Args:
            document: Document dictionary with 'content' and 'metadata' keys
            
        Returns:
            Redacted document dictionary
        """
        try:
            redacted_document = document.copy()
            
            # Redact content
            if 'content' in document:
                redacted_content, redaction_counts = self.redact_text(document['content'])
                redacted_document['content'] = redacted_content
                
                # Add redaction metadata
                redacted_document['metadata'] = redacted_document.get('metadata', {})
                redacted_document['metadata']['redaction_applied'] = True
                redacted_document['metadata']['redaction_counts'] = redaction_counts
                redacted_document['metadata']['redacted_at'] = datetime.now().isoformat()
            
            return redacted_document
            
        except Exception as e:
            logger.error(f"Error redacting document: {e}")
            return document
    
    def redact_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Redact sensitive information from multiple documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of redacted document dictionaries
        """
        try:
            logger.info(f"Redacting {len(documents)} documents")
            
            redacted_documents = []
            total_redactions = {}
            
            for i, doc in enumerate(documents):
                try:
                    redacted_doc = self.redact_document(doc)
                    redacted_documents.append(redacted_doc)
                    
                    # Aggregate redaction counts
                    doc_redactions = redacted_doc.get('metadata', {}).get('redaction_counts', {})
                    for rule_name, count in doc_redactions.items():
                        total_redactions[rule_name] = total_redactions.get(rule_name, 0) + count
                        
                except Exception as e:
                    logger.error(f"Error redacting document {i}: {e}")
                    redacted_documents.append(doc)  # Return original on error
            
            logger.info(f"Redaction completed. Total redactions: {total_redactions}")
            return redacted_documents
            
        except Exception as e:
            logger.error(f"Error redacting documents: {e}")
            return documents
    
    def add_custom_rule(self, rule: RedactionRule):
        """
        Add a custom redaction rule.
        
        Args:
            rule: RedactionRule object
        """
        self.redaction_rules[rule.name] = rule
        logger.info(f"Added custom redaction rule: {rule.name}")
    
    def update_rule_status(self, rule_name: str, enabled: bool):
        """
        Enable or disable a redaction rule.
        
        Args:
            rule_name: Name of the rule to update
            enabled: Whether to enable the rule
        """
        if rule_name in self.redaction_rules:
            self.redaction_rules[rule_name].enabled = enabled
            logger.info(f"Updated rule '{rule_name}' status: {enabled}")
        else:
            logger.warning(f"Rule '{rule_name}' not found")
    
    def get_redaction_statistics(self) -> Dict[str, Any]:
        """Get statistics about redaction rules"""
        enabled_rules = sum(1 for rule in self.redaction_rules.values() if rule.enabled)
        total_rules = len(self.redaction_rules)
        
        return {
            "total_rules": total_rules,
            "enabled_rules": enabled_rules,
            "disabled_rules": total_rules - enabled_rules,
            "rule_categories": {
                "maritime": sum(1 for rule in self.redaction_rules.values() 
                              if rule.name in ["vessel_imo", "vessel_mmsi", "vessel_call_sign", "vessel_name", "berth_number", "terminal_operator"]),
                "financial": sum(1 for rule in self.redaction_rules.values() 
                               if rule.name in ["cargo_value", "currency_amount", "bank_account", "invoice_number"]),
                "pii": sum(1 for rule in self.redaction_rules.values() 
                          if rule.name in ["email_address", "phone_number", "personal_name", "ssn", "passport_number", "driver_license"]),
                "port_specific": sum(1 for rule in self.redaction_rules.values() 
                                   if rule.name in ["container_number", "booking_reference", "customs_declaration", "port_authority_code"])
            },
            "rules": {name: {"enabled": rule.enabled, "description": rule.description} 
                     for name, rule in self.redaction_rules.items()}
        }
    
    def test_redaction(self, test_text: str) -> Dict[str, Any]:
        """
        Test redaction rules on sample text.
        
        Args:
            test_text: Text to test redaction on
            
        Returns:
            Test results with before/after and redaction counts
        """
        try:
            redacted_text, redaction_counts = self.redact_text(test_text)
            
            return {
                "original_text": test_text,
                "redacted_text": redacted_text,
                "redaction_counts": redaction_counts,
                "total_redactions": sum(redaction_counts.values()),
                "rules_applied": list(redaction_counts.keys())
            }
            
        except Exception as e:
            logger.error(f"Error testing redaction: {e}")
            return {"error": str(e)}


def create_redactor(maritime: bool = True, 
                   pii: bool = True, 
                   financial: bool = True,
                   multilingual: bool = True,
                   custom_rules: Optional[List[RedactionRule]] = None) -> AdvancedRedactor:
    """
    Factory function to create a configured redactor.
    
    Args:
        maritime: Enable maritime redaction
        pii: Enable PII redaction
        financial: Enable financial redaction
        multilingual: Enable multilingual redaction
        custom_rules: Custom redaction rules
        
    Returns:
        Configured AdvancedRedactor instance
    """
    return AdvancedRedactor(
        custom_rules=custom_rules,
        enable_maritime_redaction=maritime,
        enable_pii_redaction=pii,
        enable_financial_redaction=financial,
        enable_multilingual_redaction=multilingual
    )


def quick_redact(text: str, 
                maritime: bool = True, 
                pii: bool = True, 
                financial: bool = True) -> str:
    """
    Quick redaction function for simple use cases.
    
    Args:
        text: Text to redact
        maritime: Enable maritime redaction
        pii: Enable PII redaction
        financial: Enable financial redaction
        
    Returns:
        Redacted text
    """
    redactor = create_redactor(maritime=maritime, pii=pii, financial=financial)
    redacted_text, _ = redactor.redact_text(text)
    return redacted_text
