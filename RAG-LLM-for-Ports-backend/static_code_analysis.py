#!/usr/bin/env python3
"""
Static Code Analysis - Checks code structure and logic without network access
"""

import os
import sys
import ast
import re
from typing import List, Dict, Any

class CodeAnalyzer:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.checks_passed = 0
        
    def log_issue(self, severity: str, category: str, message: str, file: str = "", line: int = 0):
        """Log an issue"""
        issue = {
            "severity": severity,
            "category": category,
            "message": message,
            "file": file,
            "line": line
        }
        if severity == "ERROR":
            self.issues.append(issue)
        else:
            self.warnings.append(issue)
        print(f"  [{severity}] {category}: {message}" + (f" ({file}:{line})" if file else ""))
    
    def log_success(self, message: str):
        """Log success"""
        print(f"  ✅ {message}")
        self.checks_passed += 1
    
    def check_file_exists(self, filepath: str) -> bool:
        """Check if file exists"""
        return os.path.exists(filepath)
    
    def check_imports(self, filepath: str):
        """Check imports in a file"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                tree = ast.parse(content)
                
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
                
                # Check for critical imports
                if 'langgraph' not in str(imports):
                    self.log_issue("WARNING", "Imports", f"LangGraph imports may be missing in {filepath}")
                else:
                    self.log_success(f"Imports check passed for {os.path.basename(filepath)}")
        except Exception as e:
            self.log_issue("ERROR", "Parsing", f"Could not parse {filepath}: {e}")
    
    def check_return_types(self, filepath: str):
        """Check return type consistency"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
                # Check for return type annotations
                # Look for async def functions returning PortWorkflowState vs Dict[str, Any]
                pattern = r'async def (\w+)\(.*?\) -> (PortWorkflowState|Dict\[str, Any\])'
                matches = re.findall(pattern, content)
                
                port_workflow_returns = []
                dict_returns = []
                
                for func_name, return_type in matches:
                    if return_type == "PortWorkflowState":
                        port_workflow_returns.append(func_name)
                    else:
                        dict_returns.append(func_name)
                
                if port_workflow_returns:
                    self.log_issue("WARNING", "Return Types", 
                                 f"Functions returning PortWorkflowState: {', '.join(port_workflow_returns)}",
                                 filepath)
                else:
                    self.log_success("All async functions return Dict[str, Any] (partial state)")
                    
        except Exception as e:
            self.log_issue("ERROR", "Analysis", f"Error analyzing {filepath}: {e}")
    
    def check_error_handling(self, filepath: str):
        """Check error handling patterns"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                
                # Check for try-except blocks
                has_try_except = 'try:' in content and 'except' in content
                
                # Check for default values in error cases
                default_patterns = [
                    r'"No answer available"',
                    r'"No recommendation available"',
                    r'\.get\([^,]+,\s*["\']No ',
                ]
                
                defaults_found = []
                for i, line in enumerate(lines, 1):
                    for pattern in default_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            defaults_found.append((i, line.strip()))
                
                if defaults_found:
                    for line_num, line_content in defaults_found:
                        self.log_issue("ERROR", "Default Values", 
                                     f"Found default value pattern: {line_content[:60]}",
                                     filepath, line_num)
                else:
                    self.log_success("No default answer patterns found")
                
                if has_try_except:
                    self.log_success("Error handling (try-except) present")
                else:
                    self.log_issue("WARNING", "Error Handling", 
                                 "No try-except blocks found", filepath)
                    
        except Exception as e:
            self.log_issue("ERROR", "Analysis", f"Error checking {filepath}: {e}")
    
    def check_schema_consistency(self):
        """Check schema consistency between graph and API"""
        graph_file = "src/graph/port_graph.py"
        api_file = "src/api/server.py"
        schemas_file = "src/utils/schemas.py"
        
        if not all(self.check_file_exists(f) for f in [graph_file, api_file, schemas_file]):
            self.log_issue("ERROR", "Files", "Required files missing")
            return
        
        # Check AlternativeScenario schema
        try:
            with open(schemas_file, 'r') as f:
                schemas_content = f.read()
                
            # Check if recommendations field exists
            if 'recommendations: List[str]' in schemas_content or 'recommendations.*List' in schemas_content:
                self.log_success("AlternativeScenario schema has recommendations field")
            else:
                self.log_issue("ERROR", "Schema", "AlternativeScenario missing recommendations field")
            
            # Check WorkflowResponse
            if 'class WorkflowResponse' in schemas_content:
                self.log_success("WorkflowResponse schema found")
            else:
                self.log_issue("ERROR", "Schema", "WorkflowResponse schema not found")
                
        except Exception as e:
            self.log_issue("ERROR", "Schema Check", f"Error checking schemas: {e}")
    
    def check_state_updates(self, filepath: str):
        """Check state update patterns"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
            # Check for return statements that return full state vs partial
            # Look for patterns like "return state" vs "return {...}"
            return_state_pattern = r'return\s+state\s*$'
            return_dict_pattern = r'return\s+\{[^}]+\}'
            
            return_state_matches = len(re.findall(return_state_pattern, content, re.MULTILINE))
            return_dict_matches = len(re.findall(return_dict_pattern, content, re.MULTILINE))
            
            if return_state_matches > 0:
                self.log_issue("WARNING", "State Updates", 
                             f"Found {return_state_matches} return state statements (should return partial dict)",
                             filepath)
            else:
                self.log_success("No full state returns found (using partial state updates)")
                
        except Exception as e:
            self.log_issue("ERROR", "Analysis", f"Error checking state updates: {e}")
    
    def check_parallel_analysis(self, filepath: str):
        """Check parallel analysis implementation"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
            # Check for asyncio.gather usage
            if 'asyncio.gather' in content:
                self.log_success("Parallel analysis uses asyncio.gather")
            else:
                self.log_issue("WARNING", "Parallel Analysis", "asyncio.gather not found")
            
            # Check for return_exceptions=True
            if 'return_exceptions=True' in content:
                self.log_success("Parallel analysis handles exceptions properly")
            else:
                self.log_issue("WARNING", "Parallel Analysis", "return_exceptions not set to True")
                
        except Exception as e:
            self.log_issue("ERROR", "Analysis", f"Error checking parallel analysis: {e}")
    
    def analyze_graph_file(self):
        """Analyze the main graph file"""
        filepath = "src/graph/port_graph.py"
        
        if not self.check_file_exists(filepath):
            self.log_issue("ERROR", "Files", f"{filepath} not found")
            return
        
        print(f"\nAnalyzing {filepath}...")
        self.check_return_types(filepath)
        self.check_error_handling(filepath)
        self.check_state_updates(filepath)
        self.check_parallel_analysis(filepath)
    
    def analyze_api_file(self):
        """Analyze the API server file"""
        filepath = "src/api/server.py"
        
        if not self.check_file_exists(filepath):
            self.log_issue("ERROR", "Files", f"{filepath} not found")
            return
        
        print(f"\nAnalyzing {filepath}...")
        self.check_error_handling(filepath)
        self.check_imports(filepath)
    
    def run_all_checks(self):
        """Run all static checks"""
        print("="*70)
        print("STATIC CODE ANALYSIS")
        print("="*70)
        
        # Change to backend directory
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(backend_dir)
        
        print("\n" + "="*70)
        print("CHECK 1: Graph File Analysis")
        print("="*70)
        self.analyze_graph_file()
        
        print("\n" + "="*70)
        print("CHECK 2: API File Analysis")
        print("="*70)
        self.analyze_api_file()
        
        print("\n" + "="*70)
        print("CHECK 3: Schema Consistency")
        print("="*70)
        self.check_schema_consistency()
        
        # Summary
        print("\n" + "="*70)
        print("ANALYSIS SUMMARY")
        print("="*70)
        print(f"Checks Passed: {self.checks_passed}")
        print(f"Errors Found: {len(self.issues)}")
        print(f"Warnings Found: {len(self.warnings)}")
        
        if self.issues:
            print("\nERRORS:")
            for issue in self.issues:
                print(f"  - {issue['category']}: {issue['message']}")
        
        if self.warnings:
            print("\nWARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning['category']}: {warning['message']}")
        
        if not self.issues and not self.warnings:
            print("\n✅ No issues found! Code structure looks good.")
        
        print("="*70)
        
        return len(self.issues) == 0

def main():
    analyzer = CodeAnalyzer()
    success = analyzer.run_all_checks()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

