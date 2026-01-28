#!/usr/bin/env python3
"""
Comprehensive Test Suite for RAG Port Project
Tests all functionalities across all modules
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import traceback

# Add all backend paths to sys.path
PROJECT_ROOT = Path(__file__).parent
BACKEND_PATHS = [
    PROJECT_ROOT / "RAG-LLM-for-Ports-backend" / "src",
    PROJECT_ROOT / "RAG-LLM-for-Ports-backend_dbsetup" / "src",
    PROJECT_ROOT / "RAG-LLM-for-Ports-backend_translate" / "src",
]

for path in BACKEND_PATHS:
    if path.exists():
        sys.path.insert(0, str(path.parent))

# Try to load .env from main folder
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        try:
            load_dotenv(env_path)
        except PermissionError:
            # If can't read .env, try environment variable directly
            pass
except ImportError:
    pass

class ComprehensiveTestSuite:
    def __init__(self):
        self.results = {
            "passed": [],
            "failed": [],
            "warnings": [],
            "errors": []
        }
        self.api_key = os.getenv('OPENAI_API_KEY')
        
    def log_test(self, test_name: str, passed: bool, message: str = "", warning: bool = False):
        """Log test result"""
        if passed:
            self.results["passed"].append({"test": test_name, "message": message})
            print(f"  ✅ {test_name}: {message}")
        elif warning:
            self.results["warnings"].append({"test": test_name, "message": message})
            print(f"  ⚠️  {test_name}: {message}")
        else:
            self.results["failed"].append({"test": test_name, "message": message})
            print(f"  ❌ {test_name}: {message}")
    
    def test_environment_setup(self):
        """Test 1: Environment and API Key"""
        print("\n" + "="*70)
        print("TEST 1: Environment Setup")
        print("="*70)
        
        if not self.api_key:
            self.log_test("API Key Check", False, "OPENAI_API_KEY not found in environment")
            return False
        
        if self.api_key.startswith("sk-"):
            self.log_test("API Key Check", True, "API key found and appears valid")
        else:
            self.log_test("API Key Check", False, "API key format invalid")
            return False
        
        # Check .env file exists in main folder
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            self.log_test(".env File", True, ".env file exists in main folder")
        else:
            self.log_test(".env File", False, ".env file not found in main folder", warning=True)
        
        return True
    
    async def test_backend_imports(self):
        """Test 2: Backend Module Imports"""
        print("\n" + "="*70)
        print("TEST 2: Backend Module Imports")
        print("="*70)
        
        modules_to_test = [
            ("RAG-LLM-for-Ports-backend", [
                "src.port_information_retrieval",
                "src.graph.port_graph",
                "src.api.server",
                "src.routers.intent_router",
                "src.routers.decision_nodes"
            ]),
        ]
        
        all_passed = True
        for backend_name, modules in modules_to_test:
            backend_path = PROJECT_ROOT / backend_name
            if not backend_path.exists():
                self.log_test(f"{backend_name} Exists", False, f"Folder not found")
                all_passed = False
                continue
            
            self.log_test(f"{backend_name} Exists", True, "Folder found")
            
            for module_path in modules:
                try:
                    # Add backend to path temporarily
                    sys.path.insert(0, str(backend_path))
                    __import__(module_path)
                    self.log_test(f"Import {module_path}", True, "Module imported successfully")
                    sys.path.remove(str(backend_path))
                except Exception as e:
                    self.log_test(f"Import {module_path}", False, f"Import failed: {str(e)[:100]}")
                    all_passed = False
                    if str(backend_path) in sys.path:
                        sys.path.remove(str(backend_path))
        
        return all_passed
    
    async def test_port_retrieval(self):
        """Test 3: Port Information Retrieval"""
        print("\n" + "="*70)
        print("TEST 3: Port Information Retrieval")
        print("="*70)
        
        if not self.api_key:
            self.log_test("Port Retrieval", False, "API key not available")
            return False
        
        try:
            backend_path = PROJECT_ROOT / "RAG-LLM-for-Ports-backend"
            if not backend_path.exists():
                self.log_test("Port Retrieval", False, "Backend folder not found")
                return False
            
            sys.path.insert(0, str(backend_path))
            from src.port_information_retrieval import PortInformationRetrieval
            
            pr = PortInformationRetrieval(
                openai_api_key=self.api_key,
                chroma_persist_directory=str(backend_path / "storage" / "chroma"),
                collection_name="port_documents"
            )
            
            # Test query
            result = await pr.query_port_information("What are safety protocols?", max_documents=2)
            
            if result.get("answer") and len(result.get("answer", "")) > 50:
                self.log_test("Port Retrieval Query", True, f"Got answer ({len(result['answer'])} chars)")
            else:
                self.log_test("Port Retrieval Query", False, "Empty or short answer")
                return False
            
            if result.get("sources"):
                self.log_test("Port Retrieval Sources", True, f"Got {len(result['sources'])} sources")
            else:
                self.log_test("Port Retrieval Sources", False, "No sources returned", warning=True)
            
            sys.path.remove(str(backend_path))
            return True
            
        except Exception as e:
            self.log_test("Port Retrieval", False, f"Error: {str(e)[:100]}")
            if str(backend_path) in sys.path:
                sys.path.remove(str(backend_path))
            return False
    
    async def test_langgraph_workflow(self):
        """Test 4: LangGraph Workflow"""
        print("\n" + "="*70)
        print("TEST 4: LangGraph Workflow")
        print("="*70)
        
        if not self.api_key:
            self.log_test("LangGraph Workflow", False, "API key not available")
            return False
        
        try:
            backend_path = PROJECT_ROOT / "RAG-LLM-for-Ports-backend"
            if not backend_path.exists():
                self.log_test("LangGraph Workflow", False, "Backend folder not found")
                return False
            
            sys.path.insert(0, str(backend_path))
            from src.port_information_retrieval import PortInformationRetrieval
            from src.graph.port_graph import PortWorkflowGraph
            
            pr = PortInformationRetrieval(
                openai_api_key=self.api_key,
                chroma_persist_directory=str(backend_path / "storage" / "chroma"),
                collection_name="port_documents"
            )
            
            wg = PortWorkflowGraph(port_retrieval=pr, enable_checkpoints=True)
            
            result = await wg.execute_workflow(
                query="What are the safety protocols for vessel berthing?",
                user_id="test_user"
            )
            
            # Check for errors
            if "error" in result:
                self.log_test("LangGraph Execution", False, f"Workflow error: {result['error']}")
                return False
            
            # Check recommendation
            recommendation = result.get("final_recommendation")
            if recommendation and len(recommendation) > 50:
                self.log_test("LangGraph Recommendation", True, f"Got recommendation ({len(recommendation)} chars)")
            else:
                self.log_test("LangGraph Recommendation", False, "Empty recommendation")
                return False
            
            # Check scenarios
            scenarios = result.get("alternative_scenarios", [])
            if len(scenarios) >= 4:
                self.log_test("LangGraph Scenarios", True, f"Got {len(scenarios)} scenarios")
            else:
                self.log_test("LangGraph Scenarios", False, f"Only {len(scenarios)}/4 scenarios", warning=True)
            
            # Check sources
            sources = result.get("sources", [])
            if sources:
                self.log_test("LangGraph Sources", True, f"Got {len(sources)} sources")
            else:
                self.log_test("LangGraph Sources", False, "No sources", warning=True)
            
            # Check execution time
            exec_time = result.get("execution_time", 0)
            if 1 < exec_time < 60:
                self.log_test("LangGraph Performance", True, f"Execution time: {exec_time:.2f}s")
            else:
                self.log_test("LangGraph Performance", False, f"Execution time suspicious: {exec_time:.2f}s", warning=True)
            
            sys.path.remove(str(backend_path))
            return True
            
        except Exception as e:
            self.log_test("LangGraph Workflow", False, f"Error: {str(e)[:100]}")
            traceback.print_exc()
            if str(backend_path) in sys.path:
                sys.path.remove(str(backend_path))
            return False
    
    async def test_api_endpoints(self):
        """Test 5: API Endpoints (if server is running)"""
        print("\n" + "="*70)
        print("TEST 5: API Endpoints")
        print("="*70)
        
        import httpx
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Test health endpoint
                try:
                    response = await client.get("http://localhost:8000/health")
                    if response.status_code == 200:
                        self.log_test("API Health Check", True, "Server is running")
                        
                        # Test /ask endpoint
                        try:
                            response = await client.post(
                                "http://localhost:8000/ask",
                                json={"query": "What are safety protocols?"},
                                timeout=30.0
                            )
                            if response.status_code == 200:
                                data = response.json()
                                if data.get("answer"):
                                    self.log_test("API /ask Endpoint", True, "Endpoint working")
                                else:
                                    self.log_test("API /ask Endpoint", False, "No answer returned")
                            else:
                                self.log_test("API /ask Endpoint", False, f"Status {response.status_code}")
                        except Exception as e:
                            self.log_test("API /ask Endpoint", False, f"Error: {str(e)[:50]}")
                        
                        # Test /ask_graph endpoint
                        try:
                            response = await client.post(
                                "http://localhost:8000/ask_graph",
                                json={"query": "What are safety protocols?"},
                                timeout=60.0
                            )
                            if response.status_code == 200:
                                data = response.json()
                                if data.get("recommendation"):
                                    self.log_test("API /ask_graph Endpoint", True, "Endpoint working")
                                else:
                                    self.log_test("API /ask_graph Endpoint", False, "No recommendation")
                            else:
                                self.log_test("API /ask_graph Endpoint", False, f"Status {response.status_code}")
                        except Exception as e:
                            self.log_test("API /ask_graph Endpoint", False, f"Error: {str(e)[:50]}")
                        
                        return True
                    else:
                        self.log_test("API Health Check", False, f"Status {response.status_code}")
                        return False
                except Exception:
                    self.log_test("API Health Check", False, "Server not running (this is OK if not started)", warning=True)
                    return True  # Not a failure if server isn't running
        except Exception as e:
            self.log_test("API Endpoints", False, f"Connection error: {str(e)[:50]}", warning=True)
            return True  # Not a failure if server isn't running
    
    def test_file_structure(self):
        """Test 6: File Structure and Organization"""
        print("\n" + "="*70)
        print("TEST 6: File Structure")
        print("="*70)
        
        all_passed = True
        
        # Check main folders exist
        required_folders = [
            "RAG-LLM-for-Ports-backend",
        ]
        
        for folder in required_folders:
            folder_path = PROJECT_ROOT / folder
            if folder_path.exists():
                self.log_test(f"Folder {folder}", True, "Exists")
            else:
                self.log_test(f"Folder {folder}", False, "Missing")
                all_passed = False
        
        # Check for .env in main folder
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            self.log_test(".env in Main", True, ".env file exists")
        else:
            self.log_test(".env in Main", False, ".env file missing", warning=True)
        
        # Check for redundant test files (will be cleaned)
        backend_path = PROJECT_ROOT / "RAG-LLM-for-Ports-backend"
        if backend_path.exists():
            test_files = list(backend_path.glob("test*.py"))
            if len(test_files) > 1:
                self.log_test("Test Files", False, f"Multiple test files found ({len(test_files)}), should consolidate", warning=True)
            else:
                self.log_test("Test Files", True, "Test files organized")
        
        return all_passed
    
    def test_no_hardcoded_keys(self):
        """Test 7: No Hardcoded API Keys"""
        print("\n" + "="*70)
        print("TEST 7: No Hardcoded API Keys")
        print("="*70)
        
        api_key_pattern = r'sk-[a-zA-Z0-9]{20,}'
        hardcoded_found = []
        
        # Check Python files in backend folders
        for backend_folder in ["RAG-LLM-for-Ports-backend", "RAG-LLM-for-Ports-backend_dbsetup", "RAG-LLM-for-Ports-backend_translate"]:
            backend_path = PROJECT_ROOT / backend_folder
            if not backend_path.exists():
                continue
            
            for py_file in backend_path.rglob("*.py"):
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                        matches = list(re.finditer(api_key_pattern, content))
                        if matches:
                            for match in matches:
                                # Check if it's in a comment or string that looks like example
                                line_start = content.rfind('\n', 0, match.start()) + 1
                                line_end = content.find('\n', match.end())
                                line = content[line_start:line_end if line_end != -1 else len(content)]
                                
                                if 'example' not in line.lower() and 'your_' not in line.lower():
                                    hardcoded_found.append({
                                        "file": str(py_file.relative_to(PROJECT_ROOT)),
                                        "line": content[:match.start()].count('\n') + 1
                                    })
                except Exception:
                    pass
        
        if hardcoded_found:
            for item in hardcoded_found:
                self.log_test("Hardcoded Keys", False, f"Found in {item['file']}:{item['line']}")
            return False
        else:
            self.log_test("Hardcoded Keys", True, "No hardcoded API keys found")
            return True
    
    async def run_all_tests(self):
        """Run all tests"""
        print("="*70)
        print("COMPREHENSIVE TEST SUITE")
        print("="*70)
        
        tests = [
            ("Environment Setup", self.test_environment_setup, False),
            ("Backend Imports", self.test_backend_imports, True),
            ("Port Retrieval", self.test_port_retrieval, True),
            ("LangGraph Workflow", self.test_langgraph_workflow, True),
            ("API Endpoints", self.test_api_endpoints, True),
            ("File Structure", self.test_file_structure, False),
            ("No Hardcoded Keys", self.test_no_hardcoded_keys, False),
        ]
        
        for test_name, test_func, is_async in tests:
            try:
                if is_async:
                    await test_func()
                else:
                    test_func()
            except Exception as e:
                self.log_test(test_name, False, f"Test crashed: {str(e)[:100]}")
                traceback.print_exc()
        
        # Print summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"✅ Passed: {len(self.results['passed'])}")
        print(f"❌ Failed: {len(self.results['failed'])}")
        print(f"⚠️  Warnings: {len(self.results['warnings'])}")
        
        if self.results['failed']:
            print("\nFailed Tests:")
            for item in self.results['failed']:
                print(f"  - {item['test']}: {item['message']}")
        
        if self.results['warnings']:
            print("\nWarnings:")
            for item in self.results['warnings']:
                print(f"  - {item['test']}: {item['message']}")
        
        print("="*70)
        
        return len(self.results['failed']) == 0

import re

async def main():
    suite = ComprehensiveTestSuite()
    success = await suite.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTests interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        traceback.print_exc()
        sys.exit(1)

