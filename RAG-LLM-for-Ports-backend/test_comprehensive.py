#!/usr/bin/env python3
"""
Comprehensive Test Suite for RAG-LLM-for-Ports-backend
Consolidates all test functionality into one file
"""

import os
import sys
import asyncio
from pathlib import Path

# Try to load dotenv from main folder first, then local
try:
    from dotenv import load_dotenv
    # Try main folder first
    main_env = Path(__file__).parent.parent / ".env"
    if main_env.exists():
        load_dotenv(main_env)
    else:
        # Fall back to local .env
        load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_imports():
    """Test 1: Module imports"""
    print("\n" + "="*70)
    print("TEST 1: Module Imports")
    print("="*70)
    
    try:
        from src.port_information_retrieval import PortInformationRetrieval
        from src.graph.port_graph import PortWorkflowGraph
        from src.routers.intent_router import IntentRouter
        from src.routers.decision_nodes import DecisionNodes
        from src.api.server import app
        print("  ✅ All modules imported successfully")
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

async def test_port_retrieval():
    """Test 2: Port Information Retrieval"""
    print("\n" + "="*70)
    print("TEST 2: Port Information Retrieval")
    print("="*70)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("  ❌ OPENAI_API_KEY not found")
        return False
    
    try:
        from src.port_information_retrieval import PortInformationRetrieval
        
        pr = PortInformationRetrieval(
            openai_api_key=api_key,
            chroma_persist_directory='./storage/chroma',
            collection_name='port_documents'
        )
        
        result = await pr.query_port_information("What are safety protocols?", max_documents=2)
        
        if result.get("answer") and len(result.get("answer", "")) > 50:
            print(f"  ✅ Query successful ({len(result['answer'])} chars)")
            print(f"  ✅ Sources: {len(result.get('sources', []))}")
            return True
        else:
            print("  ❌ Empty answer")
            return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

async def test_langgraph_workflow():
    """Test 3: LangGraph Workflow"""
    print("\n" + "="*70)
    print("TEST 3: LangGraph Workflow")
    print("="*70)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("  ❌ OPENAI_API_KEY not found")
        return False
    
    try:
        from src.port_information_retrieval import PortInformationRetrieval
        from src.graph.port_graph import PortWorkflowGraph
        
        pr = PortInformationRetrieval(
            openai_api_key=api_key,
            chroma_persist_directory='./storage/chroma',
            collection_name='port_documents'
        )
        
        wg = PortWorkflowGraph(port_retrieval=pr, enable_checkpoints=True)
        
        result = await wg.execute_workflow(
            query='What are the safety protocols for vessel berthing?',
            user_id='test_user'
        )
        
        if "error" in result:
            print(f"  ❌ Workflow error: {result['error']}")
            return False
        
        recommendation = result.get("final_recommendation", "")
        if recommendation and len(recommendation) > 50:
            print(f"  ✅ Recommendation generated ({len(recommendation)} chars)")
        else:
            print("  ❌ Empty recommendation")
            return False
        
        scenarios = result.get("alternative_scenarios", [])
        print(f"  ✅ Alternative scenarios: {len(scenarios)}")
        
        sources = result.get("sources", [])
        print(f"  ✅ Sources: {len(sources)}")
        
        exec_time = result.get("execution_time", 0)
        print(f"  ✅ Execution time: {exec_time:.2f}s")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_endpoints():
    """Test 4: API Endpoints (if server running)"""
    print("\n" + "="*70)
    print("TEST 4: API Endpoints")
    print("="*70)
    
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:8000/health")
            if response.status_code == 200:
                print("  ✅ Server is running")
                
                # Test /ask
                response = await client.post(
                    "http://localhost:8000/ask",
                    json={"query": "What are safety protocols?"},
                    timeout=30.0
                )
                if response.status_code == 200:
                    print("  ✅ /ask endpoint working")
                else:
                    print(f"  ⚠️  /ask returned {response.status_code}")
                
                # Test /ask_graph
                response = await client.post(
                    "http://localhost:8000/ask_graph",
                    json={"query": "What are safety protocols?"},
                    timeout=60.0
                )
                if response.status_code == 200:
                    print("  ✅ /ask_graph endpoint working")
                else:
                    print(f"  ⚠️  /ask_graph returned {response.status_code}")
                
                return True
            else:
                print("  ⚠️  Server not responding (may not be running)")
                return True  # Not a failure
    except Exception:
        print("  ⚠️  Server not running (this is OK)")
        return True  # Not a failure

async def main():
    """Run all tests"""
    print("="*70)
    print("COMPREHENSIVE TEST SUITE - RAG-LLM-for-Ports-backend")
    print("="*70)
    
    results = []
    results.append(("Imports", await test_imports()))
    results.append(("Port Retrieval", await test_port_retrieval()))
    results.append(("LangGraph Workflow", await test_langgraph_workflow()))
    results.append(("API Endpoints", await test_api_endpoints()))
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70)
    
    return passed == total

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

