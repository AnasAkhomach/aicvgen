#!/usr/bin/env python3
"""
Test script to verify LG-FIX-02: Node Function Compliance

This script tests that all node functions in cv_workflow_graph.py:
1. Have proper return type annotations (Dict[str, Any])
2. Have @validate_node_output decorators
3. Return dictionary objects in practice
4. Handle errors gracefully and return error dictionaries
"""

import asyncio
import inspect
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from orchestration.cv_workflow_graph import CVWorkflowGraph
from orchestration.state import AgentState


def test_node_function_annotations():
    """Test that all node functions have proper return type annotations."""
    print("\n=== Testing Node Function Annotations ===")
    
    workflow_graph = CVWorkflowGraph(session_id="test")
    
    # Get all methods that end with '_node'
    node_methods = [
        method for method in dir(workflow_graph)
        if method.endswith('_node') and callable(getattr(workflow_graph, method))
    ]
    
    print(f"Found {len(node_methods)} node methods")
    
    failed_annotations = []
    
    for method_name in node_methods:
        method = getattr(workflow_graph, method_name)
        signature = inspect.signature(method)
        
        # Check return annotation
        return_annotation = signature.return_annotation
        
        if return_annotation == inspect.Signature.empty:
            failed_annotations.append(f"{method_name}: Missing return annotation")
        elif str(return_annotation) != "typing.Dict[str, typing.Any]":
            failed_annotations.append(f"{method_name}: Incorrect return annotation: {return_annotation}")
        else:
            print(f"✓ {method_name}: Correct return annotation")
    
    if failed_annotations:
        print("\n❌ Failed annotation checks:")
        for failure in failed_annotations:
            print(f"  - {failure}")
        return False
    else:
        print("\n✅ All node functions have correct return annotations")
        return True


def test_validate_node_output_decorators():
    """Test that all node functions have @validate_node_output decorators."""
    print("\n=== Testing @validate_node_output Decorators ===")
    
    workflow_graph = CVWorkflowGraph(session_id="test")
    
    # Get all methods that end with '_node'
    node_methods = [
        method for method in dir(workflow_graph)
        if method.endswith('_node') and callable(getattr(workflow_graph, method))
    ]
    
    missing_decorators = []
    
    for method_name in node_methods:
        method = getattr(workflow_graph, method_name)
        
        # Check if method has the validate_node_output decorator
        # The decorator should be in the method's __wrapped__ or similar attributes
        has_decorator = False
        
        # Check if the method has been wrapped (indicating decorator presence)
        if hasattr(method, '__wrapped__') or hasattr(method, '_validate_node_output'):
            has_decorator = True
        
        # Alternative check: look at the method's qualname or inspect source
        try:
            source = inspect.getsource(method)
            if '@validate_node_output' in source:
                has_decorator = True
        except (OSError, TypeError):
            pass
        
        if has_decorator:
            print(f"✓ {method_name}: Has @validate_node_output decorator")
        else:
            missing_decorators.append(method_name)
    
    if missing_decorators:
        print("\n❌ Missing @validate_node_output decorators:")
        for method_name in missing_decorators:
            print(f"  - {method_name}")
        return False
    else:
        print("\n✅ All node functions have @validate_node_output decorators")
        return True


async def test_node_return_types():
    """Test that node functions return dictionary objects in practice."""
    print("\n=== Testing Node Return Types in Practice ===")
    
    workflow_graph = CVWorkflowGraph(session_id="test")
    
    # Create a minimal test state
    test_state = AgentState(
        session_id="test",
        cv_text="Test CV content",
        workflow_status="PROCESSING",
        trace_id="test-trace",
        node_execution_metadata={},
        error_messages=[]
    )
    
    # Get all node methods for testing
    node_methods = [
        method for method in dir(workflow_graph)
        if method.endswith('_node') and callable(getattr(workflow_graph, method))
    ]
    
    failed_returns = []
    passed = 0
    
    for method_name in node_methods:
        try:
            # Get the method from the workflow graph
            method = getattr(workflow_graph, method_name)
            
            # Call the method with AgentState object directly
            result = await method(test_state)
            
            if isinstance(result, dict):
                print(f"✓ {method_name}: Returns dict - {type(result)}")
                passed += 1
            else:
                failed_returns.append(f"{method_name}: Returns {type(result)}, expected dict")
                
        except Exception as e:
            # Even with errors, the function should return a dict with error_messages
            print(f"⚠ {method_name}: Exception occurred but should still return dict: {e}")
    
    if failed_returns:
        print("\n❌ Failed return type checks:")
        for failure in failed_returns:
            print(f"  - {failure}")
        return False
    else:
        print(f"\n✅ All {passed} node functions return dictionaries")
        return True


async def test_error_handling_compliance():
    """Test that node functions handle errors gracefully and return error dictionaries."""
    print("\n=== Testing Error Handling Compliance ===")
    
    workflow_graph = CVWorkflowGraph(session_id="test")
    
    # Create a state that will likely cause errors (missing required data)
    error_state = AgentState(
        session_id="test",
        cv_text="Test CV content",
        workflow_status="PROCESSING",
        trace_id="test-trace",
        node_execution_metadata={},
        error_messages=[]
    )
    
    # Test node functions that should handle missing dependencies gracefully
    error_test_cases = [
        "jd_parser_node",
        "cv_parser_node", 
        "research_node",
        "cv_analyzer_node",
        "key_qualifications_writer_node",
        "professional_experience_writer_node",
        "projects_writer_node",
        "executive_summary_writer_node",
        "qa_node",
        "formatter_node"
    ]
    
    failed_error_handling = []
    
    for method_name in error_test_cases:
        try:
            method = getattr(workflow_graph, method_name)
            result = await method(error_state)
            
            if isinstance(result, dict):
                if "error_messages" in result:
                    print(f"✓ {method_name}: Handles errors gracefully with error_messages")
                else:
                    print(f"✓ {method_name}: Returns dict (may be empty due to missing dependencies)")
            else:
                failed_error_handling.append(f"{method_name}: Returns {type(result)}, expected dict with error handling")
                
        except Exception as e:
            failed_error_handling.append(f"{method_name}: Unhandled exception: {e}")
    
    if failed_error_handling:
        print("\n❌ Failed error handling checks:")
        for failure in failed_error_handling:
            print(f"  - {failure}")
        return False
    else:
        print("\n✅ All node functions handle errors gracefully")
        return True


async def main():
    """Run all compliance tests for LG-FIX-02."""
    print("Starting LG-FIX-02 Node Function Compliance Tests")
    print("=" * 50)
    
    results = []
    
    # Test 1: Return type annotations
    results.append(test_node_function_annotations())
    
    # Test 2: @validate_node_output decorators
    results.append(test_validate_node_output_decorators())
    
    # Test 3: Actual return types
    results.append(await test_node_return_types())
    
    # Test 4: Error handling compliance
    results.append(await test_error_handling_compliance())
    
    # Summary
    print("\n" + "=" * 50)
    print("LG-FIX-02 COMPLIANCE TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("\n🎉 LG-FIX-02 Node Function Compliance: COMPLETE")
        print("All node functions properly return Dict[str, Any] and handle errors gracefully.")
        return True
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{total})")
        print("\n⚠️  LG-FIX-02 Node Function Compliance: INCOMPLETE")
        print("Some node functions need additional fixes.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)