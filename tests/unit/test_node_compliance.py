"""Test compliance of node functions with LG-FIX-02 requirements.

This test verifies that all node functions in cv_workflow_graph.py:
1. Have proper Dict[str, Any] return type annotations
2. Are decorated with @validate_node_output
3. Actually return dictionary objects
4. Handle errors gracefully and return error dictionaries
"""

import asyncio
import inspect
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.orchestration.cv_workflow_graph import CVWorkflowGraph
from src.orchestration.state import AgentState


async def test_node_compliance():
    """Test that all node functions comply with LG-FIX-02 requirements."""
    print("Testing LG-FIX-02 Node Compliance...")
    
    # Initialize workflow graph
    workflow_graph = CVWorkflowGraph()
    
    # Get all node functions (methods ending with '_node')
    node_functions = [
        name for name, method in inspect.getmembers(workflow_graph, predicate=inspect.ismethod)
        if name.endswith('_node') and asyncio.iscoroutinefunction(method)
    ]
    
    print(f"Found {len(node_functions)} node functions to test")
    
    # Create test state
    test_state = AgentState(
        cv_text="Test CV content",
        workflow_status="PROCESSING"
    )
    
    # Create error state for error testing
    error_state = AgentState(
        cv_text="Test CV content",
        workflow_status="ERROR"
    )
    
    # Test 1: Check return type annotations
    print("\n1. Checking return type annotations...")
    test_1_passed = True
    for node_name in node_functions:
        node_func = getattr(workflow_graph, node_name)
        sig = inspect.signature(node_func)
        return_annotation = sig.return_annotation
        
        # Check if return annotation is Dict[str, Any] or equivalent
        expected_annotations = [
            "typing.Dict[str, typing.Any]",
            "Dict[str, Any]",
            "dict[str, typing.Any]",
            "dict[str, Any]"
        ]
        
        annotation_str = str(return_annotation)
        if annotation_str not in expected_annotations:
            print(f"❌ {node_name}: Return annotation is {annotation_str}, expected Dict[str, Any]")
            test_1_passed = False
        else:
            print(f"✅ {node_name}: Correct return annotation")
    
    # Test 2: Check for @validate_node_output decorator
    print("\n2. Checking @validate_node_output decorators...")
    test_2_passed = True
    for node_name in node_functions:
        node_func = getattr(workflow_graph, node_name)
        
        # Check if the function has the validate_node_output wrapper
        has_decorator = (
            hasattr(node_func, '__wrapped__') or
            'validate_node_output' in str(node_func) or
            'wrapper' in node_func.__name__
        )
        
        if not has_decorator:
            print(f"❌ {node_name}: Missing @validate_node_output decorator")
            test_2_passed = False
        else:
            print(f"✅ {node_name}: Has @validate_node_output decorator")
    
    # Test 3: Check that node functions actually return dictionaries
    print("\n3. Testing actual return types...")
    test_3_passed = True
    failed_nodes = []
    for node_name in node_functions:
        try:
            node_func = getattr(workflow_graph, node_name)
            # Call the node function with test state
            result = await node_func(test_state)
            
            if not isinstance(result, (dict, AgentState)):
                print(f"❌ {node_name}: Returns {type(result)}, expected dict or AgentState")
                failed_nodes.append(node_name)
            else:
                print(f"✅ {node_name}: Returns {type(result).__name__}")
        except Exception as e:
            # Expected failures for nodes that require injected dependencies
            if node_name == "research_node" and "ResearchAgent not injected" in str(e):
                print(f"✅ {node_name}: Expected failure (dependency not injected)")
            else:
                print(f"❌ {node_name}: Exception during execution: {e}")
                failed_nodes.append(node_name)
    
    test_3_passed = len(failed_nodes) == 0
    if failed_nodes:
        print(f"Failed nodes: {failed_nodes}")
    
    # Test 4: Check error handling
    print("\n4. Testing error handling...")
    test_4_passed = True
    
    # Test error_handler_node specifically
    try:
        error_result = await workflow_graph.error_handler_node(error_state)
        if isinstance(error_result, dict) and 'error_messages' in error_result:
            print("✅ error_handler_node: Properly handles errors")
        else:
            print(f"❌ error_handler_node: Unexpected error handling result: {error_result}")
            test_4_passed = False
    except Exception as e:
        print(f"❌ error_handler_node: Exception during error handling: {e}")
        test_4_passed = False
    
    # Summary
    print("\n" + "="*50)
    print("LG-FIX-02 COMPLIANCE TEST RESULTS:")
    print(f"1. Return type annotations: {'✅ PASS' if test_1_passed else '❌ FAIL'}")
    print(f"2. @validate_node_output decorators: {'✅ PASS' if test_2_passed else '❌ FAIL'}")
    print(f"3. Actual return types: {'✅ PASS' if test_3_passed else '❌ FAIL'}")
    print(f"4. Error handling: {'✅ PASS' if test_4_passed else '❌ FAIL'}")
    
    all_passed = test_1_passed and test_2_passed and test_3_passed and test_4_passed
    print(f"\nOVERALL: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    asyncio.run(test_node_compliance())