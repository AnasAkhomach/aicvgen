#!/usr/bin/env python3
"""Test script to validate dependency injection refactoring."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.container import get_container
from src.orchestration.graphs.main_graph import create_cv_workflow_graph_with_di
from src.orchestration.state import GlobalState, create_global_state
from src.models.cv_models import JobDescriptionData, StructuredCV

def test_dependency_injection():
    """Test that dependency injection is working correctly."""
    print("Testing dependency injection refactoring...")
    
    try:
        # Initialize DI container
        container = get_container()
        
        # Create workflow graph with dependency injection
        session_id = "test_session_123"
        workflow_graph = create_cv_workflow_graph_with_di(container, session_id)
        
        print("✓ Successfully created workflow graph with dependency injection")
        
        # Verify that the graph wrapper has the expected methods
        assert hasattr(workflow_graph, 'invoke'), "Graph wrapper missing 'invoke' method"
        assert hasattr(workflow_graph, 'trigger_workflow_step'), "Graph wrapper missing 'trigger_workflow_step' method"
        assert hasattr(workflow_graph, 'session_id'), "Graph wrapper missing 'session_id' attribute"
        assert workflow_graph.session_id == session_id, "Session ID not set correctly"
        
        print("✓ Graph wrapper has all expected methods and attributes")
        
        # Test that we can create a basic state (without actually running the workflow)
        test_state = create_global_state(
            cv_text="Test CV content",
            session_id="test_session",
            structured_cv=StructuredCV(),
            job_description_data=JobDescriptionData(
                raw_text="Test job description content"
            )
        )
        
        print("✓ Successfully created test state")
        
        print("\n🎉 All dependency injection tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dependency_injection()
    sys.exit(0 if success else 1)