#!/usr/bin/env python3
"""
Test script to debug cv_parser agent creation issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.orchestration.agent_orchestrator import get_agent_orchestrator
from src.agents.agent_base import AgentExecutionContext

def test_cv_parser_creation():
    """Test if cv_parser agent can be created through orchestrator."""
    print("=== Testing CV Parser Agent Creation ===")
    
    try:
        # Get orchestrator instance
        orchestrator = get_agent_orchestrator()
        print(f"Orchestrator created: {orchestrator}")
        
        # Create execution context
        context = AgentExecutionContext(
            session_id='test_session',
            item_id='test_item'
        )
        print(f"Context created: {context}")
        
        # Try to create cv_parser task
        print("\nAttempting to create cv_parser task...")
        task = orchestrator.create_task('cv_parser', context)
        print(f"✓ Task created successfully: {task.id}")
        print(f"  Agent type: {task.agent_type}")
        print(f"  Agent instance: {task.agent_instance}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating task: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_registry():
    """Test if cv_parser is in the agent registry."""
    print("\n=== Testing Agent Registry ===")
    
    try:
        from src.agents.specialized_agents import get_agent, AGENT_REGISTRY
        
        print(f"Available agents: {list(AGENT_REGISTRY.keys())}")
        
        if 'cv_parser' in AGENT_REGISTRY:
            print("✓ cv_parser found in registry")
            
            # Test agent creation
            agent = get_agent('cv_parser')
            print(f"✓ Agent created: {agent}")
            return True
        else:
            print("✗ cv_parser not found in registry")
            return False
            
    except Exception as e:
        print(f"✗ Error testing registry: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lifecycle_manager():
    """Test if lifecycle manager has cv_parser registered."""
    print("\n=== Testing Lifecycle Manager ===")
    
    try:
        from src.core.agent_lifecycle_manager import get_agent_lifecycle_manager
        
        manager = get_agent_lifecycle_manager()
        print(f"Lifecycle manager: {manager}")
        
        # Check if cv_parser is registered
        stats = manager.get_statistics()
        print(f"Manager statistics: {stats}")
        
        # Try to get cv_parser agent
        print("Attempting to get cv_parser agent from lifecycle manager...")
        agent = manager.get_agent('cv_parser')
        if agent:
            print(f"✓ cv_parser agent retrieved: {agent}")
            return True
        else:
            print("✗ cv_parser agent not available from lifecycle manager")
            print("This suggests cv_parser is not registered with the lifecycle manager")
            return False
            
    except Exception as e:
        print(f"✗ Error testing lifecycle manager: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting CV Parser Debug Tests...\n")
    
    # Run tests
    registry_ok = test_agent_registry()
    lifecycle_ok = test_lifecycle_manager()
    orchestrator_ok = test_cv_parser_creation()
    
    print("\n=== Test Results ===")
    print(f"Agent Registry: {'✓' if registry_ok else '✗'}")
    print(f"Lifecycle Manager: {'✓' if lifecycle_ok else '✗'}")
    print(f"Orchestrator: {'✓' if orchestrator_ok else '✗'}")
    
    if all([registry_ok, lifecycle_ok, orchestrator_ok]):
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")