#!/usr/bin/env python3
"""
Test to debug agent state management issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_agent_state_management():
    """Test agent state transitions."""
    print("=== Testing Agent State Management ===")
    
    try:
        from src.core.agent_lifecycle_manager import get_agent_lifecycle_manager, AgentState
        from src.orchestration.agent_orchestrator import get_agent_orchestrator
        from src.agents.agent_base import AgentExecutionContext
        
        # Get orchestrator and lifecycle manager
        orchestrator = get_agent_orchestrator()
        manager = orchestrator.lifecycle_manager
        
        print("\n1. Initial pool state:")
        stats = manager.get_statistics()
        cv_parser_stats = stats.get('pool_statistics', {}).get('cv_parser', {})
        print(f"CV Parser pool: {cv_parser_stats}")
        
        # Create a task
        print("\n2. Creating task...")
        context = AgentExecutionContext(session_id='test', item_id='test')
        
        try:
            task = orchestrator.create_task('cv_parser', context)
            print(f"✓ Task created: {task.id}")
            
            # Check managed agent state
            managed_agent = task.metadata.get('managed_agent')
            if managed_agent:
                print(f"Managed agent state: {managed_agent.state}")
                print(f"Managed agent available: {managed_agent.is_available()}")
            
            # Check pool state after task creation
            stats = manager.get_statistics()
            cv_parser_stats = stats.get('pool_statistics', {}).get('cv_parser', {})
            print(f"Pool after task creation: {cv_parser_stats}")
            
            # Manually return agent to pool
            print("\n3. Returning agent to pool...")
            orchestrator.return_agent(managed_agent)
            
            # Check agent state after return
            print(f"Agent state after return: {managed_agent.state}")
            print(f"Agent available after return: {managed_agent.is_available()}")
            
            # Check pool state after return
            stats = manager.get_statistics()
            cv_parser_stats = stats.get('pool_statistics', {}).get('cv_parser', {})
            print(f"Pool after agent return: {cv_parser_stats}")
            
            # Try to create another task
            print("\n4. Creating second task...")
            try:
                task2 = orchestrator.create_task('cv_parser', context)
                print(f"✓ Second task created: {task2.id}")
                
                # Check if we got the same agent
                managed_agent2 = task2.metadata.get('managed_agent')
                print(f"Same agent reused: {managed_agent is managed_agent2}")
                
            except Exception as e:
                print(f"✗ Second task failed: {e}")
            
        except Exception as e:
            print(f"✗ Task creation failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in agent state test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_state_transitions():
    """Test individual agent state transitions."""
    print("\n=== Testing Agent State Transitions ===")
    
    try:
        from src.core.agent_lifecycle_manager import AgentState, ManagedAgent, AgentPoolConfig, AgentPoolStrategy
        
        # Create a mock agent instance
        class MockAgent:
            def __init__(self):
                self.name = "test_agent"
                
        # Create a managed agent directly
        config = AgentPoolConfig(
            agent_type='test_cv_parser',
            factory=lambda: MockAgent(),
            strategy=AgentPoolStrategy.LAZY
        )
        
        agent_instance = MockAgent()
        managed_agent = ManagedAgent(
            instance=agent_instance,
            config=config,
            state=AgentState.READY
        )
        
        print(f"\n1. Initial state: {managed_agent.state}")
        print(f"   Available: {managed_agent.is_available()}")
        
        # Mark as busy
        managed_agent.mark_busy()
        print(f"\n2. After mark_busy: {managed_agent.state}")
        print(f"   Available: {managed_agent.is_available()}")
        
        # Mark as idle
        managed_agent.mark_idle()
        print(f"\n3. After mark_idle: {managed_agent.state}")
        print(f"   Available: {managed_agent.is_available()}")
        
        # Test what states are considered available
        print("\n4. Testing available states:")
        for state in AgentState:
            managed_agent.state = state
            print(f"   {state.value}: {managed_agent.is_available()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in state transition test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Agent State Management Issue...\n")
    
    # Run tests
    state_mgmt_ok = test_agent_state_management()
    transitions_ok = test_agent_state_transitions()
    
    print("\n=== Results ===")
    print(f"State Management Test: {'✓' if state_mgmt_ok else '✗'}")
    print(f"State Transitions Test: {'✓' if transitions_ok else '✗'}")
    
    if state_mgmt_ok and transitions_ok:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed.")