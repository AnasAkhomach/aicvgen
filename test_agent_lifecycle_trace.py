#!/usr/bin/env python3
"""
Comprehensive test to trace agent lifecycle and identify the root cause.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_complete_agent_lifecycle():
    """Test the complete agent lifecycle from creation to reuse."""
    print("=== Testing Complete Agent Lifecycle ===")
    
    try:
        from src.core.agent_lifecycle_manager import get_agent_lifecycle_manager, AgentState
        from src.orchestration.agent_orchestrator import get_agent_orchestrator
        from src.agents.agent_base import AgentExecutionContext
        
        # Get orchestrator and lifecycle manager
        orchestrator = get_agent_orchestrator()
        manager = orchestrator.lifecycle_manager
        
        print("\n1. Initial state:")
        stats = manager.get_statistics()
        cv_parser_stats = stats.get('pool_statistics', {}).get('cv_parser', {})
        print(f"CV Parser pool: {cv_parser_stats}")
        
        # Get all agents in the pool
        pool = manager._pools.get('cv_parser', [])
        print(f"Agents in pool: {len(pool)}")
        for i, agent in enumerate(pool):
            print(f"  Agent {i}: state={agent.state}, available={agent.is_available()}")
        
        # Create first task
        print("\n2. Creating first task...")
        context = AgentExecutionContext(session_id='test', item_id='test1')
        
        try:
            task1 = orchestrator.create_task('cv_parser', context)
            print(f"✓ Task 1 created: {task1.id}")
            
            managed_agent1 = task1.metadata.get('managed_agent')
            print(f"Agent 1 state: {managed_agent1.state}")
            print(f"Agent 1 available: {managed_agent1.is_available()}")
            
            # Check pool state
            print(f"\nPool state after task 1 creation:")
            for i, agent in enumerate(pool):
                print(f"  Agent {i}: state={agent.state}, available={agent.is_available()}, is_agent1={agent is managed_agent1}")
            
            # Return agent manually
            print("\n3. Returning agent 1...")
            orchestrator.return_agent(managed_agent1)
            print(f"Agent 1 state after return: {managed_agent1.state}")
            print(f"Agent 1 available after return: {managed_agent1.is_available()}")
            
            # Check pool state after return
            print(f"\nPool state after agent 1 return:")
            for i, agent in enumerate(pool):
                print(f"  Agent {i}: state={agent.state}, available={agent.is_available()}, is_agent1={agent is managed_agent1}")
            
            # Try to create second task immediately
            print("\n4. Creating second task immediately...")
            context2 = AgentExecutionContext(session_id='test', item_id='test2')
            
            try:
                task2 = orchestrator.create_task('cv_parser', context2)
                print(f"✓ Task 2 created: {task2.id}")
                
                managed_agent2 = task2.metadata.get('managed_agent')
                print(f"Agent 2 state: {managed_agent2.state}")
                print(f"Same agent reused: {managed_agent1 is managed_agent2}")
                
                # Check final pool state
                print(f"\nFinal pool state:")
                for i, agent in enumerate(pool):
                    print(f"  Agent {i}: state={agent.state}, available={agent.is_available()}, is_agent1={agent is managed_agent1}, is_agent2={agent is managed_agent2}")
                
            except Exception as e:
                print(f"✗ Task 2 creation failed: {e}")
                
                # Debug: Check what get_agent returns
                print("\nDebugging get_agent call...")
                debug_agent = manager.get_agent('cv_parser')
                if debug_agent:
                    print(f"get_agent returned: state={debug_agent.state}, available={debug_agent.is_available()}")
                else:
                    print("get_agent returned None")
                
                # Check _get_available_agent
                print("\nChecking _get_available_agent...")
                pool = manager._pools.get('cv_parser', [])
                available_agent = manager._get_available_agent(pool)
                if available_agent:
                    print(f"_get_available_agent returned: state={available_agent.state}, available={available_agent.is_available()}")
                else:
                    print("_get_available_agent returned None")
                    
                    # Check why no available agent
                    print("\nAnalyzing why no available agent:")
                    for i, agent in enumerate(pool):
                        print(f"  Agent {i}: state={agent.state}, available={agent.is_available()}")
                        if not agent.is_available():
                            print(f"    -> Agent {i} not available because state is {agent.state}")
            
        except Exception as e:
            print(f"✗ Task 1 creation failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in lifecycle test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_get_available_agent_logic():
    """Test the _get_available_agent method logic."""
    print("\n=== Testing _get_available_agent Logic ===")
    
    try:
        from src.core.agent_lifecycle_manager import get_agent_lifecycle_manager
        
        manager = get_agent_lifecycle_manager()
        
        # Check the pool
        pool = manager._pools.get('cv_parser', [])
        print(f"\nPool has {len(pool)} agents:")
        
        for i, agent in enumerate(pool):
            print(f"  Agent {i}: state={agent.state}, available={agent.is_available()}")
        
        # Test _get_available_agent
        print("\nTesting _get_available_agent:")
        available_agent = manager._get_available_agent(pool)
        
        if available_agent:
            print(f"✓ Found available agent: state={available_agent.state}")
        else:
            print("✗ No available agent found")
            
            # Manual search for available agents
            print("\nManual search for available agents:")
            for i, agent in enumerate(pool):
                if agent.is_available():
                    print(f"  Agent {i} is available but not returned by _get_available_agent")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in available agent test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Tracing Agent Lifecycle Issue...\n")
    
    # Run tests
    lifecycle_ok = test_complete_agent_lifecycle()
    available_ok = test_get_available_agent_logic()
    
    print("\n=== Results ===")
    print(f"Complete Lifecycle Test: {'✓' if lifecycle_ok else '✗'}")
    print(f"Available Agent Test: {'✓' if available_ok else '✗'}")
    
    if lifecycle_ok and available_ok:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed.")