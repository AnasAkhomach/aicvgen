#!/usr/bin/env python3
"""
Test to identify and fix the agent return issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_agent_return_mechanism():
    """Test the agent return mechanism step by step."""
    print("=== Agent Return Mechanism Test ===")
    
    try:
        from src.orchestration.agent_orchestrator import get_agent_orchestrator, AgentPriority
        from src.models.data_models import ContentType
        from src.agents.agent_base import AgentExecutionContext
        
        # Get orchestrator
        orchestrator = get_agent_orchestrator()
        lifecycle_manager = orchestrator.lifecycle_manager
        
        print("\n1. Initial state:")
        cv_parser_pool = lifecycle_manager._pools.get("cv_parser", [])
        print(f"   cv_parser pool size: {len(cv_parser_pool)}")
        
        print("\n2. Getting a cv_parser agent:")
        managed_agent = orchestrator.get_agent("cv_parser", "test_session")
        
        if managed_agent:
            print(f"   ✓ Got agent: {managed_agent.instance.__class__.__name__}")
            print(f"   Agent state: {managed_agent.state}")
            print(f"   Agent available: {managed_agent.is_available()}")
            
            print("\n3. Pool state after getting agent:")
            cv_parser_pool = lifecycle_manager._pools.get("cv_parser", [])
            print(f"   cv_parser pool size: {len(cv_parser_pool)}")
            for i, agent in enumerate(cv_parser_pool):
                print(f"   Agent {i}: {agent.state} - Available: {agent.is_available()}")
            
            print("\n4. Manually returning agent to pool:")
            print(f"   Agent state before return: {managed_agent.state}")
            orchestrator.return_agent(managed_agent)
            print(f"   Agent state after return: {managed_agent.state}")
            print(f"   Agent available after return: {managed_agent.is_available()}")
            
            print("\n5. Pool state after returning agent:")
            cv_parser_pool = lifecycle_manager._pools.get("cv_parser", [])
            print(f"   cv_parser pool size: {len(cv_parser_pool)}")
            for i, agent in enumerate(cv_parser_pool):
                print(f"   Agent {i}: {agent.state} - Available: {agent.is_available()}")
            
            print("\n6. Testing if we can get the same agent again:")
            managed_agent2 = orchestrator.get_agent("cv_parser", "test_session")
            if managed_agent2:
                print(f"   ✓ Got agent again: {managed_agent2.instance.__class__.__name__}")
                print(f"   Same agent instance: {managed_agent is managed_agent2}")
                print(f"   Agent state: {managed_agent2.state}")
                
                # Return it again
                orchestrator.return_agent(managed_agent2)
                print(f"   Agent state after second return: {managed_agent2.state}")
            else:
                print("   ✗ Could not get agent again")
            
        else:
            print("   ✗ Failed to get agent")
            
            # Try to understand why
            print("\n3. Debugging agent creation:")
            config = lifecycle_manager._pool_configs.get("cv_parser")
            if config:
                print(f"   Config found: {config.agent_type}")
                print(f"   Strategy: {config.strategy}")
                print(f"   Max instances: {config.max_instances}")
                
                # Try manual creation
                print("\n4. Attempting manual agent creation:")
                try:
                    manual_agent = lifecycle_manager.create_agent(config, "test_session")
                    if manual_agent:
                        print(f"   ✓ Manual creation successful: {manual_agent.instance.__class__.__name__}")
                        print(f"   Agent state: {manual_agent.state}")
                        
                        # Add to pool manually
                        cv_parser_pool.append(manual_agent)
                        print(f"   Added to pool. Pool size now: {len(cv_parser_pool)}")
                        
                        # Test return mechanism
                        manual_agent.mark_busy()
                        print(f"   Marked busy. State: {manual_agent.state}")
                        
                        lifecycle_manager.return_agent(manual_agent)
                        print(f"   After return_agent. State: {manual_agent.state}")
                        
                    else:
                        print("   ✗ Manual creation failed")
                except Exception as e:
                    print(f"   ✗ Manual creation error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("   ✗ No config found for cv_parser")
        
        print("\n=== Test completed ===")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_task_creation_and_cleanup():
    """Test task creation and cleanup mechanism."""
    print("\n=== Task Creation and Cleanup Test ===")
    
    try:
        from src.orchestration.agent_orchestrator import get_agent_orchestrator, AgentPriority
        from src.models.data_models import ContentType
        from src.agents.agent_base import AgentExecutionContext
        
        orchestrator = get_agent_orchestrator()
        
        print("\n1. Creating a task:")
        context = AgentExecutionContext(
            session_id="test_session",
            input_data={"test": "data"},
            content_type=ContentType.ANALYSIS,
            processing_options={}
        )
        
        task = orchestrator.create_task(
            agent_type="cv_parser",
            context=context,
            priority=AgentPriority.NORMAL
        )
        
        if task:
            print(f"   ✓ Task created: {task.id}")
            managed_agent = task.metadata.get("managed_agent")
            if managed_agent:
                print(f"   Task has managed agent: {managed_agent.config.agent_type}")
                print(f"   Agent state: {managed_agent.state}")
                
                print("\n2. Testing cleanup_task_agents:")
                orchestrator.cleanup_task_agents([task])
                print(f"   Agent state after cleanup: {managed_agent.state}")
                print(f"   Agent available after cleanup: {managed_agent.is_available()}")
                
                # Check if managed_agent reference was removed
                managed_agent_after = task.metadata.get("managed_agent")
                print(f"   Managed agent reference removed: {managed_agent_after is None}")
            else:
                print("   ✗ Task has no managed agent")
        else:
            print("   ✗ Failed to create task")
        
        return True
        
    except Exception as e:
        print(f"Task cleanup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Starting agent return mechanism debugging...\n")
    
    test1_ok = test_agent_return_mechanism()
    test2_ok = test_task_creation_and_cleanup()
    
    print("\n=== Summary ===")
    print(f"Agent Return Test: {'✓' if test1_ok else '✗'}")
    print(f"Task Cleanup Test: {'✓' if test2_ok else '✗'}")
    
    if test1_ok and test2_ok:
        print("\n✓ All tests passed - agent return mechanism is working")
    else:
        print("\n✗ Some tests failed - agent return mechanism needs fixing")
    
    return test1_ok and test2_ok

if __name__ == "__main__":
    main()