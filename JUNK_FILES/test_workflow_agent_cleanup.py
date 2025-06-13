#!/usr/bin/env python3
"""
Test to verify agent cleanup in actual workflow execution scenarios.
This test simulates the real workflow execution to identify where agents get stuck.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_workflow_agent_cleanup():
    """Test agent cleanup in workflow execution."""
    print("=== Workflow Agent Cleanup Test ===")
    
    try:
        from src.orchestration.workflow_definitions import get_workflow_builder, WorkflowType
        from src.orchestration.agent_orchestrator import get_agent_orchestrator
        from src.models.data_models import ContentType
        
        # Get orchestrator to check initial state
        orchestrator = get_agent_orchestrator()
        lifecycle_manager = orchestrator.lifecycle_manager
        
        print("\n1. Initial pool state:")
        cv_parser_pool = lifecycle_manager._pools.get("cv_parser", [])
        print(f"   cv_parser pool size: {len(cv_parser_pool)}")
        for i, agent in enumerate(cv_parser_pool):
            print(f"   Agent {i}: {agent.state} - Available: {agent.is_available()}")
        
        # Test inputs for workflow
        workflow_inputs = {
            "job_description": "Software Engineer position requiring Python and AI experience.",
            "existing_cv_content": "John Doe\nSoftware Developer\nExperience with Python, JavaScript, and machine learning.",
            "target_role": "Senior Software Engineer",
            "company_info": "Tech startup focused on AI solutions"
        }
        
        print("\n2. Executing workflow...")
        try:
            workflow_builder = get_workflow_builder(orchestrator)
            result = await workflow_builder.execute_workflow(
                workflow_type=WorkflowType.JOB_TAILORED_CV,
                input_data=workflow_inputs,
                session_id="test_cleanup_session"
            )
            print(f"   Workflow result success: {result.get('success', False)}")
            if not result.get('success', False):
                print(f"   Workflow error: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"   Workflow execution failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n3. Pool state after workflow:")
        cv_parser_pool = lifecycle_manager._pools.get("cv_parser", [])
        print(f"   cv_parser pool size: {len(cv_parser_pool)}")
        for i, agent in enumerate(cv_parser_pool):
            print(f"   Agent {i}: {agent.state} - Available: {agent.is_available()}")
        
        # Check if any agents are stuck in BUSY state
        busy_agents = [agent for agent in cv_parser_pool if agent.state.name == 'BUSY']
        if busy_agents:
            print(f"\n⚠️  WARNING: {len(busy_agents)} agents stuck in BUSY state!")
            
            # Try to manually clean them up
            print("\n4. Attempting manual cleanup of stuck agents:")
            for i, agent in enumerate(busy_agents):
                print(f"   Cleaning up agent {i}: {agent.state}")
                try:
                    orchestrator.return_agent(agent)
                    print(f"   Agent {i} after cleanup: {agent.state}")
                except Exception as e:
                    print(f"   Failed to cleanup agent {i}: {e}")
        else:
            print("\n✓ No agents stuck in BUSY state")
        
        print("\n5. Final pool state:")
        cv_parser_pool = lifecycle_manager._pools.get("cv_parser", [])
        print(f"   cv_parser pool size: {len(cv_parser_pool)}")
        for i, agent in enumerate(cv_parser_pool):
            print(f"   Agent {i}: {agent.state} - Available: {agent.is_available()}")
        
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_multiple_workflow_executions():
    """Test multiple workflow executions to see if agents accumulate in BUSY state."""
    print("\n=== Multiple Workflow Executions Test ===")
    
    try:
        from src.orchestration.workflow_definitions import get_workflow_builder, WorkflowType
        from src.orchestration.agent_orchestrator import get_agent_orchestrator
        
        orchestrator = get_agent_orchestrator()
        lifecycle_manager = orchestrator.lifecycle_manager
        
        workflow_inputs = {
            "job_description": "Data Scientist position requiring Python and ML experience.",
            "existing_cv_content": "Jane Smith\nData Analyst\nExperience with Python, R, and data visualization.",
            "target_role": "Senior Data Scientist",
            "company_info": "Healthcare analytics company"
        }
        
        num_executions = 3
        print(f"\nExecuting workflow {num_executions} times...")
        
        for i in range(num_executions):
            print(f"\n--- Execution {i+1} ---")
            
            # Check pool state before execution
            cv_parser_pool = lifecycle_manager._pools.get("cv_parser", [])
            busy_count = len([a for a in cv_parser_pool if a.state.name == 'BUSY'])
            available_count = len([a for a in cv_parser_pool if a.is_available()])
            print(f"   Before: Pool size={len(cv_parser_pool)}, Busy={busy_count}, Available={available_count}")
            
            # Execute workflow
            try:
                workflow_builder = get_workflow_builder(orchestrator)
                result = await workflow_builder.execute_workflow(
                    workflow_type=WorkflowType.JOB_TAILORED_CV,
                    input_data=workflow_inputs,
                    session_id=f"test_multi_session_{i}"
                )
                print(f"   Result: {result.get('success', False)}")
                if not result.get('success', False):
                    print(f"   Error: {result.get('error', 'Unknown')}")
            except Exception as e:
                print(f"   Exception: {e}")
            
            # Check pool state after execution
            cv_parser_pool = lifecycle_manager._pools.get("cv_parser", [])
            busy_count = len([a for a in cv_parser_pool if a.state.name == 'BUSY'])
            available_count = len([a for a in cv_parser_pool if a.is_available()])
            print(f"   After:  Pool size={len(cv_parser_pool)}, Busy={busy_count}, Available={available_count}")
        
        # Final summary
        print("\n--- Final Summary ---")
        cv_parser_pool = lifecycle_manager._pools.get("cv_parser", [])
        busy_agents = [a for a in cv_parser_pool if a.state.name == 'BUSY']
        available_agents = [a for a in cv_parser_pool if a.is_available()]
        
        print(f"Total agents: {len(cv_parser_pool)}")
        print(f"Busy agents: {len(busy_agents)}")
        print(f"Available agents: {len(available_agents)}")
        
        if busy_agents:
            print(f"\n⚠️  {len(busy_agents)} agents are stuck in BUSY state after {num_executions} executions!")
            return False
        else:
            print(f"\n✓ All agents properly returned to available state after {num_executions} executions")
            return True
        
    except Exception as e:
        print(f"Multiple executions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("Starting workflow agent cleanup testing...\n")
    
    test1_ok = await test_workflow_agent_cleanup()
    test2_ok = await test_multiple_workflow_executions()
    
    print("\n=== Summary ===")
    print(f"Single Workflow Test: {'✓' if test1_ok else '✗'}")
    print(f"Multiple Workflows Test: {'✓' if test2_ok else '✗'}")
    
    if test1_ok and test2_ok:
        print("\n✓ All tests passed - workflow agent cleanup is working correctly")
    else:
        print("\n✗ Some tests failed - workflow agent cleanup needs fixing")
    
    return test1_ok and test2_ok

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())