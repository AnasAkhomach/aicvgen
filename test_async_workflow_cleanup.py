#!/usr/bin/env python3
"""
Test to verify async workflow execution and agent cleanup.
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_async_workflow_cleanup():
    """Test async workflow execution and agent cleanup."""
    print("=== Async Workflow Cleanup Test ===")
    
    try:
        from src.orchestration.workflow_definitions import WorkflowBuilder, WorkflowType
        from src.orchestration.agent_orchestrator import get_agent_orchestrator
        
        # Get orchestrator and workflow builder
        orchestrator = get_agent_orchestrator()
        workflow_builder = WorkflowBuilder(orchestrator)
        
        print("\n1. Initial agent state:")
        lifecycle_manager = orchestrator.lifecycle_manager
        cv_parser_pool = lifecycle_manager._pools.get("cv_parser", [])
        print(f"   cv_parser pool size: {len(cv_parser_pool)}")
        for i, agent in enumerate(cv_parser_pool):
            print(f"   Agent {i}: {agent.state} - Available: {agent.is_available()}")
        
        # Trace cleanup calls
        original_cleanup = orchestrator.cleanup_task_agents
        cleanup_calls = []
        
        def traced_cleanup(tasks):
            cleanup_calls.append(f"cleanup_task_agents called with {len(tasks)} tasks")
            print(f"🔍 cleanup_task_agents called with {len(tasks)} tasks")
            for i, task in enumerate(tasks):
                managed_agent = task.metadata.get("managed_agent")
                if managed_agent:
                    print(f"   Task {i}: Agent {managed_agent.config.agent_type} in state {managed_agent.state}")
            result = original_cleanup(tasks)
            print(f"🔍 cleanup_task_agents completed")
            return result
        
        orchestrator.cleanup_task_agents = traced_cleanup
        
        # Trace return_agent calls
        original_return_agent = orchestrator.return_agent
        return_agent_calls = []
        
        def traced_return_agent(agent):
            return_agent_calls.append(f"return_agent called for {agent.config.agent_type} in state {agent.state}")
            print(f"🔍 return_agent called for {agent.config.agent_type} in state {agent.state}")
            result = original_return_agent(agent)
            print(f"🔍 Agent state after return_agent: {agent.state}")
            return result
        
        orchestrator.return_agent = traced_return_agent
        
        print("\n2. Executing workflow asynchronously...")
        
        # Prepare input data
        input_data = {
            "personal_info": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "+1-555-0123"
            },
            "experience": [
                {
                    "title": "Software Engineer",
                    "company": "Tech Corp",
                    "duration": "2020-2023",
                    "description": "Developed web applications"
                }
            ],
            "job_description": "Looking for a senior software engineer position"
        }
        
        # Execute workflow
        try:
            result = await workflow_builder.execute_workflow(
                WorkflowType.JOB_TAILORED_CV,
                input_data,
                session_id="test_session_async"
            )
            
            print(f"\n3. Workflow execution result:")
            print(f"   Success: {result.success}")
            print(f"   Completed tasks: {len(result.completed_tasks)}")
            print(f"   Failed tasks: {len(result.failed_tasks)}")
            if not result.success:
                print(f"   Error: {result.error_summary}")
            
        except Exception as e:
            print(f"   ✗ Workflow execution failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n4. Final agent state:")
        cv_parser_pool = lifecycle_manager._pools.get("cv_parser", [])
        print(f"   cv_parser pool size: {len(cv_parser_pool)}")
        for i, agent in enumerate(cv_parser_pool):
            print(f"   Agent {i}: {agent.state} - Available: {agent.is_available()}")
        
        print("\n5. Cleanup trace results:")
        print(f"   cleanup_task_agents calls: {len(cleanup_calls)}")
        for call in cleanup_calls:
            print(f"   {call}")
        
        print(f"   return_agent calls: {len(return_agent_calls)}")
        for call in return_agent_calls:
            print(f"   {call}")
        
        # Restore original methods
        orchestrator.cleanup_task_agents = original_cleanup
        orchestrator.return_agent = original_return_agent
        
        print("\n=== Test completed ===")
        
        # Return success if cleanup was called
        return len(cleanup_calls) > 0
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the async test."""
    try:
        success = asyncio.run(test_async_workflow_cleanup())
        print(f"\nAsync Workflow Cleanup Test: {'✓' if success else '✗'}")
        return success
    except Exception as e:
        print(f"Failed to run async test: {e}")
        return False

if __name__ == "__main__":
    main()