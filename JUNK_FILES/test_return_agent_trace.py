#!/usr/bin/env python3
"""
Test to trace when return_agent is called during task execution.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_return_agent_calls():
    """Test when return_agent is actually called."""
    print("=== Testing return_agent Calls ===")
    
    try:
        from src.core.agent_lifecycle_manager import get_agent_lifecycle_manager
        from src.orchestration.agent_orchestrator import get_agent_orchestrator
        from src.agents.agent_base import AgentExecutionContext
        
        # Get orchestrator and lifecycle manager
        orchestrator = get_agent_orchestrator()
        manager = orchestrator.lifecycle_manager
        
        # Monkey patch return_agent to trace calls
        original_return_agent = manager.return_agent
        return_agent_calls = []
        
        def traced_return_agent(agent):
            return_agent_calls.append(f"return_agent called for agent in state: {agent.state}")
            print(f"🔍 return_agent called for agent in state: {agent.state}")
            result = original_return_agent(agent)
            print(f"🔍 Agent state after return_agent: {agent.state}")
            return result
        
        manager.return_agent = traced_return_agent
        
        # Also trace orchestrator's return_agent
        original_orch_return_agent = orchestrator.return_agent
        orch_return_agent_calls = []
        
        def traced_orch_return_agent(agent):
            orch_return_agent_calls.append(f"orchestrator.return_agent called for agent in state: {agent.state}")
            print(f"🔍 orchestrator.return_agent called for agent in state: {agent.state}")
            result = original_orch_return_agent(agent)
            print(f"🔍 Agent state after orchestrator.return_agent: {agent.state}")
            return result
        
        orchestrator.return_agent = traced_orch_return_agent
        
        # Also trace cleanup_task_agents
        original_cleanup = orchestrator.cleanup_task_agents
        cleanup_calls = []
        
        def traced_cleanup(tasks):
            cleanup_calls.append(f"cleanup_task_agents called with {len(tasks)} tasks")
            print(f"🔍 cleanup_task_agents called with {len(tasks)} tasks")
            for task in tasks:
                managed_agent = task.metadata.get("managed_agent")
                if managed_agent:
                    print(f"🔍   Task {task.id} has managed_agent in state: {managed_agent.state}")
                else:
                    print(f"🔍   Task {task.id} has no managed_agent")
            result = original_cleanup(tasks)
            print(f"🔍 cleanup_task_agents completed")
            return result
        
        orchestrator.cleanup_task_agents = traced_cleanup
        
        print("\n1. Creating task...")
        context = AgentExecutionContext(session_id='test', item_id='test1')
        task = orchestrator.create_task('cv_parser', context)
        
        managed_agent = task.metadata.get('managed_agent')
        print(f"Task created with agent in state: {managed_agent.state}")
        
        print("\n2. Manually calling return_agent...")
        orchestrator.return_agent(managed_agent)
        
        print("\n3. Manually calling cleanup_task_agents...")
        orchestrator.cleanup_task_agents([task])
        
        print("\n4. Summary of calls:")
        print(f"return_agent calls: {len(return_agent_calls)}")
        for call in return_agent_calls:
            print(f"  - {call}")
        
        print(f"orchestrator.return_agent calls: {len(orch_return_agent_calls)}")
        for call in orch_return_agent_calls:
            print(f"  - {call}")
        
        print(f"cleanup_task_agents calls: {len(cleanup_calls)}")
        for call in cleanup_calls:
            print(f"  - {call}")
        
        # Restore original methods
        manager.return_agent = original_return_agent
        orchestrator.return_agent = original_orch_return_agent
        orchestrator.cleanup_task_agents = original_cleanup
        
        return True
        
    except Exception as e:
        print(f"✗ Error in return_agent trace test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_workflow_execution_cleanup():
    """Test if workflow execution calls cleanup properly."""
    print("\n=== Testing Workflow Execution Cleanup ===")
    
    try:
        from src.core.agent_lifecycle_manager import get_agent_lifecycle_manager
        from src.orchestration.agent_orchestrator import get_agent_orchestrator
        from src.orchestration.workflow_definitions import get_workflow_builder
        from src.agents.agent_base import AgentExecutionContext
        
        # Get orchestrator and workflow builder
        orchestrator = get_agent_orchestrator()
        workflow_builder = get_workflow_builder(orchestrator)
        
        # Trace cleanup calls during workflow execution
        original_cleanup = orchestrator.cleanup_task_agents
        cleanup_calls = []
        
        def traced_cleanup(tasks):
            cleanup_calls.append(f"cleanup_task_agents called with {len(tasks)} tasks during workflow")
            print(f"🔍 WORKFLOW: cleanup_task_agents called with {len(tasks)} tasks")
            for task in tasks:
                managed_agent = task.metadata.get("managed_agent")
                if managed_agent:
                    print(f"🔍   WORKFLOW: Task {task.id} has managed_agent in state: {managed_agent.state}")
            result = original_cleanup(tasks)
            print(f"🔍 WORKFLOW: cleanup_task_agents completed")
            return result
        
        orchestrator.cleanup_task_agents = traced_cleanup
        
        print("\nExecuting CV generation workflow...")
        try:
            import asyncio
            
            # Create input data for workflow execution
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
                "job_description": "Looking for a senior software engineer with Python experience"
            }
            
            # Execute the workflow asynchronously
            from src.orchestration.workflow_definitions import WorkflowType
            
            async def run_workflow():
                return await workflow_builder.execute_workflow(
                    WorkflowType.JOB_TAILORED_CV, 
                    input_data,
                    session_id="test_session"
                )
            
            result = asyncio.run(run_workflow())
            print(f"Workflow result: {result.success if result else 'None'}")
            
        except Exception as e:
            print(f"Workflow execution failed: {e}")
            # This is expected since we don't have real CV content
        
        print(f"\nWorkflow cleanup calls: {len(cleanup_calls)}")
        for call in cleanup_calls:
            print(f"  - {call}")
        
        # Restore original method
        orchestrator.cleanup_task_agents = original_cleanup
        
        return True
        
    except Exception as e:
        print(f"✗ Error in workflow cleanup test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Tracing return_agent Calls...\n")
    
    # Run tests
    return_agent_ok = test_return_agent_calls()
    workflow_ok = test_workflow_execution_cleanup()
    
    print("\n=== Results ===")
    print(f"Return Agent Trace Test: {'✓' if return_agent_ok else '✗'}")
    print(f"Workflow Cleanup Test: {'✓' if workflow_ok else '✗'}")
    
    if return_agent_ok and workflow_ok:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed.")