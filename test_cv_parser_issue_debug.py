#!/usr/bin/env python3
"""
Comprehensive test to debug the 'Cannot create task for agent type: cv_parser' issue.
This test will trace the entire agent lifecycle and identify the root cause.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.agent_lifecycle_manager import AgentLifecycleManager
from src.orchestration.agent_orchestrator import AgentOrchestrator
from src.agents.agent_base import AgentExecutionContext
from src.orchestration.agent_orchestrator import AgentPriority
from src.models.data_models import ContentType
from src.core.dependency_injection import DependencyContainer
from src.core.session_manager import SessionManager
from src.core.error_recovery import ErrorRecoveryManager
from src.core.progress_tracker import ProgressTracker
from datetime import timedelta

def test_cv_parser_issue():
    """Test the cv_parser agent issue comprehensively."""
    print("=== CV Parser Issue Debug Test ===")
    
    try:
        # Initialize dependency container
        container = DependencyContainer()
        container.register_singleton("session_manager", lambda: SessionManager())
        container.register_singleton("error_recovery", lambda: ErrorRecoveryManager())
        container.register_singleton("progress_tracker", lambda: ProgressTracker())
        
        # Initialize orchestrator
        orchestrator = AgentOrchestrator(session_id="test_session", container=container)
        
        print("\n1. Checking agent pool status:")
        lifecycle_manager = orchestrator.lifecycle_manager
        
        # Check cv_parser pool status
        if "cv_parser" in lifecycle_manager._pools:
            pool = lifecycle_manager._pools["cv_parser"]
            print(f"   cv_parser pool size: {len(pool)}")
            print(f"   cv_parser pool contents: {[str(agent.state) for agent in pool]}")
        else:
            print("   cv_parser pool not found!")
        
        print("\n2. Attempting to get cv_parser agent:")
        managed_agent = orchestrator.get_agent("cv_parser")
        if managed_agent:
            print(f"   ✓ Got cv_parser agent: {managed_agent.instance.__class__.__name__}")
            print(f"   Agent state: {managed_agent.state}")
        else:
            print("   ✗ Failed to get cv_parser agent")
            
        print("\n3. Attempting to create task:")
        context = AgentExecutionContext(
            session_id="test_session",
            user_id="test_user",
            request_id="test_request"
        )
        
        try:
            task = orchestrator.create_task(
                agent_type="cv_parser",
                context=context,
                priority=AgentPriority.HIGH,
                timeout=timedelta(minutes=5)
            )
            print(f"   ✓ Task created successfully: {task.id}")
            print(f"   Agent state after task creation: {task.metadata['managed_agent'].state}")
            
            print("\n4. Returning agent:")
            orchestrator.return_agent(task.metadata['managed_agent'])
            print(f"   Agent state after return: {task.metadata['managed_agent'].state}")
            
            print("\n5. Checking if agent is available again:")
            available_agent = orchestrator.get_agent("cv_parser")
            if available_agent:
                print(f"   ✓ Agent is available: {available_agent.state}")
            else:
                print("   ✗ Agent is not available")
                
        except Exception as e:
            print(f"   ✗ Failed to create task: {e}")
            
            # Let's check what's in the lifecycle manager
            print("\n   Debugging lifecycle manager:")
            print(f"   Registered agent types: {list(lifecycle_manager._pools.keys())}")
            
            for agent_type, pool in lifecycle_manager._pools.items():
                if agent_type == "cv_parser":
                    print(f"   {agent_type} pool:")
                    print(f"     - Pool size: {len(pool)}")
                    print(f"     - Agent states: {[str(agent.state) for agent in pool]}")
                    print(f"     - Available agents: {len([agent for agent in pool if agent.is_available()])}")
        
        print("\n=== Test completed ===")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cv_parser_issue()