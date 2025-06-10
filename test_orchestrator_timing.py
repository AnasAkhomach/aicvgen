#!/usr/bin/env python3
"""
Test script to check orchestrator initialization timing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_orchestrator_timing():
    """Test the timing of orchestrator initialization."""
    print("=== Testing Orchestrator Initialization Timing ===")
    
    try:
        # Test 1: Direct orchestrator creation
        print("\n1. Creating orchestrator directly...")
        from src.orchestration.agent_orchestrator import get_agent_orchestrator
        
        orchestrator1 = get_agent_orchestrator()
        print(f"Orchestrator 1 ID: {id(orchestrator1)}")
        
        # Check if cv_parser is available
        from src.agents.agent_base import AgentExecutionContext
        context = AgentExecutionContext(session_id='test1', item_id='test1')
        
        try:
            task1 = orchestrator1.create_task('cv_parser', context)
            print(f"✓ Task 1 created: {task1.id}")
        except Exception as e:
            print(f"✗ Task 1 failed: {e}")
        
        # Test 2: Get orchestrator again
        print("\n2. Getting orchestrator again...")
        orchestrator2 = get_agent_orchestrator()
        print(f"Orchestrator 2 ID: {id(orchestrator2)}")
        print(f"Same instance: {orchestrator1 is orchestrator2}")
        
        # Test 3: Create orchestrator with session
        print("\n3. Creating orchestrator with session...")
        orchestrator3 = get_agent_orchestrator('test_session')
        print(f"Orchestrator 3 ID: {id(orchestrator3)}")
        print(f"Same as 1: {orchestrator1 is orchestrator3}")
        print(f"Same as 2: {orchestrator2 is orchestrator3}")
        
        try:
            context3 = AgentExecutionContext(session_id='test3', item_id='test3')
            task3 = orchestrator3.create_task('cv_parser', context3)
            print(f"✓ Task 3 created: {task3.id}")
        except Exception as e:
            print(f"✗ Task 3 failed: {e}")
        
        # Test 4: Test workflow builder orchestrator
        print("\n4. Testing workflow builder orchestrator...")
        from src.orchestration.workflow_definitions import get_workflow_builder
        
        builder = get_workflow_builder()
        print(f"Builder orchestrator: {id(builder.orchestrator) if builder.orchestrator else 'None'}")
        
        if builder.orchestrator:
            print(f"Builder orchestrator same as 1: {builder.orchestrator is orchestrator1}")
            try:
                context4 = AgentExecutionContext(session_id='test4', item_id='test4')
                task4 = builder.orchestrator.create_task('cv_parser', context4)
                print(f"✓ Builder task created: {task4.id}")
            except Exception as e:
                print(f"✗ Builder task failed: {e}")
        else:
            print("Builder has no orchestrator initially")
        
        # Test 5: Test EnhancedCVIntegration orchestrator
        print("\n5. Testing EnhancedCVIntegration orchestrator...")
        from src.integration.enhanced_cv_system import EnhancedCVIntegration
        
        enhanced_cv = EnhancedCVIntegration()
        cv_orchestrator = enhanced_cv.get_orchestrator()
        print(f"CV orchestrator: {id(cv_orchestrator) if cv_orchestrator else 'None'}")
        
        if cv_orchestrator:
            print(f"CV orchestrator same as 1: {cv_orchestrator is orchestrator1}")
            try:
                context5 = AgentExecutionContext(session_id='test5', item_id='test5')
                task5 = cv_orchestrator.create_task('cv_parser', context5)
                print(f"✓ CV task created: {task5.id}")
            except Exception as e:
                print(f"✗ CV task failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in timing test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Orchestrator Initialization Timing...\n")
    
    success = test_orchestrator_timing()
    
    print("\n=== Results ===")
    if success:
        print("🎉 Timing test completed successfully!")
    else:
        print("❌ Timing test failed.")