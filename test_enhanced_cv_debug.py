#!/usr/bin/env python3
"""
Detailed test to debug EnhancedCVIntegration orchestrator issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_cv_orchestrator():
    """Test the EnhancedCVIntegration orchestrator in detail."""
    print("=== Testing EnhancedCVIntegration Orchestrator ===")
    
    try:
        # Step 1: Create direct orchestrator
        print("\n1. Creating direct orchestrator...")
        from src.orchestration.agent_orchestrator import get_agent_orchestrator
        
        direct_orchestrator = get_agent_orchestrator()
        print(f"Direct orchestrator ID: {id(direct_orchestrator)}")
        
        # Check lifecycle manager
        direct_manager = direct_orchestrator.lifecycle_manager
        print(f"Direct manager ID: {id(direct_manager)}")
        
        # Check agent pools
        direct_stats = direct_manager.get_statistics()
        direct_pools = direct_stats.get('pool_statistics', {})
        print(f"Direct pools: {list(direct_pools.keys())}")
        
        # Step 2: Create EnhancedCVIntegration
        print("\n2. Creating EnhancedCVIntegration...")
        from src.integration.enhanced_cv_system import EnhancedCVIntegration
        
        enhanced_cv = EnhancedCVIntegration()
        cv_orchestrator = enhanced_cv.get_orchestrator()
        print(f"CV orchestrator ID: {id(cv_orchestrator)}")
        print(f"Same as direct: {cv_orchestrator is direct_orchestrator}")
        
        # Check CV lifecycle manager
        cv_manager = cv_orchestrator.lifecycle_manager
        print(f"CV manager ID: {id(cv_manager)}")
        print(f"Same manager: {cv_manager is direct_manager}")
        
        # Check CV agent pools
        cv_stats = cv_manager.get_statistics()
        cv_pools = cv_stats.get('pool_statistics', {})
        print(f"CV pools: {list(cv_pools.keys())}")
        
        # Step 3: Compare pool contents
        print("\n3. Comparing pool contents...")
        if 'cv_parser' in direct_pools:
            print(f"Direct cv_parser pool: {direct_pools['cv_parser']}")
        else:
            print("Direct orchestrator missing cv_parser pool")
            
        if 'cv_parser' in cv_pools:
            print(f"CV cv_parser pool: {cv_pools['cv_parser']}")
        else:
            print("CV orchestrator missing cv_parser pool")
        
        # Step 4: Test agent creation
        print("\n4. Testing agent creation...")
        from src.agents.agent_base import AgentExecutionContext
        
        context = AgentExecutionContext(session_id='test', item_id='test')
        
        # Test direct orchestrator
        try:
            direct_task = direct_orchestrator.create_task('cv_parser', context)
            print(f"✓ Direct task created: {direct_task.id}")
        except Exception as e:
            print(f"✗ Direct task failed: {e}")
        
        # Test CV orchestrator
        try:
            cv_task = cv_orchestrator.create_task('cv_parser', context)
            print(f"✓ CV task created: {cv_task.id}")
        except Exception as e:
            print(f"✗ CV task failed: {e}")
        
        # Step 5: Check workflow builder
        print("\n5. Testing workflow builder...")
        from src.orchestration.workflow_definitions import get_workflow_builder
        
        builder = get_workflow_builder()
        print(f"Builder orchestrator before: {id(builder.orchestrator) if builder.orchestrator else 'None'}")
        
        # Trigger orchestrator creation in builder
        from src.orchestration.workflow_definitions import WorkflowType
        workflow = builder.get_workflow(WorkflowType.JOB_TAILORED_CV)
        print(f"Workflow found: {workflow is not None}")
        
        # Check if builder gets orchestrator when needed
        if not builder.orchestrator:
            print("Builder will create orchestrator on demand")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in enhanced CV test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_workflow_execution_path():
    """Test the actual workflow execution path."""
    print("\n=== Testing Workflow Execution Path ===")
    
    try:
        # Create enhanced CV system
        from src.integration.enhanced_cv_system import EnhancedCVIntegration
        from src.orchestration.workflow_definitions import WorkflowType
        
        enhanced_cv = EnhancedCVIntegration()
        
        # Get the workflow builder from enhanced CV
        builder = enhanced_cv._workflow_builder
        print(f"Enhanced CV builder: {id(builder) if builder else 'None'}")
        
        if builder:
            print(f"Builder orchestrator: {id(builder.orchestrator) if builder.orchestrator else 'None'}")
            
            # Check if builder orchestrator has cv_parser
            if builder.orchestrator:
                manager = builder.orchestrator.lifecycle_manager
                stats = manager.get_statistics()
                pools = stats.get('pool_statistics', {})
                print(f"Builder pools: {list(pools.keys())}")
                
                if 'cv_parser' in pools:
                    print(f"✓ Builder has cv_parser pool: {pools['cv_parser']}")
                else:
                    print("✗ Builder missing cv_parser pool")
        
        # Test workflow execution
        print("\nTesting workflow execution...")
        input_data = {
            "personal_info": {"name": "Test User"},
            "experience": [{"title": "Test Job"}],
            "job_description": {"description": "Test job description"}
        }
        
        # This should trigger the workflow builder to create an orchestrator
        try:
            import asyncio
            result = asyncio.run(enhanced_cv.execute_workflow(
                WorkflowType.JOB_TAILORED_CV,
                input_data,
                'test_session'
            ))
            print(f"✓ Workflow executed: {result['success']}")
            if not result['success']:
                print(f"Errors: {result['errors']}")
        except Exception as e:
            print(f"✗ Workflow execution failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in workflow execution test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing EnhancedCVIntegration Orchestrator Issue...\n")
    
    # Run tests
    orchestrator_ok = test_enhanced_cv_orchestrator()
    workflow_ok = test_workflow_execution_path()
    
    print("\n=== Results ===")
    print(f"Orchestrator Test: {'✓' if orchestrator_ok else '✗'}")
    print(f"Workflow Test: {'✓' if workflow_ok else '✗'}")
    
    if orchestrator_ok and workflow_ok:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed.")