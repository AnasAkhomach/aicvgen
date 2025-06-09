#!/usr/bin/env python3
"""
Comprehensive test script to debug CV generation workflow.
This script tests each component step by step to identify where the issue occurs.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Set environment variables
os.environ["PYTHONPATH"] = str(project_root / "src")

def test_imports():
    """Test if all required modules can be imported."""
    print("=== Testing Imports ===")
    try:
        from src.integration.enhanced_cv_system import EnhancedCVIntegration
        print("✓ EnhancedCVIntegration imported successfully")
        
        from src.orchestration.workflow_definitions import WorkflowType
        print("✓ WorkflowType imported successfully")
        
        from src.orchestration.agent_orchestrator import AgentOrchestrator
        print("✓ AgentOrchestrator imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_cv_system_initialization():
    """Test CV system initialization."""
    print("\n=== Testing CV System Initialization ===")
    try:
        from src.integration.enhanced_cv_system import EnhancedCVIntegration
        cv_system = EnhancedCVIntegration()
        print("✓ CV system initialized successfully")
        return cv_system
    except Exception as e:
        print(f"✗ CV system initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_workflow_execution(cv_system):
    """Test workflow execution with sample data."""
    print("\n=== Testing Workflow Execution ===")
    
    # Sample data
    personal_info = {
        "name": "John Doe",
        "email": "john.doe@email.com",
        "phone": "+1-555-0123",
        "location": "New York, NY"
    }
    
    experience = [
        {
            "title": "Software Engineer",
            "company": "Tech Corp",
            "duration": "2020-2023",
            "responsibilities": [
                "Developed web applications using Python and React",
                "Collaborated with cross-functional teams",
                "Implemented automated testing procedures"
            ]
        }
    ]
    
    job_description = """
    We are looking for a Senior Software Engineer to join our team.
    Requirements:
    - 3+ years of Python experience
    - Experience with web frameworks
    - Strong problem-solving skills
    - Team collaboration experience
    """
    
    try:
        print("Input data:")
        print(f"  Personal Info: {personal_info}")
        print(f"  Experience: {len(experience)} items")
        print(f"  Job Description: {len(job_description)} characters")
        
        result = await cv_system.generate_job_tailored_cv(
            personal_info=personal_info,
            experience=experience,
            job_description=job_description
        )
        
        print("\nWorkflow execution result:")
        print(f"  Success: {result.get('success', False)}")
        print(f"  Results type: {type(result.get('results', {}))}")
        print(f"  Results content: {result.get('results', {})}")
        print(f"  Metadata: {result.get('metadata', {})}")
        print(f"  Processing time: {result.get('processing_time', 0)}")
        print(f"  Errors: {result.get('errors', [])}")
        
        return result
        
    except Exception as e:
        print(f"✗ Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_orchestrator_directly():
    """Test the orchestrator directly to see if it's working."""
    print("\n=== Testing Orchestrator Directly ===")
    
    try:
        from src.orchestration.agent_orchestrator import AgentOrchestrator
        from src.orchestration.workflow_definitions import WorkflowBuilder, WorkflowType
        import asyncio
        
        # Create orchestrator
        orchestrator = AgentOrchestrator()
        print("✓ Orchestrator created")
        
        # Create workflow builder
        workflow_builder = WorkflowBuilder()
        print("✓ Workflow builder created")
        
        # Test workflow execution
        input_data = {
            "personal_info": {"name": "Test User"},
            "experience": [{"title": "Test Role"}],
            "job_description": "Test job description"
        }
        
        result = asyncio.run(workflow_builder.execute_workflow(WorkflowType.JOB_TAILORED_CV, input_data))
        print(f"✓ Workflow executed, result: {result}")
        
        return result
        
    except Exception as e:
        print(f"✗ Direct orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_agent_creation():
    """Test if agents can be created and executed."""
    print("\n=== Testing Agent Creation ===")
    
    try:
        from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
        from src.agents.agent_base import AgentExecutionContext
        from src.models.data_models import ContentType
        
        # Create agent
        agent = EnhancedContentWriterAgent(
            name="TestAgent",
            content_type=ContentType.EXPERIENCE
        )
        print("✓ Agent created successfully")
        
        # Create context
        context = AgentExecutionContext(
            session_id="test-session",
            input_data={
                "experience": [{"title": "Test Role", "company": "Test Company"}],
                "job_description": "Test job description"
            }
        )
        print("✓ Context created successfully")
        
        # Test agent execution
        result = await agent.run_async(context.input_data, context)
        print(f"✓ Agent executed, result: {result}")
        
        return result
        
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Run all tests."""
    print("Starting comprehensive CV workflow debugging...\n")
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed. Cannot proceed.")
        return
    
    # Test 2: CV System Initialization
    cv_system = test_cv_system_initialization()
    if not cv_system:
        print("\n❌ CV system initialization failed. Cannot proceed.")
        return
    
    # Test 3: Workflow Execution
    workflow_result = await test_workflow_execution(cv_system)
    
    # Test 4: Direct Orchestrator Test
    orchestrator_result = test_orchestrator_directly()
    
    # Test 5: Agent Creation and Execution
    agent_result = await test_agent_creation()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Workflow result success: {workflow_result.get('success', False) if workflow_result else False}")
    print(f"Orchestrator result success: {orchestrator_result.success if orchestrator_result else False}")
    print(f"Agent result success: {agent_result.success if agent_result else False}")
    
    if workflow_result and workflow_result.get('success'):
        print("\n✅ CV generation workflow is working!")
    else:
        print("\n❌ CV generation workflow has issues that need to be resolved.")
        
        # Additional debugging info
        print("\n=== Debugging Information ===")
        if workflow_result:
            print(f"Workflow errors: {workflow_result.get('errors', [])}")
        if orchestrator_result:
            print(f"Orchestrator success: {orchestrator_result.success}")
            print(f"Orchestrator completed tasks: {len(orchestrator_result.completed_tasks)}")
            print(f"Orchestrator failed tasks: {len(orchestrator_result.failed_tasks)}")
            print(f"Orchestrator task results: {orchestrator_result.task_results}")
            if orchestrator_result.error_summary:
                print(f"Orchestrator error summary: {orchestrator_result.error_summary}")
        if agent_result and hasattr(agent_result, 'error_details'):
            print(f"Agent errors: {agent_result.error_details}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())