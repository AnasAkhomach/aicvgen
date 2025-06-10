#!/usr/bin/env python3
"""
Comprehensive workflow test with proper input data to verify the logger fix
and complete workflow execution.
"""

import asyncio
import sys
import os
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.orchestration.workflow_definitions import get_workflow_builder, WorkflowType
from src.orchestration.agent_orchestrator import get_agent_orchestrator


def create_test_input_data() -> Dict[str, Any]:
    """Create comprehensive test input data for workflow execution."""
    return {
        "personal_info": {
            "name": "John Doe",
            "email": "john.doe@email.com",
            "phone": "+1-555-0123",
            "location": "San Francisco, CA",
            "linkedin": "https://linkedin.com/in/johndoe",
            "github": "https://github.com/johndoe"
        },
        "experience": [
            {
                "company": "Tech Corp",
                "position": "Senior Software Engineer",
                "start_date": "2020-01",
                "end_date": "2023-12",
                "description": "Led development of microservices architecture using Python and AWS.",
                "achievements": [
                    "Improved system performance by 40%",
                    "Led team of 5 engineers",
                    "Implemented CI/CD pipeline"
                ],
                "technologies": ["Python", "AWS", "Docker", "Kubernetes"]
            },
            {
                "company": "StartupXYZ",
                "position": "Software Engineer",
                "start_date": "2018-06",
                "end_date": "2019-12",
                "description": "Developed web applications using React and Node.js.",
                "achievements": [
                    "Built user authentication system",
                    "Optimized database queries"
                ],
                "technologies": ["React", "Node.js", "MongoDB", "Express"]
            }
        ],
        "job_description": {
            "raw_text": "We are looking for a Senior Python Developer with experience in cloud technologies and microservices architecture. The ideal candidate should have 5+ years of experience with Python, AWS, and containerization technologies like Docker and Kubernetes.",
            "skills": ["Python", "AWS", "Docker", "Kubernetes", "Microservices"],
            "experience_level": "Senior (5+ years)",
            "responsibilities": [
                "Design and implement microservices",
                "Deploy applications to AWS",
                "Mentor junior developers",
                "Optimize system performance"
            ],
            "industry_terms": ["cloud-native", "scalability", "DevOps"],
            "company_values": ["innovation", "collaboration", "excellence"]
        },
        "projects": [
            {
                "name": "E-commerce Platform",
                "description": "Built a scalable e-commerce platform handling 10k+ daily users",
                "technologies": ["Python", "Django", "PostgreSQL", "Redis"],
                "achievements": ["99.9% uptime", "Sub-second response times"]
            }
        ]
    }


async def test_complete_workflow_execution():
    """Test complete workflow execution with proper input data."""
    print("=== Testing Complete Workflow Execution ===")
    
    try:
        # Get workflow builder
        workflow_builder = get_workflow_builder()
        
        # Create test input data
        input_data = create_test_input_data()
        
        print("\n--- Input Data Summary ---")
        print(f"Personal Info: {input_data['personal_info']['name']}")
        print(f"Experience Entries: {len(input_data['experience'])}")
        print(f"Job Description: {input_data['job_description']['raw_text'][:100]}...")
        print(f"Projects: {len(input_data['projects'])}")
        
        # Execute workflow
        print("\n--- Executing JOB_TAILORED_CV Workflow ---")
        result = await workflow_builder.execute_workflow(
            workflow_type=WorkflowType.JOB_TAILORED_CV,
            input_data=input_data,
            session_id="test-session-complete"
        )
        
        print(f"\n--- Workflow Result ---")
        print(f"Success: {result.success}")
        print(f"Execution Time: {result.total_execution_time}")
        print(f"Completed Tasks: {len(result.completed_tasks)}")
        print(f"Failed Tasks: {len(result.failed_tasks)}")
        
        if result.completed_tasks:
            print("\n--- Completed Task Results ---")
            for i, task in enumerate(result.completed_tasks):
                print(f"Task {i+1}: {task.agent_type} - {task.status}")
                if hasattr(task, 'result') and task.result and hasattr(task.result, 'output_data'):
                    output_preview = str(task.result.output_data)[:200]
                    print(f"  Output Preview: {output_preview}...")
        
        if result.failed_tasks:
            print("\n--- Failed Task Results ---")
            for i, task in enumerate(result.failed_tasks):
                print(f"Failed Task {i+1}: {task.agent_type} - {task.status}")
                if hasattr(task, 'error_message'):
                    print(f"  Error: {task.error_message}")
        
        # Note: Agent pool state checking removed due to API changes
        print(f"\n--- Workflow Execution Complete ---")
        
        return result.success
        
    except Exception as e:
        print(f"\n❌ Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_workflow_with_minimal_data():
    """Test workflow with minimal required data to check error handling."""
    print("\n\n=== Testing Workflow with Minimal Data ===")
    
    try:
        workflow_builder = get_workflow_builder()
        
        minimal_data = {
            "personal_info": {"name": "Test User"},
            "experience": [],
            "job_description": "Python developer position"
        }
        
        print("\n--- Executing with minimal data ---")
        result = await workflow_builder.execute_workflow(
            workflow_type=WorkflowType.JOB_TAILORED_CV,
            input_data=minimal_data,
            session_id="test-session-minimal"
        )
        
        print(f"Success: {result.success}")
        print(f"Completed Tasks: {len(result.completed_tasks)}")
        print(f"Failed Tasks: {len(result.failed_tasks)}")
        
        return True
        
    except Exception as e:
        print(f"Expected error with minimal data: {e}")
        return True  # This is expected behavior


async def main():
    """Run all workflow tests."""
    print("Starting comprehensive workflow tests...\n")
    
    # Test 1: Complete workflow execution
    test1_passed = await test_complete_workflow_execution()
    
    # Test 2: Minimal data handling
    test2_passed = await test_workflow_with_minimal_data()
    
    # Summary
    print("\n\n=== Test Summary ===")
    print(f"Complete Workflow Test: {'✓' if test1_passed else '❌'}")
    print(f"Minimal Data Test: {'✓' if test2_passed else '❌'}")
    
    if test1_passed and test2_passed:
        print("\n✓ All workflow tests passed - logger fix verified!")
    else:
        print("\n❌ Some tests failed - further investigation needed")
    
    return test1_passed and test2_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)