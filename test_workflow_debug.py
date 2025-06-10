#!/usr/bin/env python3
"""
Comprehensive workflow test to debug CV generation issues
"""

import sys
import os
import asyncio
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.integration.enhanced_cv_system import EnhancedCVIntegration
from src.core.state_manager import StateManager
from src.models.data_models import ContentType
from src.orchestration.workflow_definitions import WorkflowBuilder, WorkflowType

async def test_full_workflow():
    """Test the complete CV generation workflow"""
    print("🔍 Testing Full CV Generation Workflow...")
    
    try:
        # Initialize the system
        print("\n1. Initializing Enhanced CV System...")
        cv_system = EnhancedCVIntegration()
        
        # Create test data
        print("\n2. Creating test data...")
        test_job_description = """
        Senior Software Engineer
        
        We are looking for a Senior Software Engineer with experience in:
        - Python development
        - Web frameworks (Django, Flask)
        - Database design
        - Team leadership
        - CI/CD pipelines
        
        Requirements:
        - 5+ years of software development experience
        - Strong problem-solving skills
        - Experience with cloud platforms
        """
        
        test_cv_text = """
        John Doe
        Senior Software Engineer
        
        Experience:
        • Senior Software Engineer at TechCorp Inc (2020-2023)
          - Developed scalable web applications using Python and Django
          - Led a team of 5 junior developers
          - Implemented CI/CD pipelines reducing deployment time by 50%
        
        • Data Analyst at DataCorp (2018-2020)
          - Analyzed large datasets using SQL and Python
          - Created automated reporting dashboards
          - Improved data processing efficiency by 30%
        
        Skills:
        - Python, Django, Flask
        - SQL, PostgreSQL
        - AWS, Docker
        - Team Leadership
        """
        
        # Test workflow creation
        print("\n3. Creating workflow builder...")
        workflow_builder = WorkflowBuilder()
        
        # Prepare input data
        print("\n4. Preparing input data...")
        input_data = {
            "job_description": test_job_description,
            "cv_text": test_cv_text,
            "personal_info": {"name": "John Doe", "title": "Senior Software Engineer"},
            "experience": test_cv_text,
            "education": "Bachelor's in Computer Science"
        }
        
        # Execute workflow
        print("\n5. Executing comprehensive CV workflow...")
        result = await workflow_builder.execute_workflow(
            WorkflowType.COMPREHENSIVE_CV,
            input_data,
            session_id="test_session"
        )
        
        print(f"\n7. Workflow execution result:")
        print(f"   Success: {result.success}")
        print(f"   Completed tasks: {len(result.completed_tasks)}")
        print(f"   Failed tasks: {len(result.failed_tasks)}")
        print(f"   Error summary: {result.error_summary}")
        
        # Analyze individual task results
        if result.completed_tasks:
            print("\n8. Analyzing completed task results:")
            for i, task in enumerate(result.completed_tasks):
                print(f"   Task {i+1} ({task.agent_type}):")
                print(f"     Status: {task.status}")
                print(f"     Success: {task.result.success if task.result else False}")
                if task.result and task.result.output_data:
                    if isinstance(task.result.output_data, dict):
                        content_keys = list(task.result.output_data.keys())
                        print(f"     Output keys: {content_keys}")
                    else:
                        print(f"     Output type: {type(task.result.output_data)}")
                if task.error:
                    print(f"     Error: {task.error}")
        
        if result.failed_tasks:
            print("\n8b. Analyzing failed task results:")
            for i, task in enumerate(result.failed_tasks):
                print(f"   Failed Task {i+1} ({task.agent_type}):")
                print(f"     Status: {task.status}")
                print(f"     Error: {task.error}")
        
        # Test content aggregation
        if result.success and result.completed_tasks:
            print("\n9. Testing content aggregation...")
            from src.core.content_aggregator import ContentAggregator
            
            # Convert task results to the format expected by aggregator
            task_results = result.task_results
            
            aggregator = ContentAggregator()
            cv_content = aggregator.aggregate_results(task_results)
            
            if cv_content:
                print(f"   Aggregation successful!")
                print(f"   Populated fields: {[k for k, v in cv_content.items() if v]}")
                return True
            else:
                print(f"   Aggregation failed!")
                return False
        else:
            print(f"\n❌ Workflow failed - cannot test aggregation")
            return False
            
    except Exception as e:
        print(f"\n❌ Error in workflow test: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

async def test_individual_agent():
    """Test individual agent execution"""
    print("\n🔍 Testing Individual Agent Execution...")
    
    try:
        from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
        from src.core.state_manager import AgentExecutionContext
        
        # Create agent
        agent = EnhancedContentWriterAgent(content_type=ContentType.EXPERIENCE)
        
        # Create test context
        context = AgentExecutionContext(
            session_id="test_session",
            content_item={
                "name": "Professional Experience",
                "subsections": [
                    {
                        "name": "Senior Software Engineer at TechCorp Inc (2020-2023)",
                        "items": [
                            {"item_type": "bullet_point", "content": "Developed scalable web applications using Python and Django"},
                            {"item_type": "bullet_point", "content": "Led a team of 5 junior developers"}
                        ]
                    }
                ]
            },
            job_description="Senior Software Engineer position requiring Python and leadership skills"
        )
        
        # Execute agent
        result = await agent.execute(context)
        
        print(f"   Agent execution result:")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Content keys: {list(result.get('content', {}).keys()) if result.get('content') else []}")
        if result.get('error'):
            print(f"   Error: {result['error']}")
            
        return result.get('success', False)
        
    except Exception as e:
        print(f"\n❌ Error in individual agent test: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting Comprehensive CV Workflow Debug Tests\n")
    
    # Test individual agent first
    agent_success = await test_individual_agent()
    
    # Test full workflow
    workflow_success = await test_full_workflow()
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY:")
    print(f"   Individual Agent Test: {'✅ PASSED' if agent_success else '❌ FAILED'}")
    print(f"   Full Workflow Test: {'✅ PASSED' if workflow_success else '❌ FAILED'}")
    
    if agent_success and workflow_success:
        print("\n🎉 All tests passed! The workflow should be working correctly.")
    elif agent_success and not workflow_success:
        print("\n⚠️  Individual agents work, but workflow orchestration has issues.")
    else:
        print("\n❌ There are still issues with the individual agents.")

if __name__ == "__main__":
    asyncio.run(main())