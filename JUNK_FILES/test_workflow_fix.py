#!/usr/bin/env python3
"""
Test Workflow Fix

Simple test to verify the workflow data adapter fix is working.
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, List, Any

# Add project root and src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

# Set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = str(src_path)

from src.models.data_models import ContentType
from src.agents.agent_base import AgentExecutionContext
from src.orchestration.workflow_definitions import WorkflowBuilder, WorkflowType
from src.orchestration.agent_orchestrator import get_agent_orchestrator
from src.config.logging_config import setup_logging

# Setup logging
setup_logging()

async def test_workflow_data_fix():
    """Test the workflow data adapter fix."""
    print("=== Testing Workflow Data Fix ===")
    
    # Sample workflow input
    workflow_input = {
        "personal_info": {
            "name": "Nouha KARIM",
            "email": "nouhakarim305@gmail.com",
            "phone": "+33 765732271",
            "location": "Reims",
            "title": "ETUDIANTE EN M2 OPTION GÉNIE DES SYSTÈMES INDUSTRIELS"
        },
        "experience": [
            {
                "title": "Student in Industrial Systems Engineering",
                "company": "University",
                "duration": "2023-2024",
                "description": "Studying advanced industrial systems",
                "skills": ["Systems Analysis", "Process Optimization", "Project Management"]
            },
            {
                "title": "Intern",
                "company": "Tech Company",
                "duration": "Summer 2023",
                "description": "Worked on automation projects",
                "skills": ["Automation", "Python", "Data Analysis"]
            }
        ],
        "job_description": {
            "title": "Industrial Engineer",
            "company": "Manufacturing Corp",
            "description": "Looking for an industrial engineer to optimize manufacturing processes",
            "requirements": ["Bachelor's degree in Industrial Engineering", "Experience with process optimization"],
            "skills": ["Process Optimization", "Lean Manufacturing", "Six Sigma", "Python"],
            "responsibilities": ["Analyze manufacturing processes", "Implement improvements", "Lead projects"]
        }
    }
    
    print("\n1. Testing WorkflowBuilder data adaptation...")
    
    # Create workflow builder
    orchestrator = get_agent_orchestrator("test_workflow_fix")
    builder = WorkflowBuilder(orchestrator)
    
    # Test the data adaptation method directly
    adapted_data = builder._adapt_input_data_for_agent(
        workflow_input, 
        "content_writer", 
        ContentType.EXPERIENCE
    )
    
    print(f"✓ Data adapted successfully")
    print(f"  - Keys: {list(adapted_data.keys())}")
    print(f"  - Job title: {adapted_data['job_description_data']['title']}")
    print(f"  - Content type: {adapted_data['content_item']['type']}")
    print(f"  - Roles count: {len(adapted_data['content_item']['data']['roles'])}")
    
    print("\n2. Testing content writer with adapted data...")
    
    # Get content writer agent
    content_writer = orchestrator.get_agent("content_writer")
    if not content_writer:
        print("❌ Content writer agent not found!")
        return False
    
    # Create execution context
    context = AgentExecutionContext(
        session_id="test_workflow_fix",
        input_data=adapted_data,
        content_type=ContentType.EXPERIENCE
    )
    
    # Run the agent with timeout
    try:
        print("  Running content writer agent...")
        result = await asyncio.wait_for(
            content_writer.run_async(adapted_data, context),
            timeout=30.0  # 30 second timeout
        )
        
        print(f"✓ Content writer completed")
        print(f"  - Success: {result.success}")
        print(f"  - Confidence: {result.confidence_score}")
        print(f"  - Processing time: {result.processing_time:.2f}s")
        
        if result.success and result.output_data:
            content = result.output_data.get("content", "")
            print(f"  - Content length: {len(content)} characters")
            if content:
                print(f"  - Content preview: {content[:150]}...")
                return True
            else:
                print("  - ⚠️  No content generated!")
                return False
        else:
            print(f"  - ❌ Error: {result.error_message}")
            return False
            
    except asyncio.TimeoutError:
        print("  - ❌ Timeout: Agent took too long to respond")
        return False
    except Exception as e:
        print(f"  - ❌ Exception: {e}")
        return False

async def main():
    """Main test function."""
    try:
        success = await test_workflow_data_fix()
        print("\n=== Test Results ===")
        if success:
            print("✅ Workflow data fix is working!")
        else:
            print("❌ Workflow data fix needs more work.")
        return success
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)