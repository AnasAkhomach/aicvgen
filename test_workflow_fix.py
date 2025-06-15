#!/usr/bin/env python3
"""
Test script to verify the workflow precondition error fix.
"""

import sys
import os
import asyncio

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Change to src directory to handle relative imports
os.chdir(src_path)

from integration.enhanced_cv_system import EnhancedCVSystem
from core.config import get_config
from models.workflow_models import WorkflowType

async def test_workflow_fix():
    """Test the workflow initialization fix."""
    try:
        print("Testing enhanced CV system initialization...")
        config = get_config()
        system = EnhancedCVSystem(config)
        print("✓ Enhanced CV system initialized successfully")
        
        # Test job-tailored CV workflow with sample data
        print("\nTesting job-tailored CV workflow...")
        test_input = {
            "personal_info": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "+1-555-0123"
            },
            "experience": [
                {
                    "title": "Software Engineer",
                    "company": "Tech Corp",
                    "description": "Developed web applications using Python and JavaScript"
                }
            ],
            "job_description": {
                "title": "Senior Software Engineer",
                "company": "Example Inc",
                "description": "We are looking for a senior software engineer with experience in Python and web development.",
                "requirements": "5+ years of Python experience, web development skills",
                "responsibilities": "Design and implement software solutions, mentor junior developers"
            }
        }
        
        # This should now work without the workflow precondition error
        result = await system.execute_workflow(
            WorkflowType.JOB_TAILORED_CV,
            test_input,
            session_id="test_session"
        )
        
        if result["success"]:
            print("✓ Workflow executed successfully - fix verified!")
        else:
            print(f"✗ Workflow failed: {result.get('errors', 'Unknown error')}")
            
    except Exception as e:
        print(f"✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_workflow_fix())