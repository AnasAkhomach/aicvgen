#!/usr/bin/env python3
"""
Debug script to reproduce the workflow execution error.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.integration.enhanced_cv_system import EnhancedCVIntegration
from src.models.data_models import WorkflowType
from src.config.settings import get_config

async def test_workflow_execution():
    """Test workflow execution to reproduce the error."""
    try:
        print("Initializing EnhancedCVSystem...")
        cv_system = EnhancedCVIntegration()
        
        print("Testing basic CV generation workflow...")
        
        # Test with minimal input data
        input_data = {
            "job_description": "Software Engineer position requiring Python and AI experience.",
            "existing_cv_data": {
                "personal_info": {
                    "name": "Test User",
                    "email": "test@example.com"
                },
                "experience": [
                    {
                        "title": "Software Developer",
                        "company": "Tech Corp",
                        "duration": "2020-2023",
                        "description": "Developed Python applications"
                    }
                ]
            }
        }
        
        result = await cv_system.execute_workflow(
            workflow_type=WorkflowType.BASIC_CV_GENERATION,
            input_data=input_data,
            session_id="debug_session"
        )
        
        print(f"Workflow result: {result}")
        
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {str(e)}")
        print(f"Full traceback:")
        traceback.print_exc()
        return e
    
    return None

if __name__ == "__main__":
    error = asyncio.run(test_workflow_execution())
    if error:
        print(f"\nDebugging complete. Error reproduced: {type(error).__name__}")
    else:
        print("\nNo error occurred during test.")