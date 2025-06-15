#!/usr/bin/env python3
"""
Debug script to examine the workflow state and results.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.integration.enhanced_cv_system import EnhancedCVIntegration
from src.models.data_models import WorkflowType

async def debug_workflow_state():
    """Debug the workflow state and results."""
    print("=== Debugging Workflow State ===")
    
    try:
        print("1. Initializing EnhancedCVIntegration...")
        cv_system = EnhancedCVIntegration()
        print("   ✓ EnhancedCVIntegration initialized")
        
        print("\n2. Preparing test input...")
        test_input = {
            "job_description": "Software Engineer position requiring Python and AI experience.",
            "existing_cv_data": {
                "personal_info": {
                    "name": "Test User",
                    "email": "test@example.com"
                },
                "experience": [{
                    "title": "Software Developer",
                    "company": "Tech Corp",
                    "duration": "2020-2023",
                    "description": "Developed Python applications"
                }]
            }
        }
        print("   ✓ Test input prepared")
        
        print("\n3. Executing workflow...")
        result = await cv_system.execute_workflow(
            workflow_type=WorkflowType.BASIC_CV_GENERATION,
            input_data=test_input,
            session_id="debug_state_session"
        )
        
        print("\n4. Analyzing workflow result...")
        print(f"   Result type: {type(result)}")
        print(f"   Success: {result.get('success', 'N/A')}")
        print(f"   Results type: {type(result.get('results', 'N/A'))}")
        
        if 'results' in result and result['results']:
            results = result['results']
            print(f"   Results keys: {list(results.keys())}")
            
            # Check specific fields
            if 'structured_cv' in results:
                cv = results['structured_cv']
                print(f"   Structured CV type: {type(cv)}")
                if hasattr(cv, 'sections'):
                    print(f"   CV sections count: {len(cv.sections)}")
                    
            if 'job_description_data' in results:
                job_data = results['job_description_data']
                print(f"   Job description data type: {type(job_data)}")
                
            if 'final_output_path' in results:
                print(f"   Final output path: {results['final_output_path']}")
                
            if 'error_messages' in results:
                print(f"   Error messages: {results['error_messages']}")
        else:
            print("   ⚠️ Results is empty or None")
            
        print(f"   Processing time: {result.get('processing_time', 'N/A')}")
        print(f"   Errors: {result.get('errors', 'N/A')}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during workflow execution:")
        print(f"   Exception type: {type(e).__name__}")
        print(f"   Exception message: {str(e)}")
        import traceback
        print(f"   Full traceback:\n{traceback.format_exc()}")
        return None

if __name__ == "__main__":
    result = asyncio.run(debug_workflow_state())
    print(f"\n=== Final Result ===\n{result}")