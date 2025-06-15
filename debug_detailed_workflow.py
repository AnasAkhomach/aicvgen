#!/usr/bin/env python3
"""
Detailed debug script to trace the workflow execution step by step.
"""

import asyncio
import sys
import traceback
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.integration.enhanced_cv_system import EnhancedCVIntegration
from src.models.data_models import WorkflowType
from src.config.settings import get_config
from src.agents.parser_agent import ParserAgent
from src.services.llm import LLM

# Set up logging to see detailed output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_parser_agent_detailed():
    """Test the parser agent in detail."""
    print("\n=== Testing Parser Agent ===")
    
    try:
        llm = LLM()
        parser_agent = ParserAgent(
            name="debug_parser",
            description="Debug parser agent for testing",
            llm=llm
        )
        
        # Test input similar to what the workflow uses
        parser_input = {
            "job_description": "Software Engineer position requiring Python and AI experience.",
            "cv_text": "Test User\ntest@example.com\n\nSoftware Developer at Tech Corp (2020-2023)\nDeveloped Python applications"
        }
        
        print(f"Parser input: {parser_input}")
        
        result = await parser_agent.run(parser_input)
        
        print(f"Parser result type: {type(result)}")
        print(f"Parser result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        print(f"Parser result: {result}")
        
        return result
        
    except Exception as e:
        print(f"Parser agent error: {type(e).__name__}: {str(e)}")
        print(f"Full traceback:")
        traceback.print_exc()
        
        # Try to get more details about the error
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(f"Exception type: {exc_type}")
        print(f"Exception value: {exc_value}")
        print(f"Exception args: {getattr(exc_value, 'args', 'No args')}")
        
        return None

async def test_orchestrator_detailed():
    """Test the orchestrator initialization and execution."""
    print("\n=== Testing Orchestrator ===")
    
    try:
        cv_system = EnhancedCVIntegration()
        orchestrator = cv_system._orchestrator
        
        print(f"Orchestrator type: {type(orchestrator)}")
        print(f"State manager type: {type(orchestrator.state_manager)}")
        
        # Test state manager methods
        print("\n--- Testing State Manager ---")
        
        # Check if job description data exists
        try:
            job_desc_data = orchestrator.state_manager.get_job_description_data()
            print(f"Initial job description data: {job_desc_data}")
        except Exception as e:
            print(f"No initial job description data: {e}")
        
        # Check if structured CV exists
        try:
            structured_cv = orchestrator.state_manager.get_structured_cv()
            print(f"Initial structured CV: {structured_cv}")
        except Exception as e:
            print(f"No initial structured CV: {e}")
        
        return orchestrator
        
    except Exception as e:
        print(f"Orchestrator error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return None

async def test_full_workflow_detailed():
    """Test the full workflow with detailed logging."""
    print("\n=== Testing Full Workflow ===")
    
    try:
        cv_system = EnhancedCVIntegration()
        
        # Test input data
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
        
        print(f"Input data: {input_data}")
        
        # Execute workflow
        result = await cv_system.execute_workflow(
            workflow_type=WorkflowType.BASIC_CV_GENERATION,
            input_data=input_data,
            session_id="debug_detailed_session"
        )
        
        print(f"\nWorkflow result:")
        print(f"  Success: {result.get('success')}")
        print(f"  Results keys: {list(result.get('results', {}).keys())}")
        print(f"  Metadata: {result.get('metadata')}")
        print(f"  Processing time: {result.get('processing_time')}")
        print(f"  Errors: {result.get('errors')}")
        
        if result.get('results'):
            print(f"  Results content: {result['results']}")
        
        return result
        
    except Exception as e:
        print(f"Full workflow error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return None

async def main():
    """Main debug function."""
    print("Starting detailed workflow debugging...")
    
    # Test parser agent
    parser_result = await test_parser_agent_detailed()
    
    # Test orchestrator
    orchestrator = await test_orchestrator_detailed()
    
    # Test full workflow
    workflow_result = await test_full_workflow_detailed()
    
    print("\n=== Debug Summary ===")
    print(f"Parser agent result: {'Success' if parser_result else 'Failed'}")
    print(f"Orchestrator initialization: {'Success' if orchestrator else 'Failed'}")
    print(f"Full workflow result: {'Success' if workflow_result and workflow_result.get('success') else 'Failed'}")
    
    if workflow_result and not workflow_result.get('success'):
        print(f"Workflow failure reason: Empty results with no errors - likely implementation issue")

if __name__ == "__main__":
    asyncio.run(main())