#!/usr/bin/env python3
"""
Debug script to test the workflow graph directly.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.orchestration.cv_workflow_graph import get_cv_workflow_graph
from src.orchestration.state import AgentState
from src.models.data_models import StructuredCV, JobDescriptionData

async def test_workflow_direct():
    """Test the workflow graph directly."""
    print("=== Testing Workflow Graph Directly ===")
    
    try:
        print("1. Getting workflow graph...")
        workflow_app = get_cv_workflow_graph()
        print(f"   ✓ Workflow app type: {type(workflow_app)}")
        
        print("\n2. Creating initial state...")
        # Create minimal test data
        structured_cv = StructuredCV()
        job_description_data = JobDescriptionData(
            raw_text="Software Engineer position requiring Python and AI experience.",
            skills=[],
            experience_level="N/A",
            responsibilities=[],
            industry_terms=[],
            company_values=[]
        )
        
        initial_state = AgentState(
            structured_cv=structured_cv,
            job_description_data=job_description_data
        )
        print(f"   ✓ Initial state created")
        print(f"   State type: {type(initial_state)}")
        
        print("\n3. Converting state to dict...")
        state_dict = initial_state.model_dump()
        print(f"   ✓ State dict keys: {list(state_dict.keys())}")
        
        print("\n4. Invoking workflow...")
        print("   This may take a while...")
        
        # Run the workflow
        final_state_dict = await workflow_app.ainvoke(state_dict)
        
        print("\n5. Workflow completed!")
        print(f"   Final state type: {type(final_state_dict)}")
        print(f"   Final state keys: {list(final_state_dict.keys())}")
        
        # Convert back to AgentState
        final_state = AgentState.model_validate(final_state_dict)
        print(f"   ✓ Final AgentState created")
        
        # Check results
        if final_state.structured_cv:
            print(f"   CV sections count: {len(final_state.structured_cv.sections)}")
        
        if final_state.final_output_path:
            print(f"   Final output path: {final_state.final_output_path}")
            
        if final_state.error_messages:
            print(f"   Error messages: {final_state.error_messages}")
        else:
            print("   ✓ No error messages")
            
        return final_state
        
    except Exception as e:
        print(f"\n❌ Error during workflow execution:")
        print(f"   Exception type: {type(e).__name__}")
        print(f"   Exception message: {str(e)}")
        import traceback
        print(f"   Full traceback:\n{traceback.format_exc()}")
        return None

if __name__ == "__main__":
    result = asyncio.run(test_workflow_direct())
    print(f"\n=== Final Result ===\n{result}")