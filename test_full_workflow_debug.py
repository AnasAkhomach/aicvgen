#!/usr/bin/env python3
"""
Test to debug the full workflow that's causing the NoneType await error.
"""

import sys
import os
import asyncio
import traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.parser_agent import ParserAgent
from src.orchestration.state import AgentState
from src.models.data_models import StructuredCV, PersonalInfo, Section

async def test_full_workflow_debug():
    """Debug the full workflow from ParserAgent to LLM service."""
    print("=== Debugging Full Workflow ===")
    
    try:
        # Initialize ParserAgent
        print("1. Initializing ParserAgent...")
        parser_agent = ParserAgent(name="TestParserAgent", description="Test parser agent for debugging")
        print("   ✓ ParserAgent initialized")
        
        # Check if LLM service is properly initialized
        print("\n2. Checking LLM service...")
        llm_service = getattr(parser_agent, 'llm', None)
        if llm_service:
            print(f"   ✓ LLM service found: {type(llm_service).__name__}")
            print(f"   - Executor is None: {getattr(llm_service, 'executor', None) is None}")
        else:
            print("   - No LLM service found on agent")
        
        # Create minimal StructuredCV for testing
        personal_info = PersonalInfo(
            name="Test User",
            email="test@example.com",
            phone="123-456-7890"
        )
        
        structured_cv = StructuredCV(
            sections=[]
        )
        
        # Create a test state
        print("\n3. Creating test state...")
        test_state = AgentState(
            structured_cv=structured_cv,
            cv_text="Test CV text"
        )
        print("   ✓ Test state created")
        
        # Test the parse_job_description method directly
        print("\n4. Testing parse_job_description...")
        try:
            raw_job_text = "Software Engineer position at Tech Company. Requirements: Python, JavaScript, React."
            job_data = await parser_agent.parse_job_description(
                raw_job_text,
                trace_id="test-trace"
            )
            print(f"   ✓ Job description parsed successfully")
            print(f"   - Job title: {getattr(job_data, 'job_title', 'N/A')}")
            
        except Exception as e:
            print(f"   ✗ Job description parsing failed: {str(e)}")
            print(f"   - Exception type: {type(e).__name__}")
            
            # Check if this is the NoneType await error
            if "NoneType" in str(e) and "await" in str(e):
                print("   *** Found the NoneType await error! ***")
                print("   - Full traceback:")
                traceback.print_exc()
                
                # Let's trace through the call stack
                print("\n5. Investigating the call stack...")
                
                # Check the _generate_and_parse_json method
                print("   - Testing _generate_and_parse_json directly...")
                try:
                    test_prompt = "Test prompt for debugging"
                    result = await parser_agent._generate_and_parse_json(prompt=test_prompt)
                    print(f"     ✓ _generate_and_parse_json worked: {type(result)}")
                except Exception as inner_e:
                    print(f"     ✗ _generate_and_parse_json failed: {str(inner_e)}")
                    if "NoneType" in str(inner_e) and "await" in str(inner_e):
                        print("     *** This is where the error originates! ***")
                        
                        # Let's check the LLM service call specifically
                        print("\n6. Investigating LLM service call...")
                        try:
                            # Get the LLM service
                            from src.services.llm_service import get_llm_service
                            llm_service = get_llm_service()
                            print(f"     - LLM service type: {type(llm_service).__name__}")
                            print(f"     - Executor is None: {getattr(llm_service, 'executor', None) is None}")
                            
                            # Try the generate_content call
                            response = await llm_service.generate_content(
                                prompt=test_prompt,
                                session_id="test-session"
                            )
                            print(f"     ✓ LLM service call worked: {response.success}")
                            
                        except Exception as llm_e:
                            print(f"     ✗ LLM service call failed: {str(llm_e)}")
                            if "NoneType" in str(llm_e) and "await" in str(llm_e):
                                print("     *** LLM service is the source! ***")
                                traceback.print_exc()
            
            return False
        
        # Test the full run_as_node method
        print("\n7. Testing run_as_node...")
        try:
            result = await parser_agent.run_as_node(test_state)
            print(f"   ✓ run_as_node successful")
            print(f"   - Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
        except Exception as e:
            print(f"   ✗ run_as_node failed: {str(e)}")
            if "NoneType" in str(e) and "await" in str(e):
                print("   *** Found the NoneType await error in run_as_node! ***")
                traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test setup failed: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_full_workflow_debug())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")