#!/usr/bin/env python3
"""
Test to debug the _make_llm_api_call method specifically.
"""

import sys
import os
import asyncio
import traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.llm_service import EnhancedLLMService

async def test_api_call_debug():
    """Debug the _make_llm_api_call method."""
    print("=== Debugging _make_llm_api_call Method ===")
    
    try:
        # Initialize LLM service
        print("1. Initializing LLM service...")
        llm_service = EnhancedLLMService()
        print("   ✓ LLM service initialized")
        
        # Test the _make_llm_api_call method directly
        print("\n2. Testing _make_llm_api_call directly...")
        try:
            response = llm_service._make_llm_api_call("Say 'Hello test' exactly.")
            print(f"   ✓ API call successful")
            print(f"   - Response type: {type(response).__name__}")
            print(f"   - Response is None: {response is None}")
            print(f"   - Has text attribute: {hasattr(response, 'text')}")
            
            if hasattr(response, 'text'):
                print(f"   - Text is None: {response.text is None}")
                print(f"   - Text content: {repr(response.text[:100] if response.text else 'None')}")
            
            # Check all attributes
            print(f"   - All attributes: {dir(response)}")
            
        except Exception as e:
            print(f"   ✗ API call failed: {str(e)}")
            print(f"   - Exception type: {type(e).__name__}")
            traceback.print_exc()
            return False
        
        # Test the _generate_with_timeout method
        print("\n3. Testing _generate_with_timeout...")
        try:
            response = llm_service._generate_with_timeout("Say 'Hello test' exactly.", "test_session")
            print(f"   ✓ Timeout call successful")
            print(f"   - Response type: {type(response).__name__}")
            print(f"   - Response is None: {response is None}")
            print(f"   - Has text attribute: {hasattr(response, 'text')}")
            
            if hasattr(response, 'text'):
                print(f"   - Text is None: {response.text is None}")
                print(f"   - Text content: {repr(response.text[:100] if response.text else 'None')}")
            
        except Exception as e:
            print(f"   ✗ Timeout call failed: {str(e)}")
            print(f"   - Exception type: {type(e).__name__}")
            traceback.print_exc()
            return False
        
        # Test with executor
        print("\n4. Testing with executor...")
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                llm_service.executor, 
                llm_service._generate_with_timeout, 
                "Say 'Hello test' exactly.", 
                "test_session",
                None  # trace_id
            )
            print(f"   ✓ Executor call successful")
            print(f"   - Response type: {type(response).__name__}")
            print(f"   - Response is None: {response is None}")
            print(f"   - Has text attribute: {hasattr(response, 'text')}")
            
            if hasattr(response, 'text'):
                print(f"   - Text is None: {response.text is None}")
                print(f"   - Text content: {repr(response.text[:100] if response.text else 'None')}")
            
        except Exception as e:
            print(f"   ✗ Executor call failed: {str(e)}")
            print(f"   - Exception type: {type(e).__name__}")
            traceback.print_exc()
            
            # This is where the error might be coming from
            if "NoneType" in str(e) and "await" in str(e):
                print("   *** This is the source of the NoneType await error! ***")
                
                # Let's check what's being returned
                print("\n5. Investigating the return value...")
                try:
                    # Try without await to see what we get
                    result = loop.run_in_executor(
                        llm_service.executor, 
                        llm_service._generate_with_timeout, 
                        "Say 'Hello test' exactly.", 
                        "test_session",
                        None
                    )
                    print(f"   - run_in_executor returns: {type(result).__name__}")
                    print(f"   - Is coroutine: {asyncio.iscoroutine(result)}")
                    print(f"   - Is future: {asyncio.isfuture(result)}")
                    
                except Exception as inner_e:
                    print(f"   - Investigation failed: {str(inner_e)}")
            
            return False
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test setup failed: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_api_call_debug())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")