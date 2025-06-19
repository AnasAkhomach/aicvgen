import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.llm_service import EnhancedLLMService

async def test_detailed_trace():
    """Detailed trace of the await issue."""
    try:
        print("=== Detailed Trace Test ===")
        
        # Initialize LLM service
        print("1. Initializing LLM service...")
        llm_service = EnhancedLLMService()
        print(f"   Service initialized: {type(llm_service)}")
        print(f"   LLM model: {getattr(llm_service, 'llm', 'NOT_FOUND')}")
        print(f"   Model name: {getattr(llm_service, 'model_name', 'NOT_FOUND')}")
        
        # Test _make_llm_api_call directly
        print("\n2. Testing _make_llm_api_call directly...")
        try:
            direct_response = llm_service._make_llm_api_call("Test prompt")
            print(f"   Direct response type: {type(direct_response)}")
            print(f"   Direct response is None: {direct_response is None}")
            if hasattr(direct_response, 'text'):
                print(f"   Direct response text: {direct_response.text[:100] if direct_response.text else 'None'}")
        except Exception as e:
            print(f"   Direct call failed: {type(e).__name__}: {e}")
        
        # Test _generate_with_timeout
        print("\n3. Testing _generate_with_timeout...")
        try:
            timeout_response = await llm_service._generate_with_timeout("Test prompt")
            print(f"   Timeout response type: {type(timeout_response)}")
            print(f"   Timeout response is None: {timeout_response is None}")
            if hasattr(timeout_response, 'text'):
                print(f"   Timeout response text: {timeout_response.text[:100] if timeout_response.text else 'None'}")
        except Exception as e:
            print(f"   Timeout call failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        
        # Test full generate_content
        print("\n4. Testing full generate_content...")
        try:
            full_response = await llm_service.generate_content("Test prompt")
            print(f"   Full response type: {type(full_response)}")
            print(f"   Full response is None: {full_response is None}")
            if hasattr(full_response, 'content'):
                print(f"   Full response content: {full_response.content[:100] if full_response.content else 'None'}")
        except Exception as e:
            print(f"   Full call failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_detailed_trace())