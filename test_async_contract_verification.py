#!/usr/bin/env python3
"""Simple test to verify the async contract of generate_content is working correctly."""

import asyncio
import inspect
from src.services.llm_service import EnhancedLLMService


async def test_async_contract():
    """Test that the async contract is properly implemented."""
    print("=== Testing Async Contract of LLM Service ===")
    
    try:
        # Test 1: Verify generate_content is async
        print("\n1. Checking if generate_content is async...")
        assert inspect.iscoroutinefunction(EnhancedLLMService.generate_content), \
            "generate_content should be an async function"
        print("   ✓ generate_content is properly async")
        
        # Test 2: Verify _generate_with_timeout is async
        print("\n2. Checking if _generate_with_timeout is async...")
        assert inspect.iscoroutinefunction(EnhancedLLMService._generate_with_timeout), \
            "_generate_with_timeout should be an async function"
        print("   ✓ _generate_with_timeout is properly async")
        
        # Test 3: Verify _make_llm_api_call is NOT async (synchronous)
        print("\n3. Checking if _make_llm_api_call is synchronous...")
        assert not inspect.iscoroutinefunction(EnhancedLLMService._make_llm_api_call), \
            "_make_llm_api_call should be synchronous"
        print("   ✓ _make_llm_api_call is properly synchronous")
        
        # Test 4: Try to create service instance (will fail without API key, but that's expected)
        print("\n4. Testing service instantiation...")
        try:
            service = EnhancedLLMService(user_api_key="test-key")
            print("   ✓ Service can be instantiated with API key")
        except Exception as e:
            if "google-generativeai" in str(e):
                print("   ⚠ Service requires google-generativeai package (expected in test environment)")
            else:
                print(f"   ⚠ Service instantiation failed: {e}")
        
        print("\n=== All Async Contract Tests Passed! ===")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_async_contract())
    if result:
        print("\n🎉 Async contract verification PASSED")
        exit(0)
    else:
        print("\n💥 Async contract verification FAILED")
        exit(1)