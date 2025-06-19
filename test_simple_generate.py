import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from dotenv import load_dotenv
    from pathlib import Path
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from {env_path}")
except ImportError:
    print("dotenv not available")

from src.services.llm_service import EnhancedLLMService

async def test_simple_generate():
    """Simple test of generate_content."""
    try:
        print("=== Simple Generate Test ===")
        
        # Initialize LLM service
        print("1. Initializing LLM service...")
        llm_service = EnhancedLLMService()
        print(f"   Service initialized: {type(llm_service)}")
        
        # Test generate_content with detailed error handling
        print("\n2. Testing generate_content...")
        try:
            print("   About to call generate_content...")
            response = await llm_service.generate_content("Hello, this is a test prompt.")
            print(f"   Response received: {type(response)}")
            print(f"   Response success: {getattr(response, 'success', 'NO_SUCCESS_ATTR')}")
            print(f"   Response content: {getattr(response, 'content', 'NO_CONTENT_ATTR')[:100] if hasattr(response, 'content') else 'NO_CONTENT_ATTR'}")
            print(f"   Response error_message: {getattr(response, 'error_message', 'NO_ERROR_MSG_ATTR')}")
        except Exception as e:
            print(f"   generate_content failed: {type(e).__name__}: {e}")
            import traceback
            print("   Full traceback:")
            traceback.print_exc()
        
    except Exception as e:
        print(f"Test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple_generate())