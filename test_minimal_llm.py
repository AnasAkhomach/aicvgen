import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.llm_service import get_llm_service

async def test_minimal():
    try:
        print("Getting LLM service...")
        service = get_llm_service()
        print(f"Service type: {type(service)}")
        print(f"Service is None: {service is None}")
        
        print("Calling generate_content...")
        result = await service.generate_content('test prompt')
        print(f"Result type: {type(result)}")
        print(f"Result is None: {result is None}")
        print(f"Result success: {getattr(result, 'success', 'N/A')}")
        print(f"Result content: {getattr(result, 'content', 'N/A')[:100]}")
        print(f"Result error_message: {getattr(result, 'error_message', 'N/A')}")
        
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_minimal())