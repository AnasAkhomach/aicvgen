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

async def test_direct_timeout():
    """Test _generate_with_timeout directly."""
    try:
        print("=== Direct Timeout Test ===")
        
        # Initialize LLM service
        print("1. Initializing LLM service...")
        llm_service = EnhancedLLMService()
        print(f"   Service initialized: {type(llm_service)}")
        
        # Test _generate_with_timeout directly
        print("\n2. Testing _generate_with_timeout directly...")
        try:
            print("   About to call _generate_with_timeout...")
            result = await llm_service._generate_with_timeout("Test prompt")
            print(f"   Result type: {type(result)}")
            print(f"   Result is None: {result is None}")
            if hasattr(result, 'text'):
                print(f"   Result text: {result.text[:100] if result.text else 'None'}")
        except Exception as e:
            print(f"   _generate_with_timeout failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_direct_timeout())