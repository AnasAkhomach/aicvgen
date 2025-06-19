#!/usr/bin/env python3
"""
Test script to debug LLM service initialization issues.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.llm_service import EnhancedLLMService
from src.config.settings import get_config

def test_llm_service_initialization():
    """Test LLM service initialization and basic functionality."""
    print("=== LLM Service Debug Test ===")
    
    try:
        # Test configuration loading
        print("1. Loading configuration...")
        config = get_config()
        print(f"   ✓ Configuration loaded successfully")
        print(f"   - Primary API key configured: {bool(config.llm.gemini_api_key_primary)}")
        print(f"   - Fallback API key configured: {bool(config.llm.gemini_api_key_fallback)}")
        print(f"   - Default model: {config.llm_settings.default_model}")
        
        # Test LLM service initialization
        print("\n2. Initializing LLM service...")
        llm_service = EnhancedLLMService()
        print(f"   ✓ LLM service initialized successfully")
        print(f"   - Service type: {type(llm_service).__name__}")
        print(f"   - Model name: {llm_service.model_name}")
        print(f"   - Using user key: {llm_service.using_user_key}")
        print(f"   - Using fallback: {llm_service.using_fallback}")
        print(f"   - LLM model object: {type(llm_service.llm).__name__ if llm_service.llm else 'None'}")
        
        # Test basic LLM call
        print("\n3. Testing basic LLM generation...")
        import asyncio
        
        async def test_generation():
            try:
                response = await llm_service.generate_content(
                    prompt="Say 'Hello, this is a test!' in exactly those words.",
                    session_id="test_session"
                )
                print(f"   ✓ LLM generation successful")
                print(f"   - Response type: {type(response).__name__}")
                print(f"   - Success: {response.success}")
                print(f"   - Content length: {len(response.content) if response.content else 0}")
                print(f"   - Content preview: {response.content[:100] if response.content else 'None'}...")
                return True
            except Exception as e:
                print(f"   ✗ LLM generation failed: {str(e)}")
                print(f"   - Error type: {type(e).__name__}")
                return False
        
        success = asyncio.run(test_generation())
        
        if success:
            print("\n=== All tests passed! ===")
        else:
            print("\n=== LLM generation test failed ===")
            
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_llm_service_initialization()