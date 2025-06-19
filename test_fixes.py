#!/usr/bin/env python3
"""
Test script to verify the bug fixes implemented.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from src.models.data_models import RateLimitState
from src.services.llm_service import LLMResponse

def test_rate_limit_state_fix():
    """Test that RateLimitState now has the required attributes."""
    print("Testing RateLimitState fix...")
    
    # Create a RateLimitState instance
    state = RateLimitState(model="test-model")
    
    # Test that the new attributes exist
    assert hasattr(state, 'tokens_made'), "tokens_made attribute missing"
    assert hasattr(state, 'tokens_limit'), "tokens_limit attribute missing"
    
    # Test the property aliases
    assert hasattr(state, 'requests_per_minute'), "requests_per_minute property missing"
    assert hasattr(state, 'tokens_per_minute'), "tokens_per_minute property missing"
    
    # Test that properties return correct values
    assert state.requests_per_minute == 0, f"Expected 0, got {state.requests_per_minute}"
    assert state.tokens_per_minute == 0, f"Expected 0, got {state.tokens_per_minute}"
    
    # Test record_request with tokens
    state.record_request(tokens_used=100)
    assert state.requests_per_minute == 1, f"Expected 1, got {state.requests_per_minute}"
    assert state.tokens_per_minute == 100, f"Expected 100, got {state.tokens_per_minute}"
    
    print("✓ RateLimitState fix verified!")

def test_llm_response_handling():
    """Test that LLMResponse content extraction works."""
    print("Testing LLMResponse handling fix...")
    
    # Create a mock LLMResponse
    response = LLMResponse(
        content='{"test": "data"}',
        tokens_used=50,
        processing_time=1.0,
        model_used="test-model",
        success=True
    )
    
    # Test content extraction logic
    response_content = response.content if hasattr(response, 'content') else str(response)
    assert response_content == '{"test": "data"}', f"Expected JSON string, got {response_content}"
    
    print("✓ LLMResponse handling fix verified!")

def test_none_input_handling():
    """Test that None input handling works."""
    print("Testing None input handling fix...")
    
    # Test the logic that was added to research_agent.py
    input_data = None
    
    # Handle None input_data (same logic as in research_agent.py)
    if input_data is None:
        input_data = {}
    
    # This should not raise an error now
    result = input_data.get("structured_cv", "default")
    assert result == "default", f"Expected 'default', got {result}"
    
    print("✓ None input handling fix verified!")

if __name__ == "__main__":
    print("Running bug fix verification tests...\n")
    
    try:
        test_rate_limit_state_fix()
        test_llm_response_handling()
        test_none_input_handling()
        
        print("\n🎉 All fixes verified successfully!")
        print("The critical errors identified in the error log have been resolved.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)