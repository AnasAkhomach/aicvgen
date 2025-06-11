#!/usr/bin/env python3
"""
Simple test script to verify error propagation from agents.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.agents.agent_base import AgentResult
    print("✓ Successfully imported AgentResult")
except ImportError as e:
    print(f"✗ Failed to import AgentResult: {e}")
    sys.exit(1)

# Test AgentResult creation
print("\nTesting AgentResult creation...")

# Test successful result
success_result = AgentResult(
    success=True,
    output_data={"test": "data"},
    confidence_score=0.9,
    metadata={"test": True}
)
print(f"✓ Success result: success={success_result.success}")

# Test failure result
failure_result = AgentResult(
    success=False,
    output_data={},
    confidence_score=0.0,
    error_message="Test error message",
    metadata={"test": True}
)
print(f"✓ Failure result: success={failure_result.success}, error='{failure_result.error_message}'")

print("\n" + "="*50)
print("BASIC AGENTRESULT TESTS PASSED")
print("="*50)

# Now test a simple agent
print("\nTesting agent error handling...")

try:
    from src.agents.formatter_agent import FormatterAgent
    from src.agents.agent_base import AgentExecutionContext
    from src.models.data_models import ContentType
    
    async def test_formatter_agent():
        """Test the FormatterAgent error handling."""
        agent = FormatterAgent(
            name="TestFormatterAgent",
            description="Test formatter agent for error propagation testing"
        )
        context = AgentExecutionContext(
            session_id="test_session",
            item_id="test_item",
            content_type=ContentType.EXPERIENCE
        )
        
        # Test with None input (should trigger error handling)
        print("Testing with None input...")
        result = await agent.run_async(None, context)
        
        if hasattr(result, 'success') and hasattr(result, 'error_message'):
            print(f"✓ Agent returned proper AgentResult structure")
            print(f"  - success: {result.success}")
            if not result.success:
                print(f"  - error_message: {result.error_message}")
            return True
        else:
            print(f"✗ Agent did not return proper AgentResult structure")
            return False
    
    # Run the async test
    success = asyncio.run(test_formatter_agent())
    
    if success:
        print("\n" + "="*50)
        print("AGENT ERROR HANDLING TEST PASSED")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("AGENT ERROR HANDLING TEST FAILED")
        print("="*50)
        sys.exit(1)
        
except ImportError as e:
    print(f"✗ Failed to import required modules: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error during agent testing: {e}")
    sys.exit(1)

print("\nAll tests completed successfully!")