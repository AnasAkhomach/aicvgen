#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
from agents.enhanced_content_writer import EnhancedContentWriterAgent
from agents.agent_base import AgentExecutionContext
from core.content_types import ContentType

async def test_string_input():
    """Test the enhanced content writer with string input that causes the error."""
    
    # Create agent
    agent = EnhancedContentWriterAgent(
        name="TestContentWriter",
        description="Test agent for debugging string error",
        content_type=ContentType.EXPERIENCE
    )
    
    # Create context with string content_item (this should cause the error)
    context = AgentExecutionContext(
        session_id="test_session",
        item_id="test_item",
        content_type=ContentType.EXPERIENCE,
        input_data={
            "content_item": "This is a string instead of a dictionary",  # This should trigger the error
            "job_description_data": {
                "title": "Software Engineer",
                "raw_text": "Looking for a software engineer with Python experience"
            }
        }
    )
    
    # Test input data
    input_data = {
        "content_item": "This is a string instead of a dictionary",  # This should trigger the error
        "job_description_data": {
            "title": "Software Engineer", 
            "raw_text": "Looking for a software engineer with Python experience"
        }
    }
    
    print("Testing enhanced content writer with string input...")
    print(f"Input content_item type: {type(input_data['content_item'])}")
    print(f"Input content_item value: {input_data['content_item']}")
    
    try:
        result = await agent.run_async(input_data, context)
        print(f"\nResult success: {result.success}")
        print(f"Result error: {result.error_message}")
        print(f"Result content: {result.output_data.get('content', 'No content') if result.output_data else 'No output_data'}")
        return result
    except Exception as e:
        print(f"\nException occurred: {e}")
        print(f"Exception type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    result = asyncio.run(test_string_input())
    print("\n=== Test Complete ===")