#!/usr/bin/env python3

import sys
import os
import traceback

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.models.data_models import ContentType
from src.agents.agent_base import AgentExecutionContext

async def test_string_error():
    """Test to isolate the exact location of the string error."""
    try:
        # Initialize the content writer
        writer = EnhancedContentWriterAgent()
        
        # Create test data that matches the problematic scenario
        input_data = {
            'job_description_data': {
                'title': 'Supply Chain Apprentice',
                'raw_text': 'Test job description',
                'company': 'Test Company',
                'skills': ['Python', 'Data Analysis']
            },
            'content_item': {
                'type': 'experience',
                'data': {
                    'roles': [],  # Empty roles list
                    'projects': [],
                    'personal_info': {}
                }
            },
            'context': {
                'workflow_type': 'job_tailored_cv',
                'content_type': 'experience'
            }
        }
        
        # Create context
        context = AgentExecutionContext(
            session_id="test_session",
            item_id="test_item",
            content_type=ContentType.EXPERIENCE,
            metadata={}
        )
        
        print("=== Testing _format_role_info directly ===")
        
        # Test the _format_role_info method directly
        content_item = input_data['content_item']
        generation_context = input_data['context']
        
        print(f"content_item type: {type(content_item)}")
        print(f"content_item: {content_item}")
        print(f"generation_context type: {type(generation_context)}")
        print(f"generation_context: {generation_context}")
        
        # Call the method that's causing the error
        result = writer._format_role_info(content_item, generation_context)
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"\n=== ERROR CAPTURED ===")
        print(f"Error: {e}")
        print(f"Error type: {type(e)}")
        print("\n=== FULL TRACEBACK ===")
        traceback.print_exc()
        print("\n=== END TRACEBACK ===")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_string_error())