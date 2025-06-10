#!/usr/bin/env python3

import sys
import os
import traceback

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.models.data_models import ContentType
from src.agents.agent_base import AgentExecutionContext

# Monkey patch to add debugging
original_format_role_info = None
original_build_prompt = None
original_build_experience_prompt = None

def debug_format_role_info(self, content_item, generation_context):
    print(f"\n=== _format_role_info called ===")
    print(f"content_item type: {type(content_item)}")
    print(f"content_item: {content_item}")
    print(f"generation_context type: {type(generation_context)}")
    print(f"generation_context: {generation_context}")
    
    if isinstance(content_item, str):
        print(f"ERROR: content_item is a string: {content_item[:100]}...")
        raise ValueError(f"content_item should be dict, got string: {content_item[:100]}...")
    
    return original_format_role_info(content_item, generation_context)

def debug_build_prompt(self, job_data, content_item, generation_context, content_type):
    print(f"\n=== _build_prompt called ===")
    print(f"job_data type: {type(job_data)}")
    print(f"content_item type: {type(content_item)}")
    print(f"generation_context type: {type(generation_context)}")
    print(f"content_type: {content_type}")
    
    if isinstance(content_item, str):
        print(f"ERROR: content_item is a string in _build_prompt: {content_item[:100]}...")
        raise ValueError(f"content_item should be dict, got string: {content_item[:100]}...")
    
    return original_build_prompt(job_data, content_item, generation_context, content_type)

def debug_build_experience_prompt(self, template, job_data, content_item, generation_context):
    print(f"\n=== _build_experience_prompt called ===")
    print(f"template type: {type(template)}")
    print(f"job_data type: {type(job_data)}")
    print(f"content_item type: {type(content_item)}")
    print(f"generation_context type: {type(generation_context)}")
    
    if isinstance(content_item, str):
        print(f"ERROR: content_item is a string in _build_experience_prompt: {content_item[:100]}...")
        raise ValueError(f"content_item should be dict, got string: {content_item[:100]}...")
    
    return original_build_experience_prompt(template, job_data, content_item, generation_context)

async def test_full_workflow():
    """Test the full workflow to capture where the string error occurs."""
    global original_format_role_info, original_build_prompt, original_build_experience_prompt
    
    try:
        # Initialize the content writer
        writer = EnhancedContentWriterAgent()
        
        # Monkey patch the methods to add debugging
        original_format_role_info = writer._format_role_info
        original_build_prompt = writer._build_prompt
        original_build_experience_prompt = writer._build_experience_prompt
        
        writer._format_role_info = lambda content_item, generation_context: debug_format_role_info(writer, content_item, generation_context)
        writer._build_prompt = lambda job_data, content_item, generation_context, content_type: debug_build_prompt(writer, job_data, content_item, generation_context, content_type)
        writer._build_experience_prompt = lambda template, job_data, content_item, generation_context: debug_build_experience_prompt(writer, template, job_data, content_item, generation_context)
        
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
        
        print("=== Testing full workflow ===")
        print(f"Input data: {input_data}")
        
        # Run the full async workflow
        result = await writer.run_async(input_data, context)
        print(f"\nFinal result: {result}")
        
    except Exception as e:
        print(f"\n=== ERROR CAPTURED ===")
        print(f"Error: {e}")
        print(f"Error type: {type(e)}")
        print("\n=== FULL TRACEBACK ===")
        traceback.print_exc()
        print("\n=== END TRACEBACK ===")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_full_workflow())