import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test Phase 2 fixes
from agents.parser_agent import ParserAgent
from agents.enhanced_content_writer import EnhancedContentWriterAgent
from orchestration.workflow_definitions import WorkflowBuilder
from models.data_models import ContentType
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_phase2_fixes():
    print('=== Testing Phase 2: Data Structure Fixes ===')
    
    # Test 1: WorkflowBuilder data adaptation
    print('\n1. Testing WorkflowBuilder._adapt_for_content_writer...')
    workflow_builder = WorkflowBuilder()
    
    # Test with string job description (should be converted)
    test_input = {
        'parser_results': {
            'job_description_data': 'Software Engineer position requiring Python skills',
            'structured_cv': {
                'personal_info': {'name': 'John Doe'},
                'experience': [{'role': 'Developer'}],
                'projects': [{'name': 'Test Project'}]
            }
        }
    }
    
    try:
        adapted_data = workflow_builder._adapt_for_content_writer(
            test_input, ContentType.EXPERIENCE
        )
        job_desc_data = adapted_data['job_description_data']
        
        # Verify structure
        required_fields = ['raw_text', 'skills', 'experience_level', 'responsibilities', 'industry_terms', 'company_values']
        missing_fields = [f for f in required_fields if f not in job_desc_data]
        
        if not missing_fields:
            print('✓ WorkflowBuilder correctly structures job_description_data')
        else:
            print(f'✗ Missing fields in job_description_data: {missing_fields}')
            
        print(f'  - job_description_data type: {type(job_desc_data)}')
        print(f'  - Available fields: {list(job_desc_data.keys())}')
        
    except Exception as e:
        print(f'✗ WorkflowBuilder test failed: {e}')
    
    # Test 2: EnhancedContentWriter defensive programming
    print('\n2. Testing EnhancedContentWriter defensive programming...')
    
    try:
        content_writer = EnhancedContentWriterAgent()
        
        # Test with malformed input data
        test_cases = [
            # Case 1: String job_description_data
            {
                'job_description_data': 'Raw string job description',
                'content_item': {'type': 'experience', 'data': {}},
                'context': {}
            },
            # Case 2: Missing fields in job_description_data
            {
                'job_description_data': {'raw_text': 'Test job'},
                'content_item': {'type': 'experience', 'data': {}},
                'context': {}
            },
            # Case 3: None job_description_data
            {
                'job_description_data': None,
                'content_item': {'type': 'experience', 'data': {}},
                'context': {}
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            job_data_type = type(test_case['job_description_data']).__name__
            print(f'\n  Test case {i}: {job_data_type} job_description_data')
            try:
                # This should not raise an exception due to defensive programming
                result = await content_writer.run_async(test_case, {})
                print(f'  ✓ Case {i} handled gracefully')
            except Exception as e:
                print(f'  ✗ Case {i} failed: {e}')
    
    except Exception as e:
        print(f'✗ EnhancedContentWriter test setup failed: {e}')
    
    print('\n=== Phase 2 Testing Complete ===')

# Run the test
if __name__ == '__main__':
    asyncio.run(test_phase2_fixes())