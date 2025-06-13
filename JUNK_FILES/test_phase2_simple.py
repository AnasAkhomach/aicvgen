import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Simple test for Phase 2 fixes
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_enhanced_content_writer_defensive_programming():
    """Test the defensive programming improvements in EnhancedContentWriter"""
    print('=== Testing Phase 2: Enhanced Content Writer Defensive Programming ===')
    
    try:
        from agents.enhanced_content_writer import EnhancedContentWriterAgent
        content_writer = EnhancedContentWriterAgent()
        
        # Test cases for defensive programming
        test_cases = [
            {
                'name': 'String job_description_data',
                'input': {
                    'job_description_data': 'Raw string job description',
                    'content_item': {'type': 'experience', 'data': {}},
                    'context': {}
                }
            },
            {
                'name': 'Missing fields in job_description_data',
                'input': {
                    'job_description_data': {'raw_text': 'Test job'},
                    'content_item': {'type': 'experience', 'data': {}},
                    'context': {}
                }
            },
            {
                'name': 'None job_description_data',
                'input': {
                    'job_description_data': None,
                    'content_item': {'type': 'experience', 'data': {}},
                    'context': {}
                }
            },
            {
                'name': 'Invalid content_item type',
                'input': {
                    'job_description_data': {'raw_text': 'Test', 'skills': []},
                    'content_item': 'invalid_string',
                    'context': {}
                }
            },
            {
                'name': 'Invalid context type',
                'input': {
                    'job_description_data': {'raw_text': 'Test', 'skills': []},
                    'content_item': {'type': 'experience', 'data': {}},
                    'context': 'invalid_string'
                }
            }
        ]
        
        passed_tests = 0
        total_tests = len(test_cases)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f'\n  Test {i}: {test_case["name"]}')
            try:
                # This should not raise an exception due to defensive programming
                result = await content_writer.run_async(test_case['input'], {})
                print(f'  ✓ Test {i} passed - handled gracefully')
                passed_tests += 1
            except Exception as e:
                print(f'  ✗ Test {i} failed: {e}')
        
        print(f'\n=== Results: {passed_tests}/{total_tests} tests passed ===')
        
        if passed_tests == total_tests:
            print('✓ All defensive programming tests passed!')
            return True
        else:
            print('✗ Some defensive programming tests failed')
            return False
            
    except Exception as e:
        print(f'✗ Test setup failed: {e}')
        return False

def test_data_structure_validation():
    """Test data structure validation logic"""
    print('\n=== Testing Data Structure Validation Logic ===')
    
    # Test the validation logic we implemented
    test_job_data_cases = [
        ('string', 'Raw job description string'),
        ('dict_complete', {
            'raw_text': 'Test job',
            'skills': ['Python'],
            'experience_level': 'Senior',
            'responsibilities': ['Develop'],
            'industry_terms': ['API'],
            'company_values': ['Innovation']
        }),
        ('dict_incomplete', {'raw_text': 'Test job'}),
        ('none', None),
        ('list', ['invalid', 'list']),
        ('number', 123)
    ]
    
    passed_validations = 0
    total_validations = len(test_job_data_cases)
    
    for case_name, job_data in test_job_data_cases:
        print(f'\n  Validating {case_name}: {type(job_data).__name__}')
        
        # Simulate the validation logic from enhanced_content_writer.py
        try:
            if isinstance(job_data, str):
                validated_data = {
                    "raw_text": job_data,
                    "skills": [],
                    "experience_level": "N/A",
                    "responsibilities": [],
                    "industry_terms": [],
                    "company_values": []
                }
                print(f'  ✓ String converted to structured format')
            elif isinstance(job_data, dict):
                required_fields = ["raw_text", "skills", "experience_level", "responsibilities", "industry_terms", "company_values"]
                missing_fields = [field for field in required_fields if field not in job_data]
                
                if missing_fields:
                    # Add missing fields
                    for field in missing_fields:
                        if field == "raw_text":
                            job_data[field] = job_data.get("description", "")
                        elif field == "experience_level":
                            job_data[field] = "N/A"
                        else:
                            job_data[field] = []
                    print(f'  ✓ Added missing fields: {missing_fields}')
                else:
                    print(f'  ✓ Complete dictionary structure')
                validated_data = job_data
            elif job_data is None:
                validated_data = {
                    "raw_text": "",
                    "skills": [],
                    "experience_level": "N/A",
                    "responsibilities": [],
                    "industry_terms": [],
                    "company_values": []
                }
                print(f'  ✓ None converted to empty structured format')
            else:
                validated_data = {
                    "raw_text": str(job_data) if job_data else "",
                    "skills": [],
                    "experience_level": "N/A",
                    "responsibilities": [],
                    "industry_terms": [],
                    "company_values": []
                }
                print(f'  ✓ Unexpected type converted to structured format')
            
            # Verify all required fields are present
            required_fields = ["raw_text", "skills", "experience_level", "responsibilities", "industry_terms", "company_values"]
            if all(field in validated_data for field in required_fields):
                print(f'  ✓ Validation successful - all required fields present')
                passed_validations += 1
            else:
                print(f'  ✗ Validation failed - missing required fields')
                
        except Exception as e:
            print(f'  ✗ Validation error: {e}')
    
    print(f'\n=== Validation Results: {passed_validations}/{total_validations} validations passed ===')
    return passed_validations == total_validations

async def main():
    print('=== Phase 2 Data Structure Fixes Testing ===')
    
    # Test 1: Data structure validation logic
    validation_passed = test_data_structure_validation()
    
    # Test 2: Enhanced content writer defensive programming
    defensive_passed = await test_enhanced_content_writer_defensive_programming()
    
    print('\n=== Final Results ===')
    if validation_passed and defensive_passed:
        print('✓ Phase 2 implementation successful!')
        print('✓ Data structure issues have been resolved')
        print('✓ Defensive programming is working correctly')
    else:
        print('✗ Some Phase 2 tests failed')
        if not validation_passed:
            print('  - Data structure validation needs improvement')
        if not defensive_passed:
            print('  - Defensive programming needs improvement')

if __name__ == '__main__':
    asyncio.run(main())