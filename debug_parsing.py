import asyncio
from unittest.mock import patch, AsyncMock
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.parser_agent import ParserAgent
from src.models.data_models import CVParsingResult, JobDescriptionData

# Sample LLM response from the test fixture
sample_llm_response = {
    "personal_info": {
        "name": "John Doe",
        "email": "john.doe@email.com",
        "phone": "+1234567890",
        "linkedin": "https://linkedin.com/in/johndoe",
        "github": "https://github.com/johndoe",
        "location": "New York, NY"
    },
    "sections": [
        {
            "name": "Professional Experience",
            "items": [
                "Senior Software Engineer | Tech Corp | 2020-2023 | San Francisco, CA - Led development of web applications"
            ],
            "subsections": []
        },
        {
            "name": "Education",
            "items": [
                "Bachelor of Science in Computer Science | University of Technology | 2016-2020 | Boston, MA - Graduated with honors"
            ],
            "subsections": []
        }
    ]
}

async def test_conversion():
    print("Testing CVParsingResult creation from sample data...")
    
    try:
        # Test direct conversion
        parsing_result = CVParsingResult(**sample_llm_response)
        print(f"✓ CVParsingResult created successfully")
        print(f"Name: {parsing_result.personal_info.name}")
        print(f"Email: {parsing_result.personal_info.email}")
        
        # Test with parser agent
        parser_agent = ParserAgent(
            name="test_parser",
            description="Test parser agent"
        )
        
        sample_cv_text = """
John Doe
Email: john.doe@email.com | Phone: +1234567890
LinkedIn: https://linkedin.com/in/johndoe | GitHub: https://github.com/johndoe
Location: New York, NY

PROFESSIONAL EXPERIENCE
Senior Software Engineer | Tech Corp | 2020-2023 | San Francisco, CA
Led development of web applications using modern technologies.

EDUCATION
Bachelor of Science in Computer Science | University of Technology | 2016-2020 | Boston, MA
Graduated with honors, focusing on software engineering principles.
"""
        
        job_data = JobDescriptionData(
            raw_text="Software Engineer position",
            title="Software Engineer",
            company="Test Company",
            requirements=["Python", "JavaScript"],
            responsibilities=["Develop software"],
            benefits=["Health insurance"]
        )
        
        # Test without mocking first to see the actual error
        print("\nTesting parse_cv_with_llm without mocking...")
        try:
            result = await parser_agent.parse_cv_with_llm(sample_cv_text, job_data)
            print(f"✓ parse_cv_with_llm completed")
            print(f"Result metadata keys: {result.metadata.keys()}")
            
            if "parsing_error" in result.metadata:
                print(f"✗ Parsing error: {repr(result.metadata['parsing_error'])}")
            else:
                print(f"✓ Name found: {result.metadata.get('name', 'NOT FOUND')}")
                print(f"✓ Email found: {result.metadata.get('email', 'NOT FOUND')}")
        except Exception as e:
            print(f"✗ Exception during parse_cv_with_llm: {e}")
            
        # Now test with mocking
        print("\nTesting parse_cv_with_llm with mocking...")
        with patch.object(parser_agent, '_generate_and_parse_json', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = sample_llm_response
            
            result = await parser_agent.parse_cv_with_llm(sample_cv_text, job_data)
            print(f"✓ parse_cv_with_llm completed")
            print(f"Result metadata keys: {result.metadata.keys()}")
            
            if "parsing_error" in result.metadata:
                print(f"✗ Parsing error: {repr(result.metadata['parsing_error'])}")
            else:
                print(f"✓ Name found: {result.metadata.get('name', 'NOT FOUND')}")
                print(f"✓ Email found: {result.metadata.get('email', 'NOT FOUND')}")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_conversion())