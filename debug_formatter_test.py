#!/usr/bin/env python3

import sys
sys.path.append('.')

from unittest.mock import Mock, patch
from src.agents.formatter_agent import FormatterAgent

def test_formatter_directly():
    """Test the formatter agent directly to debug the issue."""
    
    # Create formatter agent
    formatter_agent = FormatterAgent(
        name="TestFormatterAgent",
        description="Test formatter agent"
    )
    
    # Mock the LLM service directly on the agent instance
    mock_llm = Mock()
    mock_llm.generate_response.return_value = """
# Tailored CV

## Professional Profile

Test summary

---

## Key Qualifications

Python, Machine Learning

---

## Professional Experience

• Developed applications using Python and machine learning frameworks
• Led cross-functional team of 5 engineers to deliver project on time

---
""".strip()
    formatter_agent.llm_service = mock_llm
    
    input_data = {
        "content_data": {
            "summary": "Test summary",
            "skills_section": "Python, Machine Learning",
            "experience_bullets": [
                "Developed applications using Python and machine learning frameworks",
                "Led cross-functional team of 5 engineers to deliver project on time"
            ]
        },
        "format_specs": {}
    }
    
    try:
        result = formatter_agent.run(input_data)
        print(f"Result: {result}")
        
        if "formatted_cv_text" in result:
            print(f"\nFormatted CV text: {repr(result['formatted_cv_text'])}")
            print(f"\nTest summary in result: {'Test summary' in result['formatted_cv_text']}")
        else:
            print("No formatted_cv_text in result")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_formatter_directly()