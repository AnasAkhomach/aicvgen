#!/usr/bin/env python3
"""
Simple test script to verify the AttributeError fixes in EnhancedContentWriter.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_format_role_info_with_string():
    """Test that _format_role_info can handle string input gracefully."""
    
    # Import here to avoid import issues
    from agents.enhanced_content_writer import EnhancedContentWriterAgent
    from models.data_models import ContentType
    
    # Create agent instance
    agent = EnhancedContentWriterAgent(
        name="TestAgent",
        description="Test agent",
        content_type=ContentType.EXPERIENCE
    )
    
    # Test case 1: String input (this was causing the AttributeError)
    print("Testing string input...")
    try:
        result = agent._format_role_info(
            "This is a CV text string that should be parsed",
            {"target_skills": ["Python", "Data Analysis"]}
        )
        print("✓ String input handled successfully")
        print(f"Result: {result[:100]}...")
    except Exception as e:
        print(f"✗ String input failed: {e}")
        return False
    
    # Test case 2: Dictionary with string roles (workflow adapter scenario)
    print("\nTesting dictionary with string roles...")
    try:
        test_data = {
            "data": {
                "roles": [
                    "Senior Developer - Tech Corp (2020-2023)\n• Developed Python applications\n• Led team of 5 developers"
                ]
            }
        }
        result = agent._format_role_info(
            test_data,
            {"target_skills": ["Python", "Leadership"]}
        )
        print("✓ Dictionary with string roles handled successfully")
        print(f"Result: {result[:100]}...")
    except Exception as e:
        print(f"✗ Dictionary with string roles failed: {e}")
        return False
    
    # Test case 3: Empty/invalid input
    print("\nTesting empty input...")
    try:
        result = agent._format_role_info(
            {},
            {"target_skills": []}
        )
        print("✓ Empty input handled successfully")
        print(f"Result: {result[:100]}...")
    except Exception as e:
        print(f"✗ Empty input failed: {e}")
        return False
    
    return True

def test_parse_cv_text():
    """Test the improved CV text parsing."""
    
    from agents.enhanced_content_writer import EnhancedContentWriterAgent
    from models.data_models import ContentType
    
    agent = EnhancedContentWriterAgent(
        name="TestAgent",
        description="Test agent",
        content_type=ContentType.EXPERIENCE
    )
    
    # Test CV text
    cv_text = """
    Senior Software Engineer - TechCorp Inc (2020-2023)
    • Developed scalable web applications using Python and Django
    • Led a team of 5 junior developers
    • Implemented CI/CD pipelines reducing deployment time by 50%
    
    Data Analyst - DataCorp (2018-2020)
    • Analyzed large datasets using SQL and Python
    • Created automated reporting dashboards
    • Improved data processing efficiency by 30%
    """
    
    print("Testing CV text parsing...")
    try:
        result = agent._parse_cv_text_to_content_item(
            cv_text,
            {"target_skills": ["Python", "Leadership"]}
        )
        print("✓ CV text parsing successful")
        print(f"Parsed structure: {result}")
        return True
    except Exception as e:
        print(f"✗ CV text parsing failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing EnhancedContentWriter fixes...\n")
    
    success = True
    
    # Test 1: Format role info with various inputs
    if not test_format_role_info_with_string():
        success = False
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Parse CV text
    if not test_parse_cv_text():
        success = False
    
    print("\n" + "="*50)
    
    if success:
        print("\n🎉 All tests passed! The AttributeError fixes are working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    return success

if __name__ == "__main__":
    main()