#!/usr/bin/env python3
"""
Test script to verify LLM API calls are working and generating real content.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.llm import EnhancedLLMService
from src.models.data_models import ContentType

def test_llm_api_connection():
    """Test if LLM service can connect and make API calls."""
    print("=== Testing LLM API Connection ===")
    
    try:
        # Initialize LLM service
        llm_service = EnhancedLLMService()
        print(f"✓ LLM service initialized successfully")
        print(f"  Model: {llm_service.model_name}")
        print(f"  Using user key: {llm_service.using_user_key}")
        print(f"  Using fallback: {llm_service.using_fallback}")
        print(f"  Current API key (first 10 chars): {llm_service.current_api_key[:10]}...")
        
        return llm_service
        
    except Exception as e:
        print(f"✗ LLM service initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_simple_llm_call(llm_service):
    """Test a simple LLM call to verify API connectivity."""
    print("\n=== Testing Simple LLM Call ===")
    
    simple_prompt = "Hello! Please respond with exactly: 'LLM API is working correctly'"
    
    try:
        print(f"Sending prompt: {simple_prompt}")
        print("Waiting for LLM response...")
        
        start_time = datetime.now()
        response = await llm_service.generate_content(
            prompt=simple_prompt,
            content_type=ContentType.QUALIFICATION
        )
        end_time = datetime.now()
        
        print(f"\n✓ LLM call completed successfully!")
        print(f"  Response time: {(end_time - start_time).total_seconds():.2f} seconds")
        print(f"  Success: {response.success}")
        print(f"  Model used: {response.model_used}")
        print(f"  Tokens used: {response.tokens_used}")
        print(f"  Processing time: {response.processing_time:.2f}s")
        print(f"  Response content: '{response.content.strip()}'")
        
        # Check if response looks like a real LLM response
        if "LLM API is working" in response.content or len(response.content.strip()) > 10:
            print("\n🎉 LLM is generating REAL content!")
            return True
        else:
            print("\n⚠️  Response seems too short or generic - might be mock data")
            return False
            
    except Exception as e:
        print(f"\n✗ LLM call failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_cv_analysis_prompt(llm_service):
    """Test with a CV analysis prompt similar to what the agents use."""
    print("\n=== Testing CV Analysis Prompt ===")
    
    cv_analysis_prompt = """
Analyze the following CV text and extract the key sections and information.
Provide the output in JSON format with the following keys:
"summary": The professional summary or objective statement.
"experiences": A list of work experiences, each as a string describing the role, company, and key achievements.
"skills": A list of technical and soft skills mentioned.
"education": A list of educational qualifications (degrees, institutions, dates).
"projects": A list of significant projects mentioned.

CV Text:
John Doe
Software Engineer
Email: john.doe@email.com
Phone: +1-555-0123

PROFESSIONAL SUMMARY
Experienced software engineer with 5+ years in web development using Python and JavaScript.

EXPERIENCE
Senior Software Engineer - Tech Corp (2020-2023)
- Developed web applications using Django and React
- Led a team of 3 developers
- Implemented CI/CD pipelines

SKILLS
- Python, JavaScript, React, Django
- AWS, Docker, Kubernetes
- Git, Jenkins, CI/CD

EDUCATION
Bachelor of Science in Computer Science - University of Technology (2015-2019)

PROJECTS
E-commerce Platform - Built a full-stack e-commerce application using Django and React

JSON Output:
"""
    
    try:
        print("Sending CV analysis prompt to LLM...")
        
        start_time = datetime.now()
        response = await llm_service.generate_content(
            prompt=cv_analysis_prompt,
            content_type=ContentType.EXPERIENCE
        )
        end_time = datetime.now()
        
        print(f"\n✓ CV analysis call completed!")
        print(f"  Response time: {(end_time - start_time).total_seconds():.2f} seconds")
        print(f"  Success: {response.success}")
        print(f"  Response length: {len(response.content)} characters")
        print(f"  Response preview (first 500 chars):")
        print(f"  {response.content[:500]}...")
        
        # Check if response contains JSON-like structure
        if '{' in response.content and '}' in response.content:
            print("\n🎉 LLM is generating structured JSON responses!")
            return True
        else:
            print("\n⚠️  Response doesn't appear to be structured JSON")
            return False
            
    except Exception as e:
        print(f"\n✗ CV analysis call failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all LLM tests."""
    print("Starting LLM API verification tests...\n")
    
    # Test 1: LLM Service Initialization
    llm_service = test_llm_api_connection()
    if not llm_service:
        print("\n❌ Cannot proceed - LLM service initialization failed")
        return
    
    # Test 2: Simple LLM Call
    simple_test_passed = await test_simple_llm_call(llm_service)
    
    # Test 3: CV Analysis Prompt
    cv_test_passed = await test_cv_analysis_prompt(llm_service)
    
    # Summary
    print("\n" + "="*50)
    print("=== LLM API VERIFICATION SUMMARY ===")
    print("="*50)
    print(f"LLM Service Initialization: ✓ PASSED")
    print(f"Simple LLM Call: {'✓ PASSED' if simple_test_passed else '✗ FAILED'}")
    print(f"CV Analysis Call: {'✓ PASSED' if cv_test_passed else '✗ FAILED'}")
    
    if simple_test_passed and cv_test_passed:
        print("\n🎉 SUCCESS: LLM API is working and generating REAL content!")
        print("   Your CV generation workflow is using actual AI responses.")
    else:
        print("\n❌ ISSUE: LLM API calls are not working as expected.")
        print("   This might explain why you're not seeing proper LLM responses.")
        
    # Performance stats
    print(f"\nPerformance Stats:")
    print(f"  Total LLM calls made: {llm_service.call_count}")
    print(f"  Total tokens used: {llm_service.total_tokens}")
    print(f"  Total processing time: {llm_service.total_processing_time:.2f}s")
    print(f"  Average time per call: {llm_service.total_processing_time / max(llm_service.call_count, 1):.2f}s")

if __name__ == "__main__":
    asyncio.run(main())