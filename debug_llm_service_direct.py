#!/usr/bin/env python3
"""
Direct test of our LLM service to isolate the issue.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services.llm import EnhancedLLMService
from src.models.data_models import ContentType
import logging
logging.basicConfig(level=logging.INFO)

async def test_llm_service_direct():
    """Test our LLM service directly."""
    
    print("1. Initializing LLM service...")
    llm_service = EnhancedLLMService()
    
    # Simple test prompt
    simple_prompt = "Generate a list of 3 programming languages. Output only the language names, one per line."
    
    print("2. Testing simple prompt...")
    print(f"Prompt: {simple_prompt}")
    
    try:
        response = await llm_service.generate_content(
            prompt=simple_prompt,
            content_type=ContentType.QUALIFICATION
        )
        print(f"Response success: {response.success}")
        print(f"Response content: {response.content}")
        print(f"Response type: {type(response)}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50 + "\n")
    
    # Complex skills prompt
    skills_prompt = """You are an expert CV and LinkedIn profile skill generator. Your goal is to analyze the provided job description and generate a list of the 10 most relevant and impactful skills for the "Key Qualifications" section.

[Instructions for Skill Generation]
1. **Analyze Job Requirements**: Carefully read the job description to identify the most important technical skills, soft skills, and qualifications mentioned.
2. **Prioritize Relevance**: Focus on skills that are directly mentioned or strongly implied in the job posting.
3. **Synthesize Concisely**: Create skill names that are concise but descriptive (ideally under 30 characters).
4. **Ensure Variety**: Include a mix of technical skills, soft skills, and domain-specific knowledge as appropriate.
5. **Output Format**: Provide exactly 10 skills, each on a separate line, with no numbering, bullets, or additional formatting.

[Job Description]
Software Engineer position requiring Python, JavaScript, and cloud technologies.

[My Current Talents/Background]
Experienced developer with Python and web development background.

[Output Example]
Python Development
JavaScript Expertise
Cloud Technologies
Web Development
Software Engineering
Problem Solving
Collaboration
Agile Development
Version Control (Git)
Testing & Debugging

Now generate the 10 most relevant skills for this job:"""
    
    print("3. Testing skills generation prompt...")
    print(f"Prompt length: {len(skills_prompt)}")
    
    try:
        response = await llm_service.generate_content(
            prompt=skills_prompt,
            content_type=ContentType.QUALIFICATION
        )
        print(f"Response success: {response.success}")
        print(f"Response content length: {len(response.content)}")
        print(f"Response content: {response.content}")
        
        # Check if response contains the prompt
        if "You are an expert CV" in response.content:
            print("WARNING: Response contains the original prompt!")
        else:
            print("SUCCESS: Response appears to be generated content")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Check service stats
    print("\n" + "="*30 + "\n")
    print("4. LLM Service Stats:")
    stats = llm_service.get_service_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_llm_service_direct())