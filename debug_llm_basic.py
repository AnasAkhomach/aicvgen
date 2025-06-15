#!/usr/bin/env python3
"""
Basic LLM service test to check if it's working correctly.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services.llm import get_llm_service
from src.models.data_models import ContentType
import logging
logging.basicConfig(level=logging.INFO)

async def test_basic_llm():
    """Test basic LLM functionality."""
    print("1. Getting LLM service...")
    llm_service = get_llm_service()
    
    print("2. Testing skills generation prompt...")
    skills_prompt = """You are an expert CV and LinkedIn profile skill generator. Your goal is to analyze the provided job description and generate a list of the 10 most relevant and impactful skills for a candidate's "Key Qualifications" section.

[Instructions for Skill Generation]
1. **Analyze Job Description:** Carefully read the main job description below. Pay close attention to sections like "Required Qualifications," "Responsibilities," "Ideal Candidate," and "Skills." Prioritize skills mentioned frequently and those listed as essential requirements.

2. **Identify Key Skills:** Extract the 10 most critical core skills and competencies sought by the employer.

3. **Synthesize and Condense:** Rephrase the skills to be concise and impactful. Aim for action-oriented phrases that highlight capabilities. Each skill phrase should be **no longer than 30 characters**.

4. **Format Output:** Return the 10 skills as a simple, plain text, newline-separated list. Do not use bullet points, numbers, or any other formatting.

5. **Generate the "Big 10" Skills:** Create exactly 10 skills that are:
    * Highly relevant to the job description.
    * Concise (under 30 characters).
    * Action-oriented and impactful.
    * Directly aligned with employer requirements.

[Job Description]
Software Engineer position requiring Python, JavaScript, and cloud technologies.

[Additional Context & Talents to Consider]
Experienced developer with Python and web development background.

[Output Example]
Data Analysis & Insights
Python for Machine Learning
Strategic Business Planning
Cloud Infrastructure Management
Agile Project Leadership
Advanced SQL & Database Design
Cross-Functional Communication
MLOps & Model Deployment
Stakeholder Presentations
Process Automation & Optimization"""
    
    try:
        response = await llm_service.generate_content(
            prompt=skills_prompt,
            content_type=ContentType.QUALIFICATION
        )
        
        print(f"3. LLM Response:")
        print(f"   Success: {response.success}")
        print(f"   Content: {response.content}")
        print(f"   Content length: {len(response.content)}")
        print(f"   Model used: {response.model_used}")
        
        # Parse the response like the skills generation does
        lines = [line.strip() for line in response.content.split('\n') if line.strip()]
        print(f"   Parsed lines count: {len(lines)}")
        print(f"   First 5 lines: {lines[:5]}")
        
    except Exception as e:
        print(f"   Exception during LLM call: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_basic_llm())