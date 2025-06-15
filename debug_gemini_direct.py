#!/usr/bin/env python3
"""
Direct test of Gemini API to isolate the issue.
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini_direct():
    """Test Gemini API directly."""
    
    # Configure Gemini
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment")
        return
    
    genai.configure(api_key=api_key)
    
    # Create model
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Simple test prompt
    simple_prompt = "Generate a list of 3 programming languages. Output only the language names, one per line."
    
    print("Testing simple prompt...")
    print(f"Prompt: {simple_prompt}")
    
    try:
        response = model.generate_content(simple_prompt)
        print(f"Response: {response.text}")
        print(f"Response type: {type(response)}")
        print(f"Response attributes: {dir(response)}")
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
    
    print("Testing skills generation prompt...")
    print(f"Prompt length: {len(skills_prompt)}")
    
    try:
        response = model.generate_content(skills_prompt)
        print(f"Response: {response.text}")
        print(f"Response length: {len(response.text)}")
        
        # Check if response contains the prompt
        if "You are an expert CV" in response.text:
            print("WARNING: Response contains the original prompt!")
        else:
            print("SUCCESS: Response appears to be generated content")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gemini_direct()