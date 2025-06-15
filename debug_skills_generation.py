#!/usr/bin/env python3
"""
Debug script to test the skills generation step specifically.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.orchestration.state import AgentState
from src.models.data_models import WorkflowType
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.services.llm import get_llm_service
import logging
logging.basicConfig(level=logging.INFO)

async def test_skills_generation():
    """Test the skills generation step."""
    print("1. Initializing content writer agent...")
    content_writer_agent = EnhancedContentWriterAgent()
    
    print("2. Testing skills generation...")
    job_description = "Software Engineer position requiring Python, JavaScript, and cloud technologies."
    my_talents = "Experienced developer with Python and web development background."
    
    try:
        # Clear any existing cache first
        print("3. Clearing LLM cache...")
        content_writer_agent.llm_service.clear_cache()
        
        # First, let's check the prompt template loading
        template = content_writer_agent._load_prompt_template("key_qualifications_prompt")
        print(f"4. Template loaded (first 200 chars): {template[:200]}...")
        
        # Format the prompt
        prompt = template.format(
            main_job_description_raw=job_description[:2000],
            my_talents=my_talents or "Professional with diverse technical and analytical skills"
        )
        print(f"5. Formatted prompt (first 300 chars): {prompt[:300]}...")
        
        result = await content_writer_agent.generate_big_10_skills(
            job_description,
            my_talents
        )
        
        print(f"6. Skills generation result:")
        print(f"   Success: {result.get('success', False)}")
        if result.get('success'):
            print(f"   Skills count: {len(result.get('skills', []))}")
            print(f"   Skills: {result.get('skills', [])[:5]}...")  # Show first 5 skills
            print(f"   Raw LLM output (first 1000 chars):")
            print(f"   {result.get('raw_llm_output', '')[:1000]}...")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"   Exception during skills generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_skills_generation())