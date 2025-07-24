#!/usr/bin/env python3
"""
Simple test script to verify ResearchAgent improvements.
"""

import sys
import os
import asyncio
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Set up logging to see debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_debug.log')
    ]
)

from src.core.application_startup import get_container
from src.models.cv_models import JobDescriptionData, StructuredCV

async def test_research_agent():
    """Test the ResearchAgent with sample data."""
    print("Testing ResearchAgent with improved prompt and parsing...")
    
    # Get container and research agent
    container = get_container()
    research_agent = container.research_agent()
    
    # Create sample job description data
    job_data = JobDescriptionData(
        raw_text="We are looking for a Senior Python Developer with experience in Django, FastAPI, and cloud technologies. The ideal candidate should have strong problem-solving skills, experience with microservices, and ability to work in an agile environment.",
        job_title="Senior Python Developer",
        company_name="Tech Corp",
        skills=["Python", "Django", "FastAPI", "AWS", "Docker"],
        responsibilities=["Develop web applications", "Design APIs", "Code review"],
        industry_terms=["microservices", "agile", "cloud technologies"],
        company_values=["innovation", "collaboration", "excellence"]
    )
    
    # Create sample CV data
    cv_data = StructuredCV.create_empty(
        cv_text="John Doe - Python Developer with 3 years experience in Django and web development.",
        job_data=job_data
    )
    
    try:
        # Execute research analysis
        print("\nExecuting research analysis...")
        result = await research_agent._execute(
            job_description_data=job_data,
            structured_cv=cv_data
        )
        
        print(f"\nResearch completed!")
        print(f"Success: {result.success}")
        print(f"Message: {result.metadata.get('message', 'No message available')}")
        if result.success and result.output_data:
            findings = result.output_data.research_findings
            print(f"Research status: {findings.status}")
            if findings.role_insights:
                print(f"Role insights: {findings.role_insights.role_title}")
            if findings.company_insights:
                print(f"Company insights: {findings.company_insights.company_name}")
            if findings.industry_insights:
                print(f"Industry insights: {findings.industry_insights.industry_name}")
        else:
            print(f"Error: {result.error_message}")
        
        return result.success
        
    except Exception as e:
        print(f"\nResearch failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_research_agent())
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)