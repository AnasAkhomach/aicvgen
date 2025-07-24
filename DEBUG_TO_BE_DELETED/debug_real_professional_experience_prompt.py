#!/usr/bin/env python3
"""
Debug script to show the exact prompt passed to ProfessionalExperienceWriterAgent
without any mocks or templates - showing the real prompt construction.
"""

import asyncio
import os
import sys
from pathlib import Path
from uuid import uuid4

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
from src.models.cv_models import StructuredCV, Section, Item, ItemType, JobDescriptionData, ItemStatus
from src.models.llm_data_models import LLMResponse
from src.templates.content_templates import ContentTemplateManager
from src.config.settings import AgentSettings
from src.agents.professional_experience_writer_agent import ProfessionalExperienceWriterAgent


class PromptCaptureLLMService:
    """LLM service that captures and displays the exact prompt."""
    
    def __init__(self):
        self.captured_prompts = []
        self.captured_system_instructions = []
    
    async def generate_content(
        self,
        prompt: str,
        system_instruction: str = None,
        **kwargs
    ) -> LLMResponse:
        """Capture the prompt and system instruction."""
        print("\n" + "="*80)
        print("EXACT PROMPT PASSED TO PROFESSIONAL EXPERIENCE WRITER AGENT")
        print("="*80)
        
        if system_instruction:
            print("\nSYSTEM INSTRUCTION:")
            print("-" * 40)
            print(system_instruction)
            self.captured_system_instructions.append(system_instruction)
        
        print("\nUSER PROMPT:")
        print("-" * 40)
        print(prompt)
        self.captured_prompts.append(prompt)
        
        print("\n" + "="*80)
        print("END OF PROMPT CAPTURE")
        print("="*80)
        
        # Return a mock LLMResponse
        return LLMResponse(
            content='{"role_description": "Mock response", "organization_description": "Mock org", "bullet_points": ["Mock bullet point"]}',
            tokens_used=100,
            processing_time=0.5,
            model_used="mock-model",
            success=True
        )


def create_sample_structured_cv():
    """Create a sample structured CV with experience data."""
    cv = StructuredCV()
    
    # Create experience section
    experience_section = Section(
        id=uuid4(),
        name="Professional Experience",
        content_type="DYNAMIC",
        order=2,
        status=ItemStatus.INITIAL,
        subsections=[],
        items=[]
    )
    
    # Create an experience item
    experience_item_id = str(uuid4())
    experience_item = Item(
        id=experience_item_id,
        item_type=ItemType.EXPERIENCE_ROLE_TITLE,
        content="Senior Software Engineer",
        metadata={
            "company": "Tech Corp",
            "start_date": "2020-01",
            "end_date": "2023-12",
            "location": "San Francisco, CA"
        }
    )
    
    experience_section.items.append(experience_item)
    cv.sections.append(experience_section)
    
    return cv, experience_item_id


def create_sample_job_description():
    """Create sample job description data."""
    return JobDescriptionData(
        raw_text="Senior Python Developer at Innovation Labs. We are looking for an experienced Python developer to join our remote team. Requirements: 5+ years Python experience, web frameworks, strong problem-solving skills. Responsibilities include developing scalable web applications, leading technical initiatives, and mentoring junior developers.",
        job_title="Senior Python Developer",
        company_name="Innovation Labs",
        requirements=[
            "5+ years Python experience",
            "Experience with web frameworks",
            "Strong problem-solving skills"
        ],
        responsibilities=[
            "Develop scalable web applications",
            "Lead technical initiatives",
            "Mentor junior developers"
        ],
        skills=["Python", "Django", "PostgreSQL", "AWS"],
        location="Remote",
        employment_type="Full-time"
    )


def create_sample_research_findings():
    """Create sample research findings data."""
    return {
        "industry_insights": [
            "Python development market is growing rapidly",
            "Remote work is becoming standard in tech",
            "Full-stack skills are highly valued"
        ],
        "company_insights": [
            "Innovation Labs focuses on cutting-edge technology",
            "Company values technical leadership and mentoring",
            "Strong emphasis on scalable architecture"
        ],
        "role_insights": [
            "Senior developers expected to lead initiatives",
            "Mentoring junior developers is a key responsibility",
            "Web framework expertise is essential"
        ]
    }


async def debug_real_professional_experience_prompt():
    """Debug the real prompt construction for ProfessionalExperienceWriterAgent."""
    print("Starting Professional Experience Writer Agent prompt debugging...")
    
    # Create real template manager
    template_manager = ContentTemplateManager()
    
    # Load real agent settings
    agent_settings = AgentSettings()
    settings_dict = {
        "writer_agent_system_instruction": agent_settings.writer_agent_system_instruction,
        "max_tokens_content_generation": 1024,
        "temperature_content_generation": 0.7
    }
    
    # Create prompt capture LLM service
    llm_service = PromptCaptureLLMService()
    
    # Create the agent with real dependencies
    agent = ProfessionalExperienceWriterAgent(
        llm_service=llm_service,
        template_manager=template_manager,
        settings=settings_dict,
        session_id="debug-session"
    )
    
    # Create sample data
    structured_cv, experience_item_id = create_sample_structured_cv()
    job_description_data = create_sample_job_description()
    research_findings = create_sample_research_findings()
    
    # Execute the agent to capture the real prompt
    try:
        result = await agent._execute(
            structured_cv=structured_cv,
            job_description_data=job_description_data,
            current_item_id=experience_item_id,
            research_findings=research_findings
        )
        
        print("\nAgent execution completed successfully.")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"\nAgent execution failed: {e}")
        print("But the prompt was still captured above.")
    
    print(f"\nTotal prompts captured: {len(llm_service.captured_prompts)}")
    print(f"Total system instructions captured: {len(llm_service.captured_system_instructions)}")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the debug function
    asyncio.run(debug_real_professional_experience_prompt())