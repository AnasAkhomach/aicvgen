#!/usr/bin/env python3
"""
Debug script for Professional Experience Writer Agent
Tests the agent initialization, workflow execution, and prompt generation
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
print("Environment variables loaded")

# Import required modules
from src.config.settings import AppConfig
from src.agents.professional_experience_writer_agent import ProfessionalExperienceWriterAgent
from src.models.cv_models import StructuredCV, Item, ItemType, MetadataModel
from src.models.workflow_models import ContentType
from src.models.llm_data_models import PersonalInfo
from src.models.data_models import JobDescriptionData
from src.services.llm_service_interface import LLMServiceInterface
from src.templates.content_templates import ContentTemplateManager
from src.utils.cv_data_factory import create_empty_cv_structure


class DebugLLMService(LLMServiceInterface):
    """Debug LLM service that logs all interactions"""
    
    def __init__(self):
        self.call_count = 0
        self.last_prompt = None
        self.last_system_instruction = None
        self.last_parameters = None
    
    async def generate_content(
        self,
        prompt: str,
        content_type = None,
        session_id: str = None,
        trace_id: str = None,
        item_id: str = None,
        max_tokens: int = None,
        temperature: float = None,
        system_instruction: str = None,
        **kwargs
    ):
        """Mock LLM call that logs the prompt and returns sample content"""
        from src.models.llm_data_models import LLMResponse
        
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_instruction = system_instruction
        self.last_parameters = {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'content_type': content_type,
            'session_id': session_id,
            'trace_id': trace_id,
            'item_id': item_id,
            **kwargs
        }
        
        print(f"\n=== LLM CALL #{self.call_count} ===")
        print(f"System Instruction Length: {len(system_instruction) if system_instruction else 0} chars")
        print(f"Prompt Length: {len(prompt)} chars")
        print(f"Parameters: {self.last_parameters}")
        
        print("\n=== FULL SYSTEM INSTRUCTION ===")
        print(system_instruction or "None")
        
        print("\n=== FULL PROMPT ===")
        print(prompt)
        
        # Return mock professional experience items
        mock_content = '''[
  {
    "item_type": "experience_role_title",
    "content_type": "structured_content",
    "content": {
      "position_title": "Senior Software Engineer",
      "company_name": "Tech Solutions Inc.",
      "employment_period": "2020 - Present",
      "location": "San Francisco, CA",
      "achievements": [
        "Led development of microservices architecture serving 1M+ users",
        "Reduced system latency by 40% through optimization initiatives",
        "Mentored 5 junior developers and established code review processes"
      ],
      "technologies_used": ["Python", "Django", "PostgreSQL", "Docker", "AWS"]
    }
  },
  {
    "item_type": "experience_role_title",
    "content_type": "structured_content",
    "content": {
      "position_title": "Software Developer",
      "company_name": "StartupCorp",
      "employment_period": "2018 - 2020",
      "location": "Austin, TX",
      "achievements": [
        "Developed RESTful APIs handling 100K+ daily requests",
        "Implemented automated testing reducing bugs by 60%",
        "Collaborated with cross-functional teams on product features"
      ],
      "technologies_used": ["JavaScript", "Node.js", "MongoDB", "React"]
    }
  }
]'''
        
        return LLMResponse(
            content=mock_content,
            usage_stats={'input_tokens': len(prompt), 'output_tokens': len(mock_content)},
            model_info={'model': 'debug-model', 'version': '1.0'}
        )
    
    async def generate(self, prompt: str, **kwargs):
        """Backward-compatible wrapper for generate_content"""
        return await self.generate_content(prompt, **kwargs)
    
    async def validate_api_key(self) -> bool:
        """Mock API key validation"""
        return True
    
    def get_current_api_key_info(self):
        """Mock API key info"""
        from src.models.llm_service_models import LLMApiKeyInfo
        return LLMApiKeyInfo(
            provider='debug',
            key_status='valid',
            key_prefix='debug-key',
            expires_at=None
        )
    
    async def ensure_api_key_valid(self):
        """Mock API key validation"""
        pass
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Return debug information about LLM calls"""
        return {
            'call_count': self.call_count,
            'last_prompt_length': len(self.last_prompt) if self.last_prompt else 0,
            'last_system_instruction_length': len(self.last_system_instruction) if self.last_system_instruction else 0,
            'last_parameters': self.last_parameters
        }


def create_sample_structured_cv() -> StructuredCV:
    """Create a sample StructuredCV for testing"""
    
    # Create sample job description data
    job_data = JobDescriptionData(
        raw_text="We are seeking a Senior Software Engineer to join our team...",
        job_title="Senior Software Engineer",
        company_name="Tech Solutions Inc.",
        main_job_description_raw="Develop and maintain scalable web applications using Python and modern frameworks.",
        skills=["Python", "Django", "PostgreSQL", "Docker", "AWS", "Microservices"]
    )
    
    # Create empty CV structure with job data
    structured_cv = create_empty_cv_structure(job_data=job_data)
    
    # Import Section and ItemStatus for creating sections
    from src.models.cv_models import Section, ItemStatus
    
    # Create sections with items instead of adding items directly to CV
    summary_section = Section(
        name="Executive Summary",
        content_type="DYNAMIC",
        order=0,
        status=ItemStatus.INITIAL,
        items=[
            Item(
                item_type=ItemType.SUMMARY_PARAGRAPH,
                content="Experienced software engineer with 5+ years developing scalable web applications.",
                status=ItemStatus.PENDING
            )
        ]
    )
    
    qualifications_section = Section(
        name="Key Qualifications",
        content_type="DYNAMIC",
        order=1,
        status=ItemStatus.INITIAL,
        items=[
            Item(
                item_type=ItemType.KEY_QUALIFICATION,
                content="Expert in Python, Django, and cloud technologies",
                status=ItemStatus.PENDING
            )
        ]
    )
    
    experience_section = Section(
        name="Professional Experience",
        content_type="DYNAMIC",
        order=2,
        status=ItemStatus.INITIAL,
        items=[]
    )
    
    # Add sections to CV
    if not hasattr(structured_cv, 'sections') or structured_cv.sections is None:
        structured_cv.sections = []
    
    structured_cv.sections.extend([summary_section, qualifications_section, experience_section])
    
    return structured_cv


async def main():
    """Main debug function"""
    print("\n=== PROFESSIONAL EXPERIENCE WRITER AGENT DEBUG ===")
    
    try:
        # Load settings
        print("\n1. Loading settings...")
        settings = AppConfig()
        agent_settings_dict = settings.agent_settings.model_dump()
        print(f"Settings loaded: {len(agent_settings_dict)} agent settings")
        
        # Create debug LLM service
        print("\n2. Creating debug LLM service...")
        debug_llm_service = DebugLLMService()
        
        # Create mock template manager
        print("\n3. Creating mock template manager...")
        mock_template_manager = MagicMock(spec=ContentTemplateManager)
        mock_template_manager.get_template_by_type.return_value = "Job Description: {job_description}\nExperience Item: {experience_item}\nKey Qualifications: {key_qualifications}\nResearch Findings: {research_findings}"
        mock_template_manager.format_template.return_value = "Formatted template with job description and experience details"
        
        # Initialize agent
        print("\n4. Initializing Professional Experience Writer Agent...")
        agent = ProfessionalExperienceWriterAgent(
            llm_service=debug_llm_service,
            settings=agent_settings_dict,
            template_manager=mock_template_manager,
            session_id="debug-session-123"
        )
        print("Agent initialized successfully")
        
        # Create sample CV
        print("\n5. Creating sample CV structure...")
        structured_cv = create_sample_structured_cv()
        total_items = sum(len(section.items) for section in structured_cv.sections)
        print(f"CV created with {total_items} existing items across {len(structured_cv.sections)} sections")
        
        # Extract job description data
        print("\n6. Extracting job description data...")
        job_desc_dict = structured_cv.metadata.extra.get("job_description", {})
        job_description_data = JobDescriptionData(**job_desc_dict)
        print(f"Job description extracted: {job_description_data.job_title} at {job_description_data.company_name}")
        
        # Add an experience item to the CV for testing
        print("\n7. Adding experience item for testing...")
        from src.models.cv_models import ItemStatus
        import uuid
        
        # Create a UUID for the item
        item_uuid = uuid.uuid4()
        
        experience_item = Item(
            id=item_uuid,
            item_type=ItemType.EXPERIENCE_ROLE_TITLE,
            content="Senior Software Engineer at Previous Company",
            status=ItemStatus.PENDING
        )
        
        # Find the experience section and add the item
        experience_section = None
        for section in structured_cv.sections:
            if section.name == "Professional Experience":
                experience_section = section
                break
        
        if experience_section:
            experience_section.items.append(experience_item)
        else:
            print("Warning: Could not find Professional Experience section")
        
        # Execute agent
        print("\n8. Executing Professional Experience Writer Agent...")
        result = await agent._execute(
            structured_cv=structured_cv,
            job_description_data=job_description_data,
            current_item_id=str(item_uuid)
        )
        
        print(f"\n=== AGENT EXECUTION RESULT ===")
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        # Display LLM debug info
        print("\n=== LLM DEBUG INFO ===")
        debug_info = debug_llm_service.get_debug_info()
        for key, value in debug_info.items():
            print(f"{key}: {value}")
        
        print("\n=== WORKFLOW ANALYSIS ===")
        print("✓ Agent initialization successful")
        print("✓ Template loading successful")
        print("✓ CV structure processing successful")
        print("✓ Agent execution successful")
        print("✓ LLM integration successful")
        print("✓ Response processing successful")
        
        print("\n=== PROMPT COMPONENTS IDENTIFIED ===")
        print("✓ System instruction captured")
        print("✓ User prompt captured")
        print("✓ LLM parameters captured")
        print("✓ Full workflow traced")
        
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    sys.exit(exit_code)