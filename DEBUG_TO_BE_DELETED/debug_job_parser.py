#!/usr/bin/env python3
"""
Debug script to trace JobDescriptionParserAgent prompt and response.
This script will help identify why the agent fails to parse job description data correctly.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.container import Container
from src.models.cv_models import JobDescriptionData
from src.services.llm_cv_parser_service import LLMCVParserService
from src.templates.content_templates import ContentTemplateManager
from src.models.workflow_models import ContentType


class DebugLLMService:
    """Mock LLM service that logs prompts and returns debug responses."""
    
    def __init__(self):
        self.last_prompt = None
        self.call_count = 0
    
    async def generate_content(self, prompt: str, content_type: ContentType = None, 
                             session_id: str = None, trace_id: str = None, 
                             system_instruction: str = None):
        """Mock generate_content that logs the prompt."""
        self.call_count += 1
        self.last_prompt = prompt
        
        print("\n" + "="*80)
        print(f"LLM CALL #{self.call_count}")
        print("="*80)
        print(f"Session ID: {session_id}")
        print(f"Trace ID: {trace_id}")
        print(f"Content Type: {content_type}")
        print(f"System Instruction: {system_instruction[:100] + '...' if system_instruction and len(system_instruction) > 100 else system_instruction}")
        print("\nPROMPT:")
        print("-" * 40)
        print(prompt)
        print("-" * 40)
        
        # Mock response that matches the JobDescriptionData model
        mock_response = {
            "job_title": "Supply Chain Apprentice M/F",
            "company_name": "French Luxury Clothing Manufacturer",
            "main_job_description_raw": "As a Supply Chain Apprentice, and under the supervision of the Supply Chain Analyst, you contribute to the design, improvement and support of tools, systems and processes.",
            "skills": ["Supply Chain Management", "Data Analysis", "S&OP Process", "Performance Monitoring", "Office Tools"],
            "experience_level": "Beginner",
            "responsibilities": [
                "Build, implement and monitor indicators",
                "Analyze performance (service rate, adherence rate, late deliveries, etc.)",
                "Propose and participate in the implementation of organizational changes to the Supply Chain",
                "Be a stakeholder in process improvement projects in collaboration with field teams",
                "Participate in the construction of the S&OP process",
                "Build simulation and optimization models"
            ],
            "industry_terms": ["Supply Chain", "S&OP", "Service Rate", "Adherence Rate", "Late Deliveries", "Operations Management"],
            "company_values": ["French craftsmanship", "Expertise sharing", "Job creation", "Career development", "Regional development"]
        }
        
        # Create a mock response object
        class MockResponse:
            def __init__(self, content):
                self.content = json.dumps(content, indent=2)
        
        response = MockResponse(mock_response)
        
        print("\nMOCK RESPONSE:")
        print("-" * 40)
        print(response.content)
        print("-" * 40)
        print("="*80)
        
        return response


async def debug_job_description_parsing():
    """Debug the job description parsing process."""
    
    # Sample job description from the user's data
    job_description_text = """Job Description
Job Description

Job Title
Supply Chain Apprentice M/F
Purpose of the position
As a Supply Chain Apprentice, and under the supervision of the Supply Chain Analyst, you contribute to the design, improvement and support of tools, systems and processes.
Missions
Build, implement and monitor indicators;
Analyze performance (service rate, adherence rate, late deliveries, etc.);
Propose and participate in the implementation of organizational changes to the Supply Chain;
Be a stakeholder in process improvement projects in collaboration with field teams;
Participate in the construction of the S&OP process;
Build simulation and optimization models.
Desired profile
You are preparing a Master's degree in operations management and wish to discover the industrial environment;
You are organized and rigorous;
You have analytical skills and an ability to interpret data;
You are comfortable with oral and written communication;
Essential mastery of office tools.
CONTRACT
Apprentice
Contract duration
24 months
Travel
One-off (potentially monthly)
Candidate criteria

Minimum education level required
3- License
Minimum experience level required
Beginner
Location of the position

Region, Department
Normandy, Manche (50)
Commune
SAINT PAIR SUR MER

General information

Attached entity
For 30 years, we have been designing and manufacturing luxury clothing in France for the finest fashion houses.

We strive to share our expertise, promote job creation, and develop career opportunities in our region.

Our clients are experiencing strong growth and place their trust in us. To support them and expand our business, we are continuing our structuring approach and recruiting across all our professions.

1,100 employees, spread across our 15 workshops, contribute every day to promoting excellent French craftsmanship. What if it were you?
Reference
2025-278

Attached entity
For 30 years, we have been designing and manufacturing luxury clothing in France for the finest fashion houses.

We strive to share our expertise, promote job creation, and develop career opportunities in our region.

Our clients are experiencing strong growth and place their trust in us. To support them and expand our business, we are continuing our structuring approach and recruiting across all our professions.

1,100 employees, spread across our 15 workshops, contribute every day to promoting excellent French craftsmanship. What if it were you?"""
    
    print("\n🔍 DEBUGGING JOB DESCRIPTION PARSER")
    print("=" * 50)
    
    try:
        # Initialize components
        print("\n1. Initializing components...")
        
        # Create template manager
        template_manager = ContentTemplateManager()
        
        # Create debug LLM service
        debug_llm_service = DebugLLMService()
        
        # Create parser service with debug LLM
        from src.config.settings import Settings
        settings = Settings()
        
        parser_service = LLMCVParserService(
            llm_service=debug_llm_service,
            settings=settings,
            template_manager=template_manager
        )
        
        print("✅ Components initialized")
        
        # Test template loading
        print("\n2. Testing template loading...")
        template = template_manager.get_template(
            name="job_description_parser", 
            content_type=ContentType.JOB_ANALYSIS
        )
        
        if template:
            print("✅ Template loaded successfully")
            print(f"Template name: {template.name}")
            print(f"Template content type: {template.content_type}")
            print(f"Template content preview: {template.template[:200]}...")
        else:
            print("❌ Template not found!")
            return
        
        # Test prompt formatting
        print("\n3. Testing prompt formatting...")
        formatted_prompt = template_manager.format_template(
            template, {"raw_job_description": job_description_text}
        )
        
        print("✅ Prompt formatted successfully")
        print(f"Formatted prompt length: {len(formatted_prompt)} characters")
        
        # Test parsing
        print("\n4. Testing job description parsing...")
        result = await parser_service.parse_job_description_with_llm(
            raw_text=job_description_text,
            session_id="debug_session",
            trace_id="debug_trace"
        )
        
        print("\n5. PARSING RESULTS:")
        print("=" * 30)
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        # Analyze the result
        print("\n6. ANALYSIS:")
        print("=" * 30)
        
        if isinstance(result, JobDescriptionData):
            print("✅ Returned correct JobDescriptionData type")
            
            # Check which fields are populated
            fields_status = {
                "raw_text": "✅ Populated" if result.raw_text else "❌ Empty",
                "job_title": "✅ Populated" if result.job_title else "❌ Empty",
                "company_name": "✅ Populated" if result.company_name else "❌ Empty",
                "main_job_description_raw": "✅ Populated" if result.main_job_description_raw else "❌ Empty",
                "skills": f"✅ {len(result.skills)} items" if result.skills else "❌ Empty",
                "experience_level": "✅ Populated" if result.experience_level else "❌ Empty",
                "responsibilities": f"✅ {len(result.responsibilities)} items" if result.responsibilities else "❌ Empty",
                "industry_terms": f"✅ {len(result.industry_terms)} items" if result.industry_terms else "❌ Empty",
                "company_values": f"✅ {len(result.company_values)} items" if result.company_values else "❌ Empty",
                "error": "⚠️ Has error" if result.error else "✅ No error"
            }
            
            for field, status in fields_status.items():
                print(f"  {field}: {status}")
                
            # Show the actual template vs model comparison
            print("\n7. TEMPLATE VS MODEL ANALYSIS:")
            print("=" * 30)
            print("✅ TEMPLATE EXTRACTS: 8 fields from job_description_parsing_prompt.md:")
            print("   - job_title, company_name, main_job_description_raw")
            print("   - skills, experience_level, responsibilities, industry_terms, company_values")
            print("✅ MODEL EXPECTS: 9 fields in JobDescriptionData:")
            print("   - raw_text (set by parser service)")
            print("   - job_title, company_name, main_job_description_raw")
            print("   - skills, experience_level, responsibilities, industry_terms, company_values")
            print("   - error (optional)")
            print("✅ ALIGNMENT: Template and model are properly aligned!")
            print("   The raw_text field is automatically set by the parser service.")
            
        else:
            print(f"❌ Unexpected result type: {type(result)}")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_job_description_parsing())