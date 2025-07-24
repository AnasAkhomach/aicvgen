#!/usr/bin/env python3
"""
Debug script to trace KeyQualificationsWriterAgent prompt and response.
This script will help identify the exact prompt passed to the agent following the workflow.
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
from src.models.cv_models import StructuredCV, JobDescriptionData, Section, Item, ItemStatus
from src.models.workflow_models import ContentType
from src.agents.key_qualifications_writer_agent import KeyQualificationsWriterAgent
from src.templates.content_templates import ContentTemplateManager
from uuid import uuid4


class DebugLLMService:
    """Mock LLM service that logs prompts and returns debug responses."""
    
    def __init__(self):
        self.last_prompt = None
        self.last_system_instruction = None
        self.call_count = 0
    
    async def generate_content(self, prompt: str, content_type: ContentType = None, 
                             session_id: str = None, trace_id: str = None, 
                             system_instruction: str = None, max_tokens: int = None,
                             temperature: float = None, **kwargs):
        """Mock generate_content that logs the prompt."""
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_instruction = system_instruction
        
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
        
        # Mock response for key qualifications
        mock_response = {
            "items": [
                {
                    "id": "kq_1",
                    "content": "Supply Chain Management: Comprehensive understanding of end-to-end supply chain processes, including procurement, inventory management, and logistics optimization.",
                    "status": "GENERATED",
                    "item_type": "QUALIFICATION",
                    "metadata": {
                        "item_id": "kq_1",
                        "source": "cv_analysis",
                        "confidence_score": 0.9
                    }
                },
                {
                    "id": "kq_2",
                    "content": "Data Analysis & Performance Monitoring: Proficient in analyzing key performance indicators (KPIs) such as service rates, adherence rates, and delivery metrics to drive operational improvements.",
                    "status": "GENERATED",
                    "item_type": "QUALIFICATION",
                    "metadata": {
                        "item_id": "kq_2",
                        "source": "cv_analysis",
                        "confidence_score": 0.85
                    }
                },
                {
                    "id": "kq_3",
                    "content": "Process Improvement & S&OP Implementation: Experience in Sales & Operations Planning (S&OP) processes and contributing to organizational change initiatives within supply chain operations.",
                    "status": "GENERATED",
                    "item_type": "QUALIFICATION",
                    "metadata": {
                        "item_id": "kq_3",
                        "source": "cv_analysis",
                        "confidence_score": 0.8
                    }
                },
                {
                    "id": "kq_4",
                    "content": "Technical Proficiency: Advanced skills in office tools and simulation modeling, with strong analytical capabilities for data interpretation and decision-making support.",
                    "status": "GENERATED",
                    "item_type": "QUALIFICATION",
                    "metadata": {
                        "item_id": "kq_4",
                        "source": "cv_analysis",
                        "confidence_score": 0.75
                    }
                }
            ]
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


def create_sample_structured_cv() -> StructuredCV:
    """Create a sample structured CV for testing."""
    
    # Sample job description data
    job_data = JobDescriptionData(
        raw_text="Supply Chain Apprentice position...",
        job_title="Supply Chain Apprentice M/F",
        company_name="French Luxury Clothing Manufacturer",
        main_job_description_raw="As a Supply Chain Apprentice, you contribute to the design, improvement and support of tools, systems and processes.",
        skills=["Supply Chain Management", "Data Analysis", "S&OP Process", "Performance Monitoring", "Office Tools"],
        experience_level="Beginner",
        responsibilities=[
            "Build, implement and monitor indicators",
            "Analyze performance (service rate, adherence rate, late deliveries, etc.)",
            "Propose and participate in the implementation of organizational changes to the Supply Chain",
            "Be a stakeholder in process improvement projects in collaboration with field teams",
            "Participate in the construction of the S&OP process",
            "Build simulation and optimization models"
        ],
        industry_terms=["Supply Chain", "S&OP", "Service Rate", "Adherence Rate", "Late Deliveries", "Operations Management"],
        company_values=["French craftsmanship", "Expertise sharing", "Job creation", "Career development", "Regional development"]
    )
    
    # Create structured CV with sample data
    structured_cv = StructuredCV.create_empty(
        cv_text="Sample CV text for testing...",
        job_data=job_data
    )
    
    # Add some sample sections with items
    executive_summary_section = Section(
        id=uuid4(),
        name="Executive Summary",
        content_type="DYNAMIC",
        order=0,
        status=ItemStatus.GENERATED,
        subsections=[],
        items=[
            Item(
                id=uuid4(),
                content="Motivated supply chain professional with strong analytical skills and experience in process improvement.",
                status=ItemStatus.GENERATED,
                item_type="summary_paragraph"
            )
        ]
    )
    
    # Replace the empty executive summary section
    for i, section in enumerate(structured_cv.sections):
        if section.name == "Executive Summary":
            structured_cv.sections[i] = executive_summary_section
            break
    
    return structured_cv


async def debug_key_qualifications_agent():
    """Debug the key qualifications agent process."""
    
    print("\n🔍 DEBUGGING KEY QUALIFICATIONS WRITER AGENT")
    print("=" * 50)
    
    try:
        # Initialize components
        print("\n1. Initializing components...")
        
        # Create template manager
        template_manager = ContentTemplateManager()
        
        # Create debug LLM service
        debug_llm_service = DebugLLMService()
        
        # Create agent with debug LLM service
        from src.config.settings import Settings
        settings = Settings()
        
        # Convert settings to dictionary format expected by agents
        agent_settings_dict = settings.agent_settings.model_dump()
        
        agent = KeyQualificationsWriterAgent(
            llm_service=debug_llm_service,
            settings=agent_settings_dict,
            template_manager=template_manager,
            session_id="debug_session_123"
        )
        
        print("✅ Components initialized")
        
        # Test template loading
        print("\n2. Testing template loading...")
        template = template_manager.get_template(
            name="key_qualifications", 
            content_type=ContentType.QUALIFICATION
        )
        
        if template:
            print("✅ Template loaded successfully")
            print(f"Template name: {template.name}")
            print(f"Template content type: {template.content_type}")
            print(f"Template content preview: {template.template[:200]}...")
        else:
            print("❌ Template not found!")
            return
        
        # Create sample structured CV
        print("\n3. Creating sample structured CV...")
        structured_cv = create_sample_structured_cv()
        print("✅ Sample CV created")
        print(f"CV sections: {[section.name for section in structured_cv.sections]}")
        
        # Test agent execution
        print("\n4. Testing key qualifications agent execution...")
        
        # Get job description data from metadata
        job_desc_dict = structured_cv.metadata.extra.get("job_description", {})
        if job_desc_dict:
            job_description_data = JobDescriptionData(**job_desc_dict)
        else:
            # Fallback to the original job data we created
            job_description_data = job_data
        
        # Run the agent with the required parameters
        result = await agent._execute(
            structured_cv=structured_cv,
            job_description_data=job_description_data
        )
        
        print("\n5. AGENT EXECUTION RESULTS:")
        print("=" * 30)
        print(f"Result type: {type(result)}")
        print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        if result and "structured_cv" in result:
            updated_cv = result["structured_cv"]
            # Find the key qualifications section
            kq_section = None
            for section in updated_cv.sections:
                if section.name == "Key Qualifications":
                    kq_section = section
                    break
            
            if kq_section:
                print("✅ Key Qualifications section found")
                print(f"Section status: {kq_section.status}")
                print(f"Number of items: {len(kq_section.items)}")
                
                for i, item in enumerate(kq_section.items, 1):
                    print(f"\nItem {i}:")
                    print(f"  ID: {item.id}")
                    print(f"  Status: {item.status}")
                    print(f"  Type: {item.item_type}")
                    print(f"  Content: {item.content[:100]}...")
                    if hasattr(item, 'metadata') and item.metadata:
                        print(f"  Metadata: {item.metadata}")
            else:
                print("❌ Key Qualifications section not found")
        elif result and "error_messages" in result:
            print("❌ Agent execution failed:")
            for error in result["error_messages"]:
                print(f"  - {error}")
        else:
            print("❌ Unexpected result format")
        
        # Analyze the workflow
        print("\n6. WORKFLOW ANALYSIS:")
        print("=" * 30)
        print("✅ AGENT WORKFLOW TRACED:")
        print("   1. Agent receives structured CV with job description data")
        print("   2. Template 'key_qualifications' is loaded and formatted")
        print("   3. CV data and job requirements are passed to LLM")
        print("   4. LLM generates key qualifications items")
        print("   5. Response is parsed and integrated into CV structure")
        print("   6. Updated structured CV is returned")
        
        print("\n✅ PROMPT COMPONENTS IDENTIFIED:")
        print("   - System instruction from agent settings")
        print("   - Template with CV data and job description")
        print("   - Session and trace IDs for tracking")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_key_qualifications_agent())