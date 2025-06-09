#!/usr/bin/env python3
"""
Workflow Data Adapter

This script fixes the data structure mismatch between the workflow input
and what the enhanced_content_writer agent expects.
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, List, Any

# Add project root and src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

# Set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = str(src_path)

from src.models.data_models import ContentType
from src.agents.agent_base import AgentExecutionContext
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.config.logging_config import setup_logging

# Setup logging
setup_logging()

def adapt_workflow_data_for_content_writer(
    workflow_input: Dict[str, Any],
    content_type: ContentType = ContentType.EXPERIENCE
) -> Dict[str, Any]:
    """
    Adapt workflow input data to the format expected by enhanced_content_writer.
    
    Args:
        workflow_input: Raw workflow input with personal_info, experience, job_description
        content_type: Type of content to generate
        
    Returns:
        Adapted data structure for enhanced_content_writer
    """
    
    # Extract components from workflow input
    personal_info = workflow_input.get("personal_info", {})
    experience = workflow_input.get("experience", [])
    job_description = workflow_input.get("job_description", {})
    
    # Create job_description_data structure
    job_description_data = {
        "title": job_description.get("title", "Unknown Position"),
        "company": job_description.get("company", "Unknown Company"),
        "description": job_description.get("description", ""),
        "requirements": job_description.get("requirements", []),
        "skills": job_description.get("skills", []),
        "responsibilities": job_description.get("responsibilities", []),
        "industry": job_description.get("industry", ""),
        "location": job_description.get("location", ""),
        "employment_type": job_description.get("employment_type", "Full-time")
    }
    
    # Create content_item structure based on content type
    if content_type == ContentType.EXPERIENCE:
        content_item = {
            "type": "experience",
            "title": "Professional Experience",
            "data": {
                "roles": experience,
                "total_years": len(experience),
                "industries": list(set(role.get("industry", "") for role in experience if role.get("industry")))
            }
        }
    elif content_type == ContentType.SKILLS:
        # Extract skills from experience
        all_skills = []
        for role in experience:
            if "skills" in role:
                all_skills.extend(role["skills"])
        
        content_item = {
            "type": "skills",
            "title": "Technical Skills",
            "data": {
                "skills": list(set(all_skills)),
                "categories": ["Technical", "Professional", "Industry-Specific"]
            }
        }
    else:
        # Default content item
        content_item = {
            "type": content_type.value,
            "title": f"{content_type.value.title()} Section",
            "data": {"raw_data": workflow_input}
        }
    
    # Create generation context
    generation_context = {
        "personal_info": personal_info,
        "target_role": job_description_data["title"],
        "target_company": job_description_data["company"],
        "original_cv_text": "\n".join([
            f"{personal_info.get('name', '')}",
            f"{personal_info.get('email', '')}",
            f"{personal_info.get('phone', '')}",
            f"{personal_info.get('location', '')}"
        ]),
        "generation_mode": "job_tailored",
        "optimization_level": "high"
    }
    
    return {
        "job_description_data": job_description_data,
        "content_item": content_item,
        "context": generation_context
    }

async def test_adapted_workflow_data():
    """Test the data adapter with the enhanced content writer."""
    print("=== Testing Workflow Data Adapter ===")
    
    # Sample workflow input (as it comes from the orchestrator)
    workflow_input = {
        "personal_info": {
            "name": "Nouha KARIM",
            "email": "nouhakarim305@gmail.com",
            "phone": "+33 765732271",
            "location": "Reims",
            "title": "ETUDIANTE EN M2 OPTION GÉNIE DES SYSTÈMES INDUSTRIELS"
        },
        "experience": [
            {
                "title": "Student in Industrial Systems Engineering",
                "company": "University",
                "duration": "2023-2024",
                "description": "Studying advanced industrial systems",
                "skills": ["Systems Analysis", "Process Optimization", "Project Management"]
            },
            {
                "title": "Intern",
                "company": "Tech Company",
                "duration": "Summer 2023",
                "description": "Worked on automation projects",
                "skills": ["Automation", "Python", "Data Analysis"]
            }
        ],
        "job_description": {
            "title": "Industrial Engineer",
            "company": "Manufacturing Corp",
            "description": "Looking for an industrial engineer to optimize manufacturing processes",
            "requirements": ["Bachelor's degree in Industrial Engineering", "Experience with process optimization"],
            "skills": ["Process Optimization", "Lean Manufacturing", "Six Sigma", "Python"],
            "responsibilities": ["Analyze manufacturing processes", "Implement improvements", "Lead projects"]
        }
    }
    
    print("\n1. Adapting workflow data for content writer...")
    
    # Test with EXPERIENCE content type
    adapted_data = adapt_workflow_data_for_content_writer(
        workflow_input, 
        ContentType.EXPERIENCE
    )
    
    print(f"✓ Adapted data structure created")
    print(f"  - job_description_data keys: {list(adapted_data['job_description_data'].keys())}")
    print(f"  - content_item type: {adapted_data['content_item']['type']}")
    print(f"  - context keys: {list(adapted_data['context'].keys())}")
    
    print("\n2. Testing with Enhanced Content Writer...")
    
    # Create agent and context
    agent = EnhancedContentWriterAgent()
    context = AgentExecutionContext(
        session_id="test_adapter",
        input_data=adapted_data,
        content_type=ContentType.EXPERIENCE
    )
    
    # Run the agent
    result = await agent.run_async(adapted_data, context)
    
    print(f"\n3. Results:")
    print(f"  - Success: {result.success}")
    print(f"  - Confidence: {result.confidence_score}")
    print(f"  - Processing time: {result.processing_time:.2f}s")
    
    if result.success and result.output_data:
        content = result.output_data.get("content", "")
        print(f"  - Content length: {len(content)} characters")
        if content:
            print(f"  - Content preview: {content[:200]}...")
        else:
            print("  - ⚠️  No content generated!")
    else:
        print(f"  - ❌ Error: {result.error_message}")
    
    print("\n=== Test Complete ===")
    return result.success

if __name__ == "__main__":
    success = asyncio.run(test_adapted_workflow_data())
    if success:
        print("\n✅ Data adapter working correctly!")
    else:
        print("\n❌ Data adapter needs fixes!")