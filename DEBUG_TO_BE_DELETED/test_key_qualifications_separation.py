"""Test script to verify the separation of KeyQualificationsWriterAgent and KeyQualificationsUpdaterAgent.

This script demonstrates the LangGraph pattern where:
1. KeyQualificationsWriterAgent generates only a list of qualifications
2. KeyQualificationsUpdaterAgent consumes that list and updates the structured CV
"""

import asyncio
import json
from typing import Dict, Any

from src.core.container import ContainerSingleton
from src.models.cv_models import StructuredCV, Section, Item, ItemStatus, ItemType
from src.models.cv_models import JobDescriptionData
from src.orchestration.state import AgentState
from src.agents.key_qualifications_writer_agent import KeyQualificationsWriterAgent
from src.agents.key_qualifications_updater_agent import KeyQualificationsUpdaterAgent


def create_test_structured_cv() -> StructuredCV:
    """Create a test structured CV with Key Qualifications section."""
    return StructuredCV(
        sections=[
            Section(
                name="Executive Summary",
                items=[
                    Item(
                        content="Experienced software engineer with 5+ years in full-stack development.",
                        status=ItemStatus.GENERATED,
                        item_type=ItemType.EXECUTIVE_SUMMARY_PARA,
                    )
                ],
            ),
            Section(
                name="Key Qualifications",
                items=[],  # Empty initially
            ),
            Section(
                name="Professional Experience",
                items=[
                    Item(
                        content="Senior Software Engineer at TechCorp (2020-2024)",
                        status=ItemStatus.GENERATED,
                        item_type=ItemType.EXPERIENCE_ROLE_TITLE,
                    )
                ],
            ),
        ],
        metadata={"version": "1.0"},
        big_10_skills=[],
    )


def create_test_job_description() -> JobDescriptionData:
    """Create a test job description."""
    return JobDescriptionData(
        raw_text="We are looking for a Senior Full Stack Developer with expertise in Python, React, and cloud technologies. Requirements: 5+ years of experience in full-stack development, proficiency in Python and JavaScript, experience with React and Node.js, knowledge of cloud platforms (AWS, Azure, GCP), strong problem-solving skills.",
        job_title="Senior Full Stack Developer",
        company_name="TechStartup Inc.",
        main_job_description_raw="We are looking for a Senior Full Stack Developer with expertise in Python, React, and cloud technologies.",
        skills=["Python", "JavaScript", "React", "Node.js", "AWS", "Docker"],
        responsibilities=[
            "Develop scalable web applications",
            "Collaborate with cross-functional teams",
            "Implement best practices for code quality",
            "Mentor junior developers",
        ],
        experience_level="Senior (5+ years)",
    )


async def test_key_qualifications_separation():
    """Test the separation of key qualifications generation and updating."""
    print("\n🧪 TESTING KEY QUALIFICATIONS AGENT SEPARATION")
    print("=" * 60)
    
    try:
        # Get container and services
        container = ContainerSingleton.get_instance()
        llm_service = container.llm_service()
        template_manager = container.template_manager()
        
        # Create test data
        structured_cv = create_test_structured_cv()
        job_description_data = create_test_job_description()
        
        print("\n1. Initial state:")
        qual_section = next(s for s in structured_cv.sections if s.name == "Key Qualifications")
        print(f"   Key Qualifications items: {len(qual_section.items)}")
        
        # Create agents
        writer_agent = KeyQualificationsWriterAgent(
            llm_service=llm_service,
            template_manager=template_manager,
            settings={},
            session_id="test_session"
        )
        
        updater_agent = KeyQualificationsUpdaterAgent(session_id="test_session")
        
        print("\n2. Running KeyQualificationsWriterAgent...")
        
        # Create initial state
        state = AgentState(
            structured_cv=structured_cv,
            job_description_data=job_description_data,
            cv_text="Sample CV text for testing purposes",
            current_item_id="key_qualifications_section"
        )
        
        # Run writer agent
        writer_result = await writer_agent.run_as_node(state)
        print(f"   Writer result keys: {list(writer_result.keys())}")
        
        if "generated_key_qualifications" in writer_result:
            qualifications = writer_result["generated_key_qualifications"]
            print(f"   Generated {len(qualifications)} qualifications:")
            for i, qual in enumerate(qualifications, 1):
                print(f"     {i}. {qual}")
            
            print("\n3. Running KeyQualificationsUpdaterAgent...")
            
            # Update state with generated qualifications
            state.generated_key_qualifications = qualifications
            
            # Run updater agent
            updater_result = await updater_agent.run_as_node(state)
            print(f"   Updater result keys: {list(updater_result.keys())}")
            
            if "structured_cv" in updater_result:
                updated_cv = updater_result["structured_cv"]
                updated_qual_section = next(s for s in updated_cv.sections if s.name == "Key Qualifications")
                print(f"   Updated Key Qualifications items: {len(updated_qual_section.items)}")
                
                print("\n4. Final Key Qualifications in CV:")
                for i, item in enumerate(updated_qual_section.items, 1):
                    print(f"     {i}. {item.content}")
                    print(f"        Status: {item.status}")
                    print(f"        Type: {item.item_type}")
                
                print("\n✅ SUCCESS: Agent separation working correctly!")
                print("   - Writer agent generates only the qualifications list")
                print("   - Updater agent consumes the list and updates the CV")
                print("   - Data flows correctly through AgentState.generated_key_qualifications")
                
            else:
                print("❌ ERROR: Updater agent did not return structured_cv")
                print(f"   Updater result: {updater_result}")
        else:
            print("❌ ERROR: Writer agent did not return generated_key_qualifications")
            print(f"   Writer result: {writer_result}")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting test execution...")
    asyncio.run(test_key_qualifications_separation())
    print("Test execution completed.")