"""Integration tests for UserCVParserAgent fix to ensure required sections."""

import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.agents.user_cv_parser_agent import UserCVParserAgent
from src.models.cv_models import StructuredCV
from src.models.llm_data_models import CVParsingResult, CVParsingPersonalInfo, CVParsingSection, CVParsingSubsection


class TestUserCVParserAgentRequiredSections:
    """Integration tests for UserCVParserAgent required sections fix."""

    @pytest.fixture
    def mock_llm_response_incomplete(self):
        """Mock LLM response that's missing some required sections."""
        return CVParsingResult(
            personal_info=CVParsingPersonalInfo(
                name="John Doe",
                email="john.doe@example.com",
                phone="+1-555-0123",
                linkedin=None,
                github=None,
                location="New York, NY"
            ),
            sections=[
                CVParsingSection(
                     name="Professional Experience",
                     items=[],
                     subsections=[
                         CVParsingSubsection(
                             name="Senior Developer @ TechCorp | 2020-Present",
                             items=["Developed web applications", "Led team of 5 developers"]
                         )
                     ]
                 ),
                CVParsingSection(
                    name="Education",
                    items=["BS Computer Science, University of Technology, 2018"],
                    subsections=[]
                )
                # Note: Missing "Executive Summary", "Key Qualifications", "Project Experience"
            ]
        )

    @pytest.fixture
    def agent(self):
        """Create UserCVParserAgent instance for testing."""
        # Mock all required dependencies
        mock_llm_service = Mock()
        mock_vector_store_service = Mock()
        mock_template_manager = Mock()
        mock_settings = {}
        mock_session_id = "test-session-123"
        
        return UserCVParserAgent(
            llm_service=mock_llm_service,
            vector_store_service=mock_vector_store_service,
            template_manager=mock_template_manager,
            settings=mock_settings,
            session_id=mock_session_id
        )

    @pytest.mark.asyncio
    @patch('src.agents.user_cv_parser_agent.UserCVParserAgent._store_cv_vectors')
    @patch('src.services.llm_cv_parser_service.LLMCVParserService.parse_cv_with_llm')
    async def test_parser_ensures_required_sections_after_llm_response(
        self, mock_parse_cv, mock_store_vectors, agent, mock_llm_response_incomplete
    ):
        """Test that parser adds missing required sections after LLM parsing."""
        # Setup mocks
        mock_parse_cv.return_value = mock_llm_response_incomplete
        mock_store_vectors.return_value = None
        
        # Test CV text
        cv_text = "John Doe\nSoftware Engineer\nExperience: ABC Corp"
        
        # Execute the parsing
        result = await agent.parse_cv(cv_text)
        
        # Verify result is StructuredCV
        assert isinstance(result, StructuredCV)
        structured_cv = result
        
        # Debug: Print actual section names and detailed info
        print(f"\nResult type: {type(result)}")
        print(f"Result sections type: {type(result.sections)}")
        print(f"Result sections length: {len(result.sections)}")
        section_names = {section.name for section in structured_cv.sections}
        print(f"\nActual sections: {list(section_names)}")
        print(f"Total sections: {len(structured_cv.sections)}")
        
        # Check that all required sections are present
        required_sections = [
            "Executive Summary",
            "Key Qualifications",
            "Professional Experience",
            "Project Experience",
            "Education"
        ]
        
        for required_section in required_sections:
            assert required_section in section_names, f"Missing required section: {required_section}. Available sections: {list(section_names)}"
        
        # Should have at least the 5 required sections
        assert len(structured_cv.sections) >= 5, f"Expected at least 5 sections, got {len(structured_cv.sections)}: {list(section_names)}"
        
        # Verify that the sections from LLM response are preserved
        prof_exp_section = next((s for s in structured_cv.sections if s.name == "Professional Experience"), None)
        assert prof_exp_section is not None
        # Should have subsections from the LLM response
        assert len(prof_exp_section.subsections) > 0
        
        education_section = next((s for s in structured_cv.sections if s.name == "Education"), None)
        assert education_section is not None
        # Should have subsections with items from the LLM response (items are stored in "Main" subsection)
        assert len(education_section.subsections) > 0
        assert len(education_section.subsections[0].items) > 0
        
        # Verify that missing sections were added as empty
        key_qual_section = next((s for s in structured_cv.sections if s.name == "Key Qualifications"), None)
        assert key_qual_section is not None, f"Key Qualifications section not found in {list(section_names)}"
        assert len(key_qual_section.subsections) == 0
        
        exec_summary_section = next((s for s in structured_cv.sections if s.name == "Executive Summary"), None)
        assert exec_summary_section is not None, f"Executive Summary section not found in {list(section_names)}"
        assert len(exec_summary_section.subsections) == 0
        
        project_exp_section = next((s for s in structured_cv.sections if s.name == "Project Experience"), None)
        assert project_exp_section is not None, f"Project Experience section not found in {list(section_names)}"
        assert len(project_exp_section.subsections) == 0

    @pytest.mark.asyncio
    @patch('src.agents.user_cv_parser_agent.UserCVParserAgent._store_cv_vectors')
    @patch('src.services.llm_cv_parser_service.LLMCVParserService.parse_cv_with_llm')
    async def test_parser_with_complete_llm_response(self, mock_parse_cv, mock_store_vectors, agent):
        """Test that parser works correctly when LLM returns all required sections."""
        # Mock complete LLM response
        complete_response = CVParsingResult(
            personal_info=CVParsingPersonalInfo(
                name="Jane Smith",
                email="jane.smith@example.com",
                phone="+1-555-0456",
                linkedin=None,
                github=None,
                location="San Francisco, CA"
            ),
            sections=[
                CVParsingSection(name="Executive Summary", items=["Experienced software engineer"], subsections=[]),
                CVParsingSection(name="Key Qualifications", items=["Python", "JavaScript", "React"], subsections=[]),
                CVParsingSection(name="Professional Experience", items=[], subsections=[]),
                CVParsingSection(name="Project Experience", items=[], subsections=[]),
                CVParsingSection(name="Education", items=["MS Computer Science"], subsections=[])
            ]
        )
        
        mock_parse_cv.return_value = complete_response
        mock_store_vectors.return_value = None
        
        # Parse the CV
        structured_cv = await agent.parse_cv("Jane Smith CV text...")
        
        # Verify the result
        assert isinstance(structured_cv, StructuredCV)
        
        # Debug: Print actual section names and detailed info
        print(f"\nResult type: {type(structured_cv)}")
        print(f"Result sections type: {type(structured_cv.sections)}")
        print(f"Result sections length: {len(structured_cv.sections)}")
        section_names = [section.name for section in structured_cv.sections]
        print(f"\nActual sections: {section_names}")
        print(f"Total sections: {len(structured_cv.sections)}")
        
        # Should have exactly 5 sections (no duplicates added)
        assert len(structured_cv.sections) == 5, f"Expected exactly 5 sections, got {len(structured_cv.sections)}: {section_names}"
        
        # Verify content is preserved - items are stored in subsections
        exec_summary = next((s for s in structured_cv.sections if s.name == "Executive Summary"), None)
        assert exec_summary is not None
        print(f"Executive Summary subsections: {len(exec_summary.subsections)}")
        if exec_summary.subsections:
            print(f"Executive Summary items: {[item.content for item in exec_summary.subsections[0].items]}")
            assert any("Experienced software engineer" in item.content for item in exec_summary.subsections[0].items)
        
        key_quals = next((s for s in structured_cv.sections if s.name == "Key Qualifications"), None)
        assert key_quals is not None
        print(f"Key Qualifications subsections: {len(key_quals.subsections)}")
        if key_quals.subsections:
            print(f"Key Qualifications items: {[item.content for item in key_quals.subsections[0].items]}")
            items_content = [item.content for item in key_quals.subsections[0].items]
            assert any("Python" in content for content in items_content)
            assert any("JavaScript" in content for content in items_content)
            assert any("React" in content for content in items_content)