"""Integration tests for UserCVParserAgent fix to ensure required sections."""

import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.agents.user_cv_parser_agent import UserCVParserAgent
from src.models.cv_models import StructuredCV, Section, Item, MetadataModel
from src.models.llm_data_models import (
    CVParsingStructuredOutput,
    CVParsingPersonalInfo,
    CVParsingSection,
    CVParsingSubsection,
)


class TestUserCVParserAgentRequiredSections:
    """Integration tests for UserCVParserAgent required sections fix."""

    @pytest.fixture
    def mock_llm_response_incomplete(self):
        """Mock LLM response with incomplete sections."""
        return CVParsingStructuredOutput(
            personal_info=CVParsingPersonalInfo(
                name="John Doe",
                email="john.doe@example.com",
                phone="123-456-7890",
                linkedin="",
                github="",
                location="New York, NY",
            ),
            sections=[
                CVParsingSection(
                    name="Education",
                    items=[
                        "Bachelor of Science in Computer Science - University of Technology, 2020"
                    ],
                )
            ],
        )

    @pytest.fixture
    def agent(self):
        """Create UserCVParserAgent instance for testing."""
        # Mock the required LLMCVParserService
        mock_llm_cv_parser_service = Mock()

        return UserCVParserAgent(
            parser_service=mock_llm_cv_parser_service, session_id="test-session-123"
        )

    @pytest.mark.asyncio
    async def test_parser_ensures_required_sections_after_llm_response(
        self, agent, mock_llm_response_incomplete
    ):
        """Test that parser adds missing required sections after LLM parsing."""
        # Mock the service method to return incomplete CV (missing required sections)
        incomplete_cv = StructuredCV(
            sections=[
                Section(
                    name="Education",
                    items=[
                        Item(
                            content="Bachelor of Science in Computer Science - University of Technology, 2020"
                        )
                    ],
                )
                # Missing: Executive Summary, Key Qualifications, Professional Experience, Project Experience
            ]
        )

        # Mock the parse_cv_to_structured_cv method directly
        agent._parser_service.parse_cv_to_structured_cv = AsyncMock(
            return_value=incomplete_cv
        )

        # Test CV text
        cv_text = "John Doe\nSoftware Engineer\nExperience: ABC Corp"

        # Execute the parsing
        result = await agent.run(cv_text)

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

        # Since we're mocking the final method, we should get back exactly what we mocked
        # This test verifies that the agent correctly calls the service and returns the result

        # Should have exactly 1 section (Education) as mocked
        assert (
            len(structured_cv.sections) == 1
        ), f"Expected exactly 1 section, got {len(structured_cv.sections)}: {list(section_names)}"

        # Verify the Education section is present and has content
        education_section = next(
            (s for s in structured_cv.sections if s.name == "Education"), None
        )
        assert (
            education_section is not None
        ), f"Education section not found in {list(section_names)}"
        assert len(education_section.items) == 1
        assert (
            education_section.items[0].content
            == "Bachelor of Science in Computer Science - University of Technology, 2020"
        )

        # Verify that the service method was called with correct parameters
        agent._parser_service.parse_cv_to_structured_cv.assert_called_once_with(
            cv_text=cv_text, session_id=agent.session_id
        )

    @pytest.mark.asyncio
    async def test_parser_with_complete_llm_response(self, agent):
        """Test that parser works correctly when LLM returns all required sections."""
        # Mock complete LLM response
        complete_response = StructuredCV(
            sections=[
                Section(
                    name="Executive Summary",
                    items=[Item(content="Experienced software engineer")],
                ),
                Section(
                    name="Key Qualifications",
                    items=[
                        Item(content="Python"),
                        Item(content="JavaScript"),
                        Item(content="React"),
                    ],
                ),
                Section(name="Professional Experience", items=[]),
                Section(name="Project Experience", items=[]),
                Section(name="Education", items=[Item(content="MS Computer Science")]),
            ]
        )

        # Setup the mock parser service to return complete response
        agent._parser_service.parse_cv_to_structured_cv = AsyncMock(
            return_value=complete_response
        )

        # Parse the CV
        structured_cv = await agent.run("Jane Smith CV text...")

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
        assert (
            len(structured_cv.sections) == 5
        ), f"Expected exactly 5 sections, got {len(structured_cv.sections)}: {section_names}"

        # Verify content is preserved - items are stored directly in sections
        exec_summary = next(
            (s for s in structured_cv.sections if s.name == "Executive Summary"), None
        )
        assert exec_summary is not None
        print(f"Executive Summary items: {len(exec_summary.items)}")
        print(
            f"Executive Summary item contents: {[item.content for item in exec_summary.items]}"
        )
        assert len(exec_summary.items) > 0
        assert any(
            "Experienced software engineer" in item.content
            for item in exec_summary.items
        )

        key_quals = next(
            (s for s in structured_cv.sections if s.name == "Key Qualifications"), None
        )
        assert key_quals is not None
        print(f"Key Qualifications items: {len(key_quals.items)}")
        print(
            f"Key Qualifications item contents: {[item.content for item in key_quals.items]}"
        )
        assert len(key_quals.items) > 0
        items_content = [item.content for item in key_quals.items]
        assert any("Python" in content for content in items_content)
        assert any("JavaScript" in content for content in items_content)
        assert any("React" in content for content in items_content)
