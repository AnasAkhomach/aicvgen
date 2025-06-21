"""Unit tests for refactored ParserAgent helper methods.

This module tests the private helper methods created during the Task 8 refactoring
of the parse_cv_with_llm method in ParserAgent.
"""

import pytest
from unittest.mock import Mock, patch
from src.agents.parser_agent import ParserAgent
from src.models.data_models import (
    StructuredCV,
    JobDescriptionData,
    CVParsingResult,
    CVParsingPersonalInfo,
    CVParsingSection,
    CVParsingSubsection,
    Section,
    Subsection,
    Item,
    ItemType,
    ItemStatus,
)
from src.services.llm_service import EnhancedLLMService


class TestParserAgentRefactoredMethods:
    """Test cases for the refactored ParserAgent helper methods."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        mock_service = Mock(spec=EnhancedLLMService)
        return mock_service

    @pytest.fixture
    def parser_agent(self, mock_llm_service):
        """Create a ParserAgent instance with mocked dependencies."""
        agent = ParserAgent(
            name="test_parser_agent",
            description="Test parser agent for refactored methods",
            llm_service=mock_llm_service,
        )
        return agent

    @pytest.fixture
    def sample_job_data(self):
        """Sample job description data for testing."""
        return JobDescriptionData(
            raw_text="Software Engineer position",
            skills=["Python", "JavaScript"],
            experience_level="Mid-level",
            responsibilities=["Develop software", "Code review"],
            industry_terms=["Agile", "CI/CD"],
            company_values=["Innovation", "Quality"],
            status=ItemStatus.GENERATED,
        )

    @pytest.fixture
    def sample_parsing_result(self):
        """Sample CVParsingResult for testing."""
        personal_info = CVParsingPersonalInfo(
            name="John Doe",
            email="john.doe@email.com",
            phone="+1234567890",
            linkedin="https://linkedin.com/in/johndoe",
            github="https://github.com/johndoe",
            location="San Francisco, CA",
        )

        sections = [
            CVParsingSection(
                name="Executive Summary",
                items=["Experienced software engineer with 5+ years."],
                subsections=[],
            ),
            CVParsingSection(
                name="Professional Experience",
                items=[],
                subsections=[
                    CVParsingSubsection(
                        name="Senior Software Engineer at TechCorp",
                        items=[
                            "Developed scalable applications",
                            "Led team of 3 developers",
                        ],
                    )
                ],
            ),
            CVParsingSection(
                name="Education",
                items=["Bachelor of Science in Computer Science"],
                subsections=[],
            ),
        ]

        return CVParsingResult(personal_info=personal_info, sections=sections)

    def test_initialize_structured_cv(self, parser_agent, sample_job_data):
        """Test _initialize_structured_cv helper method."""
        cv_text = "Sample CV text"

        result = parser_agent._initialize_structured_cv(cv_text, sample_job_data)

        assert isinstance(result, StructuredCV)
        assert result.metadata["original_cv_text"] == cv_text
        assert "job_description" in result.metadata
        assert result.metadata["job_description"] == sample_job_data.model_dump()
        assert len(result.sections) == 0

    def test_initialize_structured_cv_with_dict_job_data(self, parser_agent):
        """Test _initialize_structured_cv with dictionary job data."""
        cv_text = "Sample CV text"
        job_data_dict = {"title": "Software Engineer", "skills": ["Python"]}

        result = parser_agent._initialize_structured_cv(cv_text, job_data_dict)

        assert isinstance(result, StructuredCV)
        assert result.metadata["job_description"] == job_data_dict

    def test_initialize_structured_cv_with_none_job_data(self, parser_agent):
        """Test _initialize_structured_cv with None job data."""
        cv_text = "Sample CV text"

        result = parser_agent._initialize_structured_cv(cv_text, None)

        assert isinstance(result, StructuredCV)
        assert result.metadata["job_description"] == {}

    def test_create_structured_cv_with_metadata(self, parser_agent, sample_job_data):
        """Test _create_structured_cv_with_metadata helper method."""
        cv_text = "Sample CV text"

        result = parser_agent._create_structured_cv_with_metadata(
            cv_text, sample_job_data
        )

        assert isinstance(result, StructuredCV)
        assert result.metadata["original_cv_text"] == cv_text
        assert "job_description" in result.metadata
        assert len(result.sections) == 0

    def test_add_contact_info_to_metadata(self, parser_agent, sample_parsing_result):
        """Test _add_contact_info_to_metadata helper method."""
        structured_cv = StructuredCV()

        parser_agent._add_contact_info_to_metadata(structured_cv, sample_parsing_result)

        assert structured_cv.metadata["name"] == "John Doe"
        assert structured_cv.metadata["email"] == "john.doe@email.com"
        assert structured_cv.metadata["phone"] == "+1234567890"
        assert structured_cv.metadata["linkedin"] == "https://linkedin.com/in/johndoe"
        assert structured_cv.metadata["github"] == "https://github.com/johndoe"
        assert structured_cv.metadata["location"] == "San Francisco, CA"

    def test_add_contact_info_to_metadata_with_error(self, parser_agent):
        """Test _add_contact_info_to_metadata with invalid parsing result."""
        structured_cv = StructuredCV()
        invalid_parsing_result = Mock()
        invalid_parsing_result.personal_info = None

        # This should trigger an AttributeError when accessing personal_info attributes
        parser_agent._add_contact_info_to_metadata(
            structured_cv, invalid_parsing_result
        )

        # Should have added parsing_error to metadata
        assert "parsing_error" in structured_cv.metadata

    def test_determine_section_content_type_dynamic(self, parser_agent):
        """Test _determine_section_content_type for dynamic sections."""
        dynamic_sections = [
            "Executive Summary",
            "Key Qualifications",
            "Professional Experience",
            "Project Experience",
        ]

        for section_name in dynamic_sections:
            result = parser_agent._determine_section_content_type(section_name)
            assert result == "DYNAMIC"

            # Test case insensitive
            result = parser_agent._determine_section_content_type(section_name.lower())
            assert result == "DYNAMIC"

    def test_determine_section_content_type_static(self, parser_agent):
        """Test _determine_section_content_type for static sections."""
        static_sections = ["Education", "Certifications", "Languages", "Publications"]

        for section_name in static_sections:
            result = parser_agent._determine_section_content_type(section_name)
            assert result == "STATIC"

    def test_add_section_items_dynamic(self, parser_agent):
        """Test _add_section_items for dynamic content."""
        section = Section(name="Executive Summary", content_type="DYNAMIC", order=0)
        parsed_section = Mock()
        parsed_section.name = "Executive Summary"
        parsed_section.items = ["Item 1", "Item 2", "  Item 3  ", ""]

        parser_agent._add_section_items(section, parsed_section, "DYNAMIC")

        assert len(section.items) == 3  # Empty string should be filtered out
        assert section.items[0].content == "Item 1"
        assert section.items[0].status == ItemStatus.INITIAL
        assert section.items[0].item_type == ItemType.EXECUTIVE_SUMMARY_PARA
        assert section.items[2].content == "Item 3"  # Whitespace should be stripped

    def test_add_section_items_static(self, parser_agent):
        """Test _add_section_items for static content."""
        section = Section(name="Education", content_type="STATIC", order=0)
        parsed_section = Mock()
        parsed_section.name = "Education"
        parsed_section.items = ["Bachelor's Degree"]

        parser_agent._add_section_items(section, parsed_section, "STATIC")

        assert len(section.items) == 1
        assert section.items[0].status == ItemStatus.STATIC
        assert section.items[0].item_type == ItemType.EDUCATION_ENTRY

    def test_add_section_subsections(self, parser_agent):
        """Test _add_section_subsections helper method."""
        section = Section(
            name="Professional Experience", content_type="DYNAMIC", order=0
        )
        parsed_section = Mock()
        parsed_section.name = "Professional Experience"

        subsection_mock = Mock()
        subsection_mock.name = "Senior Engineer"
        subsection_mock.items = ["Developed applications", "Led team"]
        parsed_section.subsections = [subsection_mock]

        parser_agent._add_section_subsections(section, parsed_section, "DYNAMIC")

        assert len(section.subsections) == 1
        assert section.subsections[0].name == "Senior Engineer"
        assert len(section.subsections[0].items) == 2
        assert section.subsections[0].items[0].status == ItemStatus.INITIAL

    def test_convert_sections_to_structured_cv(
        self, parser_agent, sample_parsing_result
    ):
        """Test _convert_sections_to_structured_cv helper method."""
        structured_cv = StructuredCV()

        parser_agent._convert_sections_to_structured_cv(
            structured_cv, sample_parsing_result
        )

        assert len(structured_cv.sections) == 3

        # Check Executive Summary section (dynamic)
        exec_summary = structured_cv.sections[0]
        assert exec_summary.name == "Executive Summary"
        assert exec_summary.content_type == "DYNAMIC"
        assert exec_summary.order == 0
        assert len(exec_summary.items) == 1
        assert exec_summary.items[0].status == ItemStatus.INITIAL

        # Check Professional Experience section (dynamic with subsections)
        prof_exp = structured_cv.sections[1]
        assert prof_exp.name == "Professional Experience"
        assert prof_exp.content_type == "DYNAMIC"
        assert len(prof_exp.subsections) == 1
        assert prof_exp.subsections[0].name == "Senior Software Engineer at TechCorp"
        assert len(prof_exp.subsections[0].items) == 2

        # Check Education section (static)
        education = structured_cv.sections[2]
        assert education.name == "Education"
        assert education.content_type == "STATIC"
        assert len(education.items) == 1
        assert education.items[0].status == ItemStatus.STATIC

    @pytest.mark.asyncio
    async def test_parse_cv_content_with_llm(self, parser_agent):
        """Test _parse_cv_content_with_llm helper method."""
        cv_text = "Sample CV text"
        mock_parsing_data = {
            "personal_info": {
                "name": "John Doe",
                "email": "john@example.com",
                "phone": "+1234567890",
                "linkedin": "",
                "github": "",
                "location": "San Francisco",
            },
            "sections": [],
        }

        with patch.object(
            parser_agent.settings,
            "get_prompt_path_by_key",
            return_value="test_prompt.txt",
        ), patch("builtins.open", create=True) as mock_open, patch.object(
            parser_agent, "_generate_and_parse_json", return_value=mock_parsing_data
        ) as mock_generate:

            mock_open.return_value.__enter__.return_value.read.return_value = (
                "Template: {{raw_cv_text}}"
            )

            result = await parser_agent._parse_cv_content_with_llm(cv_text)

            assert isinstance(result, CVParsingResult)
            assert result.personal_info.name == "John Doe"
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_cv_content_with_llm_validation_error(self, parser_agent):
        """Test _parse_cv_content_with_llm with validation error."""
        cv_text = "Sample CV text"
        invalid_parsing_data = {"invalid": "data"}

        with patch.object(
            parser_agent.settings,
            "get_prompt_path_by_key",
            return_value="test_prompt.txt",
        ), patch("builtins.open", create=True) as mock_open, patch.object(
            parser_agent, "_generate_and_parse_json", return_value=invalid_parsing_data
        ):

            mock_open.return_value.__enter__.return_value.read.return_value = (
                "Template: {{raw_cv_text}}"
            )

            # Should raise ValueError due to validation error
            with pytest.raises(ValueError):
                await parser_agent._parse_cv_content_with_llm(cv_text)

    @pytest.mark.parametrize(
        "cv_text,expected_subsection_count,expected_item_type",
        [
            (
                "Software Engineer at TechCorp (2018-2022)\n• Developed scalable APIs\n• Led migration to cloud\nSenior Developer at DataSoft (2015-2018)\n- Built ETL pipelines\n- Mentored junior staff",
                2,
                "bullet_point",
            ),
            (
                "Project Manager at BuildIt (2020-2023)\n1. Managed $1M budget\n2. Delivered on time",
                1,
                "bullet_point",
            ),
        ],
    )
    def test_parse_cv_text_to_content_item_structures_experience(
        self, parser_agent, cv_text, expected_subsection_count, expected_item_type
    ):
        """Test that ParserAgent parses raw CV text into correct subsections and items."""
        result = parser_agent.parse_cv_text_to_content_item(
            cv_text, generation_context={}
        )
        from src.models.data_models import Subsection, ItemType

        assert isinstance(result, Subsection)
        assert result.name == "Professional Experience"
        assert hasattr(result, "items")
        for item in result.items:
            assert item.item_type == ItemType.BULLET_POINT
