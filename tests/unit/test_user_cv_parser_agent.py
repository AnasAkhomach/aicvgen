"""Unit tests for UserCVParserAgent."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from src.agents.user_cv_parser_agent import UserCVParserAgent
from src.models.data_models import StructuredCV, Section, Subsection, Item, ItemStatus
from src.models.llm_data_models import CVParsingResult, CVParsingSection, CVParsingSubsection, CVParsingPersonalInfo
from src.models.agent_models import AgentResult
from src.models.agent_output_models import ParserAgentOutput
from src.error_handling.exceptions import AgentExecutionError
from src.constants.agent_constants import AgentConstants


class TestUserCVParserAgent:
    """Test suite for UserCVParserAgent."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        service = AsyncMock()
        return service

    @pytest.fixture
    def mock_vector_store_service(self):
        """Create a mock vector store service."""
        service = AsyncMock()
        service.add_structured_cv = AsyncMock()
        return service

    @pytest.fixture
    def mock_template_manager(self):
        """Create a mock template manager."""
        manager = Mock()
        return manager

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        return {"test_setting": "test_value"}

    @pytest.fixture
    def user_cv_parser_agent(self, mock_llm_service, mock_vector_store_service, mock_template_manager, mock_settings):
        """Create a UserCVParserAgent instance for testing."""
        return UserCVParserAgent(
            llm_service=mock_llm_service,
            vector_store_service=mock_vector_store_service,
            template_manager=mock_template_manager,
            settings=mock_settings,
            session_id="test_session"
        )

    @pytest.fixture
    def sample_cv_text(self):
        """Sample CV text for testing."""
        return """
        John Doe
        Software Engineer
        
        Experience:
        - Senior Developer at Tech Corp (2020-2023)
        - Junior Developer at StartupCo (2018-2020)
        
        Education:
        - BS Computer Science, University of Tech (2018)
        
        Skills:
        - Python, JavaScript, React
        """

    @pytest.fixture
    def sample_parsing_result(self):
        """Sample CVParsingResult for testing."""
        return CVParsingResult(
            personal_info=CVParsingPersonalInfo(
                name="John Doe",
                email="john.doe@example.com",
                phone="555-123-4567",
                linkedin="https://linkedin.com/in/johndoe"
            ),
            sections=[
                CVParsingSection(
                    name="Professional Experience",
                    subsections=[
                        CVParsingSubsection(
                            name="Work History",
                            items=[
                                "Senior Developer at Tech Corp (2020-2023)",
                                "Junior Developer at StartupCo (2018-2020)"
                            ]
                        )
                    ]
                ),
                CVParsingSection(
                    name="Education",
                    subsections=[
                        CVParsingSubsection(
                            name="Degrees",
                            items=[
                                "BS Computer Science, University of Tech (2018)"
                            ]
                        )
                    ]
                ),
                CVParsingSection(
                    name="Key Qualifications",
                    subsections=[
                        CVParsingSubsection(
                            name="Technical Skills",
                            items=[
                                "Python, JavaScript, React"
                            ]
                        )
                    ]
                )
            ]
        )

    def test_agent_initialization(self, user_cv_parser_agent):
        """Test that the agent initializes correctly."""
        assert user_cv_parser_agent.name == "UserCVParserAgent"
        assert user_cv_parser_agent.session_id == "test_session"
        assert user_cv_parser_agent.settings == {"test_setting": "test_value"}
        assert hasattr(user_cv_parser_agent, 'llm_cv_parser_service')
        assert hasattr(user_cv_parser_agent, 'vector_store_service')

    @pytest.mark.asyncio
    async def test_parse_cv_empty_text(self, user_cv_parser_agent):
        """Test parsing with empty CV text."""
        with pytest.raises(AgentExecutionError, match="Cannot parse an empty CV"):
            await user_cv_parser_agent.parse_cv("")

    @pytest.mark.asyncio
    async def test_parse_cv_whitespace_only(self, user_cv_parser_agent):
        """Test parsing with whitespace-only CV text."""
        with pytest.raises(AgentExecutionError, match="Cannot parse an empty CV"):
            await user_cv_parser_agent.parse_cv("   \n\t   ")

    @pytest.mark.asyncio
    async def test_parse_cv_successful(self, user_cv_parser_agent, sample_cv_text, sample_parsing_result):
        """Test successful CV parsing."""
        # Mock the LLM CV parser service
        user_cv_parser_agent.llm_cv_parser_service.parse_cv_with_llm = AsyncMock(return_value=sample_parsing_result)
        
        result = await user_cv_parser_agent.parse_cv(sample_cv_text)
        
        # Verify the result
        assert isinstance(result, StructuredCV)
        
        # Verify sections were created correctly
        section_names = [section.name for section in result.sections]
        assert "Professional Experience" in section_names
        assert "Education" in section_names
        assert "Key Qualifications" in section_names
        
        # Note: Standard sections are not preserved in current implementation
        # The conversion method replaces all sections with parsed ones only

    @pytest.mark.asyncio
    async def test_execute_successful(self, user_cv_parser_agent, sample_cv_text, sample_parsing_result):
        """Test successful execution of the agent."""
        # Mock the LLM CV parser service
        user_cv_parser_agent.llm_cv_parser_service.parse_cv_with_llm = AsyncMock(return_value=sample_parsing_result)
        
        input_data = {"input_data": {"raw_text": sample_cv_text}}
        result = await user_cv_parser_agent._execute(**input_data)
        
        # Verify the result
        assert isinstance(result, AgentResult)
        assert result.success is True
        assert isinstance(result.output_data, ParserAgentOutput)
        assert isinstance(result.output_data.structured_cv, StructuredCV)

    def test_convert_cv_parsing_result_to_structured_cv(self, user_cv_parser_agent, sample_cv_text, sample_parsing_result):
        """Test conversion of CVParsingResult to StructuredCV."""
        structured_cv = user_cv_parser_agent._convert_cv_parsing_result_to_structured_cv(
            sample_parsing_result, sample_cv_text
        )
        
        assert isinstance(structured_cv, StructuredCV)
        
        # Check that sections were created with correct names
        section_names = [section.name for section in structured_cv.sections]
        assert "Professional Experience" in section_names
        assert "Education" in section_names
        assert "Key Qualifications" in section_names
        
        # Note: Standard sections are not preserved in current implementation
        # The conversion method replaces all sections with parsed ones only
        
        # Verify section structure
        prof_exp_section = next(s for s in structured_cv.sections if s.name == "Professional Experience")
        assert len(prof_exp_section.subsections) == 1
        assert prof_exp_section.subsections[0].name == "Work History"
        assert len(prof_exp_section.subsections[0].items) == 2
        
        # Verify items have correct status
        for item in prof_exp_section.subsections[0].items:
            assert item.status == ItemStatus.GENERATED

    def test_convert_parsing_result_empty_sections(self, user_cv_parser_agent, sample_cv_text):
        """Test conversion with empty parsing result."""
        empty_parsing_result = CVParsingResult(
            personal_info=CVParsingPersonalInfo(
                name="John Doe",
                email="john.doe@example.com",
                phone="555-123-4567"
            ),
            sections=[]
        )
        
        structured_cv = user_cv_parser_agent._convert_cv_parsing_result_to_structured_cv(
            empty_parsing_result, sample_cv_text
        )
        
        assert isinstance(structured_cv, StructuredCV)
        
        # With empty parsing result, required sections should still be created
        # The agent ensures all required sections exist even if LLM doesn't return them
        section_names = [section.name for section in structured_cv.sections]
        required_sections = ["Executive Summary", "Key Qualifications", "Professional Experience", "Project Experience", "Education"]
        
        # All required sections should be present
        assert len(section_names) == 5  # All required sections created
        for required_section in required_sections:
            assert required_section in section_names
        
        # All sections should have INITIAL status since no LLM data was provided
        for section in structured_cv.sections:
            assert section.status == ItemStatus.INITIAL
            assert len(section.items) == 0  # No items since no LLM data
            assert len(section.subsections) == 0  # No subsections since no LLM data

    def test_section_with_direct_items(self, user_cv_parser_agent, sample_cv_text):
        """Test conversion of sections with direct items (no subsections)."""
        parsing_result = CVParsingResult(
            personal_info=CVParsingPersonalInfo(
                name="Test User",
                email="test@example.com",
                phone="555-000-0000"
            ),
            sections=[
                CVParsingSection(
                    name="Skills",
                    items=["Python", "JavaScript", "React"],
                    subsections=[]
                )
            ]
        )
        
        structured_cv = user_cv_parser_agent._convert_cv_parsing_result_to_structured_cv(
            parsing_result, sample_cv_text
        )
        
        # Find the Skills section
        skills_section = next(s for s in structured_cv.sections if s.name == "Skills")
        
        # Should have one subsection named "Main" with the direct items
        assert len(skills_section.subsections) == 1
        assert skills_section.subsections[0].name == "Main"
        assert len(skills_section.subsections[0].items) == 3
        
        # Verify item contents
        item_contents = [item.content for item in skills_section.subsections[0].items]
        assert "Python" in item_contents
        assert "JavaScript" in item_contents
        assert "React" in item_contents

    @pytest.mark.asyncio
    async def test_vector_store_integration(self, user_cv_parser_agent, sample_cv_text, sample_parsing_result):
        """Test that vector store is called during parsing."""
        # Mock the LLM CV parser service
        user_cv_parser_agent.llm_cv_parser_service.parse_cv_with_llm = AsyncMock(return_value=sample_parsing_result)
        
        await user_cv_parser_agent.parse_cv(sample_cv_text)
        
        # Verify vector store was called
        user_cv_parser_agent.vector_store_service.add_structured_cv.assert_called_once()

    @pytest.mark.asyncio
    async def test_vector_store_error_handling(self, user_cv_parser_agent, sample_cv_text, sample_parsing_result):
        """Test that vector store errors don't fail the parsing."""
        # Mock the LLM CV parser service
        user_cv_parser_agent.llm_cv_parser_service.parse_cv_with_llm = AsyncMock(return_value=sample_parsing_result)
        
        # Mock vector store to raise an error
        from src.error_handling.exceptions import VectorStoreError
        user_cv_parser_agent.vector_store_service.add_structured_cv.side_effect = VectorStoreError("Vector store failed")
        
        # Should not raise an exception
        result = await user_cv_parser_agent.parse_cv(sample_cv_text)
        assert isinstance(result, StructuredCV)

    @pytest.mark.asyncio
    async def test_progress_tracking(self, user_cv_parser_agent, sample_cv_text, sample_parsing_result):
        """Test that progress is tracked during parsing."""
        # Mock the LLM CV parser service
        user_cv_parser_agent.llm_cv_parser_service.parse_cv_with_llm = AsyncMock(return_value=sample_parsing_result)
        
        with patch.object(user_cv_parser_agent, 'update_progress') as mock_update_progress:
            await user_cv_parser_agent.parse_cv(sample_cv_text)
            
            # Verify progress tracking was called
            assert mock_update_progress.call_count >= 3  # At least 3 progress updates
            
            # Check specific progress calls
            progress_calls = [call[0] for call in mock_update_progress.call_args_list]
            assert any(AgentConstants.PROGRESS_MAIN_PROCESSING in call for call in progress_calls)
            assert any(AgentConstants.PROGRESS_COMPLETE in call for call in progress_calls)

    def test_section_field_mapping(self, user_cv_parser_agent, sample_cv_text):
        """Test that Section and Subsection objects are created with correct field names."""
        # Create a simple parsing result to test field mapping
        parsing_result = CVParsingResult(
            personal_info=CVParsingPersonalInfo(
                name="Test User",
                email="test@example.com",
                phone="555-000-0000"
            ),
            sections=[
                CVParsingSection(
                    name="Test Section",
                    subsections=[
                        CVParsingSubsection(
                            name="Test Subsection",
                            items=[
                                "Test content"
                            ]
                        )
                    ]
                )
            ]
        )
        
        structured_cv = user_cv_parser_agent._convert_cv_parsing_result_to_structured_cv(
            parsing_result, sample_cv_text
        )
        
        # Verify that Section uses 'name' field (not 'title')
        test_section = next(s for s in structured_cv.sections if s.name == "Test Section")
        assert hasattr(test_section, 'name')
        assert test_section.name == "Test Section"
        
        # Verify that Subsection uses 'name' field (not 'title')
        test_subsection = test_section.subsections[0]
        assert hasattr(test_subsection, 'name')
        assert test_subsection.name == "Test Subsection"

    def test_structured_cv_metadata_population(self, user_cv_parser_agent, sample_cv_text, sample_parsing_result):
        """Test that StructuredCV metadata is populated correctly."""
        structured_cv = user_cv_parser_agent._convert_cv_parsing_result_to_structured_cv(
            sample_parsing_result, sample_cv_text
        )
        
        # Verify metadata contains original CV text
        assert structured_cv.metadata.extra["original_cv_text"] == sample_cv_text
        
        # Verify metadata structure
        assert hasattr(structured_cv, 'metadata')
        assert hasattr(structured_cv.metadata, 'extra')