"""Test to verify that ProfessionalExperienceWriterAgent uses LLM constants correctly."""

import pytest
from unittest.mock import Mock, AsyncMock
from src.agents.professional_experience_writer_agent import ProfessionalExperienceWriterAgent
from src.constants.llm_constants import LLMConstants
from src.models.data_models import StructuredCV, Section, Item, JobDescriptionData
from src.models.cv_models import ItemType
from src.models.llm_data_models import LLMResponse


class TestProfessionalExperienceConstants:
    """Test that ProfessionalExperienceWriterAgent uses LLM constants instead of hardcoded values."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm_service = Mock()
        self.mock_template_manager = Mock()
        self.settings = {"test_setting": "test_value"}

        self.agent = ProfessionalExperienceWriterAgent(
            llm_service=self.mock_llm_service,
            template_manager=self.mock_template_manager,
            settings=self.settings,
            session_id="test_session"
        )

    @pytest.mark.asyncio
    async def test_llm_constants_used_as_fallbacks(self):
        """Test that LLM constants are used as fallback values when settings are not provided."""
        # Create test data
        experience_item = {
            "id": "test_item_id",
            "content": "Software Engineer at TechCorp",
            "item_type": ItemType.EXPERIENCE_ROLE_TITLE
        }

        structured_cv = StructuredCV(
            sections=[
                Section(
                    name="Professional Experience",
                    items=[
                        Item(
                            content="Software Engineer at TechCorp",
                            item_type=ItemType.EXPERIENCE_ROLE_TITLE,
                            metadata={"item_id": "test_item_id"}
                        )
                    ],
                    subsections=[]
                )
            ]
        )

        job_data = JobDescriptionData(
            title="Senior Software Engineer",
            description="Looking for a senior software engineer"
        )

        # Mock template manager
        self.mock_template_manager.get_template_by_type.return_value = "test_template"
        self.mock_template_manager.format_template.return_value = "formatted_prompt"

        # Mock LLM service response
        mock_response = LLMResponse(content="Generated professional experience content")
        self.mock_llm_service.generate_content = AsyncMock(return_value=mock_response)

        # Call the method
        await self.agent._generate_professional_experience_content(
            structured_cv, job_data, experience_item, None
        )

        # Verify that LLM service was called with constants as fallbacks
        self.mock_llm_service.generate_content.assert_called_once()
        call_args = self.mock_llm_service.generate_content.call_args

        # Check that the fallback values match LLM constants
        assert call_args.kwargs["max_tokens"] == LLMConstants.MAX_TOKENS_GENERATION
        assert call_args.kwargs["temperature"] == LLMConstants.TEMPERATURE_BALANCED

    @pytest.mark.asyncio
    async def test_settings_override_constants(self):
        """Test that settings values override LLM constants when provided."""
        # Set custom settings
        custom_settings = {
            "max_tokens_content_generation": 1500,
            "temperature_content_generation": 0.5
        }

        agent_with_custom_settings = ProfessionalExperienceWriterAgent(
            llm_service=self.mock_llm_service,
            template_manager=self.mock_template_manager,
            settings=custom_settings,
            session_id="test_session"
        )

        # Create test data
        experience_item = {
            "id": "test_item_id",
            "content": "Software Engineer at TechCorp",
            "item_type": ItemType.EXPERIENCE_ROLE_TITLE
        }

        structured_cv = StructuredCV(
            sections=[
                Section(
                    name="Professional Experience",
                    items=[
                        Item(
                            content="Software Engineer at TechCorp",
                            item_type=ItemType.EXPERIENCE_ROLE_TITLE,
                            metadata={"item_id": "test_item_id"}
                        )
                    ],
                    subsections=[]
                )
            ]
        )

        job_data = JobDescriptionData(
            title="Senior Software Engineer",
            description="Looking for a senior software engineer"
        )

        # Mock template manager
        self.mock_template_manager.get_template_by_type.return_value = "test_template"
        self.mock_template_manager.format_template.return_value = "formatted_prompt"

        # Mock LLM service response
        mock_response = LLMResponse(content="Generated professional experience content")
        self.mock_llm_service.generate_content = AsyncMock(return_value=mock_response)

        # Call the method
        await agent_with_custom_settings._generate_professional_experience_content(
            structured_cv, job_data, experience_item, None
        )

        # Verify that LLM service was called with custom settings
        self.mock_llm_service.generate_content.assert_called_once()
        call_args = self.mock_llm_service.generate_content.call_args

        # Check that custom settings are used
        assert call_args.kwargs["max_tokens"] == 1500
        assert call_args.kwargs["temperature"] == 0.5

    def test_llm_constants_are_defined(self):
        """Test that the required LLM constants are properly defined."""
        assert hasattr(LLMConstants, 'MAX_TOKENS_GENERATION')
        assert hasattr(LLMConstants, 'TEMPERATURE_BALANCED')
        assert isinstance(LLMConstants.MAX_TOKENS_GENERATION, int)
        assert isinstance(LLMConstants.TEMPERATURE_BALANCED, float)
        assert LLMConstants.MAX_TOKENS_GENERATION > 0
        assert 0.0 <= LLMConstants.TEMPERATURE_BALANCED <= 1.0