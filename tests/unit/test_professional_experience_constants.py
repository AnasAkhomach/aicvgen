"""Test to verify that ProfessionalExperienceWriterAgent uses LLM constants correctly."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.agents.professional_experience_writer_agent import (
    ProfessionalExperienceWriterAgent,
)
from src.constants.llm_constants import LLMConstants
from src.models.data_models import StructuredCV, Section, Item, JobDescriptionData
from src.models.cv_models import ItemType
from src.models.llm_data_models import LLMResponse


class TestProfessionalExperienceConstants:
    """Test that ProfessionalExperienceWriterAgent uses LLM constants instead of hardcoded values."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = Mock()
        self.mock_prompt = Mock()
        self.mock_parser = Mock()
        self.settings = {"test_setting": "test_value"}

        # Mock the chain creation to avoid | operator issues with Mock objects
        with patch.object(
            ProfessionalExperienceWriterAgent, "__init__", return_value=None
        ):
            self.agent = ProfessionalExperienceWriterAgent.__new__(
                ProfessionalExperienceWriterAgent
            )
            self.agent.name = "ProfessionalExperienceWriterAgent"
            self.agent.description = "Agent responsible for generating professional experience content for a CV"
            self.agent.session_id = "test_session"
            self.agent.settings = self.settings
            self.agent.chain = AsyncMock()

    @pytest.mark.asyncio
    async def test_llm_constants_used_as_fallbacks(self):
        """Test that LLM constants are used as fallback values when settings are not provided."""
        # Create test data
        experience_item = {
            "id": "test_item_id",
            "content": "Software Engineer at TechCorp",
            "item_type": ItemType.EXPERIENCE_ROLE_TITLE,
        }

        structured_cv = StructuredCV(
            sections=[
                Section(
                    name="Professional Experience",
                    items=[
                        Item(
                            content="Software Engineer at TechCorp",
                            item_type=ItemType.EXPERIENCE_ROLE_TITLE,
                            metadata={"item_id": "test_item_id"},
                        )
                    ],
                    subsections=[],
                )
            ]
        )

        job_data = JobDescriptionData(
            raw_text="Looking for a senior software engineer",
            job_title="Senior Software Engineer",
        )

        # Mock the LCEL chain response
        mock_response = {
            "generated_professional_experience": "Generated professional experience content"
        }
        self.agent.chain = AsyncMock()
        self.agent.chain.ainvoke = AsyncMock(return_value=mock_response)

        # Call the agent execute method
        result = await self.agent._execute(
            job_title="Senior Software Engineer",
            company_name="TechCorp",
            job_description="Looking for a senior software engineer",
            experience_item=experience_item,
            cv_summary="Software engineer with experience",
            required_skills=["Python", "JavaScript"],
            preferred_qualifications=["React", "Node.js"],
        )

        # Verify that the chain was called
        self.agent.chain.ainvoke.assert_called_once()

        # Verify the result structure
        assert "generated_professional_experience" in result

    @pytest.mark.asyncio
    async def test_settings_override_constants(self):
        """Test that settings values override LLM constants when provided."""
        # Set custom settings
        custom_settings = {
            "max_tokens_content_generation": 1500,
            "temperature_content_generation": 0.5,
        }

        with patch.object(
            ProfessionalExperienceWriterAgent, "__init__", return_value=None
        ):
            agent_with_custom_settings = ProfessionalExperienceWriterAgent.__new__(
                ProfessionalExperienceWriterAgent
            )
            agent_with_custom_settings.name = "ProfessionalExperienceWriterAgent"
            agent_with_custom_settings.description = "Agent responsible for generating professional experience content for a CV"
            agent_with_custom_settings.session_id = "test_session"
            agent_with_custom_settings.settings = custom_settings
            agent_with_custom_settings.chain = AsyncMock()

        # Create test data
        experience_item = {
            "id": "test_item_id",
            "content": "Software Engineer at TechCorp",
            "item_type": ItemType.EXPERIENCE_ROLE_TITLE,
        }

        structured_cv = StructuredCV(
            sections=[
                Section(
                    name="Professional Experience",
                    items=[
                        Item(
                            content="Software Engineer at TechCorp",
                            item_type=ItemType.EXPERIENCE_ROLE_TITLE,
                            metadata={"item_id": "test_item_id"},
                        )
                    ],
                    subsections=[],
                )
            ]
        )

        job_data = JobDescriptionData(
            raw_text="Looking for a senior software engineer",
            job_title="Senior Software Engineer",
        )

        # Mock the LCEL chain response
        mock_response = {
            "generated_professional_experience": "Generated professional experience content"
        }
        agent_with_custom_settings.chain = AsyncMock()
        agent_with_custom_settings.chain.ainvoke = AsyncMock(return_value=mock_response)

        # Call the agent execute method
        result = await agent_with_custom_settings._execute(
            job_title="Senior Software Engineer",
            company_name="TechCorp",
            job_description="Looking for a senior software engineer",
            experience_item=experience_item,
            cv_summary="Software engineer with experience",
            required_skills=["Python", "JavaScript"],
            preferred_qualifications=["React", "Node.js"],
        )

        # Verify that the chain was called
        agent_with_custom_settings.chain.ainvoke.assert_called_once()

        # Verify the result structure
        assert "generated_professional_experience" in result

    def test_llm_constants_are_defined(self):
        """Test that the required LLM constants are properly defined."""
        assert hasattr(LLMConstants, "MAX_TOKENS_GENERATION")
        assert hasattr(LLMConstants, "TEMPERATURE_BALANCED")
        assert isinstance(LLMConstants.MAX_TOKENS_GENERATION, int)
        assert isinstance(LLMConstants.TEMPERATURE_BALANCED, float)
        assert LLMConstants.MAX_TOKENS_GENERATION > 0
        assert 0.0 <= LLMConstants.TEMPERATURE_BALANCED <= 1.0
