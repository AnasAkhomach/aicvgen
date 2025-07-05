"""Test to verify that QualityAssuranceAgent uses constants correctly."""

import pytest
from unittest.mock import Mock
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.constants.agent_constants import AgentConstants
from src.models.data_models import Section, Item


class TestQualityAssuranceConstants:
    """Test that QualityAssuranceAgent uses constants instead of hardcoded values."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm_service = Mock()
        self.mock_template_manager = Mock()
        self.settings = {"test_setting": "test_value"}
        
        self.agent = QualityAssuranceAgent(
            llm_service=self.mock_llm_service,
            template_manager=self.mock_template_manager,
            settings=self.settings,
            session_id="test_session"
        )

    def test_executive_summary_word_count_uses_constants(self):
        """Test that executive summary word count check uses constants."""
        # Create a short executive summary section
        short_summary = Section(
            name="Executive Summary",
            items=[
                Item(content="Short summary with only ten words here for testing purposes.")
            ],
            subsections=[]
        )
        
        # Call the method
        result = self.agent._check_section(short_summary)
        
        # Verify the result contains the expected range using constants
        expected_min = AgentConstants.MIN_WORD_COUNT_EXECUTIVE_SUMMARY
        expected_max = expected_min + AgentConstants.EXECUTIVE_SUMMARY_WORD_COUNT_RANGE
        expected_message = f"Executive Summary is too short (10 words, recommend {expected_min}-{expected_max})"
        
        assert not result.passed
        assert len(result.issues) == 1
        assert result.issues[0] == expected_message

    def test_executive_summary_word_count_passes_with_sufficient_words(self):
        """Test that executive summary passes with sufficient word count."""
        # Create a long enough executive summary
        long_content = " ".join(["word"] * (AgentConstants.MIN_WORD_COUNT_EXECUTIVE_SUMMARY + 10))
        long_summary = Section(
            name="Executive Summary",
            items=[
                Item(content=long_content)
            ],
            subsections=[]
        )
        
        # Call the method
        result = self.agent._check_section(long_summary)
        
        # Verify the result passes
        assert result.passed
        assert len(result.issues) == 0

    def test_constants_are_defined(self):
        """Test that the required constants are properly defined."""
        assert hasattr(AgentConstants, 'MIN_WORD_COUNT_EXECUTIVE_SUMMARY')
        assert hasattr(AgentConstants, 'EXECUTIVE_SUMMARY_WORD_COUNT_RANGE')
        assert isinstance(AgentConstants.MIN_WORD_COUNT_EXECUTIVE_SUMMARY, int)
        assert isinstance(AgentConstants.EXECUTIVE_SUMMARY_WORD_COUNT_RANGE, int)
        assert AgentConstants.MIN_WORD_COUNT_EXECUTIVE_SUMMARY > 0
        assert AgentConstants.EXECUTIVE_SUMMARY_WORD_COUNT_RANGE > 0