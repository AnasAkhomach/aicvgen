"""Unit tests for EnhancedContentWriterAgent after parsing refactor."""

import sys
import os

import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.agents.parser_agent import ParserAgent
from src.models.data_models import ContentType, LLMResponse

# Ensure project root (containing src/) is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestEnhancedContentWriterRefactored:
    """Test cases for EnhancedContentWriterAgent after parsing refactor."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("src.agents.enhanced_content_writer.get_llm_service"):
            with patch("src.agents.enhanced_content_writer.get_config"):
                self.agent = EnhancedContentWriterAgent()

    def test_initialization_with_parser_agent(self):
        """Test that EnhancedContentWriterAgent initializes with ParserAgent."""
        assert hasattr(self.agent, "parser_agent")
        assert isinstance(self.agent.parser_agent, ParserAgent)
        assert self.agent.parser_agent.name == "ContentWriterParser"
        assert (
            self.agent.parser_agent.description
            == "Parser agent for content writer parsing methods"
        )

    @pytest.mark.asyncio
    async def test_generate_big_10_skills_uses_parser_agent(self):
        """Test that generate_big_10_skills uses parser_agent for parsing."""
        # Mock LLM response
        mock_response = LLMResponse(
            content="• Python\n• Java\n• SQL\n• Leadership\n• Communication",
            processing_time=1.0,
            tokens_used=150,
        )

        # Mock the LLM service
        self.agent.llm_service.generate_content = AsyncMock(return_value=mock_response)

        # Mock the parser agent method
        expected_skills = [
            "Python",
            "Java",
            "SQL",
            "Leadership",
            "Communication",
            "Problem Solving",
            "Team Collaboration",
            "Analytical Thinking",
            "Project Management",
            "Technical Documentation",
        ]
        self.agent.parser_agent._parse_big_10_skills = Mock(
            return_value=expected_skills
        )

        # Mock template loading
        with patch.object(
            self.agent,
            "_load_prompt_template",
            return_value="Test template: {main_job_description_raw} {my_talents}",
        ):
            result = await self.agent.generate_big_10_skills(
                job_description="Software Engineer position",
                my_talents="Python developer",
            )

        # Verify parser agent was called
        self.agent.parser_agent._parse_big_10_skills.assert_called_once_with(
            mock_response.content
        )

        # Verify result structure
        assert result["success"] is True
        assert result["skills"] == expected_skills
        assert len(result["skills"]) == 10
        assert result["raw_llm_output"] == mock_response.content

    @pytest.mark.asyncio
    async def test_generate_big_10_skills_parser_error_handling(self):
        """Test error handling when parser_agent fails."""
        # Mock LLM response
        mock_response = LLMResponse(
            content="Some response", processing_time=1.0, tokens_used=150
        )

        self.agent.llm_service.generate_content = AsyncMock(return_value=mock_response)

        # Mock parser agent to raise exception
        self.agent.parser_agent._parse_big_10_skills = Mock(
            side_effect=Exception("Parser error")
        )

        # Mock template loading
        with patch.object(
            self.agent,
            "_load_prompt_template",
            return_value="Test template: {main_job_description_raw} {my_talents}",
        ):
            result = await self.agent.generate_big_10_skills(
                job_description="Software Engineer position",
                my_talents="Python developer",
            )

        # Verify error handling
        assert result["success"] is False
        assert "error" in result
        assert result["skills"] == []

    @pytest.mark.asyncio
    async def test_generate_big_10_skills_empty_llm_response(self):
        """Test handling of empty LLM response."""
        # Mock empty LLM response
        mock_response = LLMResponse(content="", processing_time=1.0, tokens_used=100)

        self.agent.llm_service.generate_content = AsyncMock(return_value=mock_response)

        # Mock template loading
        with patch.object(
            self.agent,
            "_load_prompt_template",
            return_value="Test template: {main_job_description_raw} {my_talents}",
        ):
            result = await self.agent.generate_big_10_skills(
                job_description="Software Engineer position",
                my_talents="Python developer",
            )

        # Verify error handling for empty response
        assert result["success"] is False
        assert "Empty response from LLM" in result["error"]
        assert result["skills"] == []

    @pytest.mark.asyncio
    async def test_generate_big_10_skills_no_llm_response(self):
        """Test handling when LLM service returns None."""
        # Mock None LLM response
        self.agent.llm_service.generate_content = AsyncMock(return_value=None)

        # Mock template loading
        with patch.object(
            self.agent,
            "_load_prompt_template",
            return_value="Test template: {main_job_description_raw} {my_talents}",
        ):
            result = await self.agent.generate_big_10_skills(
                job_description="Software Engineer position",
                my_talents="Python developer",
            )

        # Verify error handling for None response
        assert result["success"] is False
        assert "Empty response from LLM" in result["error"]
        assert result["skills"] == []

    def test_format_big_10_skills_display(self):
        """Test formatting of Big 10 skills for display."""
        skills = ["Python", "Java", "SQL", "Leadership", "Communication"]

        result = self.agent._format_big_10_skills_display(skills)

        assert "• Python" in result
        assert "• Java" in result
        assert "• SQL" in result
        assert "• Leadership" in result
        assert "• Communication" in result

    def test_format_big_10_skills_display_empty(self):
        """Test formatting of empty skills list."""
        result = self.agent._format_big_10_skills_display([])

        assert result == "No skills generated"

    @pytest.mark.asyncio
    async def test_integration_with_parser_methods(self):
        """Test integration between EnhancedContentWriter and ParserAgent methods."""
        # Test that all expected parser methods are available
        assert hasattr(self.agent.parser_agent, "_parse_big_10_skills")
        assert hasattr(self.agent.parser_agent, "_parse_bullet_points")
        assert hasattr(self.agent.parser_agent, "_extract_company_from_cv")
        assert hasattr(self.agent.parser_agent, "_parse_cv_text_to_content_item")

        # Test that methods are callable
        assert callable(self.agent.parser_agent._parse_big_10_skills)
        assert callable(self.agent.parser_agent._parse_bullet_points)
        assert callable(self.agent.parser_agent._extract_company_from_cv)
        assert callable(self.agent.parser_agent._parse_cv_text_to_content_item)

    def test_parser_agent_independence(self):
        """Test that parser_agent can be used independently."""
        # Test that parser_agent methods work independently
        test_content = "• Test bullet point\n• Another point"
        result = self.agent.parser_agent._parse_bullet_points(test_content)

        assert isinstance(result, list)
        assert len(result) > 0
        assert "Test bullet point" in result

    @pytest.mark.asyncio
    async def test_backward_compatibility(self):
        """Test that the refactored agent maintains backward compatibility."""
        # Ensure that the public interface hasn't changed
        assert hasattr(self.agent, "generate_big_10_skills")
        assert hasattr(self.agent, "_format_big_10_skills_display")

        # Ensure content_type is still properly set
        assert hasattr(self.agent, "content_type")
        assert self.agent.content_type == ContentType.QUALIFICATION

    def test_content_templates_still_loaded(self):
        """Test that content templates are still properly loaded after refactor."""
        assert hasattr(self.agent, "content_templates")
        assert isinstance(self.agent.content_templates, dict)
        # Should have at least the basic content types
        expected_types = [
            ContentType.QUALIFICATION,
            ContentType.EXPERIENCE,
            ContentType.PROJECT,
            ContentType.EXECUTIVE_SUMMARY,
        ]
        for content_type in expected_types:
            assert content_type in self.agent.content_templates
