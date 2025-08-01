"""Tests for structured output functionality."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from src.models.agent_output_models import (
    CompanyInsight,
    IndustryInsight,
    ResearchFindings,
    ResearchStatus,
    RoleInsight,
)
from src.services.llm_service import EnhancedLLMService
from src.services.llm_service_interface import LLMServiceInterface


class TestStructuredOutput:
    """Test cases for structured output functionality."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for EnhancedLLMService."""
        settings = MagicMock()
        settings.llm_settings.default_model = "test-model"

        api_key_manager = MagicMock()
        api_key_manager.ensure_api_key_valid = AsyncMock()

        return {
            "settings": settings,
            "llm_client": AsyncMock(),
            "api_key_manager": api_key_manager,
        }

    @pytest.fixture
    def llm_service(self, mock_dependencies):
        """Create EnhancedLLMService instance for testing."""
        return EnhancedLLMService(**mock_dependencies)

    def test_interface_has_structured_output_method(self):
        """Test that LLMServiceInterface includes generate_structured_content method."""
        assert hasattr(LLMServiceInterface, "generate_structured_content")
        assert callable(getattr(LLMServiceInterface, "generate_structured_content"))

    def test_enhanced_service_implements_structured_output(self, llm_service):
        """Test that EnhancedLLMService implements generate_structured_content."""
        assert hasattr(llm_service, "generate_structured_content")
        assert callable(getattr(llm_service, "generate_structured_content"))

    @pytest.mark.asyncio
    async def test_generate_structured_content_success(
        self, llm_service, mock_dependencies
    ):
        """Test successful structured content generation."""
        # Setup mock response with proper nested models
        company_insight = CompanyInsight(
            company_name="Test Company", industry="Technology", confidence_score=0.9
        )

        industry_insight = IndustryInsight(
            industry_name="Technology", trends=["AI", "Cloud"], confidence_score=0.85
        )

        role_insight = RoleInsight(
            role_title="Software Engineer",
            required_skills=["Python", "SQL"],
            confidence_score=0.8,
        )

        expected_result = ResearchFindings(
            status=ResearchStatus.SUCCESS,
            company_insights=company_insight,
            industry_insights=industry_insight,
            role_insights=role_insight,
            key_terms=["python", "ai"],
            skill_gaps=["machine learning"],
            enhancement_suggestions=["Add ML projects"],
        )

        # Mock the LangChain structured LLM
        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke.return_value = expected_result

        # Mock the get_llm method to return a mock LLM
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        llm_service.get_llm = MagicMock(return_value=mock_llm)

        # Test the method
        result = await llm_service.generate_structured_content(
            prompt="Test prompt",
            response_model=ResearchFindings,
            system_instruction="Test instruction",
        )

        # Verify the result
        assert isinstance(result, ResearchFindings)
        assert result.status == ResearchStatus.SUCCESS
        assert result.company_insights.company_name == "Test Company"
        assert result.industry_insights.industry_name == "Technology"
        assert result.role_insights.role_title == "Software Engineer"
        assert result.key_terms == ["python", "ai"]

        # Verify the LangChain methods were called correctly
        llm_service.get_llm.assert_called_once()
        mock_llm.with_structured_output.assert_called_once_with(ResearchFindings)
        # system_instruction should be filtered out and not passed to ainvoke
        mock_structured_llm.ainvoke.assert_called_once_with("Test prompt")

    @pytest.mark.asyncio
    async def test_generate_structured_content_error_handling(
        self, llm_service, mock_dependencies
    ):
        """Test error handling in structured content generation."""
        # Mock the LangChain structured LLM to raise an exception
        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke.side_effect = Exception("Test error")

        # Mock the get_llm method to return a mock LLM
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        llm_service.get_llm = MagicMock(return_value=mock_llm)

        # Test that the method propagates the exception
        with pytest.raises(Exception, match="Test error"):
            await llm_service.generate_structured_content(
                prompt="Test prompt", response_model=ResearchFindings
            )
