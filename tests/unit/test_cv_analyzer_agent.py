"""Unit tests for CVAnalyzerAgent."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.agents.cv_analyzer_agent import CVAnalyzerAgent
from src.agents.agent_base import AgentExecutionContext, AgentResult


class TestCVAnalyzerAgent:
    """Test cases for CVAnalyzerAgent."""

    @pytest.fixture
    def agent(self):
        """Create a CVAnalyzerAgent instance for testing."""
        return CVAnalyzerAgent(
            name="test_cv_analyzer",
            description="Test CV analyzer agent"
        )

    @pytest.fixture
    def mock_context(self):
        """Create a mock execution context."""
        context = Mock(spec=AgentExecutionContext)
        context.session_id = "test_session_123"
        context.current_item_id = "test_item_456"
        context.item_id = "test_item_456"
        context.content_type = None
        context.retry_count = 0
        context.trace_id = "test_trace_123"
        return context

    @pytest.fixture
    def sample_input_data(self):
        """Sample input data for testing."""
        return {
            "user_cv": {
                "raw_text": "John Doe\nSoftware Engineer\nExperience:\n- 5 years Python development\nSkills:\n- Python, JavaScript"
            },
            "job_description": "Looking for a Python developer",
            "template_cv_path": "/path/to/template.md"
        }

    @pytest.mark.asyncio
    async def test_analyze_cv_async_execution(self, agent, mock_context, sample_input_data):
        """Test that analyze_cv method executes without SyntaxError or RuntimeError.
        
        This test specifically addresses CS-01: Fix await misuse in CVAnalyzerAgent.
        """
        # Mock the LLM service and its methods
        with patch.object(agent, '_generate_and_parse_json', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = {
                "summary": "Experienced Python developer",
                "experiences": ["5 years Python development"],
                "skills": ["Python", "JavaScript"],
                "education": [],
                "projects": []
            }
            
            # This should not raise SyntaxError or RuntimeError
            result = await agent.analyze_cv(sample_input_data, mock_context)
            
            # Verify the result structure
            assert isinstance(result, dict)
            assert "summary" in result
            assert "experiences" in result
            assert "skills" in result
            assert "education" in result
            assert "projects" in result
            
            # Verify the async method was called
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_async_calls_analyze_cv_properly(self, agent, mock_context, sample_input_data):
        """Test that run_async properly awaits analyze_cv method."""
        # Mock validation and other dependencies
        with patch('src.agents.cv_analyzer_agent.validate_agent_input') as mock_validate, \
             patch('src.agents.cv_analyzer_agent.AgentErrorHandler') as mock_error_handler, \
             patch.object(agent, 'analyze_cv', new_callable=AsyncMock) as mock_analyze:
            
            # Setup mocks
            mock_validation_result = Mock()
            mock_validation_result.success = True
            mock_validation_result.result.model_dump.return_value = sample_input_data
            mock_error_handler.handle_validation_error.return_value = mock_validation_result
            
            mock_analyze.return_value = {
                "summary": "Test summary",
                "experiences": [],
                "skills": [],
                "education": [],
                "projects": []
            }
            
            # Execute run_async
            result = await agent.run_async(sample_input_data, mock_context)
            
            # Verify analyze_cv was awaited properly
            mock_analyze.assert_called_once_with(sample_input_data, mock_context)
            
            # Verify result structure
            assert isinstance(result, AgentResult)
            assert result.success is True
            assert "summary" in result.output_data

    @pytest.mark.asyncio
    async def test_analyze_cv_with_empty_cv_text(self, agent, mock_context):
        """Test analyze_cv handles empty CV text gracefully."""
        input_data = {
            "user_cv": {"raw_text": ""},
            "job_description": "Test job"
        }
        
        result = await agent.analyze_cv(input_data, mock_context)
        
        # Should return empty structure for empty input
        assert result == {
            "summary": "",
            "experiences": [],
            "skills": [],
            "education": [],
            "projects": []
        }

    @pytest.mark.asyncio
    async def test_analyze_cv_llm_failure_fallback(self, agent, mock_context, sample_input_data):
        """Test that analyze_cv falls back to basic extraction when LLM fails."""
        with patch.object(agent, '_generate_and_parse_json', new_callable=AsyncMock) as mock_generate:
            # Simulate LLM failure
            mock_generate.side_effect = Exception("LLM service unavailable")
            
            result = await agent.analyze_cv(sample_input_data, mock_context)
            
            # Should return fallback extraction
            assert isinstance(result, dict)
            assert "summary" in result
            assert "Error analyzing CV:" in result["summary"]

    def test_extract_basic_info_synchronous(self, agent):
        """Test that extract_basic_info works synchronously as a fallback."""
        cv_text = """John Doe
        EXPERIENCE:
        - Software Engineer at Company A
        - 5 years Python development
        SKILLS:
        - Python
        - JavaScript
        EDUCATION:
        - BS Computer Science
        """
        
        result = agent.extract_basic_info(cv_text)
        
        assert isinstance(result, dict)
        assert "summary" in result
        assert "experiences" in result
        assert "skills" in result
        assert len(result["experiences"]) > 0
        assert len(result["skills"]) > 0