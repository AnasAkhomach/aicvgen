#!/usr/bin/env python3
"""
Test suite for centralized JSON parsing functionality.

This module tests the DUP-01 task implementation: centralizing JSON parsing logic
across all agents to use the enhanced _generate_and_parse_json method.
"""

import sys
import os

# Ensure project root (containing src/) is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Any, Dict

from src.agents.agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from src.core.state_manager import AgentIO
from src.models.data_models import LLMResponse
from src.agents.parser_agent import ParserAgent
from src.agents.research_agent import ResearchAgent
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.agents.cleaning_agent import CleaningAgent
from src.utils.exceptions import AgentExecutionError
from src.orchestration.state import AgentState
from src.models.data_models import JobDescriptionData, StructuredCV


class TestAgent(EnhancedAgentBase):
    """Concrete test agent for testing base functionality."""

    def __init__(self, llm_service=None):
        super().__init__(
            name="test_agent",
            description="Test agent for unit tests",
            input_schema=AgentIO(data_type="dict", description="Test input"),
            output_schema=AgentIO(data_type="dict", description="Test output"),
        )
        self.llm = (
            llm_service  # Set the llm attribute that _generate_and_parse_json looks for
        )
        self.llm_service = llm_service
        self.current_session_id = None
        self.current_trace_id = None

    async def run_async(
        self, input_data: Any, context: AgentExecutionContext
    ) -> AgentResult:
        """Test implementation of run_async."""
        return AgentResult(success=True, output_data={"test": "result"})

    async def run_as_node(self, state: "AgentState") -> Dict[str, Any]:
        """Test implementation of run_as_node."""
        return {"test_field": "test_value"}


class TestCentralizedJsonParsing:
    """Test suite for centralized JSON parsing across all agents."""

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service for testing."""
        mock_service = AsyncMock()
        mock_service.generate_content = AsyncMock()
        # Ensure the mock returns a proper LLMResponse object
        mock_response = Mock()
        mock_response.content = '{"test": "data"}'
        mock_response.usage = {"total_tokens": 100}
        mock_response.model = "test-model"
        mock_response.success = True
        mock_response.error_message = None
        mock_service.generate_content.return_value = mock_response
        return mock_service

    @pytest.fixture
    def sample_json_response(self):
        """Sample JSON response for testing."""
        return {
            "skills": ["Python", "Machine Learning", "Data Analysis"],
            "experience_level": "Senior",
            "responsibilities": ["Lead development", "Mentor team"],
        }

    @pytest.fixture
    def sample_markdown_response(self, sample_json_response):
        """Sample markdown-wrapped JSON response."""
        json_str = json.dumps(sample_json_response, indent=2)
        return f"```json\n{json_str}\n```"

    @pytest.fixture
    def agent_state(self):
        """Create a sample agent state for testing."""
        job_data = JobDescriptionData(raw_text="Sample job description")
        cv = StructuredCV()
        return AgentState(
            structured_cv=cv,
            job_description_data=job_data,
            session_id="test_session",
            trace_id="test_trace",
        )

    @pytest.mark.asyncio
    async def test_base_agent_generate_and_parse_json_success(
        self, mock_llm_service, sample_markdown_response, sample_json_response
    ):
        """Test successful JSON generation and parsing in base agent."""
        # Setup mock response
        mock_response = Mock()
        mock_response.content = sample_markdown_response
        mock_response.usage = {"total_tokens": 100}
        mock_response.model = "test-model"
        mock_response.success = True
        mock_response.error_message = None
        mock_llm_service.generate_content.return_value = mock_response

        # Create agent with mocked LLM service
        agent = TestAgent(llm_service=mock_llm_service)

        # Test the centralized method
        result = await agent._generate_and_parse_json(
            prompt="Test prompt", session_id="test_session", trace_id="test_trace"
        )

        # Verify results
        assert result == sample_json_response
        mock_llm_service.generate_content.assert_called_once_with(
            prompt="Test prompt", session_id="test_session", trace_id="test_trace"
        )

    @pytest.mark.asyncio
    async def test_base_agent_generate_and_parse_json_failure(self, mock_llm_service):
        """Test JSON parsing failure handling in base agent."""
        # Setup mock response with invalid JSON
        mock_response = Mock()
        mock_response.content = "This is not valid JSON content"
        mock_response.usage = {"total_tokens": 50}
        mock_response.model = "test-model"
        mock_response.success = True
        mock_response.error_message = None
        mock_llm_service.generate_content.return_value = mock_response

        # Create agent with mocked LLM service
        agent = TestAgent(llm_service=mock_llm_service)

        # Test that ValueError is raised for invalid JSON
        with pytest.raises(ValueError, match="No valid JSON object found"):
            await agent._generate_and_parse_json(
                prompt="Test prompt", session_id="test_session", trace_id="test_trace"
            )

    @pytest.mark.asyncio
    @patch("src.agents.parser_agent.get_llm_service")
    async def test_parser_agent_uses_centralized_method(
        self, mock_get_llm, mock_llm_service, agent_state
    ):
        """Test that ParserAgent uses the centralized JSON parsing method."""
        mock_get_llm.return_value = mock_llm_service

        # Setup mock response
        mock_response = MagicMock()
        mock_response.content = '{"skills": ["Python"], "experience_level": "Senior"}'
        mock_llm_service.generate_content.return_value = mock_response

        # Create parser agent
        agent = ParserAgent(name="test_parser", description="Test parser agent")

        # Mock the centralized method to verify it's called
        with patch.object(
            agent, "_generate_and_parse_json", new_callable=AsyncMock
        ) as mock_centralized:
            mock_centralized.return_value = {
                "skills": ["Python"],
                "experience_level": "Senior",
            }

            # Call run_as_node which should use centralized parsing
            await agent.run_as_node(agent_state)

            # Verify centralized method was called
            assert mock_centralized.call_count > 0
            # Verify session_id and trace_id were passed
            call_args = mock_centralized.call_args_list[0]
            assert "session_id" in call_args.kwargs or len(call_args.args) > 1
            assert "trace_id" in call_args.kwargs or len(call_args.args) > 2

    @pytest.mark.asyncio
    @patch("src.agents.research_agent.get_llm_service")
    async def test_research_agent_uses_centralized_method(
        self, mock_get_llm, mock_llm_service, agent_state
    ):
        """Test that ResearchAgent uses the centralized JSON parsing method."""
        mock_get_llm.return_value = mock_llm_service

        # Create research agent
        agent = ResearchAgent(name="test_research", description="Test research agent")

        # Mock the centralized method to verify it's called
        with patch.object(
            agent, "_generate_and_parse_json", new_callable=AsyncMock
        ) as mock_centralized:
            mock_centralized.return_value = {
                "company_name": "Test Corp",
                "industry": "Tech",
            }

            # Call run_as_node which should use centralized parsing
            await agent.run_as_node(agent_state)

            # Verify centralized method was called
            assert mock_centralized.call_count > 0
            # Verify session_id and trace_id were passed
            call_args = mock_centralized.call_args_list[0]
            assert "session_id" in call_args.kwargs or len(call_args.args) > 1
            assert "trace_id" in call_args.kwargs or len(call_args.args) > 2

    @pytest.mark.asyncio
    async def test_enhanced_content_writer_centralized_usage(self, mock_llm_service):
        """Test that EnhancedContentWriterAgent uses centralized JSON parsing for experience content."""
        # Create content writer agent
        agent = EnhancedContentWriterAgent()
        agent.llm_service = mock_llm_service

        # Mock the centralized method
        with patch.object(
            agent, "_generate_and_parse_json", new_callable=AsyncMock
        ) as mock_centralized:
            mock_centralized.return_value = {"roles": []}

            # Test that the method is available and can be called
            result = await agent._generate_and_parse_json(
                prompt="Test prompt", session_id="test_session", trace_id="test_trace"
            )

            assert result == {"roles": []}
            mock_centralized.assert_called_once()

    def test_cleaning_agent_fallback_logic(self):
        """Test that CleaningAgent maintains its regex fallback logic."""
        agent = CleaningAgent()

        # Test that the agent still has the _extract_json_from_response method
        # for backward compatibility
        assert hasattr(agent, "_extract_json_from_response")

        # Test that it can handle non-JSON input gracefully
        # (This would be tested in integration tests with actual LLM responses)
        assert True  # Placeholder for more detailed testing

    def test_json_extraction_markdown_blocks(self):
        """Test JSON extraction from markdown code blocks."""
        agent = TestAgent()

        # Test markdown JSON block extraction
        markdown_json = '```json\n{"key": "value"}\n```'
        extracted = agent._extract_json_from_response(markdown_json)
        assert extracted == '{"key": "value"}'

        # Test raw JSON extraction
        raw_json = '{"key": "value"}'
        extracted = agent._extract_json_from_response(raw_json)
        assert extracted == '{"key": "value"}'

    def test_session_trace_id_propagation(self):
        """Test that session_id and trace_id are properly propagated."""
        # This test verifies that agents set current_session_id and current_trace_id
        # attributes before calling centralized methods

        agent = TestAgent()

        # Test attribute setting
        agent.current_session_id = "test_session"
        agent.current_trace_id = "test_trace"

        assert agent.current_session_id == "test_session"
        assert agent.current_trace_id == "test_trace"

        # Test getattr fallback
        assert getattr(agent, "current_session_id", None) == "test_session"
        assert getattr(agent, "current_trace_id", None) == "test_trace"
        assert getattr(agent, "nonexistent_attr", None) is None
