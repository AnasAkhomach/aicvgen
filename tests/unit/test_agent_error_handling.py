import sys
import os

# Ensure project root (containing src/) is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

"""Unit tests for standardized agent error handling."""

import pytest
from unittest.mock import Mock, patch
from src.utils.agent_error_handling import (
    AgentErrorHandler,
    LLMErrorHandler,
    with_error_handling,
    with_node_error_handling,
)
from src.models.validation_schemas import ValidationError
from src.agents.agent_base import AgentResult, AgentExecutionContext
from src.orchestration.state import AgentState
from src.models.data_models import (
    StructuredCV,
    JobDescriptionData,
    ContentData,
    MetadataModel,
)
from pydantic import BaseModel


class ErrorFallbackModel(BaseModel):
    error: str


class TestAgentErrorHandler:
    """Test cases for AgentErrorHandler."""

    def test_handle_validation_error(self):
        """Test validation error handling."""
        # Pydantic ValidationError requires 'errors' and 'model'
        from pydantic import ValidationError as PydanticValidationError, BaseModel

        class DummyModel(BaseModel):
            x: int

        try:
            DummyModel(x="not_an_int")
        except PydanticValidationError as e:
            error = e
        agent_type = "test_agent"
        fallback_data = JobDescriptionData(
            raw_text="test", skills=[], responsibilities=[]
        )

        result = AgentErrorHandler.handle_validation_error(
            error, agent_type, fallback_data
        )

        assert isinstance(result, AgentResult)
        assert not result.success
        assert isinstance(result.output_data, JobDescriptionData)
        assert result.output_data.raw_text == "test"
        assert result.confidence_score == 0.0
        assert "Input validation failed" in result.error_message
        assert result.metadata["agent_type"] == agent_type
        assert result.metadata["validation_error"] is True
        assert result.metadata["error_type"] == "ValidationError"

    def test_handle_general_error(self):
        """Test general error handling."""
        error = ValueError("Test error")
        agent_type = "test_agent"
        context = "test_method"
        fallback_data = JobDescriptionData(
            raw_text="test", skills=[], responsibilities=[]
        )

        result = AgentErrorHandler.handle_general_error(
            error, agent_type, fallback_data, context
        )

        assert isinstance(result, AgentResult)
        assert not result.success
        assert isinstance(result.output_data, JobDescriptionData)
        assert result.output_data.raw_text == "test"
        assert result.confidence_score == 0.0
        assert str(error) == result.error_message
        assert result.metadata["agent_type"] == agent_type
        assert result.metadata["error_type"] == "ValueError"
        assert result.metadata["context"] == context

    def test_handle_node_error(self):
        """Test node error handling for LangGraph."""
        error = RuntimeError("Test node error")
        agent_type = "test_agent"
        state = AgentState(
            structured_cv=StructuredCV(),
            job_description_data=JobDescriptionData(raw_text="test"),
        )
        context = "node_execution"

        result = AgentErrorHandler.handle_node_error(error, agent_type, state, context)

        assert "error_messages" in result
        assert len(result["error_messages"]) == 1
        assert "test_agent Error in node_execution" in result["error_messages"][0]
        assert "Test node error" in result["error_messages"][0]

    def test_handle_node_error_with_existing_errors(self):
        """Test node error handling when state already has errors."""
        error = RuntimeError("New error")
        agent_type = "test_agent"
        state = AgentState(
            structured_cv=StructuredCV(),
            job_description_data=JobDescriptionData(raw_text="test"),
            error_messages=["Existing error"],
        )

        result = AgentErrorHandler.handle_node_error(error, agent_type, state)

        assert "error_messages" in result
        assert len(result["error_messages"]) == 2
        assert "Existing error" in result["error_messages"]
        assert "test_agent Error: New error" in result["error_messages"][1]

    def test_create_fallback_data_parser(self):
        """Test fallback data creation for parser agent."""
        fallback = AgentErrorHandler.create_fallback_data("parser")

        assert "job_description_data" in fallback
        assert "structured_cv" in fallback
        assert fallback["job_description_data"]["error"] == "Parsing failed"
        assert fallback["job_description_data"]["status"] == "GENERATION_FAILED"
        assert isinstance(fallback["job_description_data"]["skills"], list)

    def test_create_fallback_data_content_writer(self):
        """Test fallback data creation for content writer agent."""
        fallback = AgentErrorHandler.create_fallback_data("content_writer")

        assert "updated_item" in fallback
        assert fallback["updated_item"]["error"] == "Content generation failed"
        assert fallback["updated_item"]["status"] == "GENERATION_FAILED"
        assert fallback["updated_item"]["content"] == "[Content generation failed]"

    def test_create_fallback_data_research(self):
        """Test fallback data creation for research agent."""
        fallback = AgentErrorHandler.create_fallback_data("research")

        assert "research_results" in fallback
        assert "enhanced_job_description" in fallback
        assert fallback["research_results"]["error"] == "Research failed"
        assert isinstance(fallback["research_results"]["company_info"], dict)
        assert isinstance(fallback["research_results"]["industry_trends"], list)

    def test_create_fallback_data_quality_assurance(self):
        """Test fallback data creation for QA agent."""
        fallback = AgentErrorHandler.create_fallback_data("quality_assurance")

        assert "quality_check_results" in fallback
        assert "updated_structured_cv" in fallback
        assert fallback["quality_check_results"]["error"] == "Quality check failed"
        assert "summary" in fallback["quality_check_results"]
        assert fallback["quality_check_results"]["summary"]["total_items"] == 0

    def test_create_fallback_data_unknown_agent(self):
        """Test fallback data creation for unknown agent type."""
        fallback = AgentErrorHandler.create_fallback_data("unknown_agent")
        # Wrap fallback dict in ErrorFallbackModel for AgentResult compatibility
        fallback_model = ErrorFallbackModel(**fallback)
        assert fallback_model.error == "unknown_agent failed"


class TestLLMErrorHandler:
    """Test cases for LLMErrorHandler."""

    def test_handle_llm_response_error(self):
        """Test LLM response error handling."""
        error = ValueError("LLM service error")
        agent_type = "test_agent"
        fallback_content = "Fallback content"

        result = LLMErrorHandler.handle_llm_response_error(
            error, agent_type, fallback_content
        )

        assert result["error"] == str(error)
        assert result["content"] == fallback_content
        assert result["status"] == "GENERATION_FAILED"
        assert result["metadata"]["error_type"] == "ValueError"
        assert result["metadata"]["agent_type"] == agent_type

    def test_handle_llm_response_error_no_fallback(self):
        """Test LLM response error handling without fallback content."""
        error = RuntimeError("Connection error")
        agent_type = "test_agent"

        result = LLMErrorHandler.handle_llm_response_error(error, agent_type)

        assert result["error"] == str(error)
        assert result["content"] == "[Content generation failed]"
        assert result["status"] == "GENERATION_FAILED"

    def test_handle_json_parsing_error(self):
        """Test JSON parsing error handling."""
        import json

        error = json.JSONDecodeError("Invalid JSON", "bad json", 0)
        raw_response = "This is not valid JSON content"
        agent_type = "test_agent"

        result = LLMErrorHandler.handle_json_parsing_error(
            error, raw_response, agent_type
        )

        assert "JSON parsing failed" in result["error"]
        assert result["raw_response"] == raw_response
        assert result["status"] == "PARSING_FAILED"
        assert result["metadata"]["error_type"] == "JSONDecodeError"
        assert result["metadata"]["agent_type"] == agent_type


class TestErrorHandlingDecorators:
    """Test cases for error handling decorators."""

    @pytest.mark.asyncio
    async def test_with_error_handling_async_success(self):
        """Test error handling decorator with successful async function."""

        @with_error_handling("test_agent", "test_context")
        async def test_function():
            return {"success": True}

        result = await test_function()
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_with_error_handling_async_validation_error(self):
        """Test error handling decorator with validation error in async function."""
        from pydantic import ValidationError as PydanticValidationError, BaseModel

        class DummyModel(BaseModel):
            x: int

        @with_error_handling("test_agent", "test_context")
        async def test_function():
            try:
                DummyModel(x="not_an_int")
            except PydanticValidationError as e:
                raise e

        result = await test_function()
        assert isinstance(result, AgentResult)
        assert not result.success
        assert "Input validation failed" in result.error_message
        assert result.metadata["validation_error"] is True
        # output_data should be a Pydantic model
        assert isinstance(result.output_data, BaseModel)
        if hasattr(result.output_data, "error"):
            assert result.output_data.error == "test_agent failed"

    @pytest.mark.asyncio
    async def test_with_error_handling_async_general_error(self):
        """Test error handling decorator with general error in async function."""

        @with_error_handling("test_agent", "test_context")
        async def test_function():
            raise ValueError("Test error")

        result = await test_function()
        assert isinstance(result, AgentResult)
        assert not result.success
        assert result.error_message == "Test error"
        assert result.metadata["error_type"] == "ValueError"
        assert result.metadata["context"] == "test_context"
        # output_data should be a Pydantic model
        assert isinstance(result.output_data, BaseModel)
        if hasattr(result.output_data, "error"):
            assert result.output_data.error == "test_agent failed"

    def test_with_error_handling_sync_success(self):
        """Test error handling decorator with successful sync function."""

        @with_error_handling("test_agent", "test_context")
        def test_function():
            return {"success": True}

        result = test_function()
        assert result["success"] is True

    def test_with_error_handling_sync_error(self):
        """Test error handling decorator with error in sync function."""

        @with_error_handling("test_agent", "test_context")
        def test_function():
            raise RuntimeError("Test error")

        result = test_function()
        assert isinstance(result, AgentResult)
        assert not result.success
        assert result.error_message == "Test error"
        # output_data should be a Pydantic model
        assert isinstance(result.output_data, BaseModel)
        if hasattr(result.output_data, "error"):
            assert result.output_data.error == "test_agent failed"

    @pytest.mark.asyncio
    async def test_with_node_error_handling_async_success(self):
        """Test node error handling decorator with successful async function."""

        @with_node_error_handling("test_agent", "test_context")
        async def test_method(self, state):
            return {"success": True}

        mock_self = Mock()
        state = AgentState(
            structured_cv=StructuredCV(),
            job_description_data=JobDescriptionData(raw_text="test"),
        )

        result = await test_method(mock_self, state)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_with_node_error_handling_async_error(self):
        """Test node error handling decorator with error in async function."""

        @with_node_error_handling("test_agent", "test_context")
        async def test_method(self, state):
            raise ValueError("Test node error")

        mock_self = Mock()
        state = AgentState(
            structured_cv=StructuredCV(),
            job_description_data=JobDescriptionData(raw_text="test"),
        )

        result = await test_method(mock_self, state)
        assert "error_messages" in result
        assert len(result["error_messages"]) == 1
        assert "test_agent Error in test_context" in result["error_messages"][0]
        assert "Test node error" in result["error_messages"][0]

    def test_with_node_error_handling_sync_error(self):
        """Test node error handling decorator with error in sync function."""

        @with_node_error_handling("test_agent", "test_context")
        def test_method(self, state):
            raise RuntimeError("Test sync error")

        mock_self = Mock()
        state = AgentState(
            structured_cv=StructuredCV(),
            job_description_data=JobDescriptionData(raw_text="test"),
        )

        result = test_method(mock_self, state)
        assert "error_messages" in result
        assert "test_agent Error in test_context" in result["error_messages"][0]
        assert "Test sync error" in result["error_messages"][0]


if __name__ == "__main__":
    pytest.main([__file__])
