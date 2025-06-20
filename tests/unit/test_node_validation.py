"""Unit tests for node validation decorator."""

import pytest
from unittest.mock import AsyncMock
from typing import Dict, Any

from src.utils.node_validation import validate_node_output, validate_output_dict
from src.orchestration.state import AgentState
from src.models.data_models import StructuredCV, JobDescriptionData


class TestValidateOutputDict:
    """Test the validate_output_dict function."""

    def test_valid_output_dict(self):
        """Test that valid output dict passes validation."""
        valid_output = {
            "trace_id": "test-123",
            "cv_text": "Updated CV text",
            "error_messages": ["Some error"]
        }
        
        # Should not raise any exception
        validate_output_dict(valid_output)

    def test_invalid_output_dict_with_invalid_key(self):
        """Test that output dict with invalid key filters out invalid keys."""
        invalid_output = {
            "trace_id": "test-123",
            "invalid_key": "some value",
            "cv_text": "Updated CV text"
        }
        
        result = validate_output_dict(invalid_output)
        
        # Should filter out invalid_key but keep valid ones
        assert "invalid_key" not in result
        assert "trace_id" in result
        assert "cv_text" in result
        assert result["trace_id"] == "test-123"
        assert result["cv_text"] == "Updated CV text"

    def test_empty_output_dict(self):
        """Test that empty output dict passes validation."""
        empty_output = {}
        
        # Should not raise any exception
        validate_output_dict(empty_output)

    def test_output_dict_with_nested_structures(self):
        """Test that output dict with complex nested structures passes validation."""
        complex_output = {
            "structured_cv": {
                "sections": [],
                "big_10_skills": []
            },
            "job_description_data": {
                "raw_text": "Job description"
            },
            "items_to_process_queue": ["item1", "item2"]
        }
        
        # Should not raise any exception
        validate_output_dict(complex_output)


class TestValidateNodeOutputDecorator:
    """Test the validate_node_output decorator."""

    @pytest.mark.asyncio
    async def test_decorator_with_valid_output(self):
        """Test decorator passes through valid output unchanged."""
        @validate_node_output
        async def mock_node(state: AgentState) -> Dict[str, Any]:
            return {
                "trace_id": "test-123",
                "cv_text": "Updated text"
            }
        
        # Create a minimal state
        state = AgentState(
            structured_cv=StructuredCV(sections=[]),
            job_description_data=JobDescriptionData(raw_text="test")
        )
        
        result = await mock_node(state)
        
        assert result == {
            "trace_id": "test-123",
            "cv_text": "Updated text"
        }

    @pytest.mark.asyncio
    async def test_decorator_with_invalid_output(self):
        """Test decorator filters out invalid output keys."""
        @validate_node_output
        async def mock_node(state: AgentState) -> Dict[str, Any]:
            return {
                "trace_id": "test-123",
                "invalid_field": "should fail"
            }
        
        # Create a minimal state
        state = AgentState(
            structured_cv=StructuredCV(sections=[]),
            job_description_data=JobDescriptionData(raw_text="test")
        )
        
        result = await mock_node(state)
        
        # Should filter out invalid_field but keep valid ones
        assert "invalid_field" not in result
        assert "trace_id" in result
        assert result["trace_id"] == "test-123"

    @pytest.mark.asyncio
    async def test_decorator_with_empty_output(self):
        """Test decorator allows empty output."""
        @validate_node_output
        async def mock_node(state: AgentState) -> Dict[str, Any]:
            return {}
        
        # Create a minimal state
        state = AgentState(
            structured_cv=StructuredCV(sections=[]),
            job_description_data=JobDescriptionData(raw_text="test")
        )
        
        result = await mock_node(state)
        assert result == {}

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves original function metadata."""
        @validate_node_output
        async def test_node(state: AgentState) -> Dict[str, Any]:
            """Test node function."""
            return {"trace_id": "test"}
        
        assert test_node.__name__ == "test_node"
        assert "Test node function." in test_node.__doc__

    @pytest.mark.asyncio
    async def test_decorator_handles_exceptions_in_node(self):
        """Test that decorator doesn't interfere with exceptions from the node function."""
        @validate_node_output
        async def failing_node(state: AgentState) -> Dict[str, Any]:
            raise RuntimeError("Node failed")
        
        # Create a minimal state
        state = AgentState(
            structured_cv=StructuredCV(sections=[]),
            job_description_data=JobDescriptionData(raw_text="test")
        )
        
        with pytest.raises(RuntimeError, match="Node failed"):
            await failing_node(state)