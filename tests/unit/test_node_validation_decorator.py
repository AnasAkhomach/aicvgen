"""Tests for the @ensure_pydantic_model decorator in node_validation.py.

This test module verifies the functionality of the centralized Pydantic validation decorator
that was implemented as part of REM-AGENT-004.
"""

import pytest
from unittest.mock import AsyncMock
from pydantic import BaseModel, ValidationError
from typing import Dict, Any

from src.utils.node_validation import ensure_pydantic_model
from src.models.cv_models import StructuredCV, JobDescriptionData


class MockModel(BaseModel):
    """Mock Pydantic model for testing."""

    name: str
    value: int


class TestEnsurePydanticModelDecorator:
    """Test cases for the @ensure_pydantic_model decorator."""

    @pytest.mark.asyncio
    async def test_decorator_converts_dict_to_pydantic_model(self):
        """Test that the decorator converts dict fields to Pydantic models."""

        @ensure_pydantic_model(
            ("test_field", MockModel),
        )
        async def mock_function(state: Dict[str, Any]) -> Dict[str, Any]:
            # Verify the field was converted to a Pydantic model
            assert isinstance(state["test_field"], MockModel)
            assert state["test_field"].name == "test"
            assert state["test_field"].value == 42
            return {"result": "success"}

        # Test with dict input
        test_state = {
            "test_field": {"name": "test", "value": 42},
            "other_field": "unchanged",
        }

        result = await mock_function(test_state)
        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_decorator_skips_already_pydantic_models(self):
        """Test that the decorator skips fields that are already Pydantic models."""

        @ensure_pydantic_model(
            ("test_field", MockModel),
        )
        async def mock_function(state: Dict[str, Any]) -> Dict[str, Any]:
            # Verify the field remains unchanged
            assert isinstance(state["test_field"], MockModel)
            assert state["test_field"].name == "existing"
            return {"result": "success"}

        # Test with existing Pydantic model
        existing_model = MockModel(name="existing", value=100)
        test_state = {"test_field": existing_model, "other_field": "unchanged"}

        result = await mock_function(test_state)
        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_decorator_handles_validation_errors(self):
        """Test that the decorator properly handles Pydantic validation errors."""

        @ensure_pydantic_model(
            ("test_field", MockModel),
        )
        async def mock_function(state: Dict[str, Any]) -> Dict[str, Any]:
            return {"result": "success"}

        # Test with invalid data
        test_state = {
            "test_field": {"name": "test"},  # Missing required 'value' field
        }

        with pytest.raises(ValidationError):
            await mock_function(test_state)

    @pytest.mark.asyncio
    async def test_decorator_handles_multiple_fields(self):
        """Test that the decorator can handle multiple field validations."""

        class AnotherModel(BaseModel):
            description: str

        @ensure_pydantic_model(
            ("field1", MockModel),
            ("field2", AnotherModel),
        )
        async def mock_function(state: Dict[str, Any]) -> Dict[str, Any]:
            assert isinstance(state["field1"], MockModel)
            assert isinstance(state["field2"], AnotherModel)
            return {"result": "success"}

        test_state = {
            "field1": {"name": "test1", "value": 1},
            "field2": {"description": "test description"},
            "field3": "unchanged",
        }

        result = await mock_function(test_state)
        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_decorator_handles_missing_fields(self):
        """Test that the decorator gracefully handles missing fields."""

        @ensure_pydantic_model(
            ("missing_field", MockModel),
        )
        async def mock_function(state: Dict[str, Any]) -> Dict[str, Any]:
            # Field should remain missing
            assert "missing_field" not in state
            return {"result": "success"}

        test_state = {"other_field": "present"}

        result = await mock_function(test_state)
        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_decorator_handles_non_dict_fields(self):
        """Test that the decorator handles non-dict field values gracefully."""

        @ensure_pydantic_model(
            ("test_field", MockModel),
        )
        async def mock_function(state: Dict[str, Any]) -> Dict[str, Any]:
            # Field should remain unchanged as it's not a dict
            assert state["test_field"] == "not_a_dict"
            return {"result": "success"}

        test_state = {"test_field": "not_a_dict"}

        result = await mock_function(test_state)
        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_decorator_with_real_cv_models(self):
        """Test the decorator with actual CV models used in the application."""

        @ensure_pydantic_model(
            ("structured_cv", StructuredCV),
            ("job_description", JobDescriptionData),
        )
        async def mock_agent_execute(state: Dict[str, Any]) -> Dict[str, Any]:
            assert isinstance(state["structured_cv"], StructuredCV)
            assert isinstance(state["job_description"], JobDescriptionData)
            return {"result": "success"}

        # Create minimal valid data for the models
        cv_data = {"sections": [], "big_10_skills": [], "metadata": {}}

        job_data = {
            "raw_text": "Test job description for Software Engineer at Test Company",
            "job_title": "Software Engineer",
            "company_name": "Test Company",
            "skills": [],
            "experience_level": "Mid-level",
            "responsibilities": [],
            "industry_terms": [],
            "company_values": [],
        }

        test_state = {"structured_cv": cv_data, "job_description": job_data}

        result = await mock_agent_execute(test_state)
        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_decorator_raises_error_without_state(self):
        """Test that the decorator raises an error when called without state."""

        @ensure_pydantic_model(
            ("test_field", MockModel),
        )
        async def mock_function() -> Dict[str, Any]:
            return {"result": "success"}

        with pytest.raises(ValueError, match="requires state parameter"):
            await mock_function()

    @pytest.mark.asyncio
    async def test_decorator_preserves_original_state(self):
        """Test that the decorator doesn't modify the original state object."""

        @ensure_pydantic_model(
            ("test_field", MockModel),
        )
        async def mock_function(state: Dict[str, Any]) -> Dict[str, Any]:
            return {"result": "success"}

        original_state = {"test_field": {"name": "test", "value": 42}}
        original_state_copy = original_state.copy()

        await mock_function(original_state)

        # Original state should remain unchanged
        assert original_state == original_state_copy
        assert isinstance(original_state["test_field"], dict)
