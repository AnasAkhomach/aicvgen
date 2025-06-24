"""Tests for Task C-03: Enforce Pydantic Model Contracts.

This module tests that agents return proper Pydantic models in their
AgentResult.output_data field and that the validator correctly rejects
raw dictionaries.
"""

import pytest
from pydantic import BaseModel, ValidationError

from src.agents.agent_base import AgentResult
from src.models.agent_output_models import (
    ParserAgentOutput,
    CVAnalyzerAgentOutput,
    ResearchAgentOutput,
)


class TestAgentResultPydanticValidator:
    """Test that AgentResult enforces Pydantic model contracts."""

    def test_agent_result_accepts_pydantic_model(self):
        """Test that AgentResult accepts a valid Pydantic model."""
        output = ParserAgentOutput(job_description_data=None)

        result = AgentResult(success=True, output_data=output, confidence_score=1.0)

        assert result.success is True
        assert isinstance(result.output_data, ParserAgentOutput)

    def test_agent_result_accepts_dict_of_pydantic_models(self):
        """Test that AgentResult accepts a dict of Pydantic models."""
        output1 = ParserAgentOutput(job_description_data=None)
        output2 = CVAnalyzerAgentOutput(analysis_results=None)

        result = AgentResult(
            success=True,
            output_data={"parser": output1, "analyzer": output2},
            confidence_score=1.0,
        )

        assert result.success is True
        assert isinstance(result.output_data, dict)
        assert isinstance(result.output_data["parser"], ParserAgentOutput)
        assert isinstance(result.output_data["analyzer"], CVAnalyzerAgentOutput)

    def test_agent_result_rejects_raw_dictionary(self):
        """Test that AgentResult validator rejects raw dictionaries."""
        with pytest.raises(TypeError) as exc_info:
            AgentResult(
                success=True, output_data={"raw": "dictionary"}, confidence_score=1.0
            )

        assert "All values in output_data dict must be Pydantic models" in str(
            exc_info.value
        )

    def test_agent_result_rejects_none_output_data(self):
        """Test that AgentResult validator rejects None output_data."""
        with pytest.raises(ValueError) as exc_info:
            AgentResult(success=True, output_data=None, confidence_score=1.0)

        assert "output_data must not be None" in str(exc_info.value)

    def test_agent_result_rejects_invalid_type(self):
        """Test that AgentResult validator rejects invalid types."""
        with pytest.raises(TypeError) as exc_info:
            AgentResult(success=True, output_data="string_output", confidence_score=1.0)

        assert "output_data must be a Pydantic model" in str(exc_info.value)

    def test_agent_result_rejects_mixed_dict(self):
        """Test that AgentResult validator rejects dict with mixed types."""
        output = ParserAgentOutput(job_description_data=None)

        with pytest.raises(TypeError) as exc_info:
            AgentResult(
                success=True,
                output_data={"valid": output, "invalid": "string"},
                confidence_score=1.0,
            )

        assert "All values in output_data dict must be Pydantic models" in str(
            exc_info.value
        )


class TestAgentOutputModels:
    """Test that agent output models are properly defined."""

    def test_parser_agent_output_model(self):
        """Test ParserAgentOutput model."""
        output = ParserAgentOutput(job_description_data=None, structured_cv=None)
        assert output.job_description_data is None
        assert output.structured_cv is None

    def test_cv_analyzer_agent_output_model(self):
        """Test CVAnalyzerAgentOutput model."""
        output = CVAnalyzerAgentOutput(
            analysis_results={"test": "data"},
            recommendations=["rec1", "rec2"],
            compatibility_score=0.85,
        )
        assert output.analysis_results == {"test": "data"}
        assert output.recommendations == ["rec1", "rec2"]
        assert output.compatibility_score == 0.85

    def test_cv_analyzer_agent_output_score_validation(self):
        """Test that compatibility_score is validated to be between 0 and 1."""
        # Valid score
        output = CVAnalyzerAgentOutput(compatibility_score=0.5)
        assert output.compatibility_score == 0.5

        # Invalid scores should raise validation errors
        with pytest.raises(ValidationError):
            CVAnalyzerAgentOutput(compatibility_score=-0.1)

        with pytest.raises(ValidationError):
            CVAnalyzerAgentOutput(compatibility_score=1.1)

    def test_research_agent_output_model(self):
        """Test ResearchAgentOutput model."""
        output = ResearchAgentOutput(
            research_findings={"insights": "data"},
            sources=["source1", "source2"],
            confidence_level=0.9,
        )
        assert output.research_findings == {"insights": "data"}
        assert output.sources == ["source1", "source2"]
        assert output.confidence_level == 0.9

    def test_research_agent_output_defaults(self):
        """Test ResearchAgentOutput default values."""
        output = ResearchAgentOutput()
        assert output.research_findings is None
        assert output.sources == []
        assert output.confidence_level is None
