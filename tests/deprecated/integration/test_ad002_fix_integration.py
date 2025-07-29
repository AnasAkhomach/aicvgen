"""Integration test for AD-002 architectural fix.

This test verifies that the explicit input mapping works with real agent classes
and reduces coupling between agents and AgentState.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.agents.cv_analyzer_agent import CVAnalyzerAgent
from src.agents.executive_summary_writer_agent import ExecutiveSummaryWriterAgent
from src.models.agent_input_models import extract_agent_inputs
from src.models.cv_models import JobDescriptionData, StructuredCV
from src.orchestration.state import AgentState


class TestAD002FixIntegration:
    """Integration tests for the AD-002 architectural fix."""

    def create_real_agent_state(self) -> AgentState:
        """Create a real AgentState for integration testing."""
        # Create minimal real objects instead of mocks
        structured_cv = StructuredCV(
            personal_info={"name": "John Doe", "email": "john@example.com"},
            experience=[],
            education=[],
            skills=[],
            projects=[],
        )

        job_description_data = JobDescriptionData(
            title="Software Engineer",
            company="Test Company",
            requirements=["Python", "Testing"],
            responsibilities=["Develop software"],
            raw_text="Software Engineer position at Test Company",
        )

        return AgentState(
            session_id="integration-test-session",
            trace_id="trace-123",
            structured_cv=structured_cv,
            job_description_data=job_description_data,
            cv_text="John Doe CV text",
            current_item_id="item-123",
            error_messages=[],
            research_findings=None,
        )

    def test_cv_analyzer_agent_input_extraction(self):
        """Test that CVAnalyzerAgent works with explicit input mapping."""
        state = self.create_real_agent_state()

        # Extract inputs using the new system
        inputs = extract_agent_inputs("CVAnalyzerAgent", state)

        # Verify the extracted inputs contain only what CVAnalyzerAgent needs
        expected_fields = {"session_id", "cv_data", "job_description"}
        assert set(inputs.keys()) == expected_fields

        # Verify the values are correctly mapped
        assert inputs["session_id"] == state.session_id
        assert inputs["cv_data"] == state.structured_cv.model_dump()
        assert inputs["job_description"] == state.job_description_data.model_dump()

        # Verify it doesn't contain unnecessary state fields
        assert "error_messages" not in inputs
        assert "trace_id" not in inputs
        assert "current_item_id" not in inputs

    def test_executive_summary_writer_input_extraction(self):
        """Test that ExecutiveSummaryWriterAgent works with explicit input mapping."""
        state = self.create_real_agent_state()

        # Extract inputs using the new system
        inputs = extract_agent_inputs("ExecutiveSummaryWriter", state)

        # Verify the extracted inputs contain only what ExecutiveSummaryWriter needs
        expected_fields = {
            "session_id",
            "structured_cv",
            "job_description_data",
            "research_findings",
        }
        assert set(inputs.keys()) == expected_fields

        # Verify the values are correctly mapped
        assert inputs["session_id"] == state.session_id
        assert inputs["structured_cv"] == state.structured_cv.model_dump()
        assert inputs["job_description_data"] == state.job_description_data.model_dump()
        assert inputs["research_findings"] is None  # Not set in test state

        # Verify it doesn't contain unnecessary state fields
        assert "error_messages" not in inputs
        assert "trace_id" not in inputs
        assert "cv_text" not in inputs  # Not needed by ExecutiveSummaryWriter

    @pytest.mark.asyncio
    async def test_cv_analyzer_agent_run_as_node_integration(self):
        """Test CVAnalyzerAgent.run_as_node with the new input mapping."""
        state = self.create_real_agent_state()

        # Mock the LLM service and other dependencies
        with patch(
            "src.agents.cv_analyzer_agent.CVAnalyzerAgent._execute"
        ) as mock_execute:
            # Mock the execute method to return a valid result
            mock_result = Mock()
            mock_result.success = True
            mock_result.output_data = Mock()
            mock_result.agent_name = "CVAnalyzerAgent"
            mock_execute.return_value = mock_result

            # Create agent with mocked dependencies
            agent = CVAnalyzerAgent(llm_service=Mock(), session_id="test-session")

            # Run the agent using the base class run_as_node
            result_state = await agent.run_as_node(state)

            # Verify the agent was called
            mock_execute.assert_called_once()

            # Verify the agent received only the expected inputs (not the full state)
            call_kwargs = mock_execute.call_args[1]
            # The agent should receive extracted inputs, not the full state
            assert "session_id" in call_kwargs
            assert "cv_data" in call_kwargs
            assert "job_description" in call_kwargs

            # Verify it didn't receive unnecessary state fields
            assert "error_messages" not in call_kwargs
            assert "trace_id" not in call_kwargs
            assert "current_item_id" not in call_kwargs

    def test_input_mapping_reduces_coupling(self):
        """Test that the new input mapping reduces coupling between agents and state."""
        state = self.create_real_agent_state()

        # Extract inputs for different agents
        cv_analyzer_inputs = extract_agent_inputs("CVAnalyzerAgent", state)
        exec_summary_inputs = extract_agent_inputs("ExecutiveSummaryWriter", state)

        # Verify each agent only gets its required fields (no extra state fields)
        expected_cv_fields = {"session_id", "cv_data", "job_description"}
        expected_exec_fields = {
            "session_id",
            "structured_cv",
            "job_description_data",
            "research_findings",
        }

        assert set(cv_analyzer_inputs.keys()) == expected_cv_fields
        assert set(exec_summary_inputs.keys()) == expected_exec_fields

        # Verify that state-specific fields are not passed to agents
        state_only_fields = {
            "trace_id",
            "current_section_key",
            "current_section_index",
            "items_to_process_queue",
            "current_item_id",
            "is_initial_generation",
            "content_generation_queue",
            "error_messages",
        }

        for field in state_only_fields:
            assert field not in cv_analyzer_inputs
            assert field not in exec_summary_inputs

    def test_agent_input_models_are_type_safe(self):
        """Test that agent input models provide type safety."""
        from src.models.agent_input_models import (
            CVAnalyzerAgentInput,
            ExecutiveSummaryWriterAgentInput,
        )

        state = self.create_real_agent_state()

        # Test CVAnalyzerAgentInput validation
        cv_inputs = extract_agent_inputs("CVAnalyzerAgent", state)
        validated_cv_input = CVAnalyzerAgentInput(**cv_inputs)

        assert validated_cv_input.session_id == state.session_id
        assert validated_cv_input.cv_data is not None  # Should contain CV data
        assert (
            validated_cv_input.job_description is not None
        )  # Should contain job description data

        # Test ExecutiveSummaryWriterAgentInput validation
        exec_inputs = extract_agent_inputs("ExecutiveSummaryWriter", state)

        # extract_agent_inputs already validates and returns a dict
        assert exec_inputs["session_id"] == state.session_id
        assert isinstance(exec_inputs["structured_cv"], dict)
        assert isinstance(exec_inputs["job_description_data"], dict)
        # research_findings should be None since it's explicitly set to None in the test state
        assert exec_inputs["research_findings"] is None

    def test_backward_compatibility_maintained(self):
        """Test that the fix maintains backward compatibility."""
        state = self.create_real_agent_state()

        # The AgentState should still have all its original fields
        assert hasattr(state, "session_id")
        assert hasattr(state, "structured_cv")
        assert hasattr(state, "job_description_data")
        assert hasattr(state, "cv_text")
        assert hasattr(state, "error_messages")

        # The state should still be serializable
        state_dict = state.model_dump()
        assert isinstance(state_dict, dict)
        assert "session_id" in state_dict
        assert "structured_cv" in state_dict

        # Agents should still be able to access state through run_as_node
        # (This is tested in other integration tests)

    def test_error_handling_for_invalid_agent_names(self):
        """Test error handling when extracting inputs for non-existent agents."""
        state = self.create_real_agent_state()

        with pytest.raises(ValueError, match="No input model found for agent"):
            extract_agent_inputs("NonExistentAgent", state)

        with pytest.raises(ValueError, match="No input model found for agent"):
            extract_agent_inputs("", state)

        with pytest.raises(ValueError, match="No input model found for agent"):
            extract_agent_inputs("InvalidAgentName123", state)
