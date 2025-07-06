"""Test for session_id validation fix.

This test verifies that the Pydantic validation error where session_id is None
in AgentState has been resolved.
"""

import pytest
from unittest.mock import Mock, patch

from src.orchestration.state import AgentState
from src.models.cv_models import StructuredCV
from src.models.agent_input_models import ResearchAgentInput, extract_agent_inputs
from src.agents.agent_base import AgentBase
from src.core.workflow_manager import WorkflowManager


class TestSessionIdFix:
    """Test cases for session_id validation fix."""

    def test_agent_state_has_session_id_by_default(self):
        """Test that AgentState generates a session_id by default."""
        state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Sample CV text"
        )
        
        assert state.session_id is not None
        assert isinstance(state.session_id, str)
        assert len(state.session_id) > 0

    def test_agent_state_accepts_explicit_session_id(self):
        """Test that AgentState accepts an explicitly provided session_id."""
        explicit_session_id = "test-session-123"
        state = AgentState(
            session_id=explicit_session_id,
            structured_cv=StructuredCV(),
            cv_text="Sample CV text"
        )
        
        assert state.session_id == explicit_session_id

    def test_extract_agent_inputs_with_session_id(self):
        """Test that extract_agent_inputs works with session_id from AgentState."""
        from src.models.cv_models import JobDescriptionData
        
        state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=JobDescriptionData(raw_text="Sample job description")
        )
        
        # Ensure session_id is generated
        assert state.session_id is not None
        
        # Test extracting ResearchAgentInput
        inputs = extract_agent_inputs("ResearchAgent", state)
        
        assert "session_id" in inputs
        assert inputs["session_id"] == state.session_id
        
        # Test that ResearchAgentInput can be created with these inputs
        research_input = ResearchAgentInput(**inputs)
        assert research_input.session_id == state.session_id

    def test_agent_base_initialization_with_session_id(self):
        """Test that AgentBase can initialize with a valid session_id from AgentState."""
        from src.models.agent_models import AgentResult
        
        state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Sample CV text"
        )
        
        # Create a mock agent that inherits from AgentBase
        class MockAgent(AgentBase):
            async def _execute(self, **kwargs):
                return AgentResult.create_success(agent_name="MockAgent", output_data={"result": "success"})
        
        # This should not raise a validation error
        agent = MockAgent(
            name="MockAgent",
            description="Test agent",
            session_id=state.session_id
        )
        assert agent.session_id == state.session_id

    def test_workflow_manager_creates_valid_session_id(self):
        """Test that WorkflowManager creates workflows with valid session_ids."""
        mock_cv_workflow_graph = Mock()
        workflow_manager = WorkflowManager(cv_workflow_graph=mock_cv_workflow_graph)
        
        # Test with no explicit session_id (should auto-generate)
        session_id = workflow_manager.create_new_workflow(
            cv_text="Sample CV text",
            jd_text="Sample job description"
        )
        
        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        
        # Verify the state was saved with the correct session_id
        state = workflow_manager.get_workflow_status(session_id)
        assert state is not None
        assert state.session_id == session_id

    def test_workflow_manager_with_explicit_session_id(self):
        """Test that WorkflowManager works with explicitly provided session_id."""
        import uuid
        
        mock_cv_workflow_graph = Mock()
        workflow_manager = WorkflowManager(cv_workflow_graph=mock_cv_workflow_graph)
        explicit_session_id = f"explicit-test-session-{uuid.uuid4()}"
        
        # Test with explicit session_id
        session_id = workflow_manager.create_new_workflow(
            cv_text="Sample CV text",
            jd_text="Sample job description",
            session_id=explicit_session_id
        )
        
        assert session_id == explicit_session_id
        
        # Verify the state was saved with the correct session_id
        state = workflow_manager.get_workflow_status(session_id)
        assert state is not None
        assert state.session_id == explicit_session_id

    def test_all_agent_input_models_work_with_session_id(self):
        """Test that all agent input models can be created with session_id from AgentState."""
        from src.models.agent_input_models import AGENT_INPUT_MODELS
        from src.models.cv_models import JobDescriptionData
        
        state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=JobDescriptionData(raw_text="Sample job description")
        )
        
        # Test each agent input model
        for agent_name, model_class in AGENT_INPUT_MODELS.items():
            # Skip agents that require special data not available in basic state
            if agent_name in ["CleaningAgent", "JobDescriptionParserAgent"]:
                continue
                
            inputs = extract_agent_inputs(agent_name, state)
            
            # This should not raise a validation error
            try:
                model_instance = model_class(**inputs)
                assert hasattr(model_instance, 'session_id')
                assert model_instance.session_id == state.session_id
            except Exception as e:
                pytest.fail(f"Failed to create {agent_name} input model: {e}")