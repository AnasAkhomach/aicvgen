"""Test for session_id validation fix.

This test verifies that the Pydantic validation error where session_id is None
in AgentState has been resolved.
"""

from unittest.mock import Mock, patch

import pytest

from src.agents.agent_base import AgentBase
from src.core.workflow_manager import WorkflowManager
from src.models.agent_input_models import ResearchAgentInput, extract_agent_inputs
from src.models.cv_models import StructuredCV
from src.orchestration.state import AgentState


class TestSessionIdFix:
    """Test cases for session_id validation fix."""

    def test_agent_state_has_session_id_by_default(self):
        """Test that AgentState generates a session_id by default."""
        from src.orchestration.state import create_global_state

        state = create_global_state(cv_text="Sample CV text")

        assert state["session_id"] is not None
        assert isinstance(state["session_id"], str)
        assert len(state["session_id"]) > 0

    def test_agent_state_accepts_explicit_session_id(self):
        """Test that AgentState accepts an explicitly provided session_id."""
        from src.orchestration.state import create_global_state

        explicit_session_id = "test-session-123"
        state = create_global_state(
            cv_text="Sample CV text",
            session_id=explicit_session_id,
            structured_cv=StructuredCV(),
        )

        assert state["session_id"] == explicit_session_id

    def test_extract_agent_inputs_with_session_id(self):
        """Test that extract_agent_inputs works with session_id from AgentState."""
        from src.models.cv_models import JobDescriptionData
        from src.orchestration.state import create_global_state

        state = create_global_state(
            cv_text="Sample CV text",
            structured_cv=StructuredCV(),
            job_description_data=JobDescriptionData(raw_text="Sample job description"),
        )

        # Ensure session_id is generated
        assert state["session_id"] is not None

        # Test extracting ResearchAgentInput
        inputs = extract_agent_inputs("ResearchAgent", state)

        assert "session_id" in inputs
        assert inputs["session_id"] == state["session_id"]

        # Test that ResearchAgentInput can be created with these inputs
        research_input = ResearchAgentInput(**inputs)
        assert research_input.session_id == state["session_id"]

    def test_agent_base_initialization_with_session_id(self):
        """Test that AgentBase can initialize with a valid session_id from AgentState."""
        from src.models.agent_models import AgentResult
        from src.orchestration.state import create_global_state

        state = create_global_state(
            cv_text="Sample CV text", structured_cv=StructuredCV()
        )

        # Create a mock agent that inherits from AgentBase
        class MockAgent(AgentBase):
            async def _execute(self, **kwargs):
                return AgentResult.create_success(
                    agent_name="MockAgent", output_data={"result": "success"}
                )

        # This should not raise a validation error
        agent = MockAgent(
            session_id=state["session_id"], name="MockAgent", description="Test agent"
        )
        assert agent.session_id == state["session_id"]

    def test_workflow_manager_creates_valid_session_id(self):
        """Test that WorkflowManager creates workflows with valid session_ids."""
        mock_container = Mock()
        mock_cv_template_loader_service = Mock()
        mock_container.cv_template_loader_service.return_value = (
            mock_cv_template_loader_service
        )

        # Mock the template loader to return a basic StructuredCV
        mock_cv_template_loader_service.load_from_markdown.return_value = StructuredCV()

        workflow_manager = WorkflowManager(container=mock_container)

        # Test with no explicit session_id (should auto-generate)
        session_id = workflow_manager.create_new_workflow(
            cv_text="Sample CV text", jd_text="Sample job description"
        )

        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) > 0

        # Verify the state was saved with the correct session_id
        state = workflow_manager.get_workflow_status(session_id)
        assert state is not None
        assert state["session_id"] == session_id

    def test_workflow_manager_with_explicit_session_id(self):
        """Test that WorkflowManager works with explicitly provided session_id."""
        import uuid

        mock_container = Mock()
        mock_cv_template_loader_service = Mock()
        mock_container.cv_template_loader_service.return_value = (
            mock_cv_template_loader_service
        )

        # Mock the template loader to return a basic StructuredCV
        mock_cv_template_loader_service.load_from_markdown.return_value = StructuredCV()

        workflow_manager = WorkflowManager(container=mock_container)
        explicit_session_id = f"explicit-test-session-{uuid.uuid4()}"

        # Test with explicit session_id
        session_id = workflow_manager.create_new_workflow(
            cv_text="Sample CV text",
            jd_text="Sample job description",
            session_id=explicit_session_id,
        )

        assert session_id == explicit_session_id

        # Verify the state was saved with the correct session_id
        state = workflow_manager.get_workflow_status(session_id)
        assert state is not None
        assert state["session_id"] == explicit_session_id

    def test_all_agent_input_models_work_with_session_id(self):
        """Test that all agent input models can be created with session_id from AgentState."""
        from src.models.agent_input_models import AGENT_INPUT_MODELS
        from src.models.cv_models import Item, JobDescriptionData, Section
        from src.orchestration.state import create_global_state

        # Create a StructuredCV with the required sections for ExecutiveSummaryWriter
        structured_cv = StructuredCV()
        structured_cv.sections = [
            Section(
                name="Key Qualifications",
                items=[
                    Item(content="Python programming"),
                    Item(content="Machine learning"),
                ],
            ),
            Section(
                name="Professional Experience",
                items=[
                    Item(content="Senior Developer at TechCorp"),
                    Item(content="Led team of 5 developers"),
                ],
            ),
            Section(
                name="Project Experience",
                items=[
                    Item(content="Built AI-powered CV generator"),
                    Item(content="Developed REST API"),
                ],
            ),
        ]

        state = create_global_state(
            cv_text="Sample CV text",
            structured_cv=structured_cv,
            job_description_data=JobDescriptionData(raw_text="Sample job description"),
            # Add generated fields that updater agents expect
            generated_key_qualifications=["Python programming", "Machine learning"],
            generated_professional_experience="Senior Developer with 5+ years experience",
            generated_projects="AI-powered applications and REST APIs",
            generated_executive_summary="Experienced developer with strong technical skills",
            raw_data="sample raw data",
            data_type="cv_text",
            job_description_text="Sample job description text",
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

                # Check that session_id is present and correct only if the model has this field
                if (
                    hasattr(model_class, "model_fields")
                    and "session_id" in model_class.model_fields
                ):
                    assert hasattr(
                        model_instance, "session_id"
                    ), f"Model {agent_name} should have session_id attribute"
                    assert (
                        model_instance.session_id == state["session_id"]
                    ), f"Model {agent_name} session_id should be {state['session_id']}"
                else:
                    print(
                        f"Model {agent_name} does not have session_id field, skipping session_id check"
                    )

            except Exception as e:
                print(f"\nFailed to create {agent_name} input model:")
                print(f"Error: {e}")
                print(f"Inputs: {inputs}")
                pytest.fail(f"Failed to create {agent_name} input model: {e}")
