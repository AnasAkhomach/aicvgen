"""Tests for agent input mapping functionality (AD-002 fix).

This module tests the explicit input mapping implementation that reduces
coupling between agents and the global AgentState.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import Any

from src.models.agent_input_models import (
    extract_agent_inputs,
    get_agent_input_model,
    CVAnalyzerAgentInput,
    ExecutiveSummaryWriterAgentInput,
    AGENT_INPUT_MODELS,
)
from src.orchestration.state import AgentState
from src.models.cv_models import StructuredCV, JobDescriptionData
from src.agents.agent_base import AgentBase
from src.models.agent_models import AgentResult
from src.error_handling.exceptions import AgentExecutionError


class TestAgentInputModels:
    """Test agent-specific input models."""

    def test_get_agent_input_model_existing_agent(self):
        """Test getting input model for existing agent."""
        model_class = get_agent_input_model("CVAnalyzerAgent")
        assert model_class == CVAnalyzerAgentInput

    def test_get_agent_input_model_nonexistent_agent(self):
        """Test getting input model for non-existent agent."""
        model_class = get_agent_input_model("NonExistentAgent")
        assert model_class is None

    def test_cv_analyzer_input_model_validation(self):
        """Test CVAnalyzerAgentInput validation."""
        # Create mock data
        structured_cv = Mock(spec=StructuredCV)
        job_description = Mock(spec=JobDescriptionData)
        
        # Valid input
        valid_input = CVAnalyzerAgentInput(
            cv_data=structured_cv,
            job_description=job_description,
            session_id="test-session"
        )
        
        assert valid_input.cv_data == structured_cv
        assert valid_input.job_description == job_description
        assert valid_input.session_id == "test-session"

    def test_cv_analyzer_input_model_validation_missing_fields(self):
        """Test CVAnalyzerAgentInput validation with missing fields."""
        with pytest.raises(Exception):  # Pydantic validation error
            CVAnalyzerAgentInput(
                cv_data=Mock(spec=StructuredCV),
                # Missing job_description and session_id
            )

    def test_executive_summary_input_model_validation(self):
        """Test ExecutiveSummaryWriterAgentInput validation."""
        structured_cv = Mock(spec=StructuredCV)
        job_description_data = Mock(spec=JobDescriptionData)
        
        valid_input = ExecutiveSummaryWriterAgentInput(
            structured_cv=structured_cv,
            job_description_data=job_description_data,
            session_id="test-session"
        )
        
        assert valid_input.structured_cv == structured_cv
        assert valid_input.job_description_data == job_description_data
        assert valid_input.session_id == "test-session"
        assert valid_input.research_findings is None  # Optional field


class TestExtractAgentInputs:
    """Test the extract_agent_inputs function."""

    def create_mock_agent_state(self) -> AgentState:
        """Create a mock AgentState for testing."""
        structured_cv = Mock(spec=StructuredCV)
        job_description_data = Mock(spec=JobDescriptionData)
        
        # Create a minimal AgentState with required fields
        state = Mock(spec=AgentState)
        state.session_id = "test-session"
        state.structured_cv = structured_cv
        state.job_description_data = job_description_data
        state.cv_text = "Sample CV text"
        state.current_item_id = "item-123"
        
        # Add getattr support for optional fields
        def mock_getattr(name, default=None):
            if hasattr(state, name):
                return getattr(state, name)
            return default
        
        # Mock getattr to handle optional fields
        state.__class__.__getattr__ = lambda self, name: mock_getattr(name)
        
        state.model_dump.return_value = {
            "session_id": "test-session",
            "structured_cv": structured_cv,
            "job_description_data": job_description_data,
            "cv_text": "Sample CV text",
            "current_item_id": "item-123",
        }
        
        return state

    def test_extract_cv_analyzer_inputs(self):
        """Test extracting inputs for CVAnalyzerAgent."""
        state = self.create_mock_agent_state()
        
        inputs = extract_agent_inputs("CVAnalyzerAgent", state)
        
        # Verify correct fields are extracted
        assert "session_id" in inputs
        assert "cv_data" in inputs
        assert "job_description" in inputs
        assert inputs["session_id"] == "test-session"
        
        # Mock objects serialize to empty dicts, which is expected
        assert isinstance(inputs["cv_data"], dict)
        assert isinstance(inputs["job_description"], dict)

    def test_extract_executive_summary_inputs(self):
        """Test extracting inputs for ExecutiveSummaryWriterAgent."""
        state = self.create_mock_agent_state()
        
        inputs = extract_agent_inputs("ExecutiveSummaryWriter", state)
        
        # Verify correct fields are extracted
        assert "session_id" in inputs
        assert "structured_cv" in inputs
        assert "job_description_data" in inputs
        assert "research_findings" in inputs
        assert inputs["session_id"] == "test-session"
        
        # Mock objects serialize to empty dicts, which is expected
        assert isinstance(inputs["structured_cv"], dict)
        assert isinstance(inputs["job_description_data"], dict)
        assert inputs["research_findings"] is None  # Not set in mock state

    def test_extract_inputs_nonexistent_agent(self):
        """Test extracting inputs for non-existent agent raises error."""
        state = self.create_mock_agent_state()
        
        with pytest.raises(ValueError, match="No input model found for agent"):
            extract_agent_inputs("NonExistentAgent", state)

    def test_extract_inputs_validation_failure(self):
        """Test input extraction with validation failure."""
        # Create state missing required fields
        state = Mock(spec=AgentState)
        state.session_id = None  # Missing required field
        state.structured_cv = None  # Missing required field
        state.job_description_data = None  # Missing required field
        
        # Add getattr support
        def mock_getattr(name, default=None):
            if hasattr(state, name):
                return getattr(state, name)
            return default
        state.__class__.__getattr__ = lambda self, name: mock_getattr(name)
        
        state.model_dump.return_value = {
            "session_id": None,
            "structured_cv": None,
            "job_description_data": None
        }
        
        with pytest.raises(ValueError, match="Input validation failed"):
            extract_agent_inputs("CVAnalyzerAgent", state)


class MockAgent(AgentBase):
    """Mock agent for testing AgentBase functionality."""
    
    def __init__(self, name: str = "MockAgent"):
        super().__init__(name=name, description="Mock agent for testing", session_id="test-session")
        self.execute_called = False
        self.validate_called = False
        self.received_kwargs = {}
    
    def _validate_inputs(self, input_data: dict) -> None:
        """Mock validation."""
        self.validate_called = True
        if "invalid" in input_data:
            raise AgentExecutionError("Mock validation error")
    
    async def _execute(self, **kwargs: Any) -> AgentResult:
        """Mock execution."""
        self.execute_called = True
        self.received_kwargs = kwargs
        return AgentResult(
            success=True,
            output_data=Mock(),
            agent_name=self.name
        )


class TestAgentBaseInputMapping:
    """Test AgentBase with explicit input mapping."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        return MockAgent()

    @pytest.fixture
    def mock_state(self):
        """Create a mock AgentState."""
        structured_cv = Mock(spec=StructuredCV)
        job_description_data = Mock(spec=JobDescriptionData)
        
        state = Mock(spec=AgentState)
        state.session_id = "test-session"
        state.structured_cv = structured_cv
        state.job_description_data = job_description_data
        state.cv_text = "Sample CV text"
        state.current_item_id = "item-123"
        state.error_messages = []
        state.model_copy.return_value = state
        state.model_dump.return_value = {
            "session_id": "test-session",
            "structured_cv": structured_cv,
            "job_description_data": job_description_data,
            "cv_text": "Sample CV text",
            "current_item_id": "item-123",
            "error_messages": [],
        }
        
        return state

    @pytest.mark.asyncio
    async def test_run_as_node_with_valid_inputs(self, mock_agent, mock_state):
        """Test run_as_node with valid inputs using explicit mapping."""
        # Mock the agent to be in the registry
        original_models = AGENT_INPUT_MODELS.copy()
        try:
            # Add mock agent to registry for testing
            AGENT_INPUT_MODELS["MockAgent"] = CVAnalyzerAgentInput
            
            result_state = await mock_agent.run_as_node(mock_state)
            
            # Verify agent was called with extracted inputs, not full state
            assert mock_agent.execute_called
            assert "session_id" in mock_agent.received_kwargs
            assert "cv_data" in mock_agent.received_kwargs
            assert "job_description" in mock_agent.received_kwargs
            
            # Verify session_id is correctly passed
            assert mock_agent.received_kwargs["session_id"] == "test-session"
            
            # Should not receive the entire state dump
            assert "error_messages" not in mock_agent.received_kwargs
            assert "current_item_id" not in mock_agent.received_kwargs  # Not in CVAnalyzerAgentInput
            
        finally:
            # Restore original registry
            AGENT_INPUT_MODELS.clear()
            AGENT_INPUT_MODELS.update(original_models)

    @pytest.mark.asyncio
    async def test_run_as_node_input_extraction_failure(self, mock_agent, mock_state):
        """Test run_as_node when input extraction fails."""
        # Agent not in registry, should cause extraction failure
        result_state = await mock_agent.run_as_node(mock_state)
        
        # Agent should not have been executed
        assert not mock_agent.execute_called
        
        # Error should be added to state
        mock_state.model_copy.assert_called_once()
        call_args = mock_state.model_copy.call_args[1]['update']
        assert 'error_messages' in call_args
        assert any('Input extraction failed' in msg for msg in call_args['error_messages'])

    @pytest.mark.asyncio
    async def test_run_as_node_session_id_update(self, mock_agent, mock_state):
        """Test that run_as_node updates agent session_id from state."""
        original_session_id = mock_agent.session_id
        mock_state.session_id = "new-session-id"
        
        # Mock the agent to be in the registry
        original_models = AGENT_INPUT_MODELS.copy()
        try:
            AGENT_INPUT_MODELS["MockAgent"] = CVAnalyzerAgentInput
            
            await mock_agent.run_as_node(mock_state)
            
            # Session ID should be updated
            assert mock_agent.session_id == "new-session-id"
            assert mock_agent.session_id != original_session_id
            
        finally:
            AGENT_INPUT_MODELS.clear()
            AGENT_INPUT_MODELS.update(original_models)


class TestArchitecturalImprovements:
    """Test that the architectural improvements are working as expected."""

    def test_agent_input_models_registry_completeness(self):
        """Test that all expected agents have input models defined."""
        expected_agents = [
            "CVAnalyzerAgent",
            "ExecutiveSummaryWriter",
            "ProfessionalExperienceWriter",
            "KeyQualificationsWriter",
            "ProjectsWriter",
            "ResearchAgent",
            "FormatterAgent",
            "QualityAssuranceAgent",
            "CleaningAgent",
            "UserCVParserAgent",
            "JobDescriptionParserAgent",
        ]
        
        for agent_name in expected_agents:
            assert agent_name in AGENT_INPUT_MODELS, f"Missing input model for {agent_name}"
            assert AGENT_INPUT_MODELS[agent_name] is not None

    def test_input_models_reduce_coupling(self):
        """Test that input models only expose necessary fields."""
        # CVAnalyzerAgent should only need cv_data, job_description, and session_id
        cv_analyzer_fields = set(CVAnalyzerAgentInput.model_fields.keys())
        expected_fields = {"cv_data", "job_description", "session_id"}
        assert cv_analyzer_fields == expected_fields
        
        # ExecutiveSummaryWriter should have more fields but still be limited
        exec_summary_fields = set(ExecutiveSummaryWriterAgentInput.model_fields.keys())
        expected_exec_fields = {"structured_cv", "job_description_data", "research_findings", "session_id"}
        assert exec_summary_fields == expected_exec_fields

    def test_input_extraction_does_not_pass_full_state(self):
        """Test that input extraction doesn't pass unnecessary state fields."""
        # Create a state with many fields
        state = Mock(spec=AgentState)
        state.session_id = "test-session"
        state.structured_cv = Mock(spec=StructuredCV)
        state.job_description_data = Mock(spec=JobDescriptionData)
        state.cv_text = "Sample CV text"
        state.current_item_id = "item-123"
        state.error_messages = ["some error"]
        state.trace_id = "trace-123"
        state.current_section_key = "experience"
        state.model_dump.return_value = {
            "session_id": "test-session",
            "structured_cv": state.structured_cv,
            "job_description_data": state.job_description_data,
            "cv_text": "Sample CV text",
            "current_item_id": "item-123",
            "error_messages": ["some error"],
            "trace_id": "trace-123",
            "current_section_key": "experience",
        }
        
        # Extract inputs for CVAnalyzerAgent
        inputs = extract_agent_inputs("CVAnalyzerAgent", state)
        
        # Should only contain fields needed by CVAnalyzerAgent
        expected_keys = {"session_id", "cv_data", "job_description"}
        assert set(inputs.keys()) == expected_keys
        
        # Should not contain unnecessary state fields
        assert "error_messages" not in inputs
        assert "trace_id" not in inputs
        assert "current_section_key" not in inputs
        assert "current_item_id" not in inputs  # Not needed by CVAnalyzerAgent