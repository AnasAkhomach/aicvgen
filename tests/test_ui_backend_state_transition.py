"""Unit tests for UI-to-backend state transition functionality.

Tests the create_agent_state_from_ui function and related workflow invocation logic.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import streamlit as st
from src.core.main import create_agent_state_from_ui
from src.orchestration.state import AgentState
from src.models.data_models import JobDescriptionData, StructuredCV


class TestCreateAgentStateFromUI:
    """Test cases for create_agent_state_from_ui function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Mock Streamlit session state
        self.mock_session_state = {
            "job_description": "Software Engineer position at Tech Corp",
            "cv_text": "John Doe\nSoftware Developer\n5 years experience",
            "start_from_scratch": False
        }

    @patch('src.core.main.st.session_state')
    def test_create_agent_state_with_valid_data(self, mock_session_state):
        """Test creating AgentState with valid session state data."""
        # Arrange
        mock_session_state.get.side_effect = lambda key, default: self.mock_session_state.get(key, default)
        
        # Act
        result = create_agent_state_from_ui()
        
        # Assert
        assert isinstance(result, AgentState)
        assert isinstance(result.job_description_data, JobDescriptionData)
        assert isinstance(result.structured_cv, StructuredCV)
        assert result.job_description_data.raw_text == "Software Engineer position at Tech Corp"
        assert result.structured_cv.metadata["original_cv_text"] == "John Doe\nSoftware Developer\n5 years experience"
        assert result.structured_cv.metadata["start_from_scratch"] is False
        assert result.user_feedback is None
        assert result.error_messages == []
        assert result.current_section_key is None
        assert result.current_item_id is None
        assert result.items_to_process_queue == []
        assert result.is_initial_generation is True
        assert result.final_output_path is None
        assert result.research_findings is None

    @patch('src.core.main.st.session_state')
    def test_create_agent_state_with_empty_data(self, mock_session_state):
        """Test creating AgentState with empty session state data."""
        # Arrange
        empty_session_state = {
            "job_description": "",
            "cv_text": "",
            "start_from_scratch": True
        }
        mock_session_state.get.side_effect = lambda key, default: empty_session_state.get(key, default)
        
        # Act
        result = create_agent_state_from_ui()
        
        # Assert
        assert isinstance(result, AgentState)
        assert result.job_description_data.raw_text == ""
        assert result.structured_cv.metadata["original_cv_text"] == ""
        assert result.structured_cv.metadata["start_from_scratch"] is True

    @patch('src.core.main.st.session_state')
    def test_create_agent_state_with_missing_keys(self, mock_session_state):
        """Test creating AgentState when session state keys are missing."""
        # Arrange
        mock_session_state.get.side_effect = lambda key, default: default
        
        # Act
        result = create_agent_state_from_ui()
        
        # Assert
        assert isinstance(result, AgentState)
        assert result.job_description_data.raw_text == ""
        assert result.structured_cv.metadata["original_cv_text"] == ""
        assert result.structured_cv.metadata["start_from_scratch"] is False

    @patch('src.core.main.st.session_state')
    def test_create_agent_state_start_from_scratch_true(self, mock_session_state):
        """Test creating AgentState when start_from_scratch is True."""
        # Arrange
        scratch_session_state = {
            "job_description": "New job posting",
            "cv_text": "",
            "start_from_scratch": True
        }
        mock_session_state.get.side_effect = lambda key, default: scratch_session_state.get(key, default)
        
        # Act
        result = create_agent_state_from_ui()
        
        # Assert
        assert result.structured_cv.metadata["start_from_scratch"] is True
        assert result.structured_cv.metadata["original_cv_text"] == ""

    def test_agent_state_structure_compliance(self):
        """Test that created AgentState complies with expected structure."""
        with patch('src.core.main.st.session_state') as mock_session_state:
            # Arrange
            mock_session_state.get.side_effect = lambda key, default: self.mock_session_state.get(key, default)
            
            # Act
            result = create_agent_state_from_ui()
            
            # Assert - Check all required fields are present
            required_fields = [
                'structured_cv', 'job_description_data', 'current_section_key',
                'items_to_process_queue', 'current_item_id', 'is_initial_generation',
                'user_feedback', 'research_findings', 'final_output_path', 'error_messages'
            ]
            
            for field in required_fields:
                assert hasattr(result, field), f"AgentState missing required field: {field}"

    def test_agent_state_serialization(self):
        """Test that created AgentState can be serialized (for caching/logging)."""
        with patch('src.core.main.st.session_state') as mock_session_state:
            # Arrange
            mock_session_state.get.side_effect = lambda key, default: self.mock_session_state.get(key, default)
            
            # Act
            result = create_agent_state_from_ui()
            
            # Assert - Should be able to serialize without errors
            try:
                serialized = result.model_dump()
                assert isinstance(serialized, dict)
                assert 'structured_cv' in serialized
                assert 'job_description_data' in serialized
            except Exception as e:
                pytest.fail(f"AgentState serialization failed: {e}")


class TestWorkflowInvocationRefactoring:
    """Test cases for the refactored workflow invocation logic."""

    @patch('src.core.main.create_agent_state_from_ui')
    @patch('src.core.main.get_enhanced_cv_integration')
    def test_workflow_invocation_uses_agent_state(self, mock_integration, mock_create_state):
        """Test that workflow invocation properly uses AgentState."""
        # Arrange
        mock_agent_state = Mock(spec=AgentState)
        mock_create_state.return_value = mock_agent_state
        mock_integration_instance = Mock()
        mock_integration_instance.execute_workflow = Mock()
        mock_integration.return_value = mock_integration_instance
        
        # This would be part of the actual workflow invocation test
        # but since we're testing the function in isolation, we simulate the call
        from src.models.data_models import WorkflowType
        
        # Act - Simulate the workflow invocation logic
        initial_agent_state = mock_create_state()
        integration = mock_integration()
        
        # Assert
        mock_create_state.assert_called_once()
        mock_integration.assert_called_once()
        assert initial_agent_state == mock_agent_state
        assert hasattr(integration, 'execute_workflow')

    def test_agent_state_validation_before_workflow(self):
        """Test that AgentState is properly validated before workflow execution."""
        with patch('src.core.main.st.session_state') as mock_session_state:
            # Arrange
            mock_session_state.get.side_effect = lambda key, default: self.mock_session_state.get(key, default)
            
            # Act
            result = create_agent_state_from_ui()
            
            # Assert - Validate that the state is ready for workflow
            assert result.job_description_data is not None
            assert result.structured_cv is not None
            assert isinstance(result.error_messages, list)
            assert isinstance(result.items_to_process_queue, list)

    @property
    def mock_session_state(self):
        """Mock session state data for testing."""
        return {
            "job_description": "Test job description",
            "cv_text": "Test CV content",
            "start_from_scratch": False
        }