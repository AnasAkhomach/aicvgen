"""Unit tests for core state helpers.

Tests the state management functions for the core module.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
from src.core.state_helpers import (
    create_agent_state_from_ui,
    update_processing_state,
    update_token_usage,
    check_budget_limits,
    add_error_message,
    set_processing_state
)
from src.orchestration.state import AgentState
from src.models.data_models import JobDescriptionData, StructuredCV


class TestCoreStateHelpers:
    """Test cases for core state helper functions."""

    @pytest.fixture
    def mock_streamlit(self):
        """Mock Streamlit session state."""
        with patch('streamlit.session_state') as mock_session_state:
            mock_session_state.get.return_value = None
            mock_session_state.__setitem__ = Mock()
            mock_session_state.__getitem__ = Mock()
            yield mock_session_state

    def test_create_agent_state_from_ui_with_data(self, mock_streamlit):
        """Test creating agent state from UI with job description and CV data."""
        # Mock session state data
        mock_streamlit.get.side_effect = lambda key, default=None: {
            'job_description_input': 'Software Engineer position',
            'cv_text_input': 'John Doe\nSoftware Developer',
            'start_from_scratch_input': False
        }.get(key, default)
        
        with patch('streamlit.session_state', mock_streamlit):
            agent_state = create_agent_state_from_ui()
        
        # Verify agent state creation
        assert isinstance(agent_state, AgentState)
        assert isinstance(agent_state.job_description_data, JobDescriptionData)
        assert isinstance(agent_state.structured_cv, StructuredCV)
        
        # Verify job description data
        assert agent_state.job_description_data.raw_text == 'Software Engineer position'
        
        # Verify structured CV metadata
        assert agent_state.structured_cv.metadata['original_cv_text'] == 'John Doe\nSoftware Developer'
        assert agent_state.structured_cv.metadata['start_from_scratch'] is False

    def test_create_agent_state_from_ui_empty_data(self, mock_streamlit):
        """Test creating agent state from UI with empty data."""
        # Mock empty session state data
        mock_streamlit.get.side_effect = lambda key, default=None: {
            'job_description_input': '',
            'cv_text_input': '',
            'start_from_scratch_input': True
        }.get(key, default)
        
        with patch('streamlit.session_state', mock_streamlit):
            agent_state = create_agent_state_from_ui()
        
        # Verify agent state creation with empty data
        assert isinstance(agent_state, AgentState)
        assert agent_state.job_description_data.raw_text == ''
        assert agent_state.structured_cv.metadata['original_cv_text'] == ''
        assert agent_state.structured_cv.metadata['start_from_scratch'] is True

    def test_create_agent_state_from_ui_missing_keys(self, mock_streamlit):
        """Test creating agent state when session state keys are missing."""
        # Mock session state with missing keys (returns default values)
        mock_streamlit.get.side_effect = lambda key, default=None: default
        
        with patch('streamlit.session_state', mock_streamlit):
            agent_state = create_agent_state_from_ui()
        
        # Verify agent state creation with default values
        assert isinstance(agent_state, AgentState)
        assert agent_state.job_description_data.raw_text == ''
        assert agent_state.structured_cv.metadata['original_cv_text'] == ''
        assert agent_state.structured_cv.metadata['start_from_scratch'] is False

    @patch('streamlit.info')
    def test_update_processing_state_start(self, mock_info, mock_streamlit):
        """Test updating processing state to start."""
        with patch('streamlit.session_state', mock_streamlit):
            update_processing_state(True, "Starting CV generation...")
        
        mock_streamlit.__setitem__.assert_called_with('processing', True)
        mock_info.assert_called_with("Starting CV generation...")

    @patch('streamlit.success')
    def test_update_processing_state_stop(self, mock_success, mock_streamlit):
        """Test updating processing state to stop."""
        with patch('streamlit.session_state', mock_streamlit):
            update_processing_state(False, "CV generation completed!")
        
        mock_streamlit.__setitem__.assert_called_with('processing', False)
        mock_success.assert_called_with("CV generation completed!")

    def test_update_processing_state_no_message(self, mock_streamlit):
        """Test updating processing state without message."""
        with patch('streamlit.session_state', mock_streamlit):
            update_processing_state(True)
        
        mock_streamlit.__setitem__.assert_called_with('processing', True)

    def test_update_token_usage(self, mock_streamlit):
        """Test updating token usage."""
        with patch('streamlit.session_state', mock_streamlit):
            update_token_usage(150)
        
        mock_streamlit.__setitem__.assert_called_with('session_tokens', 150)

    def test_check_budget_limits_under_limit(self, mock_streamlit):
        """Test budget check when under limit."""
        mock_streamlit.get.side_effect = lambda key, default=None: {
            'session_tokens': 5000,
            'session_token_limit': 10000
        }.get(key, default)
        
        with patch('streamlit.session_state', mock_streamlit):
            result = check_budget_limits()
        
        assert result is True

    @patch('streamlit.error')
    def test_check_budget_limits_over_limit(self, mock_error, mock_streamlit):
        """Test budget check when over limit."""
        mock_streamlit.get.side_effect = lambda key, default=None: {
            'session_tokens': 15000,
            'session_token_limit': 10000
        }.get(key, default)
        
        with patch('streamlit.session_state', mock_streamlit):
            result = check_budget_limits()
        
        assert result is False
        mock_error.assert_called_with("Token limit exceeded! Used: 15000, Limit: 10000")

    @patch('streamlit.warning')
    def test_check_budget_limits_near_limit(self, mock_warning, mock_streamlit):
        """Test budget check when near limit (>80%)."""
        mock_streamlit.get.side_effect = lambda key, default=None: {
            'session_tokens': 8500,
            'session_token_limit': 10000
        }.get(key, default)
        
        with patch('streamlit.session_state', mock_streamlit):
            result = check_budget_limits()
        
        assert result is True
        mock_warning.assert_called_with("Warning: 85.0% of token limit used")

    def test_check_budget_limits_missing_data(self, mock_streamlit):
        """Test budget check with missing session data."""
        mock_streamlit.get.return_value = None
        
        with patch('streamlit.session_state', mock_streamlit):
            result = check_budget_limits()
        
        assert result is True  # Should return True when data is missing

    @patch('streamlit.error')
    def test_add_error_message(self, mock_error, mock_streamlit):
        """Test adding error message."""
        error_msg = "Test error message"
        
        with patch('streamlit.session_state', mock_streamlit):
            add_error_message(error_msg)
        
        mock_error.assert_called_with(error_msg)

    def test_set_processing_state_true(self, mock_streamlit):
        """Test setting processing state to True."""
        with patch('streamlit.session_state', mock_streamlit):
            set_processing_state(True)
        
        mock_streamlit.__setitem__.assert_called_with('processing', True)

    def test_set_processing_state_false(self, mock_streamlit):
        """Test setting processing state to False."""
        with patch('streamlit.session_state', mock_streamlit):
            set_processing_state(False)
        
        mock_streamlit.__setitem__.assert_called_with('processing', False)

    def test_create_agent_state_structured_cv_sections(self, mock_streamlit):
        """Test that structured CV is created with empty sections."""
        mock_streamlit.get.side_effect = lambda key, default=None: {
            'job_description_input': 'Test job',
            'cv_text_input': 'Test CV',
            'start_from_scratch_input': False
        }.get(key, default)
        
        with patch('streamlit.session_state', mock_streamlit):
            agent_state = create_agent_state_from_ui()
        
        # Verify structured CV has empty sections initially
        assert isinstance(agent_state.structured_cv.sections, list)
        assert len(agent_state.structured_cv.sections) == 0

    def test_create_agent_state_job_description_fields(self, mock_streamlit):
        """Test that job description data has correct initial fields."""
        mock_streamlit.get.side_effect = lambda key, default=None: {
            'job_description_input': 'Software Engineer at Tech Corp',
            'cv_text_input': 'Test CV',
            'start_from_scratch_input': False
        }.get(key, default)
        
        with patch('streamlit.session_state', mock_streamlit):
            agent_state = create_agent_state_from_ui()
        
        # Verify job description data structure
        job_data = agent_state.job_description_data
        assert job_data.raw_text == 'Software Engineer at Tech Corp'
        assert hasattr(job_data, 'company_name')
        assert hasattr(job_data, 'position_title')
        assert hasattr(job_data, 'requirements')
        assert hasattr(job_data, 'responsibilities')

    def test_token_usage_calculation_percentage(self, mock_streamlit):
        """Test token usage percentage calculation in budget check."""
        mock_streamlit.get.side_effect = lambda key, default=None: {
            'session_tokens': 7500,
            'session_token_limit': 10000
        }.get(key, default)
        
        with patch('streamlit.session_state', mock_streamlit):
            result = check_budget_limits()
        
        assert result is True
        # 7500/10000 = 75%, which is under the 80% warning threshold

    def test_budget_limits_edge_case_exact_limit(self, mock_streamlit):
        """Test budget check when exactly at limit."""
        mock_streamlit.get.side_effect = lambda key, default=None: {
            'session_tokens': 10000,
            'session_token_limit': 10000
        }.get(key, default)
        
        with patch('streamlit.session_state', mock_streamlit), \
             patch('streamlit.error') as mock_error:
            result = check_budget_limits()
        
        assert result is False
        mock_error.assert_called_with("Token limit exceeded! Used: 10000, Limit: 10000")