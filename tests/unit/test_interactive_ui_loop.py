#!/usr/bin/env python3
"""
Unit tests for the Interactive UI Loop implementation in app.py.
Tests the workflow-driven UI functionality and callback handlers.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import streamlit as st
from src.models.workflow_models import UserFeedback, UserAction
from src.orchestration.state import AgentState
from src.core.state_manager import StateManager
# Import will be mocked in tests
from src.config.logging_config import get_logger


class TestInteractiveUILoop:
    """Test suite for Interactive UI Loop functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear streamlit session state
        if hasattr(st, 'session_state'):
            st.session_state.clear()
        
        # Mock container and workflow manager
        self.mock_container = Mock()
        self.mock_workflow_manager = Mock()
        self.mock_container.workflow_manager = Mock(return_value=self.mock_workflow_manager)
        
        # Mock state manager
        self.mock_state_manager = Mock()
        
        # Sample agent state for testing
        self.sample_agent_state = Mock()
        self.sample_agent_state.workflow_status = "AWAITING_FEEDBACK"
        self.sample_agent_state.ui_display_data = {
            "content": "Sample generated content",
            "section_name": "Professional Experience"
        }
        self.sample_agent_state.error_messages = []
        self.sample_agent_state.final_output_path = "/path/to/output.pdf"

    @patch('app.get_container')
    @patch('streamlit.session_state')
    @patch('streamlit.error')
    @patch('asyncio.run')
    def test_on_start_generation_success(self, mock_asyncio_run, mock_st_error, mock_session_state, mock_get_container):
        """Test successful workflow start generation."""
        from app import on_start_generation
        
        # Setup mocks
        mock_get_container.return_value = self.mock_container
        mock_session_state.get.side_effect = lambda key, default="": {
            "cv_text_input": "Sample CV text",
            "job_description_input": "Sample job description"
        }.get(key, default)
        
        self.mock_workflow_manager.create_new_workflow.return_value = "test_session_id"
        self.mock_workflow_manager.get_workflow_status.return_value = self.sample_agent_state
        
        # Execute function
        on_start_generation()
        
        # Verify workflow manager calls
        self.mock_workflow_manager.create_new_workflow.assert_called_once_with(
            "Sample CV text", "Sample job description"
        )
        self.mock_workflow_manager.get_workflow_status.assert_called_once_with("test_session_id")
        mock_asyncio_run.assert_called_once()
        
        # Verify session state update
        assert mock_session_state.session_id == "test_session_id"
        
        # Verify no error was shown
        mock_st_error.assert_not_called()

    @patch('app.get_container')
    @patch('streamlit.session_state')
    @patch('streamlit.error')
    def test_on_start_generation_missing_inputs(self, mock_st_error, mock_session_state, mock_get_container):
        """Test start generation with missing inputs."""
        from app import on_start_generation
        
        # Setup mocks with missing CV text
        mock_session_state.get.side_effect = lambda key, default="": {
            "cv_text_input": "",  # Missing CV text
            "job_description_input": "Sample job description"
        }.get(key, default)
        
        # Execute function
        on_start_generation()
        
        # Verify error was shown
        mock_st_error.assert_called_once_with("Please provide both CV text and job description.")
        
        # Verify workflow manager was not called
        mock_get_container.assert_not_called()

    @patch('app.get_container')
    @patch('streamlit.session_state')
    @patch('streamlit.success')
    @patch('asyncio.run')
    def test_on_approve_success(self, mock_asyncio_run, mock_st_success, mock_session_state, mock_get_container):
        """Test successful approve action."""
        from app import on_approve
        
        # Setup mocks
        mock_get_container.return_value = self.mock_container
        mock_session_state.get.return_value = "test_session_id"
        self.mock_workflow_manager.get_workflow_status.return_value = self.sample_agent_state
        
        # Execute function
        on_approve()
        
        # Verify feedback was sent
        self.mock_workflow_manager.send_feedback.assert_called_once()
        sent_feedback = self.mock_workflow_manager.send_feedback.call_args[0][1]
        assert isinstance(sent_feedback, UserFeedback)
        assert sent_feedback.action == UserAction.APPROVE
        assert sent_feedback.item_id == "current_section"
        
        # Verify workflow step was triggered
        mock_asyncio_run.assert_called_once()
        
        # Verify success message
        mock_st_success.assert_called_once_with("✅ Content approved and workflow resumed.")

    @patch('app.get_container')
    @patch('streamlit.session_state')
    @patch('streamlit.info')
    @patch('asyncio.run')
    def test_on_regenerate_success(self, mock_asyncio_run, mock_st_info, mock_session_state, mock_get_container):
        """Test successful regenerate action."""
        from app import on_regenerate
        
        # Setup mocks
        mock_get_container.return_value = self.mock_container
        mock_session_state.get.return_value = "test_session_id"
        self.mock_workflow_manager.get_workflow_status.return_value = self.sample_agent_state
        
        # Execute function
        on_regenerate()
        
        # Verify feedback was sent
        self.mock_workflow_manager.send_feedback.assert_called_once()
        sent_feedback = self.mock_workflow_manager.send_feedback.call_args[0][1]
        assert isinstance(sent_feedback, UserFeedback)
        assert sent_feedback.action == UserAction.REGENERATE
        assert sent_feedback.item_id == "current_section"
        
        # Verify workflow step was triggered
        mock_asyncio_run.assert_called_once()
        
        # Verify info message
        mock_st_info.assert_called_once_with("🔄 Content regeneration requested and workflow resumed.")

    @patch('streamlit.session_state')
    @patch('streamlit.error')
    def test_on_approve_no_session(self, mock_st_error, mock_session_state):
        """Test approve action with no active session."""
        from app import on_approve
        
        # Setup mock with no session_id
        mock_session_state.get.return_value = None
        
        # Execute function
        on_approve()
        
        # Verify error was shown
        mock_st_error.assert_called_once_with("No active workflow session found.")

    @patch('streamlit.subheader')
    @patch('streamlit.write')
    @patch('streamlit.text_area')
    @patch('streamlit.columns')
    @patch('streamlit.button')
    def test_render_awaiting_feedback_ui(self, mock_button, mock_columns, mock_text_area, mock_write, mock_subheader):
        """Test rendering of awaiting feedback UI."""
        from app import render_awaiting_feedback_ui
        
        # Setup mock columns
        mock_col1, mock_col2 = Mock(), Mock()
        mock_columns.return_value = [mock_col1, mock_col2]
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        
        # Execute function
        render_awaiting_feedback_ui(self.sample_agent_state)
        
        # Verify UI elements were rendered
        mock_subheader.assert_called_once_with("Please Review This Section")
        mock_write.assert_called_once_with("**Section:** Professional Experience")
        mock_text_area.assert_called_once_with(
            "Generated Content", 
            value="Sample generated content", 
            height=300, 
            disabled=True
        )
        
        # Verify buttons were created
        assert mock_button.call_count == 2

    @patch('streamlit.success')
    @patch('streamlit.download_button')
    @patch('builtins.open', new_callable=mock_open, read_data=b'PDF content')
    def test_render_completed_ui_with_file(self, mock_file_open, mock_download_button, mock_success):
        """Test rendering of completed UI with available output file."""
        from app import render_completed_ui
        
        # Execute function
        render_completed_ui(self.sample_agent_state)
        
        # Verify success message
        mock_success.assert_called_once_with("🎉 CV Generation Complete!")
        
        # Verify file was opened and download button created
        mock_file_open.assert_called_once_with("/path/to/output.pdf", 'rb')
        mock_download_button.assert_called_once_with(
            label="📄 Download Generated CV",
            data=b'PDF content',
            file_name="generated_cv.pdf",
            mime="application/pdf"
        )

    @patch('streamlit.success')
    @patch('streamlit.info')
    def test_render_completed_ui_no_file(self, mock_info, mock_success):
        """Test rendering of completed UI with no output file."""
        from app import render_completed_ui
        
        # Setup state without final_output_path
        state_no_file = Mock()
        state_no_file.final_output_path = None
        
        # Execute function
        render_completed_ui(state_no_file)
        
        # Verify success message and info about missing file
        mock_success.assert_called_once_with("🎉 CV Generation Complete!")
        mock_info.assert_called_once_with("CV generation completed but download file is not available.")

    @patch('streamlit.error')
    @patch('streamlit.button')
    def test_render_error_ui(self, mock_button, mock_error):
        """Test rendering of error UI."""
        from app import render_error_ui
        
        # Setup state with error messages
        error_state = Mock()
        error_state.error_messages = ["First error", "Second error"]
        
        # Execute function
        render_error_ui(error_state)
        
        # Verify error message shows latest error
        mock_error.assert_called_once_with("❌ An error occurred: Second error")
        
        # Verify restart button is shown
        mock_button.assert_called_once_with("🔄 Start New Generation")

    @patch('src.error_handling.boundaries.safe_streamlit_component', lambda **kwargs: lambda f: f)
    @patch('app.get_container')
    @patch('app.StateManager')
    @patch('app.display_sidebar')
    @patch('app.display_input_form')
    @patch('streamlit.session_state')
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    @patch('streamlit.header')
    @patch('streamlit.button')
    def test_main_no_session_id(self, mock_button, mock_header, mock_markdown, mock_title, 
                               mock_session_state, mock_display_input_form, mock_display_sidebar, 
                               mock_state_manager_class, mock_get_container):
        """Test main function with no active session."""
        from app import main
        
        # Setup mocks
        mock_get_container.return_value = self.mock_container
        mock_session_state.get.return_value = None  # No session_id
        mock_state_manager = Mock()
        mock_state_manager_class.return_value = mock_state_manager
        
        # Execute function
        main()
        
        # Verify initialization
        mock_state_manager.initialize_session_state.assert_called_once()
        
        # Verify UI elements for no session
        mock_title.assert_called_once_with("🤖 AI CV Generator")
        mock_markdown.assert_called_once_with("Generate tailored CVs using AI-powered analysis")
        mock_display_sidebar.assert_called_once()
        mock_header.assert_called_once_with("📝 Input & Generate")
        mock_display_input_form.assert_called_once()
        mock_button.assert_called_once()

    @patch('src.error_handling.boundaries.safe_streamlit_component', lambda **kwargs: lambda f: f)
    @patch('app.get_container')
    @patch('app.StateManager')
    @patch('app.render_awaiting_feedback_ui')
    @patch('app.display_sidebar')
    @patch('streamlit.session_state')
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    def test_main_with_awaiting_feedback_status(self, mock_markdown, mock_title, mock_session_state, mock_display_sidebar, mock_render_awaiting_feedback, 
                                               mock_state_manager_class, mock_get_container):
        """Test main function with AWAITING_FEEDBACK status."""
        from app import main
        
        # Setup mocks
        mock_get_container.return_value = self.mock_container
        mock_session_state.get.return_value = "test_session_id"
        self.mock_workflow_manager.get_workflow_status.return_value = self.sample_agent_state
        mock_state_manager = Mock()
        mock_state_manager_class.return_value = mock_state_manager
        
        # Execute function
        main()
        
        # Verify workflow status was checked
        self.mock_workflow_manager.get_workflow_status.assert_called_once_with("test_session_id")
        
        # Verify awaiting feedback UI was rendered
        mock_render_awaiting_feedback.assert_called_once_with(self.sample_agent_state)

    @patch('src.error_handling.boundaries.safe_streamlit_component', lambda **kwargs: lambda f: f)
    @patch('app.get_container')
    @patch('app.StateManager')
    @patch('app.display_sidebar')
    @patch('streamlit.session_state')
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    @patch('streamlit.info')
    @patch('streamlit.spinner')
    @patch('time.sleep')
    @patch('streamlit.rerun')
    def test_main_with_processing_status(self, mock_rerun, mock_sleep, mock_spinner, mock_info, mock_markdown, mock_title,
                                        mock_session_state, mock_display_sidebar, mock_state_manager_class, mock_get_container):
        """Test main function with PROCESSING status."""
        from app import main
        
        # Setup mocks
        mock_get_container.return_value = self.mock_container
        mock_session_state.get.return_value = "test_session_id"
        mock_state_manager = Mock()
        mock_state_manager_class.return_value = mock_state_manager
        
        processing_state = Mock()
        processing_state.workflow_status = "PROCESSING"
        self.mock_workflow_manager.get_workflow_status.return_value = processing_state
        
        # Mock spinner context manager
        mock_spinner_context = Mock()
        mock_spinner_context.__enter__ = Mock(return_value=mock_spinner_context)
        mock_spinner_context.__exit__ = Mock(return_value=None)
        mock_spinner.return_value = mock_spinner_context
        
        # Execute function
        main()
        
        # Verify processing UI elements
        mock_info.assert_called_once_with("⏳ Processing your request...")
        mock_spinner.assert_called_once_with("Generating CV content...")
        mock_sleep.assert_called_once_with(2)
        mock_rerun.assert_called_once()