"""Unit tests for frontend UI components.

Tests the new frontend UI components that replace the legacy core UI components.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
from src.frontend.ui_components import (
    display_sidebar,
    display_input_form,
    display_review_and_edit_tab,
    display_export_tab,
    _display_reviewable_item
)
from src.orchestration.state import AgentState, UserFeedback
from src.models.data_models import StructuredCV, Section, Item, ItemStatus, JobDescriptionData


class TestFrontendUIComponents:
    """Test cases for frontend UI components."""

    @pytest.fixture
    def mock_streamlit(self):
        """Mock Streamlit session state and components."""
        with patch('streamlit.session_state') as mock_session_state:
            # Set up default session state
            mock_session_state.get.return_value = None
            mock_session_state.user_gemini_api_key = 'test_api_key'
            mock_session_state.api_key_validated = True
            mock_session_state.session_tokens_used = 100
            mock_session_state.session_token_limit = 1000
            mock_session_state.processing = False
            yield mock_session_state

    @pytest.fixture
    def sample_agent_state(self):
        """Create a sample agent state for testing."""
        job_data = JobDescriptionData(
            raw_text="Test job description",
            title="Software Engineer",
            company="Test Company",
            requirements=["Python", "React"],
            skills_required=["Programming", "Problem Solving"]
        )
        
        # Create sample CV structure
        item1 = Item(
            id="item1",
            content="Test experience item",
            status=ItemStatus.PENDING,
            raw_llm_output="Raw LLM output for item 1"
        )
        
        section1 = Section(
            name="Professional Experience",
            items=[item1]
        )
        
        structured_cv = StructuredCV(
            sections=[section1],
            metadata={"test": "data"}
        )
        
        return AgentState(
            job_description_data=job_data,
            structured_cv=structured_cv,
            current_step="review",
            processing_status="completed",
            final_output_path="/path/to/output.pdf"
        )

    @patch('streamlit.sidebar')
    @patch('streamlit.title')
    @patch('streamlit.subheader')
    @patch('streamlit.text_input')
    @patch('streamlit.success')
    @patch('streamlit.divider')
    def test_display_sidebar_with_valid_api_key(self, mock_divider, mock_success, 
                                               mock_text_input, mock_subheader, 
                                               mock_title, mock_sidebar, 
                                               mock_streamlit, sample_agent_state):
        """Test sidebar display with valid API key."""
        mock_text_input.return_value = 'test_api_key'
        
        with patch('streamlit.session_state', mock_streamlit):
            display_sidebar(sample_agent_state)
        
        mock_sidebar.assert_called()
        mock_title.assert_called_with("🔧 Session Management")
        mock_success.assert_called_with("✅ API Key validated and ready to use!")

    @patch('streamlit.header')
    @patch('streamlit.text_area')
    @patch('streamlit.checkbox')
    @patch('streamlit.button')
    def test_display_input_form_valid_inputs(self, mock_button, mock_checkbox, 
                                           mock_text_area, mock_header, 
                                           mock_streamlit):
        """Test input form with valid inputs."""
        mock_streamlit.get.side_effect = lambda key, default=None: {
            'user_gemini_api_key': 'test_key',
            'processing': False
        }.get(key, default)
        
        mock_text_area.side_effect = ['Test job description', 'Test CV content']
        mock_checkbox.return_value = False
        mock_button.return_value = True
        
        with patch('streamlit.session_state', mock_streamlit), \
             patch('streamlit.rerun') as mock_rerun:
            display_input_form(None)
        
        mock_header.assert_called_with("1. Input Your Information")
        assert mock_text_area.call_count == 2
        mock_checkbox.assert_called_once()
        mock_button.assert_called_once()
        mock_rerun.assert_called_once()

    @patch('streamlit.header')
    @patch('streamlit.warning')
    def test_display_input_form_no_api_key(self, mock_warning, mock_header, mock_streamlit):
        """Test input form without API key."""
        mock_streamlit.get.return_value = None
        
        with patch('streamlit.session_state', mock_streamlit):
            display_input_form(None)
        
        mock_header.assert_called_with("1. Input Your Information")
        mock_warning.assert_called_with("⚠️ Please enter your Gemini API key in the sidebar before proceeding.")

    @patch('streamlit.expander')
    @patch('streamlit.markdown')
    def test_display_review_and_edit_tab_with_cv(self, mock_markdown, mock_expander, 
                                               sample_agent_state):
        """Test review and edit tab with CV data."""
        mock_expander_context = MagicMock()
        mock_expander.return_value.__enter__.return_value = mock_expander_context
        
        with patch('src.frontend.ui_components._display_reviewable_item') as mock_display_item:
            display_review_and_edit_tab(sample_agent_state)
        
        mock_expander.assert_called_once()
        mock_display_item.assert_called_once()

    @patch('streamlit.info')
    def test_display_review_and_edit_tab_no_cv(self, mock_info):
        """Test review and edit tab without CV data."""
        display_review_and_edit_tab(None)
        mock_info.assert_called_with("Please generate a CV first to review it here.")

    @patch('streamlit.success')
    @patch('streamlit.download_button')
    @patch('builtins.open', create=True)
    def test_display_export_tab_with_output(self, mock_open, mock_download, 
                                          mock_success, sample_agent_state):
        """Test export tab with generated output."""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        display_export_tab(sample_agent_state)
        
        mock_success.assert_called_with("✅ Your CV has been generated!")
        mock_download.assert_called_once()

    @patch('streamlit.info')
    def test_display_export_tab_no_output(self, mock_info):
        """Test export tab without generated output."""
        state = AgentState()
        display_export_tab(state)
        mock_info.assert_called_with("Generate and finalize your CV to enable export options.")

    @patch('streamlit.markdown')
    @patch('streamlit.expander')
    @patch('streamlit.code')
    @patch('streamlit.columns')
    @patch('streamlit.button')
    def test_display_reviewable_item(self, mock_button, mock_columns, mock_code, 
                                   mock_expander, mock_markdown, sample_agent_state):
        """Test display of reviewable item."""
        item = sample_agent_state.structured_cv.sections[0].items[0]
        
        # Mock columns
        mock_col1, mock_col2 = MagicMock(), MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2, MagicMock()]
        
        # Mock expander context
        mock_expander_context = MagicMock()
        mock_expander.return_value.__enter__.return_value = mock_expander_context
        
        with patch('src.frontend.callbacks.handle_user_action') as mock_handle_action:
            _display_reviewable_item(item, sample_agent_state)
        
        mock_markdown.assert_called_with(f"> {item.content}")
        mock_expander.assert_called_once()
        mock_code.assert_called_with(item.raw_llm_output, language="text")
        assert mock_button.call_count == 2  # Accept and Regenerate buttons

    @patch('streamlit.text_area')
    @patch('streamlit.checkbox')
    @patch('streamlit.button')
    @patch('streamlit.info')
    def test_display_input_form_validation_messages(self, mock_info, mock_button, 
                                                  mock_checkbox, mock_text_area, 
                                                  mock_streamlit):
        """Test input form validation messages."""
        mock_streamlit.get.side_effect = lambda key, default=None: {
            'user_gemini_api_key': 'test_key',
            'processing': False
        }.get(key, default)
        
        # Test with empty job description
        mock_text_area.side_effect = ['', 'Test CV content']
        mock_checkbox.return_value = False
        mock_button.return_value = False
        
        with patch('streamlit.session_state', mock_streamlit):
            display_input_form(None)
        
        mock_info.assert_called_with("💡 Please provide a job description to get started.")
        
        # Test with empty CV and no start from scratch
        mock_text_area.side_effect = ['Test job description', '']
        mock_checkbox.return_value = False
        
        with patch('streamlit.session_state', mock_streamlit):
            display_input_form(None)
        
        mock_info.assert_called_with("💡 Please provide your CV content or check 'Start from scratch'.")

    def test_display_input_form_button_disabled_when_processing(self, mock_streamlit):
        """Test that generate button is disabled when processing."""
        mock_streamlit.get.side_effect = lambda key, default=None: {
            'user_gemini_api_key': 'test_key',
            'processing': True
        }.get(key, default)
        
        with patch('streamlit.session_state', mock_streamlit), \
             patch('streamlit.text_area') as mock_text_area, \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.button') as mock_button:
            
            mock_text_area.side_effect = ['Test job description', 'Test CV content']
            mock_checkbox.return_value = False
            
            display_input_form(None)
            
            # Check that button was called with disabled=True
            button_call_args = mock_button.call_args
            assert button_call_args[1]['disabled'] is True