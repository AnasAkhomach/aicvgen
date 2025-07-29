"""Unit tests for UIManager class.

These tests verify the UIManager's UI rendering logic by mocking the Streamlit
library and asserting that the appropriate Streamlit functions are called.
"""

from unittest.mock import Mock, call, patch

from src.core.state_manager import StateManager
from src.ui.ui_manager import UIManager


class TestUIManager:
    """Test suite for UIManager class."""

    @patch("src.ui.ui_manager.st")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_configure_page_sets_correct_config(self, _, mock_st):
        """Test that page configuration is set correctly on initialization."""
        # Arrange
        state_manager = StateManager()

        # Act
        UIManager(state_manager)

        # Assert
        mock_st.set_page_config.assert_called_once_with(
            page_title="🤖 AI CV Generator",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    @patch("src.ui.ui_manager.st")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_render_header_displays_title_and_description(self, _, mock_st):
        """Test that header renders title and description correctly."""
        # Arrange
        state_manager = StateManager()
        ui_manager = UIManager(state_manager)

        # Act
        ui_manager.render_header()

        # Assert
        mock_st.title.assert_called_once_with("🤖 AI CV Generator")
        mock_st.markdown.assert_called_once()
        # Verify the markdown content contains expected text
        markdown_call = mock_st.markdown.call_args[0][0]
        assert "Transform your CV" in markdown_call
        assert "ATS-friendly" in markdown_call

    @patch("src.ui.ui_manager.display_sidebar")
    @patch("src.ui.ui_manager.st")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_render_sidebar_calls_display_function(self, _, __, mock_display_sidebar):
        """Test that render_sidebar calls the display_sidebar function."""
        # Arrange
        state_manager = StateManager()
        ui_manager = UIManager(state_manager)

        # Act
        ui_manager.render_sidebar()

        # Assert
        mock_display_sidebar.assert_called_once()

    @patch("src.ui.ui_manager.display_sidebar")
    @patch("src.ui.ui_manager.st")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_render_sidebar_handles_exception(self, _, mock_st, mock_display_sidebar):
        """Test that render_sidebar handles exceptions gracefully."""
        # Arrange
        state_manager = StateManager()
        ui_manager = UIManager(state_manager)
        mock_display_sidebar.side_effect = ValueError("Sidebar error")

        # Act (should not raise exception)
        ui_manager.render_sidebar()

        # Assert
        mock_st.sidebar.error.assert_called_once_with(
            "Error loading sidebar components"
        )

    @patch("src.ui.ui_manager.time")
    @patch("src.ui.ui_manager.st")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_render_processing_indicator_shows_spinner_when_processing(
        self, _, mock_st, mock_time
    ):
        """Test that processing indicator shows spinner when processing is active."""
        # Arrange
        state_manager = StateManager()
        state_manager.is_processing = True
        ui_manager = UIManager(state_manager)

        # Mock context manager for spinner
        mock_spinner = Mock()
        mock_st.spinner.return_value.__enter__ = Mock(return_value=mock_spinner)
        mock_st.spinner.return_value.__exit__ = Mock(return_value=None)

        # Simulate processing ending after one iteration
        def side_effect_processing(*_, **__):
            state_manager.is_processing = False

        mock_time.sleep.side_effect = side_effect_processing

        # Act
        ui_manager.render_processing_indicator()

        # Assert
        mock_st.spinner.assert_called_once_with("Processing your CV... Please wait.")
        mock_time.sleep.assert_called_with(0.1)

    @patch("src.ui.ui_manager.st")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_render_status_messages_shows_success_when_just_finished(self, _, mock_st):
        """Test that success message is shown when just_finished is True."""
        # Arrange
        state_manager = StateManager()
        state_manager.just_finished = True
        ui_manager = UIManager(state_manager)

        # Act
        ui_manager.render_status_messages()

        # Assert
        mock_st.success.assert_called_once_with("CV Generation Complete!")
        assert state_manager.just_finished is False  # Should be reset

    @patch("src.ui.ui_manager.st")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_render_status_messages_shows_error_when_workflow_error(self, _, mock_st):
        """Test that error message is shown when workflow_error is present."""
        # Arrange
        state_manager = StateManager()
        state_manager.workflow_error = "Test error message"
        ui_manager = UIManager(state_manager)

        # Act
        ui_manager.render_status_messages()

        # Assert
        mock_st.error.assert_called_once_with(
            "An error occurred during CV generation: Test error message"
        )
        assert state_manager.workflow_error is None  # Should be reset

    @patch("src.ui.ui_manager.st")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_render_status_messages_shows_agent_errors(self, _, mock_st):
        """Test that agent state errors are displayed."""
        # Arrange
        mock_agent_state = Mock()
        mock_agent_state.error_messages = ["Error 1", "Error 2"]
        state_manager = StateManager()
        state_manager.agent_state = mock_agent_state
        ui_manager = UIManager(state_manager)

        # Act
        ui_manager.render_status_messages()

        # Assert
        expected_calls = [call("Error 1"), call("Error 2")]
        mock_st.error.assert_has_calls(expected_calls)

    @patch("src.ui.ui_manager.st")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_render_main_tabs_creates_three_tabs(self, _, mock_st):
        """Test that main tabs are created with correct labels."""
        # Arrange
        state_manager = StateManager()
        ui_manager = UIManager(state_manager)

        # Mock tab context managers
        mock_tab1, mock_tab2, mock_tab3 = Mock(), Mock(), Mock()
        # Make them proper context managers
        for tab in [mock_tab1, mock_tab2, mock_tab3]:
            tab.__enter__ = Mock(return_value=tab)
            tab.__exit__ = Mock(return_value=None)

        mock_st.tabs.return_value = [mock_tab1, mock_tab2, mock_tab3]

        # Mock the UI component functions that are called within tabs
        with patch("src.ui.ui_manager.display_input_form"), patch(
            "src.ui.ui_manager.display_review_and_edit_tab"
        ), patch("src.ui.ui_manager.display_export_tab"):
            # Act
            ui_manager.render_main_tabs()

        # Assert
        mock_st.tabs.assert_called_once_with(
            ["📝 Input & Generate", "✏️ Review & Edit", "📄 Export"]
        )

    @patch("src.ui.ui_manager.st")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_get_user_inputs_returns_correct_tuple(self, _, __):
        """Test that get_user_inputs returns correct cv and job description texts."""
        # Arrange
        state_manager = StateManager()
        state_manager.cv_text = "test cv"
        state_manager.job_description_text = "test job"
        ui_manager = UIManager(state_manager)

        # Act
        cv_text, job_text = ui_manager.get_user_inputs()

        # Assert
        assert cv_text == "test cv"
        assert job_text == "test job"

    @patch("src.ui.ui_manager.st")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_show_startup_error_displays_errors_and_details(self, _, mock_st):
        """Test that startup errors are displayed with details."""
        # Arrange
        state_manager = StateManager()
        ui_manager = UIManager(state_manager)
        errors = ["Error 1", "Error 2"]
        services_info = {
            "service1": Mock(initialized=True, initialization_time=0.5, error=None),
            "service2": Mock(
                initialized=False, initialization_time=1.0, error="Service error"
            ),
        }

        # Mock context manager for expander
        mock_expander = Mock()
        mock_st.expander.return_value.__enter__ = Mock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = Mock(return_value=None)

        # Act
        ui_manager.show_startup_error(errors, services_info)

        # Assert
        mock_st.error.assert_any_call("**Application Startup Failed:**")
        mock_st.error.assert_any_call("• Error 1")
        mock_st.error.assert_any_call("• Error 2")
        mock_st.warning.assert_called_once()
        mock_st.expander.assert_called_once_with("🔍 Startup Details", expanded=False)

    @patch("src.ui.ui_manager.st")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_show_validation_error_displays_all_errors(self, _, mock_st):
        """Test that validation errors are displayed."""
        # Arrange
        state_manager = StateManager()
        ui_manager = UIManager(state_manager)
        validation_errors = ["Validation Error 1", "Validation Error 2"]

        # Act
        ui_manager.show_validation_error(validation_errors)

        # Assert
        mock_st.error.assert_any_call("**Critical Service Validation Failed:**")
        mock_st.error.assert_any_call("• Validation Error 1")
        mock_st.error.assert_any_call("• Validation Error 2")

    @patch("src.ui.ui_manager.traceback")
    @patch("src.ui.ui_manager.st")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_show_unexpected_error_displays_error_with_traceback(
        self, _, mock_st, mock_traceback
    ):
        """Test that unexpected errors are displayed with traceback."""
        # Arrange
        state_manager = StateManager()
        ui_manager = UIManager(state_manager)
        test_error = Exception("Test exception")
        mock_traceback.format_exc.return_value = "Traceback details"

        # Mock context manager for expander
        mock_expander = Mock()
        mock_st.expander.return_value.__enter__ = Mock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = Mock(return_value=None)

        # Act
        ui_manager.show_unexpected_error(test_error)

        # Assert
        mock_st.error.assert_called_once_with(
            "An unexpected error occurred: Test exception"
        )
        mock_st.expander.assert_called_once_with("🔍 Error Details", expanded=False)
