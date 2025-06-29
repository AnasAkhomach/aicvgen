"""Integration tests for the refactored main.py orchestration logic.

These tests verify that the StateManager and UIManager work together correctly
and that the main function properly orchestrates the application flow following
the new clean architecture pattern.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from src.core.main import main, initialize_application
from src.core.state_manager import StateManager
from src.ui.ui_manager import UIManager


class TestMainOrchestration:
    """Test suite for main.py orchestration logic integration."""

    @patch("src.core.main.st")
    @patch("src.core.main.UIManager")
    @patch("src.core.main.StateManager")
    @patch("src.core.main.initialize_application")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_successful_application_startup_and_ui_render(
        self,
        mock_session_state,
        mock_init_app,
        mock_state_manager_class,
        mock_ui_manager_class,
        mock_st,
    ):
        """Test that main() successfully initializes and renders UI when startup succeeds."""
        # Arrange
        mock_state_manager = Mock()
        mock_ui_manager = Mock()
        mock_state_manager_class.return_value = mock_state_manager
        mock_ui_manager_class.return_value = mock_ui_manager
        mock_init_app.return_value = True

        # Act
        main()

        # Assert
        mock_state_manager_class.assert_called_once()
        mock_ui_manager_class.assert_called_once_with(mock_state_manager)
        mock_init_app.assert_called_once_with(mock_state_manager)
        mock_ui_manager.render_full_ui.assert_called_once()

    @patch("src.core.main.st")
    @patch("src.core.main.UIManager")
    @patch("src.core.main.StateManager")
    @patch("src.core.main.initialize_application")
    @patch("src.core.main.get_startup_manager")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_startup_failure_handling_with_result_details(
        self,
        mock_session_state,
        mock_get_startup_manager,
        mock_init_app,
        mock_state_manager_class,
        mock_ui_manager_class,
        mock_st,
    ):
        """Test that main() handles startup failure and shows detailed error information."""
        # Arrange
        mock_state_manager = Mock()
        mock_ui_manager = Mock()
        mock_startup_service = Mock()
        mock_startup_result = Mock()
        mock_startup_result.errors = ["Service A failed", "Service B timeout"]
        mock_startup_result.services = {"serviceA": Mock(), "serviceB": Mock()}

        mock_state_manager_class.return_value = mock_state_manager
        mock_ui_manager_class.return_value = mock_ui_manager
        mock_init_app.return_value = False
        mock_get_startup_manager.return_value = mock_startup_service
        mock_startup_service.last_startup_result = mock_startup_result

        # Act
        main()

        # Assert
        mock_init_app.assert_called_once_with(mock_state_manager)
        mock_ui_manager.show_startup_error.assert_called_once_with(
            mock_startup_result.errors, mock_startup_result.services
        )
        mock_st.stop.assert_called()

    @patch("src.core.main.st")
    @patch("src.core.main.UIManager")
    @patch("src.core.main.StateManager")
    @patch("src.core.main.initialize_application")
    @patch("src.core.main.get_startup_manager")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_startup_failure_handling_without_result_details(
        self,
        mock_session_state,
        mock_get_startup_manager,
        mock_init_app,
        mock_state_manager_class,
        mock_ui_manager_class,
        mock_st,
    ):
        """Test that main() handles startup failure gracefully when no result details available."""
        # Arrange
        mock_state_manager = Mock()
        mock_ui_manager = Mock()
        mock_startup_service = Mock()

        mock_state_manager_class.return_value = mock_state_manager
        mock_ui_manager_class.return_value = mock_ui_manager
        mock_init_app.return_value = False
        mock_get_startup_manager.return_value = mock_startup_service
        # No last_startup_result attribute - ensure getattr returns None
        mock_startup_service.configure_mock(**{"last_startup_result": None})
        del mock_startup_service.last_startup_result  # Remove the attribute completely

        # Act
        main()

        # Assert
        mock_st.error.assert_called_with(
            "Application startup failed. Please check configuration."
        )
        mock_st.stop.assert_called()

    @patch("src.core.main.st")
    @patch("src.core.main.UIManager")
    @patch("src.core.main.StateManager")
    @patch("src.core.main.initialize_application")
    @patch("src.core.main.get_startup_manager")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_validation_error_handling(
        self,
        mock_session_state,
        mock_get_startup_manager,
        mock_init_app,
        mock_state_manager_class,
        mock_ui_manager_class,
        mock_st,
    ):
        """Test that main() handles validation errors after successful startup."""
        # Arrange
        mock_state_manager = Mock()
        mock_ui_manager = Mock()
        mock_startup_service = Mock()
        validation_errors = ["Database connection failed", "API key invalid"]

        mock_state_manager_class.return_value = mock_state_manager
        mock_ui_manager_class.return_value = mock_ui_manager
        mock_init_app.return_value = True
        mock_get_startup_manager.return_value = mock_startup_service
        mock_startup_service.validate_application.return_value = validation_errors

        # Act
        main()

        # Assert
        mock_ui_manager.show_validation_error.assert_called_once_with(validation_errors)
        mock_st.stop.assert_called()

    @patch("src.core.main.st")
    @patch("src.core.main.UIManager")
    @patch("src.core.main.StateManager")
    @patch("src.core.main.initialize_application")
    @patch("src.core.main.get_startup_manager")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_successful_flow_without_validation_errors(
        self,
        mock_session_state,
        mock_get_startup_manager,
        mock_init_app,
        mock_state_manager_class,
        mock_ui_manager_class,
        mock_st,
    ):
        """Test the complete successful flow without any errors."""
        # Arrange
        mock_state_manager = Mock()
        mock_ui_manager = Mock()
        mock_startup_service = Mock()

        mock_state_manager_class.return_value = mock_state_manager
        mock_ui_manager_class.return_value = mock_ui_manager
        mock_init_app.return_value = True
        mock_get_startup_manager.return_value = mock_startup_service
        mock_startup_service.validate_application.return_value = []  # No errors

        # Act
        main()

        # Assert
        mock_state_manager_class.assert_called_once()
        mock_ui_manager_class.assert_called_once_with(mock_state_manager)
        mock_init_app.assert_called_once_with(mock_state_manager)
        mock_startup_service.validate_application.assert_called_once()
        mock_ui_manager.render_full_ui.assert_called_once()
        mock_st.stop.assert_not_called()

    @patch("src.core.main.st")
    @patch("src.core.main.UIManager")
    @patch("src.core.main.StateManager")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_configuration_error_handling(
        self,
        mock_session_state,
        mock_state_manager_class,
        mock_ui_manager_class,
        mock_st,
    ):
        """Test that main() handles ConfigurationError properly."""
        # Arrange
        from src.error_handling.exceptions import ConfigurationError

        error_message = "Missing configuration file"
        mock_state_manager_class.side_effect = ConfigurationError(error_message)

        # Act
        main()

        # Assert
        mock_st.error.assert_called()
        mock_st.warning.assert_called()
        mock_st.stop.assert_called()
        mock_st.warning.assert_called()
        mock_st.stop.assert_called()

    @patch("src.core.main.st")
    @patch("src.core.main.UIManager")
    @patch("src.core.main.StateManager")
    @patch("src.core.main.logger")
    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_catchable_exception_handling_with_fallback(
        self,
        mock_session_state,
        mock_logger,
        mock_state_manager_class,
        mock_ui_manager_class,
        mock_st,
    ):
        """Test that main() handles catchable exceptions with UI fallback."""
        # Arrange
        error = ValueError("Test error")
        mock_fallback_state_manager = Mock()
        mock_fallback_ui_manager = Mock()

        # First StateManager call fails, second succeeds for fallback
        mock_state_manager_class.side_effect = [error, mock_fallback_state_manager]
        # First UIManager call won't happen due to StateManager failure, second succeeds for fallback
        mock_ui_manager_class.return_value = mock_fallback_ui_manager

        # Act
        main()

        # Assert
        mock_logger.error.assert_called()
        # Should create fallback UI manager for error display
        assert (
            mock_ui_manager_class.call_count == 1
        )  # Only the fallback UI manager is created
        mock_fallback_ui_manager.show_unexpected_error.assert_called_with(error)
        mock_st.stop.assert_called()
        mock_st.stop.assert_called()


class TestInitializeApplication:
    """Test suite for initialize_application function."""

    @patch("src.core.main.get_startup_manager")
    @patch("src.core.main.atexit")
    @patch("src.core.main.setup_logging")
    def test_successful_initialization(
        self, mock_setup_logging, mock_atexit, mock_get_startup_manager
    ):
        """Test successful application initialization."""
        # Arrange
        mock_state_manager = Mock()
        mock_state_manager.user_gemini_api_key = "test_api_key"

        mock_startup_service = Mock()
        mock_startup_service.is_initialized = False
        mock_startup_result = Mock()
        mock_startup_result.success = True
        mock_startup_result.total_time = 1.5

        mock_startup_service.initialize_application.return_value = mock_startup_result
        mock_startup_service.validate_application.return_value = []
        mock_get_startup_manager.return_value = mock_startup_service

        # Act
        result = initialize_application(mock_state_manager)

        # Assert
        assert result is True
        mock_setup_logging.assert_called_once()
        mock_startup_service.initialize_application.assert_called_once_with(
            user_api_key="test_api_key"
        )
        mock_startup_service.validate_application.assert_called_once()

    @patch("src.core.main.get_startup_manager")
    @patch("src.core.main.atexit")
    @patch("src.core.main.setup_logging")
    def test_initialization_failure(
        self, mock_setup_logging, mock_atexit, mock_get_startup_manager
    ):
        """Test application initialization failure."""
        # Arrange
        mock_state_manager = Mock()
        mock_state_manager.user_gemini_api_key = "test_api_key"

        mock_startup_service = Mock()
        mock_startup_service.is_initialized = False
        mock_startup_result = Mock()
        mock_startup_result.success = False

        mock_startup_service.initialize_application.return_value = mock_startup_result
        mock_get_startup_manager.return_value = mock_startup_service

        # Act
        result = initialize_application(mock_state_manager)

        # Assert
        assert result is False
        mock_startup_service.validate_application.assert_not_called()

    @patch("src.core.main.get_startup_manager")
    @patch("src.core.main.atexit")
    @patch("src.core.main.setup_logging")
    def test_validation_errors_cause_failure(
        self, mock_setup_logging, mock_atexit, mock_get_startup_manager
    ):
        """Test that validation errors cause initialization to fail."""
        # Arrange
        mock_state_manager = Mock()
        mock_state_manager.user_gemini_api_key = "test_api_key"

        mock_startup_service = Mock()
        mock_startup_service.is_initialized = False
        mock_startup_result = Mock()
        mock_startup_result.success = True
        mock_startup_result.total_time = 1.5

        mock_startup_service.initialize_application.return_value = mock_startup_result
        mock_startup_service.validate_application.return_value = ["Validation error"]
        mock_get_startup_manager.return_value = mock_startup_service

        # Act
        result = initialize_application(mock_state_manager)

        # Assert
        assert result is False

    @patch("src.core.main.get_startup_manager")
    def test_already_initialized_service(self, mock_get_startup_manager):
        """Test that already initialized service is handled correctly."""
        # Arrange
        mock_state_manager = Mock()

        mock_startup_service = Mock()
        mock_startup_service.is_initialized = True
        mock_startup_service.validate_application.return_value = []
        mock_get_startup_manager.return_value = mock_startup_service

        # Act
        result = initialize_application(mock_state_manager)

        # Assert
        assert result is True
        mock_startup_service.initialize_application.assert_not_called()
        mock_startup_service.validate_application.assert_called_once()


class TestStateManagerUIManagerIntegration:
    """Test integration between StateManager and UIManager."""

    @patch("streamlit.session_state", new_callable=lambda: {})
    @patch("src.ui.ui_manager.st")
    def test_state_manager_ui_manager_integration(self, mock_st, mock_session_state):
        """Test that StateManager and UIManager work together correctly."""
        # Arrange
        state_manager = StateManager()

        # Mock UI components to avoid import issues
        with patch("src.ui.ui_manager.display_sidebar"), patch(
            "src.ui.ui_manager.display_input_form"
        ), patch("src.ui.ui_manager.display_review_and_edit_tab"), patch(
            "src.ui.ui_manager.display_export_tab"
        ):

            ui_manager = UIManager(state_manager)

            # Act
            state_manager.cv_text = "Test CV content"
            state_manager.job_description_text = "Test job description"
            state_manager.is_processing = True

            cv_text, job_text = ui_manager.get_user_inputs()

            # Assert
            assert cv_text == "Test CV content"
            assert job_text == "Test job description"
            assert state_manager.has_required_data() is True
            assert state_manager.is_processing is True

    @patch("streamlit.session_state", new_callable=lambda: {})
    @patch("src.ui.ui_manager.st")
    def test_state_manager_ui_manager_error_handling_integration(
        self, mock_st, mock_session_state
    ):
        """Test error handling integration between StateManager and UIManager."""
        # Arrange
        state_manager = StateManager()

        with patch("src.ui.ui_manager.display_sidebar"), patch(
            "src.ui.ui_manager.display_input_form"
        ), patch("src.ui.ui_manager.display_review_and_edit_tab"), patch(
            "src.ui.ui_manager.display_export_tab"
        ):

            ui_manager = UIManager(state_manager)

            # Act
            state_manager.workflow_error = "Test error message"
            state_manager.just_finished = True

            ui_manager.render_status_messages()

            # Assert
            mock_st.success.assert_called_once_with("CV Generation Complete!")
            mock_st.error.assert_called_once_with(
                "An error occurred during CV generation: Test error message"
            )
            assert state_manager.workflow_error is None  # Should be cleared
            assert state_manager.just_finished is False  # Should be reset

    @patch("streamlit.session_state", new_callable=lambda: {})
    @patch("src.ui.ui_manager.st")
    def test_state_persistence_across_ui_operations(self, mock_st, mock_session_state):
        """Test that state persists correctly across UI operations."""
        # Arrange
        state_manager = StateManager()

        with patch("src.ui.ui_manager.display_sidebar"), patch(
            "src.ui.ui_manager.display_input_form"
        ), patch("src.ui.ui_manager.display_review_and_edit_tab"), patch(
            "src.ui.ui_manager.display_export_tab"
        ):

            ui_manager = UIManager(state_manager)

            # Act - simulate multiple operations
            state_manager.cv_text = "Initial CV"
            state_manager.user_gemini_api_key = "api_key_123"

            # Get initial summary
            initial_summary = state_manager.get_state_summary()

            # Update state
            state_manager.is_processing = True
            state_manager.cv_text = "Updated CV"

            # Get updated summary
            updated_summary = state_manager.get_state_summary()

            # Assert
            assert initial_summary["has_cv_text"] is True
            assert initial_summary["is_processing"] is False
            assert updated_summary["has_cv_text"] is True
            assert updated_summary["is_processing"] is True
            assert state_manager.cv_text == "Updated CV"
            assert state_manager.user_gemini_api_key == "api_key_123"
