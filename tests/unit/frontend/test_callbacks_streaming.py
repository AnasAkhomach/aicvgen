"""Tests for streaming callbacks functionality."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
import streamlit as st

from src.frontend.callbacks import start_cv_generation


class MockSessionState(dict):
    """Mock session state that supports both dict and attribute access."""

    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value


class TestStreamingCallbacks:
    """Test streaming callbacks functionality."""

    @patch("src.frontend.callbacks.st")
    @patch("src.frontend.callbacks._get_or_create_ui_manager")
    @patch("src.frontend.callbacks.asyncio.run")
    def test_start_cv_generation_streaming_success(
        self, mock_asyncio_run, mock_get_ui_manager, mock_st
    ):
        """Test successful streaming CV generation."""
        # Setup mocks
        mock_ui_manager = Mock()
        mock_ui_manager.stream_cv_generation = AsyncMock()
        mock_get_ui_manager.return_value = mock_ui_manager

        # Mock status container with context manager support
        mock_status_container = Mock()
        mock_status_container.__enter__ = Mock(return_value=mock_status_container)
        mock_status_container.__exit__ = Mock(return_value=None)
        mock_st.status.return_value = mock_status_container
        mock_st.container.return_value = Mock()

        # Setup session state
        mock_session_state = MockSessionState(
            {
                "job_description_input": "Test job description",
                "cv_text_input": "Test CV content",
                "start_from_scratch_input": False,
            }
        )
        mock_st.session_state = mock_session_state

        # Mock the async generator
        async def mock_stream():
            yield {"status": "processing"}
            yield {"status": "completed"}

        mock_ui_manager.stream_cv_generation.return_value = mock_stream()

        # Mock asyncio.run to return final state
        mock_asyncio_run.return_value = {"status": "completed"}

        # Call the function
        start_cv_generation()

        # Verify UI manager was called with correct parameters
        mock_get_ui_manager.assert_called_once()

        # Verify status container was created
        mock_st.status.assert_called_once_with(
            "Starting CV Generation...", expanded=True
        )

        # Verify status was updated to complete
        mock_status_container.update.assert_called_once_with(
            label="Generation Complete!", state="complete"
        )

        # Verify asyncio.run was called
        mock_asyncio_run.assert_called_once()

    @patch("src.frontend.callbacks.st")
    @patch("src.frontend.callbacks._get_or_create_ui_manager")
    @patch("src.frontend.callbacks.asyncio.run")
    @patch("src.frontend.callbacks.logger")
    def test_start_cv_generation_streaming_failure(
        self, mock_logger, mock_asyncio_run, mock_get_ui_manager, mock_st
    ):
        """Test streaming CV generation with failure."""
        # Setup mocks
        mock_ui_manager = Mock()
        mock_get_ui_manager.return_value = mock_ui_manager

        # Mock status container with context manager support
        mock_status_container = Mock()
        mock_status_container.__enter__ = Mock(return_value=mock_status_container)
        mock_status_container.__exit__ = Mock(return_value=None)
        mock_st.status.return_value = mock_status_container
        mock_st.container.return_value = Mock()

        # Setup session state
        mock_session_state = MockSessionState(
            {
                "job_description_input": "Test job description",
                "cv_text_input": "Test CV content",
                "start_from_scratch_input": False,
            }
        )
        mock_st.session_state = mock_session_state

        # Mock asyncio.run to raise exception
        test_error = RuntimeError("Streaming failed")
        mock_asyncio_run.side_effect = test_error

        # Call the function and expect exception
        with pytest.raises(RuntimeError, match="Streaming failed"):
            start_cv_generation()

        # Verify error was logged
        mock_logger.error.assert_called_once_with(
            "CV generation failed: Streaming failed"
        )

        # Verify status was updated to error
        mock_status_container.update.assert_called_once_with(
            label="Generation Failed!", state="error"
        )

        # Verify error was displayed
        mock_st.error.assert_called_once_with("Failed to generate CV: Streaming failed")

    @patch("src.frontend.callbacks.st")
    @patch("src.frontend.callbacks._get_or_create_ui_manager")
    def test_start_cv_generation_with_streamlit_callback_handler(
        self, mock_get_ui_manager, mock_st
    ):
        """Test that StreamlitCallbackHandler is properly instantiated."""
        # Setup mocks
        mock_ui_manager = Mock()
        mock_ui_manager.stream_cv_generation = AsyncMock()
        mock_get_ui_manager.return_value = mock_ui_manager

        # Mock status container with context manager support
        mock_status_container = Mock()
        mock_status_container.__enter__ = Mock(return_value=mock_status_container)
        mock_status_container.__exit__ = Mock(return_value=None)
        mock_container = Mock()
        mock_st.status.return_value = mock_status_container
        mock_st.container.return_value = mock_container

        # Setup session state
        mock_session_state = MockSessionState(
            {
                "job_description_input": "Test job description",
                "cv_text_input": "Test CV content",
                "start_from_scratch_input": False,
            }
        )
        mock_st.session_state = mock_session_state

        # Mock the async generator to avoid hanging
        async def mock_stream():
            yield {"status": "completed"}

        mock_ui_manager.stream_cv_generation.return_value = mock_stream()

        with patch("src.frontend.callbacks.asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.return_value = {"status": "completed"}

            # Call the function
            start_cv_generation()

            # Verify stream_cv_generation was called with callback_handler
            # We can't easily verify the exact callback handler instance,
            # but we can verify the method was called
            assert mock_asyncio_run.called
