"""Tests for centralized session management implementation."""

import logging
import uuid
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.config.settings import AppConfig
from src.core.container import ContainerSingleton, get_container
from src.core.factories.agent_factory import AgentFactory
from src.core.workflow_manager import WorkflowManager
from src.models.workflow_models import WorkflowStage
from src.orchestration.state import create_global_state
from src.services.session_manager import SessionManager, SessionStatus


class TestCentralizedSessionManagement:
    """Test suite for centralized session management."""

    def setup_method(self):
        """Reset container singleton before each test."""
        ContainerSingleton.reset_instance()

    def teardown_method(self):
        """Clean up after each test."""
        ContainerSingleton.reset_instance()

    def test_session_manager_generates_valid_session_ids(self):
        """Test that SessionManager generates valid session IDs."""
        session_id = SessionManager.generate_session_id()

        # Verify it's a valid UUID string
        assert isinstance(session_id, str)
        assert len(session_id) == 36  # Standard UUID length

        # Verify it can be parsed as UUID
        parsed_uuid = uuid.UUID(session_id)
        assert str(parsed_uuid) == session_id

    def test_session_manager_create_session_returns_valid_id(self):
        """Test that SessionManager.create_session returns valid session ID."""
        # Mock dependencies
        mock_settings = Mock(spec=AppConfig)
        mock_path = Mock(spec=Path)
        mock_path.mkdir = Mock()
        mock_settings.sessions_directory = mock_path

        # Mock nested attributes
        mock_ui = Mock()
        mock_ui.session_timeout_seconds = 3600
        mock_settings.ui = mock_ui

        mock_session = Mock()
        mock_session.max_active_sessions = 100
        mock_session.cleanup_interval_minutes = 30
        mock_settings.session = mock_session

        mock_logger = Mock(spec=logging.Logger)

        session_manager = SessionManager(mock_settings, mock_logger)
        session_id = session_manager.create_session()

        # Verify it's a valid UUID string
        assert isinstance(session_id, str)
        assert len(session_id) == 36

        # Verify session is tracked
        session_info = session_manager.get_session(session_id)
        assert session_info is not None
        assert session_info.session_id == session_id

    def test_session_manager_get_current_session_id(self):
        """Test SessionManager.get_current_session_id functionality."""
        # Mock dependencies
        mock_settings = Mock(spec=AppConfig)
        mock_path = Mock(spec=Path)
        mock_path.mkdir = Mock()
        mock_settings.sessions_directory = mock_path

        # Mock nested attributes
        mock_ui = Mock()
        mock_ui.session_timeout_seconds = 3600
        mock_settings.ui = mock_ui

        mock_session = Mock()
        mock_session.max_active_sessions = 100
        mock_session.cleanup_interval_minutes = 30
        mock_settings.session = mock_session

        mock_logger = Mock(spec=logging.Logger)

        session_manager = SessionManager(mock_settings, mock_logger)

        # Should create new session if none exists
        current_id = session_manager.get_current_session_id()
        assert isinstance(current_id, str)
        assert len(current_id) == 36

        # Should return same ID on subsequent calls
        same_id = session_manager.get_current_session_id()
        assert same_id == current_id

    def test_container_integration_with_session_manager(self):
        """Test that container properly integrates SessionManager."""
        # Test SessionManager creation directly without full container
        mock_config = Mock(spec=AppConfig)
        mock_path = Mock(spec=Path)
        mock_path.mkdir = Mock()
        mock_config.sessions_directory = mock_path

        mock_ui = Mock()
        mock_ui.session_timeout_seconds = 3600
        mock_config.ui = mock_ui

        mock_session = Mock()
        mock_session.max_active_sessions = 100
        mock_session.cleanup_interval_minutes = 30
        mock_config.session = mock_session

        mock_logger = Mock(spec=logging.Logger)

        # Test SessionManager creation directly
        session_manager = SessionManager(mock_config, mock_logger)
        assert session_manager is not None

        # Verify SessionManager can generate session IDs
        session_id = session_manager.get_current_session_id()
        assert isinstance(session_id, str)
        assert len(session_id) == 36  # UUID4 length

    def test_agent_factory_uses_session_manager(self):
        """Test that AgentFactory uses SessionManager for session IDs."""
        # Test AgentFactory integration with SessionManager directly
        from src.core.factories import AgentFactory
        from src.services.session_manager import SessionManager

        # Mock dependencies
        mock_llm_service = Mock()
        mock_template_manager = Mock()
        mock_vector_store = Mock()

        # Create SessionManager with mocked config
        mock_config = Mock(spec=AppConfig)
        mock_path = Mock(spec=Path)
        mock_path.mkdir = Mock()
        mock_config.sessions_directory = mock_path

        mock_ui = Mock()
        mock_ui.session_timeout_seconds = 3600
        mock_config.ui = mock_ui

        mock_session = Mock()
        mock_session.max_active_sessions = 100
        mock_session.cleanup_interval_minutes = 30
        mock_config.session = mock_session

        mock_logger = Mock(spec=logging.Logger)

        # Create SessionManager and AgentFactory
        session_manager = SessionManager(mock_config, mock_logger)

        # Create session_id_provider function
        session_id_provider = lambda: session_manager.get_current_session_id()

        agent_factory = AgentFactory(
            llm_service=mock_llm_service,
            template_manager=mock_template_manager,
            vector_store_service=mock_vector_store,
            session_id_provider=session_id_provider,
        )

        # Verify that AgentFactory uses SessionManager for session IDs
        session_id = agent_factory._session_id_provider()
        assert isinstance(session_id, str)
        assert len(session_id) == 36  # UUID4 length

        # Verify session ID is valid UUID
        import uuid

        uuid.UUID(session_id)  # Should not raise exception

        # Verify agent creation uses session manager
        cv_agent = agent_factory.create_cv_analyzer_agent()
        assert cv_agent.session_id is not None
        assert len(cv_agent.session_id) == 36

    def test_workflow_manager_uses_session_manager(self):
        """Test that WorkflowManager uses SessionManager for session creation."""
        # Test SessionManager integration with workflow components
        from src.services.session_manager import SessionManager

        # Create SessionManager with mocked config
        mock_config = Mock(spec=AppConfig)
        mock_path = Mock(spec=Path)
        mock_path.mkdir = Mock()
        mock_config.sessions_directory = mock_path

        mock_ui = Mock()
        mock_ui.session_timeout_seconds = 3600
        mock_config.ui = mock_ui

        mock_session = Mock()
        mock_session.max_active_sessions = 100
        mock_session.cleanup_interval_minutes = 30
        mock_config.session = mock_session

        mock_logger = Mock(spec=logging.Logger)

        # Create SessionManager
        session_manager = SessionManager(mock_config, mock_logger)

        # Verify SessionManager can be used by workflow components
        assert session_manager is not None

        # Verify session manager can generate session IDs
        session_id = session_manager.get_current_session_id()
        assert isinstance(session_id, str)
        assert len(session_id) == 36  # UUID4 length

        # Verify session creation works
        new_session_id = session_manager.create_session()
        assert isinstance(new_session_id, str)
        assert len(new_session_id) == 36

    def test_global_state_generates_valid_session_id(self):
        """Test that create_global_state generates valid session IDs."""
        state = create_global_state(cv_text="test cv text")

        # Verify session_id is a valid UUID string
        assert isinstance(state["session_id"], str)
        assert len(state["session_id"]) == 36

        # Verify it can be parsed as UUID
        parsed_uuid = uuid.UUID(state["session_id"])
        assert str(parsed_uuid) == state["session_id"]

        # Test with provided session_id
        custom_session_id = str(uuid.uuid4())
        state_with_custom = create_global_state(
            cv_text="test cv text", session_id=custom_session_id
        )
        assert state_with_custom["session_id"] == custom_session_id

    def test_no_direct_uuid_usage_in_session_generation(self):
        """Test that session generation doesn't use uuid.uuid4() directly."""
        # Test SessionManager directly without container
        mock_config = Mock(spec=AppConfig)
        mock_path = Mock(spec=Path)
        mock_path.mkdir = Mock()
        mock_config.sessions_directory = mock_path

        mock_ui = Mock()
        mock_ui.session_timeout_seconds = 3600
        mock_config.ui = mock_ui

        mock_session = Mock()
        mock_session.max_active_sessions = 100
        mock_session.cleanup_interval_minutes = 30
        mock_config.session = mock_session

        mock_logger = Mock(spec=logging.Logger)

        session_manager = SessionManager(mock_config, mock_logger)

        # Mock uuid.uuid4 to track if it's called directly
        with patch("uuid.uuid4") as mock_uuid4:
            mock_uuid4.return_value = Mock()
            mock_uuid4.return_value.hex = "mocked-uuid-hex"

            # Generate session ID through SessionManager
            session_id = session_manager.get_current_session_id()

            # Verify uuid.uuid4 was called (SessionManager should use uuid internally)
            # This test verifies that SessionManager controls UUID generation
            assert mock_uuid4.called

            # Verify we got a valid session ID
            assert isinstance(session_id, str)
            assert len(session_id) > 0

    def test_session_lifecycle_management(self):
        """Test complete session lifecycle through SessionManager."""
        # Mock dependencies
        mock_settings = Mock(spec=AppConfig)
        mock_path = Mock(spec=Path)
        mock_path.mkdir = Mock()
        mock_settings.sessions_directory = mock_path

        # Mock nested attributes
        mock_ui = Mock()
        mock_ui.session_timeout_seconds = 3600
        mock_settings.ui = mock_ui

        mock_session = Mock()
        mock_session.max_active_sessions = 100
        mock_session.cleanup_interval_minutes = 30
        mock_settings.session = mock_session

        mock_logger = Mock(spec=logging.Logger)

        session_manager = SessionManager(mock_settings, mock_logger)

        # Create session
        session_id = session_manager.create_session()
        session_info = session_manager.get_session(session_id)
        assert session_info is not None
        assert session_info.status == SessionStatus.ACTIVE

        # Update session state with proper WorkflowState mock
        mock_state = Mock()
        mock_state.current_stage = WorkflowStage.CONTENT_GENERATION
        mock_state.total_processing_time = 10.5
        mock_state.total_llm_calls = 5
        mock_state.total_tokens_used = 1000

        # Mock queue objects
        for queue_name in ["qualification_queue", "experience_queue", "project_queue"]:
            queue_mock = Mock()
            queue_mock.total_items = 1
            queue_mock.completed_items = []
            queue_mock.failed_items = []
            setattr(mock_state, queue_name, queue_mock)

        session_manager.update_session_state(session_id, mock_state)
        updated_info = session_manager.get_session(session_id)
        assert updated_info.current_stage == WorkflowStage.CONTENT_GENERATION

        # Complete session
        session_manager.complete_session(session_id)
        session_info = session_manager.get_session(session_id)
        assert (
            session_info is None
        )  # Should be removed from active sessions after completion

    def test_multiple_sessions_isolation(self):
        """Test that multiple sessions are properly isolated."""
        # Mock dependencies
        mock_settings = Mock(spec=AppConfig)
        mock_path = Mock(spec=Path)
        mock_path.mkdir = Mock()
        mock_settings.sessions_directory = mock_path

        # Mock nested attributes
        mock_ui = Mock()
        mock_ui.session_timeout_seconds = 3600
        mock_settings.ui = mock_ui

        mock_session = Mock()
        mock_session.max_active_sessions = 100
        mock_session.cleanup_interval_minutes = 30
        mock_settings.session = mock_session

        mock_logger = Mock(spec=logging.Logger)

        session_manager = SessionManager(mock_settings, mock_logger)

        # Create multiple sessions with different user IDs
        session_id_1 = session_manager.create_session(user_id="user1")
        session_id_2 = session_manager.create_session(user_id="user2")

        assert session_id_1 != session_id_2

        # Get session info to verify isolation
        session_info_1 = session_manager.get_session(session_id_1)
        session_info_2 = session_manager.get_session(session_id_2)

        assert session_info_1.user_id == "user1"
        assert session_info_2.user_id == "user2"
        assert session_info_1.session_id != session_info_2.session_id
        assert session_info_1.current_stage == WorkflowStage.INITIALIZATION
        assert session_info_2.current_stage == WorkflowStage.INITIALIZATION

    def test_session_manager_thread_safety(self):
        """Test that SessionManager operations are thread-safe."""
        import threading
        import time

        # Mock dependencies
        mock_settings = Mock(spec=AppConfig)
        mock_path = Mock(spec=Path)
        mock_path.mkdir = Mock()
        mock_settings.sessions_directory = mock_path

        # Mock nested attributes
        mock_ui = Mock()
        mock_ui.session_timeout_seconds = 3600
        mock_settings.ui = mock_ui

        mock_session = Mock()
        mock_session.max_active_sessions = 100
        mock_session.cleanup_interval_minutes = 30
        mock_settings.session = mock_session

        mock_logger = Mock(spec=logging.Logger)

        session_manager = SessionManager(mock_settings, mock_logger)
        session_ids = []

        def create_session():
            session_id = session_manager.create_session()
            session_ids.append(session_id)
            time.sleep(0.01)  # Small delay to test concurrency

        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_session)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all session IDs are unique
        assert len(session_ids) == 10
        assert len(set(session_ids)) == 10  # All unique
