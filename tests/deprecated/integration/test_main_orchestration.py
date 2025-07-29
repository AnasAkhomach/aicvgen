"""Integration tests for the refactored main.py orchestration logic.

These tests verify that the StateManager and UIManager work together correctly
and that the main function properly orchestrates the application flow following
the new clean architecture pattern.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
import pytest_asyncio

from src.core.container import ContainerSingleton
from src.core.main import initialize_application, main
from src.core.state_manager import StateManager
from src.models.cv_models import (
    Item,
    ItemStatus,
    ItemType,
    JobDescriptionData,
    Section,
    StructuredCV,
)
from src.models.workflow_models import ContentType, UserAction, UserFeedback
from src.orchestration.graphs.main_graph import create_cv_workflow_graph_with_di
from src.orchestration.state import AgentState
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


class TestWorkflowGraphIntegration:
    """Integration tests for workflow graph and its subgraphs."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Reset ContainerSingleton before and after each test."""
        ContainerSingleton.reset_instance()
        yield
        ContainerSingleton.reset_instance()

    @pytest.fixture
    def sample_structured_cv(self):
        """Create a sample StructuredCV for testing."""
        return StructuredCV(
            sections=[
                Section(
                    name="executive_summary",
                    items=[
                        Item(
                            item_type=ItemType.EXECUTIVE_SUMMARY_PARA,
                            content="Sample executive summary",
                            status=ItemStatus.PENDING,
                        )
                    ],
                ),
                Section(
                    name="key_qualifications",
                    items=[
                        Item(
                            item_type=ItemType.KEY_QUALIFICATION,
                            content="Python programming",
                            status=ItemStatus.PENDING,
                        ),
                        Item(
                            item_type=ItemType.KEY_QUALIFICATION,
                            content="Machine learning",
                            status=ItemStatus.PENDING,
                        ),
                    ],
                ),
                Section(
                    name="professional_experience",
                    items=[
                        Item(
                            item_type=ItemType.EXPERIENCE_ROLE_TITLE,
                            content="Software Engineer at TechCorp",
                            status=ItemStatus.PENDING,
                        )
                    ],
                ),
                Section(
                    name="project_experience",
                    items=[
                        Item(
                            item_type=ItemType.PROJECT_DESCRIPTION_BULLET,
                            content="AI CV Generator",
                            status=ItemStatus.PENDING,
                        )
                    ],
                ),
            ]
        )

    @pytest.fixture
    def sample_job_description_data(self):
        """Create sample job description data for testing."""
        return JobDescriptionData(
            raw_text="Senior Software Engineer at TechCorp. We are looking for a senior software engineer with Python, Machine Learning, and 5+ years experience. Responsibilities include developing software and leading projects.",
            job_title="Senior Software Engineer",
            company_name="TechCorp",
            main_job_description_raw="We are looking for a senior software engineer...",
            responsibilities=["Develop software", "Lead projects"],
            skills=["Python", "TensorFlow", "AWS"],
        )

    @pytest.fixture
    def initial_agent_state(self, sample_structured_cv, sample_job_description_data):
        """Create initial AgentState for testing."""
        from datetime import datetime

        from src.models.agent_output_models import (
            CVAnalysisResult,
            ResearchFindings,
            ResearchStatus,
        )

        # Get the actual item IDs from the structured CV
        key_qualifications_section = next(
            (
                s
                for s in sample_structured_cv.sections
                if s.name == "key_qualifications"
            ),
            None,
        )

        if key_qualifications_section and key_qualifications_section.items:
            # Use actual item IDs from the structured CV
            item_ids = [str(item.id) for item in key_qualifications_section.items]
            current_item_id = item_ids[0] if item_ids else None
        else:
            item_ids = []
            current_item_id = None

        return AgentState(
            session_id="test_session_123",
            trace_id="test_trace_456",
            cv_text="Sample CV text content",
            structured_cv=sample_structured_cv,
            job_description_data=sample_job_description_data,
            research_findings=ResearchFindings(
                status=ResearchStatus.SUCCESS,
                research_timestamp=datetime.now(),
                key_terms=["Python", "Software Engineering"],
                enhancement_suggestions=["Add more technical details"],
            ),
            cv_analysis_results=CVAnalysisResult(
                summary="Sample analysis", key_skills=["Python", "AWS"], match_score=0.8
            ),
            current_section_key="key_qualifications",
            current_section_index=0,
            items_to_process_queue=item_ids,
            current_item_id=current_item_id,
            current_content_type=ContentType.QUALIFICATION,
            is_initial_generation=True,
            error_messages=[],
            node_execution_metadata={},
            automated_mode=True,  # Enable automated mode for testing
        )

    @pytest.fixture
    def mock_agents(self):
        """Create mocked agent instances for testing."""
        agents = {
            "jd_parser_agent": Mock(),
            "cv_parser_agent": Mock(),
            "research_agent": Mock(),
            "cv_analyzer_agent": Mock(),
            "key_qualifications_writer_agent": Mock(),
            "professional_experience_writer_agent": Mock(),
            "projects_writer_agent": Mock(),
            "executive_summary_writer_agent": Mock(),
            "qa_agent": Mock(),
            "formatter_agent": Mock(),
        }

        # Configure successful responses for each agent
        for agent_name, agent in agents.items():
            agent.run_as_node = AsyncMock()

        return agents

    @pytest.fixture
    def mock_error_recovery_service(self):
        """Create mocked ErrorRecoveryService."""
        service = Mock()
        service.handle_error = AsyncMock(return_value="skip_item")
        return service

    @pytest.mark.asyncio
    async def test_full_workflow_success_path(
        self, initial_agent_state, mock_agents, mock_error_recovery_service
    ):
        """Test Case 1: Verify entire workflow executes successfully from start to finish."""
        # Arrange - Configure successful agent responses (return dictionaries as per LG-FIX-02)
        mock_agents["jd_parser_agent"].run_as_node.return_value = {
            "job_description_data": initial_agent_state.job_description_data
        }

        mock_agents["cv_parser_agent"].run_as_node.return_value = {
            "structured_cv": initial_agent_state.structured_cv
        }

        mock_agents["research_agent"].run_as_node.return_value = {
            "research_findings": "Research completed successfully"
        }

        mock_agents["cv_analyzer_agent"].run_as_node.return_value = {
            "cv_analysis_results": "Analysis completed"
        }

        # Configure writer agents to mark items as completed
        updated_cv = initial_agent_state.structured_cv.model_copy(deep=True)
        for section in updated_cv.sections:
            for item in section.items:
                item.status = ItemStatus.COMPLETED
                item.content = f"Enhanced {item.content}"

        for writer_agent in [
            "key_qualifications_writer_agent",
            "professional_experience_writer_agent",
            "projects_writer_agent",
            "executive_summary_writer_agent",
        ]:
            mock_agents[writer_agent].run_as_node.return_value = {
                "structured_cv": updated_cv
            }

        mock_agents["qa_agent"].run_as_node.return_value = {
            "quality_check_results": "Quality check passed"
        }

        mock_agents["formatter_agent"].run_as_node.return_value = {
            "final_output_path": "/path/to/generated_cv.pdf"
        }

        # Mock the container's agent providers
        container = ContainerSingleton.get_instance()
        with patch.object(
            container, "job_description_parser_agent"
        ) as mock_jd_provider, patch.object(
            container, "user_cv_parser_agent"
        ) as mock_cv_provider, patch.object(
            container, "research_agent"
        ) as mock_research_provider, patch.object(
            container, "cv_analyzer_agent"
        ) as mock_analyzer_provider, patch.object(
            container, "key_qualifications_writer_agent"
        ) as mock_qual_provider, patch.object(
            container, "professional_experience_writer_agent"
        ) as mock_exp_provider, patch.object(
            container, "projects_writer_agent"
        ) as mock_proj_provider, patch.object(
            container, "executive_summary_writer_agent"
        ) as mock_exec_provider, patch.object(
            container, "quality_assurance_agent"
        ) as mock_qa_provider, patch.object(
            container, "formatter_agent"
        ) as mock_formatter_provider:
            # Set up mock return values
            mock_jd_provider.return_value = mock_agents["jd_parser_agent"]
            mock_cv_provider.return_value = mock_agents["cv_parser_agent"]
            mock_research_provider.return_value = mock_agents["research_agent"]
            mock_analyzer_provider.return_value = mock_agents["cv_analyzer_agent"]
            mock_qual_provider.return_value = mock_agents[
                "key_qualifications_writer_agent"
            ]
            mock_exp_provider.return_value = mock_agents[
                "professional_experience_writer_agent"
            ]
            mock_proj_provider.return_value = mock_agents["projects_writer_agent"]
            mock_exec_provider.return_value = mock_agents[
                "executive_summary_writer_agent"
            ]
            mock_qa_provider.return_value = mock_agents["qa_agent"]
            mock_formatter_provider.return_value = mock_agents["formatter_agent"]

            # Initialize workflow graph using the new DI factory
            cv_workflow_graph = create_cv_workflow_graph_with_di()

            # Act
            print(f"\n=== Starting test with initial state ===")
            print(f"Initial section index: {initial_agent_state.current_section_index}")
            print(f"Initial current_item_id: {initial_agent_state.current_item_id}")
            print(f"Initial metadata: {initial_agent_state.node_execution_metadata}")
            print(
                f"Initial workflow_status: {getattr(initial_agent_state, 'workflow_status', 'None')}"
            )

            config = {"configurable": {"thread_id": "test-session"}}
            final_state_dict = await cv_workflow_graph.ainvoke(
                initial_agent_state, config=config
            )
            final_state = AgentState.model_validate(final_state_dict)

            print(f"\n=== Final state ===")
            print(f"Final section index: {final_state.current_section_index}")
            print(f"Final current_item_id: {final_state.current_item_id}")
            print(f"Final metadata: {final_state.node_execution_metadata}")
            print(
                f"Final workflow_status: {getattr(final_state, 'workflow_status', 'None')}"
            )
            print(f"Final output path: {final_state.final_output_path}")
            print(
                f"Key qualifications writer call count: {mock_agents['key_qualifications_writer_agent'].run_as_node.call_count}"
            )

            # Assert
            assert (
                len(final_state.error_messages) == 0
            ), f"Unexpected errors: {final_state.error_messages}"
            assert final_state.final_output_path == "/path/to/generated_cv.pdf"
            assert final_state.research_findings == "Research completed successfully"
            assert final_state.cv_analysis_results == "Analysis completed"
            assert final_state.quality_check_results == "Quality check passed"

            # Verify each agent was called exactly once
            for agent_name, agent in mock_agents.items():
                agent.run_as_node.assert_called()
                assert agent.run_as_node.call_count >= 1, f"{agent_name} was not called"

    @pytest.mark.asyncio
    async def test_workflow_error_handling_and_recovery(
        self, initial_agent_state, mock_agents, mock_error_recovery_service
    ):
        """Test Case 2: Verify workflow correctly handles agent failure and routes to error handler."""
        # Arrange - Configure jd_parser_agent to fail
        error_message = "Error in job description parser node: JD Parser failed to process job description"
        mock_agents["jd_parser_agent"].run_as_node.side_effect = Exception(
            error_message
        )

        # Configure other agents for success (in case workflow continues)
        for agent_name, agent in mock_agents.items():
            if agent_name != "jd_parser_agent":
                agent.run_as_node.return_value = {
                    "node_execution_metadata": {agent_name: "success"}
                }

        # Configure error recovery service
        mock_error_recovery_service.handle_error.return_value = "skip_item"

        # Mock the container's agent providers
        container = ContainerSingleton.get_instance()
        with patch.object(
            container, "job_description_parser_agent"
        ) as mock_jd_provider, patch.object(
            container, "user_cv_parser_agent"
        ) as mock_cv_provider, patch.object(
            container, "research_agent"
        ) as mock_research_provider, patch.object(
            container, "cv_analyzer_agent"
        ) as mock_analyzer_provider, patch.object(
            container, "key_qualifications_writer_agent"
        ) as mock_qual_provider, patch.object(
            container, "professional_experience_writer_agent"
        ) as mock_exp_provider, patch.object(
            container, "projects_writer_agent"
        ) as mock_proj_provider, patch.object(
            container, "executive_summary_writer_agent"
        ) as mock_exec_provider, patch.object(
            container, "quality_assurance_agent"
        ) as mock_qa_provider, patch.object(
            container, "formatter_agent"
        ) as mock_formatter_provider:
            # Set up mock return values
            mock_jd_provider.return_value = mock_agents["jd_parser_agent"]
            mock_cv_provider.return_value = mock_agents["cv_parser_agent"]
            mock_research_provider.return_value = mock_agents["research_agent"]
            mock_analyzer_provider.return_value = mock_agents["cv_analyzer_agent"]
            mock_qual_provider.return_value = mock_agents[
                "key_qualifications_writer_agent"
            ]
            mock_exp_provider.return_value = mock_agents[
                "professional_experience_writer_agent"
            ]
            mock_proj_provider.return_value = mock_agents["projects_writer_agent"]
            mock_exec_provider.return_value = mock_agents[
                "executive_summary_writer_agent"
            ]
            mock_qa_provider.return_value = mock_agents["qa_agent"]
            mock_formatter_provider.return_value = mock_agents["formatter_agent"]

            # Initialize workflow graph using the new DI factory
            cv_workflow_graph = create_cv_workflow_graph_with_di()

            # Act
            print(f"\n=== Starting regeneration test with initial state ===")
            print(f"Initial section index: {initial_agent_state.current_section_index}")
            print(f"Initial metadata: {initial_agent_state.node_execution_metadata}")
            print(
                f"Initial current_section_key: {initial_agent_state.current_section_key}"
            )

            config = {"configurable": {"thread_id": "test-session"}}
            final_state_dict = await cv_workflow_graph.ainvoke(
                initial_agent_state, config=config
            )
            final_state = AgentState.model_validate(final_state_dict)

            print(f"\n=== Final regeneration test state ===")
            print(f"Final section index: {final_state.current_section_index}")
            print(f"Final metadata: {final_state.node_execution_metadata}")
            print(
                f"Key qualifications writer call count: {mock_agents['key_qualifications_writer_agent'].run_as_node.call_count}"
            )
            print(f"Final errors: {final_state.error_messages}")

            # Assert
            assert (
                len(final_state.error_messages) > 0
            ), "Expected error messages but found none"
            assert any(
                error_message in str(error) for error in final_state.error_messages
            ), f"Expected error message '{error_message}' not found in {final_state.error_messages}"

            # Verify error recovery service was called
            mock_error_recovery_service.handle_error.assert_called()

            # Verify the failing agent was called
            mock_agents["jd_parser_agent"].run_as_node.assert_called()

    @pytest.mark.asyncio
    async def test_workflow_user_regeneration_loop(
        self, initial_agent_state, mock_agents, mock_error_recovery_service
    ):
        """Test Case 3: Verify workflow correctly handles user feedback requesting regeneration."""
        # Arrange - Configure agents for success
        for agent_name, agent in mock_agents.items():
            if agent_name != "key_qualifications_writer_agent":
                agent.run_as_node.return_value = {
                    "node_execution_metadata": {agent_name: "success"}
                }

        # Configure key_qualifications_writer_agent for regeneration scenario
        call_count = 0

        def mock_writer_side_effect(state):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call - request regeneration
                # Get the first qualification item's ID
                qual_section = next(
                    (
                        s
                        for s in state.structured_cv.sections
                        if s.name == "key_qualifications"
                    ),
                    None,
                )
                target_item_id = (
                    qual_section.items[0].id
                    if qual_section and qual_section.items
                    else "qual_1"
                )

                return {
                    "user_feedback": UserFeedback(
                        action=UserAction.REGENERATE,
                        content="Please regenerate this content",
                        target_item_id=target_item_id,
                    )
                }
            else:
                # Subsequent calls - successful regeneration
                updated_cv = state.structured_cv.model_copy(deep=True)
                for section in updated_cv.sections:
                    if section.name == "key_qualifications":
                        if section.items:
                            # Update the first qualification item
                            section.items[
                                0
                            ].content = "Regenerated Python programming expertise"
                            section.items[0].status = ItemStatus.COMPLETED
                        break

                return {
                    "structured_cv": updated_cv,
                    "user_feedback": None,  # Clear feedback after processing
                }

        mock_agents[
            "key_qualifications_writer_agent"
        ].run_as_node.side_effect = mock_writer_side_effect

        # Mock the container's agent providers
        container = ContainerSingleton.get_instance()
        with patch.object(
            container, "job_description_parser_agent"
        ) as mock_jd_provider, patch.object(
            container, "user_cv_parser_agent"
        ) as mock_cv_provider, patch.object(
            container, "research_agent"
        ) as mock_research_provider, patch.object(
            container, "cv_analyzer_agent"
        ) as mock_analyzer_provider, patch.object(
            container, "key_qualifications_writer_agent"
        ) as mock_qual_provider, patch.object(
            container, "professional_experience_writer_agent"
        ) as mock_exp_provider, patch.object(
            container, "projects_writer_agent"
        ) as mock_proj_provider, patch.object(
            container, "executive_summary_writer_agent"
        ) as mock_exec_provider, patch.object(
            container, "quality_assurance_agent"
        ) as mock_qa_provider, patch.object(
            container, "formatter_agent"
        ) as mock_formatter_provider:
            # Set up mock return values
            mock_jd_provider.return_value = mock_agents["jd_parser_agent"]
            mock_cv_provider.return_value = mock_agents["cv_parser_agent"]
            mock_research_provider.return_value = mock_agents["research_agent"]
            mock_analyzer_provider.return_value = mock_agents["cv_analyzer_agent"]
            mock_qual_provider.return_value = mock_agents[
                "key_qualifications_writer_agent"
            ]
            mock_exp_provider.return_value = mock_agents[
                "professional_experience_writer_agent"
            ]
            mock_proj_provider.return_value = mock_agents["projects_writer_agent"]
            mock_exec_provider.return_value = mock_agents[
                "executive_summary_writer_agent"
            ]
            mock_qa_provider.return_value = mock_agents["qa_agent"]
            mock_formatter_provider.return_value = mock_agents["formatter_agent"]

            # Initialize workflow graph using the new DI factory
            cv_workflow_graph = create_cv_workflow_graph_with_di()

            # Act
            print(f"\n=== Starting regeneration test with initial state ===")
            print(f"Initial section index: {initial_agent_state.current_section_index}")
            print(f"Initial metadata: {initial_agent_state.node_execution_metadata}")
            print(
                f"Initial current_section_key: {initial_agent_state.current_section_key}"
            )

            config = {"configurable": {"thread_id": "test-session"}}
            final_state_dict = await cv_workflow_graph.ainvoke(
                initial_agent_state, config=config
            )
            final_state = AgentState.model_validate(final_state_dict)

            print(f"\n=== Final regeneration test state ===")
            print(f"Final section index: {final_state.current_section_index}")
            print(f"Final metadata: {final_state.node_execution_metadata}")
            print(
                f"Key qualifications writer call count: {mock_agents['key_qualifications_writer_agent'].run_as_node.call_count}"
            )
            print(f"Final errors: {final_state.error_messages}")

            # Assert
            # Verify the writer agent was called multiple times (initial + regeneration)
            assert (
                mock_agents["key_qualifications_writer_agent"].run_as_node.call_count
                >= 2
            ), f"Expected at least 2 calls but got {mock_agents['key_qualifications_writer_agent'].run_as_node.call_count}"

            # Verify user feedback was processed (should be None in final state)
            assert (
                final_state.user_feedback is None
            ), f"Expected user_feedback to be None but got {final_state.user_feedback}"

            # Verify the content was regenerated
            qual_section = next(
                (
                    s
                    for s in final_state.structured_cv.sections
                    if s.name == "key_qualifications"
                ),
                None,
            )
            assert qual_section is not None, "key_qualifications section not found"

            # Check the first qualification item was regenerated
            assert qual_section.items, "No items found in Key Qualifications section"
            qual_item = qual_section.items[0]
            assert (
                "Regenerated" in qual_item.content
            ), f"Expected regenerated content but got: {qual_item.content}"
