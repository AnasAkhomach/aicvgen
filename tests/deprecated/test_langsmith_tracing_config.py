"""Unit tests for LangSmith tracing configuration.

This module tests the OBSERVE-01 implementation:
- LangSmith environment variables configuration
- WorkflowGraphWrapper configurable parameter with thread_id
- Proper tracing correlation using session_id as thread_id
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.container import Container
from src.orchestration.graphs.main_graph import create_cv_workflow_graph_with_di
from src.orchestration.state import GlobalState


class TestLangSmithTracingConfig:
    """Test LangSmith tracing configuration for observability."""

    def test_langsmith_environment_variables_configured(self):
        """Test that LangSmith environment variables are properly configured."""
        # Test that the required environment variables are available
        # Note: In actual usage, these would be set in .env file
        expected_vars = [
            "LANGSMITH_TRACING",
            "LANGSMITH_API_KEY",
            "LANGSMITH_PROJECT",
            "LANGSMITH_ENDPOINT",
        ]

        # Check that variables are defined in .env.example
        env_example_path = os.path.join(
            os.path.dirname(__file__), "..", "..", ".env.example"
        )
        with open(env_example_path, "r") as f:
            env_content = f.read()

        for var in expected_vars:
            assert (
                var in env_content
            ), f"Environment variable {var} not found in .env.example"

    @pytest.mark.asyncio
    async def test_workflow_graph_wrapper_configurable_parameter(self):
        """Test that WorkflowGraphWrapper passes configurable parameter with thread_id."""
        # Mock dependencies
        mock_container = MagicMock(spec=Container)
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={"test": "result"})

        session_id = "test-session-123"
        test_state = GlobalState(
            session_id=session_id,
            trace_id="test-trace-456",
            current_stage="TEST",
            workflow_status="RUNNING",
        )

        # Mock the graph creation to return our mock graph
        with patch(
            "src.orchestration.graphs.main_graph.StateGraph"
        ) as mock_state_graph:
            mock_state_graph.return_value.compile.return_value = mock_graph

            # Create workflow graph wrapper
            wrapper = create_cv_workflow_graph_with_di(mock_container)

            # Test invoke method
            await wrapper.invoke(test_state)

            # Verify that ainvoke was called with configurable parameter
            expected_config = {"configurable": {"thread_id": session_id}}
            mock_graph.ainvoke.assert_called_with(test_state, config=expected_config)

    @pytest.mark.asyncio
    async def test_workflow_graph_wrapper_trigger_workflow_step_configurable(self):
        """Test that trigger_workflow_step passes configurable parameter with thread_id."""
        # Mock dependencies
        mock_container = MagicMock(spec=Container)
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={"test": "result"})

        session_id = "test-session-789"
        test_state = GlobalState(
            session_id=session_id,
            trace_id="test-trace-101",
            current_stage="TEST",
            workflow_status="RUNNING",
        )

        # Mock the graph creation to return our mock graph
        with patch(
            "src.orchestration.graphs.main_graph.StateGraph"
        ) as mock_state_graph:
            mock_state_graph.return_value.compile.return_value = mock_graph

            # Create workflow graph wrapper
            wrapper = create_cv_workflow_graph_with_di(mock_container)

            # Test trigger_workflow_step method
            await wrapper.trigger_workflow_step(test_state)

            # Verify that ainvoke was called with configurable parameter
            expected_config = {"configurable": {"thread_id": session_id}}
            mock_graph.ainvoke.assert_called_with(test_state, config=expected_config)

    def test_session_id_as_thread_id_correlation(self):
        """Test that session_id is properly used as thread_id for tracing correlation."""
        # Mock dependencies
        mock_container = MagicMock(spec=Container)
        mock_graph = MagicMock()

        session_id = "correlation-test-session"

        # Mock the graph creation
        with patch(
            "src.orchestration.graphs.main_graph.StateGraph"
        ) as mock_state_graph:
            mock_state_graph.return_value.compile.return_value = mock_graph

            # Create workflow graph wrapper
            wrapper = create_cv_workflow_graph_with_di(mock_container)

            # Verify that the wrapper stores the session_id correctly
            assert wrapper.session_id == session_id

            # This ensures that when ainvoke is called, the session_id will be used as thread_id
            # for proper trace correlation in LangSmith

    @pytest.mark.asyncio
    async def test_langsmith_tracing_integration_mock(self):
        """Test LangSmith tracing integration with mocked environment."""
        # Mock LangSmith environment variables
        with patch.dict(
            os.environ,
            {
                "LANGSMITH_TRACING": "true",
                "LANGSMITH_API_KEY": "test-api-key",
                "LANGSMITH_PROJECT": "aicvgen-test",
                "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
            },
        ):
            # Verify environment is properly set
            assert os.getenv("LANGSMITH_TRACING") == "true"
            assert os.getenv("LANGSMITH_API_KEY") == "test-api-key"
            assert os.getenv("LANGSMITH_PROJECT") == "aicvgen-test"
            assert os.getenv("LANGSMITH_ENDPOINT") == "https://api.smith.langchain.com"

            # Mock dependencies
            mock_container = MagicMock(spec=Container)
            mock_graph = AsyncMock()
            mock_graph.ainvoke = AsyncMock(return_value={"traced": "result"})

            session_id = "tracing-test-session"
            test_state = GlobalState(
                session_id=session_id,
                trace_id="tracing-test-trace",
                current_stage="TEST",
                workflow_status="RUNNING",
            )

            # Mock the graph creation
            with patch(
                "src.orchestration.graphs.main_graph.StateGraph"
            ) as mock_state_graph:
                mock_state_graph.return_value.compile.return_value = mock_graph

                # Create and invoke workflow
                wrapper = create_cv_workflow_graph_with_di(mock_container)
                result = await wrapper.invoke(test_state)

                # Verify the call was made with proper configuration
                expected_config = {"configurable": {"thread_id": session_id}}
                mock_graph.ainvoke.assert_called_once_with(
                    test_state, config=expected_config
                )
                assert result == {"traced": "result"}
