"""Tests for CB-009 fix: LangGraph state contract violations.

This module tests the fix for contract breaches where workflow nodes
violated LangGraph's immutable state pattern by directly mutating state objects.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from src.orchestration.cv_workflow_graph import CVWorkflowGraph
from src.orchestration.state import AgentState
from src.models.cv_models import StructuredCV, JobDescriptionData
from src.models.workflow_models import UserFeedback, UserAction, ContentType
from src.models.agent_output_models import ResearchFindings, CVAnalysisResult


class TestCB009StateImmutabilityFix:
    """Test cases for CB-009 fix in state immutability."""

    @pytest.fixture
    def base_agent_state(self):
        """Create a base AgentState for testing."""
        return AgentState(
            session_id="test-session",
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            error_messages=["existing_error"],
            node_execution_metadata={"existing_key": "existing_value"}
        )

    @pytest.fixture
    def workflow_graph(self):
        """Create a workflow graph with mocked dependencies."""
        with patch.object(CVWorkflowGraph, '_build_graph'), \
             patch.object(CVWorkflowGraph, '__init__', lambda x: None):
            graph = CVWorkflowGraph()
            graph.session_id = "test-session"
            
            # Mock all agents
            graph.job_description_parser_agent = AsyncMock()
            graph.user_cv_parser_agent = AsyncMock()
            graph.research_agent = AsyncMock()
            graph.cv_analyzer_agent = AsyncMock()
            graph.key_qualifications_writer_agent = AsyncMock()
            graph.professional_experience_writer_agent = AsyncMock()
            graph.projects_writer_agent = AsyncMock()
            graph.executive_summary_writer_agent = AsyncMock()
            graph.qa_agent = AsyncMock()
            graph.formatter_agent = AsyncMock()
            
            return graph

    @pytest.mark.asyncio
    async def test_jd_parser_node_error_immutability(self, workflow_graph, base_agent_state):
        """Test that jd_parser_node doesn't mutate original error_messages list."""
        # Arrange
        original_errors = base_agent_state.error_messages.copy()
        workflow_graph.job_description_parser_agent.run_as_node.side_effect = RuntimeError("Test error")
        
        # Act
        result_state = await workflow_graph.jd_parser_node(base_agent_state)
        
        # Assert
        assert base_agent_state.error_messages == original_errors  # Original unchanged
        assert len(result_state.error_messages) == len(original_errors) + 1
        assert "JobDescriptionParserAgent failed: Test error" in result_state.error_messages
        assert base_agent_state is not result_state  # Different objects

    @pytest.mark.asyncio
    async def test_cv_parser_node_error_immutability(self, workflow_graph, base_agent_state):
        """Test that cv_parser_node doesn't mutate original error_messages list."""
        # Arrange - Remove structured_cv to force parsing
        state_without_cv = base_agent_state.model_copy(update={"structured_cv": None})
        original_errors = state_without_cv.error_messages.copy()
        workflow_graph.user_cv_parser_agent.run_as_node.side_effect = RuntimeError("Test error")
        
        # Act
        result_state = await workflow_graph.cv_parser_node(state_without_cv)
        
        # Assert
        assert state_without_cv.error_messages == original_errors  # Original unchanged
        assert len(result_state.error_messages) == len(original_errors) + 1
        assert "UserCVParserAgent failed: Test error" in result_state.error_messages
        assert state_without_cv is not result_state  # Different objects

    @pytest.mark.asyncio
    async def test_supervisor_node_metadata_immutability(self, workflow_graph, base_agent_state):
        """Test that supervisor_node doesn't mutate original node_execution_metadata dict."""
        # Arrange
        original_metadata = base_agent_state.node_execution_metadata.copy()
        
        # Act
        result_state = await workflow_graph.supervisor_node(base_agent_state)
        
        # Assert
        assert base_agent_state.node_execution_metadata == original_metadata  # Original unchanged
        assert "next_node" in result_state.node_execution_metadata
        assert "last_executed_node" in result_state.node_execution_metadata
        assert result_state.node_execution_metadata["existing_key"] == "existing_value"  # Preserved
        assert base_agent_state is not result_state  # Different objects

    @pytest.mark.asyncio
    async def test_mark_subgraph_completion_node_metadata_immutability(self, workflow_graph, base_agent_state):
        """Test that mark_subgraph_completion_node doesn't mutate original metadata."""
        # Arrange
        original_metadata = base_agent_state.node_execution_metadata.copy()
        
        # Act
        result_state = await workflow_graph.mark_subgraph_completion_node(base_agent_state)
        
        # Assert
        assert base_agent_state.node_execution_metadata == original_metadata  # Original unchanged
        assert "last_executed_node" in result_state.node_execution_metadata
        assert result_state.node_execution_metadata["existing_key"] == "existing_value"  # Preserved
        assert base_agent_state is not result_state  # Different objects

    @pytest.mark.asyncio
    async def test_entry_router_node_metadata_immutability(self, workflow_graph, base_agent_state):
        """Test that _entry_router_node doesn't mutate original metadata."""
        # Arrange
        original_metadata = base_agent_state.node_execution_metadata.copy()
        
        # Act
        result_state = await workflow_graph._entry_router_node(base_agent_state)
        
        # Assert
        assert base_agent_state.node_execution_metadata == original_metadata  # Original unchanged
        assert "entry_route" in result_state.node_execution_metadata
        assert result_state.node_execution_metadata["existing_key"] == "existing_value"  # Preserved
        assert base_agent_state is not result_state  # Different objects

    @pytest.mark.asyncio
    async def test_error_handler_node_error_immutability(self, workflow_graph, base_agent_state):
        """Test that error_handler_node doesn't mutate original error_messages when it fails."""
        # Arrange
        original_errors = base_agent_state.error_messages.copy()
        
        # Mock ErrorRecoveryService to raise an exception
        with patch('src.orchestration.cv_workflow_graph.ErrorRecoveryService') as mock_service:
            mock_service.return_value.handle_error.side_effect = Exception("Recovery failed")
            
            # Act
            result_state = await workflow_graph.error_handler_node(base_agent_state)
            
            # Assert
            assert base_agent_state.error_messages == original_errors  # Original unchanged
            assert len(result_state.error_messages) == len(original_errors) + 1
            assert "Error handler failed: Recovery failed" in result_state.error_messages
            assert base_agent_state is not result_state  # Different objects

    @pytest.mark.asyncio
    async def test_formatter_node_error_messages_immutability(self, workflow_graph, base_agent_state):
        """Test that formatter_node doesn't mutate original error_messages when handling dict result."""
        # Arrange
        original_errors = base_agent_state.error_messages.copy()
        dict_result = {
            "final_output_path": "/path/to/output.pdf",
            "error_messages": ["formatter_error_1", "formatter_error_2"]
        }
        workflow_graph.formatter_agent.run_as_node.return_value = dict_result
        
        # Act
        result_state = await workflow_graph.formatter_node(base_agent_state)
        
        # Assert
        assert base_agent_state.error_messages == original_errors  # Original unchanged
        assert len(result_state.error_messages) == len(original_errors) + 2
        assert "formatter_error_1" in result_state.error_messages
        assert "formatter_error_2" in result_state.error_messages
        assert result_state.final_output_path == "/path/to/output.pdf"
        assert base_agent_state is not result_state  # Different objects

    @pytest.mark.asyncio
    async def test_state_objects_are_never_same_reference(self, workflow_graph, base_agent_state):
        """Test that all node methods return new state objects, never the same reference."""
        # Test multiple nodes to ensure they all return new objects
        nodes_to_test = [
            workflow_graph.handle_feedback_node,
            workflow_graph.mark_subgraph_completion_node,
            workflow_graph._entry_router_node,
            workflow_graph.supervisor_node
        ]
        
        for node_method in nodes_to_test:
            result_state = await node_method(base_agent_state)
            assert result_state is not base_agent_state, f"{node_method.__name__} returned same object reference"

    def test_immutable_patterns_used_correctly(self):
        """Test that the code uses proper immutable patterns."""
        # Read the source file to verify patterns
        with open("c:\\Users\\Nitro\\Desktop\\aicvgen\\src\\orchestration\\cv_workflow_graph.py", "r") as f:
            content = f.read()
        
        # Verify that we're using spread operator for lists instead of concatenation
        assert "[*state.error_messages," in content, "Should use spread operator for error_messages"
        
        # Verify that we're using dict spread instead of .copy() and mutation
        assert "**state.node_execution_metadata," in content, "Should use dict spread for metadata"
        
        # Verify we're not using .extend() on error_messages
        assert ".extend(" not in content, "Should not use .extend() on error_messages"
        
        # Verify we're not using direct assignment after .copy()
        lines_with_copy = [line for line in content.split('\n') if '.copy()' in line and 'node_execution_metadata' in line]
        assert len(lines_with_copy) == 0, "Should not use .copy() followed by direct assignment"