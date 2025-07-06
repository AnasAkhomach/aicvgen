"""Unit tests for the async WorkflowManager.

This module contains comprehensive unit tests for the WorkflowManager class,
focusing on async method testing with proper mocking of CVWorkflowGraph and file system.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import uuid
import json
from pathlib import Path
from typing import Dict, Any

from src.core.workflow_manager import WorkflowManager
from src.orchestration.cv_workflow_graph import CVWorkflowGraph
from src.orchestration.state import AgentState
from src.models.workflow_models import WorkflowState, WorkflowStage, UserFeedback, UserAction
from src.models.cv_models import StructuredCV, JobDescriptionData


class TestAsyncWorkflowManager:
    """Test class for async WorkflowManager functionality."""

    @pytest.fixture
    def mock_cv_workflow_graph(self):
        """Create a mock CVWorkflowGraph with async astream method for testing."""
        mock_graph = MagicMock(spec=CVWorkflowGraph)
        mock_graph.app = MagicMock()
        
        # Mock ainvoke to return dict (as per actual implementation)
        async def mock_ainvoke(*args, **kwargs):
            return {
                "trace_id": "test-trace",
                "structured_cv": StructuredCV().model_dump(),
                "cv_text": "Generated CV text",
                "session_id": "test-session",
                "error_messages": []
            }
        
        mock_graph.app.ainvoke = mock_ainvoke
        
        # Mock astream as an async generator as required by the task
        async def mock_astream(*args, **kwargs):
            """Mock async generator for astream method."""
            yield {"test_node": AgentState(
                structured_cv=StructuredCV(),
                cv_text="Generated CV text",
                trace_id="test-trace"
            )}
        
        mock_graph.app.astream = mock_astream
        
        # Mock trigger_workflow_step method that WorkflowManager now calls
        async def mock_trigger_workflow_step(agent_state):
            """Mock trigger_workflow_step method."""
            # Return updated agent state with generated CV text
            updated_state = AgentState(
                trace_id=agent_state.trace_id,
                structured_cv=agent_state.structured_cv,
                cv_text="Generated CV text",
                session_id=agent_state.session_id,
                error_messages=agent_state.error_messages
            )
            return updated_state
        
        mock_graph.trigger_workflow_step = mock_trigger_workflow_step
        return mock_graph
    
    @pytest.fixture
    def workflow_manager(self, mock_cv_workflow_graph):
        """Create a WorkflowManager instance with mocked dependencies."""
        with patch('src.core.workflow_manager.Path') as mock_path:
            # Mock the sessions directory
            mock_sessions_dir = MagicMock()
            mock_sessions_dir.exists.return_value = False
            mock_sessions_dir.mkdir = MagicMock()
            mock_path.return_value = mock_sessions_dir
            
            manager = WorkflowManager(cv_workflow_graph=mock_cv_workflow_graph)
            manager.sessions_dir = mock_sessions_dir
            
            yield manager
    
    def test_initialization(self, mock_cv_workflow_graph):
        """Test that WorkflowManager initializes correctly."""
        manager = WorkflowManager(cv_workflow_graph=mock_cv_workflow_graph)
        
        assert manager.cv_workflow_graph == mock_cv_workflow_graph
        assert manager.sessions_dir.name == "sessions"
        assert manager.logger is not None
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.core.workflow_manager.Path')
    def test_create_new_workflow(self, mock_path, mock_file, mock_cv_workflow_graph):
        """Test creating a new workflow with file system mocking."""
        # Setup mocks
        mock_sessions_dir = MagicMock()
        mock_sessions_dir.exists.return_value = True
        mock_sessions_dir.mkdir = MagicMock()
        mock_path.return_value = mock_sessions_dir
        
        mock_session_file = MagicMock()
        mock_session_file.exists.return_value = False
        mock_sessions_dir.__truediv__.return_value = mock_session_file
        
        manager = WorkflowManager(cv_workflow_graph=mock_cv_workflow_graph)
        manager.sessions_dir = mock_sessions_dir
        
        session_id = "test-session-123"
        cv_text = "Sample CV content"
        jd_text = "Sample job description"
        
        result = manager.create_new_workflow(cv_text, jd_text, session_id)
        
        assert result == session_id
        mock_file.assert_called_once()
        mock_sessions_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.core.workflow_manager.Path')
    def test_create_duplicate_workflow_raises_error(self, mock_path, mock_file, mock_cv_workflow_graph):
        """Test that creating a workflow with existing ID raises an error."""
        # Setup mocks
        mock_sessions_dir = MagicMock()
        mock_sessions_dir.exists.return_value = True
        mock_path.return_value = mock_sessions_dir
        
        mock_session_file = MagicMock()
        mock_session_file.exists.return_value = True  # File already exists
        mock_sessions_dir.__truediv__.return_value = mock_session_file
        
        manager = WorkflowManager(cv_workflow_graph=mock_cv_workflow_graph)
        manager.sessions_dir = mock_sessions_dir
        
        session_id = "duplicate-session"
        cv_text = "Sample CV content"
        jd_text = "Sample job description"
        
        # Attempt to create duplicate should raise ValueError
        with pytest.raises(ValueError, match=f"Workflow with session_id {session_id} already exists"):
            manager.create_new_workflow(cv_text, jd_text, session_id)
    
    @pytest.mark.asyncio
    async def test_trigger_workflow_step_success(self, workflow_manager, mock_cv_workflow_graph):
        """Test successful async workflow step execution."""
        session_id = "test-session"
        agent_state = AgentState(
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text="Test CV content"
        )
        
        # Mock get_workflow_status to return existing workflow
        with patch.object(workflow_manager, 'get_workflow_status') as mock_get_status:
            mock_get_status.return_value = agent_state
            
            # Mock file operations
            with patch('builtins.open', mock_open()) as mock_file:
                # Trigger workflow step
                result = await workflow_manager.trigger_workflow_step(session_id, agent_state)
                
                # Verify result is an AgentState with expected properties
                assert isinstance(result, AgentState)
                assert result.trace_id == "test-trace"
                assert result.cv_text == "Generated CV text"
                mock_file.assert_called()
    
    @pytest.mark.asyncio
    async def test_trigger_workflow_step_nonexistent_workflow(self, workflow_manager):
        """Test triggering workflow step for non-existent workflow."""
        agent_state = AgentState(
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text="Test CV content"
        )
        
        # Mock get_workflow_status to return None (no workflow found)
        with patch.object(workflow_manager, 'get_workflow_status') as mock_get_status:
            mock_get_status.return_value = None
            
            with pytest.raises(ValueError, match="No active workflow found for session_id: nonexistent"):
                await workflow_manager.trigger_workflow_step("nonexistent", agent_state)
    
    @pytest.mark.asyncio
    async def test_trigger_workflow_step_with_exception(self, workflow_manager, mock_cv_workflow_graph):
        """Test triggering workflow step with exception handling."""
        session_id = "test-session"
        agent_state = AgentState(
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text="Test CV content"
        )
        
        # Mock get_workflow_status to return existing workflow
        with patch.object(workflow_manager, 'get_workflow_status') as mock_get_status:
            mock_get_status.return_value = agent_state
            
            # Create a new mock that raises exception
            async def failing_trigger_workflow_step(*args, **kwargs):
                raise Exception("Graph execution failed")
            
            mock_cv_workflow_graph.trigger_workflow_step = failing_trigger_workflow_step
            
            # Mock file operations
            with patch('builtins.open', mock_open()) as mock_file:
                # Trigger workflow step should raise RuntimeError
                with pytest.raises(RuntimeError, match="Workflow execution failed: Graph execution failed"):
                    await workflow_manager.trigger_workflow_step(session_id, agent_state)
                
                # Verify file write was attempted (for error state)
                mock_file.assert_called()
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"session_id": "test-session", "cv_text": "Test CV", "structured_cv": {}, "error_messages": [], "trace_id": "test-trace"}')
    @patch('src.core.workflow_manager.Path')
    def test_get_workflow_status_existing(self, mock_path, mock_file, workflow_manager):
        """Test getting status of existing workflow with file system mocking."""
        session_id = "test-session"
        
        # Setup mocks
        mock_sessions_dir = MagicMock()
        mock_path.return_value = mock_sessions_dir
        
        mock_session_file = MagicMock()
        mock_sessions_dir.__truediv__.return_value = mock_session_file
        
        workflow_manager.sessions_dir = mock_sessions_dir
        
        # Get status
        status = workflow_manager.get_workflow_status(session_id)
        
        assert isinstance(status, AgentState)
        assert status.session_id == session_id
        mock_file.assert_called_once()
    
    @patch('builtins.open', side_effect=FileNotFoundError())
    @patch('src.core.workflow_manager.Path')
    def test_get_workflow_status_nonexistent(self, mock_path, mock_file, workflow_manager):
        """Test getting status of non-existent workflow."""
        # Setup mocks
        mock_sessions_dir = MagicMock()
        mock_path.return_value = mock_sessions_dir
        
        mock_session_file = MagicMock()
        mock_sessions_dir.__truediv__.return_value = mock_session_file
        
        workflow_manager.sessions_dir = mock_sessions_dir
        
        result = workflow_manager.get_workflow_status("nonexistent")
        assert result is None
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"session_id": "test-session", "cv_text": "Test CV", "structured_cv": {}, "error_messages": [], "trace_id": "test-trace"}')
    @patch('src.core.workflow_manager.Path')
    def test_send_feedback_existing_workflow(self, mock_path, mock_file, workflow_manager):
        """Test sending user feedback to existing workflow."""
        session_id = "test-session"
        
        # Setup mocks
        mock_sessions_dir = MagicMock()
        mock_path.return_value = mock_sessions_dir
        
        mock_session_file = MagicMock()
        mock_sessions_dir.__truediv__.return_value = mock_session_file
        
        workflow_manager.sessions_dir = mock_sessions_dir
        
        # Create user feedback
        feedback = UserFeedback(
            action=UserAction.REGENERATE,
            item_id="executive_summary",
            feedback_text="Please improve the summary"
        )
        
        # Send feedback
        result = workflow_manager.send_feedback(session_id, feedback)
        
        # Verify feedback was recorded
        assert result is True
        # Verify file operations
        assert mock_file.call_count >= 2  # One read, one write
    
    def test_send_feedback_nonexistent_workflow(self, workflow_manager):
        """Test sending user feedback to non-existent workflow."""
        feedback = UserFeedback(
            action=UserAction.REGENERATE,
            item_id="executive_summary",
            feedback_text="Please improve the summary"
        )
        
        with patch.object(workflow_manager, 'get_workflow_status') as mock_get_status:
            mock_get_status.return_value = None
            
            result = workflow_manager.send_feedback("non_existent", feedback)
            assert result is False
    
    def test_determine_next_stage_internal(self, workflow_manager):
        """Test internal next stage determination logic."""
        # Test the internal _determine_next_stage method
        next_stage = workflow_manager._determine_next_stage(WorkflowStage.INITIALIZATION)
        assert next_stage == WorkflowStage.CV_PARSING
        
        next_stage = workflow_manager._determine_next_stage(WorkflowStage.CV_PARSING)
        assert next_stage == WorkflowStage.JOB_ANALYSIS
        
        next_stage = workflow_manager._determine_next_stage(WorkflowStage.COMPLETED)
        assert next_stage is None
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"session_id": "test-session", "cv_text": "Test CV", "structured_cv": {}, "error_messages": [], "trace_id": "test-trace"}')
    @patch('src.core.workflow_manager.Path')
    def test_cleanup_workflow_existing(self, mock_path, mock_file, workflow_manager):
        """Test cleaning up existing workflow."""
        session_id = "test-session"
        
        # Setup mocks
        mock_sessions_dir = MagicMock()
        mock_path.return_value = mock_sessions_dir
        
        mock_session_file = MagicMock()
        mock_session_file.exists.return_value = True
        mock_session_file.unlink = MagicMock()
        mock_sessions_dir.__truediv__.return_value = mock_session_file
        
        workflow_manager.sessions_dir = mock_sessions_dir
        
        # Clean up workflow
        result = workflow_manager.cleanup_workflow(session_id)
        
        # Verify cleanup
        assert result is True
        mock_session_file.unlink.assert_called_once()
    
    @patch('src.core.workflow_manager.Path')
    def test_cleanup_workflow_nonexistent(self, mock_path, workflow_manager):
        """Test cleaning up non-existent workflow."""
        # Setup mocks
        mock_sessions_dir = MagicMock()
        mock_path.return_value = mock_sessions_dir
        
        mock_session_file = MagicMock()
        mock_session_file.exists.return_value = False
        mock_sessions_dir.__truediv__.return_value = mock_session_file
        
        workflow_manager.sessions_dir = mock_sessions_dir
        
        result = workflow_manager.cleanup_workflow("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_astream_method_mocking(self, mock_cv_workflow_graph):
        """Test that the astream method is properly mocked as an async generator."""
        # Verify that astream is an async generator
        astream_result = mock_cv_workflow_graph.app.astream({}, config={})
        
        # Test async iteration
        results = []
        async for step in astream_result:
            results.append(step)
        
        assert len(results) == 1
        assert "test_node" in results[0]
        assert isinstance(results[0]["test_node"], AgentState)
    
    @pytest.mark.asyncio
    async def test_multiple_async_operations(self, workflow_manager, mock_cv_workflow_graph):
        """Test multiple async operations running concurrently."""
        session_ids = ["session-1", "session-2", "session-3"]
        agent_states = [
            AgentState(
                trace_id=f"trace-{i}",
                structured_cv=StructuredCV(),
                cv_text=f"Test CV content {i}"
            )
            for i in range(3)
        ]
        
        # Track calls to trigger_workflow_step
        call_count = 0
        original_trigger_workflow_step = mock_cv_workflow_graph.trigger_workflow_step
        
        async def counting_trigger_workflow_step(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return await original_trigger_workflow_step(*args, **kwargs)
        
        mock_cv_workflow_graph.trigger_workflow_step = counting_trigger_workflow_step
        
        # Mock get_workflow_status to return existing workflows
        with patch.object(workflow_manager, 'get_workflow_status') as mock_get_status:
            mock_get_status.side_effect = agent_states
            
            # Mock file operations
            with patch('builtins.open', mock_open()):
                # Run multiple async operations concurrently
                tasks = [
                    workflow_manager.trigger_workflow_step(session_id, state)
                    for session_id, state in zip(session_ids, agent_states)
                ]
                
                results = await asyncio.gather(*tasks)
                
                assert len(results) == 3
                for result in results:
                    assert isinstance(result, AgentState)
                
                # Verify trigger_workflow_step was called for each task
                assert call_count == 3