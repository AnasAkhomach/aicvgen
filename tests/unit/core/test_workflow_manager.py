"""Unit tests for the async WorkflowManager.

This module contains comprehensive unit tests for the WorkflowManager class,
focusing on async method testing with proper mocking of workflow graph and file system.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import uuid
import json
from pathlib import Path
from typing import Dict, Any

from src.core.workflow_manager import WorkflowManager
from src.orchestration.graphs.main_graph import create_cv_workflow_graph_with_di
from src.orchestration.state import GlobalState
from src.models.workflow_models import WorkflowState, WorkflowStage, UserFeedback, UserAction
from src.models.cv_models import StructuredCV, JobDescriptionData
from src.agents.agent_base import AgentBase


class TestAsyncWorkflowManager:
    """Test class for async WorkflowManager functionality."""

    @pytest.fixture
    def mock_container(self):
        """Create mock dependency injection container."""
        container = MagicMock()
        
        # Mock agents
        agent_types = [
            "job_description_parser_agent",
            "user_cv_parser_agent", 
            "research_agent",
            "cv_analyzer_agent",
            "key_qualifications_writer_agent",
            "professional_experience_writer_agent",
            "projects_writer_agent",
            "executive_summary_writer_agent",
            "qa_agent",
            "formatter_agent",
        ]
        
        for agent_type in agent_types:
            mock_agent = MagicMock(spec=AgentBase)
            mock_agent.run_as_node = AsyncMock()
            container.get.return_value = mock_agent
            
        return container

    @pytest.fixture
    def mock_workflow_wrapper(self, mock_container):
        """Create a mock CompiledStateGraph."""
        # Create a mock compiled graph
        mock_compiled_graph = MagicMock()
        
        # Mock ainvoke method
        async def mock_ainvoke(global_state, config=None):
            """Mock ainvoke method."""
            # Return updated global state with generated CV text
            updated_state = global_state.copy()
            updated_state["cv_text"] = "Generated CV text"
            return updated_state
        
        mock_compiled_graph.ainvoke = mock_ainvoke
        
        # Mock the create_cv_workflow_graph_with_di function
        with patch('src.core.workflow_manager.create_cv_workflow_graph_with_di') as mock_create:
            mock_create.return_value = mock_compiled_graph
            return mock_compiled_graph
    
    @pytest.fixture
    def workflow_manager(self, mock_container):
        """Create a WorkflowManager instance with mocked dependencies."""
        with patch('src.core.workflow_manager.Path') as mock_path:
            # Mock the sessions directory
            mock_sessions_dir = MagicMock()
            mock_sessions_dir.exists.return_value = False
            mock_sessions_dir.mkdir = MagicMock()
            mock_path.return_value = mock_sessions_dir
            
            manager = WorkflowManager(container=mock_container)
            manager.sessions_dir = mock_sessions_dir
            
            yield manager
    
    def test_initialization(self, mock_container):
        """Test that WorkflowManager initializes correctly."""
        manager = WorkflowManager(container=mock_container)
        
        assert manager.container == mock_container
        assert manager.sessions_dir.name == "sessions"
        assert manager.logger is not None
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.core.workflow_manager.Path')
    def test_create_new_workflow(self, mock_path, mock_file, mock_container):
        """Test creating a new workflow with file system mocking."""
        # Setup mocks
        mock_sessions_dir = MagicMock()
        mock_sessions_dir.exists.return_value = True
        mock_sessions_dir.mkdir = MagicMock()
        mock_path.return_value = mock_sessions_dir
        
        mock_session_file = MagicMock()
        mock_session_file.exists.return_value = False
        mock_sessions_dir.__truediv__.return_value = mock_session_file
        
        manager = WorkflowManager(container=mock_container)
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
    def test_create_duplicate_workflow_raises_error(self, mock_path, mock_file, mock_container):
        """Test that creating a workflow with existing ID raises an error."""
        # Setup mocks
        mock_sessions_dir = MagicMock()
        mock_sessions_dir.exists.return_value = True
        mock_path.return_value = mock_sessions_dir
        
        mock_session_file = MagicMock()
        mock_session_file.exists.return_value = True  # File already exists
        mock_sessions_dir.__truediv__.return_value = mock_session_file
        
        manager = WorkflowManager(container=mock_container)
        manager.sessions_dir = mock_sessions_dir
        
        session_id = "duplicate-session"
        cv_text = "Sample CV content"
        jd_text = "Sample job description"
        
        # Attempt to create duplicate should raise ValueError
        with pytest.raises(ValueError, match=f"Workflow with session_id {session_id} already exists"):
            manager.create_new_workflow(cv_text, jd_text, session_id)
    
    @pytest.mark.asyncio
    async def test_trigger_workflow_step_success(self, workflow_manager, mock_workflow_wrapper):
        """Test successful async workflow step execution."""
        session_id = "test-session"
        global_state = {
            "trace_id": "test-trace",
            "structured_cv": StructuredCV().model_dump(),
            "cv_text": "Test CV content",
            "session_id": session_id,
            "error_messages": [],
            "node_execution_metadata": {},
            "workflow_status": "PROCESSING",
            "ui_display_data": {},
            "automated_mode": False
        }
        
        # Mock get_workflow_status to return existing workflow
        with patch.object(workflow_manager, 'get_workflow_status') as mock_get_status:
            mock_get_status.return_value = global_state
            
            # Mock the workflow graph creation
            with patch.object(workflow_manager, '_get_workflow_graph') as mock_get_graph:
                mock_get_graph.return_value = mock_workflow_wrapper
                
                # Mock file operations
                with patch('builtins.open', mock_open()) as mock_file:
                    # Trigger workflow step
                    result = await workflow_manager.trigger_workflow_step(session_id, global_state)
                    
                    # Verify result is a dict with expected properties
                    assert isinstance(result, dict)
                    assert result["trace_id"] == "test-trace"
                    assert result["cv_text"] == "Generated CV text"
                    mock_file.assert_called()
    
    @pytest.mark.asyncio
    async def test_trigger_workflow_step_nonexistent_workflow(self, workflow_manager):
        """Test triggering workflow step for non-existent workflow."""
        global_state = {
            "trace_id": "test-trace",
            "structured_cv": StructuredCV().model_dump(),
            "cv_text": "Test CV content",
            "session_id": "nonexistent",
            "error_messages": [],
            "node_execution_metadata": {},
            "workflow_status": "PROCESSING",
            "ui_display_data": {},
            "automated_mode": False
        }
        
        # Mock get_workflow_status to return None (no workflow found)
        with patch.object(workflow_manager, 'get_workflow_status') as mock_get_status:
            mock_get_status.return_value = None
            
            with pytest.raises(ValueError, match="No active workflow found for session_id: nonexistent"):
                await workflow_manager.trigger_workflow_step("nonexistent", global_state)
    
    @pytest.mark.asyncio
    async def test_trigger_workflow_step_with_exception(self, workflow_manager, mock_workflow_wrapper):
        """Test triggering workflow step with exception handling."""
        session_id = "test-session"
        global_state = {
            "trace_id": "test-trace",
            "structured_cv": StructuredCV().model_dump(),
            "cv_text": "Test CV content",
            "session_id": session_id,
            "error_messages": [],
            "node_execution_metadata": {},
            "workflow_status": "PROCESSING",
            "ui_display_data": {},
            "automated_mode": False
        }
        
        # Mock get_workflow_status to return existing workflow
        with patch.object(workflow_manager, 'get_workflow_status') as mock_get_status:
            mock_get_status.return_value = global_state
            
            # Create a new mock that raises exception
            async def failing_ainvoke(*args, **kwargs):
                raise Exception("Graph execution failed")
            
            mock_workflow_wrapper.ainvoke = failing_ainvoke
            
            # Mock the workflow graph creation
            with patch.object(workflow_manager, '_get_workflow_graph') as mock_get_graph:
                mock_get_graph.return_value = mock_workflow_wrapper
                
                # Mock file operations
                with patch('builtins.open', mock_open()) as mock_file:
                    # Trigger workflow step should raise RuntimeError
                    with pytest.raises(RuntimeError, match="Workflow execution failed: Graph execution failed"):
                        await workflow_manager.trigger_workflow_step(session_id, global_state)
                    
                    # Verify file write was attempted (for error state)
                    mock_file.assert_called()
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"session_id": "test-session", "cv_text": "Test CV", "structured_cv": {}, "error_messages": [], "trace_id": "test-trace", "node_execution_metadata": {}, "workflow_status": "PROCESSING", "ui_display_data": {}, "automated_mode": false}')
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
        
        assert isinstance(status, dict)
        assert status["session_id"] == session_id
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
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"session_id": "test-session", "cv_text": "Test CV", "structured_cv": {}, "error_messages": [], "trace_id": "test-trace", "node_execution_metadata": {}, "workflow_status": "PROCESSING", "ui_display_data": {}, "automated_mode": false}')
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
    
    @patch('src.core.workflow_manager.Path')
    def test_cleanup_workflow_existing(self, mock_path, workflow_manager):
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
    async def test_workflow_wrapper_mocking(self, mock_workflow_wrapper):
        """Test that the workflow wrapper is properly mocked."""
        # Test ainvoke method
        test_state = {
            "trace_id": "test-trace",
            "cv_text": "Test CV content"
        }
        
        result = await mock_workflow_wrapper.ainvoke(test_state)
        
        assert isinstance(result, dict)
        assert result["cv_text"] == "Generated CV text"
    
    @pytest.mark.asyncio
    async def test_multiple_async_operations(self, workflow_manager, mock_workflow_wrapper):
        """Test multiple async operations running concurrently."""
        session_ids = ["session-1", "session-2", "session-3"]
        global_states = [
            {
                "trace_id": f"trace-{i}",
                "structured_cv": StructuredCV().model_dump(),
                "cv_text": f"Test CV content {i}",
                "session_id": session_ids[i],
                "error_messages": [],
                "node_execution_metadata": {},
                "workflow_status": "PROCESSING",
                "ui_display_data": {},
                "automated_mode": False
            }
            for i in range(3)
        ]
        
        # Track calls to ainvoke
        call_count = 0
        original_ainvoke = mock_workflow_wrapper.ainvoke
        
        async def counting_ainvoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return await original_ainvoke(*args, **kwargs)
        
        mock_workflow_wrapper.ainvoke = counting_ainvoke
        
        # Mock get_workflow_status to return existing workflows
        with patch.object(workflow_manager, 'get_workflow_status') as mock_get_status:
            mock_get_status.side_effect = global_states
            
            # Mock the workflow graph creation
            with patch.object(workflow_manager, '_get_workflow_graph') as mock_get_graph:
                mock_get_graph.return_value = mock_workflow_wrapper
                
                # Mock file operations
                with patch('builtins.open', mock_open()):
                    # Run multiple async operations concurrently
                    tasks = [
                        workflow_manager.trigger_workflow_step(session_id, state)
                        for session_id, state in zip(session_ids, global_states)
                    ]
                    
                    results = await asyncio.gather(*tasks)
                    
                    assert len(results) == 3
                    for result in results:
                        assert isinstance(result, dict)
                        assert result["cv_text"] == "Generated CV text"
                    
                    # Verify ainvoke was called for each task
                    assert call_count == 3