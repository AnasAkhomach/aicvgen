"""Unit tests for the simplified execute_workflow method.

Tests the enhanced_cv_system.execute_workflow method to ensure it properly
handles AgentState inputs and rejects non-AgentState inputs.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import hashlib

from src.integration.enhanced_cv_system import EnhancedCVIntegration
from src.orchestration.state import AgentState
from src.models.data_models import JobDescriptionData, StructuredCV
from src.models.data_models import WorkflowType


class TestExecuteWorkflowSimplification:
    """Test cases for the simplified execute_workflow method."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a mock enhanced CV integration instance
        self.mock_orchestrator = Mock()
        self.mock_state_manager = Mock()
        self.mock_orchestrator.state_manager = self.mock_state_manager
        
        # Create the integration instance with mocked dependencies
        self.integration = EnhancedCVIntegration()
        self.integration._orchestrator = self.mock_orchestrator
        self.integration._intelligent_cache = Mock()
        self.integration._performance_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "requests_processed": 0,
            "total_processing_time": 0
        }
        self.integration.logger = Mock()
        
        # Create a sample AgentState for testing
        self.sample_agent_state = AgentState(
            structured_cv=StructuredCV(
                metadata={"original_cv_text": "Test CV", "start_from_scratch": False}
            ),
            job_description_data=JobDescriptionData(raw_text="Test job description")
        )

    @pytest.mark.asyncio
    async def test_execute_workflow_accepts_agent_state(self):
        """Test that execute_workflow accepts AgentState input."""
        # Arrange
        self.mock_orchestrator.initialize_workflow = AsyncMock()
        self.mock_orchestrator.execute_full_workflow = AsyncMock()
        self.mock_orchestrator.execute_full_workflow.return_value = self.sample_agent_state
        self.integration._intelligent_cache.get.return_value = None
        
        with patch.object(self.integration, '_get_performance_context'), \
             patch.object(self.integration, '_get_async_context'):
            
            # Act
            result = await self.integration.execute_workflow(
                workflow_type=WorkflowType.JOB_TAILORED_CV,
                input_data=self.sample_agent_state
            )
            
            # Assert
            assert result is not None
            self.mock_state_manager.set_structured_cv.assert_called_once_with(
                self.sample_agent_state.structured_cv
            )
            self.mock_state_manager.set_job_description_data.assert_called_once_with(
                self.sample_agent_state.job_description_data
            )
            self.mock_orchestrator.initialize_workflow.assert_called_once()
            self.mock_orchestrator.execute_full_workflow.assert_called_once()

    def test_execute_workflow_signature_only_accepts_agent_state(self):
        """Test that execute_workflow method signature only accepts AgentState."""
        # This test verifies the type hints and method signature
        import inspect
        from typing import get_type_hints
        
        # Get the method signature
        sig = inspect.signature(self.integration.execute_workflow)
        type_hints = get_type_hints(self.integration.execute_workflow)
        
        # Assert that input_data parameter expects AgentState
        assert 'input_data' in sig.parameters
        assert type_hints['input_data'] == AgentState

    @pytest.mark.asyncio
    async def test_execute_workflow_with_empty_structured_cv(self):
        """Test execute_workflow when AgentState has no structured_cv."""
        # Arrange
        agent_state_no_cv = AgentState(
            structured_cv=None,
            job_description_data=JobDescriptionData(raw_text="Test job description")
        )
        
        self.mock_orchestrator.initialize_workflow = AsyncMock()
        self.mock_orchestrator.execute_full_workflow = AsyncMock()
        self.mock_orchestrator.execute_full_workflow.return_value = agent_state_no_cv
        self.integration._intelligent_cache.get.return_value = None
        
        with patch.object(self.integration, '_get_performance_context'), \
             patch.object(self.integration, '_get_async_context'), \
             patch('src.integration.enhanced_cv_system.StructuredCV') as mock_cv_class:
            
            mock_empty_cv = Mock()
            mock_cv_class.return_value = mock_empty_cv
            
            # Act
            await self.integration.execute_workflow(
                workflow_type=WorkflowType.JOB_TAILORED_CV,
                input_data=agent_state_no_cv
            )
            
            # Assert
            mock_cv_class.assert_called_once()
            self.mock_state_manager.set_structured_cv.assert_called_once_with(mock_empty_cv)
            self.integration.logger.info.assert_any_call("Empty Structured CV data set in state manager")

    @pytest.mark.asyncio
    async def test_execute_workflow_with_missing_job_description(self):
        """Test execute_workflow when AgentState has no job_description_data."""
        # Arrange
        agent_state_no_job = AgentState(
            structured_cv=StructuredCV(),
            job_description_data=None
        )
        
        self.mock_orchestrator.initialize_workflow = AsyncMock()
        self.mock_orchestrator.execute_full_workflow = AsyncMock()
        self.mock_orchestrator.execute_full_workflow.return_value = agent_state_no_job
        self.integration._intelligent_cache.get.return_value = None
        
        with patch.object(self.integration, '_get_performance_context'), \
             patch.object(self.integration, '_get_async_context'):
            
            # Act
            await self.integration.execute_workflow(
                workflow_type=WorkflowType.JOB_TAILORED_CV,
                input_data=agent_state_no_job
            )
            
            # Assert
            self.integration.logger.warning.assert_called_with("Job description data missing in AgentState")

    @pytest.mark.asyncio
    async def test_execute_workflow_caching_with_agent_state(self):
        """Test that caching works correctly with AgentState input."""
        # Arrange
        cached_result = {"cached": True, "result": "test"}
        self.integration._intelligent_cache.get.return_value = cached_result
        
        # Act
        result = await self.integration.execute_workflow(
            workflow_type=WorkflowType.JOB_TAILORED_CV,
            input_data=self.sample_agent_state
        )
        
        # Assert
        assert result == cached_result
        assert self.integration._performance_stats["cache_hits"] == 1
        
        # Verify cache key generation uses model_dump()
        expected_cache_data = {
            "workflow_type": WorkflowType.JOB_TAILORED_CV.value,
            "input_data": self.sample_agent_state.model_dump(),
            "custom_options": {}
        }
        expected_cache_key = hashlib.md5(str(expected_cache_data).encode()).hexdigest()
        self.integration._intelligent_cache.get.assert_called_once_with(expected_cache_key)

    @pytest.mark.asyncio
    async def test_execute_workflow_performance_tracking(self):
        """Test that performance metrics are properly tracked."""
        # Arrange
        self.mock_orchestrator.initialize_workflow = AsyncMock()
        self.mock_orchestrator.execute_full_workflow = AsyncMock()
        self.mock_orchestrator.execute_full_workflow.return_value = self.sample_agent_state
        self.integration._intelligent_cache.get.return_value = None
        
        initial_requests = self.integration._performance_stats["requests_processed"]
        initial_time = self.integration._performance_stats["total_processing_time"]
        
        with patch.object(self.integration, '_get_performance_context'), \
             patch.object(self.integration, '_get_async_context'):
            
            # Act
            await self.integration.execute_workflow(
                workflow_type=WorkflowType.JOB_TAILORED_CV,
                input_data=self.sample_agent_state
            )
            
            # Assert
            assert self.integration._performance_stats["requests_processed"] == initial_requests + 1
            assert self.integration._performance_stats["total_processing_time"] > initial_time

    @pytest.mark.asyncio
    async def test_execute_workflow_logging(self):
        """Test that proper logging occurs during workflow execution."""
        # Arrange
        self.mock_orchestrator.initialize_workflow = AsyncMock()
        self.mock_orchestrator.execute_full_workflow = AsyncMock()
        self.mock_orchestrator.execute_full_workflow.return_value = self.sample_agent_state
        self.integration._intelligent_cache.get.return_value = None
        
        with patch.object(self.integration, '_get_performance_context'), \
             patch.object(self.integration, '_get_async_context'):
            
            # Act
            await self.integration.execute_workflow(
                workflow_type=WorkflowType.JOB_TAILORED_CV,
                input_data=self.sample_agent_state
            )
            
            # Assert
            self.integration.logger.info.assert_any_call(
                "Executing workflow",
                extra={
                    "workflow_type": WorkflowType.JOB_TAILORED_CV.value,
                    "session_id": None,
                    "input_data_type": "AgentState"
                }
            )
            self.integration.logger.info.assert_any_call("State manager populated from AgentState")

    @pytest.mark.asyncio
    async def test_execute_workflow_unsupported_workflow_type(self):
        """Test execute_workflow with unsupported workflow type."""
        # Arrange
        unsupported_workflow = "UNSUPPORTED_WORKFLOW"
        self.integration._intelligent_cache.get.return_value = None
        
        with patch.object(self.integration, '_get_performance_context'), \
             patch.object(self.integration, '_get_async_context'):
            
            # Act
            result = await self.integration.execute_workflow(
                workflow_type=unsupported_workflow,
                input_data=self.sample_agent_state
            )
            
            # Assert
            self.integration.logger.warning.assert_called_with(
                f"Workflow type {unsupported_workflow} not fully implemented for AgentState input or not recognized."
            )

    def test_no_dict_input_support(self):
        """Test that the method no longer supports dictionary inputs."""
        # This test verifies that the complex dict handling logic has been removed
        # by checking that the method signature only accepts AgentState
        
        import inspect
        source = inspect.getsource(self.integration.execute_workflow)
        
        # Assert that dict handling code has been removed
        assert "isinstance(input_data, dict)" not in source
        assert "job_description" not in source or "input_data[\"job_description\"]" not in source
        assert "Fallback to parsing" not in source
        assert "dictionary inputs" not in source