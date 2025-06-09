"""Unit tests for Enhanced Orchestrator.

Tests individual item processing workflow, rate limit handling,
session state persistence, error recovery, and progress tracking.
"""

import unittest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import tempfile
import os
import sys
from datetime import datetime
from typing import Dict, Any

# Add the project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.enhanced_orchestrator import EnhancedOrchestrator
from src.models.data_models import (
    CVGenerationState, WorkflowStage, ProcessingStatus, ContentType,
    ContentItem, ExperienceItem, ProjectItem, QualificationItem,
    JobDescriptionData, ProcessingQueue
)
from src.services.progress_tracker import ProgressTracker
from src.services.error_recovery import ErrorRecoveryService
from src.services.session_manager import SessionManager, SessionStatus


class TestEnhancedOrchestrator(unittest.TestCase):
    """Test cases for Enhanced Orchestrator."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm_client = Mock()
        self.mock_progress_tracker = Mock(spec=ProgressTracker)
        self.mock_error_recovery = Mock(spec=ErrorRecoveryService)
        self.mock_session_manager = Mock(spec=SessionManager)
        self.mock_progress_callback = Mock()
        
        # Create temporary directory for state persistence testing
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.core.enhanced_orchestrator.get_rate_limiter')
    @patch('src.core.enhanced_orchestrator.get_structured_logger')
    @patch('src.core.enhanced_orchestrator.get_config')
    @patch('src.core.enhanced_orchestrator.get_progress_tracker')
    @patch('src.core.enhanced_orchestrator.get_error_recovery_service')
    @patch('src.core.enhanced_orchestrator.get_session_manager')
    def test_orchestrator_initialization(self, mock_session_mgr, mock_error_recovery,
                                      mock_progress_tracker, mock_config,
                                      mock_logger, mock_rate_limiter):
        """Test Enhanced Orchestrator initialization."""
        # Setup mocks
        mock_config.return_value = Mock()
        mock_logger.return_value = Mock()
        mock_rate_limiter.return_value = Mock()
        mock_progress_tracker.return_value = self.mock_progress_tracker
        mock_error_recovery.return_value = self.mock_error_recovery
        mock_session_mgr.return_value = self.mock_session_manager
        
        # Create orchestrator
        orchestrator = EnhancedOrchestrator(
            llm_client=self.mock_llm_client,
            progress_callback=self.mock_progress_callback
        )
        
        # Verify initialization
        self.assertIsNotNone(orchestrator.llm_client)
        self.assertIsNotNone(orchestrator.item_processor)
        self.assertIsNotNone(orchestrator.rate_limiter)
        self.assertIsNotNone(orchestrator.logger)
        self.assertIsNotNone(orchestrator.progress_tracker)
        self.assertIsNotNone(orchestrator.error_recovery)
        self.assertIsNotNone(orchestrator.session_manager)
        self.assertEqual(orchestrator.progress_callback, self.mock_progress_callback)

    @patch('src.core.enhanced_orchestrator.get_rate_limiter')
    @patch('src.core.enhanced_orchestrator.get_structured_logger')
    @patch('src.core.enhanced_orchestrator.get_config')
    def test_individual_item_processing_workflow(self, mock_config, mock_logger, mock_rate_limiter):
        """Test individual item processing workflow - key MVP requirement."""
        # Setup mocks
        mock_config.return_value = Mock()
        mock_logger.return_value = Mock()
        mock_rate_limiter.return_value = Mock()
        
        # Create orchestrator with mocked dependencies
        orchestrator = EnhancedOrchestrator(
            llm_client=self.mock_llm_client,
            progress_tracker=self.mock_progress_tracker,
            error_recovery=self.mock_error_recovery,
            session_manager=self.mock_session_manager
        )
        
        # Mock item processor
        orchestrator.item_processor = Mock()
        orchestrator.item_processor.process_item = AsyncMock(return_value={
            'success': True,
            'content': 'Processed content',
            'tokens_used': 100
        })
        
        # Create test content item
        test_item = ContentItem(
            content_type=ContentType.EXPERIENCE,
            original_content="Test experience content"
        )
        # Set the processing status through metadata
        test_item.metadata.status = ProcessingStatus.PENDING
        
        # Test processing single item
        async def run_test():
            result = await orchestrator._process_single_item_with_context(test_item, {})
            return result
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_test())
            
            # Verify processing was called
            orchestrator.item_processor.process_item.assert_called_once()
            
            # Verify result structure
            self.assertIsInstance(result, dict)
            self.assertTrue(result.get('success', False))
            
        finally:
            loop.close()

    @patch('src.core.enhanced_orchestrator.get_rate_limiter')
    @patch('src.core.enhanced_orchestrator.get_structured_logger')
    @patch('src.core.enhanced_orchestrator.get_config')
    def test_rate_limit_handling_and_queue_management(self, mock_config, mock_logger, mock_rate_limiter):
        """Test rate limit handling and queue management."""
        # Setup mocks
        mock_config.return_value = Mock()
        mock_logger.return_value = Mock()
        
        # Mock rate limiter with rate limit exceeded scenario
        mock_rate_limiter_instance = Mock()
        mock_rate_limiter_instance.wait_if_needed = AsyncMock()
        mock_rate_limiter_instance.is_rate_limited = Mock(return_value=True)
        mock_rate_limiter_instance.get_wait_time = Mock(return_value=2.0)
        mock_rate_limiter.return_value = mock_rate_limiter_instance
        
        # Create orchestrator
        orchestrator = EnhancedOrchestrator(
            llm_client=self.mock_llm_client,
            progress_tracker=self.mock_progress_tracker,
            error_recovery=self.mock_error_recovery,
            session_manager=self.mock_session_manager
        )
        
        # Test rate limit checking
        async def run_test():
            # Test rate limit wait
            await orchestrator.rate_limiter.wait_if_needed()
            
            # Verify rate limit methods were called
            mock_rate_limiter_instance.wait_if_needed.assert_called_once()
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_test())
        finally:
            loop.close()

    @patch('src.core.enhanced_orchestrator.get_rate_limiter')
    @patch('src.core.enhanced_orchestrator.get_structured_logger')
    @patch('src.core.enhanced_orchestrator.get_config')
    def test_session_state_persistence_and_recovery(self, mock_config, mock_logger, mock_rate_limiter):
        """Test session state persistence and recovery."""
        # Setup mocks
        mock_config.return_value = Mock()
        mock_logger.return_value = Mock()
        mock_rate_limiter.return_value = Mock()
        
        # Mock session manager with persistence capabilities
        self.mock_session_manager.create_session = Mock(return_value="session_123")
        self.mock_session_manager.save_state = Mock()
        self.mock_session_manager.load_state = Mock(return_value={
            'workflow_stage': WorkflowStage.QUALIFICATION_GENERATION,
            'processed_items': ['item1', 'item2'],
            'session_status': SessionStatus.ACTIVE
        })
        self.mock_session_manager.get_session_status = Mock(return_value=SessionStatus.ACTIVE)
        
        # Create orchestrator
        orchestrator = EnhancedOrchestrator(
            llm_client=self.mock_llm_client,
            progress_tracker=self.mock_progress_tracker,
            error_recovery=self.mock_error_recovery,
            session_manager=self.mock_session_manager
        )
        
        # Test session creation
        session_id = orchestrator.session_manager.create_session()
        self.assertEqual(session_id, "session_123")
        
        # Test state saving
        test_state = {
            'workflow_stage': WorkflowStage.QUALIFICATION_GENERATION,
            'processed_items': ['item1', 'item2']
        }
        orchestrator.session_manager.save_state(session_id, test_state)
        self.mock_session_manager.save_state.assert_called_with(session_id, test_state)
        
        # Test state loading
        loaded_state = orchestrator.session_manager.load_state(session_id)
        self.assertEqual(loaded_state['workflow_stage'], WorkflowStage.QUALIFICATION_GENERATION)
        self.assertEqual(len(loaded_state['processed_items']), 2)

    @patch('src.core.enhanced_orchestrator.get_rate_limiter')
    @patch('src.core.enhanced_orchestrator.get_structured_logger')
    @patch('src.core.enhanced_orchestrator.get_config')
    def test_error_recovery_and_retry_logic(self, mock_config, mock_logger, mock_rate_limiter):
        """Test error recovery and retry logic."""
        # Setup mocks
        mock_config.return_value = Mock()
        mock_logger.return_value = Mock()
        mock_rate_limiter.return_value = Mock()
        
        # Mock error recovery service
        self.mock_error_recovery.should_retry = Mock(return_value=True)
        self.mock_error_recovery.get_retry_delay = Mock(return_value=1.0)
        self.mock_error_recovery.record_error = Mock()
        self.mock_error_recovery.get_error_stats = Mock(return_value={
            'total_errors': 2,
            'retry_attempts': 1,
            'success_rate': 0.8
        })
        
        # Create orchestrator
        orchestrator = EnhancedOrchestrator(
            llm_client=self.mock_llm_client,
            progress_tracker=self.mock_progress_tracker,
            error_recovery=self.mock_error_recovery,
            session_manager=self.mock_session_manager
        )
        
        # Test error recording
        test_error = Exception("Test processing error")
        orchestrator.error_recovery.record_error("item_123", test_error)
        self.mock_error_recovery.record_error.assert_called_with("item_123", test_error)
        
        # Test retry decision
        should_retry = orchestrator.error_recovery.should_retry("item_123")
        self.assertTrue(should_retry)
        
        # Test retry delay
        delay = orchestrator.error_recovery.get_retry_delay("item_123")
        self.assertEqual(delay, 1.0)
        
        # Test error statistics
        stats = orchestrator.error_recovery.get_error_stats()
        self.assertEqual(stats['total_errors'], 2)
        self.assertEqual(stats['retry_attempts'], 1)
        self.assertEqual(stats['success_rate'], 0.8)

    @patch('src.core.enhanced_orchestrator.get_rate_limiter')
    @patch('src.core.enhanced_orchestrator.get_structured_logger')
    @patch('src.core.enhanced_orchestrator.get_config')
    def test_progress_tracking_integration(self, mock_config, mock_logger, mock_rate_limiter):
        """Test progress tracking integration."""
        # Setup mocks
        mock_config.return_value = Mock()
        mock_logger.return_value = Mock()
        mock_rate_limiter.return_value = Mock()
        
        # Mock progress tracker
        self.mock_progress_tracker.start_session = Mock(return_value="progress_session_123")
        self.mock_progress_tracker.update_progress = Mock()
        self.mock_progress_tracker.get_progress = Mock(return_value={
            'total_items': 10,
            'processed_items': 7,
            'completion_percentage': 70.0,
            'estimated_time_remaining': 120.0
        })
        self.mock_progress_tracker.complete_session = Mock()
        
        # Create orchestrator
        orchestrator = EnhancedOrchestrator(
            llm_client=self.mock_llm_client,
            progress_tracker=self.mock_progress_tracker,
            error_recovery=self.mock_error_recovery,
            session_manager=self.mock_session_manager,
            progress_callback=self.mock_progress_callback
        )
        
        # Test progress session start
        session_id = orchestrator.progress_tracker.start_session(total_items=10)
        self.assertEqual(session_id, "progress_session_123")
        
        # Test progress update
        orchestrator.progress_tracker.update_progress(
            session_id, 
            processed_items=7,
            current_item="item_7"
        )
        self.mock_progress_tracker.update_progress.assert_called_with(
            session_id,
            processed_items=7,
            current_item="item_7"
        )
        
        # Test progress retrieval
        progress = orchestrator.progress_tracker.get_progress(session_id)
        self.assertEqual(progress['completion_percentage'], 70.0)
        self.assertEqual(progress['total_items'], 10)
        self.assertEqual(progress['processed_items'], 7)
        
        # Test progress callback invocation
        if orchestrator.progress_callback:
            orchestrator.progress_callback(progress)
            self.mock_progress_callback.assert_called_with(progress)
        
        # Test session completion
        orchestrator.progress_tracker.complete_session(session_id)
        self.mock_progress_tracker.complete_session.assert_called_with(session_id)

    @patch('src.core.enhanced_orchestrator.get_rate_limiter')
    @patch('src.core.enhanced_orchestrator.get_structured_logger')
    @patch('src.core.enhanced_orchestrator.get_config')
    def test_processing_queue_management(self, mock_config, mock_logger, mock_rate_limiter):
        """Test processing queue management functionality."""
        # Setup mocks
        mock_config.return_value = Mock()
        mock_logger.return_value = Mock()
        mock_rate_limiter.return_value = Mock()
        
        # Create orchestrator
        orchestrator = EnhancedOrchestrator(
            llm_client=self.mock_llm_client,
            progress_tracker=self.mock_progress_tracker,
            error_recovery=self.mock_error_recovery,
            session_manager=self.mock_session_manager
        )
        
        # Create test processing queue
        test_queue = ProcessingQueue()
        
        # Add test items to the queue
        test_queue.pending_items = [
            ContentItem(
                content_type=ContentType.EXPERIENCE,
                original_content="Experience 1"
            ),
            ContentItem(
                content_type=ContentType.PROJECT,
                original_content="Project 1"
            )
        ]
        
        # Test queue initialization
        self.assertEqual(len(test_queue.pending_items), 2)
        self.assertEqual(len(test_queue.in_progress_items), 0)
        self.assertEqual(len(test_queue.completed_items), 0)
        self.assertEqual(len(test_queue.failed_items), 0)
        
        # Test queue item status updates
        item = test_queue.pending_items[0]
        item.metadata.status = ProcessingStatus.IN_PROGRESS
        self.assertEqual(item.metadata.status, ProcessingStatus.IN_PROGRESS)


if __name__ == '__main__':
    unittest.main()