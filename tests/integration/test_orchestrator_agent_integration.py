"""Integration tests for Orchestrator ↔ Agents workflow.

Tests complete workflow from job description to generated CV sections,
focusing on data flow through StructuredCV state management and proper
status transitions (INITIAL → GENERATED → ACCEPTED).
"""

import unittest
import asyncio
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.enhanced_orchestrator import EnhancedOrchestrator
from src.models.data_models import (
    CVGenerationState, WorkflowStage, ProcessingStatus, ContentType,
    ContentItem, ExperienceItem, ProjectItem, QualificationItem,
    JobDescriptionData, ProcessingQueue, ProcessingMetadata
)
from src.agents.content_writer_agent import EnhancedContentWriterAgent
from src.agents.cv_analyzer_agent import CVAnalyzerAgent
from src.agents.parser_agent import ParserAgent
from src.services.progress_tracker import ProgressTracker
from src.services.error_recovery import ErrorRecoveryService
from src.services.session_manager import SessionManager


class TestOrchestratorAgentIntegration(unittest.TestCase):
    """Integration tests for orchestrator and agent interactions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock LLM client with realistic responses
        self.mock_llm_client = Mock()
        self.mock_llm_client.generate_content = AsyncMock()
        
        # Mock services
        self.mock_progress_tracker = Mock(spec=ProgressTracker)
        self.mock_error_recovery = Mock(spec=ErrorRecoveryService)
        self.mock_session_manager = Mock(spec=SessionManager)
        
        # Progress callback for tracking
        self.progress_updates = []
        self.progress_callback = lambda update: self.progress_updates.append(update)
        
        # Create orchestrator with mocked dependencies
        self.orchestrator = EnhancedOrchestrator(
            llm_client=self.mock_llm_client,
            progress_tracker=self.mock_progress_tracker,
            error_recovery=self.mock_error_recovery,
            session_manager=self.mock_session_manager,
            progress_callback=self.progress_callback
        )
        
        # Sample job description data
        self.job_description = JobDescriptionData(
            title="Senior Software Engineer",
            company="TechCorp",
            description="We are looking for a senior software engineer with Python expertise...",
            requirements=["5+ years Python", "Django/Flask", "AWS"],
            responsibilities=["Lead development", "Mentor junior developers"],
            skills=["Python", "Django", "AWS", "PostgreSQL"]
        )
        
        # Sample CV data for processing
        self.sample_cv_data = {
            "personal_info": {
                "name": "John Doe",
                "email": "john@example.com",
                "phone": "+1234567890"
            },
            "experience": [
                {
                    "title": "Software Engineer",
                    "company": "Previous Corp",
                    "duration": "2020-2023",
                    "description": "Developed web applications using Python and Django"
                }
            ],
            "projects": [
                {
                    "name": "E-commerce Platform",
                    "description": "Built scalable e-commerce platform",
                    "technologies": ["Python", "Django", "PostgreSQL"]
                }
            ]
        }

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_workflow_data_flow(self):
        """Test complete workflow from job description to generated CV sections."""
        # Setup mock responses for different content types
        self.mock_llm_client.generate_content.side_effect = [
            # Qualification generation response
            {
                "content": "Senior Software Engineer with 5+ years of Python development experience",
                "tokens_used": 50
            },
            # Experience processing response
            {
                "content": "Enhanced web application development using Python and Django frameworks",
                "tokens_used": 45
            },
            # Project processing response
            {
                "content": "Architected and developed scalable e-commerce platform serving 10k+ users",
                "tokens_used": 40
            }
        ]
        
        # Create initial state
        initial_state = CVGenerationState(
            session_id="test-session-123",
            job_description=self.job_description,
            current_stage=WorkflowStage.INITIALIZATION,
            cv_data=self.sample_cv_data
        )
        
        # Run the workflow
        result = asyncio.run(self._run_complete_workflow(initial_state))
        
        # Verify workflow completion
        self.assertIsNotNone(result)
        self.assertEqual(result.current_stage, WorkflowStage.COMPLETED)
        
        # Verify data flow through StructuredCV state management
        self.assertIsNotNone(result.processing_queue)
        self.assertTrue(len(result.processing_queue.items) > 0)
        
        # Verify status transitions for processed items
        completed_items = [
            item for item in result.processing_queue.items
            if item.metadata.status == ProcessingStatus.COMPLETED
        ]
        self.assertTrue(len(completed_items) > 0)
        
        # Verify content generation
        for item in completed_items:
            self.assertIsNotNone(item.generated_content)
            self.assertGreater(len(item.generated_content), 0)
        
        # Verify progress tracking
        self.assertTrue(len(self.progress_updates) > 0)
        self.mock_progress_tracker.update_progress.assert_called()

    async def _run_complete_workflow(self, initial_state: CVGenerationState) -> CVGenerationState:
        """Helper method to run complete workflow."""
        # Initialize workflow
        state = await self.orchestrator.initialize_workflow(
            job_description=initial_state.job_description,
            cv_data=initial_state.cv_data,
            session_id=initial_state.session_id
        )
        
        # Process through all stages
        state = await self.orchestrator.process_qualification_generation(state)
        state = await self.orchestrator.process_experience_items(state)
        state = await self.orchestrator.process_project_items(state)
        
        # Finalize workflow
        state = await self.orchestrator.finalize_workflow(state)
        
        return state

    def test_structured_cv_state_transitions(self):
        """Test proper status transitions (INITIAL → GENERATED → ACCEPTED)."""
        # Create content item with initial status
        content_item = ContentItem(
            content_type=ContentType.QUALIFICATION,
            original_content="Software Engineer with Python experience",
            metadata=ProcessingMetadata(status=ProcessingStatus.PENDING)
        )
        
        # Verify initial state
        self.assertEqual(content_item.metadata.status, ProcessingStatus.PENDING)
        
        # Simulate processing
        content_item.metadata.update_status(ProcessingStatus.IN_PROGRESS)
        self.assertEqual(content_item.metadata.status, ProcessingStatus.IN_PROGRESS)
        
        # Simulate completion
        content_item.generated_content = "Senior Software Engineer with 5+ years Python expertise"
        content_item.metadata.update_status(ProcessingStatus.COMPLETED)
        self.assertEqual(content_item.metadata.status, ProcessingStatus.COMPLETED)
        
        # Verify metadata updates
        self.assertIsNotNone(content_item.metadata.updated_at)
        self.assertGreater(content_item.metadata.updated_at, content_item.metadata.created_at)

    def test_dependency_resolution_workflow(self):
        """Test workflow handles item dependencies correctly."""
        # Create items with dependencies
        qualification_item = ContentItem(
            content_type=ContentType.QUALIFICATION,
            original_content="Base qualification",
            metadata=ProcessingMetadata(item_id="qual-1")
        )
        
        experience_item = ContentItem(
            content_type=ContentType.EXPERIENCE,
            original_content="Experience details",
            metadata=ProcessingMetadata(item_id="exp-1"),
            dependencies=["qual-1"]  # Depends on qualification
        )
        
        # Create processing queue
        queue = ProcessingQueue()
        queue.add_item(experience_item)
        queue.add_item(qualification_item)
        
        # Verify dependency ordering
        next_items = queue.get_ready_items()
        
        # Only qualification should be ready (no dependencies)
        ready_ids = [item.metadata.item_id for item in next_items]
        self.assertIn("qual-1", ready_ids)
        self.assertNotIn("exp-1", ready_ids)  # Should wait for qualification
        
        # Complete qualification
        qualification_item.metadata.update_status(ProcessingStatus.COMPLETED)
        
        # Now experience should be ready
        next_items = queue.get_ready_items()
        ready_ids = [item.metadata.item_id for item in next_items]
        self.assertIn("exp-1", ready_ids)

    def test_error_handling_in_workflow(self):
        """Test error handling and recovery in complete workflow."""
        # Setup mock to simulate error
        self.mock_llm_client.generate_content.side_effect = [
            Exception("Rate limit exceeded"),  # First call fails
            {  # Second call succeeds after retry
                "content": "Recovered content",
                "tokens_used": 30
            }
        ]
        
        # Setup error recovery mock
        self.mock_error_recovery.should_retry.return_value = True
        self.mock_error_recovery.get_retry_delay.return_value = 0.1  # Fast retry for testing
        
        # Create content item
        content_item = ContentItem(
            content_type=ContentType.QUALIFICATION,
            original_content="Test content"
        )
        
        # Process item (should handle error and retry)
        result = asyncio.run(self._process_item_with_retry(content_item))
        
        # Verify recovery
        self.assertEqual(result.metadata.status, ProcessingStatus.COMPLETED)
        self.assertEqual(result.generated_content, "Recovered content")
        self.assertEqual(result.metadata.processing_attempts, 1)  # One retry
        
        # Verify error recovery service was called
        self.mock_error_recovery.should_retry.assert_called()

    async def _process_item_with_retry(self, item: ContentItem) -> ContentItem:
        """Helper method to process item with retry logic."""
        try:
            # Simulate first attempt (will fail)
            item.metadata.update_status(ProcessingStatus.IN_PROGRESS)
            response = await self.mock_llm_client.generate_content(
                prompt=f"Process: {item.original_content}",
                content_type=item.content_type.value
            )
            
        except Exception as e:
            # Handle error and retry
            item.metadata.update_status(ProcessingStatus.FAILED, str(e))
            
            if self.mock_error_recovery.should_retry(e, item.metadata.processing_attempts):
                # Wait for retry delay
                await asyncio.sleep(self.mock_error_recovery.get_retry_delay())
                
                # Retry
                item.metadata.update_status(ProcessingStatus.IN_PROGRESS)
                response = await self.mock_llm_client.generate_content(
                    prompt=f"Process: {item.original_content}",
                    content_type=item.content_type.value
                )
            else:
                raise
        
        # Success
        item.generated_content = response["content"]
        item.metadata.tokens_used = response["tokens_used"]
        item.metadata.update_status(ProcessingStatus.COMPLETED)
        
        return item

    def test_session_state_persistence(self):
        """Test session state can be saved and restored during workflow."""
        # Create initial state
        state = CVGenerationState(
            session_id="persist-test-123",
            job_description=self.job_description,
            current_stage=WorkflowStage.EXPERIENCE_PROCESSING,
            cv_data=self.sample_cv_data
        )
        
        # Add some processed items
        content_item = ContentItem(
            content_type=ContentType.QUALIFICATION,
            original_content="Original qualification",
            generated_content="Generated qualification content",
            metadata=ProcessingMetadata(status=ProcessingStatus.COMPLETED)
        )
        
        if not state.processing_queue:
            state.processing_queue = ProcessingQueue()
        state.processing_queue.add_item(content_item)
        
        # Mock session manager save/load
        saved_state = None
        
        def mock_save_session(session_id: str, state_data: Dict[str, Any]):
            nonlocal saved_state
            saved_state = state_data
            
        def mock_load_session(session_id: str) -> Dict[str, Any]:
            return saved_state
        
        self.mock_session_manager.save_session.side_effect = mock_save_session
        self.mock_session_manager.load_session.side_effect = mock_load_session
        
        # Save session
        self.orchestrator.save_session_state(state)
        
        # Verify save was called
        self.mock_session_manager.save_session.assert_called_once()
        self.assertIsNotNone(saved_state)
        
        # Verify state data integrity
        self.assertEqual(saved_state["session_id"], "persist-test-123")
        self.assertEqual(saved_state["current_stage"], WorkflowStage.EXPERIENCE_PROCESSING.value)
        
        # Load session
        restored_state = self.orchestrator.load_session_state("persist-test-123")
        
        # Verify restoration
        self.assertEqual(restored_state.session_id, state.session_id)
        self.assertEqual(restored_state.current_stage, state.current_stage)
        self.assertEqual(restored_state.job_description.title, state.job_description.title)
        
        # Verify processing queue restoration
        self.assertIsNotNone(restored_state.processing_queue)
        restored_items = restored_state.processing_queue.items
        self.assertEqual(len(restored_items), 1)
        self.assertEqual(restored_items[0].content_type, ContentType.QUALIFICATION)
        self.assertEqual(restored_items[0].metadata.status, ProcessingStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()