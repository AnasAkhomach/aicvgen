"""Integration tests for State Persistence across workflow interruptions.

Tests session save/restore across workflow interruptions,
focusing on data integrity and recovery capabilities with
complete state reconstruction from saved sessions.
"""

import unittest
import asyncio
import tempfile
import shutil
import json
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.enhanced_orchestrator import EnhancedOrchestrator
from src.services.session_manager import SessionManager
from src.core.state_manager import StateManager
from src.models.data_models import (
    CVGenerationState, WorkflowStage, ProcessingStatus, ContentType,
    ContentItem, ProcessingMetadata, ProcessingQueue, JobDescriptionData,
    ExperienceItem, ProjectItem, QualificationItem, RateLimitState
)
# Define test-specific exceptions
class StateCorruptionError(Exception):
    pass

class SessionNotFoundError(Exception):
    pass


class TestStatePersistenceIntegration(unittest.TestCase):
    """Integration tests for state persistence and recovery."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.session_dir = os.path.join(self.temp_dir, "sessions")
        self.state_dir = os.path.join(self.temp_dir, "state")
        
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)
        
        # Create session and state managers
        self.session_manager = SessionManager(storage_path=Path(self.session_dir))
        self.state_manager = StateManager(session_id="test-session")
        
        # Mock LLM client
        self.mock_llm_client = Mock()
        self.mock_llm_client.generate_content = AsyncMock()
        
        # Create orchestrator with real persistence services
        self.orchestrator = EnhancedOrchestrator(
            llm_client=self.mock_llm_client,
            session_manager=self.session_manager
        )
        
        # Sample data for testing
        self.job_description = JobDescriptionData(
            raw_text="We seek a senior full stack developer with 5+ years experience in React, Node.js, and Python...",
            company_name="InnovateTech",
            position_title="Senior Full Stack Developer",
            required_skills=["React", "Node.js", "Python", "PostgreSQL", "AWS"],
            responsibilities=["Lead development", "Architecture design"],
            qualifications=["5+ years experience", "React", "Node.js", "Python"]
        )
        
        self.cv_data = {
            "personal_info": {
                "name": "Jane Smith",
                "email": "jane.smith@email.com",
                "phone": "+1-555-0123",
                "location": "San Francisco, CA"
            },
            "experience": [
                {
                    "title": "Senior Software Engineer",
                    "company": "TechStart Inc",
                    "duration": "2021-2023",
                    "description": "Led development of microservices architecture"
                },
                {
                    "title": "Software Engineer",
                    "company": "DevCorp",
                    "duration": "2019-2021",
                    "description": "Developed web applications using React and Node.js"
                }
            ],
            "projects": [
                {
                    "name": "E-commerce Platform",
                    "description": "Built scalable e-commerce platform",
                    "technologies": ["React", "Node.js", "PostgreSQL"]
                },
                {
                    "name": "Analytics Dashboard",
                    "description": "Real-time analytics dashboard",
                    "technologies": ["Python", "Django", "Redis"]
                }
            ],
            "education": [
                {
                    "degree": "BS Computer Science",
                    "institution": "Tech University",
                    "year": "2019"
                }
            ]
        }

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_workflow_state_persistence(self):
        """Test saving and restoring complete workflow state."""
        # Create initial workflow state (let orchestrator create session)
        initial_state = asyncio.run(self.orchestrator.start_cv_generation(
            job_description="Senior Full Stack Developer position at InnovateTech",
            user_data=self.cv_data
        ))
        
        session_id = initial_state.session_id
        
        # Update workflow stage to simulate some progress
        initial_state.current_stage = WorkflowStage.EXPERIENCE_PROCESSING
        
        # Save the state
        self.session_manager.update_session_state(session_id, initial_state)
        
        # Simulate workflow interruption and restart
        # Create new orchestrator instance
        new_orchestrator = EnhancedOrchestrator(
            llm_client=self.mock_llm_client,
            session_manager=self.session_manager
        )
        
        # Restore the state
        restored_state = self.session_manager.get_session_state(session_id)
        
        # Verify complete state restoration
        self.assertEqual(restored_state.session_id, initial_state.session_id)
        self.assertEqual(restored_state.current_stage, initial_state.current_stage)
        
        # Verify job description restoration
        self.assertIsNotNone(restored_state.job_description)
        self.assertEqual(restored_state.job_description.position_title, initial_state.job_description.position_title)
        self.assertEqual(restored_state.job_description.company_name, initial_state.job_description.company_name)
        if initial_state.job_description.required_skills:
            self.assertEqual(len(restored_state.job_description.required_skills), len(initial_state.job_description.required_skills))
        
        # Verify CV data restoration
        self.assertEqual(len(restored_state.key_qualifications), len(initial_state.key_qualifications))
        self.assertEqual(len(restored_state.professional_experiences), len(initial_state.professional_experiences))
        self.assertEqual(len(restored_state.side_projects), len(initial_state.side_projects))
        
        # Verify processing queues restoration
        qual_items = (restored_state.qualification_queue.pending_items + 
                     restored_state.qualification_queue.in_progress_items + 
                     restored_state.qualification_queue.completed_items + 
                     restored_state.qualification_queue.failed_items)
        exp_items = (restored_state.experience_queue.pending_items + 
                    restored_state.experience_queue.in_progress_items + 
                    restored_state.experience_queue.completed_items + 
                    restored_state.experience_queue.failed_items)
        proj_items = (restored_state.project_queue.pending_items + 
                     restored_state.project_queue.in_progress_items + 
                     restored_state.project_queue.completed_items + 
                     restored_state.project_queue.failed_items)
        
        # Verify that queues have the same number of items as the initial state
        initial_qual_items = (initial_state.qualification_queue.pending_items + 
                             initial_state.qualification_queue.in_progress_items + 
                             initial_state.qualification_queue.completed_items + 
                             initial_state.qualification_queue.failed_items)
        initial_exp_items = (initial_state.experience_queue.pending_items + 
                            initial_state.experience_queue.in_progress_items + 
                            initial_state.experience_queue.completed_items + 
                            initial_state.experience_queue.failed_items)
        initial_proj_items = (initial_state.project_queue.pending_items + 
                             initial_state.project_queue.in_progress_items + 
                             initial_state.project_queue.completed_items + 
                             initial_state.project_queue.failed_items)
        
        self.assertEqual(len(qual_items), len(initial_qual_items))
        self.assertEqual(len(exp_items), len(initial_exp_items))
        self.assertEqual(len(proj_items), len(initial_proj_items))

    def test_incremental_state_updates(self):
        """Test incremental state updates during workflow execution."""
        session_id = "incremental-updates-001"
        
        # Initialize workflow
        state = asyncio.run(self.orchestrator.start_cv_generation(
            job_description="Senior Full Stack Developer position at InnovateTech",
            user_data=self.cv_data,
            session_id=session_id
        ))
        
        # Save initial state
        self.session_manager.save_session(session_id, state.to_dict())
        
        # Simulate processing steps with incremental updates
        processing_steps = [
            {
                "stage": WorkflowStage.QUALIFICATION_GENERATION,
                "item": ContentItem(
                    content_type=ContentType.QUALIFICATION,
                    original_content="Full stack developer",
                    generated_content="Senior Full Stack Developer with expertise in modern web technologies",
                    metadata=ProcessingMetadata(item_id="qual-step1", status=ProcessingStatus.COMPLETED)
                )
            },
            {
                "stage": WorkflowStage.EXPERIENCE_PROCESSING,
                "item": ContentItem(
                    content_type=ContentType.EXPERIENCE,
                    original_content="Microservices development",
                    generated_content="Designed and implemented microservices architecture serving 1M+ users",
                    metadata=ProcessingMetadata(item_id="exp-step1", status=ProcessingStatus.COMPLETED)
                )
            },
            {
                "stage": WorkflowStage.PROJECT_PROCESSING,
                "item": ContentItem(
                    content_type=ContentType.PROJECT,
                    original_content="E-commerce platform",
                    generated_content="Architected scalable e-commerce platform with 99.9% uptime",
                    metadata=ProcessingMetadata(item_id="proj-step1", status=ProcessingStatus.COMPLETED)
                )
            }
        ]
        
        # Process each step and save incrementally
        for i, step in enumerate(processing_steps):
            state.current_stage = step["stage"]
            
            if not state.processing_queue:
                state.processing_queue = ProcessingQueue()
            
            state.processing_queue.add_item(step["item"])
            
            # Save incremental update
            self.session_manager.save_session(session_id, state.to_dict())
            
            # Verify state can be restored at each step
            restored_state = self.session_manager.load_session(session_id)
            self.assertEqual(restored_state["current_stage"], step["stage"].value)
            self.assertEqual(len(restored_state["processing_queue"]["items"]), i + 1)
        
        # Final verification
        final_state = CVGenerationState.from_dict(self.session_manager.load_session(session_id))
        self.assertEqual(final_state.current_stage, WorkflowStage.PROJECT_PROCESSING)
        self.assertEqual(len(final_state.processing_queue.items), 3)

    def test_concurrent_session_management(self):
        """Test handling multiple concurrent sessions."""
        session_ids = [f"concurrent-session-{i:03d}" for i in range(5)]
        
        # Create multiple concurrent sessions
        async def create_concurrent_sessions():
            tasks = []
            for session_id in session_ids:
                task = self.orchestrator.start_cv_generation(
                    job_description="Senior Full Stack Developer position at InnovateTech",
                    user_data=self.cv_data,
                    session_id=session_id
                )
                tasks.append(task)
            
            return await asyncio.gather(*tasks)
        
        states = asyncio.run(create_concurrent_sessions())
        
        # Save all sessions
        for state in states:
            # Add unique content to each session
            unique_item = ContentItem(
                content_type=ContentType.QUALIFICATION,
                original_content=f"Qualification for {state.session_id}",
                generated_content=f"Generated qualification for {state.session_id}",
                metadata=ProcessingMetadata(
                    item_id=f"qual-{state.session_id}",
                    status=ProcessingStatus.COMPLETED
                )
            )
            
            if not state.processing_queue:
                state.processing_queue = ProcessingQueue()
            state.processing_queue.add_item(unique_item)
            
            self.session_manager.save_session(state.session_id, state.to_dict())
        
        # Verify each session can be restored independently
        for session_id in session_ids:
            restored_state = CVGenerationState.from_dict(
                self.session_manager.load_session(session_id)
            )
            
            self.assertEqual(restored_state.session_id, session_id)
            self.assertIsNotNone(restored_state.processing_queue)
            
            # Verify unique content
            items = restored_state.processing_queue.items
            self.assertEqual(len(items), 1)
            self.assertIn(session_id, items[0].generated_content)

    def test_state_corruption_detection_and_recovery(self):
        """Test detection and handling of corrupted state data."""
        session_id = "corruption-test-001"
        
        # Create valid state
        state = asyncio.run(self.orchestrator.start_cv_generation(
            job_description=self.job_description,
            user_data=self.cv_data,
            session_id=session_id
        ))
        
        # Save valid state
        self.session_manager.update_session_state(session_id, state)
        
        # Corrupt the saved state file
        session_file = os.path.join(self.session_dir, f"{session_id}.json")
        with open(session_file, 'w') as f:
            f.write('{"corrupted": "data", "missing_required_fields":')
        
        # Attempt to load corrupted state
        with self.assertRaises((ValueError, json.JSONDecodeError, KeyError)):
            self.session_manager.get_session_state(session_id)
        
        # Test recovery with backup
        # Create backup of valid state
        backup_file = os.path.join(self.session_dir, f"{session_id}.backup.json")
        valid_state_data = state.to_dict()
        with open(backup_file, 'w') as f:
            json.dump(valid_state_data, f)
        
        # Implement backup recovery logic
        try:
            restored_state = self.session_manager.load_session(session_id)
        except Exception:
            # Fallback to backup
            if os.path.exists(backup_file):
                with open(backup_file, 'r') as f:
                    backup_data = json.load(f)
                restored_state = CVGenerationState.from_dict(backup_data)
            else:
                raise
        
        # Verify recovery
        self.assertEqual(restored_state.session_id, session_id)
        self.assertEqual(restored_state.job_description.title, self.job_description.title)

    def test_large_state_persistence_performance(self):
        """Test performance with large state objects."""
        session_id = "large-state-001"
        
        # Create state with large amount of data
        state = asyncio.run(self.orchestrator.start_cv_generation(
            job_description=self.job_description,
            user_data=self.cv_data,
            session_id=session_id
        ))
        
        # Add many content items
        if not state.processing_queue:
            state.processing_queue = ProcessingQueue()
        
        for i in range(100):  # Large number of items
            item = ContentItem(
                content_type=ContentType.EXPERIENCE,
                original_content=f"Experience item {i} with detailed description...",
                generated_content=f"Enhanced experience item {i} with comprehensive details and optimized content for job matching...",
                metadata=ProcessingMetadata(
                    item_id=f"exp-{i:03d}",
                    status=ProcessingStatus.COMPLETED,
                    tokens_used=50 + (i % 20),
                    processing_time=0.5 + (i % 10) * 0.1
                )
            )
            state.processing_queue.add_item(item)
        
        # Measure save performance
        import time
        start_time = time.time()
        self.session_manager.save_session(session_id, state.to_dict())
        save_time = time.time() - start_time
        
        # Measure load performance
        start_time = time.time()
        restored_data = self.session_manager.load_session(session_id)
        load_time = time.time() - start_time
        
        # Verify performance is reasonable (adjust thresholds as needed)
        self.assertLess(save_time, 5.0, "Save operation took too long")
        self.assertLess(load_time, 5.0, "Load operation took too long")
        
        # Verify data integrity
        restored_state = CVGenerationState.from_dict(restored_data)
        self.assertEqual(len(restored_state.processing_queue.items), 100)
        
        # Verify random samples
        items = restored_state.processing_queue.items
        self.assertEqual(items[0].metadata.item_id, "exp-000")
        self.assertEqual(items[50].metadata.item_id, "exp-050")
        self.assertEqual(items[99].metadata.item_id, "exp-099")

    def test_session_cleanup_and_expiration(self):
        """Test automatic cleanup of expired sessions."""
        # Create sessions with different timestamps
        current_time = datetime.now()
        
        sessions_data = [
            {
                "id": "recent-session",
                "timestamp": current_time - timedelta(hours=1)
            },
            {
                "id": "old-session",
                "timestamp": current_time - timedelta(days=7)
            },
            {
                "id": "expired-session",
                "timestamp": current_time - timedelta(days=30)
            }
        ]
        
        # Create and save sessions
        for session_data in sessions_data:
            state = asyncio.run(self.orchestrator.start_cv_generation(
                job_description=self.job_description,
                user_data=self.cv_data,
                session_id=session_data["id"]
            ))
            
            # Manually set timestamp
            state.created_at = session_data["timestamp"]
            state.updated_at = session_data["timestamp"]
            
            self.session_manager.save_session(session_data["id"], state.to_dict())
        
        # Verify all sessions exist
        for session_data in sessions_data:
            session_file = os.path.join(self.session_dir, f"{session_data['id']}.json")
            self.assertTrue(os.path.exists(session_file))
        
        # Implement cleanup logic (would be part of SessionManager)
        def cleanup_expired_sessions(max_age_days=14):
            cutoff_time = current_time - timedelta(days=max_age_days)
            
            for filename in os.listdir(self.session_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.session_dir, filename)
                    
                    try:
                        with open(filepath, 'r') as f:
                            session_data = json.load(f)
                        
                        updated_at = datetime.fromisoformat(session_data.get('updated_at', '1970-01-01T00:00:00'))
                        
                        if updated_at < cutoff_time:
                            os.remove(filepath)
                    
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # Remove corrupted files
                        os.remove(filepath)
        
        # Run cleanup
        cleanup_expired_sessions(max_age_days=14)
        
        # Verify cleanup results
        remaining_files = [f for f in os.listdir(self.session_dir) if f.endswith('.json')]
        
        # Recent session should remain
        self.assertIn('recent-session.json', remaining_files)
        
        # Expired session should be removed
        self.assertNotIn('expired-session.json', remaining_files)
        
        # Old session (7 days) should remain (within 14-day limit)
        self.assertIn('old-session.json', remaining_files)

    def test_cross_platform_state_compatibility(self):
        """Test state files are compatible across different platforms."""
        session_id = "cross-platform-001"
        
        # Create state with various data types
        state = asyncio.run(self.orchestrator.start_cv_generation(
            job_description=self.job_description,
            user_data=self.cv_data,
            session_id=session_id
        ))
        
        # Add content with special characters and unicode
        special_item = ContentItem(
            content_type=ContentType.QUALIFICATION,
            original_content="Développeur Senior avec expérience en IA/ML 🚀",
            generated_content="Senior Developer with AI/ML expertise and innovative solutions 💡",
            metadata=ProcessingMetadata(
                item_id="special-chars-001",
                status=ProcessingStatus.COMPLETED
            )
        )
        
        if not state.processing_queue:
            state.processing_queue = ProcessingQueue()
        state.processing_queue.add_item(special_item)
        
        # Save state
        self.session_manager.save_session(session_id, state.to_dict())
        
        # Read raw file content
        session_file = os.path.join(self.session_dir, f"{session_id}.json")
        with open(session_file, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        # Verify unicode characters are properly encoded
        self.assertIn('🚀', raw_content)
        self.assertIn('💡', raw_content)
        self.assertIn('Développeur', raw_content)
        
        # Verify state can be restored
        restored_state = CVGenerationState.from_dict(
            self.session_manager.load_session(session_id)
        )
        
        restored_item = restored_state.processing_queue.items[0]
        self.assertEqual(restored_item.original_content, "Développeur Senior avec expérience en IA/ML 🚀")
        self.assertEqual(restored_item.generated_content, "Senior Developer with AI/ML expertise and innovative solutions 💡")


if __name__ == "__main__":
    unittest.main()