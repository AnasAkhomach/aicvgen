"""Unit tests for SessionManager.

Tests session lifecycle management, state persistence,
and session data operations.
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any

from src.services.session_manager import SessionManager
from src.models.data_models import (
    StructuredCV, Section, Subsection, Item, ItemMetadata,
    JobDescriptionData, ProcessingStatus
)
from src.orchestration.state import AgentState
from src.utils.exceptions import StateManagerError


class TestSessionManager:
    """Test cases for SessionManager."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_manager(self, temp_data_dir):
        """Create a SessionManager instance for testing."""
        return SessionManager(data_dir=str(temp_data_dir))

    @pytest.fixture
    def sample_agent_state(self):
        """Create a sample AgentState for testing."""
        return AgentState(
            session_id="test_session_123",
            structured_cv=StructuredCV(
                sections=[
                    Section(
                        id="section_1",
                        title="Experience",
                        subsections=[
                            Subsection(
                                id="subsection_1",
                                title="Software Engineer",
                                items=[
                                    Item(
                                        id="item_1",
                                        type="experience",
                                        content="Developed web applications",
                                        metadata=ItemMetadata(
                                            status=ProcessingStatus.COMPLETED,
                                            created_at=datetime.now()
                                        )
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            job_description_data=JobDescriptionData(
                raw_text="Software Engineer position",
                company_name="Test Company",
                role_title="Software Engineer",
                key_requirements=["Python", "React"],
                nice_to_have=["Docker"],
                company_info="Tech startup"
            ),
            current_item_id="item_1",
            items_to_process_queue=["item_2", "item_3"],
            research_findings={"key_skills": ["Python", "JavaScript"]},
            user_feedback=None,
            error_messages=[],
            processing_complete=False
        )

    def test_session_creation(self, session_manager):
        """Test creating a new session."""
        session_id = session_manager.create_session()
        
        assert session_id is not None
        assert len(session_id) > 0
        assert session_id in session_manager._sessions
        
        # Check that session directory is created
        session_dir = session_manager.data_dir / "sessions" / session_id
        assert session_dir.exists()
        assert session_dir.is_dir()

    def test_session_state_saving_and_loading(self, session_manager, sample_agent_state):
        """Test saving and loading session state."""
        session_id = sample_agent_state.session_id
        
        # Save state
        session_manager.save_state(session_id, sample_agent_state)
        
        # Load state
        loaded_state = session_manager.load_state(session_id)
        
        assert loaded_state is not None
        assert loaded_state.session_id == session_id
        assert loaded_state.current_item_id == sample_agent_state.current_item_id
        assert len(loaded_state.structured_cv.sections) == 1
        assert loaded_state.job_description_data.company_name == "Test Company"

    def test_session_state_persistence_across_instances(self, temp_data_dir, sample_agent_state):
        """Test that session state persists across SessionManager instances."""
        session_id = sample_agent_state.session_id
        
        # Create first instance and save state
        session_manager1 = SessionManager(data_dir=str(temp_data_dir))
        session_manager1.save_state(session_id, sample_agent_state)
        
        # Create second instance and load state
        session_manager2 = SessionManager(data_dir=str(temp_data_dir))
        loaded_state = session_manager2.load_state(session_id)
        
        assert loaded_state is not None
        assert loaded_state.session_id == session_id
        assert loaded_state.current_item_id == sample_agent_state.current_item_id

    def test_session_listing(self, session_manager, sample_agent_state):
        """Test listing all sessions."""
        # Create multiple sessions
        session_id1 = session_manager.create_session()
        session_id2 = session_manager.create_session()
        
        # Save state for one session
        sample_agent_state.session_id = session_id1
        session_manager.save_state(session_id1, sample_agent_state)
        
        # List sessions
        sessions = session_manager.list_sessions()
        
        assert len(sessions) >= 2
        assert session_id1 in [s["session_id"] for s in sessions]
        assert session_id2 in [s["session_id"] for s in sessions]
        
        # Check that session with state has more details
        session1_info = next(s for s in sessions if s["session_id"] == session_id1)
        assert "last_modified" in session1_info
        assert "has_state" in session1_info
        assert session1_info["has_state"] is True

    def test_session_deletion(self, session_manager, sample_agent_state):
        """Test deleting a session."""
        session_id = sample_agent_state.session_id
        
        # Save state
        session_manager.save_state(session_id, sample_agent_state)
        
        # Verify session exists
        assert session_manager.session_exists(session_id)
        
        # Delete session
        session_manager.delete_session(session_id)
        
        # Verify session is deleted
        assert not session_manager.session_exists(session_id)
        
        # Verify session directory is removed
        session_dir = session_manager.data_dir / "sessions" / session_id
        assert not session_dir.exists()

    def test_session_cleanup_old_sessions(self, session_manager, sample_agent_state):
        """Test cleanup of old sessions."""
        # Create an old session by manually creating directory and state file
        old_session_id = "old_session_123"
        old_session_dir = session_manager.data_dir / "sessions" / old_session_id
        old_session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create old state file with old timestamp
        old_state_file = old_session_dir / "state.json"
        old_timestamp = datetime.now() - timedelta(days=8)  # 8 days old
        
        sample_agent_state.session_id = old_session_id
        state_data = sample_agent_state.model_dump()
        state_data["_timestamp"] = old_timestamp.isoformat()
        
        with open(old_state_file, 'w') as f:
            json.dump(state_data, f, default=str)
        
        # Set file modification time to old timestamp
        old_timestamp_seconds = old_timestamp.timestamp()
        old_session_dir.touch(times=(old_timestamp_seconds, old_timestamp_seconds))
        old_state_file.touch(times=(old_timestamp_seconds, old_timestamp_seconds))
        
        # Create a recent session
        recent_session_id = session_manager.create_session()
        
        # Run cleanup (assuming 7 days retention)
        cleaned_count = session_manager.cleanup_old_sessions(max_age_days=7)
        
        assert cleaned_count >= 1
        assert not session_manager.session_exists(old_session_id)
        assert session_manager.session_exists(recent_session_id)

    def test_session_exists_check(self, session_manager, sample_agent_state):
        """Test checking if a session exists."""
        session_id = sample_agent_state.session_id
        
        # Initially should not exist
        assert not session_manager.session_exists(session_id)
        
        # After saving state, should exist
        session_manager.save_state(session_id, sample_agent_state)
        assert session_manager.session_exists(session_id)

    def test_get_session_info(self, session_manager, sample_agent_state):
        """Test getting detailed session information."""
        session_id = sample_agent_state.session_id
        
        # Save state
        session_manager.save_state(session_id, sample_agent_state)
        
        # Get session info
        info = session_manager.get_session_info(session_id)
        
        assert info is not None
        assert info["session_id"] == session_id
        assert info["has_state"] is True
        assert "last_modified" in info
        assert "created_at" in info
        assert "state_size" in info
        assert info["state_size"] > 0

    def test_save_state_creates_session_directory(self, session_manager, sample_agent_state):
        """Test that saving state creates session directory if it doesn't exist."""
        session_id = sample_agent_state.session_id
        session_dir = session_manager.data_dir / "sessions" / session_id
        
        # Ensure directory doesn't exist
        assert not session_dir.exists()
        
        # Save state
        session_manager.save_state(session_id, sample_agent_state)
        
        # Directory should now exist
        assert session_dir.exists()
        assert session_dir.is_dir()

    def test_load_state_nonexistent_session(self, session_manager):
        """Test loading state for a non-existent session."""
        nonexistent_session_id = "nonexistent_session_123"
        
        loaded_state = session_manager.load_state(nonexistent_session_id)
        
        assert loaded_state is None

    def test_save_state_with_invalid_data(self, session_manager):
        """Test saving state with invalid data."""
        session_id = "test_session_123"
        
        # Try to save None state
        with pytest.raises((ValueError, TypeError)):
            session_manager.save_state(session_id, None)

    def test_concurrent_session_operations(self, session_manager, sample_agent_state):
        """Test concurrent session operations."""
        session_id = sample_agent_state.session_id
        
        async def save_and_load():
            # Save state
            session_manager.save_state(session_id, sample_agent_state)
            
            # Load state
            loaded_state = session_manager.load_state(session_id)
            return loaded_state
        
        # Run multiple concurrent operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            tasks = [save_and_load() for _ in range(5)]
            results = loop.run_until_complete(asyncio.gather(*tasks))
            
            # All operations should succeed
            assert len(results) == 5
            assert all(result is not None for result in results)
            assert all(result.session_id == session_id for result in results)
        finally:
            loop.close()

    def test_session_state_versioning(self, session_manager, sample_agent_state):
        """Test that session state includes versioning information."""
        session_id = sample_agent_state.session_id
        
        # Save state
        session_manager.save_state(session_id, sample_agent_state)
        
        # Read raw state file to check versioning
        state_file = session_manager.data_dir / "sessions" / session_id / "state.json"
        with open(state_file, 'r') as f:
            raw_data = json.load(f)
        
        # Should include timestamp
        assert "_timestamp" in raw_data
        
        # Timestamp should be recent
        timestamp = datetime.fromisoformat(raw_data["_timestamp"])
        assert (datetime.now() - timestamp).total_seconds() < 60  # Within last minute

    def test_session_data_integrity(self, session_manager, sample_agent_state):
        """Test that session data maintains integrity through save/load cycles."""
        session_id = sample_agent_state.session_id
        
        # Perform multiple save/load cycles
        for i in range(3):
            # Modify state slightly
            sample_agent_state.current_item_id = f"item_{i}"
            sample_agent_state.items_to_process_queue.append(f"new_item_{i}")
            
            # Save and load
            session_manager.save_state(session_id, sample_agent_state)
            loaded_state = session_manager.load_state(session_id)
            
            # Verify integrity
            assert loaded_state.current_item_id == f"item_{i}"
            assert f"new_item_{i}" in loaded_state.items_to_process_queue
            assert len(loaded_state.structured_cv.sections) == 1
            
            # Update sample_agent_state for next iteration
            sample_agent_state = loaded_state

    def test_session_manager_initialization_creates_directories(self, temp_data_dir):
        """Test that SessionManager initialization creates necessary directories."""
        # Remove the data directory
        if temp_data_dir.exists():
            shutil.rmtree(temp_data_dir)
        
        # Create SessionManager
        session_manager = SessionManager(data_dir=str(temp_data_dir))
        
        # Check that directories are created
        assert temp_data_dir.exists()
        assert (temp_data_dir / "sessions").exists()

    def test_get_session_statistics(self, session_manager, sample_agent_state):
        """Test getting session statistics."""
        # Create multiple sessions with different states
        for i in range(3):
            session_id = f"test_session_{i}"
            sample_agent_state.session_id = session_id
            sample_agent_state.processing_complete = (i == 2)  # Last one is complete
            session_manager.save_state(session_id, sample_agent_state)
        
        # Get statistics
        stats = session_manager.get_session_statistics()
        
        assert "total_sessions" in stats
        assert "active_sessions" in stats
        assert "completed_sessions" in stats
        assert stats["total_sessions"] >= 3
        assert stats["completed_sessions"] >= 1
        assert stats["active_sessions"] >= 2