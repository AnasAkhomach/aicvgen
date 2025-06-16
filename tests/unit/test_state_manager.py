"""Unit tests for StateManager.

Tests state management, persistence, validation,
and state transition functionality.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, Any

from src.services.state_manager import StateManager
from src.orchestration.state import AgentState
from src.models.data_models import (
    StructuredCV, Section, Subsection, Item, ItemMetadata,
    JobDescriptionData, ProcessingStatus
)
from src.utils.exceptions import StateManagerError


class TestStateManager:
    """Test cases for StateManager."""

    @pytest.fixture
    def temp_state_dir(self):
        """Create a temporary directory for state storage."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def state_manager(self, temp_state_dir):
        """Create a StateManager instance for testing."""
        return StateManager(state_dir=str(temp_state_dir))

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

    def test_state_manager_initialization(self, temp_state_dir):
        """Test StateManager initialization."""
        state_manager = StateManager(state_dir=str(temp_state_dir))
        
        assert state_manager.state_dir == Path(temp_state_dir)
        assert hasattr(state_manager, '_current_state')
        assert hasattr(state_manager, '_state_history')
        assert hasattr(state_manager, '_state_lock')

    @pytest.mark.asyncio
    async def test_save_state_success(self, state_manager, sample_agent_state):
        """Test successful state saving."""
        session_id = sample_agent_state.session_id
        
        # Save state
        await state_manager.save_state(session_id, sample_agent_state)
        
        # Verify state file exists
        state_file = state_manager.state_dir / f"{session_id}_state.json"
        assert state_file.exists()
        
        # Verify current state is updated
        assert state_manager._current_state[session_id] == sample_agent_state

    @pytest.mark.asyncio
    async def test_load_state_success(self, state_manager, sample_agent_state):
        """Test successful state loading."""
        session_id = sample_agent_state.session_id
        
        # Save state first
        await state_manager.save_state(session_id, sample_agent_state)
        
        # Clear current state
        state_manager._current_state.clear()
        
        # Load state
        loaded_state = await state_manager.load_state(session_id)
        
        assert loaded_state is not None
        assert loaded_state.session_id == session_id
        assert loaded_state.current_item_id == sample_agent_state.current_item_id
        assert len(loaded_state.structured_cv.sections) == 1

    @pytest.mark.asyncio
    async def test_load_state_not_found(self, state_manager):
        """Test loading state for non-existent session."""
        nonexistent_session_id = "nonexistent_session_123"
        
        loaded_state = await state_manager.load_state(nonexistent_session_id)
        
        assert loaded_state is None

    @pytest.mark.asyncio
    async def test_get_current_state(self, state_manager, sample_agent_state):
        """Test getting current state."""
        session_id = sample_agent_state.session_id
        
        # Initially should be None
        current_state = await state_manager.get_current_state(session_id)
        assert current_state is None
        
        # After saving, should return the state
        await state_manager.save_state(session_id, sample_agent_state)
        current_state = await state_manager.get_current_state(session_id)
        
        assert current_state is not None
        assert current_state.session_id == session_id

    @pytest.mark.asyncio
    async def test_update_state_field(self, state_manager, sample_agent_state):
        """Test updating specific state fields."""
        session_id = sample_agent_state.session_id
        
        # Save initial state
        await state_manager.save_state(session_id, sample_agent_state)
        
        # Update specific field
        new_item_id = "new_item_123"
        await state_manager.update_state_field(session_id, "current_item_id", new_item_id)
        
        # Verify update
        current_state = await state_manager.get_current_state(session_id)
        assert current_state.current_item_id == new_item_id

    @pytest.mark.asyncio
    async def test_update_state_field_invalid_session(self, state_manager):
        """Test updating state field for invalid session."""
        with pytest.raises(StateManagerError, match="Session not found"):
            await state_manager.update_state_field("invalid_session", "current_item_id", "new_value")

    @pytest.mark.asyncio
    async def test_update_state_field_invalid_field(self, state_manager, sample_agent_state):
        """Test updating invalid state field."""
        session_id = sample_agent_state.session_id
        
        # Save initial state
        await state_manager.save_state(session_id, sample_agent_state)
        
        # Try to update invalid field
        with pytest.raises(StateManagerError, match="Invalid field"):
            await state_manager.update_state_field(session_id, "invalid_field", "new_value")

    @pytest.mark.asyncio
    async def test_add_error_message(self, state_manager, sample_agent_state):
        """Test adding error messages to state."""
        session_id = sample_agent_state.session_id
        
        # Save initial state
        await state_manager.save_state(session_id, sample_agent_state)
        
        # Add error message
        error_message = "Test error occurred"
        await state_manager.add_error_message(session_id, error_message)
        
        # Verify error message is added
        current_state = await state_manager.get_current_state(session_id)
        assert error_message in current_state.error_messages

    @pytest.mark.asyncio
    async def test_clear_error_messages(self, state_manager, sample_agent_state):
        """Test clearing error messages from state."""
        session_id = sample_agent_state.session_id
        
        # Add some error messages first
        sample_agent_state.error_messages = ["Error 1", "Error 2"]
        await state_manager.save_state(session_id, sample_agent_state)
        
        # Clear error messages
        await state_manager.clear_error_messages(session_id)
        
        # Verify error messages are cleared
        current_state = await state_manager.get_current_state(session_id)
        assert len(current_state.error_messages) == 0

    @pytest.mark.asyncio
    async def test_update_processing_queue(self, state_manager, sample_agent_state):
        """Test updating processing queue."""
        session_id = sample_agent_state.session_id
        
        # Save initial state
        await state_manager.save_state(session_id, sample_agent_state)
        
        # Update processing queue
        new_queue = ["item_4", "item_5", "item_6"]
        await state_manager.update_processing_queue(session_id, new_queue)
        
        # Verify queue is updated
        current_state = await state_manager.get_current_state(session_id)
        assert current_state.items_to_process_queue == new_queue

    @pytest.mark.asyncio
    async def test_add_to_processing_queue(self, state_manager, sample_agent_state):
        """Test adding items to processing queue."""
        session_id = sample_agent_state.session_id
        
        # Save initial state
        await state_manager.save_state(session_id, sample_agent_state)
        
        # Add item to queue
        new_item = "item_4"
        await state_manager.add_to_processing_queue(session_id, new_item)
        
        # Verify item is added
        current_state = await state_manager.get_current_state(session_id)
        assert new_item in current_state.items_to_process_queue
        assert current_state.items_to_process_queue[-1] == new_item

    @pytest.mark.asyncio
    async def test_remove_from_processing_queue(self, state_manager, sample_agent_state):
        """Test removing items from processing queue."""
        session_id = sample_agent_state.session_id
        
        # Save initial state
        await state_manager.save_state(session_id, sample_agent_state)
        
        # Remove item from queue
        item_to_remove = "item_2"
        await state_manager.remove_from_processing_queue(session_id, item_to_remove)
        
        # Verify item is removed
        current_state = await state_manager.get_current_state(session_id)
        assert item_to_remove not in current_state.items_to_process_queue

    @pytest.mark.asyncio
    async def test_mark_processing_complete(self, state_manager, sample_agent_state):
        """Test marking processing as complete."""
        session_id = sample_agent_state.session_id
        
        # Save initial state
        await state_manager.save_state(session_id, sample_agent_state)
        
        # Mark processing complete
        await state_manager.mark_processing_complete(session_id)
        
        # Verify processing is marked complete
        current_state = await state_manager.get_current_state(session_id)
        assert current_state.processing_complete is True

    @pytest.mark.asyncio
    async def test_state_validation(self, state_manager, sample_agent_state):
        """Test state validation."""
        session_id = sample_agent_state.session_id
        
        # Valid state should pass validation
        is_valid, errors = await state_manager.validate_state(sample_agent_state)
        assert is_valid is True
        assert len(errors) == 0
        
        # Invalid state should fail validation
        invalid_state = sample_agent_state.model_copy()
        invalid_state.session_id = ""  # Empty session ID
        
        is_valid, errors = await state_manager.validate_state(invalid_state)
        assert is_valid is False
        assert len(errors) > 0

    @pytest.mark.asyncio
    async def test_state_history_tracking(self, state_manager, sample_agent_state):
        """Test state history tracking."""
        session_id = sample_agent_state.session_id
        
        # Save initial state
        await state_manager.save_state(session_id, sample_agent_state)
        
        # Update state multiple times
        for i in range(3):
            await state_manager.update_state_field(session_id, "current_item_id", f"item_{i}")
        
        # Get state history
        history = await state_manager.get_state_history(session_id)
        
        assert len(history) >= 3  # At least 3 state changes
        
        # Verify history contains different states
        item_ids = [state.current_item_id for state in history]
        assert "item_0" in item_ids
        assert "item_1" in item_ids
        assert "item_2" in item_ids

    @pytest.mark.asyncio
    async def test_rollback_state(self, state_manager, sample_agent_state):
        """Test rolling back to previous state."""
        session_id = sample_agent_state.session_id
        
        # Save initial state
        original_item_id = sample_agent_state.current_item_id
        await state_manager.save_state(session_id, sample_agent_state)
        
        # Update state
        new_item_id = "new_item_123"
        await state_manager.update_state_field(session_id, "current_item_id", new_item_id)
        
        # Verify state is updated
        current_state = await state_manager.get_current_state(session_id)
        assert current_state.current_item_id == new_item_id
        
        # Rollback to previous state
        await state_manager.rollback_state(session_id, steps=1)
        
        # Verify state is rolled back
        current_state = await state_manager.get_current_state(session_id)
        assert current_state.current_item_id == original_item_id

    @pytest.mark.asyncio
    async def test_concurrent_state_operations(self, state_manager, sample_agent_state):
        """Test concurrent state operations."""
        session_id = sample_agent_state.session_id
        
        # Save initial state
        await state_manager.save_state(session_id, sample_agent_state)
        
        async def update_state(field_value):
            await state_manager.update_state_field(session_id, "current_item_id", f"item_{field_value}")
        
        # Run concurrent updates
        tasks = [update_state(i) for i in range(10)]
        await asyncio.gather(*tasks)
        
        # Verify final state is consistent
        current_state = await state_manager.get_current_state(session_id)
        assert current_state.current_item_id.startswith("item_")

    @pytest.mark.asyncio
    async def test_state_persistence_across_instances(self, temp_state_dir, sample_agent_state):
        """Test state persistence across StateManager instances."""
        session_id = sample_agent_state.session_id
        
        # Create first instance and save state
        state_manager1 = StateManager(state_dir=str(temp_state_dir))
        await state_manager1.save_state(session_id, sample_agent_state)
        
        # Create second instance and load state
        state_manager2 = StateManager(state_dir=str(temp_state_dir))
        loaded_state = await state_manager2.load_state(session_id)
        
        assert loaded_state is not None
        assert loaded_state.session_id == session_id
        assert loaded_state.current_item_id == sample_agent_state.current_item_id

    @pytest.mark.asyncio
    async def test_cleanup_old_states(self, state_manager, sample_agent_state):
        """Test cleanup of old state files."""
        # Create multiple sessions
        session_ids = [f"session_{i}" for i in range(5)]
        
        for session_id in session_ids:
            state = sample_agent_state.model_copy()
            state.session_id = session_id
            await state_manager.save_state(session_id, state)
        
        # Cleanup old states (keep only 3 most recent)
        cleaned_count = await state_manager.cleanup_old_states(max_states=3)
        
        assert cleaned_count >= 2  # Should have cleaned at least 2 states
        
        # Verify remaining states
        remaining_files = list(state_manager.state_dir.glob("*_state.json"))
        assert len(remaining_files) <= 3

    @pytest.mark.asyncio
    async def test_get_state_summary(self, state_manager, sample_agent_state):
        """Test getting state summary."""
        session_id = sample_agent_state.session_id
        
        # Save state
        await state_manager.save_state(session_id, sample_agent_state)
        
        # Get summary
        summary = await state_manager.get_state_summary(session_id)
        
        assert summary is not None
        assert "session_id" in summary
        assert "current_item_id" in summary
        assert "processing_complete" in summary
        assert "total_items" in summary
        assert "completed_items" in summary
        assert "error_count" in summary

    @pytest.mark.asyncio
    async def test_export_state(self, state_manager, sample_agent_state, temp_state_dir):
        """Test exporting state to file."""
        session_id = sample_agent_state.session_id
        
        # Save state
        await state_manager.save_state(session_id, sample_agent_state)
        
        # Export state
        export_path = temp_state_dir / "exported_state.json"
        await state_manager.export_state(session_id, str(export_path))
        
        # Verify export file exists and contains data
        assert export_path.exists()
        
        import json
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        assert exported_data["session_id"] == session_id
        assert "structured_cv" in exported_data
        assert "job_description_data" in exported_data

    @pytest.mark.asyncio
    async def test_import_state(self, state_manager, sample_agent_state, temp_state_dir):
        """Test importing state from file."""
        session_id = sample_agent_state.session_id
        
        # Export state first
        await state_manager.save_state(session_id, sample_agent_state)
        export_path = temp_state_dir / "exported_state.json"
        await state_manager.export_state(session_id, str(export_path))
        
        # Clear current state
        state_manager._current_state.clear()
        
        # Import state
        imported_session_id = await state_manager.import_state(str(export_path))
        
        assert imported_session_id == session_id
        
        # Verify imported state
        current_state = await state_manager.get_current_state(session_id)
        assert current_state is not None
        assert current_state.session_id == session_id
        assert current_state.current_item_id == sample_agent_state.current_item_id