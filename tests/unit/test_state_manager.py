import pytest
import asyncio
import os
import shutil
import uuid
from src.core.state_manager import StateManager
from src.models.data_models import StructuredCV


@pytest.mark.asyncio
async def test_state_manager_async_save_and_load(tmp_path):
    # Setup
    session_id = str(uuid.uuid4())
    test_dir = tmp_path / "data" / "sessions" / session_id
    os.makedirs(test_dir, exist_ok=True)
    cv = StructuredCV(id=session_id, sections=[], metadata={})
    manager = StateManager(session_id=session_id)
    # Save state
    state_file = await manager.save_state(cv)
    assert os.path.exists(state_file)
    # Load state
    manager2 = StateManager(session_id=session_id)
    loaded_cv = await manager2.load_state(StructuredCV)
    assert loaded_cv is not None
    assert str(loaded_cv.id) == session_id
    # Cleanup
    shutil.rmtree(tmp_path / "data")
