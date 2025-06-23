from typing import Optional, Type, TypeVar, Generic
import json
import uuid
import os
import time
import asyncio
from datetime import datetime
from pydantic import BaseModel
from ..config.logging_config import get_structured_logger
from uuid import UUID

logger = get_structured_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        import enum

        if isinstance(obj, enum.Enum):
            return obj.value
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


class StateManager(Generic[T]):
    """
    Generic persistence layer for workflow state objects (e.g., AgentState, StructuredCV).
    Handles only save/load operations at workflow boundaries.

    Usage:
        manager = StateManager(session_id)
        await manager.save_state(state_obj)
        state = await manager.load_state(StateClass)
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self._last_save_time = None
        self._state_changes = []
        logger.info("Initialized StateManager with session ID: %s", self.session_id)

    async def load_state(self, model_class: Type[T]) -> Optional[T]:
        """
        Asynchronously load a state object (Pydantic model) from a saved state using asyncio.to_thread.
        """
        try:
            start_time = time.time()
            state_file = f"data/sessions/{self.session_id}/state.json"
            log_file = f"data/sessions/{self.session_id}/state_changes.json"

            def blocking_io():
                if not os.path.exists(state_file):
                    return None, []
                with open(state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                state_changes = []
                if os.path.exists(log_file):
                    with open(log_file, "r", encoding="utf-8") as f:
                        state_changes = json.load(f)
                return data, state_changes

            data, state_changes = await asyncio.to_thread(blocking_io)
            if data is None:
                logger.warning("State file not found: %s", state_file)
                return None
            state_obj = model_class.model_validate(data)
            self._state_changes = state_changes
            duration = time.time() - start_time
            logger.info("State loaded from %s in %.2fs", state_file, duration)
            return state_obj
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load state: %s", str(e))
            return None

    async def save_state(self, state_obj: BaseModel):
        """
        Asynchronously save the current state object (Pydantic model) using asyncio.to_thread.
        """
        try:
            start_time = time.time()
            if not state_obj:
                logger.warning("No state data to save")
                return None
            os.makedirs(f"data/sessions/{self.session_id}", exist_ok=True)
            state_file = f"data/sessions/{self.session_id}/state.json"
            log_file = f"data/sessions/{self.session_id}/state_changes.json"

            def blocking_io():
                with open(state_file, "w", encoding="utf-8") as f:
                    json.dump(state_obj.model_dump(), f, indent=2, cls=EnumEncoder)
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(self._state_changes, f, indent=2)

            await asyncio.to_thread(blocking_io)
            duration = time.time() - start_time
            self._last_save_time = datetime.now().isoformat()
            logger.info("State saved to %s in %.2fs", state_file, duration)
            return state_file
        except (OSError, TypeError) as e:
            logger.error(f"Failed to save state: {str(e)}")
            return None

    def log_state_change(self, change: dict):
        """Append a state change event to the log."""
        self._state_changes.append({"timestamp": datetime.now().isoformat(), **change})


# State Management Layer Documentation
#
# 1. Streamlit session_state (UI Layer):
#    - Holds only raw user input and UI flags (e.g., text areas, checkboxes, API key, progress).
#    - Should NOT store complex objects like StructuredCV or AgentState.
#    - See: src/frontend/state_helpers.py
# 2. AgentState (Workflow Layer):
#    - The single source of truth for workflow execution.
#    - Created from UI input at workflow start, destroyed or archived at end.
#    - All agents read/write only to this object.
#    - See: src/orchestration/state.py
# 3. StateManager (Persistence Layer):
#    - Used only to save/load AgentState or StructuredCV at workflow boundaries.
#    - No agent or service should call StateManager directly during workflow execution.
#    - See: src/core/state_manager.py
#
# For details, see TASK_BLUEPRINT.md section 12.
