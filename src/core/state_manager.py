from typing import Optional
import json
import uuid
import os
import time
import asyncio
from datetime import datetime
from ..models.data_models import StructuredCV
from ..config.logging_config import get_structured_logger

logger = get_structured_logger(__name__)


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):  # Use 'obj' for compatibility with base class
        import enum

        if isinstance(obj, enum.Enum):
            return obj.value
        return super().default(obj)


class StateManager:
    """
    Persistence layer for StructuredCV state. Handles only save/load operations.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.__structured_cv = None
        self._last_save_time = None
        self._state_changes = []
        logger.info("Initialized StateManager with session ID: %s", self.session_id)

    def set_structured_cv(self, structured_cv):
        self.__structured_cv = structured_cv
        logger.info(
            "StructuredCV set with session ID: %s",
            structured_cv.id if structured_cv else "None",
        )

    def get_structured_cv(self):
        return self.__structured_cv

    async def load_state(self, session_id=None):
        """
        Asynchronously load a StructuredCV from a saved state using asyncio.to_thread.
        """
        try:
            start_time = time.time()
            session_id = session_id or self.session_id
            state_file = f"data/sessions/{session_id}/state.json"
            log_file = f"data/sessions/{session_id}/state_changes.json"

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
            self.__structured_cv = StructuredCV.from_dict(data)
            self._state_changes = state_changes
            duration = time.time() - start_time
            logger.info("State loaded from %s in %.2fs", state_file, duration)
            return self.__structured_cv
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load state: %s", str(e))
            return None

    async def save_state(self):
        """
        Asynchronously save the current StructuredCV state using asyncio.to_thread.
        """
        try:
            start_time = time.time()
            structured_cv = self.get_structured_cv()
            if not structured_cv:
                logger.warning("No CV data to save")
                return None

            os.makedirs(f"data/sessions/{structured_cv.id}", exist_ok=True)
            state_file = f"data/sessions/{structured_cv.id}/state.json"
            log_file = f"data/sessions/{structured_cv.id}/state_changes.json"

            def blocking_io():
                with open(state_file, "w", encoding="utf-8") as f:
                    json.dump(structured_cv.to_dict(), f, indent=2, cls=EnumEncoder)
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(self._state_changes, f, indent=2)

            await asyncio.to_thread(blocking_io)
            duration = time.time() - start_time
            self._last_save_time = datetime.now().isoformat()
            logger.info("State saved to %s in %.2fs", state_file, duration)
            return state_file
        except (OSError, TypeError) as e:
            logger.error("Failed to save state: %s", str(e))
            return None


# State Management Layer Documentation
#
# State Management Layers in AI CV Generator:
#
# 1. Streamlit session_state (UI Layer):
#    - Holds only raw user input and UI flags (e.g., text areas, checkboxes, API key, progress).
#    - Should NOT store complex objects like StructuredCV or AgentState.
#    - See: src/frontend/state_helpers.py
#
# 2. AgentState (Workflow Layer):
#    - The single source of truth for workflow execution.
#    - Created from UI input at workflow start, destroyed or archived at end.
#    - All agents read/write only to this object.
#    - See: src/orchestration/state.py
#
# 3. StateManager (Persistence Layer):
#    - Used only to save/load AgentState or StructuredCV at workflow boundaries.
#    - No agent or service should call StateManager directly during workflow execution.
#    - See: src/core/state_manager.py
#
# For details, see TASK_BLUEPRINT.md section 12.
