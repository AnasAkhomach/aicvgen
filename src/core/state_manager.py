from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import enum
import json
import uuid
import os
import time
from datetime import datetime

# Import standardized Pydantic models
from ..models.data_models import (
    JobDescriptionData,
    StructuredCV,
    Section,
    Subsection,
    Item,
    ItemStatus,
    ItemType,
    ContentData,
    AgentIO,
    VectorStoreConfig,
)
from ..config.logging_config import get_structured_logger

logger = get_structured_logger(__name__)


# VectorStoreConfig is now imported from standardized data models


# ContentPiece is now imported from standardized data models

# CVData is now imported from standardized data models

# SkillEntry and ExperienceEntry are now imported from standardized data models

# ContentData is now imported from standardized data models

# WorkflowState is now imported from standardized data models

# ItemStatus and ItemType are now imported from standardized data models

# Item is now imported from standardized data models

# Subsection is now imported from standardized data models

# Section is now imported from standardized data models


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, enum.Enum):
            return obj.value
        return super().default(obj)


# StructuredCV is now imported from standardized data models

# All StructuredCV methods are now part of the imported class from data_models.py

# Orphaned methods removed - functionality now handled by imported StructuredCV class

# Additional orphaned methods removed - functionality now handled by imported StructuredCV class

# All orphaned methods removed - functionality now handled by imported StructuredCV class


class StateManager:
    """
    Manages the state of the CV tailoring process.
    """

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize the StateManager.

        Args:
            session_id: Optional ID for the session. If not provided, a new one will be generated.
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.__structured_cv = None  # Private attribute using name mangling
        self._last_save_time = None
        self._state_changes = []  # Track state transitions
        logger.info("Initialized StateManager with session ID: %s", session_id)

    def create_new_cv(self, metadata=None):
        """
        Create a new StructuredCV.

        Args:
            metadata: Optional metadata for the CV.

        Returns:
            The new StructuredCV instance.
        """
        self.__structured_cv = StructuredCV(id=self.session_id, metadata=metadata or {})
        return self.__structured_cv

    def load_state(self, session_id=None):
        """
        Load a StructuredCV from a saved state.

        Args:
            session_id: The ID of the session to load. If not provided, uses the instance's session_id.

        Returns:
            The loaded StructuredCV instance, or None if loading failed.
        """
        try:
            start_time = time.time()
            session_id = session_id or self.session_id
            state_file = f"data/sessions/{session_id}/state.json"
            if not os.path.exists(state_file):
                logger.warning("State file not found: %s", state_file)
                return None

            with open(state_file, "r") as f:
                data = json.load(f)
                self.__structured_cv = StructuredCV.from_dict(data)

            log_file = f"data/sessions/{session_id}/state_changes.json"
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    self._state_changes = json.load(f)

            duration = time.time() - start_time
            logger.info("State loaded from %s in %.2fs", state_file, duration)
            return self.__structured_cv
        except Exception as e:
            logger.error("Failed to load state: %s", str(e))
            return None

    def save_state(self):
        """
        Save the current StructuredCV state.

        Returns:
            The path to the saved file, or None if saving failed.
        """
        try:
            start_time = time.time()
            structured_cv = self.get_structured_cv()
            if not structured_cv:
                logger.warning("No CV data to save")
                return None

            os.makedirs(f"data/sessions/{structured_cv.id}", exist_ok=True)

            state_file = f"data/sessions/{structured_cv.id}/state.json"
            with open(state_file, "w") as f:
                json.dump(structured_cv.to_dict(), f, indent=2, cls=EnumEncoder)

            log_file = f"data/sessions/{structured_cv.id}/state_changes.json"
            with open(log_file, "w") as f:
                json.dump(self._state_changes, f, indent=2)

            duration = time.time() - start_time
            self._last_save_time = datetime.now().isoformat()
            logger.info("State saved to %s in %.2fs", state_file, duration)
            return state_file
        except Exception as e:
            logger.error("Failed to save state: %s", str(e))
            return None

    def get_structured_cv(self):
        """
        Get the current StructuredCV instance.

        Returns:
            The current StructuredCV instance, or None if it doesn't exist.
        """
        return self.__structured_cv

    def set_structured_cv(self, structured_cv):
        """
        Set the current StructuredCV instance.

        Args:
            structured_cv: The StructuredCV instance to set.
        """
        self.__structured_cv = structured_cv
        logger.info(
            "StructuredCV set with session ID: %s",
            structured_cv.id if structured_cv else "None",
        )

    def get_job_description_data(self) -> Optional[JobDescriptionData]:
        """
        Get the current JobDescriptionData from the structured CV's metadata.

        Returns:
            The current JobDescriptionData instance, or None if it doesn't exist.
        """
        structured_cv = self.get_structured_cv()
        if structured_cv and structured_cv.metadata:
            job_data = structured_cv.metadata.get("job_description")
            if isinstance(job_data, JobDescriptionData):
                return job_data
            elif isinstance(job_data, dict):
                # Attempt to load from dict if it was serialized
                try:
                    return JobDescriptionData.model_validate(job_data)
                except Exception as e:
                    logger.error("Failed to validate job_data from dict: %s", e)
                    return None
        return None

    def set_job_description_data(
        self, job_description_data: JobDescriptionData
    ) -> bool:
        """
        Set the JobDescriptionData in the structured CV's metadata.

        Args:
            job_description_data: The JobDescriptionData instance to set.

        Returns:
            True if the data was set successfully, False otherwise.
        """
        try:
            structured_cv = self.get_structured_cv()
            if not structured_cv:
                logger.error(
                    "Cannot set job description data: No StructuredCV instance exists."
                )
                return False

            # Ensure metadata exists
            if not structured_cv.metadata:
                structured_cv.metadata = {}

            # Set the job description data
            structured_cv.metadata["job_description"] = job_description_data
            logger.info("Job description data set successfully in state manager")
            return True
        except Exception as e:
            logger.error("Error setting job description data: %s", e)
            return False

    # Methods that delegate to StructuredCV - these provide a convenient interface
    # while keeping the actual functionality in the StructuredCV class

    def update_item_content(self, item_id, new_content):
        """Delegate to StructuredCV.update_item_content"""
        structured_cv = self.get_structured_cv()
        if not structured_cv:
            logger.error("Cannot update item: No StructuredCV instance exists.")
            return False
        return structured_cv.update_item_content(item_id, new_content)

    def update_item_status(self, item_id: str, new_status: str) -> bool:
        """Delegate to StructuredCV.update_item_status"""
        structured_cv = self.get_structured_cv()
        if not structured_cv:
            logger.error("Cannot update item status: No StructuredCV instance exists.")
            return False
        return structured_cv.update_item_status(item_id, new_status)

    def update_subsection_status(self, item_id: str, new_status: ItemStatus) -> bool:
        """Update the status of a subsection item.

        Args:
            item_id: The ID of the item to update
            new_status: The new ItemStatus to set

        Returns:
            True if the update was successful, False otherwise
        """
        structured_cv = self.get_structured_cv()
        if not structured_cv:
            logger.error(
                "Cannot update subsection status: No StructuredCV instance exists."
            )
            return False
        return structured_cv.update_item_status(
            item_id,
            new_status.value if hasattr(new_status, "value") else str(new_status),
        )

    def get_item(self, item_id):
        """Delegate to StructuredCV.find_item_by_id"""
        structured_cv = self.get_structured_cv()
        if not structured_cv:
            logger.error("Cannot get item: No StructuredCV instance exists.")
            return None
        item, _, _ = structured_cv.find_item_by_id(item_id)
        return item

    def convert_to_content_data(self):
        """Delegate to StructuredCV.to_content_data"""
        structured_cv = self.get_structured_cv()
        if not structured_cv:
            return None
        return structured_cv.to_content_data()

    # Content management methods - these delegate to StructuredCV

    @property
    def cv_data(self):
        """Delegate to StructuredCV for CV data retrieval"""
        structured_cv = self.get_structured_cv()
        if not structured_cv:
            return None
        return structured_cv.get_cv_data()

    def update_cv_data(self, new_content):
        """Delegate to StructuredCV for CV data updates"""
        try:
            structured_cv = self.get_structured_cv()
            if not structured_cv:
                logger.error("Cannot update CV data: No StructuredCV instance exists.")
                return False

            # Update the CV data
            structured_cv.cv_data = new_content
            logger.info("CV data updated successfully")
            return True
        except Exception as e:
            logger.error("Error updating CV data: %s", e)
            return False

    def update_section(self, section_data: Dict[str, Any]) -> bool:
        """Update a section in the StructuredCV.

        Args:
            section_data: Dictionary containing section data with 'id' and other fields

        Returns:
            True if update was successful, False otherwise
        """
        try:
            structured_cv = self.get_structured_cv()
            if not structured_cv:
                logger.error("Cannot update section: No StructuredCV instance exists.")
                return False

            section_id = section_data.get("id")
            if not section_id:
                logger.error("Cannot update section: No section ID provided.")
                return False

            # Find and update the section
            for section in structured_cv.sections:
                if section.id == section_id:
                    # Update section fields
                    if "title" in section_data:
                        section.name = section_data["title"]
                    if "description" in section_data:
                        section.description = section_data["description"]
                    logger.info("Section %s updated successfully", section_id)
                    return True

            logger.error("Section with ID %s not found", section_id)
            return False
        except Exception as e:
            logger.error("Error updating section: %s", e)
            return False

    def update_subsection(
        self, parent_section: Dict[str, Any], subsection_data: Dict[str, Any]
    ) -> bool:
        """Update a subsection within a parent section.

        Args:
            parent_section: Dictionary containing parent section data with 'id'
            subsection_data: Dictionary containing subsection data with 'id' and other fields

        Returns:
            True if update was successful, False otherwise
        """
        try:
            structured_cv = self.get_structured_cv()
            if not structured_cv:
                logger.error(
                    "Cannot update subsection: No StructuredCV instance exists."
                )
                return False

            parent_section_id = parent_section.get("id")
            subsection_id = subsection_data.get("id")

            if not parent_section_id or not subsection_id:
                logger.error(
                    "Cannot update subsection: Missing section or subsection ID."
                )
                return False

            # Find parent section and update subsection
            for section in structured_cv.sections:
                if section.id == parent_section_id:
                    for subsection in section.subsections:
                        if subsection.id == subsection_id:
                            # Update subsection fields
                            if "title" in subsection_data:
                                subsection.name = subsection_data["title"]
                            if "description" in subsection_data:
                                subsection.description = subsection_data["description"]
                            logger.info(
                                "Subsection %s updated successfully", subsection_id
                            )
                            return True

            logger.error(
                "Subsection with ID %s not found in section %s",
                subsection_id,
                parent_section_id,
            )
            return False
        except Exception as e:
            logger.error("Error updating subsection: %s", e)
            return False

    def update_item_feedback(self, item_id: str, feedback: str) -> bool:
        """Update feedback for a specific item.

        Args:
            item_id: The ID of the item to update
            feedback: The feedback text to store

        Returns:
            True if update was successful, False otherwise
        """
        try:
            structured_cv = self.get_structured_cv()
            if not structured_cv:
                logger.error(
                    "Cannot update item feedback: No StructuredCV instance exists."
                )
                return False

            # Find and update the item
            for section in structured_cv.sections:
                for subsection in section.subsections:
                    for item in subsection.items:
                        if item.id == item_id:
                            # Store feedback in metadata if available
                            if hasattr(item, "metadata") and item.metadata:
                                item.metadata["feedback"] = feedback
                            else:
                                # Create metadata if it doesn't exist
                                item.metadata = {"feedback": feedback}
                            logger.info("Feedback updated for item %s", item_id)
                            return True

            logger.error("Item with ID %s not found", item_id)
            return False
        except Exception as e:
            logger.error("Error updating item feedback: %s", e)
            return False

    def update_item(
        self,
        section_data: Dict[str, Any],
        subsection_data: Dict[str, Any],
        item_data: Dict[str, Any],
    ) -> bool:
        """Update an item within a subsection.

        Args:
            section_data: Dictionary containing section data with 'id'
            subsection_data: Dictionary containing subsection data with 'id'
            item_data: Dictionary containing item data with 'id' and other fields

        Returns:
            True if update was successful, False otherwise
        """
        try:
            structured_cv = self.get_structured_cv()
            if not structured_cv:
                logger.error("Cannot update item: No StructuredCV instance exists.")
                return False

            section_id = section_data.get("id")
            subsection_id = subsection_data.get("id")
            item_id = item_data.get("id")

            if not all([section_id, subsection_id, item_id]):
                logger.error(
                    "Cannot update item: Missing section, subsection, or item ID."
                )
                return False

            # Find and update the item
            for section in structured_cv.sections:
                if section.id == section_id:
                    for subsection in section.subsections:
                        if subsection.id == subsection_id:
                            for item in subsection.items:
                                if item.id == item_id:
                                    # Update item fields
                                    if "title" in item_data:
                                        item.title = item_data["title"]
                                    if "content" in item_data:
                                        item.content = item_data["content"]
                                    if "status" in item_data:
                                        item.status = item_data["status"]
                                    logger.info("Item %s updated successfully", item_id)
                                    return True

            logger.error("Item with ID %s not found", item_id)
            return False
        except Exception as e:
            logger.error("Error updating item: %s", e)
            return False

    def update_item_metadata(self, item_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a specific item.

        Args:
            item_id: The ID of the item to update
            metadata: Dictionary containing metadata to update

        Returns:
            True if update was successful, False otherwise
        """
        try:
            structured_cv = self.get_structured_cv()
            if not structured_cv:
                logger.error(
                    "Cannot update item metadata: No StructuredCV instance exists."
                )
                return False

            # Find and update the item
            for section in structured_cv.sections:
                for subsection in section.subsections:
                    for item in subsection.items:
                        if item.id == item_id:
                            # Update or create metadata
                            if hasattr(item, "metadata") and item.metadata:
                                item.metadata.update(metadata)
                            else:
                                item.metadata = metadata.copy()
                            logger.info("Metadata updated for item %s", item_id)
                            return True

            logger.error("Item with ID %s not found", item_id)
            return False
        except Exception as e:
            logger.error("Error updating item metadata: %s", e)
            return False

    def save_session(self, session_dir: str) -> bool:
        """Save the current session state to a directory.

        Args:
            session_dir: Directory path to save the session

        Returns:
            True if save was successful, False otherwise
        """
        try:
            os.makedirs(session_dir, exist_ok=True)

            # Save StructuredCV state
            structured_cv = self.get_structured_cv()
            if structured_cv:
                cv_file = os.path.join(session_dir, "structured_cv.json")
                with open(cv_file, "w", encoding="utf-8") as f:
                    json.dump(
                        structured_cv.model_dump(),
                        f,
                        indent=2,
                        ensure_ascii=False,
                        cls=EnumEncoder,
                    )

            # Save StateManager metadata
            state_file = os.path.join(session_dir, "state_manager.json")
            state_data = {
                "session_id": self.session_id,
                "last_save_time": self._last_save_time,
                "state_changes": self._state_changes,
            }
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)

            self._last_save_time = time.time()
            logger.info("Session saved to %s", session_dir)
            return True
        except Exception as e:
            logger.error("Error saving session: %s", e)
            return False

    def load_session(self, session_dir: str) -> bool:
        """Load session state from a directory.

        Args:
            session_dir: Directory path to load the session from

        Returns:
            True if load was successful, False otherwise
        """
        try:
            # Load StructuredCV state
            cv_file = os.path.join(session_dir, "structured_cv.json")
            if os.path.exists(cv_file):
                with open(cv_file, "r", encoding="utf-8") as f:
                    cv_data = json.load(f)
                structured_cv = StructuredCV.model_validate(cv_data)
                self.set_structured_cv(structured_cv)

            # Load StateManager metadata
            state_file = os.path.join(session_dir, "state_manager.json")
            if os.path.exists(state_file):
                with open(state_file, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                self.session_id = state_data.get("session_id", self.session_id)
                self._last_save_time = state_data.get("last_save_time")
                self._state_changes = state_data.get("state_changes", [])

            logger.info("Session loaded from %s", session_dir)
            return True
        except Exception as e:
            logger.error("Error loading session: %s", e)
            return False


# State Management Layer Documentation
"""
State Management Layers in AI CV Generator:

1. Streamlit session_state (UI Layer):
   - Holds only raw user input and UI flags (e.g., text areas, checkboxes, API key, progress).
   - Should NOT store complex objects like StructuredCV or AgentState.
   - See: src/frontend/state_helpers.py

2. AgentState (Workflow Layer):
   - The single source of truth for workflow execution.
   - Created from UI input at workflow start, destroyed or archived at end.
   - All agents read/write only to this object.
   - See: src/orchestration/state.py

3. StateManager (Persistence Layer):
   - Used only to save/load AgentState or StructuredCV at workflow boundaries.
   - No agent or service should call StateManager directly during workflow execution.
   - See: src/core/state_manager.py

For details, see TASK_BLUEPRINT.md section 12.
"""
