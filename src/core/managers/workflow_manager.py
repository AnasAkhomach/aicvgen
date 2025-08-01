"""Core WorkflowManager component for managing CV generation workflows.

This module provides the WorkflowManager class that orchestrates the CV generation
workflow using the modular workflow graph pattern and manages workflow lifecycle.
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from src.config.logging_config import get_structured_logger
from src.models.cv_models import JobDescriptionData, StructuredCV
from src.models.workflow_models import UserFeedback, WorkflowStage
from src.orchestration.graphs.main_graph import create_cv_workflow_graph_with_di
from src.orchestration.state import GlobalState, create_global_state
from src.services.cv_template_loader_service import CVTemplateLoaderService
from src.services.session_manager import SessionManager

logger = get_structured_logger(__name__)


def _serialize_for_json(obj: Any) -> Any:
    """Custom serializer for JSON that handles Pydantic models and UUID objects properly.

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable representation
    """
    if isinstance(obj, BaseModel):
        return {
            "__pydantic_model__": obj.__class__.__module__
            + "."
            + obj.__class__.__name__,
            "data": obj.model_dump(mode="json"),
        }
    elif isinstance(obj, UUID):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_for_json(item) for item in obj]
    else:
        return obj


def _deserialize_from_json(obj: Any) -> Any:
    """Custom deserializer for JSON that reconstructs Pydantic models.

    Args:
        obj: JSON object to deserialize

    Returns:
        Deserialized object with Pydantic models reconstructed
    """
    if isinstance(obj, dict):
        if "__pydantic_model__" in obj:
            # Reconstruct Pydantic model
            model_path = obj["__pydantic_model__"]
            data = obj["data"]

            # Import the model class
            module_name, class_name = model_path.rsplit(".", 1)
            try:
                module = __import__(module_name, fromlist=[class_name])
                model_class = getattr(module, class_name)
                return model_class.model_validate(data)
            except (ImportError, AttributeError) as e:
                logger.warning(
                    f"Failed to deserialize Pydantic model {model_path}: {e}"
                )
                return data  # Fallback to raw data
        else:
            return {k: _deserialize_from_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deserialize_from_json(item) for item in obj]
    else:
        return obj


class WorkflowManager:
    """Manages CV generation workflows and their lifecycle.

    This class provides a high-level interface for creating, executing,
    and monitoring CV generation workflows using the modular workflow graph.
    """

    def __init__(
        self,
        cv_template_loader_service: CVTemplateLoaderService,
        session_manager: SessionManager,
        container,
    ):
        """Initialize the WorkflowManager.

        Args:
            cv_template_loader_service: Service for loading CV templates
            session_manager: SessionManager for centralized session management
            container: Dependency injection container for workflow graph creation
        """
        self.cv_template_loader_service = cv_template_loader_service
        self.session_manager = session_manager
        self.container = container
        self.logger = logger
        self.sessions_dir = Path("instance/sessions")

        logger.info("WorkflowManager initialized with injected services")

    def create_new_workflow(
        self,
        cv_text: str,
        jd_text: str,
        session_id: Optional[str] = None,
        workflow_type: Optional[str] = None,
    ) -> str:
        """Create a new workflow instance with file-based state persistence.

        Args:
            cv_text: The raw text of the user's CV
            jd_text: The job description text
            session_id: Optional session ID, will generate one if not provided
            workflow_type: Optional workflow type identifier

        Returns:
            str: The session ID of the created workflow

        Raises:
            ValueError: If a workflow with the given session_id already exists
            OSError: If there's an error creating the session file
        """
        if session_id is None:
            session_id = self.session_manager.create_session()

        # Check if workflow already exists
        session_file = self.sessions_dir / f"{session_id}.json"
        if session_file.exists():
            raise ValueError(f"Workflow with session_id {session_id} already exists")

        # Load template and create structured CV skeleton
        try:
            template_path = "src/templates/default_cv_template.md"
            structured_cv = self.cv_template_loader_service.load_from_markdown(
                template_path
            )
            # Set the original CV text for reference in metadata
            structured_cv.metadata.extra["original_cv_text"] = cv_text
        except (FileNotFoundError, ValueError) as e:
            logger.warning(
                "Failed to load CV template, falling back to empty structure",
                extra={"error": str(e), "template_path": template_path},
            )
            # Fallback to empty structure if template loading fails
            structured_cv = StructuredCV.create_empty(cv_text=cv_text)

        # Create initial GlobalState object
        if session_id:
            agent_state = create_global_state(
                cv_text=cv_text, session_id=session_id, structured_cv=structured_cv
            )
        else:
            agent_state = create_global_state(
                cv_text=cv_text, structured_cv=structured_cv
            )
            session_id = agent_state["session_id"]  # Use the auto-generated session_id

        # Set job description data if provided
        if jd_text:
            agent_state["job_description_data"] = JobDescriptionData(raw_text=jd_text)

        # Create sessions directory if it doesn't exist
        try:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(
                "Failed to create sessions directory",
                extra={"error": str(e), "path": str(self.sessions_dir)},
            )
            raise OSError(f"Failed to create sessions directory: {str(e)}") from e

        # Serialize GlobalState to JSON file
        session_file = self.sessions_dir / f"{session_id}.json"
        try:
            with open(session_file, "w", encoding="utf-8") as f:
                serialized_state = _serialize_for_json(agent_state)
                json.dump(serialized_state, f, indent=2)
        except (OSError, IOError) as e:
            logger.error(
                "Failed to save session file",
                extra={"session_id": session_id, "error": str(e)},
            )
            raise OSError(f"Failed to save session file: {str(e)}") from e

        logger.info(
            "New workflow created with file persistence",
            extra={
                "session_id": session_id,
                "workflow_type": workflow_type,
                "stage": WorkflowStage.INITIALIZATION,
                "session_file": str(session_file),
            },
        )

        return session_id

    async def trigger_workflow_step(
        self, session_id: str, agent_state: GlobalState
    ) -> GlobalState:
        """Trigger the next step in the workflow.

        Args:
            session_id: The session ID of the workflow
            agent_state: The current agent state

        Returns:
            GlobalState: The updated agent state after workflow execution

        Raises:
            ValueError: If no active workflow found for the session
            RuntimeError: If workflow execution fails
        """
        # Check if session exists
        current_state = self.get_workflow_status(session_id)
        if current_state is None:
            raise ValueError(f"No active workflow found for session_id: {session_id}")

        logger.info(
            "Triggering workflow step",
            extra={"session_id": session_id, "trace_id": agent_state.get("trace_id")},
        )

        try:
            # Get or create workflow graph for this session
            compiled_graph = self._get_workflow_graph(session_id)

            # Execute workflow using the compiled graph directly
            updated_state = await compiled_graph.ainvoke(agent_state)

            # Save updated state to file
            self._save_state(session_id, updated_state)

            logger.info(
                "Workflow step completed",
                extra={
                    "session_id": session_id,
                    "trace_id": updated_state.get("trace_id"),
                    "has_errors": bool(updated_state.get("error_messages", [])),
                    "workflow_status": updated_state.get("workflow_status"),
                },
            )

            return updated_state

        except Exception as e:
            # Check if this is a normal workflow completion (END state)
            if str(e) == "END":
                logger.info(
                    f"Workflow completed successfully for session {session_id}",
                    extra={"session_id": session_id, "final_state": "END"},
                )
                # Update workflow status to completed
                agent_state["workflow_status"] = "COMPLETED"
                self._save_state(session_id, agent_state)
                return agent_state

            # Handle actual errors
            if agent_state.get("error_messages") is None:
                agent_state["error_messages"] = []
            agent_state["error_messages"].append(str(e))

            # Save error state to file
            try:
                self._save_state(session_id, agent_state)
            except (OSError, IOError) as save_error:
                logger.error(
                    "Failed to save error state",
                    extra={"session_id": session_id, "error": str(save_error)},
                )

            logger.error(
                f"Workflow step execution failed for session {session_id}: {str(e)}",
                extra={"session_id": session_id, "error": str(e)},
            )
            raise RuntimeError(f"Workflow execution failed: {str(e)}") from e

    def get_workflow_status(self, session_id: str) -> Optional[GlobalState]:
        """Get the current status of a workflow by reading from the session file.

        Args:
            session_id: The session ID of the workflow

        Returns:
            Optional[GlobalState]: The agent state if found, None otherwise
        """
        session_file = self.sessions_dir / f"{session_id}.json"

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            # Deserialize with proper Pydantic model reconstruction
            agent_state = _deserialize_from_json(raw_data)

            logger.debug(
                "Retrieved workflow status from file",
                extra={
                    "session_id": session_id,
                    "session_file": str(session_file),
                    "trace_id": agent_state.get("trace_id"),
                },
            )
            return agent_state

        except FileNotFoundError:
            logger.warning(
                "Session file not found",
                extra={"session_id": session_id, "session_file": str(session_file)},
            )
            return None

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to decode session file JSON",
                extra={
                    "session_id": session_id,
                    "session_file": str(session_file),
                    "error": str(e),
                },
            )
            return None

        except (OSError, IOError, UnicodeDecodeError) as e:
            logger.error(
                "Unexpected error reading session file",
                extra={
                    "session_id": session_id,
                    "session_file": str(session_file),
                    "error": str(e),
                },
            )
            return None

    def send_feedback(self, session_id: str, feedback: UserFeedback) -> bool:
        """Send user feedback to a workflow.

        Args:
            session_id: The session ID of the workflow
            feedback: The user feedback to send

        Returns:
            bool: True if feedback was successfully recorded, False otherwise
        """
        # Get current agent state
        agent_state = self.get_workflow_status(session_id)
        if agent_state is None:
            logger.warning(
                "Cannot send feedback to non-existent workflow",
                extra={"session_id": session_id},
            )
            return False

        # Set user feedback
        agent_state["user_feedback"] = feedback

        # Save updated state back to file
        try:
            self._save_state(session_id, agent_state)

            logger.info(
                "User feedback recorded",
                extra={
                    "session_id": session_id,
                    "action": feedback.action,
                    "item_id": feedback.item_id,
                    "has_text": bool(feedback.feedback_text),
                    "rating": feedback.rating,
                },
            )
            return True

        except (OSError, IOError) as e:
            logger.error(
                "Failed to save feedback to session file",
                extra={"session_id": session_id, "error": str(e)},
            )
            return False

    def _determine_next_stage(
        self, current_stage: WorkflowStage
    ) -> Optional[WorkflowStage]:
        """Determine the next workflow stage based on current stage.

        Args:
            current_stage: The current workflow stage

        Returns:
            Optional[WorkflowStage]: The next stage, or None if workflow is complete
        """
        stage_progression = {
            WorkflowStage.INITIALIZATION: WorkflowStage.CV_PARSING,
            WorkflowStage.CV_PARSING: WorkflowStage.JOB_ANALYSIS,
            WorkflowStage.JOB_ANALYSIS: WorkflowStage.CONTENT_GENERATION,
            WorkflowStage.CONTENT_GENERATION: WorkflowStage.REVIEW,
            WorkflowStage.REVIEW: WorkflowStage.COMPLETED,
        }

        return stage_progression.get(current_stage)

    def _save_state(self, session_id: str, agent_state: GlobalState) -> None:
        """Save agent state to session file.

        Args:
            session_id: The session ID
            agent_state: The agent state to save

        Raises:
            OSError: If there's an error saving the file
        """
        session_file = self.sessions_dir / f"{session_id}.json"

        try:
            # Serialize with proper Pydantic model handling
            serialized_state = _serialize_for_json(agent_state)
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(serialized_state, f, indent=2)
        except (OSError, IOError) as e:
            logger.error(
                "Failed to save session file",
                extra={"session_id": session_id, "error": str(e)},
            )
            raise OSError(f"Failed to save session file: {str(e)}") from e

    def cleanup_workflow(self, session_id: str) -> bool:
        """Clean up workflow resources.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if cleanup successful, False if workflow not found
        """
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
                logger.info(
                    "Cleaned up workflow for session", extra={"session_id": session_id}
                )
                return True
            else:
                logger.warning(
                    "No workflow found for cleanup", extra={"session_id": session_id}
                )
                return False
        except (OSError, IOError, PermissionError) as e:
            logger.error(
                "Error cleaning up workflow",
                extra={"session_id": session_id, "error": str(e)},
            )
            return False

    async def astream_workflow(self, session_id: str, callback_handler=None):
        """Stream workflow execution asynchronously.

        Args:
            session_id: The session ID
            callback_handler: Optional callback handler for streaming updates

        Yields:
            dict: Workflow state updates during execution
        """
        try:
            # Get current workflow status
            status = self.get_workflow_status(session_id)
            if not status or status.get("status") == "completed":
                logger.warning(
                    "Cannot stream workflow - invalid or completed session",
                    extra={"session_id": session_id},
                )
                return

            # Load current state
            session_file = self.sessions_dir / f"{session_id}.json"
            if not session_file.exists():
                logger.error(
                    "Session file not found for streaming",
                    extra={"session_id": session_id},
                )
                return

            with open(session_file, "r", encoding="utf-8") as f:
                state_data = json.load(f)

            current_state = _deserialize_from_json(state_data)

            # Get workflow graph
            compiled_graph = self._get_workflow_graph(session_id)

            # Stream workflow execution
            async for state_update in compiled_graph.astream(current_state):
                # Yield state update to callback handler
                if callback_handler:
                    await callback_handler.on_workflow_update(state_update)

                yield state_update

                # Save intermediate state
                self._save_state(session_id, state_update)

        except Exception as e:
            # Check if this is a normal workflow completion (END state)
            if str(e) == "END":
                logger.info(
                    "Workflow streaming completed successfully",
                    extra={"session_id": session_id, "final_state": "END"},
                )
                # Update workflow status to completed in the current state
                current_state["workflow_status"] = "COMPLETED"
                self._save_state(session_id, current_state)

                # Notify callback handler of successful completion
                if callback_handler:
                    await callback_handler.on_workflow_update(
                        {"workflow_status": "COMPLETED"}
                    )
                return

            # Handle actual errors
            logger.error(
                "Error during workflow streaming",
                extra={"session_id": session_id, "error": str(e)},
            )
            if callback_handler:
                await callback_handler.on_workflow_error(str(e))
            raise

    def _get_workflow_graph(self, session_id: str):
        """Create a compiled workflow graph for the given session.

        Args:
            session_id: The session ID

        Returns:
            CompiledStateGraph: The compiled workflow graph instance
        """
        # Create new compiled workflow graph using DI container
        compiled_graph = create_cv_workflow_graph_with_di(self.container)
        logger.debug(
            "Created new compiled workflow graph for session",
            extra={"session_id": session_id},
        )

        return compiled_graph
