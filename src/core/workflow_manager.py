"""Core WorkflowManager component for managing CV generation workflows.

This module provides the WorkflowManager class that orchestrates the CV generation
workflow using the CVWorkflowGraph and manages workflow lifecycle.
"""

import json
import uuid
from pathlib import Path
from typing import Optional

from src.config.logging_config import get_structured_logger
from src.models.workflow_models import WorkflowStage, UserFeedback
from src.orchestration.cv_workflow_graph import CVWorkflowGraph
from src.orchestration.state import AgentState
from src.models.cv_models import StructuredCV
from src.models.cv_models import JobDescriptionData



logger = get_structured_logger(__name__)


class WorkflowManager:
    """Manages CV generation workflows and their lifecycle.

    This class provides a high-level interface for creating, executing,
    and monitoring CV generation workflows using the CVWorkflowGraph.
    """

    def __init__(self, cv_workflow_graph: CVWorkflowGraph):
        """Initialize the WorkflowManager.

        Args:
            cv_workflow_graph: The CV workflow graph instance
        """
        self.cv_workflow_graph = cv_workflow_graph
        self.logger = logger
        self.sessions_dir = Path("instance/sessions")

        logger.info("WorkflowManager initialized")

    def create_new_workflow(
        self,
        cv_text: str,
        jd_text: str,
        session_id: Optional[str] = None,
        workflow_type: Optional[str] = None
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
            session_id = str(uuid.uuid4())

        # Check if workflow already exists
        session_file = self.sessions_dir / f"{session_id}.json"
        if session_file.exists():
            raise ValueError(f"Workflow with session_id {session_id} already exists")

        # Create initial AgentState object
        if session_id:
            agent_state = AgentState(
                session_id=session_id,
                structured_cv=StructuredCV.create_empty(cv_text=cv_text),
                cv_text=cv_text
            )
        else:
            agent_state = AgentState(
                structured_cv=StructuredCV.create_empty(cv_text=cv_text),
                cv_text=cv_text
            )
            session_id = agent_state.session_id  # Use the auto-generated session_id

        # Set job description data if provided
        if jd_text:
            agent_state.job_description_data = JobDescriptionData(raw_text=jd_text)

        # Create sessions directory if it doesn't exist
        try:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(
                "Failed to create sessions directory",
                extra={"error": str(e), "path": str(self.sessions_dir)}
            )
            raise OSError(f"Failed to create sessions directory: {str(e)}") from e

        # Serialize AgentState to JSON file
        session_file = self.sessions_dir / f"{session_id}.json"
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                f.write(agent_state.model_dump_json(indent=2))
        except (OSError, IOError) as e:
            logger.error(
                "Failed to save session file",
                extra={"session_id": session_id, "error": str(e)}
            )
            raise OSError(f"Failed to save session file: {str(e)}") from e

        logger.info(
            "New workflow created with file persistence",
            extra={
                "session_id": session_id,
                "workflow_type": workflow_type,
                "stage": WorkflowStage.INITIALIZATION,
                "session_file": str(session_file)
            }
        )

        return session_id

    async def trigger_workflow_step(
        self,
        session_id: str,
        agent_state: AgentState
    ) -> AgentState:
        """Trigger the next step in the workflow.

        Args:
            session_id: The session ID of the workflow
            agent_state: The current agent state

        Returns:
            AgentState: The updated agent state after workflow execution

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
            extra={
                "session_id": session_id,
                "trace_id": agent_state.trace_id
            }
        )

        try:
            # Use the CVWorkflowGraph's trigger_workflow_step method which includes pause mechanism
            updated_state = await self.cv_workflow_graph.trigger_workflow_step(agent_state)

            # Save updated state to file
            self._save_state(session_id, updated_state)

            logger.info(
                "Workflow step completed",
                extra={
                    "session_id": session_id,
                    "trace_id": updated_state.trace_id,
                    "has_errors": bool(updated_state.error_messages),
                    "workflow_status": updated_state.workflow_status
                }
            )

            return updated_state

        except (RuntimeError, ValueError, OSError, IOError) as e:
            # Add error to agent state and save
            if agent_state.error_messages is None:
                agent_state.error_messages = []
            agent_state.error_messages.append(str(e))

            # Save error state to file
            try:
                self._save_state(session_id, agent_state)
            except (OSError, IOError) as save_error:
                logger.error(
                    "Failed to save error state",
                    extra={"session_id": session_id, "error": str(save_error)}
                )

            logger.error(
                "Workflow step execution failed",
                extra={
                    "session_id": session_id,
                    "error": str(e)
                }
            )
            raise RuntimeError(f"Workflow execution failed: {str(e)}") from e

    def get_workflow_status(self, session_id: str) -> Optional[AgentState]:
        """Get the current status of a workflow by reading from the session file.

        Args:
            session_id: The session ID of the workflow

        Returns:
            Optional[AgentState]: The agent state if found, None otherwise
        """
        session_file = self.sessions_dir / f"{session_id}.json"

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                json_data = f.read()
                agent_state = AgentState.model_validate_json(json_data)

            logger.debug(
                "Retrieved workflow status from file",
                extra={
                    "session_id": session_id,
                    "session_file": str(session_file),
                    "trace_id": agent_state.trace_id
                }
            )
            return agent_state

        except FileNotFoundError:
            logger.warning(
                "Session file not found",
                extra={"session_id": session_id, "session_file": str(session_file)}
            )
            return None

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to decode session file JSON",
                extra={
                    "session_id": session_id,
                    "session_file": str(session_file),
                    "error": str(e)
                }
            )
            return None

        except (OSError, IOError, UnicodeDecodeError) as e:
            logger.error(
                "Unexpected error reading session file",
                extra={
                    "session_id": session_id,
                    "session_file": str(session_file),
                    "error": str(e)
                }
            )
            return None

    def send_feedback(
        self,
        session_id: str,
        feedback: UserFeedback
    ) -> bool:
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
                extra={"session_id": session_id}
            )
            return False

        # Set user feedback
        agent_state.user_feedback = feedback

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
                    "rating": feedback.rating
                }
            )
            return True

        except (OSError, IOError) as e:
            logger.error(
                "Failed to save feedback to session file",
                extra={
                    "session_id": session_id,
                    "error": str(e)
                }
            )
            return False

    def _determine_next_stage(self, current_stage: WorkflowStage) -> Optional[WorkflowStage]:
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

    def _save_state(self, session_id: str, agent_state: AgentState) -> None:
        """Save agent state to session file.

        Args:
            session_id: The session ID
            agent_state: The agent state to save

        Raises:
            OSError: If there's an error saving the file
        """
        session_file = self.sessions_dir / f"{session_id}.json"

        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                f.write(agent_state.model_dump_json(indent=2))
        except (OSError, IOError) as e:
            logger.error(
                "Failed to save session file",
                extra={"session_id": session_id, "error": str(e)}
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
                    "Cleaned up workflow for session",
                    extra={"session_id": session_id}
                )
                return True
            else:
                logger.warning(
                    "No workflow found for cleanup",
                    extra={"session_id": session_id}
                )
                return False
        except (OSError, IOError, PermissionError) as e:
            logger.error(
                "Error cleaning up workflow",
                extra={"session_id": session_id, "error": str(e)}
            )
            return False