"""CV Generation Facade for simplified workflow orchestration.

This module provides the CvGenerationFacade class that encapsulates the complexity
of the WorkflowManager and provides a high-level interface for CV generation.
"""

import asyncio
import threading
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from src.config.logging_config import get_structured_logger
from src.core.facades.cv_template_manager_facade import CVTemplateManagerFacade
from src.core.facades.cv_vector_store_facade import CVVectorStoreFacade
from src.core.managers.workflow_manager import WorkflowManager
from src.models.cv_models import JobDescriptionData, StructuredCV
from src.models.workflow_models import ContentType, UserFeedback, WorkflowType
from src.orchestration.state import GlobalState, create_global_state

if TYPE_CHECKING:
    from src.agents.user_cv_parser_agent import UserCVParserAgent

logger = get_structured_logger(__name__)


def _run_async_in_thread(workflow_func, *args, **kwargs):
    """Run an async workflow function in a separate thread with its own event loop.

    This is necessary to avoid 'Event loop is closed' errors in Streamlit,
    which already runs in an event loop context.

    Args:
        workflow_func: The async function to run
        *args: Positional arguments for the workflow function
        **kwargs: Keyword arguments for the workflow function

    Returns:
        threading.Thread: The thread running the workflow
    """

    def run_in_thread():
        """Run the workflow in a separate thread with its own event loop."""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(workflow_func(*args, **kwargs))
            finally:
                loop.close()
        except Exception as e:
            logger.error("Error in workflow thread: %s", e)

    # Start workflow in background thread
    workflow_thread = threading.Thread(target=run_in_thread, daemon=True)
    workflow_thread.start()
    return workflow_thread


class CvGenerationFacade:
    """Facade for CV generation workflow operations.

    This facade encapsulates the complexity of WorkflowManager and provides
    a simplified interface for the UI layer to interact with CV generation workflows.
    """

    def __init__(
        self,
        workflow_manager: WorkflowManager,
        user_cv_parser_agent: "UserCVParserAgent",
        template_facade: Optional[CVTemplateManagerFacade] = None,
        vector_store_facade: Optional[CVVectorStoreFacade] = None,
    ):
        """Initialize the CV Generation Facade.

        Args:
            workflow_manager: The WorkflowManager instance to encapsulate
            user_cv_parser_agent: The UserCVParserAgent for CV parsing
            template_facade: Optional template manager facade
            vector_store_facade: Optional vector store facade
        """
        self.workflow_manager = workflow_manager
        self.user_cv_parser_agent = user_cv_parser_agent
        self.template_facade = template_facade
        self.vector_store_facade = vector_store_facade
        self.logger = logger

        logger.info(
            "CvGenerationFacade initialized with WorkflowManager, UserCVParserAgent and facades"
        )

    async def generate_cv(
        self,
        cv_text: str,
        jd_text: str,
        workflow_type: WorkflowType = WorkflowType.JOB_TAILORED_CV,
        session_id: Optional[str] = None,
    ) -> tuple[str, GlobalState]:
        """Generate a CV using the specified workflow.

        This is the main high-level method that encapsulates the entire
        CV generation workflow complexity.

        Args:
            cv_text: The original CV text content
            jd_text: The job description text
            workflow_type: Type of workflow to execute
            session_id: Optional session ID for workflow tracking

        Returns:
            tuple[str, GlobalState]: Session ID and initial workflow state

        Raises:
            ValueError: If input validation fails
            RuntimeError: If workflow creation or execution fails
        """
        try:
            # Validate inputs
            if not cv_text or not cv_text.strip():
                raise ValueError("CV text cannot be empty")
            if not jd_text or not jd_text.strip():
                raise ValueError("Job description text cannot be empty")

            logger.info(
                "Starting CV generation workflow",
                extra={
                    "workflow_type": workflow_type.value,
                    "has_session_id": session_id is not None,
                    "cv_text_length": len(cv_text),
                    "jd_text_length": len(jd_text),
                },
            )

            # Create new workflow
            session_id = self.workflow_manager.create_new_workflow(
                cv_text=cv_text,
                jd_text=jd_text,
                session_id=session_id,
                workflow_type=workflow_type,
            )

            # Get initial state
            initial_state = self.workflow_manager.get_workflow_status(session_id)
            if initial_state is None:
                raise RuntimeError(
                    f"Failed to retrieve initial workflow state for session {session_id}"
                )

            logger.info(
                "CV generation workflow created successfully",
                extra={
                    "session_id": session_id,
                    "workflow_type": workflow_type.value,
                    "initial_stage": initial_state.get("workflow_status"),
                },
            )

            return session_id, initial_state

        except (ValueError, RuntimeError) as e:
            logger.error(
                "Failed to start CV generation workflow",
                extra={"error": str(e), "workflow_type": workflow_type.value},
            )
            raise
        except Exception as e:
            logger.error(
                "Unexpected error during CV generation workflow creation",
                extra={"error": str(e), "workflow_type": workflow_type.value},
            )
            raise RuntimeError(
                f"Unexpected error during workflow creation: {str(e)}"
            ) from e

    async def execute_workflow_step(self, session_id: str) -> GlobalState:
        """Execute the next step in the workflow.

        Args:
            session_id: The session ID of the workflow to execute

        Returns:
            GlobalState: The updated workflow state

        Raises:
            ValueError: If session not found or invalid
            RuntimeError: If workflow execution fails
        """
        try:
            # Get current state
            current_state = self.workflow_manager.get_workflow_status(session_id)
            if current_state is None:
                raise ValueError(f"No workflow found for session {session_id}")

            logger.info(
                "Executing workflow step",
                extra={
                    "session_id": session_id,
                    "current_stage": current_state.get("workflow_status"),
                },
            )

            # Execute workflow step
            updated_state = await self.workflow_manager.trigger_workflow_step(
                session_id=session_id, agent_state=current_state
            )

            logger.info(
                "Workflow step executed successfully",
                extra={
                    "session_id": session_id,
                    "new_stage": updated_state.get("workflow_status"),
                    "has_errors": bool(updated_state.get("error_messages", [])),
                },
            )

            return updated_state

        except (ValueError, RuntimeError) as e:
            logger.error(
                "Failed to execute workflow step",
                extra={"session_id": session_id, "error": str(e)},
            )
            raise
        except Exception as e:
            logger.error(
                "Unexpected error during workflow step execution",
                extra={"session_id": session_id, "error": str(e)},
            )
            raise RuntimeError(
                f"Unexpected error during workflow execution: {str(e)}"
            ) from e

    def get_workflow_status(self, session_id: str) -> Optional[GlobalState]:
        """Get the current status of a workflow.

        Args:
            session_id: The session ID of the workflow

        Returns:
            Optional[GlobalState]: The current workflow state, or None if not found
        """
        try:
            state = self.workflow_manager.get_workflow_status(session_id)

            if state:
                logger.debug(
                    "Retrieved workflow status",
                    extra={
                        "session_id": session_id,
                        "workflow_status": state.get("workflow_status"),
                        "has_errors": bool(state.get("error_messages", [])),
                    },
                )
            else:
                logger.warning("Workflow not found", extra={"session_id": session_id})

            return state

        except Exception as e:
            logger.error(
                "Error retrieving workflow status",
                extra={"session_id": session_id, "error": str(e)},
            )
            return None

    def submit_user_feedback(self, session_id: str, feedback: UserFeedback) -> bool:
        """Submit user feedback for a workflow.

        Args:
            session_id: The session ID of the workflow
            feedback: The user feedback to submit

        Returns:
            bool: True if feedback was successfully submitted, False otherwise
        """
        try:
            success = self.workflow_manager.send_feedback(session_id, feedback)

            if success:
                logger.info(
                    "User feedback submitted successfully",
                    extra={
                        "session_id": session_id,
                        "action": feedback.action.value,
                        "item_id": feedback.item_id,
                        "has_feedback_text": bool(feedback.feedback_text),
                        "rating": feedback.rating,
                    },
                )
            else:
                logger.warning(
                    "Failed to submit user feedback",
                    extra={"session_id": session_id, "action": feedback.action.value},
                )

            return success

        except Exception as e:
            logger.error(
                "Error submitting user feedback",
                extra={
                    "session_id": session_id,
                    "action": feedback.action.value,
                    "error": str(e),
                },
            )
            return False

    # Template Management Methods
    def get_template(
        self, template_id: str, category: str = None
    ) -> Optional[Dict[str, Any]]:
        """Get a content template.

        Args:
            template_id: The ID of the template to retrieve
            category: Optional category filter

        Returns:
            Optional[Dict[str, Any]]: The template data, or None if not found
        """
        if not self.template_facade:
            self.logger.warning("Template facade not available")
            return None
        return self.template_facade.get_template(template_id, category)

    def format_template(
        self, template_id: str, variables: Dict[str, Any], category: str = None
    ) -> Optional[str]:
        """Format a template with variables.

        Args:
            template_id: The ID of the template to format
            variables: Variables to substitute in the template
            category: Optional category filter

        Returns:
            Optional[str]: The formatted template, or None if formatting failed
        """
        if not self.template_facade:
            self.logger.warning("Template facade not available")
            return None
        return self.template_facade.format_template(template_id, variables, category)

    def list_templates(self, category: str = None) -> List[str]:
        """List available templates.

        Args:
            category: Optional category filter

        Returns:
            List[str]: List of available template IDs
        """
        if not self.template_facade:
            self.logger.warning("Template facade not available")
            return []
        return self.template_facade.list_templates(category)

    # Vector Database Operations
    async def store_content(
        self,
        content: str,
        content_type: ContentType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Store content in vector database.

        Args:
            content: The content to store
            content_type: Type of content being stored
            metadata: Optional metadata to associate with the content

        Returns:
            Optional[str]: The ID of the stored content, or None if storage failed
        """
        if not self.vector_store_facade:
            self.logger.warning("Vector store facade not available")
            return None
        return await self.vector_store_facade.store_content(
            content, content_type, metadata
        )

    async def search_content(
        self, query: str, content_type: Optional[ContentType] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar content.

        Args:
            query: The search query
            content_type: Optional content type filter
            limit: Maximum number of results to return

        Returns:
            List[Dict[str, Any]]: List of matching content items
        """
        if not self.vector_store_facade:
            self.logger.warning("Vector store facade not available")
            return []
        return await self.vector_store_facade.search_content(query, content_type, limit)

    async def find_similar_content(
        self, content: str, content_type: Optional[ContentType] = None, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Find content similar to the provided content.

        Args:
            content: The content to find similarities for
            content_type: Optional content type filter
            limit: Maximum number of results to return

        Returns:
            List[Dict[str, Any]]: List of similar content items
        """
        if not self.vector_store_facade:
            self.logger.warning("Vector store facade not available")
            return []
        return await self.vector_store_facade.find_similar_content(
            content, content_type, limit
        )

    # High-level Workflow Methods
    async def generate_basic_cv(
        self,
        personal_info: Dict[str, Any],
        experience: List[Dict[str, Any]],
        education: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a basic CV.

        Args:
            personal_info: Personal information data
            experience: List of experience entries
            education: List of education entries
            session_id: Optional session ID
            **kwargs: Additional metadata

        Returns:
            Dict[str, Any]: The workflow execution result
        """
        try:
            # Create structured CV from the provided data
            structured_cv = StructuredCV.create_empty()

            # Create global state
            agent_state = create_global_state(
                cv_text="",
                structured_cv=structured_cv,
                session_metadata={
                    "personal_info": personal_info,
                    "experience": experience,
                    "education": education,
                    **kwargs,
                },
            )

            # Execute workflow through workflow manager
            return await self.workflow_manager.execute_workflow(
                WorkflowType.BASIC_CV_GENERATION,
                agent_state,
                session_id,
            )

        except Exception as e:
            self.logger.error(
                "Failed to generate basic CV",
                extra={"error": str(e), "session_id": session_id},
            )
            raise RuntimeError(f"Basic CV generation failed: {str(e)}") from e

    async def generate_job_tailored_cv(
        self,
        personal_info: Dict[str, Any],
        experience: List[Dict[str, Any]],
        job_description: Union[str, Dict[str, Any]],
        session_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a job-tailored CV.

        Args:
            personal_info: Personal information data
            experience: List of experience entries
            job_description: Job description as string or dict
            session_id: Optional session ID
            **kwargs: Additional metadata

        Returns:
            Dict[str, Any]: The workflow execution result
        """
        try:
            # Handle both string and dict job descriptions
            if isinstance(job_description, str):
                job_desc_dict = {
                    "description": job_description.strip(),
                    "raw_text": job_description.strip(),
                }
            else:
                job_desc_dict = job_description

            # Create structured CV
            structured_cv = StructuredCV.create_empty()

            # Create job description data
            job_description_data = JobDescriptionData(
                raw_text=job_desc_dict.get("raw_text", ""),
                description=job_desc_dict.get("description", ""),
                requirements=job_desc_dict.get("requirements", []),
                skills=job_desc_dict.get("skills", []),
                company_info=job_desc_dict.get("company_info", {}),
            )

            # Create global state
            agent_state = create_global_state(
                cv_text="",
                structured_cv=structured_cv,
                job_description_data=job_description_data,
                session_metadata={
                    "personal_info": personal_info,
                    "experience": experience,
                    **kwargs,
                },
            )

            # Execute workflow through workflow manager
            return await self.workflow_manager.execute_workflow(
                WorkflowType.JOB_TAILORED_CV,
                agent_state,
                session_id,
            )

        except Exception as e:
            self.logger.error(
                "Failed to generate job-tailored CV",
                extra={"error": str(e), "session_id": session_id},
            )
            raise RuntimeError(f"Job-tailored CV generation failed: {str(e)}") from e

    async def optimize_cv(
        self, existing_cv: Dict[str, Any], session_id: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Optimize an existing CV.

        Args:
            existing_cv: The existing CV data to optimize
            session_id: Optional session ID
            **kwargs: Additional metadata

        Returns:
            Dict[str, Any]: The workflow execution result
        """
        try:
            # Convert existing_cv to StructuredCV if it's not already
            if isinstance(existing_cv, dict):
                structured_cv = StructuredCV.model_validate(existing_cv)
            else:
                structured_cv = existing_cv

            # Create global state
            agent_state = create_global_state(
                cv_text="",
                structured_cv=structured_cv,
                session_metadata={"optimization_request": True, **kwargs},
            )

            # Execute workflow through workflow manager
            return await self.workflow_manager.execute_workflow(
                WorkflowType.CV_OPTIMIZATION,
                agent_state,
                session_id,
            )

        except Exception as e:
            self.logger.error(
                "Failed to optimize CV",
                extra={"error": str(e), "session_id": session_id},
            )
            raise RuntimeError(f"CV optimization failed: {str(e)}") from e

    async def check_cv_quality(
        self, cv_content: Dict[str, Any], session_id: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Perform quality assurance on CV content.

        Args:
            cv_content: The CV content to check
            session_id: Optional session ID
            **kwargs: Additional metadata

        Returns:
            Dict[str, Any]: The workflow execution result
        """
        try:
            # Convert cv_content to StructuredCV if it's not already
            if isinstance(cv_content, dict):
                structured_cv = StructuredCV.model_validate(cv_content)
            else:
                structured_cv = cv_content

            # Create global state
            agent_state = create_global_state(
                cv_text="",
                structured_cv=structured_cv,
                session_metadata={"quality_check_request": True, **kwargs},
            )

            # Execute workflow through workflow manager
            return await self.workflow_manager.execute_workflow(
                WorkflowType.QUALITY_ASSURANCE,
                agent_state,
                session_id,
            )

        except Exception as e:
            self.logger.error(
                "Failed to check CV quality",
                extra={"error": str(e), "session_id": session_id},
            )
            raise RuntimeError(f"CV quality check failed: {str(e)}") from e

    def cleanup_workflow(self, session_id: str) -> bool:
        """Clean up workflow resources.

        Args:
            session_id: The session ID of the workflow to clean up

        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        try:
            success = self.workflow_manager.cleanup_workflow(session_id)

            if success:
                logger.info(
                    "Workflow cleanup completed", extra={"session_id": session_id}
                )
            else:
                logger.warning(
                    "Workflow cleanup failed or workflow not found",
                    extra={"session_id": session_id},
                )

            return success

        except Exception as e:
            logger.error(
                "Error during workflow cleanup",
                extra={"session_id": session_id, "error": str(e)},
            )
            return False

    # Methods required by TICKET REM-P2-01
    def start_cv_generation(
        self, cv_content: str, job_description: str, user_api_key: Optional[str] = None
    ) -> str:
        """Initializes and triggers the first step of the CV generation workflow.

        This method encapsulates the full sequence of workflow_manager calls:
        1. Creates a new workflow with raw CV text (equivalent to start_workflow)
        2. Immediately triggers the first workflow step (equivalent to resume_workflow)

        Args:
            cv_content: The original CV text content
            job_description: The job description text
            user_api_key: Optional user API key

        Returns:
            str: Session ID for the new workflow

        Raises:
            ValueError: If input validation fails
            RuntimeError: If workflow creation or execution fails
        """
        try:
            # Validate inputs
            if not cv_content or not cv_content.strip():
                raise ValueError("CV content cannot be empty")
            if not job_description or not job_description.strip():
                raise ValueError("Job description cannot be empty")

            logger.info(
                "Starting CV generation workflow with CV parsing and immediate execution",
                extra={
                    "has_user_api_key": user_api_key is not None,
                    "cv_content_length": len(cv_content),
                    "job_description_length": len(job_description),
                },
            )

            # Create new workflow with raw CV text (parsing will be done by workflow node)
            session_id = self.workflow_manager.create_new_workflow(
                cv_text=cv_content, jd_text=job_description
            )

            # Get initial state and immediately trigger first workflow step (equivalent to resume_workflow)
            initial_state = self.workflow_manager.get_workflow_status(session_id)
            if initial_state is None:
                raise RuntimeError(
                    f"Failed to retrieve initial workflow state for session {session_id}"
                )

            # The facade now immediately resumes the workflow after starting it in a background thread
            _run_async_in_thread(
                self.workflow_manager.trigger_workflow_step, session_id, initial_state
            )

            logger.info(
                "CV generation workflow started and first step triggered",
                extra={"session_id": session_id},
            )

            return session_id

        except (ValueError, RuntimeError) as e:
            logger.error(
                "Failed to start CV generation workflow", extra={"error": str(e)}
            )
            raise
        except Exception as e:
            logger.error(
                "Unexpected error during CV generation workflow start",
                extra={"error": str(e)},
            )
            raise RuntimeError(
                f"Unexpected error during workflow start: {str(e)}"
            ) from e

    def get_workflow_state(self, session_id: str) -> Optional[GlobalState]:
        """Retrieve the full, current state of a given workflow.

        This method matches the exact signature required by TICKET REM-P2-01.

        Args:
            session_id: The session ID of the workflow

        Returns:
            Optional[GlobalState]: The current workflow state, or None if not found
        """
        # Delegate to existing method
        return self.get_workflow_status(session_id)

    def provide_user_feedback(self, session_id: str, feedback: UserFeedback) -> None:
        """Provide user feedback to the workflow and resume it.

        This method encapsulates the full sequence of workflow_manager calls:
        1. Submits user feedback to the workflow
        2. Immediately triggers the next workflow step (equivalent to resume_workflow)

        Args:
            session_id: The session ID of the workflow
            feedback: The user feedback to provide

        Raises:
            ValueError: If session not found or feedback invalid
            RuntimeError: If feedback submission or workflow resumption fails
        """
        try:
            # Step 1: Submit feedback
            success = self.submit_user_feedback(session_id, feedback)
            if not success:
                raise RuntimeError(
                    f"Failed to submit user feedback for session {session_id}"
                )

            # Step 2: Get current state and immediately trigger next workflow step (equivalent to resume_workflow)
            current_state = self.workflow_manager.get_workflow_status(session_id)
            if current_state is None:
                raise RuntimeError(
                    f"Failed to retrieve workflow state for session {session_id}"
                )

            # The facade now immediately resumes the workflow after providing feedback in a background thread
            _run_async_in_thread(
                self.workflow_manager.trigger_workflow_step, session_id, current_state
            )

            logger.info(
                "User feedback provided and next workflow step triggered",
                extra={
                    "session_id": session_id,
                    "action": feedback.action.value,
                    "item_id": feedback.item_id,
                },
            )

        except Exception as e:
            logger.error(
                "Failed to provide user feedback or trigger workflow step",
                extra={"session_id": session_id, "error": str(e)},
            )
            raise RuntimeError(f"Failed to provide user feedback: {str(e)}") from e

    async def stream_cv_generation(
        self,
        cv_text: str,
        jd_text: str,
        workflow_type: WorkflowType = WorkflowType.JOB_TAILORED_CV,
        callback_handler=None,
    ):
        """Stream CV generation workflow with real-time updates.

        This method provides streaming capabilities for CV generation,
        yielding intermediate states and progress updates.

        Args:
            cv_text: The original CV text content
            jd_text: The job description text
            workflow_type: Type of workflow to execute
            callback_handler: Optional callback handler for streaming updates

        Yields:
            dict: Workflow state updates during execution

        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If workflow execution fails
        """
        try:
            # Start the workflow
            session_id = self.start_cv_generation(
                cv_content=cv_text, job_description=jd_text
            )

            logger.info(
                "Started streaming CV generation workflow",
                extra={"session_id": session_id},
            )

            # Stream workflow execution
            async for state_update in self.workflow_manager.astream_workflow(
                session_id, callback_handler
            ):
                yield state_update

            logger.info(
                "Completed streaming CV generation workflow",
                extra={"session_id": session_id},
            )

        except Exception as e:
            logger.error(
                "Error during streaming CV generation", extra={"error": str(e)}
            )
            if callback_handler:
                await callback_handler.on_workflow_error(str(e))
            raise RuntimeError(f"Streaming CV generation failed: {str(e)}") from e
