# In src/frontend/workflow_controller.py
import asyncio
import threading
import uuid
import streamlit as st

from src.config.logging_config import get_logger
from src.error_handling.exceptions import AgentExecutionError, ConfigurationError
from src.orchestration.state import GlobalState, UserFeedback
from src.models.data_models import UserAction

# Initialize logger
logger = get_logger(__name__)


class WorkflowController:
    """
    Centralized controller for managing workflow thread and asyncio event loop.

    This class encapsulates all threading and asyncio logic for background workflow execution,
    providing a clean interface for starting workflows and submitting user feedback.
    """

    def __init__(self, workflow_manager):
        """
        Initialize the WorkflowController with a workflow manager.

        Args:
            workflow_manager: The WorkflowManager instance to use for workflow operations
        """
        self.workflow_manager = workflow_manager
        self._background_thread = None
        self._background_loop = None
        self._is_running = False

        # Start the background thread and event loop immediately
        self._start_background_thread()

    def _start_background_thread(self):
        """
        Start the background thread with a persistent event loop.
        """
        if self._is_running:
            logger.warning("Background thread already running")
            return

        logger.info("Starting WorkflowController background thread")

        self._background_thread = threading.Thread(
            target=self._run_background_event_loop,
            daemon=True
        )
        self._background_thread.start()
        self._is_running = True

    def _run_background_event_loop(self):
        """
        The target function for the background thread.
        Creates and runs a persistent asyncio event loop.
        """
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Store loop reference for access from main thread
            self._background_loop = loop
            self.workflow_manager._background_loop = loop

            logger.info("Background event loop started and ready")

            # Keep the loop running to handle submitted tasks
            loop.run_forever()

        except Exception as e:
            logger.error(f"Error in background event loop: {e}", exc_info=True)
        finally:
            self._is_running = False
            logger.info("Background event loop stopped")

    def start_generation(self, initial_state: GlobalState, workflow_session_id: str):
        """
        Start CV generation workflow with the given initial state.

        Args:
            initial_state: The initial GlobalState to start the workflow with
            workflow_session_id: The workflow session ID
        """
        if not self._is_running or not self._background_loop:
            logger.error("Background event loop not available")
            st.error("Workflow controller not ready")
            return

        # Set processing flags
        trace_id = str(uuid.uuid4())
        initial_state["trace_id"] = trace_id
        st.session_state.is_processing = True
        st.session_state.workflow_error = None
        st.session_state.just_finished = False

        logger.info(
            "Starting CV generation workflow",
            extra={"trace_id": trace_id, "workflow_session_id": workflow_session_id}
        )

        try:
            # Submit the initial workflow task to the background event loop
            future = asyncio.run_coroutine_threadsafe(
                self._execute_initial_workflow(initial_state, workflow_session_id, trace_id),
                self._background_loop
            )

            # Don't wait for the result here - let it run in background
            logger.info("Initial workflow task submitted to background loop")

        except Exception as e:
            logger.error(f"Error starting generation: {e}", exc_info=True)
            st.session_state.is_processing = False
            st.session_state.workflow_error = e
            st.error(f"Failed to start workflow: {e}")

    async def _execute_initial_workflow(self, initial_state: GlobalState, workflow_session_id: str, trace_id: str):
        """
        Execute the initial workflow step asynchronously.

        Args:
            initial_state: The initial GlobalState
            workflow_session_id: The workflow session ID
            trace_id: The trace ID for logging
        """
        try:
            logger.info(
                "Executing initial workflow step",
                extra={"trace_id": trace_id, "workflow_session_id": workflow_session_id}
            )

            # Execute the workflow step using WorkflowManager
            result_state = await self.workflow_manager.trigger_workflow_step(
                session_id=workflow_session_id,
                agent_state=initial_state
            )

            # Update session state with results
            st.session_state.agent_state = result_state

            logger.info(
                "Initial workflow completed successfully",
                extra={"trace_id": trace_id, "workflow_session_id": workflow_session_id}
            )

        except (AgentExecutionError, ConfigurationError, RuntimeError) as e:
            logger.error(
                f"Initial workflow failed: {e}",
                extra={"trace_id": trace_id, "workflow_session_id": workflow_session_id, "error_type": type(e).__name__},
                exc_info=True
            )
            st.session_state.workflow_error = e
        finally:
            st.session_state.is_processing = False
            st.session_state.just_finished = True
            # Signal UI refresh after background workflow update
            st.session_state.needs_rerun = True

    def submit_user_feedback(self, action: str, item_id: str, workflow_session_id: str) -> bool:
        """
        Submit user feedback and trigger workflow continuation.

        Args:
            action: The user action ('accept' or 'regenerate')
            item_id: The ID of the item being acted upon
            workflow_session_id: The workflow session ID

        Returns:
            bool: True if feedback was submitted successfully, False otherwise
        """
        if not self._is_running or not self._background_loop:
            logger.error("Background event loop not available")
            st.error("Workflow controller not ready")
            return False

        # Map UI actions to feedback actions
        if action == "accept":
            feedback_action = "approve"
            user_action = UserAction.APPROVE
            feedback_text = "User approved the item."
            st.success("✅ Item approved")
        elif action == "regenerate":
            feedback_action = "regenerate"
            user_action = UserAction.REGENERATE
            feedback_text = "User requested to regenerate the item."
            st.info("🔄 Regenerating item...")
        else:
            st.error(f"Unknown action: {action}")
            return False

        try:
            # Create user feedback
            user_feedback = UserFeedback(
                action=user_action,
                item_id=item_id,
                feedback_text=feedback_text
            )

            # Step 1: Send feedback to workflow manager
            feedback_sent = self.workflow_manager.send_feedback(workflow_session_id, user_feedback)
            if not feedback_sent:
                st.error("Failed to send feedback to workflow")
                return False

            # Step 2: Get updated agent state after feedback
            updated_agent_state = self.workflow_manager.get_workflow_status(workflow_session_id)
            if not updated_agent_state:
                st.error("Failed to get updated workflow state")
                return False

            # Step 3: Submit continuation task to the persistent event loop
            future = asyncio.run_coroutine_threadsafe(
                self.workflow_manager.trigger_workflow_step(workflow_session_id, updated_agent_state),
                self._background_loop
            )

            # Wait for the result with a timeout
            result_state = future.result(timeout=30)  # 30 second timeout

            # Update session state with result
            st.session_state.agent_state = result_state
            # Signal UI refresh after background workflow update
            st.session_state.needs_rerun = True
            st.success(f"Workflow resumed successfully after {feedback_action}")

            return True

        except Exception as e:
            logger.error(f"Error submitting user feedback: {e}", exc_info=True)
            st.error(f"Failed to process {action} action: {e}")
            return False

    def is_ready(self) -> bool:
        """
        Check if the controller is ready to handle requests.

        Returns:
            bool: True if the background loop is running and ready
        """
        return self._is_running and self._background_loop is not None

    def shutdown(self):
        """
        Gracefully shutdown the background thread and event loop.

        This method:
        1. Cancels all pending asyncio tasks in the background loop
        2. Safely stops the event loop
        3. Waits for the background thread to exit cleanly
        """
        if not self._is_running or not self._background_loop:
            logger.debug("WorkflowController already shutdown or not running")
            return

        logger.info("Starting graceful shutdown of WorkflowController")

        try:
            # Cancel all pending tasks in the background loop
            def cancel_all_tasks():
                try:
                    # Get all tasks in the loop
                    tasks = asyncio.all_tasks(loop=self._background_loop)

                    # Cancel each task (except the current one if any)
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                            logger.debug(f"Cancelled task: {task}")

                    logger.info(f"Cancelled {len(tasks)} pending tasks")

                    # Stop the event loop
                    self._background_loop.stop()

                except Exception as e:
                    logger.error(f"Error during task cancellation: {e}", exc_info=True)
                    # Still try to stop the loop
                    self._background_loop.stop()

            # Execute the cancellation in the background loop thread-safely
            self._background_loop.call_soon_threadsafe(cancel_all_tasks)

            # Wait for the background thread to finish
            if self._background_thread and self._background_thread.is_alive():
                logger.info("Waiting for background thread to exit...")
                self._background_thread.join(timeout=10)  # 10 second timeout

                if self._background_thread.is_alive():
                    logger.warning("Background thread did not exit within timeout")
                else:
                    logger.info("Background thread exited successfully")

            self._is_running = False
            self._background_loop = None
            self._background_thread = None

            logger.info("WorkflowController graceful shutdown complete")

        except Exception as e:
            logger.error(f"Error during WorkflowController shutdown: {e}", exc_info=True)
            # Force cleanup even if there were errors
            self._is_running = False
            self._background_loop = None
            self._background_thread = None

    def __del__(self):
        """
        Destructor to ensure graceful shutdown when the object is garbage collected.

        This is triggered when the Streamlit session ends and the WorkflowController
        instance is no longer referenced, providing automatic cleanup of resources.
        """
        try:
            if self._is_running:
                logger.info("WorkflowController being garbage collected, triggering shutdown")
                self.shutdown()
        except Exception as e:
            # Use print instead of logger since logging might not be available during cleanup
            print(f"Error in WorkflowController.__del__: {e}")