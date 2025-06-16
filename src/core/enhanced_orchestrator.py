"""Enhanced Orchestrator for AI CV Generator.

Orchestrator refactored to use a compiled LangGraph application for workflow execution.
"""

import logging
import time
from typing import Optional, Dict, Any

from src.core.state_manager import StateManager
from src.models.data_models import StructuredCV, JobDescriptionData, ItemStatus
from src.orchestration.state import AgentState
from src.orchestration.cv_workflow_graph import cv_graph_app  # Import the compiled graph
from src.agents.research_agent import ResearchAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.services.llm_service import get_llm_service
from src.services.vector_db import get_enhanced_vector_db
from src.utils.exceptions import (
    WorkflowPreconditionError,
    LLMResponseParsingError,
    AgentExecutionError,
    ConfigurationError,
    StateManagerError,
    ValidationError
)


logger = logging.getLogger(__name__)

class EnhancedOrchestrator:
    """
    A thin wrapper around the compiled LangGraph application.
    Manages state translation between the UI and the graph.
    """

    def __init__(self, state_manager: StateManager):
        """
        Initializes the orchestrator with the compiled LangGraph app.

        Args:
            state_manager: The state manager for the current session.
        """
        self.state_manager = state_manager
        self.workflow_app = cv_graph_app

        # Initialize LLM service
        self.llm_service = get_llm_service()

        # Initialize Research and QA agents
        vector_db = get_enhanced_vector_db()
        self.research_agent = ResearchAgent(
            name="ResearchAgent",
            description="Agent for populating vector store with CV content",
            vector_db=vector_db
        )

        self.quality_assurance_agent = QualityAssuranceAgent(
            name="QualityAssuranceAgent",
            description="Agent for quality assurance of generated content"
        )

        logger.info("EnhancedOrchestrator initialized with compiled LangGraph application and MVP agents.")

    async def initialize_workflow(self) -> None:
        """
        Initialize the workflow by running the research agent to populate the vector store.
        This should be called before processing any items.
        
        Raises:
            WorkflowPreconditionError: If job description data or structured CV data is missing
        """
        try:
            logger.info("Initializing workflow with research agent...")

            # Get current job description and CV data
            job_description_data = self.state_manager.get_job_description_data()
            structured_cv = self.state_manager.get_structured_cv()

            # Validate that required data is available before proceeding
            if not job_description_data:
                raise WorkflowPreconditionError("Job description data is required to initialize workflow.")
            
            if not structured_cv:
                raise WorkflowPreconditionError("Structured CV data is required to initialize workflow.")

            # Run research agent to populate vector store
            research_input = {
                "job_description_data": job_description_data.model_dump() if hasattr(job_description_data, 'model_dump') else job_description_data,
                "structured_cv": structured_cv.model_dump() if structured_cv else {}
            }

            # Use run_as_node for LangGraph integration
            # Create AgentState for run_as_node compatibility
            from src.orchestration.state import AgentState
            from src.models.data_models import StructuredCV, JobDescriptionData
            
            # Create proper StructuredCV and JobDescriptionData objects
            structured_cv = research_input.get("structured_cv") or StructuredCV()
            job_desc_data = research_input.get("job_description_data")
            if not job_desc_data or isinstance(job_desc_data, dict):
                job_desc_data = JobDescriptionData(raw_text=research_input.get("job_description", ""))
            
            agent_state = AgentState(
                structured_cv=structured_cv,
                job_description_data=job_desc_data
            )
            
            node_result = await self.research_agent.run_as_node(agent_state)
            research_result = node_result.get("output_data", {})

            if research_result.get("success", False):
                logger.info("Research agent successfully populated vector store")
            else:
                logger.warning(f"Research agent completed with warnings: {research_result.get('message', 'Unknown issue')}")

        except WorkflowPreconditionError as wpe:
            logger.error(f"Workflow precondition error during initialization: {str(wpe)}")
            raise  # Re-raise WorkflowPreconditionError to prevent workflow from continuing
        except Exception as e:
            logger.error(f"Error initializing workflow with research agent: {e}", exc_info=True)
            raise  # Re-raise other exceptions to prevent workflow from continuing

    async def execute_full_workflow(self) -> AgentState:
        """
        Executes the entire workflow from the beginning.

        Returns:
            The final state of the workflow after execution.
        """
        structured_cv = self.state_manager.get_structured_cv()
        job_description_data = self.state_manager.get_job_description_data()

        if not structured_cv or not job_description_data:
            raise ValueError("Initial CV and Job Description data must be loaded before execution.")

        initial_state = AgentState(
            structured_cv=structured_cv,
            job_description_data=job_description_data
        )

        logger.info("Invoking LangGraph workflow with initial state.")
        logger.info(f"Initial state keys: {list(initial_state.model_dump().keys())}")
        logger.info(f"Initial state structured_cv present: {bool(initial_state.structured_cv)}")
        logger.info(f"Initial state job_description_data present: {bool(initial_state.job_description_data)}")
        
        # The .ainvoke() method runs the graph asynchronously until it hits an END state or needs input
        final_state_dict = await self.workflow_app.ainvoke(initial_state.model_dump())
        
        logger.info(f"Final state dict keys: {list(final_state_dict.keys()) if isinstance(final_state_dict, dict) else 'Not a dict'}")
        logger.info(f"Final state dict type: {type(final_state_dict)}")
        
        final_state = AgentState.model_validate(final_state_dict)

        # Persist the final state
        if final_state.structured_cv:
            self.state_manager.set_structured_cv(final_state.structured_cv)

        return final_state

    async def process_single_item(self, item_id: str) -> AgentState:
        """
        Process a single item using the LangGraph workflow.

        Args:
            item_id: The ID of the specific item to process

        Returns:
            Updated AgentState after processing
        """
        try:
            logger.info(f"Orchestrator processing single item: {item_id}")

            # Get current state from state manager
            structured_cv = self.state_manager.get_structured_cv()
            job_description_data = self.state_manager.get_job_description_data()

            if not structured_cv or not job_description_data:
                logger.error("Cannot process item: CV or Job Description data is missing from state.")
                return AgentState(
                    structured_cv=structured_cv or StructuredCV(),
                    job_description_data=job_description_data,
                    error_messages=["CV or Job Description data is missing from state."]
                )

            # Update status to indicate processing
            self.state_manager.update_subsection_status(item_id, ItemStatus.GENERATED)

            # Create the current state object for LangGraph
            current_state = AgentState(
                structured_cv=structured_cv,
                job_description_data=job_description_data,
                current_item_id=item_id,
                current_section_key="experience"  # This would be more dynamic in a full app
            )

            logger.info(f"Invoking LangGraph workflow to process single item: {item_id}")

            # The graph's conditional logic will route this to the correct node
            final_state_dict = await self.workflow_app.ainvoke(current_state.model_dump())
            final_state = AgentState.model_validate(final_state_dict)

            # Persist the final state
            if final_state.structured_cv:
                self.state_manager.set_structured_cv(final_state.structured_cv)

            # Update status based on success/failure
            if final_state.error_messages:
                logger.error(f"LangGraph workflow failed for item {item_id}: {final_state.error_messages}")
                self.state_manager.update_subsection_status(item_id, ItemStatus.GENERATION_FAILED)
            else:
                logger.info(f"LangGraph workflow successfully processed item: {item_id}")
                self.state_manager.update_subsection_status(item_id, ItemStatus.GENERATED)

            return final_state

        except (ValidationError, WorkflowPreconditionError) as ve:
            logger.error(f"Validation error processing item {item_id}: {ve}")
            self.state_manager.update_subsection_status(item_id, ItemStatus.GENERATION_FAILED)
            return AgentState(
                structured_cv=self.state_manager.get_structured_cv() or StructuredCV(),
                job_description_data=self.state_manager.get_job_description_data(),
                current_item_id=item_id,
                error_messages=[f"Validation error processing item {item_id}: {str(ve)}"]
            )
        except (LLMResponseParsingError, AgentExecutionError) as ae:
            logger.error(f"Agent execution error processing item {item_id}: {ae}")
            self.state_manager.update_subsection_status(item_id, ItemStatus.GENERATION_FAILED)
            return AgentState(
                structured_cv=self.state_manager.get_structured_cv() or StructuredCV(),
                job_description_data=self.state_manager.get_job_description_data(),
                current_item_id=item_id,
                error_messages=[f"Agent execution error processing item {item_id}: {str(ae)}"]
            )
        except (ConfigurationError, StateManagerError) as se:
            logger.error(f"System error processing item {item_id}: {se}")
            self.state_manager.update_subsection_status(item_id, ItemStatus.GENERATION_FAILED)
            return AgentState(
                structured_cv=self.state_manager.get_structured_cv() or StructuredCV(),
                job_description_data=self.state_manager.get_job_description_data(),
                current_item_id=item_id,
                error_messages=[f"System error processing item {item_id}: {str(se)}"]
            )
        except Exception as e:
            logger.error(f"Unexpected error processing item {item_id}: {e}", exc_info=True)
            self.state_manager.update_subsection_status(item_id, ItemStatus.GENERATION_FAILED)
            return AgentState(
                structured_cv=self.state_manager.get_structured_cv() or StructuredCV(),
                job_description_data=self.state_manager.get_job_description_data(),
                current_item_id=item_id,
                error_messages=[f"Unexpected error processing item {item_id}: {str(e)}"]
            )

    async def _run_quality_assurance(self, structured_cv: StructuredCV, item_id: str) -> Dict[str, Any]:
        """
        Run quality assurance on the generated CV content.

        Args:
            structured_cv: The CV structure to check
            item_id: The ID of the item being processed

        Returns:
            Dictionary containing QA results and potentially updated CV
        """
        try:
            logger.info(f"Running quality assurance for item: {item_id}")

            # Get job description data for context
            job_description_data = self.state_manager.get_job_description_data()

            # Prepare QA input
            qa_input = {
                "structured_cv": structured_cv.model_dump(),
                "job_description_data": job_description_data.model_dump() if job_description_data else {}
            }

            # Use run_as_node for LangGraph integration
            # Create AgentState for run_as_node compatibility
            from src.orchestration.state import AgentState
            agent_state = AgentState(
                job_description_data=qa_input.get("job_description_data", {}),
                structured_cv=qa_input.get("structured_cv", {}),
                current_stage="quality_assurance",
                metadata={"agent_type": "quality_assurance"}
            )
            
            node_result = await self.quality_assurance_agent.run_as_node(agent_state)
            qa_result = node_result.get("output_data", {})

            # Log QA results
            quality_checks = qa_result.get("quality_check_results", {})
            summary = quality_checks.get("summary", {})

            logger.info(f"QA completed for item {item_id}: "
                       f"Items checked: {summary.get('total_items', 0)}, "
                       f"Issues found: {summary.get('total_issues', 0)}, "
                       f"Score: {summary.get('overall_score', 'N/A')}")

            # Update item metadata with QA results if available
            if quality_checks and hasattr(self.state_manager, 'update_item_metadata'):
                try:
                    self.state_manager.update_item_metadata(item_id, {
                        "qa_score": summary.get('overall_score'),
                        "qa_issues": summary.get('total_issues', 0),
                        "qa_timestamp": time.time()
                    })
                except Exception as e:
                    logger.warning(f"Could not update item metadata with QA results: {e}")

            return qa_result

        except (ValidationError, WorkflowPreconditionError) as ve:
            logger.error(f"Validation error in quality assurance for item {item_id}: {ve}")
            return {
                "quality_check_results": {"validation_error": str(ve)},
                "updated_structured_cv": structured_cv
            }
        except (LLMResponseParsingError, AgentExecutionError) as ae:
            logger.error(f"Agent execution error in quality assurance for item {item_id}: {ae}")
            return {
                "quality_check_results": {"agent_error": str(ae)},
                "updated_structured_cv": structured_cv
            }
        except (ConfigurationError, StateManagerError) as se:
            logger.error(f"System error in quality assurance for item {item_id}: {se}")
            return {
                "quality_check_results": {"system_error": str(se)},
                "updated_structured_cv": structured_cv
            }
        except Exception as e:
            logger.error(f"Unexpected error in quality assurance for item {item_id}: {e}", exc_info=True)
            return {
                "quality_check_results": {"unexpected_error": str(e)},
                "updated_structured_cv": structured_cv
            }