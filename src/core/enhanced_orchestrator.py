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
from src.services.llm import LLM
from src.services.vector_db import get_enhanced_vector_db


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
        self.llm = LLM()

        # Initialize Research and QA agents
        vector_db = get_enhanced_vector_db()
        self.research_agent = ResearchAgent(
            name="ResearchAgent",
            description="Agent for populating vector store with CV content",
            llm=self.llm,
            vector_db=vector_db
        )

        self.quality_assurance_agent = QualityAssuranceAgent(
            name="QualityAssuranceAgent",
            description="Agent for quality assurance of generated content",
            llm=self.llm
        )

        logger.info("EnhancedOrchestrator initialized with compiled LangGraph application and MVP agents.")

    def initialize_workflow(self) -> None:
        """
        Initialize the workflow by running the research agent to populate the vector store.
        This should be called before processing any items.
        
        Raises:
            ValueError: If job description data or structured CV data is missing
        """
        try:
            logger.info("Initializing workflow with research agent...")

            # Get current job description and CV data
            job_description_data = self.state_manager.get_job_description_data()
            structured_cv = self.state_manager.get_structured_cv()

            # Validate that required data is available before proceeding
            if not job_description_data:
                raise ValueError("Job description data is missing. Cannot initialize workflow without job description data.")
            
            if not structured_cv:
                raise ValueError("Structured CV data is missing. Cannot initialize workflow without CV data.")

            # Run research agent to populate vector store
            research_input = {
                "job_description_data": job_description_data.model_dump() if hasattr(job_description_data, 'model_dump') else job_description_data,
                "structured_cv": structured_cv.model_dump() if structured_cv else {}
            }

            research_result = self.research_agent.run(research_input)

            if research_result.get("success", False):
                logger.info("Research agent successfully populated vector store")
            else:
                logger.warning(f"Research agent completed with warnings: {research_result.get('message', 'Unknown issue')}")

        except ValueError as ve:
            logger.error(f"Validation error during workflow initialization: {str(ve)}")
            raise  # Re-raise ValueError to prevent workflow from continuing
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
        # The .ainvoke() method runs the graph asynchronously until it hits an END state or needs input
        final_state_dict = await self.workflow_app.ainvoke(initial_state.model_dump())
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

        except Exception as e:
            logger.error(f"Error in orchestrator processing single item {item_id}: {e}", exc_info=True)
            # Revert status to show failure
            self.state_manager.update_subsection_status(item_id, ItemStatus.GENERATION_FAILED)
            return AgentState(
                structured_cv=self.state_manager.get_structured_cv() or StructuredCV(),
                job_description_data=self.state_manager.get_job_description_data(),
                current_item_id=item_id,
                error_messages=[f"Error processing item {item_id}: {str(e)}"]
            )

    def _run_quality_assurance(self, structured_cv: StructuredCV, item_id: str) -> Dict[str, Any]:
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

            # Run quality assurance
            qa_result = self.quality_assurance_agent.run(qa_input)

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

        except Exception as e:
            logger.error(f"Error running quality assurance for item {item_id}: {e}", exc_info=True)
            return {
                "quality_check_results": {"error": str(e)},
                "updated_structured_cv": structured_cv
            }