"""LangGraph-based CV Generation Workflow.

This module defines the state machine workflow using LangGraph's StateGraph.
It implements granular, item-by-item processing with user feedback loops.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from langgraph.graph import END, StateGraph

from src.agents.agent_base import AgentBase
from src.config.logging_config import get_structured_logger
from src.config.settings import get_config
from src.models.workflow_models import ContentType, UserAction
from src.orchestration.state import AgentState
from src.services.error_recovery import ErrorRecoveryService
from src.utils.node_validation import validate_node_output

logger = get_structured_logger(__name__)


class WorkflowNodes(Enum):
    """Enum for all workflow node names and routing outcomes to ensure type safety."""

    # Main graph nodes
    JD_PARSER = "jd_parser"
    CV_PARSER = "cv_parser"
    RESEARCH = "research"
    CV_ANALYZER = "cv_analyzer"
    SUPERVISOR = "supervisor"
    FORMATTER = "formatter"
    ERROR_HANDLER = "error_handler"

    # Subgraph nodes
    KEY_QUALIFICATIONS_SUBGRAPH = "key_qualifications_subgraph"
    PROFESSIONAL_EXPERIENCE_SUBGRAPH = "professional_experience_subgraph"
    PROJECTS_SUBGRAPH = "projects_subgraph"
    EXECUTIVE_SUMMARY_SUBGRAPH = "executive_summary_subgraph"

    # Workflow sequence section identifiers
    KEY_QUALIFICATIONS = "key_qualifications"
    PROFESSIONAL_EXPERIENCE = "professional_experience"
    PROJECT_EXPERIENCE = "project_experience"
    EXECUTIVE_SUMMARY = "executive_summary"

    # Subgraph internal nodes
    GENERATE = "generate"
    QA = "qa"
    HANDLE_FEEDBACK = "handle_feedback"

    # Routing outcomes
    REGENERATE = "regenerate"
    CONTINUE = "continue"
    ERROR = "error"
    PREPARE_REGENERATION = "prepare_regeneration"


# Define the workflow sequence for sections
WORKFLOW_SEQUENCE = [
    WorkflowNodes.KEY_QUALIFICATIONS.value,
    WorkflowNodes.PROFESSIONAL_EXPERIENCE.value,
    WorkflowNodes.PROJECT_EXPERIENCE.value,
    WorkflowNodes.EXECUTIVE_SUMMARY.value,
]


class CVWorkflowGraph:
    """Manages the CV generation workflow using a state graph."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        job_description_parser_agent: Optional[AgentBase] = None,
        user_cv_parser_agent: Optional[AgentBase] = None,
        research_agent: Optional[AgentBase] = None,
        cv_analyzer_agent: Optional[AgentBase] = None,
        key_qualifications_writer_agent: Optional[AgentBase] = None,
        professional_experience_writer_agent: Optional[AgentBase] = None,
        projects_writer_agent: Optional[AgentBase] = None,
        executive_summary_writer_agent: Optional[AgentBase] = None,
        qa_agent: Optional[AgentBase] = None,
        formatter_agent: Optional[AgentBase] = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())

        # Inject agent instances directly
        self.job_description_parser_agent = job_description_parser_agent
        self.user_cv_parser_agent = user_cv_parser_agent
        self.research_agent = research_agent
        self.cv_analyzer_agent = cv_analyzer_agent
        self.key_qualifications_writer_agent = key_qualifications_writer_agent
        self.professional_experience_writer_agent = professional_experience_writer_agent
        self.projects_writer_agent = projects_writer_agent
        self.executive_summary_writer_agent = executive_summary_writer_agent
        self.qa_agent = qa_agent
        self.formatter_agent = formatter_agent

        self.workflow = self._build_graph()
        self.app = self.workflow.compile()

        logger.info(f"CVWorkflowGraph initialized for session {self.session_id}")

    def _has_meaningful_cv_content(self, structured_cv) -> bool:
        """Check if the structured CV has meaningful content (items in sections).
        
        Args:
            structured_cv: The StructuredCV object to check
            
        Returns:
            bool: True if the CV has sections with items, False otherwise
        """
        if not structured_cv or not hasattr(structured_cv, 'sections'):
            return False
            
        if not structured_cv.sections:
            return False
            
        # Check if any section has items
        for section in structured_cv.sections:
            if hasattr(section, 'items') and section.items:
                return True
            # Also check subsections
            if hasattr(section, 'subsections') and section.subsections:
                for subsection in section.subsections:
                    if hasattr(subsection, 'items') and subsection.items:
                        return True
                        
        return False

    @validate_node_output
    async def jd_parser_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute parser node to process job description."""
        logger.info(f"Executing jd_parser_node")

        # Skip if job description data is already available
        if state.job_description_data:
            logger.info("Job description data already available, skipping parsing")
            return {}

        try:
            if not self.job_description_parser_agent:
                raise RuntimeError("JobDescriptionParserAgent not injected")
            result = await self.job_description_parser_agent.run_as_node(state)
            if isinstance(result, dict):
                return result
            # If result is AgentState, extract only the fields that changed
            return {"job_description_data": result.job_description_data}
        except (KeyError, AttributeError, RuntimeError) as exc:
            logger.error("JD Parser node failed: %s", exc)
            error_msg = f"JobDescriptionParserAgent failed: {str(exc)}"
            return {"error_messages": [*state.error_messages, error_msg]}

    @validate_node_output
    async def cv_parser_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute parser node to process CV."""
        logger.info(f"Executing cv_parser_node")

        # Defensive logging: Track structured_cv.sections before processing
        sections_count_before = (
            len(state.structured_cv.sections) if state.structured_cv else 0
        )
        logger.info(
            f"[DEFENSIVE] cv_parser_node - structured_cv.sections count BEFORE: {sections_count_before}"
        )

        # Skip if structured CV data is already available and has content
        if self._has_meaningful_cv_content(state.structured_cv):
            logger.info("Structured CV data with content already available, skipping parsing")
            return {}

        try:
            if not self.user_cv_parser_agent:
                raise RuntimeError("UserCVParserAgent not injected")
            result = await self.user_cv_parser_agent.run_as_node(state)

            # Defensive logging: Track structured_cv.sections after processing
            if isinstance(result, AgentState):
                sections_count_after = (
                    len(result.structured_cv.sections) if result.structured_cv else 0
                )
                logger.info(
                    f"[DEFENSIVE] cv_parser_node - structured_cv.sections count AFTER: {sections_count_after}"
                )
                # Return only the structured_cv field
                return {"structured_cv": result.structured_cv}
            elif isinstance(result, dict) and "structured_cv" in result:
                structured_cv = result["structured_cv"]
                if structured_cv and hasattr(structured_cv, 'sections'):
                    sections_count_after = len(structured_cv.sections)
                else:
                    sections_count_after = 0
                logger.info(
                    f"[DEFENSIVE] cv_parser_node - structured_cv.sections count AFTER: {sections_count_after}"
                )
                return result
            else:
                return result if isinstance(result, dict) else {}
            
        except (KeyError, AttributeError, RuntimeError) as exc:
            logger.error("CV Parser node failed: %s", exc)
            error_msg = f"UserCVParserAgent failed: {str(exc)}"
            return {"error_messages": [*state.error_messages, error_msg]}

    @validate_node_output
    async def research_node(self, state: AgentState, **kwargs) -> Dict[str, Any]:
        """Execute research node."""
        logger.info("Executing research_node")

        # Defensive logging: Track structured_cv.sections before processing
        sections_count_before = (
            len(state.structured_cv.sections) if state.structured_cv else 0
        )
        logger.info(
            f"[DEFENSIVE] research_node - structured_cv.sections count BEFORE: {sections_count_before}"
        )

        # Skip if research findings are already available
        if state.research_findings:
            logger.info("Research findings already available, skipping research")
            return {}

        if not self.research_agent:
            raise RuntimeError("ResearchAgent not injected")
        result = await self.research_agent.run_as_node(state)

        # Defensive logging: Track structured_cv.sections after processing
        if isinstance(result, AgentState):
            sections_count_after = (
                len(result.structured_cv.sections) if result.structured_cv else 0
            )
            logger.info(
                f"[DEFENSIVE] research_node - structured_cv.sections count AFTER: {sections_count_after}"
            )
            # Return only the research_findings field
            return {"research_findings": result.research_findings}
        elif isinstance(result, dict):
            # Check if structured_cv is in the result dict
            if "structured_cv" in result:
                structured_cv = result["structured_cv"]
                if structured_cv and hasattr(structured_cv, 'sections'):
                    sections_count_after = len(structured_cv.sections)
                else:
                    sections_count_after = 0
                logger.info(
                    f"[DEFENSIVE] research_node - structured_cv.sections count AFTER: {sections_count_after}"
                )
            else:
                # structured_cv should remain unchanged
                logger.info(
                    f"[DEFENSIVE] research_node - structured_cv not in result, should remain: {sections_count_before}"
                )
            return result
        else:
            return {}

    @validate_node_output
    async def cv_analyzer_node(self, state: AgentState, **kwargs) -> Dict[str, Any]:
        """Analyze the user's CV and store results in state.cv_analysis_results."""
        logger.info("Executing cv_analyzer_node")

        # Defensive logging: Track structured_cv.sections before processing
        sections_count_before = (
            len(state.structured_cv.sections) if state.structured_cv else 0
        )
        logger.info(
            f"[DEFENSIVE] cv_analyzer_node - structured_cv.sections count BEFORE: {sections_count_before}"
        )

        # Skip if CV analysis results are already available
        if state.cv_analysis_results:
            logger.info("CV analysis results already available, skipping analysis")
            return {}

        if not self.cv_analyzer_agent:
            logger.warning("CVAnalyzerAgent not injected, returning empty result")
            return {"error_messages": ["CVAnalyzerAgent not available"]}
        result = await self.cv_analyzer_agent.run_as_node(state)

        # Defensive logging: Track structured_cv.sections after processing
        if isinstance(result, AgentState):
            sections_count_after = (
                len(result.structured_cv.sections) if result.structured_cv else 0
            )
            logger.info(
                f"[DEFENSIVE] cv_analyzer_node - structured_cv.sections count AFTER: {sections_count_after}"
            )
            # Return only the cv_analysis_results field
            return {"cv_analysis_results": result.cv_analysis_results}
        elif isinstance(result, dict):
            # Check if structured_cv is in the result dict
            if "structured_cv" in result:
                structured_cv = result["structured_cv"]
                if structured_cv and hasattr(structured_cv, 'sections'):
                    sections_count_after = len(structured_cv.sections)
                else:
                    sections_count_after = 0
                logger.info(
                    f"[DEFENSIVE] cv_analyzer_node - structured_cv.sections count AFTER: {sections_count_after}"
                )
            else:
                # structured_cv should remain unchanged
                logger.info(
                    f"[DEFENSIVE] cv_analyzer_node - structured_cv not in result, should remain: {sections_count_before}"
                )
            return result
        else:
            return {}

    @validate_node_output
    async def key_qualifications_writer_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute key qualifications writer node."""
        logger.info(f"Executing key_qualifications_writer_node")
        if not self.key_qualifications_writer_agent:
            return {"error_messages": ["KeyQualificationsWriterAgent not injected"]}
        
        try:
            result = await self.key_qualifications_writer_agent.run_as_node(state)
            if isinstance(result, dict):
                return result
            elif hasattr(result, 'model_dump'):
                return result.model_dump()
            else:
                return {}
        except Exception as e:
            logger.error(f"KeyQualificationsWriterAgent failed: {e}")
            return {"error_messages": [f"KeyQualificationsWriterAgent failed: {e}"]}

    @validate_node_output
    async def professional_experience_writer_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute professional experience writer node."""
        logger.info(f"Executing professional_experience_writer_node")
        if not self.professional_experience_writer_agent:
            return {"error_messages": ["ProfessionalExperienceWriterAgent not injected"]}
        
        try:
            result = await self.professional_experience_writer_agent.run_as_node(state)
            if isinstance(result, dict):
                return result
            elif hasattr(result, 'model_dump'):
                return result.model_dump()
            else:
                return {}
        except Exception as e:
            logger.error(f"ProfessionalExperienceWriterAgent failed: {e}")
            return {"error_messages": [f"ProfessionalExperienceWriterAgent failed: {e}"]}

    @validate_node_output
    async def projects_writer_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute projects writer node."""
        logger.info(f"Executing projects_writer_node")
        if not self.projects_writer_agent:
            return {"error_messages": ["ProjectsWriterAgent not injected"]}
        
        try:
            result = await self.projects_writer_agent.run_as_node(state)
            if isinstance(result, dict):
                return result
            elif hasattr(result, 'model_dump'):
                return result.model_dump()
            else:
                return {}
        except Exception as e:
            logger.error(f"ProjectsWriterAgent failed: {e}")
            return {"error_messages": [f"ProjectsWriterAgent failed: {e}"]}

    @validate_node_output
    async def executive_summary_writer_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute executive summary writer node."""
        logger.info(f"Executing executive_summary_writer_node")
        
        try:
            if not self.executive_summary_writer_agent:
                logger.warning("ExecutiveSummaryWriterAgent not injected, returning empty result")
                return {"error_messages": ["ExecutiveSummaryWriterAgent not available"]}
            
            result = await self.executive_summary_writer_agent.run_as_node(state)
            
            # Convert result to dictionary format
            if isinstance(result, dict):
                return result
            elif hasattr(result, 'model_dump'):
                # Handle Pydantic models
                return result.model_dump()
            elif hasattr(result, '__dict__'):
                # Handle other objects with __dict__
                return result.__dict__
            else:
                # Fallback for other types
                return {"executive_summary_result": str(result)}
                
        except Exception as e:
            logger.error(f"Error in executive_summary_writer_node: {e}")
            return {"error_messages": [f"Executive summary writer failed: {str(e)}"]}

    @validate_node_output
    async def qa_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute QA node for current item."""
        logger.info(f"Executing qa_node for item: {state.current_item_id}")
        
        try:
            if not self.qa_agent:
                logger.warning("QualityAssuranceAgent not injected, returning empty result")
                return {"error_messages": ["QualityAssuranceAgent not available"]}
            
            result = await self.qa_agent.run_as_node(state)
            
            # Convert result to dictionary format
            if isinstance(result, dict):
                return result
            elif hasattr(result, 'model_dump'):
                # Handle Pydantic models
                return result.model_dump()
            elif hasattr(result, '__dict__'):
                # Handle other objects with __dict__
                return result.__dict__
            else:
                # Fallback for other types
                return {"qa_result": str(result)}
                
        except Exception as e:
            logger.error(f"Error in qa_node: {e}")
            return {"error_messages": [f"QA node failed: {str(e)}"]}

    @validate_node_output
    async def formatter_node(self, state: AgentState, **kwargs) -> Dict[str, Any]:
        """Execute formatter node to generate PDF output."""
        logger.info("Executing formatter_node")
        
        try:
            if not self.formatter_agent:
                logger.warning("FormatterAgent not injected, returning empty result")
                return {"error_messages": ["FormatterAgent not available"]}
            
            result = await self.formatter_agent.run_as_node(state)
            
            # Convert result to dictionary format
            if isinstance(result, dict):
                return result
            elif hasattr(result, 'model_dump'):
                # Handle Pydantic models (AgentState)
                result_dict = result.model_dump()
                # Extract key formatter-specific fields
                formatter_result = {}
                if "final_output_path" in result_dict:
                    formatter_result["final_output_path"] = result_dict["final_output_path"]
                if "error_messages" in result_dict:
                    formatter_result["error_messages"] = result_dict["error_messages"]
                return formatter_result
            elif hasattr(result, '__dict__'):
                # Handle other objects with __dict__
                return result.__dict__
            else:
                # Fallback for other types
                return {"formatter_result": str(result)}
                
        except Exception as e:
            logger.error(f"Error in formatter_node: {e}")
            return {"error_messages": [f"Formatter node failed: {str(e)}"]}

    @validate_node_output
    async def error_handler_node(self, state: AgentState, **kwargs) -> Dict[str, Any]:
        """
        Centralized error handling node that processes agent failures and determines
        recovery actions.
        """
        logger.info("Executing error_handler_node")
        
        try:
            if not state.error_messages:
                logger.warning("Error handler called but no error messages found")
                return {"error_messages": ["No error messages found"]}

            last_error = state.error_messages[-1]
            logger.error("Handling error: %s", last_error)

            error_recovery_service = ErrorRecoveryService()
            # Use current_content_type from state, fallback to QUALIFICATION if not set
            content_type = state.current_content_type or ContentType.QUALIFICATION
            recovery_action = await error_recovery_service.handle_error(
                Exception(last_error),
                state.current_item_id or "unknown",
                content_type,
                self.session_id,
                0,  # retry_count managed at orchestration level
                {
                    "workflow_step": "agent_execution",
                    "trace_id": state.trace_id,
                },
            )
            
            updates = {}
            if recovery_action.strategy.value == "skip_item":
                logger.info(f"Skipping failed item: {state.current_item_id}")
                updates["current_item_id"] = None
            elif recovery_action.strategy.value == "immediate_retry":
                logger.info(f"Marking item for retry: {state.current_item_id}")
            elif recovery_action.strategy.value == "fallback_content":
                logger.info("Using fallback content")
                updates["fallback_content"] = recovery_action.fallback_content
            
            updates["error_messages"] = []
            return updates
            
        except Exception as exc:
            logger.error("Error handler itself failed: %s", exc)
            return {
                "error_messages": [
                    f"Error handler failed: {exc}",
                ]
            }

    def _initialize_supervisor_state(self, state: AgentState) -> Dict[str, Any]:
        """
        Centralized method to initialize supervisor state variables.
        Sets current_section_index and current_item_id based on structured_cv contents.
        
        Args:
            state: The current AgentState
            
        Returns:
            Dict with initialized supervisor state variables
        """
        if state.structured_cv and state.structured_cv.sections:
            sections = state.structured_cv.sections
            
            # Find the first section that has items
            for section_index, section in enumerate(sections):
                if section.items:
                    # Retrieve the ID of the first item in this section
                    first_item_id = str(section.items[0].id)
                    
                    # Set supervisor state variables
                    updates = {
                        "current_section_index": section_index,
                        "current_item_id": first_item_id
                    }
                    logger.info(
                        f"Initialized supervisor state: section_index={section_index}, item_id={first_item_id}"
                    )
                    return updates
            
            # If no sections have items, log warning
            logger.warning(
                "No sections with items found in structured_cv - supervisor state not initialized"
            )
        else:
            logger.info("No structured_cv data available - supervisor state not initialized")
        
        # Return empty dict if initialization conditions not met
        return {}

    @validate_node_output
    async def initialize_supervisor_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Initialize supervisor state by calculating current_section_index and current_item_id.
        This node is responsible for setting up the initial workflow routing state.
        """
        logger.info("Executing initialize_supervisor_node")
        
        try:
            # Initialize current_section_index to 0 if not set
            current_section_index = 0 if state.current_section_index is None else state.current_section_index
            
            # Initialize current_item_id based on the first section's first item
            current_item_id = None
            if state.structured_cv and state.structured_cv.sections:
                # Get the first section that has items
                for section in state.structured_cv.sections:
                    if section.items:
                        current_item_id = str(section.items[0].id)
                        break
            
            logger.info(f"Initialized supervisor state: section_index={current_section_index}, item_id={current_item_id}")
            
            return {
                "current_section_index": current_section_index,
                "current_item_id": current_item_id
            }
            
        except Exception as e:
            logger.error(f"Error in initialize_supervisor_node: {e}")
            return {"error_messages": [f"Initialize supervisor node failed: {str(e)}"]}

    @validate_node_output
    async def supervisor_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Supervisor node to route to the correct content generation subgraph
        or to the formatter node if all content is generated.
        """
        logger.info(
            f"Executing supervisor_node. Current section index: {state.current_section_index}"
        )
        print(f"\n=== SUPERVISOR NODE DEBUG ===")
        print(f"Current section index: {state.current_section_index}")
        print(f"WORKFLOW_SEQUENCE length: {len(WORKFLOW_SEQUENCE)}")
        print(f"Last executed node: {state.node_execution_metadata.get('last_executed_node')}")
        print(f"Error messages: {state.error_messages}")
        print(f"User feedback: {state.user_feedback}")
        print(f"Automated mode: {getattr(state, 'automated_mode', False)}")

        try:
            # Defensive check: Validate current_item_id before routing to content generation
            if (
                not hasattr(state, "current_item_id")
                or state.current_item_id is None
                or not str(state.current_item_id).strip()
            ):
                error_msg = f"Supervisor node: Invalid or missing current_item_id for section index {state.current_section_index} - cannot route to content generation"
                logger.error(error_msg)
                print(f"ERROR: {error_msg}")
                # Route to error handler for graceful handling
                updated_metadata = {
                    **state.node_execution_metadata,
                    "next_node": WorkflowNodes.ERROR_HANDLER.value,
                    "last_executed_node": WorkflowNodes.SUPERVISOR.value,
                }
                return {
                    "error_messages": [error_msg],
                    "workflow_status": "ERROR",
                    "node_execution_metadata": updated_metadata,
                }

            # Check if we're returning from a completed subgraph
            # If the last executed node was a subgraph, increment the section index
            last_node = state.node_execution_metadata.get("last_executed_node")
            current_index = state.current_section_index

            if last_node and last_node.endswith("_subgraph") and not state.user_feedback:
                # A subgraph completed successfully, move to next section
                current_index += 1
                logger.info(
                    f"Subgraph {last_node} completed, incrementing section index to {current_index}"
                )
                print(f"Incrementing section index from {state.current_section_index} to {current_index}")

            print(f"Current index after increment check: {current_index}")
            print(f"Checking if {current_index} >= {len(WORKFLOW_SEQUENCE)}")

            if state.error_messages:
                logger.warning("Errors detected in state, routing to error handler.")
                next_node = WorkflowNodes.ERROR_HANDLER.value
                print(f"Routing to ERROR_HANDLER due to errors")
            elif (
                state.user_feedback and state.user_feedback.action == UserAction.REGENERATE
            ):
                logger.info("User requested regeneration, staying in current section.")
                next_section_key = WORKFLOW_SEQUENCE[current_index]
                # Map section keys to subgraph node names
                section_to_subgraph = {
                    WorkflowNodes.KEY_QUALIFICATIONS.value: WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value,
                    WorkflowNodes.PROFESSIONAL_EXPERIENCE.value: WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value,
                    WorkflowNodes.PROJECT_EXPERIENCE.value: WorkflowNodes.PROJECTS_SUBGRAPH.value,
                    WorkflowNodes.EXECUTIVE_SUMMARY.value: WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value,
                }
                next_node = section_to_subgraph.get(
                    next_section_key, WorkflowNodes.ERROR_HANDLER.value
                )
                print(f"Routing to {next_node} for regeneration")
            elif current_index >= len(WORKFLOW_SEQUENCE):
                logger.info("All content sections processed, routing to formatter.")
                next_node = WorkflowNodes.FORMATTER.value
                print(f"Routing to FORMATTER - all sections complete")
            else:
                next_section_key = WORKFLOW_SEQUENCE[current_index]
                logger.info(f"Routing to {next_section_key} subgraph.")
                # Map section keys to subgraph node names
                section_to_subgraph = {
                    WorkflowNodes.KEY_QUALIFICATIONS.value: WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value,
                    WorkflowNodes.PROFESSIONAL_EXPERIENCE.value: WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value,
                    WorkflowNodes.PROJECT_EXPERIENCE.value: WorkflowNodes.PROJECTS_SUBGRAPH.value,
                    WorkflowNodes.EXECUTIVE_SUMMARY.value: WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value,
                }
                next_node = section_to_subgraph.get(
                    next_section_key, WorkflowNodes.ERROR_HANDLER.value
                )
                print(f"Routing to {next_node} for section {next_section_key}")

            print(f"Final routing decision: {next_node}")
            print(f"=== END SUPERVISOR NODE DEBUG ===\n")

            # Update node_execution_metadata with the next_node decision and mark current node as last executed
            updated_metadata = {
                **state.node_execution_metadata,
                "next_node": next_node,
                "last_executed_node": WorkflowNodes.SUPERVISOR.value,
            }

            return {
                "node_execution_metadata": updated_metadata,
                "current_section_index": current_index,
            }
            
        except Exception as e:
            logger.error(f"Error in supervisor_node: {e}")
            return {"error_messages": [f"Supervisor node failed: {str(e)}"]}

    @validate_node_output
    async def handle_feedback_node(self, state: AgentState, **kwargs) -> Dict[str, Any]:
        """Handles user feedback to refine content."""
        logger.info(
            f"Executing handle_feedback_node for item: {getattr(state, 'current_item_id', 'unknown')}"
        )

        try:
            # Strict validation: Check if user feedback exists and has valid item_id
            if state.user_feedback:
                if (
                    not state.user_feedback.item_id
                    or not state.user_feedback.item_id.strip()
                ):
                    error_msg = "Feedback received with no item_id - workflow cannot proceed without valid item identification"
                    logger.error(error_msg)
                    # Raise ValueError to prevent advancement and ensure strict validation
                    raise ValueError(error_msg)

            # If no user feedback yet, check automated mode or set status to awaiting feedback
            if not state.user_feedback:
                # In automated mode, auto-approve and continue without user feedback
                if getattr(state, 'automated_mode', False):
                    logger.info(
                        "Automated mode enabled, auto-approving content and continuing workflow"
                    )
                    return {
                        "user_feedback": None,
                        "workflow_status": "PROCESSING",
                        "ui_display_data": {},
                    }
                
                logger.info(
                    "No user feedback found, setting workflow status to AWAITING_FEEDBACK"
                )

                # Prepare UI display data based on current section and content
                current_section = (
                    WORKFLOW_SEQUENCE[state.current_section_index]
                    if state.current_section_index < len(WORKFLOW_SEQUENCE)
                    else "unknown"
                )

                ui_data = {
                    "section": current_section,
                    "item_id": state.current_item_id,
                    "content_type": (
                        state.current_content_type.value
                        if state.current_content_type
                        else "unknown"
                    ),
                    "timestamp": datetime.now().isoformat(),
                    "requires_feedback": True,
                    "feedback_options": ["approve", "regenerate"],
                }

                # Add section-specific content to UI data
                if hasattr(state, "structured_cv") and state.structured_cv:
                    # Find the relevant section content for display
                    for section in state.structured_cv.sections:
                        if section.name == current_section:
                            ui_data["content_preview"] = {
                                "section_name": section.name,
                                "items_count": len(section.items),
                                "items": [
                                    {
                                        "id": item.id,
                                        "content": (
                                            item.content[:200] + "..."
                                            if len(item.content) > 200
                                            else item.content
                                        ),
                                    }
                                    for item in section.items[:3]
                                ],  # Show first 3 items as preview
                            }
                            break

                return {
                    "workflow_status": "AWAITING_FEEDBACK",
                    "ui_display_data": ui_data,
                }

            # Process existing user feedback
            if state.user_feedback.action == UserAction.REGENERATE:
                logger.info(f"User feedback: Regenerate for item {state.current_item_id}")
                return {
                    "user_feedback": None,
                    "workflow_status": "PROCESSING",
                    "ui_display_data": {},
                }
            elif state.user_feedback.action == UserAction.APPROVE:
                logger.info(f"User feedback: Approved for item {state.current_item_id}")
                return {
                    "user_feedback": None,
                    "workflow_status": "PROCESSING",
                    "ui_display_data": {},
                }

            # Fallback: return empty dict
            return {}
            
        except Exception as e:
            logger.error(f"Error in handle_feedback_node: {e}")
            return {"error_messages": [f"Handle feedback node failed: {str(e)}"]}

    @validate_node_output
    async def mark_subgraph_completion_node(
        self, state: AgentState, **kwargs
    ) -> Dict[str, Any]:
        """
        Node to mark that a subgraph has completed successfully.
        This updates the metadata to indicate which subgraph just finished.
        """
        try:
            current_section_key = WORKFLOW_SEQUENCE[state.current_section_index]
            subgraph_name = f"{current_section_key}_subgraph"

            logger.info(f"Marking completion of {subgraph_name}")

            # Update metadata to indicate this subgraph completed
            updated_metadata = {
                **state.node_execution_metadata,
                "last_executed_node": subgraph_name,
            }

            return {"node_execution_metadata": updated_metadata}
            
        except Exception as e:
            logger.error(f"Error in mark_subgraph_completion_node: {e}")
            return {"error_messages": [f"Mark subgraph completion node failed: {str(e)}"]}

    def _route_after_content_generation(self, state: Dict[str, Any]) -> str:
        """
        Router function for subgraphs after content generation and QA.
        Determines if content needs regeneration or if the subgraph should end.
        """
        agent_state = AgentState.model_validate(state)

        if agent_state.error_messages:
            logger.warning(
                "Errors detected in state, routing to error handler from content generation subgraph."
            )
            return WorkflowNodes.ERROR.value

        if (
            agent_state.user_feedback
            and agent_state.user_feedback.action == UserAction.REGENERATE
        ):
            logger.info("User requested regeneration, looping back within subgraph.")
            return WorkflowNodes.REGENERATE.value

        # If no regeneration requested and no errors, route to completion marker
        logger.info(
            "Content approved or no regeneration requested, marking completion."
        )
        return "MARK_COMPLETION"

    def _route_from_supervisor(self, state: Dict[str, Any]) -> str:
        """
        Router function for the main graph, called by the supervisor node.
        Determines the next step based on the 'next_node' field set by the supervisor.
        """
        agent_state = AgentState.model_validate(state)
        return agent_state.node_execution_metadata.get(
            "next_node", WorkflowNodes.ERROR_HANDLER.value
        )

    @validate_node_output
    async def _entry_router_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Entry router node that determines whether to start from initial parsing
        or skip directly to content generation based on existing data.
        """
        logger.info("Executing entry router to determine workflow starting point")

        try:
            # Check if initial parsing steps are already completed
            has_job_data = state.job_description_data is not None
            has_cv_data = state.structured_cv is not None
            has_research = state.research_findings is not None
            has_analysis = state.cv_analysis_results is not None

            if has_job_data and has_cv_data and has_research and has_analysis:
                logger.info("Initial parsing already completed, routing to supervisor")
                next_node = WorkflowNodes.SUPERVISOR.value
                # Initialize supervisor state since we're skipping parsing nodes
                supervisor_updates = self._initialize_supervisor_state(state)
                # Merge supervisor updates with routing decision
                updated_metadata = {**state.node_execution_metadata, "entry_route": next_node}
                return {**supervisor_updates, "node_execution_metadata": updated_metadata}
            else:
                logger.info("Starting from initial parsing steps")
                next_node = WorkflowNodes.JD_PARSER.value

            # Update metadata with routing decision
            updated_metadata = {**state.node_execution_metadata, "entry_route": next_node}
            
            return {"node_execution_metadata": updated_metadata}
            
        except Exception as e:
            logger.error(f"Error in _entry_router_node: {e}")
            return {"error_messages": [f"Entry router node failed: {str(e)}"]}

    def _route_from_entry(self, state: Dict[str, Any]) -> str:
        """
        Router function for the entry point.
        Determines whether to start from JD_PARSER or SUPERVISOR.
        """
        agent_state = AgentState.model_validate(state)
        return agent_state.node_execution_metadata.get(
            "entry_route", WorkflowNodes.JD_PARSER.value
        )

    def _build_key_qualifications_subgraph(self) -> StateGraph:
        """Builds the subgraph for Key Qualifications generation."""
        workflow = StateGraph(AgentState)
        workflow.add_node(
            WorkflowNodes.GENERATE.value, self.key_qualifications_writer_node
        )
        workflow.add_node(WorkflowNodes.QA.value, self.qa_node)
        workflow.add_node(
            WorkflowNodes.HANDLE_FEEDBACK.value, self.handle_feedback_node
        )
        workflow.add_node("MARK_COMPLETION", self.mark_subgraph_completion_node)

        workflow.set_entry_point(WorkflowNodes.GENERATE.value)
        workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.QA.value)
        workflow.add_edge(WorkflowNodes.QA.value, WorkflowNodes.HANDLE_FEEDBACK.value)

        workflow.add_conditional_edges(
            WorkflowNodes.HANDLE_FEEDBACK.value,
            self._route_after_content_generation,
            {
                WorkflowNodes.REGENERATE.value: WorkflowNodes.GENERATE.value,
                "MARK_COMPLETION": "MARK_COMPLETION",
                WorkflowNodes.ERROR.value: END,
            },
        )
        workflow.add_edge("MARK_COMPLETION", END)
        return workflow

    def _build_professional_experience_subgraph(self) -> StateGraph:
        """Builds the subgraph for Professional Experience generation."""
        workflow = StateGraph(AgentState)
        workflow.add_node(
            WorkflowNodes.GENERATE.value, self.professional_experience_writer_node
        )
        workflow.add_node(WorkflowNodes.QA.value, self.qa_node)
        workflow.add_node(
            WorkflowNodes.HANDLE_FEEDBACK.value, self.handle_feedback_node
        )
        workflow.add_node("MARK_COMPLETION", self.mark_subgraph_completion_node)

        workflow.set_entry_point(WorkflowNodes.GENERATE.value)
        workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.QA.value)
        workflow.add_edge(WorkflowNodes.QA.value, WorkflowNodes.HANDLE_FEEDBACK.value)

        workflow.add_conditional_edges(
            WorkflowNodes.HANDLE_FEEDBACK.value,
            self._route_after_content_generation,
            {
                WorkflowNodes.REGENERATE.value: WorkflowNodes.GENERATE.value,
                "MARK_COMPLETION": "MARK_COMPLETION",
                WorkflowNodes.ERROR.value: END,
            },
        )
        workflow.add_edge("MARK_COMPLETION", END)
        return workflow

    def _build_projects_subgraph(self) -> StateGraph:
        """Builds the subgraph for Projects generation."""
        workflow = StateGraph(AgentState)
        workflow.add_node(WorkflowNodes.GENERATE.value, self.projects_writer_node)
        workflow.add_node(WorkflowNodes.QA.value, self.qa_node)
        workflow.add_node(
            WorkflowNodes.HANDLE_FEEDBACK.value, self.handle_feedback_node
        )
        workflow.add_node("MARK_COMPLETION", self.mark_subgraph_completion_node)

        workflow.set_entry_point(WorkflowNodes.GENERATE.value)
        workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.QA.value)
        workflow.add_edge(WorkflowNodes.QA.value, WorkflowNodes.HANDLE_FEEDBACK.value)

        workflow.add_conditional_edges(
            WorkflowNodes.HANDLE_FEEDBACK.value,
            self._route_after_content_generation,
            {
                WorkflowNodes.REGENERATE.value: WorkflowNodes.GENERATE.value,
                "MARK_COMPLETION": "MARK_COMPLETION",
                WorkflowNodes.ERROR.value: END,
            },
        )
        workflow.add_edge("MARK_COMPLETION", END)
        return workflow

    def _build_executive_summary_subgraph(self) -> StateGraph:
        """Builds the subgraph for Executive Summary generation."""
        workflow = StateGraph(AgentState)
        workflow.add_node(
            WorkflowNodes.GENERATE.value, self.executive_summary_writer_node
        )
        workflow.add_node(WorkflowNodes.QA.value, self.qa_node)
        workflow.add_node(
            WorkflowNodes.HANDLE_FEEDBACK.value, self.handle_feedback_node
        )
        workflow.add_node("MARK_COMPLETION", self.mark_subgraph_completion_node)

        workflow.set_entry_point(WorkflowNodes.GENERATE.value)
        workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.QA.value)
        workflow.add_edge(WorkflowNodes.QA.value, WorkflowNodes.HANDLE_FEEDBACK.value)

        workflow.add_conditional_edges(
            WorkflowNodes.HANDLE_FEEDBACK.value,
            self._route_after_content_generation,
            {
                WorkflowNodes.REGENERATE.value: WorkflowNodes.GENERATE.value,
                "MARK_COMPLETION": "MARK_COMPLETION",
                WorkflowNodes.ERROR.value: END,
            },
        )
        workflow.add_edge("MARK_COMPLETION", END)
        return workflow

    def _build_graph(self) -> StateGraph:
        """Build and return the refactored CV workflow graph with supervisor and subgraphs."""
        workflow = StateGraph(AgentState)

        # Add main graph nodes
        workflow.add_node(WorkflowNodes.JD_PARSER.value, self.jd_parser_node)
        workflow.add_node(WorkflowNodes.CV_PARSER.value, self.cv_parser_node)
        workflow.add_node(WorkflowNodes.RESEARCH.value, self.research_node)
        workflow.add_node(WorkflowNodes.CV_ANALYZER.value, self.cv_analyzer_node)
        workflow.add_node("INITIALIZE_SUPERVISOR", self.initialize_supervisor_node)
        workflow.add_node(WorkflowNodes.SUPERVISOR.value, self.supervisor_node)
        workflow.add_node(WorkflowNodes.FORMATTER.value, self.formatter_node)
        workflow.add_node(WorkflowNodes.ERROR_HANDLER.value, self.error_handler_node)

        # Add subgraphs as nodes (compile them first)
        workflow.add_node(
            WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value,
            self._build_key_qualifications_subgraph().compile(),
        )
        workflow.add_node(
            WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value,
            self._build_professional_experience_subgraph().compile(),
        )
        workflow.add_node(
            WorkflowNodes.PROJECTS_SUBGRAPH.value,
            self._build_projects_subgraph().compile(),
        )
        workflow.add_node(
            WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value,
            self._build_executive_summary_subgraph().compile(),
        )

        # Add a conditional entry point router
        workflow.add_node("ENTRY_ROUTER", self._entry_router_node)
        workflow.set_entry_point("ENTRY_ROUTER")

        # Define main graph edges
        workflow.add_conditional_edges(
            "ENTRY_ROUTER",
            self._route_from_entry,
            {
                WorkflowNodes.JD_PARSER.value: WorkflowNodes.JD_PARSER.value,
                WorkflowNodes.SUPERVISOR.value: "INITIALIZE_SUPERVISOR",
            },
        )
        workflow.add_edge(WorkflowNodes.JD_PARSER.value, WorkflowNodes.CV_PARSER.value)
        workflow.add_edge(WorkflowNodes.CV_PARSER.value, WorkflowNodes.RESEARCH.value)
        workflow.add_edge(WorkflowNodes.RESEARCH.value, WorkflowNodes.CV_ANALYZER.value)
        workflow.add_edge(
            WorkflowNodes.CV_ANALYZER.value, "INITIALIZE_SUPERVISOR"
        )
        workflow.add_edge("INITIALIZE_SUPERVISOR", WorkflowNodes.SUPERVISOR.value)

        # Conditional routing from supervisor
        workflow.add_conditional_edges(
            WorkflowNodes.SUPERVISOR.value,
            self._route_from_supervisor,
            {
                WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value: WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value,
                WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value: WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value,
                WorkflowNodes.PROJECTS_SUBGRAPH.value: WorkflowNodes.PROJECTS_SUBGRAPH.value,
                WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value: WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value,
                WorkflowNodes.FORMATTER.value: WorkflowNodes.FORMATTER.value,
                WorkflowNodes.ERROR_HANDLER.value: WorkflowNodes.ERROR_HANDLER.value,
            },
        )

        # After each subgraph, return to the supervisor to determine the next section
        workflow.add_edge(
            WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value,
            WorkflowNodes.SUPERVISOR.value,
        )
        workflow.add_edge(
            WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value,
            WorkflowNodes.SUPERVISOR.value,
        )
        workflow.add_edge(
            WorkflowNodes.PROJECTS_SUBGRAPH.value, WorkflowNodes.SUPERVISOR.value
        )
        workflow.add_edge(
            WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value,
            WorkflowNodes.SUPERVISOR.value,
        )

        # Formatter ends the workflow
        workflow.add_edge(WorkflowNodes.FORMATTER.value, END)

        # Error handler terminates the workflow
        workflow.add_edge(WorkflowNodes.ERROR_HANDLER.value, END)

        return workflow

    async def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the workflow with the given inputs."""
        return await self.app.ainvoke(
            inputs, config={"configurable": {"thread_id": self.session_id}}
        )

    async def trigger_workflow_step(self, state: AgentState) -> AgentState:
        """Trigger the next workflow step based on current state.

        This method allows resuming workflow execution from a paused state
        by streaming through workflow steps and saving state after each step.
        The method stops streaming when workflow_status becomes "AWAITING_FEEDBACK".

        Args:
            state: Current workflow state

        Returns:
            Updated state after executing workflow steps
        """
        logger.info(f"Triggering workflow step for session {self.session_id}")

        # Safety check for input state
        if state is None:
            logger.error("Cannot trigger workflow step: input state is None")
            return None

        try:
            # Set workflow status to processing
            state = state.set_workflow_status("PROCESSING")
            state = state.set_ui_display_data({})

            # Convert state to dict for workflow invocation
            state_dict = state.model_dump()

            # Stream through workflow steps
            async for step in self.app.astream(
                state_dict, config={"configurable": {"thread_id": self.session_id}}
            ):
                # Update state from step result
                if isinstance(step, dict):
                    # Extract the actual state from the step result
                    # LangGraph astream yields {node_name: result} format
                    for node_name, node_result in step.items():
                        if isinstance(node_result, dict):
                            # Only update state if node_result has content
                            # Empty dictionaries should not modify the state
                            if node_result:
                                try:
                                    # Use model_copy with update pattern for correct state management
                                    updated_state = state.model_copy(update=node_result)
                                    if updated_state is not None:
                                        state = updated_state
                                        logger.debug(f"Updated state from node {node_name} with keys: {list(node_result.keys())}")
                                    else:
                                        logger.warning(f"Node {node_name} model_copy returned None, keeping original state")
                                except Exception as copy_error:
                                    logger.error(f"Error updating state from node {node_name}: {copy_error}")
                                    # Keep the original state if model_copy fails
                            else:
                                logger.debug(f"Node {node_name} returned empty dict, no state update needed")
                        elif hasattr(node_result, "model_dump"):
                            state = node_result
                        else:
                            # Only assign if node_result is a valid AgentState-like object
                            # Prevent None or invalid objects from becoming the state
                            if node_result is not None and hasattr(node_result, 'workflow_status'):
                                state = node_result
                            else:
                                logger.warning(f"Node {node_name} returned invalid result type {type(node_result)}, keeping original state")
                        break  # Take the first (and typically only) result

                # Save state to JSON file after each step
                self._save_state_to_file(state)

                # Safety check before accessing state attributes
                if state is None:
                    logger.error("State became None during workflow execution")
                    break

                # Check if we should pause for user feedback
                if hasattr(state, 'workflow_status') and state.workflow_status == "AWAITING_FEEDBACK":
                    logger.info(
                        f"Workflow paused for feedback in session {self.session_id}"
                    )
                    break

                # Check if workflow completed or errored
                if hasattr(state, 'workflow_status') and state.workflow_status in ["COMPLETED", "ERROR"]:
                    logger.info(
                        f"Workflow finished with status {state.workflow_status} in session {self.session_id}"
                    )
                    break

            return state

        except Exception as exc:
            logger.error(f"Error triggering workflow step: {exc}")
            error_msg = f"Workflow step execution failed: {str(exc)}"

            # Safety check for state before accessing its attributes
            if state is None:
                logger.error("State is None during error handling, cannot update error information")
                return None

            # Update state with error information
            error_messages = list(state.error_messages) if state.error_messages else []
            error_messages.append(error_msg)
            state = state.set_workflow_status("ERROR")
            # Create a new state with updated error messages
            state_dict = state.model_dump()
            state_dict["error_messages"] = error_messages
            state = AgentState(**state_dict)

            # Save error state to file
            try:
                self._save_state_to_file(state)
            except Exception as save_error:
                logger.error(f"Failed to save error state: {save_error}")

            return state

    def _save_state_to_file(self, state: AgentState) -> None:
        """Save the current state to a JSON file.

        Args:
            state: The state to save
        """
        try:
            # Safety check to prevent NoneType errors
            if state is None:
                logger.error("Cannot save state: state is None")
                return
                
            # Check if state has the required methods (more flexible than isinstance)
            if not hasattr(state, 'model_dump_json'):
                logger.error(f"Cannot save state: object does not have model_dump_json method, got {type(state)}")
                return
                
            config = get_config()
            sessions_dir = config.paths.project_root / "instance" / "sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)

            session_file = sessions_dir / f"{self.session_id}.json"

            # Write state to file
            with open(session_file, "w", encoding="utf-8") as f:
                f.write(state.model_dump_json(indent=2))

            logger.debug(f"State saved to {session_file}")

        except Exception as exc:
            logger.error(f"Failed to save state to file: {exc}")
            raise
