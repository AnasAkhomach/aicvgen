"""CV Analyzer Agent for processing and analyzing CV data."""

import json
from typing import Any, Dict, Optional, TYPE_CHECKING
from pydantic import ValidationError

from .agent_base import EnhancedAgentBase, AgentResult
from ..config.logging_config import get_structured_logger
from ..models.data_models import AgentIO, AgentDecisionLog
from ..models.cv_analyzer_models import BasicCVInfo
from ..models.cv_analysis_result import CVAnalysisResult
from ..models.agent_output_models import CVAnalyzerAgentOutput
from ..utils.exceptions import LLMResponseParsingError, AgentExecutionError

logger = get_structured_logger(__name__)

if TYPE_CHECKING:
    from .agent_base import AgentExecutionContext


class CVAnalyzerAgent(EnhancedAgentBase):
    """
    Agent responsible for analyzing the user's CV and extracting relevant information.
    """

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        """Initialize the CVAnalyzerAgent with required dependencies.

        Args:
            config: Dictionary containing llm_service, settings, progress_tracker, template_manager, name, and description.
        """
        super().__init__(
            name=config.get("name", "CVAnalyzerAgent"),
            description=config.get(
                "description",
                "Agent responsible for analyzing the user's CV and extracting relevant information",
            ),
            input_schema=AgentIO(
                description="User CV data and optional job description for analysis.",
                required_fields=["user_cv", "job_description", "template_cv_path"],
            ),
            output_schema=AgentIO(
                description="Extracted information from the CV.",
                required_fields=["analysis_results", "extracted_data"],
            ),
            progress_tracker=config["progress_tracker"],
        )
        self.llm_service = config["llm_service"]
        self.settings = config["settings"]
        self.template_manager = config["template_manager"]
        self.timeout = 30  # Maximum wait time in seconds

    def extract_basic_info(self, cv_text: str) -> "BasicCVInfo":
        """
        Extract basic information from CV text without using LLM for a fallback.

        Args:
            cv_text: Raw CV text

        Returns:
            BasicCVInfo: Pydantic model with basic extracted information
        """
        lines = cv_text.split("\n")
        result = {
            "summary": "",
            "experiences": [],
            "skills": [],
            "education": [],
            "projects": [],
        }

        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to identify sections
            lower_line = line.lower()
            if "experience" in lower_line and (line.endswith(":") or line.isupper()):
                current_section = "experiences"
                continue
            elif "skill" in lower_line and (line.endswith(":") or line.isupper()):
                current_section = "skills"
                continue
            elif "education" in lower_line and (line.endswith(":") or line.isupper()):
                current_section = "education"
                continue
            elif "project" in lower_line and (
                line.endswith(":") or line.isupper() or line.startswith("#")
            ):
                current_section = "projects"
                continue
            elif "summary" in lower_line and (line.endswith(":") or line.isupper()):
                current_section = "summary"
                continue

            # Add content to current section
            if current_section:
                if line.startswith("-") or line.startswith("•"):
                    if current_section == "summary":
                        result[current_section] += f" {line[1:].strip()}"
                    else:
                        result[current_section].append(line[1:].strip())
                elif current_section == "summary":
                    result[current_section] += f" {line}"
                else:
                    result[current_section].append(line)

        # If we couldn't identify any summary, use the first line as name
        if not result["summary"] and lines:
            result["summary"] = f"CV for {lines[0].strip()}"

        # Instead of returning dict, return BasicCVInfo model
        return BasicCVInfo(**result)

    async def analyze_cv(
        self, input_data: Dict[str, Any], context: "AgentExecutionContext" = None
    ) -> CVAnalyzerAgentOutput:
        """Process and analyze the user CV data.

        Args:
            input_data: Dictionary containing user_cv (CVData object or raw text)
                        and job_description.
            context: Optional execution context for logging.

        Returns:
            CVAnalyzerAgentOutput: The structured output of the analysis.

        Raises:
            LLMResponseParsingError: If the LLM output cannot be parsed or validated.
            ValueError: If the input CV text is empty.
            AgentExecutionError: For other unexpected errors during agent execution.
        """
        logger.info("Executing CVAnalyzerAgent.analyze_cv", extra={"context": context})

        user_cv = input_data.get("user_cv", {})
        raw_cv_text = user_cv.get("raw_text", "")

        if not raw_cv_text:
            logger.warning(
                "Empty CV text provided to CVAnalyzerAgent.", extra={"context": context}
            )
            raise ValueError("Input CV text cannot be empty.")

        logger.debug(
            f"Analyzing CV... Raw Text (first 200 chars): {raw_cv_text[:200]}...",
            extra={"context": context},
        )

        fallback_extraction = self.extract_basic_info(raw_cv_text)

        try:
            prompt_template = self.template_manager.get_template("cv_analysis_prompt")
            self.log_decision(
                message="Successfully loaded CV analysis prompt template",
                context=context,
                decision_type="template_loading",
                confidence_score=1.0,
            )
        except Exception as e:
            logger.error(
                f"Error loading CV analysis prompt template: {e}",
                extra={"context": context},
                exc_info=True,
            )
            raise AgentExecutionError(
                self.name, f"Could not load required prompt template: {e}"
            ) from e

        prompt = prompt_template.format(cv_text=raw_cv_text)

        llm_response_content = ""
        try:
            llm_response = await self.llm_service.invoke_async(prompt, context=context)
            llm_response_content = llm_response.content if llm_response else ""

            cleaned_json_str = self._extract_json_from_llm_response(
                llm_response_content
            )
            # Use cleaned_json_str directly in subsequent logic
            if not cleaned_json_str:
                raise LLMResponseParsingError(
                    "No JSON found in LLM response.", raw_response=llm_response_content
                )
            parsed_json = json.loads(cleaned_json_str)
            analysis_result = CVAnalysisResult.model_validate(parsed_json)

            self.log_decision(
                message="Successfully parsed and validated LLM response.",
                context=context,
                decision_type="llm_response_parsing",
                confidence_score=0.95,
            )

            return CVAnalyzerAgentOutput(
                analysis_results=analysis_result,
                extracted_data=fallback_extraction,
            )

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(
                "Failed to parse or validate LLM response for CV analysis.",
                extra={
                    "context": context,
                    "error_message": str(e),
                    "raw_response": llm_response_content,
                },
                exc_info=True,
            )
            raise LLMResponseParsingError(
                message=f"Failed to parse or validate LLM response. Error: {e}",
                raw_response=llm_response_content,
            ) from e
        except LLMResponseParsingError:
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during CV analysis: {e}",
                extra={"context": context},
                exc_info=True,
            )
            raise AgentExecutionError(
                self.name, f"An unexpected error occurred: {e}"
            ) from e

    async def run_async(
        self, input_data: Dict[str, Any], context: "AgentExecutionContext"
    ) -> AgentResult:
        """
        Executes the CV analysis workflow, strictly enforcing return contracts.
        On error, propagates exceptions instead of returning a failed AgentResult.
        This method implements the contract required by the orchestration layer.
        """
        analysis_output = await self.analyze_cv(input_data, context)
        return AgentResult(
            success=True,
            output_data=analysis_output,
            confidence_score=0.9,
        )

    async def run_as_node(self, state):
        """Stub implementation to satisfy abstract base class for testing."""
        raise NotImplementedError(
            "run_as_node is not implemented for CVAnalyzerAgent in this context."
        )

    def _extract_json_from_llm_response(self, response_text: str) -> Optional[str]:
        """Extracts a JSON object from a string, even if it's embedded in other text."""
        # Example implementation (replace with actual logic):
        import re

        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            return match.group(0)
        return None

    def log_decision(
        self,
        message: str,
        context: Optional["AgentExecutionContext"] = None,
        decision_type: str = "processing",
        confidence_score: Optional[float] = None,
    ):
        """
        Logs a decision or action taken by the agent with structured logging.
        """
        from datetime import datetime

        decision_log = AgentDecisionLog(
            timestamp=datetime.now().isoformat(),
            agent_name=self.name,
            session_id=(
                getattr(context, "trace_id", "unknown") if context else "unknown"
            ),
            item_id=getattr(context, "current_item_id", None) if context else None,
            decision_type=decision_type,
            decision_details=message,
            confidence_score=confidence_score,
            metadata={
                "content_type": (
                    getattr(context, "content_type", None).value
                    if context and getattr(context, "content_type", None) is not None
                    else None
                ),
                "retry_count": getattr(context, "retry_count", 0) if context else 0,
                "execution_count": getattr(self, "execution_count", 0),
                "success_rate": getattr(self, "success_count", 0)
                / max(getattr(self, "execution_count", 1), 1),
            },
        )
        # Only log if logger has log_agent_decision
        if hasattr(self.logger, "log_agent_decision"):
            self.logger.log_agent_decision(decision_log)
        else:
            self.logger.info(f"Decision: {decision_log}")
