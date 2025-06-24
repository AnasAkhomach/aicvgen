"""CV Analyzer Agent for processing and analyzing CV data."""

import datetime
import json
import os
import time
from typing import Any, Dict, TYPE_CHECKING

from .agent_base import EnhancedAgentBase, AgentResult
from ..config.logging_config import get_structured_logger
from ..config.settings import Settings
from ..models.data_models import AgentIO, AgentDecisionLog
from ..models.cv_analyzer_models import BasicCVInfo, CVAnalyzerNodeResult
from ..models.cv_analysis_result import CVAnalysisResult
from ..models.agent_output_models import CVAnalyzerAgentOutput
from ..models.validation_schemas import validate_agent_input
from ..services.llm_service import EnhancedLLMService

from ..services.progress_tracker import ProgressTracker
from ..templates.content_templates import ContentTemplateManager
from ..utils.agent_error_handling import AgentErrorHandler, with_node_error_handling

logger = get_structured_logger(__name__)

if TYPE_CHECKING:
    from .agent_base import AgentExecutionContext


class CVAnalyzerAgent(EnhancedAgentBase):
    """
    Agent responsible for analyzing the user's CV and extracting relevant information.
    """

    def __init__(
        self,
        llm_service: EnhancedLLMService,
        settings: Settings,
        progress_tracker: ProgressTracker,
        template_manager: ContentTemplateManager,
        name: str = "CVAnalyzerAgent",
        description: str = "Agent responsible for analyzing the user's CV and extracting relevant information",
    ):
        """Initialize the CVAnalyzerAgent with required dependencies.

        Args:
            llm_service: LLM service instance for content analysis.
            settings: Settings/config dependency for prompt loading.
            progress_tracker: Progress tracker service dependency.
            template_manager: The template manager for loading prompts.
            name: The name of the agent.
            description: A description of the agent.
        """
        super().__init__(
            name=name,
            description=description,
            input_schema=AgentIO(
                description="User CV data and optional job description for analysis.",
                required_fields=["user_cv", "job_description", "template_cv_path"],
            ),
            output_schema=AgentIO(
                description="Extracted information from the CV.",
                required_fields=["analysis_results", "extracted_data"],
            ),
            progress_tracker=progress_tracker,
        )

        # Required service dependencies (constructor injection)
        self.llm_service = llm_service
        self.settings = settings
        self.template_manager = template_manager
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

    async def analyze_cv(self, input_data, context=None):
        """Process and analyze the user CV data.

        Args:
            input_data: Dictionary containing user_cv (CVData object or raw text) and job_description
            context: Optional execution context for logging

        Returns:
            dict: Analyzed CV data with extracted information
        """
        print("Executing: CVAnalyzerAgent")

        # Check if a template CV path is provided in the input
        template_cv_path = input_data.get("template_cv_path")

        if template_cv_path and os.path.exists(template_cv_path):
            try:
                with open(template_cv_path, "r", encoding="utf-8") as file:
                    _ = (
                        file.read()
                    )  # Template content loaded but not used in current implementation
                print(f"Loaded template CV from {template_cv_path}")
            except (OSError, IOError, UnicodeDecodeError) as e:
                print(f"Error loading template CV: {e}")

        # Get user CV data from input
        user_cv = input_data.get("user_cv", {})

        raw_cv_text = user_cv.get("raw_text", "")

        if not raw_cv_text:
            print("Warning: Empty CV text provided to CVAnalyzerAgent.")
            # Return a BasicCVInfo instance for empty input
            from ..models.cv_analyzer_models import BasicCVInfo

            return BasicCVInfo(summary="")

        print(f"Analyzing CV...\n Raw Text (first 200 chars): {raw_cv_text[:200]}...")

        # Create a fallback extraction first, in case the LLM fails
        fallback_extraction = self.extract_basic_info(raw_cv_text)

        # Load prompt template from external file
        try:
            prompt_path = self.settings.get_prompt_path("cv_analysis_prompt")
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_template = f.read()
            self.log_decision(
                "Successfully loaded CV analysis prompt template",
                context,
                "template_loading",
                confidence_score=1.0,
            )
        except (OSError, IOError, UnicodeDecodeError) as e:
            self.log_decision(
                f"Error loading CV analysis prompt template: {e}",
                context,
                "template_loading",
                confidence_score=0.0,
            )
            # Fallback to basic prompt
            prompt_template = """
            Analyze the following CV text and extract key information in JSON format.
            CV Text: {raw_cv_text}
            Job Description Context: {job_description}
            """

        # Format the prompt with actual data
        prompt = prompt_template.format(
            raw_cv_text=raw_cv_text,
            job_description=input_data.get("job_description", ""),
        )

        print("Sending prompt to LLM for CV analysis...")
        try:
            # Use centralized JSON generation and parsing
            start_time = time.time()
            extracted_data = await self._generate_and_parse_json(
                prompt=prompt,
                session_id=getattr(context, "session_id", "default"),
                trace_id=getattr(context, "trace_id", "default"),
            )
            elapsed_time = time.time() - start_time

            if elapsed_time > self.timeout:
                print(
                    f"LLM response took too long ({elapsed_time:.2f}s). Using fallback extraction."
                )
                return fallback_extraction

            print("Received response from LLM.")

            # Ensure all expected keys are present, even if empty in the LLM output
            extracted_data.setdefault("summary", "")
            extracted_data.setdefault("experiences", [])
            extracted_data.setdefault("skills", [])
            extracted_data.setdefault("education", [])
            extracted_data.setdefault("projects", [])

            print("Successfully analyzed CV using LLM.")
            return extracted_data

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM response during CV analysis: {e}")
            print("Using fallback extraction instead.")
            # fallback_extraction is a BasicCVInfo, update summary
            fallback_extraction.summary = (
                f"Error parsing CV: {getattr(fallback_extraction, 'summary', '')}"
            )
            return fallback_extraction
        except Exception as e:
            print(f"An unexpected error occurred during CV analysis: {e}")
            print("Using fallback extraction instead.")
            fallback_extraction.summary = (
                f"Error analyzing CV: {getattr(fallback_extraction, 'summary', '')}"
            )
            return fallback_extraction

    async def run_async(
        self, input_data: Any, context: "AgentExecutionContext"
    ) -> "AgentResult":
        """Async run method for consistency with enhanced agent interface."""

        logger = get_structured_logger(__name__)

        try:  # Validate input data using Pydantic schemas
            validation_result = AgentErrorHandler.handle_validation_error(
                lambda: validate_agent_input("cv_analyzer", input_data),
                "CVAnalyzerAgent",
            )
            if not validation_result.success:
                return AgentResult(
                    success=False,
                    output_data=CVAnalyzerAgentOutput(),
                    confidence_score=0.0,
                    error_message=validation_result.error_message,
                    metadata={"agent_type": "cv_analyzer", "validation_error": True},
                )

            # Convert validated Pydantic model back to dict for processing
            # validation_result is likely the validated data itself, not an AgentResult
            input_data = validation_result  # Already validated input
            decision_log = AgentDecisionLog(
                timestamp=datetime.datetime.now().isoformat(),
                agent_name=self.name,
                session_id=getattr(context, "session_id", "unknown"),
                item_id=getattr(context, "current_item_id", None),
                decision_type="validation",
                decision_details="Input validation passed for CVAnalyzerAgent",
                confidence_score=1.0,
                metadata={
                    "input_keys": (
                        list(input_data.keys())
                        if isinstance(input_data, dict)
                        else ["non_dict_input"]
                    )
                },
            )
            logger.log_agent_decision(decision_log)  # Process the CV analysis directly
            if isinstance(input_data, dict):
                result = await self.analyze_cv(input_data, context)
            else:
                # Convert string input to expected format
                formatted_input = {
                    "user_cv": {"raw_text": str(input_data)},
                    "job_description": "",
                }
                result = await self.analyze_cv(formatted_input, context)

            # Wrap result in proper output model
            output_data = CVAnalyzerAgentOutput(analysis_results=result)

            return AgentResult(
                success=True,
                output_data=output_data,
                confidence_score=1.0,
                metadata={"agent_type": "cv_analyzer"},
            )

        except Exception as e:
            # TODO: Narrow exception type in future refactor
            error_result = AgentErrorHandler.handle_general_error(
                e, "CVAnalyzerAgent", context="run_async"
            )
            return error_result

    @with_node_error_handling
    async def run_as_node(self, state) -> Dict[str, Any]:
        """
        Executes the CV analyzer agent as a node within the LangGraph.

        Args:
            state: The current state of the LangGraph workflow.

        Returns:
            Dict[str, Any]: Dictionary slice with cv_analysis_results.
        """
        try:
            from ..agents.agent_base import AgentExecutionContext

            # Create execution context from state
            context = AgentExecutionContext(
                session_id=getattr(state, "session_id", "default"),
                item_id=getattr(state, "current_item_id", None),
                content_type=getattr(state, "content_type", None),
                retry_count=getattr(state, "retry_count", 0),
                metadata=getattr(state, "metadata", {}),
                input_data=getattr(state, "input_data", {}),
                processing_options=getattr(state, "processing_options", {}),
            )

            # Extract input data from state
            input_data = {
                "user_cv": getattr(state, "user_cv", {}),
                "job_description": getattr(state, "job_description", ""),
                "template_cv_path": getattr(state, "template_cv_path", ""),
            }

            # Execute the agent
            result = await self.run_async(input_data, context)

            if result.success:
                # Ensure output is a CVAnalysisResult instance
                if isinstance(result.output_data, dict):
                    cv_analysis = CVAnalysisResult(**result.output_data)
                elif isinstance(result.output_data, CVAnalysisResult):
                    cv_analysis = result.output_data
                else:
                    cv_analysis = None

                return {"cv_analysis_results": cv_analysis}
            else:
                error_msg = result.error_message or "CVAnalyzerAgent failed"
                raise RuntimeError(error_msg)
        except Exception as e:
            logger.error(f"CVAnalyzerAgent failed: {e}", exc_info=True)
            from ..utils.exceptions import AgentExecutionError

            raise AgentExecutionError(
                agent_name="CVAnalyzerAgent", message=str(e)
            ) from e
