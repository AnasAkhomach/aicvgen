"""CV Analyzer Agent for processing and analyzing CV data."""

import datetime
import json
import os
import time
from typing import Any, Dict, TYPE_CHECKING

from src.agents.agent_base import EnhancedAgentBase
from src.config.logging_config import get_structured_logger
from src.config.settings import get_config
from src.models.data_models import AgentDecisionLog
from src.models.validation_schemas import validate_agent_input, ValidationError
from src.services.llm_service import get_llm_service

if TYPE_CHECKING:
    from .agent_base import AgentExecutionContext

from .agent_base import AgentResult


class CVAnalyzerAgent(EnhancedAgentBase):
    """
    Agent responsible for analyzing the user's CV and extracting relevant information.
    """

    def __init__(self, name: str, description: str):
        """
        Initializes the CVAnalyzerAgent.

        Args:
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
        )
        self.llm_service = get_llm_service()  # Use enhanced LLM service
        self.timeout = 30  # Maximum wait time in seconds

        # Initialize settings for prompt loading
        self.settings = get_config()

    def extract_basic_info(self, cv_text: str) -> Dict[str, Any]:
        """
        Extract basic information from CV text without using LLM for a fallback.

        Args:
            cv_text: Raw CV text

        Returns:
            Dictionary with basic extracted information
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

        return result

    def analyze_cv(self, input_data, context=None):
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
            return {
                "summary": "",
                "experiences": [],
                "skills": [],
                "education": [],
                "projects": [],
            }

        print(f"Analyzing CV...\n Raw Text (first 200 chars): {raw_cv_text[:200]}...")

        # Create a fallback extraction first, in case the LLM fails
        fallback_extraction = self.extract_basic_info(raw_cv_text)

        # Load prompt template from external file
        logger = get_structured_logger(__name__)
        try:
            prompt_path = self.settings.get_prompt_path("cv_analysis_prompt")
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_template = f.read()
            self.log_decision(
                "Successfully loaded CV analysis prompt template",
                getattr(context, "current_item_id", None) if context else None,
                "template_loading",
                confidence_score=1.0,
                metadata={"template_path": str(prompt_path)},
            )
        except (OSError, IOError, UnicodeDecodeError) as e:
            self.log_decision(
                f"Error loading CV analysis prompt template: {e}",
                getattr(context, "current_item_id", None) if context else None,
                "template_loading",
                confidence_score=0.0,
                metadata={"error": str(e)},
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
            # Set a timeout for the LLM call
            start_time = time.time()
            llm_response = self.llm.generate_content(prompt)
            elapsed_time = time.time() - start_time

            if elapsed_time > self.timeout:
                print(
                    f"LLM response took too long ({elapsed_time:.2f}s). Using fallback extraction."
                )
                return fallback_extraction

            print("Received response from LLM.")

            # Attempt to parse the JSON response (handle markdown formatting)
            json_string = llm_response.strip()
            if json_string.startswith("```json"):
                json_string = json_string[len("```json") :].strip()
                if json_string.endswith("```"):
                    json_string = json_string[: -len("```")].strip()

            # Load the JSON string into a Python dictionary
            extracted_data = json.loads(json_string)

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
            fallback_extraction["summary"] = (
                f"Error parsing CV: {fallback_extraction.get('summary', '')}"
            )
            return fallback_extraction
        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError) as e:
            print(f"An unexpected error occurred during CV analysis: {e}")
            print("Using fallback extraction instead.")
            fallback_extraction["summary"] = (
                f"Error analyzing CV: {fallback_extraction.get('summary', '')}"
            )
            return fallback_extraction

    async def run_async(
        self, input_data: Any, context: "AgentExecutionContext"
    ) -> "AgentResult":
        """Async run method for consistency with enhanced agent interface."""

        logger = get_structured_logger(__name__)

        try:
            # Validate input data using Pydantic schemas
            try:
                validated_input = validate_agent_input("cv_analyzer", input_data)
                # Convert validated Pydantic model back to dict for processing
                input_data = validated_input.model_dump()
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
                logger.log_agent_decision(decision_log)
            except ValidationError as ve:
                decision_log = AgentDecisionLog(
                    timestamp=datetime.datetime.now().isoformat(),
                    agent_name=self.name,
                    session_id=getattr(context, "session_id", "unknown"),
                    item_id=getattr(context, "current_item_id", None),
                    decision_type="validation",
                    decision_details=f"Input validation failed for CVAnalyzerAgent: {ve.message}",
                    confidence_score=0.0,
                    metadata={"error": ve.message, "error_type": "ValidationError"},
                )
                logger.log_agent_decision(decision_log)
                return AgentResult(
                    success=False,
                    output_data={"error": f"Input validation failed: {ve.message}"},
                    confidence_score=0.0,
                    error_message=f"Input validation failed: {ve.message}",
                    metadata={"agent_type": "cv_analyzer", "validation_error": True},
                )
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                decision_log = AgentDecisionLog(
                    timestamp=datetime.datetime.now().isoformat(),
                    agent_name=self.name,
                    session_id=getattr(context, "session_id", "unknown"),
                    item_id=getattr(context, "current_item_id", None),
                    decision_type="validation",
                    decision_details=f"Input validation error for CVAnalyzerAgent: {str(e)}",
                    confidence_score=0.0,
                    metadata={"error": str(e), "error_type": type(e).__name__},
                )
                logger.log_agent_decision(decision_log)
                return AgentResult(
                    success=False,
                    output_data={"error": f"Input validation error: {str(e)}"},
                    confidence_score=0.0,
                    error_message=f"Input validation error: {str(e)}",
                    metadata={"agent_type": "cv_analyzer", "validation_error": True},
                )

            # Process the CV analysis directly
            if isinstance(input_data, dict):
                result = self.analyze_cv(input_data, context)
            else:
                # Convert string input to expected format
                formatted_input = {
                    "user_cv": {"raw_text": str(input_data)},
                    "job_description": "",
                }
                result = self.analyze_cv(formatted_input, context)

            return AgentResult(
                success=True,
                output_data=result,
                confidence_score=1.0,
                metadata={"agent_type": "cv_analyzer"},
            )

        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            return AgentResult(
                success=False,
                output_data={},
                confidence_score=0.0,
                error_message=str(e),
                metadata={"agent_type": "cv_analyzer"},
            )
