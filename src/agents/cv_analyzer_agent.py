from src.agents.agent_base import AgentBase
from src.core.state_manager import AgentIO, CVData, JobDescriptionData
from typing import Dict, Any, List
from src.services.llm import LLM  # Import LLM
import json  # Import json
import time
import os


class CVAnalyzerAgent(AgentBase):
    """
    Agent responsible for analyzing the user's CV and extracting relevant information.
    """

    def __init__(self, name: str, description: str, llm: LLM):
        """
        Initializes the CVAnalyzerAgent.

        Args:
            name: The name of the agent.
            description: A description of the agent.
            llm: The LLM instance to use for parsing.
        """
        super().__init__(
            name=name,
            description=description,
            input_schema=AgentIO(
                input={
                    "user_cv": CVData,
                    "job_description": JobDescriptionData,  # May need job description for context
                    "template_cv_path": str,
                },
                output=Dict[str, Any],  # Define a more specific output schema later
                description="User CV data and optional job description for analysis.",
            ),
            output_schema=AgentIO(
                input={"user_cv": CVData, "job_description": JobDescriptionData},
                output=Dict[str, Any],  # Define a more specific output schema later
                description="Extracted information from the CV.",
            ),
        )
        self.llm = llm  # Store the LLM instance
        self.timeout = 30  # Maximum wait time in seconds

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

    def run(self, input_data):
        """Process and analyze the user CV data.

        Args:
            input_data: Dictionary containing user_cv (CVData object or raw text) and job_description

        Returns:
            dict: Analyzed CV data with extracted information
        """
        print("Executing: CVAnalyzerAgent")

        # Check if a template CV path is provided in the input
        template_cv_path = input_data.get("template_cv_path")
        template_content = None

        if template_cv_path and os.path.exists(template_cv_path):
            try:
                with open(template_cv_path, "r", encoding="utf-8") as file:
                    template_content = file.read()
                print(f"Loaded template CV from {template_cv_path}")
            except Exception as e:
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

        # Craft the prompt for the LLM
        # Include job description for context to help prioritize relevant info
        prompt = f"""
        Analyze the following CV text and extract the key sections and information.
        Provide the output in JSON format with the following keys:
        "summary": The professional summary or objective statement.
        "experiences": A list of work experiences, each as a string describing the role, company, and key achievements.
        "skills": A list of technical and soft skills mentioned.
        "education": A list of educational qualifications (degrees, institutions, dates).
        "projects": A list of significant projects mentioned. IMPORTANT: Be thorough in extracting all projects, including project names, technologies used, and key accomplishments.

        If a section is not present, provide an empty string or an empty list accordingly.
        Pay special attention to the projects section, as it is critical information for the CV.

        Job Description Context (for relevance):
        {input_data.get("job_description", "")}

        CV Text:
        {raw_cv_text}

        JSON Output:
        """

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
        except Exception as e:
            print(f"An unexpected error occurred during CV analysis: {e}")
            print("Using fallback extraction instead.")
            fallback_extraction["summary"] = (
                f"Error analyzing CV: {fallback_extraction.get('summary', '')}"
            )
            return fallback_extraction
