from agent_base import AgentBase
from state_manager import AgentIO, CVData, JobDescriptionData
from typing import Dict, Any, List
from llm import LLM # Import LLM
import json # Import json

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
                    "job_description": JobDescriptionData  # May need job description for context
                },
                output=Dict[str, Any],  # Define a more specific output schema later
                description="User CV data and optional job description for analysis.",
            ),
            output_schema=AgentIO(
                input={
                    "user_cv": CVData,
                    "job_description": JobDescriptionData
                },
                output=Dict[str, Any], # Define a more specific output schema later
                description="Extracted information from the CV.",
            ),
        )
        self.llm = llm # Store the LLM instance

    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the user's CV using an LLM and extracts key information.

        Args:
            input: A dictionary containing 'user_cv' (CVData) and optional 'job_description' (JobDescriptionData).

        Returns:
            A dictionary containing the extracted information from the CV.
            Expected keys: 'summary', 'experiences', 'skills', 'education', 'projects'.
        """
        user_cv_data: CVData = input.get("user_cv")
        job_description_data: JobDescriptionData = input.get("job_description") # Use job_description_data for context

        raw_cv_text = user_cv_data.get("raw_text", "")

        if not raw_cv_text:
            print("Warning: Empty CV text provided to CVAnalyzerAgent.")
            return {
                "summary": "",
                "experiences": [],
                "skills": [],
                "education": [],
                "projects": []
            }

        print(f"Analyzing CV...\n Raw Text (first 200 chars): {raw_cv_text[:200]}...")

        # Craft the prompt for the LLM
        # Include job description for context to help prioritize relevant info
        prompt = f"""
        Analyze the following CV text and extract the key sections and information.
        Provide the output in JSON format with the following keys:
        "summary": The professional summary or objective statement.
        "experiences": A list of work experiences, each as a string describing the role, company, and key achievements.
        "skills": A list of technical and soft skills mentioned.
        "education": A list of educational qualifications (degrees, institutions, dates).
        "projects": A list of significant projects mentioned.

        If a section is not present, provide an empty string or an empty list accordingly.

        Job Description Context (for relevance):
        {job_description_data}

        CV Text:
        {raw_cv_text}

        JSON Output:
        """

        print("Sending prompt to LLM for CV analysis...")
        try:
            llm_response = self.llm.generate_content(prompt)
            print("Received response from LLM.")
            # print(f"LLM Response: {llm_response}") # Uncomment for debugging

            # Attempt to parse the JSON response (handle markdown formatting)
            json_string = llm_response.strip()
            if json_string.startswith("```json"):
                json_string = json_string[len("```json"):].strip()
                if json_string.endswith("```"):
                    json_string = json_string[:-len("```")].strip()

            # Load the JSON string into a Python dictionary
            extracted_data = json.loads(json_string)

            # Ensure all expected keys are present, even if empty in the LLM output
            extracted_data.setdefault("summary", "")
            extracted_data.setdefault("experiences", [])
            extracted_data.setdefault("skills", [])
            extracted_data.setdefault("education", [])
            extracted_data.setdefault("projects", [])

            print("Successfully analyzed CV using LLM.")
            # print(f"Extracted Data: {extracted_data}") # Uncomment for debugging
            return extracted_data

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM response during CV analysis: {e}")
            # print(f"Faulty JSON string: {json_string}") # Uncomment for debugging
            # Return empty structure on error
            return {
                "summary": f"Error parsing CV: {e}", # Indicate error in summary
                "experiences": [],
                "skills": [],
                "education": [],
                "projects": []
            }
        except Exception as e:
            print(f"An unexpected error occurred during CV analysis: {e}")
            # Return empty structure on error
            return {
                "summary": f"Error analyzing CV: {e}", # Indicate error in summary
                "experiences": [],
                "skills": [],
                "education": [],
                "projects": []
            }
