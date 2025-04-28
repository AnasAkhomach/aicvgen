from agent_base import AgentBase
from llm import LLM
from state_manager import (
    AgentIO,
    JobDescriptionData,  # Import JobDescriptionData from state_manager
)
import json  # Import json for parsing LLM output
from typing import List, Optional

class ParserAgent(AgentBase):
    """Agent responsible for parsing job descriptions and extracting key information using an LLM."""

    def __init__(self, name: str, description: str, llm: LLM):
        """Initializes the ParserAgent with name, description, and llm."""
        super().__init__(
            name=name,
            description=description,
            input_schema=AgentIO(
                input={
                    "job_description": str
                },
                output=JobDescriptionData,  # The output is a JobDescriptionData object
                description="Raw job description as a string.",
            ),
            output_schema=AgentIO(
                input={
                    "job_description": str
                },
                output=JobDescriptionData,
                description="Parsed job description data.",
            ),
        )
        self.llm = llm

    def run(self, input: dict) -> JobDescriptionData:
        """
        Parses a raw job description using an LLM and extracts key information.

        Args:
            input: A dictionary containing the raw job description as a string.

        Returns:
            A JobDescriptionData object with the parsed content.
        """
        raw_job_description = input.get("job_description", "")
        if not raw_job_description:
            print("Warning: Empty job description provided to ParserAgent.")
            # Return a default JobDescriptionData object for empty input
            return JobDescriptionData(
                raw_text=raw_job_description,
                skills=[],
                experience_level="N/A",
                responsibilities=[],
                industry_terms=[],
                company_values=[],
            )

        # Update prompt to explicitly request JSON output
        prompt = f"""
        Please extract the following key information from the job description below and provide it in JSON format.
        The JSON object should have the following keys:
        "skills": List of key technical and soft skills mentioned.
        "experience_level": The required experience level (e.g., "Entry-Level", "Mid-Level", "Senior-Level", "Manager"). If not specified, infer or use "N/A".
        "responsibilities": List of the main responsibilities listed in the job description.
        "industry_terms": List of any industry-specific jargon or terms.
        "company_values": List of any company values or cultural aspects mentioned.

        If a category is not mentioned, provide an empty list [] or "N/A" for strings.

        Job Description:
        {raw_job_description}

        IMPORTANT: Respond ONLY with the valid JSON object, starting with {{ and ending with }}.
        """

        print("Sending prompt to LLM for parsing...")
        try:
            # Get the LLM's response
            llm_response = self.llm.generate_content(prompt)
            print("Received response from LLM.")
            print(f"LLM Response: {llm_response}")  # Print raw LLM response for debugging

            # Extract JSON object from LLM response
            json_string = llm_response.strip()
            if "{" in json_string and "}" in json_string:
                json_string = json_string[json_string.find("{"):json_string.rfind("}") + 1]

            try:
                parsed_data = json.loads(json_string)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from LLM response: {e}")
                parsed_data = {}

            # Create a JobDescriptionData object from the parsed data
            job_data = JobDescriptionData(
                raw_text=raw_job_description,  # Pass raw_text here
                skills=parsed_data.get("skills", []),
                responsibilities=parsed_data.get("responsibilities", []),
                experience_level=parsed_data.get("experience_level", "N/A"),
                industry_terms=parsed_data.get("industry_terms", []),
                company_values=parsed_data.get("company_values", []),
                error=None if parsed_data else f"Error decoding JSON: {e}"  # Set error field if parsing failed
            )

            print("Successfully parsed job description using LLM.")
            return job_data

        except Exception as e:
            print(f"An unexpected error occurred during parsing: {e}")
            # Return a default JobDescriptionData object with error info
            return JobDescriptionData(
                raw_text=raw_job_description,
                skills=[],
                experience_level="N/A",
                responsibilities=[],
                industry_terms=[],
                company_values=[],
                error=str(e)  # Populate the error field
            )
