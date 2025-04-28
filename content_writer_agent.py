from agent_base import AgentBase
from llm import LLM
from state_manager import JobDescriptionData, ContentData, AgentIO, ExperienceEntry, CVData
from typing import List, Dict, Any
import json
from tools_agent import ToolsAgent # Import ToolsAgent

class ContentWriterAgent(AgentBase):
    """
    Agent responsible for generating tailored CV content based on job requirements and user experiences.
    """
    # Accept ToolsAgent in the constructor
    def __init__(self, name: str, description: str, llm: LLM, tools_agent: ToolsAgent):
        """
        Initializes the ContentWriterAgent.

        Args:
            name: The name of the agent.
            description: A description of the agent.
            llm: The LLM instance to use for content generation.
            tools_agent: The ToolsAgent instance for content processing.
        """
        super().__init__(
            name=name,
            description=description,
            input_schema=AgentIO(
                input={
                    "job_description_data": Dict[str, Any], # Using Dict as JobDescriptionData is treated as Dict in state
                    "relevant_experiences": List[str],
                    "research_results": Dict[str, Any],
                    "user_cv_data": Dict[str, Any] # Added user_cv_data to input schema
                },
                output=ContentData, # The output is a ContentData object
                description="Parsed job description data, relevant user experiences, research results, and user CV data.",
            ),
            output_schema=AgentIO(
                input={
                     "job_description_data": Dict[str, Any],
                     "relevant_experiences": List[str],
                     "research_results": Dict[str, Any],
                     "user_cv_data": Dict[str, Any]
                },
                output=ContentData,
                description="Tailored CV content.",
            ),
        )
        self.llm = llm
        self.tools_agent = tools_agent # Store ToolsAgent instance

    def run(self, input_data: Dict[str, Any]) -> ContentData:
        """
        Generates tailored CV content using the LLM and processes it with the ToolsAgent.

        Args:
            input_data: A dictionary containing 'job_description_data' (Dict), 
                      'relevant_experiences' (List[str]), 'research_results' (Dict), 
                      and 'user_cv_data' (Dict).

        Returns:
            A ContentData object with the generated and processed content.
        """
        job_description_data = input_data.get("job_description_data", {})
        relevant_experiences = input_data.get("relevant_experiences", [])
        research_results = input_data.get("research_results", {})
        user_cv_data = input_data.get("user_cv_data", {}) # Get user_cv_data

        # Prepare context for the LLM
        job_requirements = (
            f"Skills: {', '.join(job_description_data.get('skills', []))}\n"
            f"Responsibilities: {'. '.join(job_description_data.get('responsibilities', []))}"
        )
        formatted_experiences = "\n".join([f"- {exp}" for exp in relevant_experiences])

        research_context = ""
        if research_results:
            research_context = "\n Additional Research Context:"
            for key, value in research_results.items():
                research_context += f"{key.replace('_', ' ').title()}: {value}\n "

        # Include extracted CV sections for more comprehensive content generation
        cv_sections_context = "\nUser CV Sections:\n"
        if user_cv_data.get("summary"):
            cv_sections_context += f"Summary: {user_cv_data['summary']}\n"
        if user_cv_data.get("skills"):
             cv_sections_context += f"Skills: {', '.join(user_cv_data['skills'])}\n"
        if user_cv_data.get("education"):
             cv_sections_context += f"Education: {'; '.join(user_cv_data['education'])}\n"
        if user_cv_data.get("projects"):
             cv_sections_context += f"Projects: {'; '.join(user_cv_data['projects'])}\n"


        prompt = f"""
        You are a CV tailoring assistant. Your goal is to generate compelling CV content 
        that highlights the user's relevant experience based on a job description, provided research context, and their existing CV information.

        Here is the job description information:
        {job_requirements}

        Here are some of the user's experiences that are potentially relevant (draw from these, but also consider the full CV context if needed):
        {formatted_experiences}

        {research_context}

        {cv_sections_context}

        Generate the following tailored content in JSON format:
        {{
          "summary": "A 2-3 sentence professional summary tailored to the job description and research context, highlighting key skills and experience from the user's CV.",
          "experience_bullets": [
            "Tailored bullet point 1 based on relevant experience, job requirements, research context, and full CV details.",
            "Tailored bullet point 2..."
          ],
          "skills_section": "A comprehensive list of skills relevant to the job description, combining skills from the job requirements, research, and the user's CV.",
           "projects": ["Tailored project description 1..."], # Include projects based on user CV and job description
           "other_content": {{}} # Include other relevant sections if needed
        }}

        Ensure the content is concise, action-oriented, and uses keywords from the job description. Leverage the provided relevant experiences and the full user CV sections to create the most impactful content.
        """\

        print("Sending tailored prompt to LLM for content generation from ContentWriterAgent...")
        try:
            llm_response = self.llm.generate_content(prompt)
            print("Received response from LLM for content generation in ContentWriterAgent.")
            # print(f"LLM Response (Content Gen Agent): {llm_response}") # Uncomment for debugging

            # Attempt to parse the JSON response
            json_string = llm_response.strip()
            if json_string.startswith("```json"):
                json_string = json_string[len("```json"):].strip()
                if json_string.endswith("```"):
                    json_string = json_string[:-len("```")].strip()

            parsed_content = json.loads(json_string)

            # Create a ContentData object from the parsed data
            generated_content = ContentData(
                summary=parsed_content.get("summary", ""),
                experience_bullets=parsed_content.get("experience_bullets", []),
                skills_section=parsed_content.get("skills_section", ""),
                projects=parsed_content.get("projects", []), 
                other_content=parsed_content.get("other_content", {})
            )

            print("Successfully generated content using ContentWriterAgent.")
            
            # --- Demonstrate using the ToolsAgent (Simulated) ---
            print("Using ToolsAgent for formatting and validation...")
            # Simulate formatting the summary
            formatted_summary = self.tools_agent.format_text(generated_content.summary, format_type="markdown")
            print(f"Simulated Formatted Summary: {formatted_summary[:100]}...")

            # Simulate validating the generated content against job requirements
            # For validation, let's use skills and responsibilities from the job description as requirements
            validation_requirements = job_description_data.get("skills", []) + job_description_data.get("responsibilities", [])
            all_generated_text = (
                generated_content.summary + "\n\n" 
                + "\n".join(generated_content.experience_bullets) 
                + "\n\n" 
                + generated_content.skills_section
            )
            validation_results = self.tools_agent.validate_content(all_generated_text, validation_requirements)
            print(f"Simulated Validation Results: {validation_results}")
            # You would typically use these validation results to refine the generated content
            # For this simulation, we just print the results.
            # ---------------------------------------------------


            return generated_content

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM response in ContentWriterAgent: {e}")
            print(f"Faulty JSON string (Content Gen Agent): {json_string}")
            return ContentData() # Return empty ContentData on error
        except Exception as e:
            print(f"An unexpected error occurred in ContentWriterAgent: {e}")
            return ContentData() # Return empty ContentData on error

    def generate_batch(self, input_data: Dict[str, Any], batch_type: str) -> ContentData:
        """
        Generates a batch of content based on the specified type (e.g., 'summary', 'experience_bullet').

        Args:
            input_data: A dictionary containing 'job_description_data', 'relevant_experiences',
                        'research_results', and 'user_cv_data'.
            batch_type: The type of content to generate (e.g., 'summary', 'experience_bullet').

        Returns:
            A ContentData object with the generated batch content.
        """
        job_description_data = input_data.get("job_description_data", {})
        relevant_experiences = input_data.get("relevant_experiences", [])
        research_results = input_data.get("research_results", {})
        user_cv_data = input_data.get("user_cv_data", {})

        # Prepare context for the LLM
        job_requirements = (
            f"Skills: {', '.join(job_description_data.get('skills', []))}\n"
            f"Responsibilities: {'. '.join(job_description_data.get('responsibilities', []))}"
        )
        formatted_experiences = "\n".join([f"- {exp}" for exp in relevant_experiences])

        research_context = ""
        if research_results:
            research_context = "\n Additional Research Context:"
            for key, value in research_results.items():
                research_context += f"{key.replace('_', ' ').title()}: {value}\n "

        cv_sections_context = "\nUser CV Sections:\n"
        if user_cv_data.get("summary"):
            cv_sections_context += f"Summary: {user_cv_data['summary']}\n"
        if user_cv_data.get("skills"):
            cv_sections_context += f"Skills: {', '.join(user_cv_data['skills'])}\n"
        if user_cv_data.get("education"):
            cv_sections_context += f"Education: {'; '.join(user_cv_data['education'])}\n"
        if user_cv_data.get("projects"):
            cv_sections_context += f"Projects: {'; '.join(user_cv_data['projects'])}\n"

        # Adjust the prompt based on the batch type
        if batch_type == "summary":
            prompt = f"""
            Generate a professional summary tailored to the job description and research context.

            Job Description:
            {job_requirements}

            Relevant Experiences:
            {formatted_experiences}

            {research_context}

            {cv_sections_context}

            Output the summary as a single string.
            """
        elif batch_type == "experience_bullet":
            prompt = f"""
            Generate a single bullet point for the professional experience section based on the job description
            and relevant experiences.

            Job Description:
            {job_requirements}

            Relevant Experiences:
            {formatted_experiences}

            {research_context}

            {cv_sections_context}

            Output the bullet point as a single string.
            """
        else:
            raise ValueError(f"Unsupported batch type: {batch_type}")

        print(f"Sending prompt to LLM for {batch_type} generation...")
        try:
            llm_response = self.llm.generate_content(prompt)
            print(f"Received response from LLM for {batch_type} generation.")

            # Parse the response
            generated_content = ContentData()
            if batch_type == "summary":
                generated_content["summary"] = llm_response.strip()
            elif batch_type == "experience_bullet":
                generated_content["experience_bullets"] = [llm_response.strip()]

            return generated_content

        except Exception as e:
            print(f"Error generating {batch_type}: {e}")
            return ContentData()
