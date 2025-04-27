from agent_base import AgentBase
from llm import LLM
from state_manager import ContentData, AgentIO, CVData
from typing import Dict, Any

# Define a simple structure for the built CV data for now
# This can be made more sophisticated later
class BuiltCVData(Dict):
    """
    Represents the structured data for the complete CV.
    """
    pass # Inheriting from Dict for now

class CVBuilderAgent(AgentBase):
    """
    Agent responsible for building the complete CV structure from tailored sections.
    """

    def __init__(self, name: str, description: str, llm: LLM):
        """
        Initializes the CVBuilderAgent.

        Args:
            name: The name of the agent.
            description: A description of the agent.
            llm: The LLM instance (can be used later for more complex assembly).
        """
        super().__init__(
            name=name,
            description=description,
            input_schema=AgentIO(
                input={
                    "generated_content": Dict[str, Any], # Takes the generated content
                    "user_cv_data": Dict[str, Any] # Takes the original CV data
                    # Potentially other relevant data from state
                },
                output=BuiltCVData, # Outputs the structured CV data
                description="Generated content and original CV data for building.",
            ),
            output_schema=AgentIO(
                input={
                    "generated_content": Dict[str, Any],
                    "user_cv_data": Dict[str, Any]
                },
                output=BuiltCVData,
                description="Structured complete CV data.",
            ),
        )
        self.llm = llm # Storing LLM though not used for complex assembly yet

    def run(self, input_data: Dict[str, Any]) -> BuiltCVData:
        """
        Builds the complete CV structure.

        Args:
            input_data: A dictionary containing 'generated_content' and 'user_cv_data'.

        Returns:
            A BuiltCVData object representing the structured CV.
        """
        generated_content = input_data.get("generated_content", {})
        user_cv_data = input_data.get("user_cv_data", {})

        print("Executing: CVBuilderAgent")

        # --- Placeholder for actual CV building logic ---
        # For now, just combine generated content and original raw text
        # A real implementation would structure different sections (summary, experience, skills, etc.)
        built_cv_data = BuiltCVData({
            "summary": generated_content.get("summary", ""),
            "experience_bullets": generated_content.get("experience_bullets", []),
            "skills_section": generated_content.get("skills_section", ""),
            "raw_cv_text": user_cv_data.get("raw_text", ""), # Include original raw text for now
            "other_sections": {} # Placeholder for other sections parsed from original CV later
        })
        # -------------------------------------------------

        print("Completed: CVBuilderAgent")
        return built_cv_data
