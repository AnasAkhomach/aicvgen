from agent_base import AgentBase
from state_manager import ContentData, AgentIO
import google.generativeai as genai
import os
from llm import LLM

class TemplateRenderer(AgentBase):
    """Agent responsible for rendering CV templates."""

    def __init__(self, name: str, description: str, model: LLM, 
                 input_schema: AgentIO, output_schema: AgentIO):
        super().__init__(name, description, input_schema, output_schema)
        self.model = model

    def run(self, input_data: ContentData) -> str:
        """
        Renders a CV based on the given ContentData.
        Returns a string with the rendered CV in markdown format.
        """
        sections = ["# Tailored CV\n\n"]

        # Summary section
        if input_data.get("summary"):
            sections.append(f"## Summary\n{input_data.get('summary')}\n\n")

        # Experience section
        if input_data.get("experience_bullets"):
            sections.append("## Experience\n")
            sections.extend(f"- {bullet}\n" for bullet in input_data.get("experience_bullets", []))
            sections.append("\n")

        # Skills section (fixed quotes here)
        if input_data.get("skills_section"):
            sections.append(f"## Skills\n{input_data.get('skills_section')}\n\n")

        # Projects section
        if input_data.get("projects"):
            sections.append("## Projects\n")
            sections.extend(f"- {project}\n" for project in input_data.get("projects", []))
            sections.append("\n")

        # Other content sections
        if input_data.get("other_content"):
            for section_name, section_content in input_data.get("other_content", {}).items():
                sections.append(f"## {section_name}\n{section_content}\n\n")

        return "\n".join(sections).strip()