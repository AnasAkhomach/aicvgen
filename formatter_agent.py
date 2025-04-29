from agent_base import AgentBase
from state_manager import AgentIO, ContentData
from typing import Dict, Any

class FormatterAgent(AgentBase):
    """
    Agent responsible for formatting the tailored CV content.
    """

    def __init__(self, name: str, description: str):
        """
        Initializes the FormatterAgent.

        Args:
            name: The name of the agent.
            description: A description of the agent.
        """
        super().__init__(
            name=name,
            description=description,
            input_schema=AgentIO(
                input={
                    "content_data": ContentData, # Takes generated content
                    "format_specifications": Dict[str, Any] # Takes formatting rules/preferences
                },
                output=str, # Outputs formatted content as a string (e.g., Markdown, LaTeX)
                description="Tailored CV content and formatting specifications.",
            ),
            output_schema=AgentIO(
                input={
                    "content_data": ContentData,
                    "format_specifications": Dict[str, Any]
                },
                output=str,
                description="Formatted CV content string.",
            ),
        )

    def run(self, input: Dict[str, Any]) -> str:
        """
        Formats the generated CV content based on specifications.

        Args:
            input: A dictionary containing 'content_data' (ContentData) and 'format_specifications' (Dict).

        Returns:
            A string with the formatted content.
        """
        content_data: ContentData = input.get("content_data")
        format_specifications: Dict[str, Any] = input.get("format_specifications", {})

        print("Executing: FormatterAgent")
        print(f"Formatting with specifications: {format_specifications}")

        # --- Placeholder Formatting Logic ---
        # This is a basic simulation. Real implementation would apply styles, 
        # handle LaTeX syntax, layout, etc.
        formatted_content_parts = []

        if content_data.summary:
            formatted_content_parts.append(f"## Summary\n{content_data.summary}")

        if content_data.experience_bullets:
            formatted_content_parts.append("## Experience")
            for bullet in content_data.experience_bullets:
                formatted_content_parts.append(f"- {bullet}")

        if content_data.skills_section:
             formatted_content_parts.append(f"## Skills\n{content_data.skills_section}")
        
        if content_data.projects:
            formatted_content_parts.append("## Projects")
            for project in content_data.projects:
                formatted_content_parts.append(f"- {project}")

        if content_data.other_content:
            for section_name, section_content in content_data.other_content.items():
                # Add period at the end if not already present
                if section_content and not section_content.endswith('.'):
                    section_content = f"{section_content}."
                formatted_content_parts.append(f"## {section_name}\n{section_content}")

        # If we have any content, join with double newlines for proper spacing
        if formatted_content_parts:
            formatted_output = "\n\n".join(formatted_content_parts)
        else:
            formatted_output = ""
            
        # -----------------------------------

        print("Completed: FormatterAgent (Simulated Formatting)")
        return formatted_output
