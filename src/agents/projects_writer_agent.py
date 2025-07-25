"""Projects Writer Agent for generating tailored project content.

This agent specializes in creating compelling project descriptions that align
with job requirements and highlight relevant technical achievements.
"""

from typing import Any, Dict, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.models.agent_output_models import ProjectLLMOutput
from src.config.logging_config import get_logger
from src.agents.agent_base import AgentBase

logger = get_logger(__name__)


class ProjectsWriterAgentInput(BaseModel):
    """Input model for ProjectsWriterAgent following Gold Standard LCEL pattern."""
    
    job_description: str = Field(description="The job description text")
    project_item: dict = Field(description="The specific project item being processed")
    key_qualifications: str = Field(description="Extracted key qualifications from CV")
    professional_experience: str = Field(description="Extracted professional experience from CV")
    research_findings: Optional[dict] = Field(default=None, description="Research findings if available")
    template_content: str = Field(description="Formatted template content for the prompt")
    format_instructions: str = Field(description="Output format instructions from parser")


class ProjectsWriterAgent(AgentBase):
    """Agent for generating tailored project content using Gold Standard LCEL pattern."""
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        prompt: ChatPromptTemplate,
        parser: BaseOutputParser,
        settings: dict,
        session_id: str
    ):
        """Initialize the ProjectsWriterAgent with LCEL components.
        
        Args:
            llm: The language model instance
            prompt: The chat prompt template
            parser: The output parser
            settings: Settings dictionary
            session_id: Session identifier
        """
        # Call parent constructor with required parameters
        super().__init__(
            name="ProjectsWriterAgent",
            description="Agent for generating tailored project content using Gold Standard LCEL pattern",
            session_id=session_id,
            settings=settings
        )
        
        self.llm = llm
        self.prompt = prompt
        self.parser = parser
        self.settings = settings
        self.session_id = session_id
        
        # Create the LCEL chain
        self.chain = self.prompt | self.llm | self.parser
        
        logger.info(f"ProjectsWriterAgent initialized for session: {session_id}")

    async def _execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the agent using Gold Standard LCEL pattern.
        
        Args:
            **kwargs: Keyword arguments containing input data
            
        Returns:
            Dictionary containing generated project content
        """
        try:
            # 1. Validate the input dictionary against the new Pydantic model
            validated_input = ProjectsWriterAgentInput(**kwargs)
            
            # 2. Invoke the chain with the validated data
            generated_data: ProjectLLMOutput = await self.chain.ainvoke(validated_input.model_dump())
            
            # 3. Return the generated data. The agent's job is done.
            return {"generated_projects": generated_data}
            
        except Exception as exc:
            logger.error(f"Error in ProjectsWriterAgent._execute: {exc}")
            raise
