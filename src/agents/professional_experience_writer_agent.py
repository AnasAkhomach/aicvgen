"""This module defines the ProfessionalExperienceWriterAgent, responsible for generating the Professional Experience section of the CV."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.language_models import BaseLanguageModel
from src.agents.agent_base import AgentBase
from src.models.agent_output_models import ProfessionalExperienceLLMOutput
from src.config.logging_config import get_logger
from src.error_handling.exceptions import AgentExecutionError

logger = get_logger(__name__)


class ProfessionalExperienceAgentInput(BaseModel):
    """Input model for ProfessionalExperienceWriterAgent."""

    job_title: str = Field(description="Job title from the job description")
    company_name: str = Field(description="Company name from the job description")
    job_description: str = Field(description="Full job description text")
    experience_item: Dict[str, Any] = Field(
        description="The specific experience item being processed"
    )
    cv_summary: str = Field(description="Summary from the CV")
    required_skills: list = Field(
        default=[], description="Required skills from job description"
    )
    preferred_qualifications: list = Field(
        default=[], description="Preferred qualifications from job description"
    )
    research_findings: Optional[Dict[str, Any]] = Field(
        default=None, description="Research findings if available"
    )


class ProfessionalExperienceWriterAgent(AgentBase):
    """Agent responsible for generating professional experience content for a CV.

    This agent takes job description data, experience items, key qualifications,
    and research findings to generate tailored professional experience content.
    Uses LCEL (LangChain Expression Language) for declarative chain composition.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        prompt: ChatPromptTemplate,
        parser: BaseOutputParser,
        settings: dict = None,
        session_id: str = "default",
    ):
        """Initialize the ProfessionalExperienceWriterAgent.

        Args:
            llm: The language model to use
            prompt: The chat prompt template
            parser: The output parser
            settings: Optional agent settings
            session_id: Session identifier
        """
        super().__init__(
            name="ProfessionalExperienceWriterAgent",
            description="Agent responsible for generating professional experience content for a CV",
            session_id=session_id,
            settings=settings,
        )
        # Create the LCEL chain
        self.chain = prompt | llm | parser
        logger.info(f"Initialized {self.name} with LCEL chain")

    async def _execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the professional experience writer agent using Gold Standard LCEL pattern.

        Args:
            **kwargs: Input data containing job_title, company_name, job_description, etc.

        Returns:
            Dict containing the generated professional experience data
        """
        try:
            # 1. Validate the input dictionary against the new Pydantic model
            validated_input = ProfessionalExperienceAgentInput(**kwargs)

            # 2. Invoke the chain with the validated data
            # The agent's only job is to run the chain and return the result
            # It does NOT modify the structured_cv itself
            generated_data: ProfessionalExperienceLLMOutput = await self.chain.ainvoke(
                validated_input.model_dump()
            )

            # 3. Return the generated data in the format expected by the graph
            return {"generated_professional_experience": generated_data}

        except Exception as e:
            logger.error(f"Error in {self.name} execution: {e}")
            raise AgentExecutionError(
                self.name, f"Failed to generate professional experience content: {e}"
            ) from e
