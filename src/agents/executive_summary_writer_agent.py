"""
This module defines the ExecutiveSummaryWriterAgent, responsible for generating the Executive Summary section of the CV.
"""

from typing import Any, Dict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.agents.agent_base import AgentBase
from src.config.logging_config import get_structured_logger
from src.error_handling.exceptions import AgentExecutionError
from src.models.agent_input_models import ExecutiveSummaryWriterAgentInput
from src.models.agent_output_models import ExecutiveSummaryLLMOutput

logger = get_structured_logger(__name__)


class ExecutiveSummaryWriterAgent(AgentBase):
    """Agent for generating tailored Executive Summary content using LCEL pattern."""

    def __init__(
        self,
        *,
        llm: BaseLanguageModel,
        prompt: ChatPromptTemplate,
        parser: BaseOutputParser,
        session_id: str,
        settings: dict = None,
    ):
        """Initialize the Executive Summary writer agent with LCEL components."""
        super().__init__(
            name="ExecutiveSummaryWriter",
            description="Generates tailored Executive Summary for the CV using LCEL.",
            session_id=session_id,
            settings=settings or {},
        )
        # Pure LCEL chain: prompt | llm | parser
        self.chain = prompt | llm | parser

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the core content generation logic for Executive Summary using LCEL."""
        try:
            # 1. Validate input.
            validated_input = ExecutiveSummaryWriterAgentInput(**kwargs)

            # 2. Invoke the chain.
            generated_data: ExecutiveSummaryLLMOutput = await self.chain.ainvoke(
                validated_input.model_dump()
            )

            # 3. Return ONLY the generated data.
            # The agent's job is now finished.
            return {"generated_executive_summary": generated_data.executive_summary}

        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}", exc_info=True)
            # Re-raise a specific error for the graph to handle
            raise AgentExecutionError(self.name, str(e)) from e
