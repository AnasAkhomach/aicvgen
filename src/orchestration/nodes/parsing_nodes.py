"""Parsing and research nodes for the CV workflow."""

import logging
from typing import Dict, Any

from src.orchestration.state import GlobalState
from src.agents.job_description_parser_agent import JobDescriptionParserAgent
from src.agents.research_agent import ResearchAgent
from src.agents.cv_analyzer_agent import CVAnalyzerAgent
from src.agents.user_cv_parser_agent import UserCVParserAgent

logger = logging.getLogger(__name__)


async def user_cv_parser_node(
    state: GlobalState, *, agent: "UserCVParserAgent"
) -> GlobalState:
    """Parse the raw CV text from the state and update the state with the structured CV object.

    Args:
        state: Current workflow state containing cv_text
        agent: User CV parser agent

    Returns:
        Updated state with structured_cv object
    """
    logger.info("Starting User CV Parser node")

    try:
        # Extract CV text from state
        cv_text = state.get("cv_text", "") or ""

        # Parse the CV using the agent (even if empty)
        structured_cv = await agent.run(cv_text)

        # Update state with the structured CV and return the modified state
        updated_state = {
            **state,
            "structured_cv": structured_cv,
            "last_executed_node": "USER_CV_PARSER",
        }

        logger.info("User CV Parser node completed successfully")
        return updated_state

    except Exception as exc:
        logger.error(f"Error in User CV Parser node: {exc}")
        error_messages = (
            list(state["error_messages"]) if state["error_messages"] else []
        )
        error_messages.append(f"CV Parser failed: {str(exc)}")

        return {
            **state,
            "error_messages": error_messages,
            "last_executed_node": "USER_CV_PARSER",
        }


async def jd_parser_node(
    state: GlobalState, *, agent: "JobDescriptionParserAgent"
) -> GlobalState:
    """Parse job description and extract key requirements.

    Args:
        state: Current workflow state
        agent: Job description parser agent

    Returns:
        Updated state with parsed job data
    """
    logger.info("Starting JD Parser node")

    try:
        # Agent is now explicitly provided via dependency injection

        # Parse the job description using the agent
        # Extract job description text from job_description_data if available
        job_description_text = ""
        if state["job_description_data"]:
            # Handle both Pydantic object and string representation cases
            job_data = state["job_description_data"]
            if hasattr(job_data, "raw_text"):
                # Proper Pydantic object
                job_description_text = job_data.raw_text or ""
            elif isinstance(job_data, str):
                # String representation - extract raw_text value
                import re

                match = re.search(r"raw_text='([^']*)'|raw_text=([^\s]+)", job_data)
                if match:
                    job_description_text = match.group(1) or match.group(2) or ""
                else:
                    job_description_text = job_data  # Fallback to entire string
            else:
                logger.warning(
                    f"Unexpected job_description_data type: {type(job_data)}"
                )
                job_description_text = str(job_data)

        result = await agent.run(input_data={"raw_text": job_description_text})

        # Check for errors in the result
        if "error_messages" in result:
            raise Exception(f"Agent execution failed: {result['error_messages']}")

        parsed_jd = result.get("job_description_data")

        # Update state with parsed job data
        updated_state = {
            **state,
            "parsed_jd": parsed_jd,
            "last_executed_node": "JD_PARSER",
        }

        logger.info("JD Parser node completed successfully")
        return updated_state

    except Exception as exc:
        logger.error(f"Error in JD Parser node: {exc}")
        error_messages = (
            list(state["error_messages"]) if state["error_messages"] else []
        )
        error_messages.append(f"JD Parser failed: {str(exc)}")

        return {
            **state,
            "error_messages": error_messages,
            "last_executed_node": "JD_PARSER",
        }


async def research_node(state: GlobalState, *, agent: "ResearchAgent") -> GlobalState:
    """Conduct research based on parsed job description.

    Args:
        state: Current workflow state
        agent: Research agent

    Returns:
        Updated state with research data
    """
    logger.info("Starting Research node")

    try:
        # Agent is now explicitly provided via dependency injection

        # Conduct research based on parsed JD
        result = await agent.run(job_description_data=state["parsed_jd"])

        # Check for errors in the result
        if "error_messages" in result:
            raise Exception(f"Agent execution failed: {result['error_messages']}")

        research_data = result.get("research_findings")

        # Update state with research data
        updated_state = {
            **state,
            "research_data": research_data,
            "last_executed_node": "RESEARCH",
        }

        logger.info("Research node completed successfully")
        return updated_state

    except Exception as exc:
        logger.error(f"Error in Research node: {exc}")
        error_messages = (
            list(state["error_messages"]) if state["error_messages"] else []
        )
        error_messages.append(f"Research failed: {str(exc)}")

        return {
            **state,
            "error_messages": error_messages,
            "last_executed_node": "RESEARCH",
        }


async def cv_analyzer_node(
    state: GlobalState, *, agent: "CVAnalyzerAgent"
) -> GlobalState:
    """Analyze CV against job requirements.

    Args:
        state: Current workflow state
        agent: CV analyzer agent

    Returns:
        Updated state with CV analysis
    """
    logger.info("Starting CV Analyzer node")

    try:
        # Agent is now explicitly provided via dependency injection

        # Analyze CV against job requirements
        result = await agent.run(
            cv_data=state["structured_cv"], job_description=state["parsed_jd"]
        )

        # Check for errors in the result
        if "error_messages" in result:
            raise Exception(f"Agent execution failed: {result['error_messages']}")

        analysis_result = result.get("cv_analysis_results")

        # Update state with analysis
        updated_state = {
            **state,
            "cv_analysis": analysis_result,
            "last_executed_node": "CV_ANALYZER",
        }

        logger.info("CV Analyzer node completed successfully")
        return updated_state

    except Exception as exc:
        logger.error(f"Error in CV Analyzer node: {exc}")
        error_messages = (
            list(state["error_messages"]) if state["error_messages"] else []
        )
        error_messages.append(f"CV Analysis failed: {str(exc)}")

        return {
            **state,
            "error_messages": error_messages,
            "last_executed_node": "CV_ANALYZER",
        }
