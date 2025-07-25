"""Parsing and research nodes for the CV workflow."""

import logging
from typing import Dict, Any

from src.orchestration.state import GlobalState
from src.agents.job_description_parser_agent import JobDescriptionParserAgent
from src.agents.research_agent import ResearchAgent
from src.agents.cv_analyzer_agent import CVAnalyzerAgent

logger = logging.getLogger(__name__)


async def jd_parser_node(state: GlobalState, *, agent: 'JobDescriptionParserAgent') -> GlobalState:
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
        
        # Parse the job description
        # Extract job description text from job_description_data if available
        job_description_text = ""
        if state["job_description_data"]:
            job_description_text = state["job_description_data"].raw_text or ""
        
        parsed_jd = await agent.parse_job_description(
            job_description=job_description_text
        )
        
        # Update state with parsed job data
        updated_state = {
            **state,
            "parsed_jd": parsed_jd,
            "last_executed_node": "JD_PARSER"
        }
        
        logger.info("JD Parser node completed successfully")
        return updated_state
        
    except Exception as exc:
        logger.error(f"Error in JD Parser node: {exc}")
        error_messages = list(state["error_messages"]) if state["error_messages"] else []
        error_messages.append(f"JD Parser failed: {str(exc)}")
        
        return {
            **state,
            "error_messages": error_messages,
            "last_executed_node": "JD_PARSER"
        }


async def research_node(state: GlobalState, *, agent: 'ResearchAgent') -> GlobalState:
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
        research_data = await agent.conduct_research(
            parsed_jd=state["parsed_jd"]
        )
        
        # Update state with research data
        updated_state = {
            **state,
            "research_data": research_data,
            "last_executed_node": "RESEARCH"
        }
        
        logger.info("Research node completed successfully")
        return updated_state
        
    except Exception as exc:
        logger.error(f"Error in Research node: {exc}")
        error_messages = list(state["error_messages"]) if state["error_messages"] else []
        error_messages.append(f"Research failed: {str(exc)}")
        
        return {
            **state,
            "error_messages": error_messages,
            "last_executed_node": "RESEARCH"
        }


async def cv_analyzer_node(state: GlobalState, *, agent: 'CVAnalyzerAgent') -> GlobalState:
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
        analysis_result = await agent.analyze_cv(
            structured_cv=state["structured_cv"],
            parsed_jd=state["parsed_jd"],
            research_data=state["research_data"]
        )
        
        # Update state with analysis
        updated_state = {
            **state,
            "cv_analysis": analysis_result,
            "last_executed_node": "CV_ANALYZER"
        }
        
        logger.info("CV Analyzer node completed successfully")
        return updated_state
        
    except Exception as exc:
        logger.error(f"Error in CV Analyzer node: {exc}")
        error_messages = list(state["error_messages"]) if state["error_messages"] else []
        error_messages.append(f"CV Analysis failed: {str(exc)}")
        
        return {
            **state,
            "error_messages": error_messages,
            "last_executed_node": "CV_ANALYZER"
        }