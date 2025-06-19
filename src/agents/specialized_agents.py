#!/usr/bin/env python3

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from .enhanced_content_writer import EnhancedContentWriterAgent, create_content_writer
from .agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from .parser_agent import ParserAgent
from ..models.data_models import ContentType, ProcessingStatus
from ..config.logging_config import get_structured_logger
from ..config.settings import get_config
from ..services.llm_service import get_llm_service
from ..exceptions.agent_exceptions import AgentErrorHandler

logger = get_structured_logger("specialized_agents")


class CVAnalysisAgent(EnhancedAgentBase):
    """Agent specialized in analyzing CV content and job requirements."""

    def __init__(self):
        super().__init__(
            name="CVAnalysisAgent",
            description="Analyzes CV content and job requirements to provide optimization recommendations",
            input_schema={"cv_data": Dict[str, Any], "job_description": Dict[str, Any]},
            output_schema={
                "analysis_results": Dict[str, Any],
                "recommendations": List[str],
                "match_score": float,
            },
        )
        self.llm_service = get_llm_service()

        # Initialize settings for prompt loading
        self.settings = get_config()

    async def run_async(
        self, input_data: Any, context: AgentExecutionContext
    ) -> AgentResult:
        """Analyze CV content against job requirements."""
        from ..models.validation_schemas import validate_agent_input

        try:
            # Validate input data
            if not validate_agent_input(input_data, dict):
                self.logger.error("Input validation failed for CVAnalysisAgent")
                return AgentResult(
                    success=False,
                    output_data={
                        "error": "Input validation failed: expected dictionary input"
                    },
                    confidence_score=0.0,
                    error_message="Input validation failed: expected dictionary input",
                    metadata={
                        "analysis_type": "cv_job_match",
                        "validation_error": True,
                    },
                )

            # Debug logging to understand the input_data type and content
            self.logger.info(
                f"CVAnalysisAgent received input_data type: {type(input_data)}"
            )
            self.logger.info(
                f"CVAnalysisAgent received input_data: {str(input_data)[:500]}"
            )

            # Ensure input_data is a dictionary
            if not isinstance(input_data, dict):
                self.logger.warning(
                    f"Converting non-dict input_data from {type(input_data)} to empty dict"
                )
                input_data = {}

            cv_data = input_data.get("cv_data", {})
            job_description = input_data.get("job_description", {})

            self.logger.info(
                f"Extracted cv_data type: {type(cv_data)}, job_description type: {type(job_description)}"
            )

            # Perform analysis
            analysis = await self._analyze_cv_job_match(
                cv_data, job_description, context
            )

            # Generate recommendations
            recommendations = await self._generate_recommendations(analysis, context)

            # Calculate match score
            match_score = self._calculate_match_score(analysis)

            result_data = {
                "analysis_results": analysis,
                "recommendations": recommendations,
                "match_score": match_score,
                "analysis_timestamp": datetime.now().isoformat(),
            }

            return AgentResult(
                success=True,
                output_data=result_data,
                confidence_score=0.85,
                metadata={
                    "analysis_type": "cv_job_match",
                    "items_analyzed": len(cv_data),
                },
            )

        except Exception as e:
            # Use standardized error handling
            fallback_data = AgentErrorHandler.create_fallback_data("cv_analysis")
            return AgentErrorHandler.handle_general_error(
                e, "cv_analysis", fallback_data, "run_async"
            )

    async def _analyze_cv_job_match(
        self,
        cv_data: Dict[str, Any],
        job_description: Dict[str, Any],
        context: AgentExecutionContext,
    ) -> Dict[str, Any]:
        """Analyze match between CV and job requirements."""
        # Implementation for CV-job matching analysis
        analysis = {
            "skill_matches": [],
            "experience_relevance": 0.0,
            "gaps_identified": [],
            "strengths": [],
        }

        # Basic analysis logic (can be enhanced with LLM calls)
        cv_skills = cv_data.get("skills", [])
        job_requirements = job_description.get("requirements", [])

        # Simple skill matching
        for skill in cv_skills:
            for req in job_requirements:
                if skill.lower() in req.lower():
                    analysis["skill_matches"].append(skill)

        return analysis

    async def _generate_recommendations(
        self, analysis: Dict[str, Any], context: AgentExecutionContext
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if len(analysis["skill_matches"]) < 3:
            recommendations.append(
                "Consider highlighting more relevant technical skills"
            )

        if analysis["experience_relevance"] < 0.7:
            recommendations.append(
                "Emphasize experience that directly relates to the job requirements"
            )

        return recommendations

    def _calculate_match_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall match score."""
        # Simple scoring logic
        skill_score = min(len(analysis["skill_matches"]) * 0.2, 1.0)
        experience_score = analysis["experience_relevance"]
        gap_penalty = len(analysis["gaps_identified"]) * 0.1

        return max(0.0, (skill_score + experience_score) / 2 - gap_penalty)


# Helper function for quality checks (used by other agents)
def _check_item_quality_basic(
    item: Dict[str, Any], quality_criteria: Dict[str, Any]
) -> Dict[str, Any]:
    """Basic quality check for content items."""
    issues = []
    warnings = []
    score = 1.0

    content = item.get("content", "")

    # Length checks
    min_length = quality_criteria.get("min_length", 20)
    max_length = quality_criteria.get("max_length", 1000)

    if len(content) < min_length:
        issues.append(f"Content too short (minimum {min_length} characters)")
        score -= 0.3
    elif len(content) > max_length:
        warnings.append(
            f"Content might be too long (maximum {max_length} characters recommended)"
        )
        score -= 0.1

    # Grammar and formatting checks
    if not content.strip():
        issues.append("Empty content")
        score = 0.0
    elif content != content.strip():
        warnings.append("Content has leading/trailing whitespace")
        score -= 0.05

    # Professional language check (basic)
    unprofessional_words = ["awesome", "cool", "stuff", "things"]
    for word in unprofessional_words:
        if word.lower() in content.lower():
            warnings.append(f"Consider replacing informal word: '{word}'")
            score -= 0.05

    # Confidence score check
    confidence = item.get("confidence_score", 1.0)
    if confidence < 0.5:
        warnings.append("Low confidence score from content generation")
        score -= 0.1

    return {
        "passed": len(issues) == 0,
        "score": max(score, 0.0),
        "issues": issues,
        "warnings": warnings,
    }


# Factory functions for creating specialized agents
def create_cv_analysis_agent() -> CVAnalysisAgent:
    """Create a CV analysis agent."""
    return CVAnalysisAgent()


def create_quality_assurance_agent():
    """Create a quality assurance agent from the dedicated module."""
    from .quality_assurance_agent import QualityAssuranceAgent

    return QualityAssuranceAgent(
        name="QualityAssuranceAgent",
        description="Agent responsible for quality assurance of generated CV content",
    )


def create_enhanced_parser_agent() -> EnhancedAgentBase:
    """Create an enhanced parser agent that wraps the original ParserAgent."""
    return EnhancedParserAgent()


class EnhancedParserAgent(EnhancedAgentBase):
    """Enhanced wrapper for the original ParserAgent."""

    def __init__(self):
        super().__init__(
            name="EnhancedParserAgent",
            description="Enhanced parser agent for CV and job description parsing",
            input_schema={"cv_text": str, "job_description": str},
            output_schema={"structured_cv": Dict[str, Any], "job_data": Dict[str, Any]},
        )
        self.parser_agent = ParserAgent(
            name="ParserAgent",
            description="Parses CV and JD.",
            llm_service=get_llm_service(),
        )

    async def run_async(
        self, input_data: Any, context: AgentExecutionContext
    ) -> AgentResult:
        """Run the enhanced parser agent."""
        try:
            # Delegate to the original parser agent
            result = await self.parser_agent.run_as_node(input_data)

            return AgentResult(
                success=True,
                output_data=result,
                confidence_score=0.9,
                metadata={"parser_type": "enhanced"},
            )

        except Exception as e:
            self.logger.error(f"Enhanced parser failed: {str(e)}")
            return AgentResult(
                success=False,
                output_data={},
                error_message=str(e),
                confidence_score=0.0,
            )