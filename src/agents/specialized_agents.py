#!/usr/bin/env python3

import asyncio
from typing import Dict, Any, List
from datetime import datetime
import json

from pydantic import BaseModel

from .enhanced_content_writer import EnhancedContentWriterAgent, create_content_writer
from .agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from .parser_agent import ParserAgent
from ..models.data_models import StructuredCV, JobDescriptionData
from ..models.cv_analysis_result import CVAnalysisResult
from ..config.logging_config import get_structured_logger
from ..config.settings import get_config
from ..services.llm_service import EnhancedLLMService
from ..utils.agent_error_handling import AgentErrorHandler

logger = get_structured_logger("specialized_agents")


class CVAnalysisAgent(EnhancedAgentBase):
    """Agent specialized in analyzing CV content and job requirements using Pydantic models."""

    def __init__(self, llm_service, settings):
        super().__init__(
            name="CVAnalysisAgent",
            description="Analyzes CV content and job requirements to provide optimization recommendations",
            input_schema={
                "cv_data": StructuredCV,
                "job_description": JobDescriptionData,
            },
            output_schema={
                "analysis_results": CVAnalysisResult,
            },
        )
        self.llm_service = llm_service
        self.settings = settings

    async def run_async(
        self, input_data: dict, context: AgentExecutionContext
    ) -> AgentResult:
        """Analyze CV content against job requirements using Pydantic models."""
        from ..models.validation_schemas import validate_agent_input

        try:
            # Validate input data
            if not isinstance(input_data, dict):
                self.logger.error(
                    "Input validation failed for CVAnalysisAgent: input_data must be a dict"
                )
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
            cv_data = input_data.get("cv_data")
            job_description = input_data.get("job_description")
            if not isinstance(cv_data, StructuredCV):
                cv_data = StructuredCV.parse_obj(cv_data)
            if not isinstance(job_description, JobDescriptionData):
                job_description = JobDescriptionData.parse_obj(job_description)
            analysis = await self._analyze_cv_job_match(
                cv_data, job_description, context
            )
            recommendations = await self._generate_recommendations(analysis, context)
            match_score = self._calculate_match_score(analysis)
            result_data = CVAnalysisResult(
                skill_matches=analysis.skill_matches,
                experience_relevance=analysis.experience_relevance,
                gaps_identified=analysis.gaps_identified,
                strengths=analysis.strengths,
                recommendations=recommendations,
                match_score=match_score,
                analysis_timestamp=datetime.now().isoformat(),
            )
            return AgentResult(
                success=True,
                output_data=result_data,
                confidence_score=0.85,
                metadata={
                    "analysis_type": "cv_job_match",
                    "items_analyzed": len(cv_data.sections),
                },
            )
        except Exception as e:
            fallback_data = AgentErrorHandler.create_fallback_data("cv_analysis")
            return AgentErrorHandler.handle_general_error(
                e, "cv_analysis", fallback_data, "run_async"
            )

    class _AnalysisResult(BaseModel):
        skill_matches: List[str] = []
        experience_relevance: float = 0.0
        gaps_identified: List[str] = []
        strengths: List[str] = []

    async def _analyze_cv_job_match(
        self,
        cv_data: StructuredCV,
        job_description: JobDescriptionData,
        context: AgentExecutionContext,
    ) -> "CVAnalysisAgent._AnalysisResult":
        """Analyze match between CV and job requirements using Pydantic models."""
        analysis = self._AnalysisResult()
        cv_skills = getattr(cv_data, "big_10_skills", [])
        job_requirements = getattr(job_description, "skills", [])
        for skill in cv_skills:
            for req in job_requirements:
                if skill.lower() in req.lower():
                    analysis.skill_matches.append(skill)
        return analysis

    async def _generate_recommendations(
        self,
        analysis: "CVAnalysisAgent._AnalysisResult",
        context: AgentExecutionContext,
    ) -> List[str]:
        recommendations = []
        if len(analysis.skill_matches) < 3:
            recommendations.append(
                "Consider highlighting more relevant technical skills"
            )
        if analysis.experience_relevance < 0.7:
            recommendations.append(
                "Emphasize experience that directly relates to the job requirements"
            )
        return recommendations

    def _calculate_match_score(
        self, analysis: "CVAnalysisAgent._AnalysisResult"
    ) -> float:
        skill_score = min(len(analysis.skill_matches) * 0.2, 1.0)
        experience_score = analysis.experience_relevance
        gap_penalty = len(analysis.gaps_identified) * 0.1
        return max(0.0, (skill_score + experience_score) / 2 - gap_penalty)

    async def run_as_node(self, state):
        """Stub for LangGraph node execution (not used in CVAnalysisAgent)."""
        raise NotImplementedError("CVAnalysisAgent does not implement run_as_node.")


# ContentOptimizationAgent and related logic have been removed as part of the architectural refactor. All content writing is now managed explicitly in the workflow graph.


# Helper function for quality checks (used by other agents)
def _check_item_quality_basic(item: dict, quality_criteria: dict) -> dict:
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
            output_schema={
                "structured_cv": StructuredCV,
                "job_data": JobDescriptionData,
            },
        )
        self.parser_agent = ParserAgent(
            name="ParserAgent",
            description="Parses CV and JD.",
            # llm_service must be injected by the factory or DI container
        )

    async def run_async(
        self, input_data: dict, context: AgentExecutionContext
    ) -> AgentResult:
        try:
            result = await self.parser_agent.run_as_node(input_data)
            # Expect result to be a dict with 'structured_cv' and 'job_data' as Pydantic models
            structured_cv = result.get("structured_cv")
            # Use the correct key for job data
            job_data = result.get("job_description_data")
            if not isinstance(structured_cv, StructuredCV):
                structured_cv = StructuredCV.parse_obj(structured_cv)
            if not isinstance(job_data, JobDescriptionData):
                job_data = JobDescriptionData.parse_obj(job_data)
            return AgentResult(
                success=True,
                output_data={"structured_cv": structured_cv, "job_data": job_data},
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

    async def run_as_node(self, state):
        """Stub for LangGraph node execution (not used in EnhancedParserAgent)."""
        raise NotImplementedError("EnhancedParserAgent does not implement run_as_node.")
