"""CV Analyzer Agent"""

from datetime import datetime
from typing import List
from pydantic import BaseModel, ValidationError

from .agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from ..models.data_models import StructuredCV, JobDescriptionData
from ..models.agent_output_models import CVAnalysisResult, CVAnalyzerAgentOutput
from ..config.logging_config import get_structured_logger
from ..config.settings import get_config
from ..services.llm_service import EnhancedLLMService
from ..error_handling.agent_error_handler import AgentErrorHandler
from ..error_handling.exceptions import (
    AgentExecutionError,
    LLMResponseParsingError,
    DataConversionError,
)

logger = get_structured_logger("cv_analyzer_agent")


class CVAnalyzerAgent(EnhancedAgentBase):
    """Agent specialized in analyzing CV content and job requirements using Pydantic models."""

    def __init__(self, llm_service: EnhancedLLMService):
        super().__init__(
            name="CVAnalyzerAgent",
            description="Analyzes CV content and job requirements to provide optimization recommendations",
            input_schema={
                "cv_data": StructuredCV,
                "job_description": JobDescriptionData,
            },
            output_schema={
                "analysis_results": CVAnalysisResult,
            },
            error_recovery_service=None,  # TODO: Inject actual service if needed
            progress_tracker=None,  # TODO: Inject actual tracker if needed
        )
        self.llm_service = llm_service
        self.settings = get_config()

    async def run_async(
        self, input_data: dict, context: AgentExecutionContext
    ) -> AgentResult:
        """Analyze CV content against job requirements using Pydantic models."""
        from ..models.validation_schemas import validate_agent_input

        try:
            # Validate input data
            if not isinstance(input_data, dict):
                self.logger.error(
                    "Input validation failed for CVAnalyzerAgent: input_data must be a dict"
                )
                return AgentResult(
                    success=False,
                    output_data=CVAnalysisResult(
                        skill_matches=[],
                        experience_relevance=0.0,
                        gaps_identified=[
                            "Input validation failed: expected dictionary input"
                        ],
                        strengths=[],
                        recommendations=[],
                        match_score=0.0,
                        analysis_timestamp=None,
                    ),
                    confidence_score=0.0,
                    error_message="Input validation failed: expected dictionary input",
                    metadata={
                        "agent_type": "cv_analysis",
                    },
                )
            cv_data = input_data.get("cv_data")
            job_description = input_data.get("job_description")
            if not isinstance(cv_data, StructuredCV):
                cv_data = StructuredCV.model_validate(cv_data)
            if not isinstance(job_description, JobDescriptionData):
                job_description = JobDescriptionData.model_validate(job_description)
            analysis = await self._analyze_cv_job_match(
                cv_data, job_description, context
            )
            recommendations = await self._generate_recommendations(analysis, context)
            match_score = self._calculate_match_score(analysis)
            analysis_result = CVAnalysisResult(
                skill_matches=analysis.skill_matches,
                experience_relevance=analysis.experience_relevance,
                gaps_identified=analysis.gaps_identified,
                strengths=analysis.strengths,
                recommendations=recommendations,
                match_score=match_score,
                analysis_timestamp=datetime.now().isoformat(),
            )
            output_data = CVAnalyzerAgentOutput(
                analysis_results=analysis_result,
                recommendations=recommendations,
                compatibility_score=match_score,
            )
            return AgentResult(
                success=True,
                output_data=output_data,
                confidence_score=0.85,
                metadata={
                    "analysis_type": "cv_job_match",
                    "items_analyzed": (
                        len(cv_data.sections)
                        if hasattr(cv_data, "sections") and cv_data.sections
                        else 0
                    ),
                },
            )
        except (ValidationError, KeyError, TypeError, AttributeError) as e:
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
    ) -> "CVAnalyzerAgent._AnalysisResult":
        """Analyze match between CV and job requirements using Pydantic models."""
        analysis = self._AnalysisResult()
        cv_skills = getattr(cv_data, "big_10_skills", [])
        job_requirements = getattr(job_description, "skills", [])
        if cv_skills and job_requirements:
            for skill in cv_skills:
                for req in job_requirements:
                    if skill.lower() in req.lower():
                        analysis.skill_matches.append(skill)
        return analysis

    async def _generate_recommendations(
        self,
        analysis: "CVAnalyzerAgent._AnalysisResult",
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
        self, analysis: "CVAnalyzerAgent._AnalysisResult"
    ) -> float:
        skill_score = min(len(analysis.skill_matches) * 0.2, 1.0)
        experience_score = analysis.experience_relevance
        gap_penalty = len(analysis.gaps_identified) * 0.1
        return max(0.0, (skill_score + experience_score) / 2 - gap_penalty)

    async def run_as_node(self, state):
        """Stub for LangGraph node execution (not used in CVAnalyzerAgent)."""
        raise NotImplementedError("CVAnalyzerAgent does not implement run_as_node.")
