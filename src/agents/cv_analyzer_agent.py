"""CV Analyzer Agent"""

from datetime import datetime
from typing import Any, List

from pydantic import BaseModel, ValidationError

from src.agents.agent_base import AgentBase, AgentResult
from src.config.logging_config import get_structured_logger
from src.config.settings import get_config
from src.constants.analysis_constants import AnalysisConstants
from src.constants.agent_constants import AgentConstants
from src.error_handling.agent_error_handler import AgentErrorHandler

from src.models.agent_output_models import (CVAnalysisResult, CVAnalyzerAgentOutput)
from src.models.cv_models import JobDescriptionData, StructuredCV
from src.services.llm_service_interface import LLMServiceInterface

logger = get_structured_logger("cv_analyzer_agent")


class CVAnalyzerAgent(AgentBase):
    """Agent specialized in analyzing CV content and job requirements using Pydantic models."""

    def __init__(self, llm_service: LLMServiceInterface, session_id: str = "default"):
        super().__init__(
            name="CVAnalyzerAgent",
            description="Analyzes CV content and job requirements to provide optimization recommendations",
            session_id=session_id,
        )
        self.llm_service = llm_service
        self.settings = get_config()

    async def _execute(self, **kwargs: Any) -> AgentResult:
        """Analyze CV content against job requirements using Pydantic models."""
        input_data = kwargs.get("input_data")

        try:
            self.update_progress(AgentConstants.PROGRESS_START, "Starting CV analysis")
            
            cv_data = input_data.get("cv_data")
            job_description = input_data.get("job_description")
            if not isinstance(cv_data, StructuredCV):
                cv_data = StructuredCV.model_validate(cv_data)
            if not isinstance(job_description, JobDescriptionData):
                job_description = JobDescriptionData.model_validate(job_description)
            
            self.update_progress(AgentConstants.PROGRESS_INPUT_VALIDATION, "Input validation completed")
            
            analysis = await self._analyze_cv_job_match(
                cv_data, job_description
            )
            
            self.update_progress(AgentConstants.PROGRESS_MAIN_PROCESSING, "Analyzing CV-job match")
            
            recommendations = await self._generate_recommendations(analysis)
            match_score = self._calculate_match_score(analysis)
            
            self.update_progress(AgentConstants.PROGRESS_POST_PROCESSING, "Generating recommendations and scores")
            
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
            
            self.update_progress(AgentConstants.PROGRESS_COMPLETE, "CV analysis completed successfully")
            
            return AgentResult(
                success=True,
                output_data=output_data,
                confidence_score=AnalysisConstants.DEFAULT_CONFIDENCE_SCORE,
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
    ) -> List[str]:
        recommendations = []
        if len(analysis.skill_matches) < AnalysisConstants.MIN_SKILL_MATCHES:
            recommendations.append(
                "Consider highlighting more relevant technical skills"
            )
        if analysis.experience_relevance < AnalysisConstants.MIN_EXPERIENCE_RELEVANCE:
            recommendations.append(
                "Emphasize experience that directly relates to the job requirements"
            )
        return recommendations

    def _calculate_match_score(
        self, analysis: "CVAnalyzerAgent._AnalysisResult"
    ) -> float:
        skill_score = min(len(analysis.skill_matches) * AnalysisConstants.SKILL_SCORE_MULTIPLIER, AnalysisConstants.MAX_MATCH_SCORE)
        experience_score = analysis.experience_relevance
        gap_penalty = len(analysis.gaps_identified) * AnalysisConstants.GAP_PENALTY_MULTIPLIER
        return max(AnalysisConstants.MIN_MATCH_SCORE, (skill_score + experience_score) * AnalysisConstants.SCORE_WEIGHT_FACTOR - gap_penalty)
