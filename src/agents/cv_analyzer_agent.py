"""CV Analyzer Agent"""

from datetime import datetime
from typing import Any, List

from pydantic import BaseModel, ValidationError

from src.agents.agent_base import AgentBase
from src.config.logging_config import get_structured_logger
from src.config.settings import get_config
from src.constants.analysis_constants import AnalysisConstants
from src.constants.agent_constants import AgentConstants

from src.models.agent_output_models import CVAnalysisResult
from src.models.cv_models import JobDescriptionData, StructuredCV
from src.services.llm_service_interface import LLMServiceInterface
from src.utils.node_validation import ensure_pydantic_model

logger = get_structured_logger("cv_analyzer_agent")


class CVAnalyzerAgent(AgentBase):
    """Agent specialized in analyzing CV content and job requirements using Pydantic models."""

    def __init__(self, llm_service: LLMServiceInterface, settings: dict, session_id: str):
        super().__init__(
            name="CVAnalyzerAgent",
            description="Analyzes CV content and job requirements to provide optimization recommendations",
            session_id=session_id,
            settings=settings
        )
        self.llm_service = llm_service
        self.settings = get_config()

    @ensure_pydantic_model(
        ('cv_data', StructuredCV),
        ('job_description', JobDescriptionData)
    )
    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        """Analyze CV content against job requirements using Pydantic models."""
        try:
            self.update_progress(AgentConstants.PROGRESS_START, "Starting CV analysis")

            cv_data = kwargs.get("cv_data")
            job_description = kwargs.get("job_description")

            if not cv_data:
                return {"error_messages": ["cv_data is required but not provided"]}
            if not job_description:
                return {"error_messages": ["job_description is required but not provided"]}

            # Pydantic validation is now handled by the decorator
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

            self.update_progress(AgentConstants.PROGRESS_COMPLETE, "CV analysis completed successfully")

            return {
                "cv_analysis_results": analysis_result
            }
        except (ValidationError, KeyError, TypeError, AttributeError) as e:
            logger.error(f"CV analysis error: {str(e)}")
            return {"error_messages": [f"CV analysis failed: {str(e)}"]}

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
        
        # Extract system instruction from settings
        system_instruction = None
        if self.settings and hasattr(self.settings, 'agent_settings'):
            system_instruction = getattr(self.settings.agent_settings, 'cv_analyzer_system_instruction', None)
        elif self.settings and isinstance(self.settings, dict):
            system_instruction = self.settings.get('cv_analyzer_system_instruction')
        
        # Basic skill matching logic
        if cv_skills and job_requirements:
            for skill in cv_skills:
                for req in job_requirements:
                    if skill.lower() in req.lower():
                        analysis.skill_matches.append(skill)
        
        # TODO: Enhance with LLM-based analysis using system instruction
        # This would involve calling self.llm_service.generate_content with system_instruction
        
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
