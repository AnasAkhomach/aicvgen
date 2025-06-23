from .agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from ..models.data_models import AgentIO
from ..config.logging_config import get_structured_logger
from ..services.llm_service import EnhancedLLMService
from ..services.error_recovery import ErrorRecoveryService
from ..services.progress_tracker import ProgressTracker
from ..utils.exceptions import ValidationError
from ..utils.agent_error_handling import AgentErrorHandler, with_node_error_handling
from src.utils.prompt_utils import load_prompt_template, format_prompt
from src.utils.error_utils import handle_errors
from typing import Dict, Any, List
from ..models.research_models import (
    ResearchFindings,
    ResearchStatus,
    CompanyInsight,
    IndustryInsight,
    RoleInsight,
)

logger = get_structured_logger(__name__)


class ResearchAgent(EnhancedAgentBase):
    """
    Agent responsible for conducting research related to the job description and finding
    relevant content from the CV for tailoring.
    """

    def __init__(
        self,
        llm_service: EnhancedLLMService,
        error_recovery_service: ErrorRecoveryService,
        progress_tracker: ProgressTracker,
        vector_db,
        settings,
        name: str = "ResearchAgent",
        description: str = "Agent responsible for conducting research and gathering insights",
    ):
        input_schema = AgentIO(
            description="Job description and CV for research",
            required_fields=["job_description", "structured_cv"],
        )
        output_schema = AgentIO(
            description="Research findings and insights",
            required_fields=["research_findings"],
        )
        super().__init__(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            error_recovery_service=error_recovery_service,
            progress_tracker=progress_tracker,
        )
        self.llm_service = llm_service
        self.vector_db = vector_db
        self.settings = settings
        self.current_session_id = None
        self.current_trace_id = None
        self._latest_research_results = None

    async def run_async(
        self, input_data: Any, context: "AgentExecutionContext"
    ) -> "AgentResult":
        try:
            try:
                validated_input = handle_errors(lambda: input_data)("research")
                input_data = validated_input
                self.log_decision(
                    "Input validation passed for ResearchAgent", context, "validation"
                )
            except ValidationError as ve:
                fallback_data = AgentErrorHandler.create_fallback_data("research")
                return AgentErrorHandler.handle_validation_error(ve, "research")
            except ValueError as ve:
                fallback_data = AgentErrorHandler.create_fallback_data("research")
                return AgentErrorHandler.handle_general_error(
                    ve, "research", fallback_data, "run_async"
                )
            except Exception as e:
                fallback_data = AgentErrorHandler.create_fallback_data("research")
                return AgentErrorHandler.handle_general_error(
                    e, "research", fallback_data, "run_async"
                )
            if input_data is None:
                raise ValueError("input_data cannot be None for research agent")
            structured_cv = input_data.structured_cv
            job_desc_data = input_data.job_description_data
            if (
                hasattr(job_desc_data, "company_name")
                and not job_desc_data.company_name
            ):
                job_desc_data = job_desc_data.copy(
                    update={"company_name": "Unknown Company"}
                )
            if hasattr(job_desc_data, "title") and not job_desc_data.title:
                job_desc_data = job_desc_data.copy(update={"title": "Unknown Title"})
            research_findings = await self._perform_research_analysis(
                structured_cv, job_desc_data
            )
            result = {"research_findings": research_findings}
            return AgentResult(
                success=True,
                output_data=result,
                confidence_score=1.0,
                metadata={"agent_type": "research"},
            )
        except Exception as e:
            fallback_data = AgentErrorHandler.create_fallback_data("research")
            return AgentErrorHandler.handle_general_error(
                e, "research", fallback_data, "run_async"
            )

    async def _perform_research_analysis(
        self, structured_cv, job_desc_data
    ) -> ResearchFindings:
        try:
            self.current_session_id = getattr(self, "current_session_id", None)
            self.current_trace_id = getattr(self, "current_trace_id", None)
            if self.vector_db:
                self._index_cv_content(structured_cv)
            skills = getattr(job_desc_data, "skills", []) or []
            responsibilities = getattr(job_desc_data, "responsibilities", []) or []
            experience_level = getattr(
                job_desc_data, "experience_level", "Not specified"
            )
            job_analysis = await self.analyze_job_description(
                job_desc_data.raw_text, skills, responsibilities, experience_level
            )
            company_info = await self._research_company_info(job_desc_data.raw_text)
            relevant_content = []
            if job_analysis.get("key_skills"):
                for skill in job_analysis["key_skills"][:3]:
                    content = self._search_relevant_content(
                        skill, structured_cv, num_results=2
                    )
                    relevant_content.extend(content)
            industry_insights = self._research_industry_insights(
                job_analysis.get("key_skills", [])
            )
            company_name = company_info.get("company_name") or "Unknown Company"
            industry_name = str(company_info.get("industry") or "")
            key_values = company_info.get("values") or []
            raw_culture = company_info.get("values")
            if isinstance(raw_culture, list):
                culture = ", ".join(str(v) for v in raw_culture)
            else:
                culture = str(raw_culture) if raw_culture is not None else None
            research_findings = ResearchFindings(
                status=ResearchStatus.SUCCESS,
                key_terms=job_analysis.get("key_skills", []),
                skill_gaps=[],
                enhancement_suggestions=[],
                company_insights=(
                    CompanyInsight(
                        company_name=company_name,
                        industry=industry_name,
                        size=None,
                        culture=culture,
                        recent_news=[],
                        key_values=key_values,
                        confidence_score=0.8,
                    )
                    if company_info
                    else None
                ),
                industry_insights=(
                    IndustryInsight(
                        industry_name=industry_name,
                        trends=industry_insights.get("trends", []),
                        key_skills=industry_insights.get("technologies", []),
                        growth_areas=industry_insights.get("growth_areas", []),
                        challenges=[],
                        confidence_score=0.7,
                    )
                    if industry_insights
                    else None
                ),
                role_insights=(
                    RoleInsight(
                        role_title="",
                        required_skills=job_analysis.get("key_skills", []),
                        preferred_qualifications=[],
                        responsibilities=job_analysis.get("responsibilities", []),
                        career_progression=[],
                        salary_range=None,
                        confidence_score=0.8,
                    )
                    if job_analysis
                    else None
                ),
                confidence_score=0.8,
                processing_time_seconds=0.0,
            )
            self._latest_research_results = research_findings
            return research_findings
        except Exception as e:
            logger.error("Error in research analysis: %s", str(e))
            return ResearchFindings.create_failed(f"Research analysis failed: {str(e)}")

    def get_research_results(self) -> ResearchFindings:
        if hasattr(self, "_latest_research_results") and self._latest_research_results:
            if isinstance(self._latest_research_results, ResearchFindings):
                return self._latest_research_results
            elif isinstance(self._latest_research_results, dict):
                return ResearchFindings.from_dict(self._latest_research_results)
        return ResearchFindings.create_empty()

    def _extract_field(self, data: Dict[str, Any], field: str, default: Any) -> Any:
        if hasattr(data, "get") and callable(data.get):
            return data.get(field, default)
        elif hasattr(data, field):
            return getattr(data, field, default)
        return default

    def _index_cv_content(self, structured_cv) -> None:
        if not self.vector_db:
            return
        cv_id = structured_cv.id
        if (
            hasattr(self.vector_db, "indexed_cv_ids")
            and cv_id in self.vector_db.indexed_cv_ids
        ):
            return
        if not hasattr(self.vector_db, "indexed_cv_ids"):
            self.vector_db.indexed_cv_ids = set()
        for section in structured_cv.sections:
            for item in section.items:
                if item.content:
                    try:
                        metadata = {
                            "id": item.id,
                            "section": section.name,
                            "type": str(item.item_type),
                        }
                        self.vector_db.add_item(item, item.content, metadata=metadata)
                    except Exception as e:
                        logger.error("Error indexing item %s: %s", item.id, str(e))
            for subsection in section.subsections:
                for item in subsection.items:
                    if item.content:
                        try:
                            metadata = {
                                "id": item.id,
                                "section": section.name,
                                "subsection": subsection.name,
                                "type": str(item.item_type),
                            }
                            self.vector_db.add_item(
                                item, item.content, metadata=metadata
                            )
                        except Exception as e:
                            logger.error(
                                "Error indexing subsection item %s: %s", item.id, str(e)
                            )
        self.vector_db.indexed_cv_ids.add(cv_id)

    def _search_relevant_content(
        self, query: str, structured_cv, num_results: int = 5
    ) -> List[Dict[str, Any]]:
        if not self.vector_db:
            return []
        try:
            results = self.vector_db.search(query, k=num_results)
            relevant_items = []
            for result in results:
                item_id = getattr(result, "id", None)
                if not item_id:
                    continue
                item, section, subsection = structured_cv.find_item_by_id(item_id)
                if not item:
                    continue
                relevant_items.append(
                    {
                        "id": item.id,
                        "content": item.content,
                        "section": section.name if section else "",
                        "subsection": subsection.name if subsection else "",
                        "type": str(item.item_type),
                    }
                )
            return relevant_items
        except Exception as e:
            logger.error("Error searching for relevant content: %s", str(e))
            return []

    def _calculate_section_relevance(
        self, structured_cv, terms: List[str]
    ) -> Dict[str, float]:
        section_scores = {}
        for section in structured_cv.sections:
            score = 0
            matched_terms = 0
            for item in section.items:
                content = item.content.lower()
                for term in terms:
                    if term.lower() in content:
                        score += 1
                        matched_terms += 1
            for subsection in section.subsections:
                for item in subsection.items:
                    content = item.content.lower()
                    for term in terms:
                        if term.lower() in content:
                            score += 1
                            matched_terms += 1
            if terms:
                section_scores[section.name] = max(
                    0.1, min(1.0, score / (len(terms) * 0.7))
                )
            else:
                section_scores[section.name] = 0.5
        return section_scores

    @handle_errors(default_return=None)
    async def analyze_job_description(
        self, raw_jd, skills, responsibilities, experience_level
    ):
        prompt_path = self.settings.get_prompt_path("job_research_analysis_prompt")
        fallback_template = """
                Analyze the job description: {raw_jd}
                Skills context: {skills}
                Responsibilities: {responsibilities}
                Experience level: {experience_level}
                Return analysis as JSON.
                """
        prompt_template = load_prompt_template(prompt_path, fallback=fallback_template)
        prompt = format_prompt(
            prompt_template,
            raw_jd=raw_jd,
            skills=", ".join(skills),
            responsibilities=", ".join(responsibilities),
            experience_level=experience_level,
        )
        analysis = await self._generate_and_parse_json(
            prompt=prompt,
            session_id=getattr(self, "current_session_id", None),
            trace_id=getattr(self, "current_trace_id", None),
        )
        return analysis

    async def _research_company_info(self, raw_jd: str) -> Dict[str, Any]:
        try:
            company_prompt = f"""
            From the following job description, extract:
            1. The company name
            2. The industry or sector
            3. Any mentioned company values or culture aspects

            Format your response as a JSON object with keys: "company_name", "industry", "values".

            Job Description:
            {raw_jd}

            IMPORTANT: Respond ONLY with a valid JSON object.
            """
            try:
                company_info = await self._generate_and_parse_json(
                    prompt=company_prompt,
                    session_id=getattr(self, "current_session_id", None),
                    trace_id=getattr(self, "current_trace_id", None),
                )
                company_name = company_info.get("company_name", "Unknown Company")
                company_info["description"] = f"Research insights about {company_name}."
                company_info["key_products"] = ["Product A", "Product B"]
                company_info["market_position"] = "Market leader in their segment"
                return company_info
            except Exception as e:
                logger.error("Company research LLM request failed: %s", str(e))
                return {
                    "company_name": "Unknown Company",
                    "industry": "Technology",
                    "values": ["Innovation", "Teamwork"],
                    "error": "LLM company research failed",
                }
        except Exception as e:
            logger.error("Error researching company info: %s", str(e))
        return {
            "company_name": "Unknown Company",
            "industry": "Technology",
            "values": ["Innovation", "Teamwork"],
        }

    def _research_industry_insights(self, terms: List[str]) -> Dict[str, Any]:
        return {
            "trends": [
                f"Growing demand for {terms[0] if terms else 'this skill set'}",
                "Increased focus on automation and efficiency",
                "Shift toward remote and hybrid work environments",
            ],
            "technologies": [
                "Cloud-based collaboration tools",
                "Data analytics platforms",
                "Automation software",
            ],
            "growth_areas": [
                "Remote team management",
                "Process optimization",
                "Customer experience enhancement",
            ],
        }

    @with_node_error_handling("research")
    async def run_as_node(self, state):
        if getattr(self, "_dependency_error", None):
            return state.model_copy(
                update={
                    "error_messages": state.error_messages
                    + [f"ResearchAgent dependency error: {self._dependency_error}"]
                }
            )
        logger.info("Research Agent: Starting node execution")
        from ..models.validation_schemas import validate_agent_input

        validate_agent_input("research", state)
        agent_input = state
        result = await self.run_async(agent_input, state)
        if result.success:
            if result.output_data and "research_findings" in result.output_data:
                research_findings = result.output_data["research_findings"]
                if isinstance(research_findings, dict):
                    research_findings = ResearchFindings.from_dict(research_findings)
                logger.info("Research Agent: Successfully completed research")
                return state.model_copy(update={"research_findings": research_findings})
            else:
                error_msg = "Research Agent: No research findings in result"
                logger.error(error_msg)
                return state.model_copy(
                    update={"error_messages": state.error_messages + [error_msg]}
                )
        else:
            error_msg = f"Research Agent failed: {result.error_message}"
            logger.error(error_msg)
            return state.model_copy(
                update={"error_messages": state.error_messages + [error_msg]}
            )
