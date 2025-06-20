from .agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from ..services.llm_service import get_llm_service
from ..core.state_manager import (
    JobDescriptionData,
    AgentIO,
    StructuredCV,
    Section,
    Subsection,
    Item,
)
from ..services.vector_store_service import get_vector_store_service
from ..config.logging_config import get_structured_logger
from ..models.data_models import AgentDecisionLog, AgentExecutionLog
from ..config.settings import get_config
from ..services.llm_service import LLMResponse
from ..utils.exceptions import (
    ValidationError,
    LLMResponseParsingError,
    WorkflowPreconditionError,
    AgentExecutionError,
    ConfigurationError,
    StateManagerError,
)
from ..utils.agent_error_handling import (
    AgentErrorHandler,
    LLMErrorHandler,
    with_error_handling,
    with_node_error_handling,
)
from typing import Dict, Any, List, Optional
import time
import json
import os
import asyncio
from datetime import datetime
from ..models.research_models import ResearchFindings, ResearchStatus, CompanyInsight, IndustryInsight, RoleInsight
from ..orchestration.state import AgentState
from ..core.async_optimizer import optimize_async

# Set up structured logging
logger = get_structured_logger(__name__)


class ResearchAgent(EnhancedAgentBase):
    """
    Agent responsible for conducting research related to the job description and finding
    relevant content from the CV for tailoring.

    This agent fulfills REQ-FUNC-RESEARCH-1, REQ-FUNC-RESEARCH-2, REQ-FUNC-RESEARCH-3,
    and REQ-FUNC-RESEARCH-4 from the SRS.
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm_service=None,
    ):
        self._latest_research_results = {}  # Initialize research results storage
        """
        Initializes the ResearchAgent.

        Args:
            name: The name of the agent.
            description: A description of the agent.
            llm_service: Optional LLM service instance for analysis.
        """
        super().__init__(
            name=name,
            description=description,
            input_schema=AgentIO(
                description="Reads structured CV and job description data from AgentState for research analysis.",
                required_fields=["structured_cv", "job_description_data"],
                optional_fields=[],
            ),
            output_schema=AgentIO(
                description="Populates the 'research_findings' field in AgentState with analysis results.",
                required_fields=["research_findings"],
                optional_fields=["error_messages"],
            ),
        )
        self.llm = llm_service or get_llm_service()
        self.vector_db = get_vector_store_service()

        # Initialize settings for prompt loading
        self.settings = get_config()

    async def run_async(
        self, input_data: Any, context: "AgentExecutionContext"
    ) -> "AgentResult":
        """Async run method for consistency with enhanced agent interface."""
        from .agent_base import AgentResult
        from ..models.validation_schemas import validate_agent_input

        try:
            # Validate input data using Pydantic schemas
            try:
                validated_input = validate_agent_input("research", input_data)
                # Use validated input data directly
                input_data = validated_input
                # Log validation success with structured logging
                self.log_decision(
                    "Input validation passed for ResearchAgent", context, "validation"
                )
            except ValidationError as ve:
                fallback_data = AgentErrorHandler.create_fallback_data("research")
                return AgentErrorHandler.handle_validation_error(
                    ve, "research", fallback_data, "run_async"
                )
            except Exception as e:
                fallback_data = AgentErrorHandler.create_fallback_data("research")
                return AgentErrorHandler.handle_general_error(
                    e, "research", fallback_data, "run_async"
                )

            # Perform research analysis directly without circular dependency
            from ..models.data_models import StructuredCV, JobDescriptionData

            # Handle None input_data
            if input_data is None:
                input_data = {}

            # Create proper StructuredCV and JobDescriptionData objects
            structured_cv = input_data.get("structured_cv")
            if isinstance(structured_cv, dict):
                structured_cv = StructuredCV(**structured_cv)
            elif not structured_cv:
                structured_cv = StructuredCV()

            job_desc_data = input_data.get("job_description_data")
            if isinstance(job_desc_data, dict):
                job_desc_data = JobDescriptionData(**job_desc_data)
            elif not job_desc_data:
                job_desc_data = JobDescriptionData(
                    raw_text=input_data.get("job_description", "")
                )

            # Perform the actual research work
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
            # Use standardized error handling
            fallback_data = AgentErrorHandler.create_fallback_data("research")
            return AgentErrorHandler.handle_general_error(
                e, "research", fallback_data, "run_async"
            )

    async def _perform_research_analysis(
        self, structured_cv: "StructuredCV", job_desc_data: "JobDescriptionData"
    ) -> ResearchFindings:
        """Perform the core research analysis without circular dependencies."""
        try:
            # Set current session and trace IDs for centralized JSON parsing
            self.current_session_id = getattr(self, "current_session_id", None)
            self.current_trace_id = getattr(self, "current_trace_id", None)

            # Index CV content if vector DB is available
            if self.vector_db:
                self._index_cv_content(structured_cv)

            # Extract basic job data for analysis
            skills = getattr(job_desc_data, "skills", []) or []
            responsibilities = getattr(job_desc_data, "responsibilities", []) or []
            experience_level = getattr(
                job_desc_data, "experience_level", "Not specified"
            )

            # Analyze job requirements
            job_analysis = await self._analyze_job_requirements(
                job_desc_data.raw_text, skills, responsibilities, experience_level
            )

            # Research company information
            company_info = await self._research_company_info(job_desc_data.raw_text)

            # Find relevant CV content
            relevant_content = []
            if job_analysis.get("key_skills"):
                for skill in job_analysis["key_skills"][:3]:  # Limit to top 3 skills
                    content = self._search_relevant_content(
                        skill, structured_cv, num_results=2
                    )
                    relevant_content.extend(content)

            # Research industry insights
            industry_insights = self._research_industry_insights(
                job_analysis.get("key_skills", [])
            )

            # Compile research findings
            research_findings = ResearchFindings(
                status=ResearchStatus.SUCCESS,
                key_terms=job_analysis.get("key_skills", []),
                skill_gaps=[],
                enhancement_suggestions=[],
                company_insights=CompanyInsight(
                    company_name=company_info.get("company_name", ""),
                    industry=company_info.get("industry"),
                    size=None,
                    culture=company_info.get("values"),
                    recent_news=[],
                    key_values=company_info.get("values", []),
                    confidence_score=0.8
                ) if company_info else None,
                industry_insights=IndustryInsight(
                    industry_name=company_info.get("industry", "") if company_info else "",
                    trends=industry_insights.get("trends", []),
                    key_skills=industry_insights.get("technologies", []),
                    growth_areas=industry_insights.get("growth_areas", []),
                    challenges=[],
                    confidence_score=0.7
                ) if industry_insights else None,
                role_insights=RoleInsight(
                    role_title="",
                    required_skills=job_analysis.get("key_skills", []),
                    preferred_qualifications=[],
                    responsibilities=job_analysis.get("responsibilities", []),
                    career_progression=[],
                    salary_range=None,
                    confidence_score=0.8
                ) if job_analysis else None,
                confidence_score=0.8,
                processing_time_seconds=0.0
            )

            # Store for later retrieval
            self._latest_research_results = research_findings

            return research_findings

        except Exception as e:
            logger.error(f"Error in research analysis: {str(e)}")
            return ResearchFindings.create_failed(f"Research analysis failed: {str(e)}")

    def get_research_results(self) -> ResearchFindings:
        """
        Returns the previously generated research results.

        Returns:
            The research results as ResearchFindings, or empty findings if no research has been performed
        """
        if hasattr(self, "_latest_research_results") and self._latest_research_results:
            if isinstance(self._latest_research_results, ResearchFindings):
                return self._latest_research_results
            elif isinstance(self._latest_research_results, dict):
                return ResearchFindings.from_dict(self._latest_research_results)
        return ResearchFindings.create_empty()

    def _extract_field(self, data: Dict[str, Any], field: str, default: Any) -> Any:
        """Helper to extract a field from job description data, handling different formats."""
        if hasattr(data, "get") and callable(data.get):
            return data.get(field, default)
        elif hasattr(data, field):
            return getattr(data, field, default)
        return default

    def _index_cv_content(self, structured_cv: StructuredCV) -> None:
        """
        Indexes the CV content in the vector database.

        Args:
            structured_cv: The structured CV to index
        """
        if not self.vector_db:
            return

        # Track if we've already indexed this CV to avoid duplicates
        cv_id = structured_cv.id
        if (
            hasattr(self.vector_db, "indexed_cv_ids")
            and cv_id in self.vector_db.indexed_cv_ids
        ):
            return

        # Initialize tracking set if it doesn't exist
        if not hasattr(self.vector_db, "indexed_cv_ids"):
            self.vector_db.indexed_cv_ids = set()

        # Index each section and item
        for section in structured_cv.sections:
            # Index direct items in the section
            for item in section.items:
                if item.content:
                    try:
                        # Store ID mapping for easy lookup later
                        metadata = {
                            "id": item.id,
                            "section": section.name,
                            "type": str(item.item_type),
                        }
                        self.vector_db.add_item(item, item.content, metadata=metadata)
                    except Exception as e:
                        logger.error(f"Error indexing item {item.id}: {str(e)}")

            # Index subsections and their items
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
                                f"Error indexing subsection item {item.id}: {str(e)}"
                            )

        # Mark this CV as indexed
        self.vector_db.indexed_cv_ids.add(cv_id)

    def _search_relevant_content(
        self, query: str, structured_cv: StructuredCV, num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Searches for content in the CV relevant to the query.

        Args:
            query: The search query
            structured_cv: The structured CV to search in
            num_results: The number of results to return

        Returns:
            List of relevant content items with metadata
        """
        if not self.vector_db:
            return []

        try:
            results = self.vector_db.search(query, k=num_results)

            # Format the results
            relevant_items = []
            for result in results:
                # Get the full item from the structured CV
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
            logger.error(f"Error searching for relevant content: {str(e)}")
            return []

    def _calculate_section_relevance(
        self, structured_cv: StructuredCV, terms: List[str]
    ) -> Dict[str, float]:
        """
        Calculates relevance scores for each section based on search terms.

        Args:
            structured_cv: The structured CV
            terms: The search terms

        Returns:
            Dictionary mapping section names to relevance scores
        """
        # For the MVP, we'll use a simplified scoring approach
        # In a real implementation, this would use more sophisticated semantic matching

        section_scores = {}

        # Check each section
        for section in structured_cv.sections:
            score = 0
            matched_terms = 0

            # Check direct items in the section
            for item in section.items:
                content = item.content.lower()
                for term in terms:
                    if term.lower() in content:
                        score += 1
                        matched_terms += 1

            # Check subsections and their items
            for subsection in section.subsections:
                for item in subsection.items:
                    content = item.content.lower()
                    for term in terms:
                        if term.lower() in content:
                            score += 1
                            matched_terms += 1

            # Calculate a normalized score if we have terms to match
            if terms:
                section_scores[section.name] = max(
                    0.1, min(1.0, score / (len(terms) * 0.7))
                )
            else:
                section_scores[section.name] = 0.5  # Default mid-level score

        return section_scores

    async def _analyze_job_requirements(
        self,
        raw_jd: str,
        skills: List[str],
        responsibilities: List[str],
        experience_level: str,
    ) -> Dict[str, Any]:
        """
        Analyzes job requirements in depth using the LLM.

        Args:
            raw_jd: The raw job description text
            skills: Extracted skills list
            responsibilities: Extracted responsibilities list
            experience_level: The experience level required

        Returns:
            Dictionary with detailed job requirements analysis
        """
        try:
            # Load prompt template from external file
            try:
                prompt_path = self.settings.get_prompt_path(
                    "job_research_analysis_prompt"
                )
                with open(prompt_path, "r", encoding="utf-8") as f:
                    prompt_template = f.read()
                logger.info("Successfully loaded job research analysis prompt template")
            except Exception as e:
                logger.error(
                    f"Error loading job research analysis prompt template: {e}"
                )
                # Fallback to basic prompt
                prompt_template = """
                Analyze the job description: {raw_jd}
                Skills context: {skills}
                Responsibilities: {responsibilities}
                Experience level: {experience_level}
                Return analysis as JSON.
                """

            # Format the prompt with actual data
            prompt = prompt_template.format(
                raw_jd=raw_jd,
                skills=", ".join(skills),
                responsibilities=", ".join(responsibilities),
                experience_level=experience_level,
            )

            # Use centralized JSON generation and parsing
            try:
                analysis = await self._generate_and_parse_json(
                    prompt=prompt,
                    session_id=getattr(self, "current_session_id", None),
                    trace_id=getattr(self, "current_trace_id", None),
                )
                return analysis
            except Exception as e:
                logger.error(f"Failed to analyze job description with LLM: {str(e)}")
                return {
                    "core_technical_skills": skills,
                    "soft_skills": [],
                    "error": "LLM analysis failed",
                    "raw_analysis": str(e),
                }
        except Exception as e:
            logger.error(f"Error analyzing job requirements: {str(e)}")
            return {}

    async def _research_company_info(self, raw_jd: str) -> Dict[str, Any]:
        """
        Simulates research about the company mentioned in the job description.

        Args:
            raw_jd: The raw job description text

        Returns:
            Dictionary with company information
        """
        # Extract company name with the LLM
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

            # Use centralized JSON generation and parsing
            try:
                company_info = await self._generate_and_parse_json(
                    prompt=company_prompt,
                    session_id=getattr(self, "current_session_id", None),
                    trace_id=getattr(self, "current_trace_id", None),
                )

                # For MVP, simulate the rest of the company research
                company_name = company_info.get("company_name", "Unknown Company")
                company_info["description"] = f"Research insights about {company_name}."
                company_info["key_products"] = ["Product A", "Product B"]
                company_info["market_position"] = "Market leader in their segment"

                return company_info
            except Exception as e:
                logger.error(f"Company research LLM request failed: {str(e)}")
                # Return fallback company info
                return {
                    "company_name": "Unknown Company",
                    "industry": "Technology",
                    "values": ["Innovation", "Teamwork"],
                    "error": "LLM company research failed",
                }
        except Exception as e:
            logger.error(f"Error researching company info: {str(e)}")

        # Fallback basic info
        return {
            "company_name": "Unknown Company",
            "industry": "Technology",
            "values": ["Innovation", "Teamwork"],
        }

    def _research_industry_insights(self, terms: List[str]) -> Dict[str, Any]:
        """
        Simulates research about industry trends related to the job.

        Args:
            terms: Industry terms or skills to research

        Returns:
            Dictionary with industry insights
        """
        # For MVP, return simulated insights
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

    @with_node_error_handling
    async def run_as_node(self, state: AgentState) -> AgentState:
        """Run the agent as a LangGraph node."""
        logger.info("Research Agent: Starting node execution")
        
        # Create input for the agent
        agent_input = AgentIO(
            structured_cv=state.structured_cv,
            job_description=state.job_description,
            metadata={"agent_name": self.name}
        )
        
        # Run the agent
        result = await self.run_async(agent_input)
        
        if result.success:
            # Update state with research findings
            if result.data and "research_findings" in result.data:
                state.research_findings = result.data["research_findings"]
                logger.info("Research Agent: Successfully completed research")
            else:
                error_msg = "Research Agent: No research findings in result"
                logger.error(error_msg)
                state.error_messages.append(error_msg)
        else:
            error_msg = f"Research Agent failed: {result.error_message}"
            logger.error(error_msg)
            state.error_messages.append(error_msg)
            
        return state
