from src.agents.agent_base import AgentBase
from src.services.llm import LLM
from src.core.state_manager import (
    JobDescriptionData,
    AgentIO,
    StructuredCV,
    Section,
    Subsection,
    Item,
)
from src.services.vector_db import VectorDB
from src.config.logging_config import get_logger
from src.config.settings import get_config
from src.services.llm import LLMResponse
from typing import Dict, Any, List, Optional
import time
import logging
import json
import os
import asyncio
from src.orchestration.state import AgentState

# Set up logging
logger = logging.getLogger(__name__)


class ResearchAgent(AgentBase):
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
        llm: LLM,
        vector_db: Optional[VectorDB] = None,
    ):
        """
        Initializes the ResearchAgent.

        Args:
            name: The name of the agent.
            description: A description of the agent.
            llm: The LLM instance for analysis.
            vector_db: Vector database instance for storing and searching embeddings.
        """
        super().__init__(
            name=name,
            description=description,
            input_schema=AgentIO(
                input={
                    "job_description_data": Dict[str, Any],  # Takes parsed job description data
                    "structured_cv": StructuredCV,  # The current StructuredCV
                },
                output=Dict[str, Any],  # Outputs research results and relevance matches
                description="Conducts research on job information and finds relevant CV content.",
            ),
            output_schema=AgentIO(
                input={
                    "job_description_data": Dict[str, Any],
                    "structured_cv": StructuredCV,
                },
                output=Dict[str, Any],
                description="Relevant research findings and content matches for tailoring.",
            ),
        )
        self.llm = llm
        self.vector_db = vector_db
        
        # Initialize settings for prompt loading
        self.settings = get_config()

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conducts research based on the job description and finds relevant content from the CV.

        Args:
            input_data: A dictionary containing:
                - 'job_description_data' (Dict): Parsed job description data
                - 'structured_cv' (StructuredCV): The current StructuredCV

        Returns:
            A dictionary containing research results and relevance matches.
        """
        job_description_data = input_data.get("job_description_data", {})
        structured_cv = input_data.get("structured_cv")

        # Check if input data is valid
        if not job_description_data or not structured_cv:
            logger.warning("Missing required input data for ResearchAgent")
            return {}

        # Extract relevant information for research queries
        skills = self._extract_field(job_description_data, "skills", [])
        responsibilities = self._extract_field(job_description_data, "responsibilities", [])
        industry_terms = self._extract_field(job_description_data, "industry_terms", [])
        company_values = self._extract_field(job_description_data, "company_values", [])
        experience_level = self._extract_field(job_description_data, "experience_level", "")
        raw_jd = self._extract_field(job_description_data, "raw_text", "")

        logger.info(
            f"Running ResearchAgent for job with {len(skills)} skills, {len(responsibilities)} responsibilities"
        )

        # --- STEP 1: Build vector database from CV content if not already populated ---
        if self.vector_db:
            self._index_cv_content(structured_cv)

        # --- STEP 2: Perform research and analysis ---

        # Initialize the research results dictionary
        research_results = {
            "relevance_scores": {},
            "key_matches": {},
            "company_info": {},
            "industry_insights": {},
            "job_requirements_analysis": {},
        }

        # Analyze the job description with the LLM for deeper insights
        requirements_analysis = self._analyze_job_requirements(
            raw_jd, skills, responsibilities, experience_level
        )
        if requirements_analysis:
            research_results["job_requirements_analysis"] = requirements_analysis

        # Find most relevant content from the CV using vector search (if available)
        if self.vector_db:
            # Search by skills
            if skills:
                skill_query = " ".join(skills)
                relevant_skill_content = self._search_relevant_content(
                    skill_query, structured_cv, num_results=5
                )
                research_results["key_matches"]["skills"] = relevant_skill_content

            # Search by responsibilities
            if responsibilities:
                resp_query = " ".join(responsibilities)
                relevant_resp_content = self._search_relevant_content(
                    resp_query, structured_cv, num_results=5
                )
                research_results["key_matches"]["responsibilities"] = relevant_resp_content

            # Calculate overall relevance score for sections
            research_results["relevance_scores"] = self._calculate_section_relevance(
                structured_cv, skills + responsibilities + industry_terms
            )

        # Gather company information (simulated for MVP)
        if raw_jd:
            company_info = self._research_company_info(raw_jd)
            research_results["company_info"] = company_info

        # Gather industry insights (simulated for MVP)
        if industry_terms or skills:
            industry_insights = self._research_industry_insights(
                industry_terms if industry_terms else skills[:3]
            )
            research_results["industry_insights"] = industry_insights

        logger.info(
            f"ResearchAgent completed analysis with {len(research_results['key_matches'])} match categories"
        )

        self._latest_research_results = research_results
        return research_results
    
    async def run_async(self, input_data: Any, context: 'AgentExecutionContext') -> 'AgentResult':
        """Async run method for consistency with enhanced agent interface."""
        from .agent_base import AgentResult
        from src.models.validation_schemas import validate_agent_input, ValidationError
        
        try:
            # Validate input data using Pydantic schemas
            try:
                validated_input = validate_agent_input('research', input_data)
                # Convert validated Pydantic model back to dict for processing
                input_data = validated_input.model_dump()
                logger.info("Input validation passed for ResearchAgent")
            except ValidationError as ve:
                logger.error(f"Input validation failed for ResearchAgent: {ve.message}")
                fallback_result = {
                    "research_results": {
                        "error": f"Input validation failed: {ve.message}",
                        "company_info": {},
                        "industry_trends": [],
                        "role_insights": {},
                        "skill_requirements": [],
                        "market_data": {}
                    },
                    "enhanced_job_description": None
                }
                return AgentResult(
                    success=False,
                    output_data=fallback_result,
                    confidence_score=0.0,
                    error_message=f"Input validation failed: {ve.message}",
                    metadata={"agent_type": "research", "validation_error": True}
                )
            except Exception as e:
                logger.error(f"Input validation error for ResearchAgent: {str(e)}")
                fallback_result = {
                    "research_results": {
                        "error": f"Input validation error: {str(e)}",
                        "company_info": {},
                        "industry_trends": [],
                        "role_insights": {},
                        "skill_requirements": [],
                        "market_data": {}
                    },
                    "enhanced_job_description": None
                }
                return AgentResult(
                    success=False,
                    output_data=fallback_result,
                    confidence_score=0.0,
                    error_message=f"Input validation error: {str(e)}",
                    metadata={"agent_type": "research", "validation_error": True}
                )
            
            # Use the existing run method for the actual processing
            result = self.run(input_data)
            
            return AgentResult(
                success=True,
                output_data=result,
                confidence_score=1.0,
                metadata={"agent_type": "research"}
            )
            
        except Exception as e:
            logger.error(f"ResearchAgent error: {str(e)}")
            
            # Return error result with fallback empty results
            fallback_result = {
                "research_results": {
                    "error": str(e),
                    "company_info": {},
                    "industry_trends": [],
                    "role_insights": {},
                    "skill_requirements": [],
                    "market_data": {}
                },
                "enhanced_job_description": input_data.get("job_description") if isinstance(input_data, dict) else None
            }
            
            return AgentResult(
                success=False,
                output_data=fallback_result,
                confidence_score=0.0,
                error_message=str(e),
                metadata={"agent_type": "research"}
            )

    def get_research_results(self) -> Dict[str, Any]:
        """
        Returns the previously generated research results.

        Returns:
            The research results as a dictionary, or an empty dict if no research has been performed
        """
        if hasattr(self, "_latest_research_results") and self._latest_research_results:
            return self._latest_research_results
        return {}

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
        if hasattr(self.vector_db, "indexed_cv_ids") and cv_id in self.vector_db.indexed_cv_ids:
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
                            self.vector_db.add_item(item, item.content, metadata=metadata)
                        except Exception as e:
                            logger.error(f"Error indexing subsection item {item.id}: {str(e)}")

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
                section_scores[section.name] = max(0.1, min(1.0, score / (len(terms) * 0.7)))
            else:
                section_scores[section.name] = 0.5  # Default mid-level score

        return section_scores

    def _analyze_job_requirements(
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
                prompt_path = self.settings.get_prompt_path("job_research_analysis_prompt")
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt_template = f.read()
                logger.info("Successfully loaded job research analysis prompt template")
            except Exception as e:
                logger.error(f"Error loading job research analysis prompt template: {e}")
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
                skills=', '.join(skills),
                responsibilities=', '.join(responsibilities),
                experience_level=experience_level
            )

            response = self.llm.generate_content(prompt)

            # Extract JSON from response
            try:
                # Find the first JSON-like structure in the response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    analysis = json.loads(json_str)
                    return analysis

                # If no JSON found, try to create basic structure from the text
                return {
                    "core_technical_skills": skills,
                    "soft_skills": [
                        s
                        for s in skills
                        if s.lower() in ["communication", "leadership", "teamwork"]
                    ],
                    "key_performance_metrics": [],
                    "project_types": [],
                    "working_environment": {},
                }
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM output as JSON")
                return {
                    "core_technical_skills": skills,
                    "soft_skills": [],
                    "raw_analysis": response,
                }
        except Exception as e:
            logger.error(f"Error analyzing job requirements: {str(e)}")
            return {}

    def _research_company_info(self, raw_jd: str) -> Dict[str, Any]:
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

            response = self.llm.generate_content(company_prompt)

            # Extract JSON from response
            try:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    company_info = json.loads(json_str)

                    # For MVP, simulate the rest of the company research
                    company_name = company_info.get("company_name", "Unknown Company")
                    company_info["description"] = f"Research insights about {company_name}."
                    company_info["key_products"] = ["Product A", "Product B"]
                    company_info["market_position"] = "Market leader in their segment"

                    return company_info
            except json.JSONDecodeError:
                pass
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
    
    async def run_as_node(self, state: AgentState) -> dict:
        """
        Executes the research logic as a LangGraph node.
        
        Args:
            state: The current state of the workflow.
            
        Returns:
            A dictionary containing research findings.
        """
        logger.info("ResearchAgent node running.")
        cv = state.structured_cv
        job_data = state.job_description_data
        
        if not cv or not job_data:
            logger.warning("Research agent called without required CV or job data.")
            return {}
        
        try:
            # Create execution context for the async method
            context = AgentExecutionContext(
                session_id="langraph_session",
                input_data={
                    "structured_cv": cv.model_dump(),
                    "job_description_data": job_data.model_dump()
                }
            )
            
            # Call the existing async method
            result = await self.run_async(None, context)
            
            if result.success:
                # Extract research findings
                research_data = result.output_data.get("research_findings", {})
                
                # Store research findings in state
                current_findings = state.research_findings or {}
                current_findings.update(research_data)
                
                return {"research_findings": current_findings}
            
            # If not successful, add error to state
            error_list = state.error_messages or []
            error_list.append(f"Research Error: {result.error_message}")
            return {"error_messages": error_list}
            
        except Exception as e:
            logger.error(f"Error in Research node: {e}", exc_info=True)
            error_list = state.error_messages or []
            error_list.append(f"Research Error: {e}")
            return {"error_messages": error_list}
