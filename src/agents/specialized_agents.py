"""Specialized agents for different CV content types with Phase 1 infrastructure."""

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
                confidence_score=0.8,
                metadata={"analysis_type": "cv_job_match"},
            )

        except Exception as e:
            logger.error(
                "CV analysis failed", session_id=context.session_id, error=str(e)
            )
            return AgentResult(
                success=False,
                output_data={},
                error_message=str(e),
                confidence_score=0.0,
            )

    async def _analyze_cv_job_match(
        self,
        cv_data: Dict[str, Any],
        job_description: Any,
        context: AgentExecutionContext,
    ) -> Dict[str, Any]:
        """Analyze how well CV matches job requirements."""

        # Handle job_description being either a string or dict
        if isinstance(job_description, str):
            job_title = "Unknown"
            job_requirements = job_description[:1000]
        elif isinstance(job_description, dict):
            job_title = job_description.get("title", "Unknown")
            job_requirements = job_description.get("raw_text", str(job_description))[
                :1000
            ]
        else:
            job_title = "Unknown"
            job_requirements = str(job_description)[:1000]

        # Load prompt template from external file
        try:
            prompt_path = self.settings.get_prompt_path("cv_assessment_prompt")
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_template = f.read()
            logger.info("Successfully loaded CV assessment prompt template")
        except Exception as e:
            logger.error(f"Error loading CV assessment prompt template: {e}")
            # Fallback to basic prompt
            prompt_template = """
            Analyze CV against job requirements.
            Job: {job_title}
            Requirements: {job_requirements}
            CV Summary: {executive_summary}
            Return JSON analysis.
            """

        # Format the prompt with actual data
        prompt = prompt_template.format(
            job_title=job_title,
            job_requirements=job_requirements,
            executive_summary=cv_data.get("executive_summary", ""),
            experience=json.dumps(cv_data.get("experience", []), indent=2)[:1000],
            qualifications=json.dumps(cv_data.get("qualifications", []), indent=2)[
                :500
            ],
        )

        response = await self.llm_service.generate_content(
            prompt=prompt,
            content_type=ContentType.QUALIFICATION,
            session_id=context.session_id,
            item_id=context.item_id,
        )

        # Check if LLM response was successful
        if not response.success:
            logger.error(f"CV analysis LLM request failed: {response.error_message}")
            return {
                "skill_gaps": [],
                "strengths": [],
                "experience_relevance": 0.3,
                "keyword_match": 0.3,
                "overall_assessment": "Analysis failed due to LLM error",
                "error": response.error_message,
            }

        try:
            # Try to parse JSON response
            analysis = json.loads(response.content)
            return analysis
        except json.JSONDecodeError:
            # Fallback to structured text parsing
            return self._parse_analysis_text(response.content)

        return {
            "skill_gaps": [],
            "strengths": [],
            "experience_relevance": 0.5,
            "keyword_match": 0.5,
            "overall_assessment": "Analysis unavailable",
        }

    async def _generate_recommendations(
        self, analysis: Dict[str, Any], context: AgentExecutionContext
    ) -> List[str]:
        """Generate improvement recommendations based on analysis."""

        recommendations = []

        # Add skill gap recommendations
        skill_gaps = analysis.get("skill_gaps", [])
        if skill_gaps:
            recommendations.append(
                f"Consider developing these skills: {', '.join(skill_gaps[:3])}"
            )

        # Add experience recommendations
        experience_score = analysis.get("experience_relevance", 0)
        if experience_score < 0.7:
            recommendations.append("Highlight more relevant work experience")

        # Add keyword optimization
        keyword_score = analysis.get("keyword_match", 0)
        if keyword_score < 0.6:
            recommendations.append("Optimize CV with more job-relevant keywords")

        return recommendations

    def _calculate_match_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall match score."""
        exp_relevance = analysis.get("experience_relevance", 0.5)
        keyword_match = analysis.get("keyword_match", 0.5)

        # Weight the scores
        match_score = (exp_relevance * 0.6) + (keyword_match * 0.4)

        # Adjust based on skill gaps
        skill_gaps = len(analysis.get("skill_gaps", []))
        if skill_gaps > 3:
            match_score *= 0.8

        return round(match_score, 2)

    def _parse_analysis_text(self, text: str) -> Dict[str, Any]:
        """Parse analysis from unstructured text."""
        # Simple text parsing fallback
        return {
            "skill_gaps": [],
            "strengths": [],
            "experience_relevance": 0.6,
            "keyword_match": 0.6,
            "overall_assessment": text[:200] + "..." if len(text) > 200 else text,
        }

    async def run_as_node(self, state: "AgentState") -> Dict[str, Any]:
        """Execute CV analysis as a LangGraph node."""
        from ..orchestration.state import AgentState

        try:
            # Extract data from state
            cv_data = state.structured_cv.model_dump() if state.structured_cv else {}
            job_data = (
                state.job_description_data.model_dump()
                if state.job_description_data
                else {}
            )

            # Create execution context
            context = AgentExecutionContext(
                session_id=getattr(state, "session_id", "default")
            )

            # Run analysis
            result = await self.run_async(
                {"cv_data": cv_data, "job_description": job_data}, context
            )

            # Return state updates
            return {
                "cv_analysis_results": result.output_data if result.success else {},
                "processing_status": (
                    ProcessingStatus.COMPLETED
                    if result.success
                    else ProcessingStatus.FAILED
                ),
            }

        except Exception as e:
            self.logger.error(f"CV analysis node failed: {str(e)}")
            return {
                "cv_analysis_results": {},
                "processing_status": ProcessingStatus.FAILED,
            }


class ContentOptimizationAgent(EnhancedAgentBase):
    """Agent specialized in optimizing existing CV content."""

    def __init__(self):
        super().__init__(
            name="ContentOptimizationAgent",
            description="Optimizes existing CV content for better impact and relevance",
            input_schema={
                "content_items": List[Dict[str, Any]],
                "job_requirements": Dict[str, Any],
                "optimization_goals": List[str],
            },
            output_schema={
                "optimized_content": List[Dict[str, Any]],
                "optimization_summary": Dict[str, Any],
            },
        )
        self.llm_service = get_llm_service()

        # Initialize settings for prompt loading
        self.settings = get_config()
        self.content_writers = {
            content_type: create_content_writer(content_type)
            for content_type in ContentType
        }

    async def run_async(
        self, input_data: Any, context: AgentExecutionContext
    ) -> AgentResult:
        """Optimize content items for better performance."""
        from ..models.validation_schemas import validate_agent_input

        try:
            # Validate input data
            if not validate_agent_input(input_data, dict):
                logger.error("Input validation failed for ContentOptimizationAgent")
                return AgentResult(
                    success=False,
                    output_data={
                        "error": "Input validation failed: expected dictionary input"
                    },
                    confidence_score=0.0,
                    error_message="Input validation failed: expected dictionary input",
                    metadata={"optimization_type": "content", "validation_error": True},
                )

            content_items = input_data.get("content_items", [])
            job_requirements = input_data.get("job_requirements", {})
            optimization_goals = input_data.get("optimization_goals", [])

            optimized_items = []
            optimization_stats = {
                "items_processed": 0,
                "items_improved": 0,
                "average_improvement": 0.0,
            }

            for item in content_items:
                optimized_item = await self._optimize_content_item(
                    item, job_requirements, optimization_goals, context
                )
                optimized_items.append(optimized_item)

                optimization_stats["items_processed"] += 1
                if optimized_item.get("improved", False):
                    optimization_stats["items_improved"] += 1

            # Calculate average improvement
            if optimization_stats["items_processed"] > 0:
                optimization_stats["average_improvement"] = (
                    optimization_stats["items_improved"]
                    / optimization_stats["items_processed"]
                )

            result_data = {
                "optimized_content": optimized_items,
                "optimization_summary": optimization_stats,
            }

            return AgentResult(
                success=True,
                output_data=result_data,
                confidence_score=0.85,
                metadata=optimization_stats,
            )

        except Exception as e:
            logger.error(
                "Content optimization failed",
                session_id=context.session_id,
                error=str(e),
            )
            return AgentResult(
                success=False,
                output_data={},
                error_message=str(e),
                confidence_score=0.0,
            )

    async def _optimize_content_item(
        self,
        item: Dict[str, Any],
        job_requirements: Dict[str, Any],
        optimization_goals: List[str],
        context: AgentExecutionContext,
    ) -> Dict[str, Any]:
        """Optimize a single content item."""

        # Determine content type
        content_type = self._determine_item_content_type(item)

        # Get appropriate content writer
        writer = self.content_writers.get(content_type)
        if not writer:
            return {**item, "improved": False, "reason": "No suitable writer found"}

        # Prepare input for content writer
        writer_input = {
            "job_description_data": job_requirements,
            "content_item": item,
            "context": {
                "optimization_goals": optimization_goals,
                "additional_context": "Optimize for better impact and relevance",
            },
        }

        # Create context for the writer
        writer_context = AgentExecutionContext(
            session_id=context.session_id,
            item_id=f"{context.item_id}_opt_{item.get('id', 'unknown')}",
            content_type=content_type,
        )

        # Generate optimized content
        result = await writer.run_async(writer_input, writer_context)

        if result.success:
            optimized_item = {
                **item,
                "content": result.output_data.get("content", item.get("content", "")),
                "improved": True,
                "confidence_score": result.confidence_score,
                "optimization_metadata": result.metadata,
            }
        else:
            optimized_item = {
                **item,
                "improved": False,
                "reason": result.error_message or "Optimization failed",
            }

        return optimized_item

    def _determine_item_content_type(self, item: Dict[str, Any]) -> ContentType:
        """Determine content type from item structure."""

        if "position" in item or "company" in item:
            return ContentType.EXPERIENCE
        elif "name" in item and ("technologies" in item or "tech_stack" in item):
            return ContentType.PROJECT
        elif "summary" in item or item.get("type") == "executive_summary":
            return ContentType.EXECUTIVE_SUMMARY
        else:
            return ContentType.QUALIFICATION

    async def run_as_node(self, state: "AgentState") -> Dict[str, Any]:
        """Execute content optimization as a LangGraph node."""
        from ..orchestration.state import AgentState

        try:
            # Extract content items from state
            content_items = []
            if state.structured_cv:
                cv_dict = state.structured_cv.model_dump()
                # Extract various content types
                content_items.extend(cv_dict.get("experience", []))
                content_items.extend(cv_dict.get("projects", []))
                content_items.extend(cv_dict.get("qualifications", []))
                if cv_dict.get("executive_summary"):
                    content_items.append(
                        {
                            "type": "executive_summary",
                            "content": cv_dict["executive_summary"],
                        }
                    )

            job_requirements = (
                state.job_description_data.model_dump()
                if state.job_description_data
                else {}
            )

            # Create execution context
            context = AgentExecutionContext(
                session_id=getattr(state, "session_id", "default")
            )

            # Run optimization
            result = await self.run_async(
                {
                    "content_items": content_items,
                    "job_requirements": job_requirements,
                    "optimization_goals": ["relevance", "impact", "clarity"],
                },
                context,
            )

            # Return state updates
            return {
                "optimized_content": result.output_data if result.success else {},
                "processing_status": (
                    ProcessingStatus.COMPLETED
                    if result.success
                    else ProcessingStatus.FAILED
                ),
            }

        except Exception as e:
            self.logger.error(f"Content optimization node failed: {str(e)}")
            return {
                "optimized_content": {},
                "processing_status": ProcessingStatus.FAILED,
            }


# QualityAssuranceAgent removed - using the one from quality_assurance_agent.py instead


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


def create_content_optimization_agent() -> ContentOptimizationAgent:
    """Create a content optimization agent."""
    return ContentOptimizationAgent()


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
    """Enhanced wrapper for ParserAgent that integrates with the new infrastructure."""

    def __init__(self):
        super().__init__(
            name="EnhancedParserAgent",
            description="Enhanced agent for parsing CVs and job descriptions into structured formats",
            input_schema={
                "cv_text": str,
                "job_description": str,
                "start_from_scratch": bool,
            },
            output_schema={
                "structured_cv": Dict[str, Any],
                "job_description_data": Dict[str, Any],
            },
        )
        # Initialize the original parser agent
        llm_service = get_llm_service()
        self.parser_agent = ParserAgent(
            name="ParserAgent",
            description="Parses CVs and job descriptions",
            llm=llm_service,
        )

    async def run_async(
        self, input_data: Any, context: AgentExecutionContext
    ) -> AgentResult:
        """Run the parser agent asynchronously."""
        from ..models.validation_schemas import validate_agent_input

        try:
            # Validate input data
            if not validate_agent_input(input_data):
                logger.error("Input validation failed for EnhancedParserAgent")
                return AgentResult(
                    success=False,
                    output_data={"error": "Input validation failed"},
                    confidence_score=0.0,
                    error_message="Input validation failed",
                    metadata={
                        "agent_type": "enhanced_parser",
                        "validation_error": True,
                    },
                )

            # Convert input data to the format expected by ParserAgent
            parser_input = {}

            if isinstance(input_data, dict):
                # Handle structured input
                if "cv_text" in input_data:
                    parser_input["cv_text"] = input_data["cv_text"]
                if "job_description" in input_data:
                    parser_input["job_description"] = input_data["job_description"]
                if "start_from_scratch" in input_data:
                    parser_input["start_from_scratch"] = input_data[
                        "start_from_scratch"
                    ]
            elif isinstance(input_data, str):
                # Handle string input as CV text
                parser_input["cv_text"] = input_data

            # Use run_as_node for LangGraph integration
            # Create AgentState for run_as_node compatibility
            from ..orchestration.state import AgentState
            from ..models.data_models import StructuredCV, JobDescriptionData

            # Create proper StructuredCV and JobDescriptionData objects
            structured_cv = parser_input.get("structured_cv") or StructuredCV()
            job_desc_data = parser_input.get("job_description_data")
            if not job_desc_data or isinstance(job_desc_data, dict):
                job_desc_data = JobDescriptionData(
                    raw_text=parser_input.get("job_description", "")
                )

            agent_state = AgentState(
                structured_cv=structured_cv, job_description_data=job_desc_data
            )

            node_result = await self.parser_agent.run_as_node(agent_state)
            result = node_result.get("output_data", {})

            return AgentResult(
                success=True,
                output_data=result,
                confidence_score=1.0,
                metadata={"agent_type": "enhanced_parser"},
            )

        except Exception as e:
            self.logger.error(f"Enhanced parser agent failed: {str(e)}")
            return AgentResult(
                success=False,
                output_data={},
                confidence_score=0.0,
                error_message=str(e),
                metadata={"agent_type": "enhanced_parser"},
            )

    async def run_as_node(self, state: "AgentState") -> Dict[str, Any]:
        """Execute enhanced parser as a LangGraph node."""
        try:
            # Delegate to the wrapped parser agent's run_as_node method
            return await self.parser_agent.run_as_node(state)

        except Exception as e:
            self.logger.error(f"Enhanced parser node failed: {str(e)}")
            return {
                "structured_cv": state.structured_cv,
                "job_description_data": state.job_description_data,
                "processing_status": ProcessingStatus.FAILED,
            }


# Agent registry for easy access
AGENT_REGISTRY = {
    "cv_analysis": create_cv_analysis_agent,
    "cv_parser": create_enhanced_parser_agent,
    "content_optimization": create_content_optimization_agent,
    "quality_assurance": create_quality_assurance_agent,
    "qualification_writer": lambda: create_content_writer(ContentType.QUALIFICATION),
    "experience_writer": lambda: create_content_writer(ContentType.EXPERIENCE),
    "project_writer": lambda: create_content_writer(ContentType.PROJECT),
    "executive_summary_writer": lambda: create_content_writer(
        ContentType.EXECUTIVE_SUMMARY
    ),
}


def get_agent(agent_type: str) -> Optional[EnhancedAgentBase]:
    """Get an agent instance by type."""
    factory = AGENT_REGISTRY.get(agent_type)
    if factory:
        return factory()
    return None


def list_available_agents() -> List[str]:
    """List all available agent types."""
    return list(AGENT_REGISTRY.keys())
