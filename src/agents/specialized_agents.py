"""Specialized agents for different CV content types with Phase 1 infrastructure."""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from .enhanced_content_writer import EnhancedContentWriterAgent, create_content_writer
from .agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from ..models.data_models import ContentType, ProcessingStatus
from ..config.logging_config import get_structured_logger
from ..config.settings import get_config
from ..services.llm import get_llm_service

logger = get_structured_logger("specialized_agents")


class CVAnalysisAgent(EnhancedAgentBase):
    """Agent specialized in analyzing CV content and job requirements."""
    
    def __init__(self):
        super().__init__(
            name="CVAnalysisAgent",
            description="Analyzes CV content and job requirements to provide optimization recommendations",
            input_schema={
                "cv_data": Dict[str, Any],
                "job_description": Dict[str, Any]
            },
            output_schema={
                "analysis_results": Dict[str, Any],
                "recommendations": List[str],
                "match_score": float
            }
        )
        self.llm_service = get_llm_service()
        
        # Initialize settings for prompt loading
        self.settings = get_config()
    
    def run(self, input_data: Any) -> Any:
        """Synchronous wrapper for run_async method."""
        import asyncio
        
        # Create a basic context if not provided
        context = AgentExecutionContext(
            session_id="sync_session",
            item_id="sync_item",
            content_type=None
        )
        
        # Run the async method
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.run_async(input_data, context))
    
    async def run_async(self, input_data: Any, context: AgentExecutionContext) -> AgentResult:
        """Analyze CV content against job requirements."""
        try:
            # Debug logging to understand the input_data type and content
            self.logger.info(f"CVAnalysisAgent received input_data type: {type(input_data)}")
            self.logger.info(f"CVAnalysisAgent received input_data: {str(input_data)[:500]}")
            
            # Ensure input_data is a dictionary
            if not isinstance(input_data, dict):
                self.logger.warning(f"Converting non-dict input_data from {type(input_data)} to empty dict")
                input_data = {}
            
            cv_data = input_data.get("cv_data", {})
            job_description = input_data.get("job_description", {})
            
            self.logger.info(f"Extracted cv_data type: {type(cv_data)}, job_description type: {type(job_description)}")
            
            # Perform analysis
            analysis = await self._analyze_cv_job_match(cv_data, job_description, context)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(analysis, context)
            
            # Calculate match score
            match_score = self._calculate_match_score(analysis)
            
            result_data = {
                "analysis_results": analysis,
                "recommendations": recommendations,
                "match_score": match_score,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            return AgentResult(
                success=True,
                output_data=result_data,
                confidence_score=0.8,
                metadata={"analysis_type": "cv_job_match"}
            )
            
        except Exception as e:
            logger.error(
                "CV analysis failed",
                session_id=context.session_id,
                error=str(e)
            )
            return AgentResult(
                success=False,
                output_data={},
                error_message=str(e),
                confidence_score=0.0
            )
    
    def run(self, input_data: Any) -> Any:
        """Legacy synchronous run method for backward compatibility."""
        import asyncio
        from .agent_base import AgentExecutionContext
        
        # Create a basic execution context
        context = AgentExecutionContext(
            session_id="sync_execution",
            user_id="system",
            request_id="sync_request"
        )
        
        # Run the async method synchronously
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(self.run_async(input_data, context))
        
        if result.success:
            return result.output_data
        else:
            raise Exception(result.error_message or "CV analysis failed")
    
    async def _analyze_cv_job_match(
        self, 
        cv_data: Dict[str, Any], 
        job_description: Any,
        context: AgentExecutionContext
    ) -> Dict[str, Any]:
        """Analyze how well CV matches job requirements."""
        
        # Handle job_description being either a string or dict
        if isinstance(job_description, str):
            job_title = "Unknown"
            job_requirements = job_description[:1000]
        elif isinstance(job_description, dict):
            job_title = job_description.get('title', 'Unknown')
            job_requirements = job_description.get('raw_text', str(job_description))[:1000]
        else:
            job_title = "Unknown"
            job_requirements = str(job_description)[:1000]
        
        # Load prompt template from external file
        try:
            prompt_path = self.settings.get_prompt_path("cv_assessment_prompt")
            with open(prompt_path, 'r', encoding='utf-8') as f:
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
            executive_summary=cv_data.get('executive_summary', ''),
            experience=json.dumps(cv_data.get('experience', []), indent=2)[:1000],
            qualifications=json.dumps(cv_data.get('qualifications', []), indent=2)[:500]
        )
        
        response = await self.llm_service.generate_content(
            prompt=prompt,
            content_type=ContentType.QUALIFICATION,
            session_id=context.session_id,
            item_id=context.item_id
        )
        
        if response.success:
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
            "overall_assessment": "Analysis unavailable"
        }
    
    async def _generate_recommendations(
        self, 
        analysis: Dict[str, Any], 
        context: AgentExecutionContext
    ) -> List[str]:
        """Generate improvement recommendations based on analysis."""
        
        recommendations = []
        
        # Add skill gap recommendations
        skill_gaps = analysis.get('skill_gaps', [])
        if skill_gaps:
            recommendations.append(f"Consider developing these skills: {', '.join(skill_gaps[:3])}")
        
        # Add experience recommendations
        experience_score = analysis.get('experience_relevance', 0)
        if experience_score < 0.7:
            recommendations.append("Highlight more relevant work experience")
        
        # Add keyword optimization
        keyword_score = analysis.get('keyword_match', 0)
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
            "overall_assessment": text[:200] + "..." if len(text) > 200 else text
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
                "optimization_goals": List[str]
            },
            output_schema={
                "optimized_content": List[Dict[str, Any]],
                "optimization_summary": Dict[str, Any]
            }
        )
        self.llm_service = get_llm_service()
        
        # Initialize settings for prompt loading
        self.settings = get_config()
        self.content_writers = {
            content_type: create_content_writer(content_type)
            for content_type in ContentType
        }
    
    def run(self, input_data: Any) -> Any:
        """Synchronous wrapper for run_async method."""
        import asyncio
        
        # Create a basic context if not provided
        context = AgentExecutionContext(
            session_id="sync_session",
            item_id="sync_item",
            content_type=None
        )
        
        # Run the async method
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.run_async(input_data, context))
    
    async def run_async(self, input_data: Any, context: AgentExecutionContext) -> AgentResult:
        """Optimize content items for better performance."""
        try:
            content_items = input_data.get("content_items", [])
            job_requirements = input_data.get("job_requirements", {})
            optimization_goals = input_data.get("optimization_goals", [])
            
            optimized_items = []
            optimization_stats = {
                "items_processed": 0,
                "items_improved": 0,
                "average_improvement": 0.0
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
                    optimization_stats["items_improved"] / 
                    optimization_stats["items_processed"]
                )
            
            result_data = {
                "optimized_content": optimized_items,
                "optimization_summary": optimization_stats
            }
            
            return AgentResult(
                success=True,
                output_data=result_data,
                confidence_score=0.85,
                metadata=optimization_stats
            )
            
        except Exception as e:
            logger.error(
                "Content optimization failed",
                session_id=context.session_id,
                error=str(e)
            )
            return AgentResult(
                success=False,
                output_data={},
                error_message=str(e),
                confidence_score=0.0
            )
    
    async def _optimize_content_item(
        self,
        item: Dict[str, Any],
        job_requirements: Dict[str, Any],
        optimization_goals: List[str],
        context: AgentExecutionContext
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
                "additional_context": "Optimize for better impact and relevance"
            }
        }
        
        # Create context for the writer
        writer_context = AgentExecutionContext(
            session_id=context.session_id,
            item_id=f"{context.item_id}_opt_{item.get('id', 'unknown')}",
            content_type=content_type
        )
        
        # Generate optimized content
        result = await writer.run_async(writer_input, writer_context)
        
        if result.success:
            optimized_item = {
                **item,
                "content": result.output_data.get("content", item.get("content", "")),
                "improved": True,
                "confidence_score": result.confidence_score,
                "optimization_metadata": result.metadata
            }
        else:
            optimized_item = {
                **item,
                "improved": False,
                "reason": result.error_message or "Optimization failed"
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


class QualityAssuranceAgent(EnhancedAgentBase):
    """Agent specialized in quality assurance for generated content."""
    
    def __init__(self):
        super().__init__(
            name="QualityAssuranceAgent",
            description="Performs quality assurance checks on generated CV content",
            input_schema={
                "content_items": List[Dict[str, Any]],
                "quality_criteria": Dict[str, Any]
            },
            output_schema={
                "quality_report": Dict[str, Any],
                "flagged_items": List[Dict[str, Any]],
                "overall_score": float
            }
        )
    
    def run(self, input_data: Any) -> Any:
        """Synchronous wrapper for run_async method."""
        import asyncio
        
        # Create a basic context if not provided
        context = AgentExecutionContext(
            session_id="sync_session",
            item_id="sync_item",
            content_type=None
        )
        
        # Run the async method
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.run_async(input_data, context))
    
    async def run_async(self, input_data: Any, context: AgentExecutionContext) -> AgentResult:
        """Perform quality assurance on content items."""
        try:
            content_items = input_data.get("content_items", [])
            quality_criteria = input_data.get("quality_criteria", {})
            
            quality_report = {
                "total_items": len(content_items),
                "passed_items": 0,
                "failed_items": 0,
                "warnings": 0,
                "checks_performed": []
            }
            
            flagged_items = []
            item_scores = []
            
            for item in content_items:
                item_qa_result = await self._check_item_quality(
                    item, quality_criteria, context
                )
                
                if item_qa_result["passed"]:
                    quality_report["passed_items"] += 1
                else:
                    quality_report["failed_items"] += 1
                    flagged_items.append({
                        "item": item,
                        "issues": item_qa_result["issues"]
                    })
                
                quality_report["warnings"] += len(item_qa_result.get("warnings", []))
                item_scores.append(item_qa_result["score"])
            
            # Calculate overall score
            overall_score = sum(item_scores) / len(item_scores) if item_scores else 0.0
            
            result_data = {
                "quality_report": quality_report,
                "flagged_items": flagged_items,
                "overall_score": overall_score
            }
            
            return AgentResult(
                success=True,
                output_data=result_data,
                confidence_score=0.9,
                metadata={"qa_timestamp": datetime.now().isoformat()}
            )
            
        except Exception as e:
            logger.error(
                "Quality assurance failed",
                session_id=context.session_id,
                error=str(e)
            )
            return AgentResult(
                success=False,
                output_data={},
                error_message=str(e),
                confidence_score=0.0
            )
    
    async def _check_item_quality(
        self,
        item: Dict[str, Any],
        quality_criteria: Dict[str, Any],
        context: AgentExecutionContext
    ) -> Dict[str, Any]:
        """Check quality of a single content item."""
        
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
            warnings.append(f"Content might be too long (maximum {max_length} characters recommended)")
            score -= 0.1
        
        # Grammar and formatting checks
        if not content.strip():
            issues.append("Empty content")
            score = 0.0
        elif content != content.strip():
            warnings.append("Content has leading/trailing whitespace")
            score -= 0.05
        
        # Professional language check (basic)
        unprofessional_words = ['awesome', 'cool', 'stuff', 'things']
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
            "warnings": warnings
        }


# Factory functions for creating specialized agents
def create_cv_analysis_agent() -> CVAnalysisAgent:
    """Create a CV analysis agent."""
    return CVAnalysisAgent()


def create_content_optimization_agent() -> ContentOptimizationAgent:
    """Create a content optimization agent."""
    return ContentOptimizationAgent()


def create_quality_assurance_agent() -> QualityAssuranceAgent:
    """Create a quality assurance agent."""
    return QualityAssuranceAgent()


# Agent registry for easy access
AGENT_REGISTRY = {
    "cv_analysis": create_cv_analysis_agent,
    "content_optimization": create_content_optimization_agent,
    "quality_assurance": create_quality_assurance_agent,
    "qualification_writer": lambda: create_content_writer(ContentType.QUALIFICATION),
    "experience_writer": lambda: create_content_writer(ContentType.EXPERIENCE),
    "project_writer": lambda: create_content_writer(ContentType.PROJECT),
    "executive_summary_writer": lambda: create_content_writer(ContentType.EXECUTIVE_SUMMARY)
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