"""Enhanced Content Writer Agent with Phase 1 infrastructure integration."""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from .agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from ..services.llm import get_llm_service, LLMResponse
from ..models.data_models import ContentType, ProcessingStatus
from ..config.logging_config import get_structured_logger
from ..core.state_manager import (
    JobDescriptionData,
    ContentData,
    AgentIO,
    ExperienceEntry,
    CVData,
    StructuredCV,
    Section,
    Subsection,
    Item,
    ItemStatus,
    ItemType,
)

logger = get_structured_logger("enhanced_content_writer")


class EnhancedContentWriterAgent(EnhancedAgentBase):
    """Enhanced Content Writer Agent with Phase 1 infrastructure integration."""
    
    def __init__(
        self,
        name: str = "EnhancedContentWriter",
        description: str = "Enhanced agent for generating tailored CV content with advanced error handling and progress tracking",
        content_type: ContentType = ContentType.QUALIFICATION
    ):
        """Initialize the enhanced content writer agent."""
        super().__init__(
            name=name,
            description=description,
            input_schema=AgentIO(
                input={
                    "job_description_data": Dict[str, Any],
                    "content_item": Dict[str, Any],
                    "context": Dict[str, Any]
                },
                output=Dict[str, Any],
                description="Generates enhanced CV content with structured logging and error handling"
            ),
            output_schema=AgentIO(
                input={
                    "job_description_data": Dict[str, Any],
                    "content_item": Dict[str, Any],
                    "context": Dict[str, Any]
                },
                output=Dict[str, Any],
                description="Generated content with metadata and quality metrics"
            ),
            content_type=content_type
        )
        
        # Enhanced services
        self.llm_service = get_llm_service()
        
        # Content generation templates
        self.content_templates = {
            ContentType.QUALIFICATION: self._get_qualification_template(),
            ContentType.EXPERIENCE: self._get_experience_template(),
            ContentType.PROJECT: self._get_project_template(),
            ContentType.EXECUTIVE_SUMMARY: self._get_executive_summary_template()
        }
        
        logger.info(
            "Enhanced Content Writer Agent initialized",
            agent_name=name,
            supported_content_types=[ct.value for ct in self.content_templates.keys()]
        )
    
    async def run_async(self, input_data: Any, context: AgentExecutionContext) -> AgentResult:
        """Enhanced async content generation with structured processing."""
        try:
            # Extract and validate input data
            job_data = input_data.get("job_description_data", {})
            content_item = input_data.get("content_item", {})
            generation_context = input_data.get("context", {})
            
            # Determine content type
            content_type = context.content_type or self._determine_content_type(content_item)
            
            # Log generation start
            self.log_decision(
                f"Starting content generation for {content_type.value}",
                context
            )
            
            # Generate content using LLM service
            generated_content = await self._generate_content_with_llm(
                job_data, content_item, generation_context, content_type, context
            )
            
            # Post-process and validate content
            processed_content = await self._post_process_content(
                generated_content, content_type, context
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(
                processed_content, job_data, content_item
            )
            
            # Prepare result
            result_data = {
                "content": processed_content.content,
                "content_type": content_type.value,
                "confidence_score": confidence,
                "tokens_used": processed_content.tokens_used,
                "processing_time": processed_content.processing_time,
                "metadata": {
                    "job_title": job_data.get("title", "Unknown"),
                    "content_length": len(processed_content.content),
                    "llm_model": processed_content.model_used,
                    "generation_timestamp": datetime.now().isoformat(),
                    **processed_content.metadata
                }
            }
            
            self.log_decision(
                f"Content generation completed successfully for {content_type.value}",
                context
            )
            
            return AgentResult(
                success=True,
                output_data=result_data,
                confidence_score=confidence,
                processing_time=processed_content.processing_time,
                metadata=result_data["metadata"]
            )
            
        except Exception as e:
            logger.error(
                "Content generation failed",
                session_id=context.session_id,
                item_id=context.item_id,
                error=str(e),
                agent_name=self.name
            )
            
            # Return error result with fallback content
            fallback_content = self._generate_fallback_content(
                input_data.get("content_item", {}),
                context.content_type or ContentType.QUALIFICATION
            )
            
            return AgentResult(
                success=False,
                output_data={
                    "content": fallback_content,
                    "content_type": (context.content_type or ContentType.QUALIFICATION).value,
                    "confidence_score": 0.1,
                    "error": str(e),
                    "fallback_used": True
                },
                confidence_score=0.1,
                error_message=str(e),
                metadata={"fallback_used": True, "error_type": type(e).__name__}
            )
    
    def run(self, input_data: Any) -> Any:
        """Legacy synchronous method for backward compatibility."""
        # Create a basic context for legacy calls
        context = AgentExecutionContext(
            session_id="legacy_session",
            item_id="legacy_item",
            content_type=self.content_type
        )
        
        # Run async method synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.run_async(input_data, context))
            return result.output_data
        finally:
            loop.close()
    
    async def _generate_content_with_llm(
        self,
        job_data: Dict[str, Any],
        content_item: Dict[str, Any],
        generation_context: Dict[str, Any],
        content_type: ContentType,
        context: AgentExecutionContext
    ) -> LLMResponse:
        """Generate content using the enhanced LLM service."""
        
        # Build prompt based on content type
        prompt = self._build_prompt(
            job_data, content_item, generation_context, content_type
        )
        
        # Generate content with LLM service
        response = await self.llm_service.generate_content(
            prompt=prompt,
            content_type=content_type,
            session_id=context.session_id,
            item_id=context.item_id,
            max_retries=3
        )
        
        return response
    
    def _build_prompt(
        self,
        job_data: Dict[str, Any],
        content_item: Dict[str, Any],
        generation_context: Dict[str, Any],
        content_type: ContentType
    ) -> str:
        """Build a specialized prompt based on content type."""
        
        template = self.content_templates.get(
            content_type, 
            self.content_templates[ContentType.QUALIFICATION]
        )
        
        # Extract relevant information
        job_title = job_data.get("title", "the position")
        job_description = job_data.get("raw_text", job_data.get("description", ""))
        company_name = job_data.get("company", "the company")
        
        # Content-specific information
        if content_type == ContentType.QUALIFICATION:
            item_title = content_item.get("title", "qualification")
            item_description = content_item.get("description", "")
        elif content_type == ContentType.EXPERIENCE:
            item_title = content_item.get("position", "role")
            item_description = content_item.get("description", "")
            company = content_item.get("company", "previous company")
        elif content_type == ContentType.PROJECT:
            item_title = content_item.get("name", "project")
            item_description = content_item.get("description", "")
        else:  # EXECUTIVE_SUMMARY
            item_title = "Executive Summary"
            item_description = generation_context.get("cv_summary", "")
        
        # Format the prompt
        prompt = template.format(
            job_title=job_title,
            job_description=job_description[:1000],  # Limit length
            company_name=company_name,
            item_title=item_title,
            item_description=item_description,
            additional_context=generation_context.get("additional_context", "")
        )
        
        return prompt
    
    async def _post_process_content(
        self,
        llm_response: LLMResponse,
        content_type: ContentType,
        context: AgentExecutionContext
    ) -> LLMResponse:
        """Post-process generated content for quality and formatting."""
        
        if not llm_response.success:
            return llm_response
        
        content = llm_response.content
        
        # Basic cleaning
        content = content.strip()
        
        # Content-specific post-processing
        if content_type == ContentType.QUALIFICATION:
            content = self._format_qualification_content(content)
        elif content_type == ContentType.EXPERIENCE:
            content = self._format_experience_content(content)
        elif content_type == ContentType.PROJECT:
            content = self._format_project_content(content)
        elif content_type == ContentType.EXECUTIVE_SUMMARY:
            content = self._format_executive_summary_content(content)
        
        # Update the response with processed content
        llm_response.content = content
        llm_response.metadata["post_processed"] = True
        
        return llm_response
    
    def _calculate_confidence_score(
        self,
        llm_response: LLMResponse,
        job_data: Dict[str, Any],
        content_item: Dict[str, Any]
    ) -> float:
        """Calculate confidence score based on content quality metrics."""
        
        if not llm_response.success:
            return 0.0
        
        content = llm_response.content
        base_score = 0.7  # Base confidence
        
        # Length check
        if 50 <= len(content) <= 500:
            base_score += 0.1
        
        # Job relevance (simple keyword matching)
        job_keywords = self._extract_keywords(job_data.get("raw_text", ""))
        content_keywords = self._extract_keywords(content.lower())
        
        if job_keywords and content_keywords:
            overlap = len(job_keywords.intersection(content_keywords))
            relevance_score = min(overlap / len(job_keywords), 0.2)
            base_score += relevance_score
        
        # Processing time penalty (if too fast, might be low quality)
        if llm_response.processing_time > 2.0:
            base_score += 0.05
        
        return min(base_score, 1.0)
    
    def _extract_keywords(self, text: str) -> set:
        """Extract relevant keywords from text."""
        import re
        
        # Simple keyword extraction
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        # Filter out common words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 
            'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use', 'will', 'with'
        }
        
        return {word for word in words if word not in stop_words and len(word) > 3}
    
    def _determine_content_type(self, content_item: Dict[str, Any]) -> ContentType:
        """Determine content type from item data."""
        
        if "position" in content_item or "company" in content_item:
            return ContentType.EXPERIENCE
        elif "name" in content_item and "technologies" in content_item:
            return ContentType.PROJECT
        elif "summary" in content_item or "executive" in str(content_item).lower():
            return ContentType.EXECUTIVE_SUMMARY
        else:
            return ContentType.QUALIFICATION
    
    def _generate_fallback_content(
        self, 
        content_item: Dict[str, Any], 
        content_type: ContentType
    ) -> str:
        """Generate fallback content when LLM fails."""
        
        fallbacks = {
            ContentType.QUALIFICATION: "Strong technical skills and experience relevant to the position.",
            ContentType.EXPERIENCE: f"Valuable experience in {content_item.get('position', 'previous role')} contributing to professional growth.",
            ContentType.PROJECT: f"Successful completion of {content_item.get('name', 'project')} demonstrating technical capabilities.",
            ContentType.EXECUTIVE_SUMMARY: "Experienced professional with a strong background in delivering results and contributing to organizational success."
        }
        
        return fallbacks.get(content_type, "Professional experience and qualifications relevant to the position.")
    
    # Content formatting methods
    def _format_qualification_content(self, content: str) -> str:
        """Format qualification content."""
        # Ensure it starts with a strong statement
        if not content.startswith(("Strong", "Experienced", "Skilled", "Proficient")):
            content = f"Strong {content.lower()}"
        return content
    
    def _format_experience_content(self, content: str) -> str:
        """Format experience content."""
        # Ensure bullet points or clear structure
        if "•" not in content and "*" not in content and "-" not in content:
            # Add basic structure if missing
            sentences = content.split(". ")
            if len(sentences) > 1:
                content = "\n".join([f"• {sentence.strip()}" for sentence in sentences if sentence.strip()])
        return content
    
    def _format_project_content(self, content: str) -> str:
        """Format project content."""
        # Ensure clear project description
        return content
    
    def _format_executive_summary_content(self, content: str) -> str:
        """Format executive summary content."""
        # Ensure professional tone and appropriate length
        if len(content) > 300:
            # Truncate if too long
            content = content[:297] + "..."
        return content
    
    # Template methods
    def _get_qualification_template(self) -> str:
        return """
Generate a professional qualification statement for a {job_title} position at {company_name}.

Job Requirements:
{job_description}

Current Qualification:
{item_title}: {item_description}

Additional Context:
{additional_context}

Generate a concise, impactful qualification statement (2-3 sentences) that:
1. Highlights relevant skills and experience
2. Aligns with the job requirements
3. Uses professional language
4. Demonstrates value to the employer

Qualification Statement:"""
    
    def _get_experience_template(self) -> str:
        return """
Generate professional experience content for a {job_title} position at {company_name}.

Job Requirements:
{job_description}

Experience to Enhance:
{item_title}: {item_description}

Additional Context:
{additional_context}

Generate enhanced experience content that:
1. Emphasizes achievements and impact
2. Uses action verbs and quantifiable results
3. Aligns with the target job requirements
4. Maintains professional tone
5. Includes 3-5 bullet points if appropriate

Enhanced Experience:"""
    
    def _get_project_template(self) -> str:
        return """
Generate professional project description for a {job_title} position at {company_name}.

Job Requirements:
{job_description}

Project to Enhance:
{item_title}: {item_description}

Additional Context:
{additional_context}

Generate enhanced project content that:
1. Highlights technical skills and technologies used
2. Emphasizes project impact and results
3. Aligns with job requirements
4. Demonstrates problem-solving abilities
5. Uses professional technical language

Enhanced Project Description:"""
    
    def _get_executive_summary_template(self) -> str:
        return """
Generate a compelling executive summary for a {job_title} position at {company_name}.

Job Requirements:
{job_description}

Current Summary Context:
{item_description}

Additional Context:
{additional_context}

Generate a professional executive summary (3-4 sentences) that:
1. Captures key professional strengths
2. Aligns with the target role
3. Highlights unique value proposition
4. Uses confident, professional language
5. Focuses on achievements and capabilities

Executive Summary:"""


# Factory function for creating specialized content writers
def create_content_writer(content_type: ContentType) -> EnhancedContentWriterAgent:
    """Create a specialized content writer for a specific content type."""
    
    type_names = {
        ContentType.QUALIFICATION: "QualificationWriter",
        ContentType.EXPERIENCE: "ExperienceWriter", 
        ContentType.PROJECT: "ProjectWriter",
        ContentType.EXECUTIVE_SUMMARY: "ExecutiveSummaryWriter"
    }
    
    type_descriptions = {
        ContentType.QUALIFICATION: "Specialized agent for generating qualification statements",
        ContentType.EXPERIENCE: "Specialized agent for enhancing work experience descriptions",
        ContentType.PROJECT: "Specialized agent for creating project descriptions",
        ContentType.EXECUTIVE_SUMMARY: "Specialized agent for crafting executive summaries"
    }
    
    return EnhancedContentWriterAgent(
        name=type_names[content_type],
        description=type_descriptions[content_type],
        content_type=content_type
    )