"""Enhanced Content Writer Agent with Phase 1 infrastructure integration."""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from src.agents.agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from src.services.llm import get_llm_service, LLMResponse
from src.models.data_models import ContentType, ProcessingStatus
from src.config.logging_config import get_structured_logger
from src.config.settings import get_config
from src.core.state_manager import (
    ContentData,
    AgentIO,
    ExperienceEntry,
    CVData,
)
from src.models.data_models import (
    JobDescriptionData,
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
        
        # Initialize settings for prompt loading
        self.settings = get_config()
        
        # Content generation templates - load from external files
        self.content_templates = {
            ContentType.QUALIFICATION: self._load_prompt_template("key_qualifications_prompt"),
            ContentType.EXPERIENCE: self._load_prompt_template("resume_role_prompt"),
            ContentType.PROJECT: self._load_prompt_template("side_project_prompt"),
            ContentType.EXECUTIVE_SUMMARY: self._load_prompt_template("executive_summary_prompt")
        }
        
        logger.info(
            "Enhanced Content Writer Agent initialized",
            agent_name=name,
            supported_content_types=[ct.value for ct in self.content_templates.keys()]
        )
    
    async def run_async(self, input_data: Any, context: AgentExecutionContext) -> AgentResult:
        """Enhanced async content generation with structured processing."""
        from src.models.validation_schemas import validate_agent_input, ValidationError
        
        try:
            # Use provided input_data or fallback to context.input_data
            if input_data is None:
                input_data = context.input_data or {}
            
            # Validate input data using Pydantic schemas
            try:
                validated_input = validate_agent_input('enhanced_content_writer', input_data)
                # Convert validated Pydantic model back to dict for processing
                input_data = validated_input.model_dump()
                logger.info("Input validation passed for EnhancedContentWriter")
            except ValidationError as ve:
                logger.error(f"Input validation failed for EnhancedContentWriter: {ve.message}")
                return AgentResult(
                    success=False,
                    output_data={"error": "Input validation failed"},
                    confidence_score=0.0,
                    error_message=f"Input validation failed: {ve.message}",
                    metadata={"agent_type": "enhanced_content_writer", "validation_error": True}
                )
            except Exception as e:
                logger.error(f"Input validation error for EnhancedContentWriter: {str(e)}")
                return AgentResult(
                    success=False,
                    output_data={"error": "Input validation error"},
                    confidence_score=0.0,
                    error_message=f"Input validation error: {str(e)}",
                    metadata={"agent_type": "enhanced_content_writer", "validation_error": True}
                )
            
            # Debug: Log input_data type and content
            logger.info(f"EnhancedContentWriter received input_data type: {type(input_data)}")
            logger.info(f"EnhancedContentWriter received input_data: {input_data}")
            
            # Extract and validate input data with enhanced defensive programming
            job_data = input_data.get("job_description_data", {})
            
            # Enhanced type checking and validation for job_data parameter
            if isinstance(job_data, str):
                logger.warning(
                    f"DATA STRUCTURE MISMATCH: job_description_data is a string instead of dict. "
                    f"String length: {len(job_data)}, Preview: {job_data[:100]}..."
                )
                # Convert string job description to a structured JobDescriptionData format
                job_data = {
                    "raw_text": job_data,
                    "skills": [],
                    "experience_level": "N/A",
                    "responsibilities": [],
                    "industry_terms": [],
                    "company_values": []
                }
                logger.info("Successfully converted string job_description_data to structured format")
            elif isinstance(job_data, dict):
                # Validate required fields and add missing ones with detailed logging
                required_fields = ["raw_text", "skills", "experience_level", "responsibilities", "industry_terms", "company_values"]
                missing_fields = [field for field in required_fields if field not in job_data]
                
                if missing_fields:
                    logger.warning(
                        f"DATA STRUCTURE INCOMPLETE: job_description_data missing fields: {missing_fields}. "
                        f"Available fields: {list(job_data.keys())}"
                    )
                    # Add missing fields with default values
                    for field in missing_fields:
                        if field == "raw_text":
                            job_data[field] = job_data.get("description", "")
                        elif field == "experience_level":
                            job_data[field] = "N/A"
                        else:
                            job_data[field] = []
                    logger.info(f"Added missing fields to job_description_data: {missing_fields}")
                else:
                    logger.debug("job_description_data structure validation passed")
            elif job_data is None:
                logger.error("DATA STRUCTURE ERROR: job_description_data is None")
                job_data = {
                    "raw_text": "",
                    "skills": [],
                    "experience_level": "N/A",
                    "responsibilities": [],
                    "industry_terms": [],
                    "company_values": []
                }
            else:
                logger.error(
                    f"DATA STRUCTURE ERROR: job_description_data has unexpected type {type(job_data)}. "
                    f"Value: {job_data}. Converting to empty structured format."
                )
                job_data = {
                    "raw_text": str(job_data) if job_data else "",
                    "skills": [],
                    "experience_level": "N/A",
                    "responsibilities": [],
                    "industry_terms": [],
                    "company_values": []
                }
            
            # Enhanced validation for content_item with detailed error logging
            content_item = input_data.get("content_item", {})
            if not isinstance(content_item, dict):
                logger.error(
                    f"DATA STRUCTURE ERROR: content_item has unexpected type {type(content_item)}. "
                    f"Value: {content_item}. Converting to empty dict."
                )
                content_item = {}
            
            # Validate content_item structure
            if content_item:
                required_content_fields = ["type", "data"]
                missing_content_fields = [field for field in required_content_fields if field not in content_item]
                if missing_content_fields:
                    logger.warning(
                        f"DATA STRUCTURE INCOMPLETE: content_item missing fields: {missing_content_fields}. "
                        f"Available fields: {list(content_item.keys())}"
                    )
                    # Add missing fields with defaults
                    if "type" not in content_item:
                        content_item["type"] = "unknown"
                    if "data" not in content_item:
                        content_item["data"] = {}
                else:
                    logger.debug("content_item structure validation passed")
            
            # Enhanced validation for generation_context
            generation_context = input_data.get("generation_context", input_data.get("context", {}))
            if not isinstance(generation_context, dict):
                logger.error(
                    f"DATA STRUCTURE ERROR: context has unexpected type {type(generation_context)}. "
                    f"Value: {generation_context}. Converting to empty dict."
                )
                generation_context = {}
            
            # Determine content type
            content_type = context.content_type or self._determine_content_type(content_item)
            
            # Log generation start
            self.log_decision(
                f"Starting content generation for {content_type.value}",
                context if hasattr(context, 'session_id') else None
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
                context if hasattr(context, 'session_id') else None
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
                session_id=getattr(context, 'session_id', None),
                item_id=getattr(context, 'item_id', None),
                error=str(e),
                agent_name=self.name
            )
            
            # Return error result with fallback content
            content_item = input_data.get("content_item", {})
            # Ensure content_item is a dictionary for fallback generation
            if not isinstance(content_item, dict):
                content_item = {}
            content_type_fallback = getattr(context, 'content_type', None) or ContentType.QUALIFICATION
            fallback_content = self._generate_fallback_content(
                content_item,
                content_type_fallback
            )
            
            return AgentResult(
                success=False,
                output_data={
                    "content": fallback_content,
                    "content_type": content_type_fallback.value,
                    "confidence_score": 0.1,
                    "error": str(e),
                    "fallback_used": True
                },
                confidence_score=0.1,
                error_message=str(e),
                metadata={"fallback_used": True, "error_type": type(e).__name__}
            )
    
    def run(self, input_data: Any) -> Any:
        """Legacy synchronous interface for backward compatibility."""
        # Create a basic context for legacy calls
        context = AgentExecutionContext(
            session_id="legacy_session",
            item_id="legacy_item",
            content_type=self.content_type,
            input_data=input_data if isinstance(input_data, dict) else {}
        )
        
        # Run async method synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.run_async(input_data, context))
            return result
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
            session_id=getattr(context, 'session_id', None),
            item_id=getattr(context, 'item_id', None),
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
        
        # Defensive programming: Ensure job_data is a dictionary
        if not isinstance(job_data, dict):
            logger.error(f"Expected job_data to be a dict, but got {type(job_data)}. Input (first 200 chars): {str(job_data)[:200]}")
            # Fallback to an empty dictionary to prevent AttributeError and allow graceful degradation
            job_data = {}
        
        template = self.content_templates.get(
            content_type, 
            self.content_templates[ContentType.QUALIFICATION]
        )
        
        # Handle EXPERIENCE content type with resume_role_prompt template
        if content_type == ContentType.EXPERIENCE:
            return self._build_experience_prompt(template, job_data, content_item, generation_context)
        
        # Extract relevant information for other content types
        job_title = job_data.get("title", "the position")
        job_description = job_data.get("raw_text", job_data.get("description", ""))
        company_name = job_data.get("company", "the company")
        
        # Ensure content_item is a dictionary
        if not isinstance(content_item, dict):
            content_item = {}
        
        # Content-specific information
        if content_type == ContentType.QUALIFICATION:
            item_title = content_item.get("title", "qualification")
            item_description = content_item.get("description", "")
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
    
    def _build_experience_prompt(
        self,
        template: str,
        job_data: Dict[str, Any],
        content_item: Dict[str, Any],
        generation_context: Dict[str, Any]
    ) -> str:
        """Build prompt specifically for experience content using resume_role_prompt template."""
        
        # Defensive programming: Ensure job_data is a dictionary
        if not isinstance(job_data, dict):
            logger.error(f"Expected job_data to be a dict, but got {type(job_data)}. Input (first 200 chars): {str(job_data)[:200]}")
            # Fallback to an empty dictionary to prevent AttributeError and allow graceful degradation
            job_data = {}
        
        # Handle case where content_item might be a string (from data adapter)
        if isinstance(content_item, str):
            # Parse the CV text to extract experience information
            content_item = self._parse_cv_text_to_content_item(content_item, generation_context)
        
        # Extract target skills from job data
        target_skills = job_data.get("skills", [])
        if isinstance(target_skills, list):
            target_skills_text = "\n".join([f"- {skill}" for skill in target_skills])
        else:
            target_skills_text = "- No specific skills identified"
        
        # Build batched structured output for the role
        role_info = self._format_role_info(content_item, generation_context)
        
        # Format the template with the correct variables
        prompt = template.replace("{{Target Skills}}", target_skills_text)
        prompt = prompt.replace("{{batched_structured_output}}", role_info)
        
        return prompt
    
    def _format_role_info(self, content_item: Dict[str, Any], generation_context: Dict[str, Any]) -> str:
        """Format role information for the resume template."""
        
        # Handle case where content_item itself is a string (CV text)
        if isinstance(content_item, str):
            logger.info("Received CV text as content_item, parsing to structured format")
            parsed_content = self._parse_cv_text_to_content_item(content_item, generation_context)
            if isinstance(parsed_content, dict):
                return self._format_role_info(parsed_content, generation_context)
            else:
                # Fallback if parsing fails
                logger.warning("Failed to parse CV text, using fallback structure")
                content_item = {"name": "Professional Experience", "items": []}
        
        # Ensure content_item is a dictionary
        if not isinstance(content_item, dict):
            logger.error(f"Expected dict for content_item, got {type(content_item)}: {content_item}")
            content_item = {"name": "Unknown Role", "items": []}
        
        # Handle workflow adapter data structure with 'data' wrapper
        if "data" in content_item and "roles" in content_item["data"]:
            roles = content_item["data"]["roles"]
            
            # Validate roles is a list
            if not isinstance(roles, list):
                logger.error(f"Expected list for roles, got {type(roles)}: {roles}")
                roles = []
            
            # Check if roles is empty
            if not roles:
                logger.warning("Empty roles list, creating fallback structure")
                fallback_role = {"name": "Professional Experience", "items": []}
                return self._format_single_role(fallback_role, generation_context)
            
            # Check if roles is a list of strings (CV text) that needs parsing
            if roles and isinstance(roles[0], str):
                # Parse the CV text to extract structured role information
                cv_text = roles[0]  # Take the first (and likely only) CV text
                parsed_content = self._parse_cv_text_to_content_item(cv_text, generation_context)
                return self._format_role_info(parsed_content, generation_context)
            
            # Handle structured role data
            formatted_roles = []
            for i, role in enumerate(roles):
                try:
                    # Check if role is a dictionary (structured data) or string (CV text)
                    if isinstance(role, dict):
                        role_block = self._format_workflow_role(role, generation_context)
                        formatted_roles.append(role_block)
                    elif isinstance(role, str):
                        # This is CV text that needs parsing
                        parsed_content = self._parse_cv_text_to_content_item(role, generation_context)
                        # Ensure parsed_content is valid before formatting
                        if isinstance(parsed_content, dict):
                            role_block = self._format_single_role(parsed_content, generation_context)
                            formatted_roles.append(role_block)
                        else:
                            logger.warning(f"Failed to parse CV text to structured content: {role[:100]}...")
                            # Create a fallback role block
                            fallback_role = {"name": "Professional Experience", "items": []}
                            role_block = self._format_single_role(fallback_role, generation_context)
                            formatted_roles.append(role_block)
                    else:
                        logger.warning(f"Unexpected role type at index {i}: {type(role)}, content: {role}")
                        # Create a fallback role block for unexpected types
                        fallback_role = {"name": f"Role {i+1}", "items": []}
                        role_block = self._format_single_role(fallback_role, generation_context)
                        formatted_roles.append(role_block)
                except Exception as e:
                    logger.error(f"Error processing role at index {i}: {e}")
                    # Create a fallback role block for errors
                    fallback_role = {"name": f"Role {i+1}", "items": []}
                    role_block = self._format_single_role(fallback_role, generation_context)
                    formatted_roles.append(role_block)
            
            # Ensure we have at least one role
            if not formatted_roles:
                logger.warning("No roles were successfully formatted, creating fallback")
                fallback_role = {"name": "Professional Experience", "items": []}
                role_block = self._format_single_role(fallback_role, generation_context)
                formatted_roles.append(role_block)
            
            return "\n\n".join(formatted_roles)
        
        # Handle both section-level and subsection-level content (legacy format)
        elif "subsections" in content_item and content_item["subsections"]:
            # This is a section with subsections (multiple roles)
            formatted_roles = []
            for i, subsection in enumerate(content_item["subsections"]):
                try:
                    role_block = self._format_single_role(subsection, generation_context)
                    formatted_roles.append(role_block)
                except Exception as e:
                    logger.error(f"Error processing subsection at index {i}: {e}")
                    # Create a fallback role block
                    fallback_role = {"name": f"Role {i+1}", "items": []}
                    role_block = self._format_single_role(fallback_role, generation_context)
                    formatted_roles.append(role_block)
            
            # Ensure we have at least one role
            if not formatted_roles:
                logger.warning("No subsections were successfully formatted, creating fallback")
                fallback_role = {"name": "Professional Experience", "items": []}
                role_block = self._format_single_role(fallback_role, generation_context)
                formatted_roles.append(role_block)
            
            return "\n\n".join(formatted_roles)
        else:
            # This is a single role (legacy format)
            return self._format_single_role(content_item, generation_context)
    
    def _format_single_role(self, role_data: Dict[str, Any], generation_context: Dict[str, Any]) -> str:
        """Format a single role for the resume template."""
        
        # Handle case where role_data might be a string (CV text)
        if isinstance(role_data, str):
            # Parse the CV text to get structured data
            role_data = self._parse_cv_text_to_content_item(role_data, generation_context)
        
        # Ensure role_data is a dictionary
        if not isinstance(role_data, dict):
            logger.error(f"Expected dict for role_data, got {type(role_data)}: {role_data}")
            role_data = {"name": "Unknown Role", "items": []}
        
        # Extract role information
        role_name = role_data.get("name", "Unknown Role")
        
        # Extract company from original CV text if available
        company_name = self._extract_company_from_cv(role_name, generation_context)
        
        # Extract accomplishments from items
        accomplishments = []
        if "items" in role_data:
            for item in role_data["items"]:
                if isinstance(item, dict) and item.get("item_type") == "bullet_point":
                    accomplishments.append(item.get("content", ""))
        
        # Format the role info block
        role_info = f"""<role_info_start>
<info>
Role: {role_name}
Organization: {company_name}
Description: {role_name} position with focus on data analysis and technical implementation
</info>
<accomplishments>
"""
        
        for accomplishment in accomplishments:
            role_info += f"- {accomplishment}\n"
        
        role_info += "</accomplishments>\n</role_info_end>"
        
        return role_info
    
    def _format_workflow_role(self, role_data: Dict[str, Any], generation_context: Dict[str, Any]) -> str:
        """Format a single role from workflow adapter data structure."""
        
        # Ensure role_data is a dictionary
        if not isinstance(role_data, dict):
            logger.error(f"Expected dict for role_data, got {type(role_data)}: {role_data}")
            role_data = {"title": "Unknown Role", "company": "Unknown Company", "description": "", "skills": []}
        
        # Extract role information from workflow format
        role_title = role_data.get("title", "Unknown Role")
        company_name = role_data.get("company", "Unknown Company")
        description = role_data.get("description", "")
        skills = role_data.get("skills", [])
        
        # Create accomplishments from description and skills
        accomplishments = []
        if description:
            accomplishments.append(description)
        
        # Add skills as accomplishments if available
        if skills:
            skills_text = f"Key skills: {', '.join(skills)}"
            accomplishments.append(skills_text)
        
        # Format the role info block
        role_info = f"""<role_info_start>
<info>
Role: {role_title}
Organization: {company_name}
Description: {description[:200] if description else f'{role_title} position with focus on technical implementation'}
</info>
<accomplishments>
"""
        
        for accomplishment in accomplishments:
            role_info += f"- {accomplishment}\n"
        
        role_info += "</accomplishments>\n</role_info_end>"
        
        return role_info
    
    def _extract_company_from_cv(self, role_name: str, generation_context: Dict[str, Any]) -> str:
        """Extract company name from original CV text based on role name."""
        
        # Get original CV text from generation context
        original_cv = generation_context.get("original_cv_text", "")
        
        # Define company mappings based on the CV content
        company_mappings = {
            "Trainee Data Analyst": "STE Smart-Send",
            "IT trainer": "Supply Chain Management Center", 
            "Mathematics teacher": "Martile Secondary School",
            "Indie Mobile Game Developer": "Unity (Independent)"
        }
        
        # Return mapped company or try to extract from CV text
        if role_name in company_mappings:
            return company_mappings[role_name]
        
        # Fallback: try to find company in original CV text near the role name
        if original_cv and role_name in original_cv:
            # Simple extraction logic - this could be improved
            lines = original_cv.split('\n')
            for i, line in enumerate(lines):
                if role_name in line and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith('*'):
                        # Extract company from next line
                        if '|' in next_line:
                            company_part = next_line.split('|')[0].strip()
                            if company_part.startswith('[') and ']' in company_part:
                                return company_part.split(']')[0][1:]
                            return company_part
        
        return "Previous Company"
    
    def _parse_cv_text_to_content_item(self, cv_text: str, generation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse CV text to extract content item structure."""
        
        if not cv_text or not isinstance(cv_text, str):
            logger.warning(f"Invalid CV text input: {type(cv_text)}")
            return {"name": "Professional Experience", "items": []}
        
        import re
        
        # Define enhanced role patterns to look for
        role_patterns = [
            # More comprehensive patterns
            r'(?i)^\s*([A-Za-z\s,.-]+?)\s*[-–—]\s*([A-Za-z\s&,.()-]+?)\s*\(([^)]+)\)',  # Title - Company (Duration)
            r'(?i)^\s*([A-Za-z\s,.-]+?)\s*at\s+([A-Za-z\s&,.()-]+?)\s*\(([^)]+)\)',  # Title at Company (Duration)
            r'(?i)^\s*([A-Za-z\s,.-]+?)\s*[-–—]\s*([A-Za-z\s&,.()-]+?)\s*[,|]\s*([A-Za-z\s0-9-]+)',  # Title - Company, Duration
            r'(?i)^\s*([A-Za-z\s,.-]+?)\s*at\s+([A-Za-z\s&,.()-]+?)\s*[,|]\s*([A-Za-z\s0-9-]+)',  # Title at Company, Duration
            r'(?i)^\s*([A-Za-z\s,.-]+?)\s*[-–—]\s*([A-Za-z\s&,.()-]+)',  # Title - Company
            r'(?i)^\s*([A-Za-z\s,.-]+?)\s*at\s+([A-Za-z\s&,.()-]+)',  # Title at Company
            r'(?i)^\s*([A-Za-z\s,.-]+?)\s*\|\s*([A-Za-z\s&,.()-]+)',  # Title | Company
            r'(?i)^\s*([A-Za-z\s,.-]{3,})$',  # Single line that could be a role title
        ]
        
        # Split text into lines and clean them
        lines = [line.strip() for line in cv_text.strip().split('\n') if line.strip()]
        
        if not lines:
            logger.warning("No content found in CV text")
            return {"name": "Professional Experience", "items": []}
        
        roles = []
        current_role = None
        
        for i, line in enumerate(lines):
            # Skip very short lines that are unlikely to be role titles
            if len(line) < 3:
                continue
                
            # Check if this line matches a role pattern
            role_match = None
            matched_pattern_index = -1
            
            for pattern_index, pattern in enumerate(role_patterns):
                try:
                    match = re.match(pattern, line)
                    if match:
                        role_match = match
                        matched_pattern_index = pattern_index
                        break
                except re.error as e:
                    logger.warning(f"Regex error with pattern {pattern_index}: {e}")
                    continue
            
            if role_match:
                # Save previous role if exists
                if current_role and current_role.get("items"):
                    roles.append(current_role)
                
                # Start new role based on matched pattern
                groups = role_match.groups()
                
                if matched_pattern_index <= 3 and len(groups) >= 2:  # Patterns with company info
                    title = groups[0].strip()
                    company = groups[1].strip()
                    duration = groups[2].strip() if len(groups) > 2 else ""
                    
                    role_name = f"{title} at {company}"
                    if duration:
                        role_name += f" ({duration})"
                        
                    current_role = {
                        "name": role_name,
                        "items": []
                    }
                elif matched_pattern_index <= 6 and len(groups) >= 2:  # Simple Title - Company patterns
                    title = groups[0].strip()
                    company = groups[1].strip()
                    current_role = {
                        "name": f"{title} at {company}",
                        "items": []
                    }
                else:  # Single line role title
                    current_role = {
                        "name": groups[0].strip(),
                        "items": []
                    }
                    
            elif line.startswith(('•', '-', '*', '◦', '▪', '▫')) or re.match(r'^\s*\d+[.)].', line):
                # This is a bullet point/accomplishment
                if current_role:
                    # Clean the accomplishment text
                    accomplishment = re.sub(r'^[•\-*◦▪▫\d+.)\s]+', '', line).strip()
                    if accomplishment and len(accomplishment) > 5:  # Filter out very short items
                        current_role["items"].append({
                            "item_type": "bullet_point",
                            "content": accomplishment
                        })
                elif not roles:  # No current role but we have accomplishments
                    # Create a generic role for orphaned accomplishments
                    current_role = {
                        "name": "Professional Experience",
                        "items": []
                    }
                    accomplishment = re.sub(r'^[•\-*◦▪▫\d+.)\s]+', '', line).strip()
                    if accomplishment and len(accomplishment) > 5:
                        current_role["items"].append({
                            "item_type": "bullet_point",
                            "content": accomplishment
                        })
            else:
                # Check if this could be a continuation of an accomplishment
                if current_role and current_role.get("items") and not role_match:
                    # If the line doesn't look like a new role and we have a current role,
                    # it might be a continuation of the previous accomplishment
                    if len(line) > 10 and not line.isupper():  # Avoid headers
                        current_role["items"].append({
                            "item_type": "bullet_point",
                            "content": line
                        })
        
        # Add the last role
        if current_role and (current_role.get("items") or not roles):
            roles.append(current_role)
        
        # If no roles were found, try to extract any meaningful content
        if not roles:
            # Look for any bullet points or numbered lists in the text
            accomplishments = []
            for line in lines:
                if line.startswith(('•', '-', '*', '◦', '▪', '▫')) or re.match(r'^\s*\d+[.)].', line):
                    accomplishment = re.sub(r'^[•\-*◦▪▫\d+.)\s]+', '', line).strip()
                    if accomplishment and len(accomplishment) > 5:
                        accomplishments.append({
                            "item_type": "bullet_point",
                            "content": accomplishment
                        })
            
            return {
                "name": "Professional Experience",
                "items": accomplishments if accomplishments else [{"item_type": "bullet_point", "content": "No specific accomplishments extracted"}]
            }
        
        # If only one role, return it directly
        if len(roles) == 1:
            return roles[0]
        
        # Multiple roles, return as subsections
        return {
            "name": "Professional Experience",
            "subsections": roles
        }
    
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
        
        # Ensure content_item is a dictionary before checking keys
        if not isinstance(content_item, dict):
            logger.warning(f"Expected dict for content_item, got {type(content_item)}: {content_item}")
            return ContentType.QUALIFICATION
        
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
        
        # Ensure content_item is a dictionary
        if not isinstance(content_item, dict):
            content_item = {}
        
        # Safely get values from content_item (now guaranteed to be a dictionary)
        position = content_item.get('position', 'previous role')
        project_name = content_item.get('name', 'project')
        
        fallbacks = {
            ContentType.QUALIFICATION: "Strong technical skills and experience relevant to the position.",
            ContentType.EXPERIENCE: f"Valuable experience in {position} contributing to professional growth.",
            ContentType.PROJECT: f"Successful completion of {project_name} demonstrating technical capabilities.",
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
    
    def _load_prompt_template(self, prompt_name: str) -> str:
        """Load a prompt template from external file."""
        try:
            prompt_path = self.settings.get_prompt_path(prompt_name)
            with open(prompt_path, 'r', encoding='utf-8') as f:
                template = f.read()
            logger.info(f"Successfully loaded prompt template: {prompt_name}")
            return template
        except Exception as e:
            logger.error(f"Error loading prompt template {prompt_name}: {e}")
            # Fallback to basic template
            return self._get_fallback_template()
    
    def _get_fallback_template(self) -> str:
        """Fallback template if external prompt loading fails."""
        return """
Generate professional content for a {job_title} position at {company_name}.

Job Requirements:
{job_description}

Content to Enhance:
{item_title}: {item_description}

Additional Context:
{additional_context}

Generate enhanced content that aligns with the job requirements and demonstrates value to the employer.
"""


# Factory function for creating specialized content writers
def create_content_writer(content_type: ContentType) -> EnhancedContentWriterAgent:
    """Create a specialized content writer for a specific content type."""
    
    type_names = {
        ContentType.QUALIFICATION: "QualificationWriter",
        ContentType.EXPERIENCE: "ExperienceWriter", 
        ContentType.EXPERIENCE_ITEM: "ExperienceItemWriter",
        ContentType.PROJECT: "ProjectWriter",
        ContentType.PROJECT_ITEM: "ProjectItemWriter",
        ContentType.EXECUTIVE_SUMMARY: "ExecutiveSummaryWriter",
        ContentType.SKILL: "SkillWriter",
        ContentType.ACHIEVEMENT: "AchievementWriter",
        ContentType.EDUCATION: "EducationWriter",
        ContentType.SKILLS: "SkillsWriter",
        ContentType.PROJECTS: "ProjectsWriter",
        ContentType.ANALYSIS: "AnalysisWriter",
        ContentType.QUALITY_CHECK: "QualityCheckWriter",
        ContentType.OPTIMIZATION: "OptimizationWriter"
    }
    
    type_descriptions = {
        ContentType.QUALIFICATION: "Specialized agent for generating qualification statements",
        ContentType.EXPERIENCE: "Specialized agent for enhancing work experience descriptions",
        ContentType.EXPERIENCE_ITEM: "Specialized agent for enhancing individual experience items",
        ContentType.PROJECT: "Specialized agent for creating project descriptions",
        ContentType.PROJECT_ITEM: "Specialized agent for creating individual project items",
        ContentType.EXECUTIVE_SUMMARY: "Specialized agent for crafting executive summaries",
        ContentType.SKILL: "Specialized agent for enhancing skill descriptions",
        ContentType.ACHIEVEMENT: "Specialized agent for highlighting achievements",
        ContentType.EDUCATION: "Specialized agent for education content",
        ContentType.SKILLS: "Specialized agent for skills section content",
        ContentType.PROJECTS: "Specialized agent for projects section content",
        ContentType.ANALYSIS: "Specialized agent for content analysis",
        ContentType.QUALITY_CHECK: "Specialized agent for quality assurance",
        ContentType.OPTIMIZATION: "Specialized agent for content optimization"
    }
    
    return EnhancedContentWriterAgent(
        name=type_names[content_type],
        description=type_descriptions[content_type],
        content_type=content_type
    )