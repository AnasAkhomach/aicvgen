"""Enhanced Content Writer Agent with Phase 1 infrastructure integration."""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from src.agents.agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from src.services.llm_service import get_llm_service
from src.services.llm import LLMResponse
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
from src.orchestration.state import AgentState
from src.core.async_optimizer import optimize_async
from src.utils.exceptions import (
    ValidationError,
    LLMResponseParsingError,
    WorkflowPreconditionError,
    AgentExecutionError,
    ConfigurationError,
    StateManagerError,
)

logger = get_structured_logger("enhanced_content_writer")


class EnhancedContentWriterAgent(EnhancedAgentBase):
    """Enhanced Content Writer Agent with Phase 1 infrastructure integration."""

    def __init__(
        self,
        name: str = "EnhancedContentWriter",
        description: str = "Enhanced agent for generating tailored CV content with advanced error handling and progress tracking",
        content_type: ContentType = ContentType.QUALIFICATION,
    ):
        """Initialize the enhanced content writer agent."""
        super().__init__(
            name=name,
            description=description,
            input_schema=AgentIO(
                description="Generates enhanced CV content with structured logging and error handling",
                required_fields=["job_description_data", "content_item", "context"],
            ),
            output_schema=AgentIO(
                description="Generated content with metadata and quality metrics",
                required_fields=["content", "metadata", "quality_metrics"],
            ),
            content_type=content_type,
        )

        # Enhanced services
        self.llm_service = get_llm_service()

        # Initialize settings for prompt loading
        self.settings = get_config()

        # Content generation templates - load from external files
        self.content_templates = {
            ContentType.QUALIFICATION: self._load_prompt_template(
                "key_qualifications_writer"
            ),
            ContentType.EXPERIENCE: self._load_prompt_template("resume_role_writer"),
            ContentType.PROJECT: self._load_prompt_template("project_writer"),
            ContentType.EXECUTIVE_SUMMARY: self._load_prompt_template(
                "executive_summary_writer"
            ),
        }

        logger.info(
            "Enhanced Content Writer Agent initialized",
            agent_name=name,
            supported_content_types=[ct.value for ct in self.content_templates.keys()],
        )

    async def run_async(
        self, input_data: Any, context: AgentExecutionContext
    ) -> AgentResult:
        """Enhanced async content generation with structured processing and granular item support."""

        try:
            # Use provided input_data or fallback to context.input_data
            if input_data is None:
                input_data = context.input_data or {}

            # Basic input validation
            if not isinstance(input_data, dict):
                logger.error(
                    f"Input validation failed for EnhancedContentWriter: expected dict, got {type(input_data)}"
                )
                return AgentResult(
                    success=False,
                    output_data={"error": "Input validation failed"},
                    confidence_score=0.0,
                    error_message=f"Input validation failed: expected dict, got {type(input_data)}",
                    metadata={
                        "agent_type": "enhanced_content_writer",
                        "validation_error": True,
                    },
                )

            # Debug: Log input_data type and content
            logger.info(
                f"EnhancedContentWriter received input_data type: {type(input_data)}"
            )
            logger.info(f"EnhancedContentWriter received input_data: {input_data}")

            # Extract and validate input data with enhanced defensive programming
            job_data = input_data.get("job_description_data", {})

            # Check for granular processing mode (Task 3.1)
            current_item_id = input_data.get("current_item_id")
            structured_cv_data = input_data.get("structured_cv")

            if current_item_id and structured_cv_data:
                logger.info(f"Processing single item: {current_item_id}")
                return await self._process_single_item(
                    structured_cv_data, job_data, current_item_id
                )

            # FIXED: Defensive validation for job_description_data (Fix for CI-003)
            # This prevents AttributeError by ensuring job_data is always a valid JobDescriptionData structure
            try:
                from pydantic import ValidationError
                from src.models.data_models import JobDescriptionData

                raw_job_data = input_data.get("job_description_data")
                if isinstance(raw_job_data, dict):
                    job_data = JobDescriptionData.model_validate(raw_job_data)
                    logger.info("job_description_data validation passed")
                elif isinstance(raw_job_data, JobDescriptionData):
                    job_data = raw_job_data
                    logger.info(
                        "job_description_data is already a valid JobDescriptionData object"
                    )
                else:
                    # If it's not a dict or JobDescriptionData, it's malformed. Raise a validation error.
                    raise ValidationError.from_exception_data(
                        title="JobDescriptionData",
                        line_errors=[
                            {
                                "loc": ("job_description_data",),
                                "input": raw_job_data,
                                "msg": "Input should be a valid dictionary or JobDescriptionData object",
                            }
                        ],
                    )
            except ValidationError as e:
                logger.error(
                    f"Input validation failed for EnhancedContentWriterAgent: {e}",
                    exc_info=True,
                )
                return AgentResult(
                    success=False,
                    error_message=f"Invalid job description data structure: {e}",
                    output_data={
                        "error": f"Invalid job description data structure: {e}"
                    },
                    confidence_score=0.0,
                    metadata={
                        "agent_type": "enhanced_content_writer",
                        "validation_error": True,
                    },
                )
            except Exception as validation_error:
                logger.error(
                    f"job_description_data validation failed: {str(validation_error)}"
                )
                return AgentResult(
                    success=False,
                    error_message=f"Job description data validation error: {str(validation_error)}",
                    output_data={
                        "error": f"Job description data validation error: {str(validation_error)}"
                    },
                    confidence_score=0.0,
                    metadata={
                        "agent_type": "enhanced_content_writer",
                        "validation_error": True,
                    },
                )

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
                    "company_values": [],
                }
                logger.info(
                    "Successfully converted string job_description_data to structured format"
                )
            elif isinstance(job_data, dict):
                # Validate required fields and add missing ones with detailed logging
                required_fields = [
                    "raw_text",
                    "skills",
                    "experience_level",
                    "responsibilities",
                    "industry_terms",
                    "company_values",
                ]
                missing_fields = [
                    field for field in required_fields if field not in job_data
                ]

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
                    logger.info(
                        f"Added missing fields to job_description_data: {missing_fields}"
                    )
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
                    "company_values": [],
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
                    "company_values": [],
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
                missing_content_fields = [
                    field
                    for field in required_content_fields
                    if field not in content_item
                ]
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
            generation_context = input_data.get(
                "generation_context", input_data.get("context", {})
            )
            if not isinstance(generation_context, dict):
                logger.error(
                    f"DATA STRUCTURE ERROR: context has unexpected type {type(generation_context)}. "
                    f"Value: {generation_context}. Converting to empty dict."
                )
                generation_context = {}

            # Determine content type
            content_type = context.content_type or self._determine_content_type(
                content_item
            )

            # Log generation start
            self.log_decision(
                f"Starting content generation for {content_type.value}",
                context if hasattr(context, "session_id") else None,
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
                    **processed_content.metadata,
                },
            }

            self.log_decision(
                f"Content generation completed successfully for {content_type.value}",
                context if hasattr(context, "session_id") else None,
            )

            return AgentResult(
                success=True,
                output_data=result_data,
                confidence_score=confidence,
                processing_time=processed_content.processing_time,
                metadata=result_data["metadata"],
            )

        except (ValidationError, LLMResponseParsingError) as parse_error:
            logger.error(
                "Content parsing/validation failed",
                session_id=getattr(context, "session_id", None),
                item_id=getattr(context, "item_id", None),
                error=str(parse_error),
                agent_name=self.name,
                error_type="parsing_validation",
            )
            return self._create_error_result(
                input_data, context, parse_error, "parsing_validation"
            )
        except (ConfigurationError, StateManagerError) as system_error:
            logger.error(
                "System configuration error during content generation",
                session_id=getattr(context, "session_id", None),
                item_id=getattr(context, "item_id", None),
                error=str(system_error),
                agent_name=self.name,
                error_type="system",
            )
            return self._create_error_result(
                input_data, context, system_error, "system"
            )
        except AgentExecutionError as agent_error:
            logger.error(
                "Agent execution error during content generation",
                session_id=getattr(context, "session_id", None),
                item_id=getattr(context, "item_id", None),
                error=str(agent_error),
                agent_name=self.name,
                error_type="agent_execution",
            )
            return self._create_error_result(
                input_data, context, agent_error, "agent_execution"
            )
        except Exception as e:
            logger.error(
                "Unexpected error during content generation",
                session_id=getattr(context, "session_id", None),
                item_id=getattr(context, "item_id", None),
                error=str(e),
                agent_name=self.name,
                error_type="unexpected",
            )
            return self._create_error_result(input_data, context, e, "unexpected")

    # Legacy run method removed - use run_as_node for LangGraph integration

    async def _generate_content_with_llm(
        self,
        job_data: Dict[str, Any],
        content_item: Dict[str, Any],
        generation_context: Dict[str, Any],
        content_type: ContentType,
        context: AgentExecutionContext,
    ) -> LLMResponse:
        """Generate content using the enhanced LLM service with JSON parsing for role generation."""

        # Build prompt based on content type
        prompt = self._build_prompt(
            job_data, content_item, generation_context, content_type
        )

        # Generate content with LLM service
        response = await self.llm_service.generate_content(
            prompt=prompt,
            content_type=content_type,
            session_id=getattr(context, "session_id", None),
            item_id=getattr(context, "item_id", None),
            max_retries=3,
        )

        # Store raw LLM output in content_item if it's an Item model
        if hasattr(content_item, "raw_llm_output"):
            content_item.raw_llm_output = response.content
        elif isinstance(content_item, dict) and "raw_llm_output" in content_item:
            content_item["raw_llm_output"] = response.content

        # For Experience content type, parse JSON response and validate with Pydantic
        if content_type == ContentType.EXPERIENCE and response.success:
            try:
                # Extract JSON from the response
                json_content = self._extract_json_from_response(response.content)

                # Parse and validate with Pydantic model
                from src.models.validation_schemas import LLMRoleGenerationOutput

                validated_data = LLMRoleGenerationOutput.model_validate(json_content)

                # Convert validated data to formatted content
                formatted_content = self._format_role_generation_output(validated_data)

                # Update response content with formatted output
                response.content = formatted_content
                response.metadata["json_validated"] = True
                response.metadata["validation_model"] = "LLMRoleGenerationOutput"

                logger.info(
                    f"Successfully parsed and validated role generation JSON for {content_type}"
                )

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed for role generation: {e}")
                response.metadata["json_parse_error"] = str(e)
                response.metadata["json_validated"] = False
                # Keep original content as fallback

            except ValidationError as e:
                logger.error(f"Pydantic validation failed for role generation: {e}")
                response.metadata["validation_error"] = str(e)
                response.metadata["json_validated"] = False
                # Keep original content as fallback

            except Exception as e:
                logger.error(
                    f"Unexpected error during role generation JSON processing: {e}"
                )
                response.metadata["processing_error"] = str(e)
                response.metadata["json_validated"] = False
                # Keep original content as fallback

        return response

    def _build_prompt(
        self,
        job_data: Dict[str, Any],
        content_item: Dict[str, Any],
        generation_context: Dict[str, Any],
        content_type: ContentType,
    ) -> str:
        """Build a specialized prompt based on content type."""

        # Defensive programming: Ensure job_data is a dictionary
        if not isinstance(job_data, dict):
            logger.error(
                f"Expected job_data to be a dict, but got {type(job_data)}. Input (first 200 chars): {str(job_data)[:200]}"
            )
            # Fallback to an empty dictionary to prevent AttributeError and allow graceful degradation
            job_data = {}

        template = self.content_templates.get(
            content_type, self.content_templates[ContentType.QUALIFICATION]
        )

        # Handle EXPERIENCE content type with resume_role_prompt template
        if content_type == ContentType.EXPERIENCE:
            return self._build_experience_prompt(
                template, job_data, content_item, generation_context
            )

        # Extract relevant information for other content types
        job_title = job_data.get("title", "the position")
        job_description = job_data.get("raw_text", job_data.get("description", ""))
        company_name = job_data.get("company", "the company")

        # Ensure content_item is a dictionary
        if not isinstance(content_item, dict):
            content_item = {}

        # Content-specific information and prompt formatting
        if content_type == ContentType.QUALIFICATION:
            # Handle "Big 10" skills generation with updated template variables
            my_talents = generation_context.get(
                "my_talents",
                job_data.get(
                    "my_talents",
                    "Professional with diverse technical and analytical skills",
                ),
            )

            # Format the prompt with the updated key_qualifications_prompt variables
            prompt = template.format(
                main_job_description_raw=job_description[
                    :2000
                ],  # Increased limit for better analysis
                my_talents=my_talents,
            )
        elif content_type == ContentType.PROJECT:
            item_title = content_item.get("name", "project")
            item_description = content_item.get("description", "")

            # Format the prompt
            prompt = template.format(
                job_title=job_title,
                job_description=job_description[:1000],  # Limit length
                company_name=company_name,
                item_title=item_title,
                item_description=item_description,
                additional_context=generation_context.get("additional_context", ""),
            )
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
                additional_context=generation_context.get("additional_context", ""),
            )

        return prompt

    def _build_experience_prompt(
        self,
        template: str,
        job_data: Dict[str, Any],
        content_item: Dict[str, Any],
        generation_context: Dict[str, Any],
    ) -> str:
        """Build prompt specifically for experience content using resume_role_prompt template."""

        # Defensive programming: Ensure job_data is a dictionary
        if not isinstance(job_data, dict):
            logger.error(
                f"Expected job_data to be a dict, but got {type(job_data)}. Input (first 200 chars): {str(job_data)[:200]}"
            )
            # Fallback to an empty dictionary to prevent AttributeError and allow graceful degradation
            job_data = {}

        # Handle case where content_item might be a string (from data adapter)
        if isinstance(content_item, str):
            # Parse the CV text to extract experience information
            content_item = self._parse_cv_text_to_content_item(
                content_item, generation_context
            )

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

    def _format_role_info(
        self, content_item: Dict[str, Any], generation_context: Dict[str, Any]
    ) -> str:
        """Format role information for the resume template."""

        # Handle case where content_item itself is a string (CV text)
        if isinstance(content_item, str):
            logger.info(
                "Received CV text as content_item, parsing to structured format"
            )
            parsed_content = self._parse_cv_text_to_content_item(
                content_item, generation_context
            )
            if isinstance(parsed_content, dict):
                return self._format_role_info(parsed_content, generation_context)
            else:
                # Fallback if parsing fails
                logger.warning("Failed to parse CV text, using fallback structure")
                content_item = {"name": "Professional Experience", "items": []}

        # Ensure content_item is a dictionary
        if not isinstance(content_item, dict):
            logger.error(
                f"Expected dict for content_item, got {type(content_item)}: {content_item}"
            )
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
                parsed_content = self._parse_cv_text_to_content_item(
                    cv_text, generation_context
                )
                return self._format_role_info(parsed_content, generation_context)

            # Handle structured role data
            formatted_roles = []
            for i, role in enumerate(roles):
                try:
                    # Check if role is a dictionary (structured data) or string (CV text)
                    if isinstance(role, dict):
                        role_block = self._format_workflow_role(
                            role, generation_context
                        )
                        formatted_roles.append(role_block)
                    elif isinstance(role, str):
                        # This is CV text that needs parsing
                        parsed_content = self._parse_cv_text_to_content_item(
                            role, generation_context
                        )
                        # Ensure parsed_content is valid before formatting
                        if isinstance(parsed_content, dict):
                            role_block = self._format_single_role(
                                parsed_content, generation_context
                            )
                            formatted_roles.append(role_block)
                        else:
                            logger.warning(
                                f"Failed to parse CV text to structured content: {role[:100]}..."
                            )
                            # Create a fallback role block
                            fallback_role = {
                                "name": "Professional Experience",
                                "items": [],
                            }
                            role_block = self._format_single_role(
                                fallback_role, generation_context
                            )
                            formatted_roles.append(role_block)
                    else:
                        logger.warning(
                            f"Unexpected role type at index {i}: {type(role)}, content: {role}"
                        )
                        # Create a fallback role block for unexpected types
                        fallback_role = {"name": f"Role {i+1}", "items": []}
                        role_block = self._format_single_role(
                            fallback_role, generation_context
                        )
                        formatted_roles.append(role_block)
                except Exception as e:
                    logger.error(f"Error processing role at index {i}: {e}")
                    # Create a fallback role block for errors
                    fallback_role = {"name": f"Role {i+1}", "items": []}
                    role_block = self._format_single_role(
                        fallback_role, generation_context
                    )
                    formatted_roles.append(role_block)

            # Ensure we have at least one role
            if not formatted_roles:
                logger.warning(
                    "No roles were successfully formatted, creating fallback"
                )
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
                    role_block = self._format_single_role(
                        subsection, generation_context
                    )
                    formatted_roles.append(role_block)
                except Exception as e:
                    logger.error(f"Error processing subsection at index {i}: {e}")
                    # Create a fallback role block
                    fallback_role = {"name": f"Role {i+1}", "items": []}
                    role_block = self._format_single_role(
                        fallback_role, generation_context
                    )
                    formatted_roles.append(role_block)

            # Ensure we have at least one role
            if not formatted_roles:
                logger.warning(
                    "No subsections were successfully formatted, creating fallback"
                )
                fallback_role = {"name": "Professional Experience", "items": []}
                role_block = self._format_single_role(fallback_role, generation_context)
                formatted_roles.append(role_block)

            return "\n\n".join(formatted_roles)
        else:
            # This is a single role (legacy format)
            return self._format_single_role(content_item, generation_context)

    def _format_single_role(
        self, role_data: Dict[str, Any], generation_context: Dict[str, Any]
    ) -> str:
        """Format a single role for the resume template."""

        # Handle case where role_data might be a string (CV text)
        if isinstance(role_data, str):
            # Parse the CV text to get structured data
            role_data = self._parse_cv_text_to_content_item(
                role_data, generation_context
            )

        # Ensure role_data is a dictionary
        if not isinstance(role_data, dict):
            logger.error(
                f"Expected dict for role_data, got {type(role_data)}: {role_data}"
            )
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

    def _format_workflow_role(
        self, role_data: Dict[str, Any], generation_context: Dict[str, Any]
    ) -> str:
        """Format a single role from workflow adapter data structure."""

        # Ensure role_data is a dictionary
        if not isinstance(role_data, dict):
            logger.error(
                f"Expected dict for role_data, got {type(role_data)}: {role_data}"
            )
            role_data = {
                "title": "Unknown Role",
                "company": "Unknown Company",
                "description": "",
                "skills": [],
            }

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

    def _extract_company_from_cv(
        self, role_name: str, generation_context: Dict[str, Any]
    ) -> str:
        """Extract company name from original CV text based on role name."""

        # Get original CV text from generation context
        original_cv = generation_context.get("original_cv_text", "")

        # Define company mappings based on the CV content
        company_mappings = {
            "Trainee Data Analyst": "STE Smart-Send",
            "IT trainer": "Supply Chain Management Center",
            "Mathematics teacher": "Martile Secondary School",
            "Indie Mobile Game Developer": "Unity (Independent)",
        }

        # Return mapped company or try to extract from CV text
        if role_name in company_mappings:
            return company_mappings[role_name]

        # Fallback: try to find company in original CV text near the role name
        if original_cv and role_name in original_cv:
            # Simple extraction logic - this could be improved
            lines = original_cv.split("\n")
            for i, line in enumerate(lines):
                if role_name in line and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith("*"):
                        # Extract company from next line
                        if "|" in next_line:
                            company_part = next_line.split("|")[0].strip()
                            if company_part.startswith("[") and "]" in company_part:
                                return company_part.split("]")[0][1:]
                            return company_part

        return "Previous Company"

    def _parse_cv_text_to_content_item(
        self, cv_text: str, generation_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse CV text to extract content item structure."""

        if not cv_text or not isinstance(cv_text, str):
            logger.warning(f"Invalid CV text input: {type(cv_text)}")
            return {"name": "Professional Experience", "items": []}

        import re

        # Define enhanced role patterns to look for
        role_patterns = [
            # More comprehensive patterns
            r"(?i)^\s*([A-Za-z\s,.-]+?)\s*[-–—]\s*([A-Za-z\s&,.()-]+?)\s*\(([^)]+)\)",  # Title - Company (Duration)
            r"(?i)^\s*([A-Za-z\s,.-]+?)\s*at\s+([A-Za-z\s&,.()-]+?)\s*\(([^)]+)\)",  # Title at Company (Duration)
            r"(?i)^\s*([A-Za-z\s,.-]+?)\s*[-–—]\s*([A-Za-z\s&,.()-]+?)\s*[,|]\s*([A-Za-z\s0-9-]+)",  # Title - Company, Duration
            r"(?i)^\s*([A-Za-z\s,.-]+?)\s*at\s+([A-Za-z\s&,.()-]+?)\s*[,|]\s*([A-Za-z\s0-9-]+)",  # Title at Company, Duration
            r"(?i)^\s*([A-Za-z\s,.-]+?)\s*[-–—]\s*([A-Za-z\s&,.()-]+)",  # Title - Company
            r"(?i)^\s*([A-Za-z\s,.-]+?)\s*at\s+([A-Za-z\s&,.()-]+)",  # Title at Company
            r"(?i)^\s*([A-Za-z\s,.-]+?)\s*\|\s*([A-Za-z\s&,.()-]+)",  # Title | Company
            r"(?i)^\s*([A-Za-z\s,.-]{3,})$",  # Single line that could be a role title
        ]

        # Split text into lines and clean them
        lines = [line.strip() for line in cv_text.strip().split("\n") if line.strip()]

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

                if (
                    matched_pattern_index <= 3 and len(groups) >= 2
                ):  # Patterns with company info
                    title = groups[0].strip()
                    company = groups[1].strip()
                    duration = groups[2].strip() if len(groups) > 2 else ""

                    role_name = f"{title} at {company}"
                    if duration:
                        role_name += f" ({duration})"

                    current_role = {"name": role_name, "items": []}
                elif (
                    matched_pattern_index <= 6 and len(groups) >= 2
                ):  # Simple Title - Company patterns
                    title = groups[0].strip()
                    company = groups[1].strip()
                    current_role = {"name": f"{title} at {company}", "items": []}
                else:  # Single line role title
                    current_role = {"name": groups[0].strip(), "items": []}

            elif line.startswith(("•", "-", "*", "◦", "▪", "▫")) or re.match(
                r"^\s*\d+[.)].", line
            ):
                # This is a bullet point/accomplishment
                if current_role:
                    # Clean the accomplishment text
                    accomplishment = re.sub(r"^[•\-*◦▪▫\d+.)\s]+", "", line).strip()
                    if (
                        accomplishment and len(accomplishment) > 5
                    ):  # Filter out very short items
                        current_role["items"].append(
                            {"item_type": "bullet_point", "content": accomplishment}
                        )
                elif not roles:  # No current role but we have accomplishments
                    # Create a generic role for orphaned accomplishments
                    current_role = {"name": "Professional Experience", "items": []}
                    accomplishment = re.sub(r"^[•\-*◦▪▫\d+.)\s]+", "", line).strip()
                    if accomplishment and len(accomplishment) > 5:
                        current_role["items"].append(
                            {"item_type": "bullet_point", "content": accomplishment}
                        )
            else:
                # Check if this could be a continuation of an accomplishment
                if current_role and current_role.get("items") and not role_match:
                    # If the line doesn't look like a new role and we have a current role,
                    # it might be a continuation of the previous accomplishment
                    if len(line) > 10 and not line.isupper():  # Avoid headers
                        current_role["items"].append(
                            {"item_type": "bullet_point", "content": line}
                        )

        # Add the last role
        if current_role and (current_role.get("items") or not roles):
            roles.append(current_role)

        # If no roles were found, try to extract any meaningful content
        if not roles:
            # Look for any bullet points or numbered lists in the text
            accomplishments = []
            for line in lines:
                if line.startswith(("•", "-", "*", "◦", "▪", "▫")) or re.match(
                    r"^\s*\d+[.)].", line
                ):
                    accomplishment = re.sub(r"^[•\-*◦▪▫\d+.)\s]+", "", line).strip()
                    if accomplishment and len(accomplishment) > 5:
                        accomplishments.append(
                            {"item_type": "bullet_point", "content": accomplishment}
                        )

            return {
                "name": "Professional Experience",
                "items": (
                    accomplishments
                    if accomplishments
                    else [
                        {
                            "item_type": "bullet_point",
                            "content": "No specific accomplishments extracted",
                        }
                    ]
                ),
            }

        # If only one role, return it directly
        if len(roles) == 1:
            return roles[0]

        # Multiple roles, return as subsections
        return {"name": "Professional Experience", "subsections": roles}

    async def _post_process_content(
        self,
        llm_response: LLMResponse,
        content_type: ContentType,
        context: AgentExecutionContext,
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
        content_item: Dict[str, Any],
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
        words = re.findall(r"\b\w{3,}\b", text.lower())

        # Filter out common words
        stop_words = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "had",
            "her",
            "was",
            "one",
            "our",
            "out",
            "day",
            "get",
            "has",
            "him",
            "his",
            "how",
            "its",
            "may",
            "new",
            "now",
            "old",
            "see",
            "two",
            "who",
            "boy",
            "did",
            "man",
            "men",
            "put",
            "say",
            "she",
            "too",
            "use",
            "will",
            "with",
        }

        return {word for word in words if word not in stop_words and len(word) > 3}

    def _determine_content_type(self, content_item: Dict[str, Any]) -> ContentType:
        """Determine content type from item data."""

        # Ensure content_item is a dictionary before checking keys
        if not isinstance(content_item, dict):
            logger.warning(
                f"Expected dict for content_item, got {type(content_item)}: {content_item}"
            )
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
        self, content_item: Dict[str, Any], content_type: ContentType
    ) -> str:
        """Generate fallback content when LLM fails."""

        # Ensure content_item is a dictionary
        if not isinstance(content_item, dict):
            content_item = {}

        # Safely get values from content_item (now guaranteed to be a dictionary)
        position = content_item.get("position", "previous role")
        project_name = content_item.get("name", "project")

        fallbacks = {
            ContentType.QUALIFICATION: "Strong technical skills and experience relevant to the position.",
            ContentType.EXPERIENCE: f"Valuable experience in {position} contributing to professional growth.",
            ContentType.PROJECT: f"Successful completion of {project_name} demonstrating technical capabilities.",
            ContentType.EXECUTIVE_SUMMARY: "Experienced professional with a strong background in delivering results and contributing to organizational success.",
        }

        return fallbacks.get(
            content_type,
            "Professional experience and qualifications relevant to the position.",
        )

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
                content = "\n".join(
                    [
                        f"• {sentence.strip()}"
                        for sentence in sentences
                        if sentence.strip()
                    ]
                )
        return content

    def _build_single_item_prompt(
        self,
        subsection: Subsection,
        section: Section,
        job_description: Optional[JobDescriptionData],
        research_findings: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a focused prompt for a single subsection/item."""
        try:
            # Get the appropriate template for the content type
            template = self.content_templates.get(
                section.content_type, self._get_fallback_template()
            )

            # Prepare context variables
            context_vars = {
                "job_title": (
                    job_description.job_title if job_description else "Target Position"
                ),
                "company_name": (
                    job_description.company_name
                    if job_description
                    else "Target Company"
                ),
                "job_description": (
                    job_description.main_job_description_raw
                    if job_description
                    else "No job description provided"
                ),
                "item_title": subsection.title,
                "item_description": (
                    subsection.items[0].content
                    if subsection.items
                    else "No existing content"
                ),
                "additional_context": f"This is for the {section.title} section of a CV",
            }

            # Add research findings to context if available
            if research_findings:
                research_context = self._format_research_findings(research_findings)
                context_vars["research_insights"] = research_context
                # Update template to include research insights
                template = template.replace(
                    "{additional_context}",
                    "{additional_context}\n\nResearch Insights:\n{research_insights}",
                )

            # Format the template with context variables
            formatted_prompt = template.format(**context_vars)

            logger.info(
                f"Built single item prompt for {subsection.title} with research insights: {bool(research_findings)}"
            )
            return formatted_prompt

        except Exception as e:
            logger.error(f"Error building single item prompt: {str(e)}")
            return self._get_fallback_template().format(
                job_title="Target Position",
                company_name="Target Company",
                job_description="No job description provided",
                item_title=subsection.title,
                item_description="No existing content",
                additional_context="CV content generation",
            )

    def _format_project_content(self, content: str) -> str:
        """Format project content."""
        # Ensure clear project description
        return content

    def _format_research_findings(self, research_findings: Dict[str, Any]) -> str:
        """Format research findings for inclusion in prompts."""
        if not research_findings:
            return "No research insights available."

        formatted_insights = []

        # Format job requirements analysis
        if "job_requirements" in research_findings:
            job_reqs = research_findings["job_requirements"]
            if job_reqs.get("key_skills"):
                formatted_insights.append(
                    f"Key Skills Required: {', '.join(job_reqs['key_skills'])}"
                )
            if job_reqs.get("experience_level"):
                formatted_insights.append(
                    f"Experience Level: {job_reqs['experience_level']}"
                )

        # Format relevant CV content
        if "relevant_cv_content" in research_findings:
            cv_content = research_findings["relevant_cv_content"]
            if cv_content.get("matching_skills"):
                formatted_insights.append(
                    f"Matching Skills from CV: {', '.join(cv_content['matching_skills'])}"
                )
            if cv_content.get("relevant_experiences"):
                exp_list = [
                    exp.get("title", "Unknown")
                    for exp in cv_content["relevant_experiences"][:3]
                ]
                formatted_insights.append(
                    f"Relevant Experiences: {', '.join(exp_list)}"
                )

        # Format section relevance scores
        if "section_relevance" in research_findings:
            relevance = research_findings["section_relevance"]
            high_relevance = [
                section for section, score in relevance.items() if score > 0.7
            ]
            if high_relevance:
                formatted_insights.append(
                    f"High Relevance Sections: {', '.join(high_relevance)}"
                )

        # Format company information
        if "company_info" in research_findings:
            company = research_findings["company_info"]
            if company.get("industry"):
                formatted_insights.append(f"Company Industry: {company['industry']}")
            if company.get("size"):
                formatted_insights.append(f"Company Size: {company['size']}")

        return (
            "\n".join(formatted_insights)
            if formatted_insights
            else "Research insights available but not formatted."
        )

    def _format_executive_summary_content(self, content: str) -> str:
        """Format executive summary content."""
        # Ensure professional tone and appropriate length
        if len(content) > 300:
            # Truncate if too long
            content = content[:297] + "..."
        return content

    def _extract_json_from_response(self, response_content: str) -> dict:
        """Extract JSON content from LLM response."""
        import re

        # Try to find JSON block markers
        json_match = re.search(r"```json\s*\n(.*?)\n```", response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find content between { and }
            start_idx = response_content.find("{")
            if start_idx == -1:
                raise json.JSONDecodeError(
                    "No JSON object found in response", response_content, 0
                )

            # Find the matching closing brace
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(response_content[start_idx:], start_idx):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

            if brace_count != 0:
                raise json.JSONDecodeError(
                    "Unmatched braces in JSON", response_content, start_idx
                )

            json_str = response_content[start_idx:end_idx]

        # Parse the JSON string
        return json.loads(json_str)

    def _format_role_generation_output(self, validated_data) -> str:
        """Format the validated role generation data into the expected content format."""
        from src.models.validation_schemas import LLMRoleGenerationOutput

        if not isinstance(validated_data, LLMRoleGenerationOutput):
            logger.error(
                f"Expected LLMRoleGenerationOutput, got {type(validated_data)}"
            )
            return "Error: Invalid data format for role generation"

        formatted_content = ""

        for role in validated_data.roles:
            formatted_content += f"## {role.organization_description}\n\n"
            formatted_content += f"**Role:** {role.role_description}\n\n"
            formatted_content += f"**Suggested Resume Bullet Points:**\n\n"

            for skill_bullets in role.suggested_resume_bullet_points:
                formatted_content += f"### {skill_bullets.skill}\n\n"
                for bullet in skill_bullets.bullet_points:
                    formatted_content += f"• {bullet}\n"
                formatted_content += "\n"

            formatted_content += "\n---\n\n"

        return formatted_content.strip()

    def _load_prompt_template(self, prompt_key: str) -> str:
        """Load a prompt template from external file using centralized configuration."""
        try:
            prompt_path = self.settings.get_prompt_path_by_key(prompt_key)
            with open(prompt_path, "r", encoding="utf-8") as f:
                template = f.read()
            logger.info(f"Successfully loaded prompt template: {prompt_key}")
            return template
        except Exception as e:
            logger.error(f"Error loading prompt template {prompt_key}: {e}")
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

    @optimize_async("agent_execution", "enhanced_content_writer")
    async def run_as_node(self, state: AgentState) -> dict:
        """
        Executes the content generation logic as a LangGraph node.
        Processes a single item specified by `current_item_id` in the state.

        Args:
            state: The current state of the workflow.

        Returns:
            A dictionary containing the updated 'structured_cv'.
        """
        logger.info(
            f"EnhancedContentWriterAgent processing item: {state.current_item_id}"
        )

        if not state.current_item_id:
            logger.error("Content writer called without current_item_id")
            error_list = state.error_messages or []
            error_list.append("ContentWriter failed: No item ID.")
            return {"error_messages": error_list}

        try:
            # Process the single item using the async method with research findings
            result = await self._process_single_item(
                state.structured_cv.model_dump(),
                (
                    state.job_description_data.model_dump()
                    if state.job_description_data
                    else {}
                ),
                state.current_item_id,
                state.research_findings,
            )

            if result.success:
                # Extract the updated CV from the result
                updated_cv_data = result.output_data.get("structured_cv")
                if updated_cv_data:
                    # Convert back to StructuredCV model
                    updated_cv = StructuredCV.model_validate(updated_cv_data)
                    logger.info(f"Successfully processed item {state.current_item_id}")
                    return {"structured_cv": updated_cv}

            # If not successful, add error to state
            logger.error(
                f"Failed to process item {state.current_item_id}: {result.error_message}"
            )
            error_list = state.error_messages or []
            error_list.append(f"ContentWriter Error: {result.error_message}")
            return {"error_messages": error_list}

        except Exception as e:
            logger.error(
                f"Exception in Content Writer node for item {state.current_item_id}: {e}",
                exc_info=True,
            )
            error_list = state.error_messages or []
            error_list.append(f"ContentWriter Exception: {str(e)}")
            return {"error_messages": error_list}

    async def generate_big_10_skills(
        self, job_description: str, my_talents: str = ""
    ) -> Dict[str, Any]:
        """
        Generate the "Big 10" skills specifically for Key Qualifications section.
        Returns a structured response with the skills list and raw LLM output.
        """
        try:
            # Load the key qualifications prompt template
            template = self._load_prompt_template("key_qualifications_writer")

            # Format the prompt with job description and talents
            prompt = template.format(
                main_job_description_raw=job_description[:2000],
                my_talents=my_talents
                or "Professional with diverse technical and analytical skills",
            )

            logger.info("Generating Big 10 skills with enhanced content writer")

            # Generate content using LLM
            response = await self.llm_service.generate_content(
                prompt=prompt, content_type=ContentType.QUALIFICATION
            )

            if not response or not response.content or not response.content.strip():
                logger.warning("Empty response from LLM for Big 10 skills generation")
                return {
                    "skills": [],
                    "raw_llm_output": "",
                    "success": False,
                    "error": "Empty response from LLM",
                }

            # Parse the response into individual skills
            skills_list = self._parse_big_10_skills(response.content)

            logger.info(f"Successfully generated {len(skills_list)} skills")

            return {
                "skills": skills_list,
                "raw_llm_output": response.content,
                "success": True,
                "formatted_content": self._format_big_10_skills_display(skills_list),
            }

        except Exception as e:
            logger.error(f"Error generating Big 10 skills: {str(e)}")
            return {
                "skills": [],
                "raw_llm_output": "",
                "success": False,
                "error": str(e),
            }

    def _parse_big_10_skills(self, llm_response: str) -> List[str]:
        """
        Parse the LLM response to extract exactly 10 skills.
        """
        try:
            # Split by newlines and clean up
            lines = [line.strip() for line in llm_response.split("\n") if line.strip()]

            # Filter out template content and system instructions
            filtered_lines = []
            skip_patterns = [
                "[System Instruction]",
                "[Instructions for Skill Generation]",
                "[Job Description]",
                "[Additional Context",
                "[Output Example]",
                "You are an expert",
                "Analyze Job Description",
                "Identify Key Skills",
                "Synthesize and Condense",
                "Format Output",
                'Generate the "Big 10" Skills',
                "Highly relevant to the job",
                "Concise (under 30 characters)",
                "Action-oriented and impactful",
                "Directly aligned with employer",
                "{{main_job_description_raw}}",
                "{{my_talents}}",
            ]

            for line in lines:
                # Skip lines that contain template instructions
                should_skip = False
                for pattern in skip_patterns:
                    if pattern.lower() in line.lower():
                        should_skip = True
                        break

                # Skip lines that are too long (likely instructions)
                if len(line) > 100:
                    should_skip = True

                # Skip lines with brackets (likely template markers)
                if line.startswith("[") and line.endswith("]"):
                    should_skip = True

                if not should_skip:
                    filtered_lines.append(line)

            # Remove any bullet points, numbers, or formatting
            skills = []
            for line in filtered_lines:
                # Remove common prefixes
                cleaned_line = line
                for prefix in [
                    "•",
                    "*",
                    "-",
                    "1.",
                    "2.",
                    "3.",
                    "4.",
                    "5.",
                    "6.",
                    "7.",
                    "8.",
                    "9.",
                    "10.",
                ]:
                    if cleaned_line.startswith(prefix):
                        cleaned_line = cleaned_line[len(prefix) :].strip()
                        break

                # Remove numbering patterns like "1)", "2)", etc.
                import re

                cleaned_line = re.sub(r"^\d+[.):]\s*", "", cleaned_line)

                if cleaned_line and len(cleaned_line) <= 50:  # Reasonable skill length
                    skills.append(cleaned_line)

            # Ensure we have exactly 10 skills
            if len(skills) > 10:
                skills = skills[:10]
            elif len(skills) < 10:
                # Pad with generic skills if needed
                generic_skills = [
                    "Problem Solving",
                    "Team Collaboration",
                    "Communication Skills",
                    "Analytical Thinking",
                    "Project Management",
                    "Technical Documentation",
                    "Quality Assurance",
                    "Process Improvement",
                    "Client Relations",
                    "Strategic Planning",
                ]
                while len(skills) < 10 and generic_skills:
                    skills.append(generic_skills.pop(0))

            return skills[:10]  # Ensure exactly 10

        except Exception as e:
            logger.error(f"Error parsing Big 10 skills: {str(e)}")
            # Return fallback skills
            return [
                "Problem Solving",
                "Team Collaboration",
                "Communication Skills",
                "Analytical Thinking",
                "Project Management",
                "Technical Documentation",
                "Quality Assurance",
                "Process Improvement",
                "Client Relations",
                "Strategic Planning",
            ]

    def _format_big_10_skills_display(self, skills: List[str]) -> str:
        """
        Format the Big 10 skills for display in the CV.
        """
        if not skills:
            return "No skills generated"

        # Format as bullet points
        formatted_skills = "\n".join([f"• {skill}" for skill in skills])
        return formatted_skills

    def _ensure_job_data_structure(self, job_data: dict) -> dict:
        """
        Ensure job_data has all required fields with default values to prevent AttributeError
        """
        default_structure = {
            "title": "",
            "company": "",
            "location": "",
            "experience_level": "",
            "employment_type": "",
            "description": "",
            "requirements": [],
            "responsibilities": [],
            "benefits": [],
            "skills": [],
            "salary_range": "",
            "remote_work": False,
            "industry": "",
            "department": "",
        }

        # Merge provided data with defaults, ensuring all required fields exist
        for key, default_value in default_structure.items():
            if key not in job_data or job_data[key] is None:
                job_data[key] = default_value

        return job_data

    def _convert_string_to_job_data(self, job_string: str) -> dict:
        """
        Convert a string representation to a structured job_data dictionary
        """
        # Basic structure with the string as description
        return self._ensure_job_data_structure(
            {
                "description": job_string,
                "title": "Unknown Position",
                "company": "Unknown Company",
            }
        )

    async def _process_single_item(
        self,
        structured_cv_data: Dict[str, Any],
        job_data: Dict[str, Any],
        item_id: str,
        research_findings: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Process a single subsection item for granular workflow (Task 3.1).

        Args:
            structured_cv_data: The current CV data structure
            job_data: Job description data for context
            item_id: The ID of the specific item to process
            research_findings: Research insights from ResearchAgent

        Returns:
            AgentResult with the updated CV structure
        """

        try:
            # Validate and parse the structured CV data
            if isinstance(structured_cv_data, dict):
                structured_cv = StructuredCV.model_validate(structured_cv_data)
            else:
                structured_cv = structured_cv_data

            # Parse job description data
            job_description = (
                JobDescriptionData.model_validate(job_data) if job_data else None
            )

            # Find the target item by ID using the built-in method
            target_item, target_section, target_subsection = (
                structured_cv.find_item_by_id(str(item_id))
            )

            if not target_item:
                logger.error(f"Could not find item with ID: {item_id}")
                return AgentResult(
                    success=False,
                    error_message=f"Item with ID {item_id} not found",
                    output_data={"structured_cv": structured_cv.model_dump()},
                )

            # Build focused prompt for this specific item with research insights
            prompt = self._build_single_item_prompt(
                target_subsection, target_section, job_description, research_findings
            )

            # Generate content using LLM with fallback handling
            try:
                response: LLMResponse = await self.llm_service.generate_content(prompt)

                # Parse the response and update the specific item
                if response.success and response.content:
                    # Store raw LLM output for transparency (REQ-FUNC-UI-6)
                    raw_output = response.content

                    # Update the target item with new content
                    target_item.content = response.content.strip()
                    target_item.status = ItemStatus.GENERATED
                    target_item.raw_llm_output = raw_output

                    logger.info(f"Successfully processed item: {item_id}")

                    return AgentResult(
                        success=True,
                        output_data={"structured_cv": structured_cv.model_dump()},
                        confidence_score=0.8,
                        metadata={
                            "processed_item_id": item_id,
                            "raw_llm_output": raw_output,
                        },
                    )
                else:
                    # LLM failed, use fallback content generation
                    logger.warning(
                        f"LLM generation failed for item {item_id}, using fallback: {response.error_message}"
                    )
                    fallback_content = self._generate_fallback_content(
                        target_item, target_section, target_subsection, job_description
                    )

                    # Update the target item with fallback content
                    target_item.content = fallback_content
                    target_item.status = ItemStatus.GENERATED_FALLBACK
                    target_item.raw_llm_output = f"LLM_FAILED: {response.error_message}"
                    if not target_item.metadata:
                        target_item.metadata = {}
                    target_item.metadata.update(
                        {"fallback_used": True, "error": response.error_message}
                    )

                    logger.info(f"Applied fallback content for item: {item_id}")

                    return AgentResult(
                        success=True,  # Still successful, just with fallback
                        output_data={"structured_cv": structured_cv.model_dump()},
                        confidence_score=0.3,  # Lower confidence for fallback
                        metadata={
                            "processed_item_id": item_id,
                            "fallback_used": True,
                            "llm_error": response.error_message,
                        },
                    )

            except Exception as llm_error:
                # Handle any LLM service exceptions
                logger.error(f"LLM service exception for item {item_id}: {llm_error}")
                fallback_content = self._generate_fallback_content(
                    target_item, target_section, target_subsection, job_description
                )

                # Update the target item with fallback content
                target_item.content = fallback_content
                target_item.status = ItemStatus.GENERATED_FALLBACK
                target_item.raw_llm_output = f"LLM_EXCEPTION: {str(llm_error)}"
                if not target_item.metadata:
                    target_item.metadata = {}
                target_item.metadata.update(
                    {"fallback_used": True, "error": str(llm_error)}
                )

                logger.info(
                    f"Applied fallback content after LLM exception for item: {item_id}"
                )

                return AgentResult(
                    success=True,  # Still successful, just with fallback
                    output_data={"structured_cv": structured_cv.model_dump()},
                    confidence_score=0.3,  # Lower confidence for fallback
                    metadata={
                        "processed_item_id": item_id,
                        "fallback_used": True,
                        "llm_exception": str(llm_error),
                    },
                )

        except Exception as e:
            logger.error(f"Error processing single item {item_id}: {e}")
            return AgentResult(
                success=False,
                error_message=str(e),
                output_data={"structured_cv": structured_cv_data},
            )

    def _build_experience_prompt_for_subsection(
        self, subsection: Subsection, job_data: Dict[str, Any]
    ) -> str:
        """
        Build a focused prompt for a single subsection/role.

        Args:
            subsection: The subsection to generate content for
            job_data: Job description data for context

        Returns:
            Formatted prompt string
        """
        # Get the experience prompt template
        template = self.content_templates.get(ContentType.EXPERIENCE, "")

        # Extract job information
        job_title = job_data.get("title", "the position")
        job_description = job_data.get("raw_text", job_data.get("description", ""))
        target_skills = job_data.get("skills", [])

        # Format target skills
        if isinstance(target_skills, list):
            target_skills_text = "\n".join([f"- {skill}" for skill in target_skills])
        else:
            target_skills_text = "- No specific skills identified"

        # Format role information from subsection
        role_name = subsection.name
        company = (
            subsection.metadata.get("company", "Unknown Company")
            if subsection.metadata
            else "Unknown Company"
        )
        dates = (
            subsection.metadata.get("dates", "Unknown Dates")
            if subsection.metadata
            else "Unknown Dates"
        )

        # Build role info string
        role_info = f"Role: {role_name}\nCompany: {company}\nDates: {dates}\n"

        # Add existing bullet points if any
        if subsection.items:
            role_info += "\nCurrent bullet points:\n"
            for item in subsection.items:
                role_info += f"- {item.content}\n"

        # Format the template
        prompt = template.replace("{{Target Skills}}", target_skills_text)
        prompt = prompt.replace("{{batched_structured_output}}", role_info)
        prompt = prompt.replace("{{job_title}}", job_title)
        prompt = prompt.replace(
            "{{job_description}}", job_description[:1000]
        )  # Limit length

        return prompt

    def _parse_bullet_points(self, content: str) -> List[str]:
        """
        Parse generated content into individual bullet points.

        Args:
            content: Raw LLM output content

        Returns:
            List of bullet point strings
        """
        lines = content.strip().split("\n")
        bullet_points = []

        for line in lines:
            line = line.strip()
            if line and (
                line.startswith("-") or line.startswith("•") or line.startswith("*")
            ):
                # Remove bullet point markers and clean up
                clean_line = line.lstrip("-•* ").strip()
                if clean_line:
                    bullet_points.append(clean_line)
            elif line and not any(
                marker in line
                for marker in ["Role:", "Company:", "Dates:", "bullet points:"]
            ):
                # Handle lines that might be bullet points without markers
                if len(line) > 10:  # Avoid very short lines that might be headers
                    bullet_points.append(line)

        # Ensure we have at least some content
        if not bullet_points and content.strip():
            bullet_points = [content.strip()]

        return bullet_points[:5]  # Limit to 5 bullet points max

    def _create_error_result(
        self,
        input_data: Dict[str, Any],
        context: AgentExecutionContext,
        error: Exception,
        error_type: str,
    ) -> AgentResult:
        """Create a standardized error result with fallback content."""
        # Return error result with fallback content
        content_item = input_data.get("content_item", {})
        # Ensure content_item is a dictionary for fallback generation
        if not isinstance(content_item, dict):
            content_item = {}
        content_type_fallback = (
            getattr(context, "content_type", None) or ContentType.QUALIFICATION
        )

        # Simple fallback content for error cases
        fallback_content = "⚠️ Content generation failed. Please try again."

        return AgentResult(
            success=False,
            output_data={
                "content": fallback_content,
                "content_type": content_type_fallback.value,
                "confidence_score": 0.1,
                "error": str(error),
                "fallback_used": True,
            },
            confidence_score=0.1,
            error_message=str(error),
            metadata={"fallback_used": True, "error_type": error_type},
        )

    def _generate_fallback_content(
        self,
        target_item: "Item",
        target_section: "Section",
        target_subsection: "Subsection",
        job_description: Optional["JobDescriptionData"],
    ) -> str:
        """
        Generate fallback content when LLM fails.

        Args:
            target_item: The item to generate fallback content for
            target_section: The section containing the item
            target_subsection: The subsection containing the item
            job_description: Job description data for context

        Returns:
            Fallback content string with user-friendly message
        """
        return "⚠️ The LLM did not respond or the content was not correctly generated. Please wait 10 seconds and try to regenerate!"


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
        ContentType.OPTIMIZATION: "OptimizationWriter",
        ContentType.CV_ANALYSIS: "CVAnalysisWriter",
        ContentType.CV_PARSING: "CVParsingWriter",
        ContentType.ACHIEVEMENTS: "AchievementsWriter",
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
        ContentType.OPTIMIZATION: "Specialized agent for content optimization",
        ContentType.CV_ANALYSIS: "Specialized agent for CV analysis and assessment",
        ContentType.CV_PARSING: "Specialized agent for CV parsing and extraction",
        ContentType.ACHIEVEMENTS: "Specialized agent for achievements section content",
    }

    return EnhancedContentWriterAgent(
        name=type_names[content_type],
        description=type_descriptions[content_type],
        content_type=content_type,
    )
