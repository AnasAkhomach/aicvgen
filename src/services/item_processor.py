"""Individual item processor for CV generation.

This module handles the processing of individual CV items (qualifications, experiences, projects)
with rate limiting and retry logic to mitigate LLM API limits.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import asdict

from ..config.logging_config import get_structured_logger, LLMCallLog
from ..config.settings import get_config
from ..models.data_models import (
    Item, ProcessingStatus, ContentType, CVGenerationState,
    ExperienceItem, ProjectItem, QualificationItem
)
from ..services.rate_limiter import get_rate_limiter, RateLimitExceeded, APIError


class ItemProcessor:
    """Processor for individual CV content items."""

    def __init__(self, llm_client=None, qa_callback=None):
        self.llm_client = llm_client
        self.rate_limiter = get_rate_limiter()
        self.logger = get_structured_logger("item_processor")
        self.settings = get_config()
        self.qa_callback = qa_callback  # Callback for quality assurance

        # Processing statistics
        self.total_processed = 0
        self.total_failed = 0
        self.total_rate_limited = 0
        self.processing_start_time = None

    async def process_item(
        self,
        item: Item,
        job_context: Dict[str, Any],
        template_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Process a single content item.

        Args:
            item: The content item to process
            job_context: Job description context for tailoring
            template_context: Additional template context

        Returns:
            bool: True if processing succeeded, False otherwise
        """
        if item.metadata.status != ProcessingStatus.PENDING:
            self.logger.warning(
                f"Item {item.metadata.item_id} is not in pending status",
                current_status=item.metadata.status.value
            )
            return False

        # Update status to in progress
        item.metadata.update_status(ProcessingStatus.IN_PROGRESS)
        start_time = time.time()

        try:
            # Generate content based on item type
            generated_content = await self._generate_content_for_item(
                item, job_context, template_context
            )

            # Update item with generated content
            item.generated_content = generated_content
            item.metadata.update_status(ProcessingStatus.COMPLETED)
            item.metadata.processing_time_seconds = time.time() - start_time

            # Run quality assurance if callback is provided
            if self.qa_callback:
                try:
                    await self.qa_callback(item, job_context)
                except Exception as qa_error:
                    self.logger.warning(
                        f"QA callback failed for item {item.metadata.item_id}: {qa_error}"
                    )

            self.total_processed += 1

            self.logger.info(
                f"Successfully processed item {item.metadata.item_id}",
                item_type=item.content_type.value,
                processing_time=item.metadata.processing_time_seconds,
                tokens_used=item.metadata.tokens_used
            )

            return True

        except RateLimitExceeded as e:
            item.metadata.update_status(ProcessingStatus.RATE_LIMITED, str(e))
            item.metadata.processing_time_seconds = time.time() - start_time
            self.total_rate_limited += 1

            self.logger.warning(
                f"Rate limit hit for item {item.metadata.item_id}",
                model=e.model,
                retry_after=e.retry_after,
                attempt=item.metadata.processing_attempts
            )

            return False

        except Exception as e:
            item.metadata.update_status(ProcessingStatus.FAILED, str(e))
            item.metadata.processing_time_seconds = time.time() - start_time
            self.total_failed += 1

            self.logger.error(
                f"Failed to process item {item.metadata.item_id}",
                error=str(e),
                item_type=item.content_type.value,
                attempt=item.metadata.processing_attempts
            )

            return False

    async def _generate_content_for_item(
        self,
        item: Item,
        job_context: Dict[str, Any],
        template_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate content for a specific item type."""

        if item.content_type == ContentType.QUALIFICATION:
            return await self._generate_qualification_content(item, job_context)
        elif item.content_type == ContentType.EXPERIENCE_ITEM:
            return await self._generate_experience_content(item, job_context)
        elif item.content_type == ContentType.PROJECT_ITEM:
            return await self._generate_project_content(item, job_context)
        elif item.content_type == ContentType.EXECUTIVE_SUMMARY:
            return await self._generate_summary_content(item, job_context, template_context)
        else:
            raise ValueError(f"Unsupported content type: {item.content_type}")

    async def _generate_qualification_content(
        self,
        item: QualificationItem,
        job_context: Dict[str, Any]
    ) -> str:
        """Generate tailored qualification content."""

        prompt = self._build_qualification_prompt(item, job_context)
        model = self.settings.llm.primary_model

        # Estimate tokens (rough approximation)
        estimated_tokens = len(prompt) // 4 + 200

        response = await self._make_llm_call(
            prompt=prompt,
            model=model,
            estimated_tokens=estimated_tokens,
            item_id=item.metadata.item_id,
            prompt_type="qualification_generation"
        )

        return self._extract_content_from_response(response)

    async def _generate_experience_content(
        self,
        item: ExperienceItem,
        job_context: Dict[str, Any]
    ) -> str:
        """Generate tailored experience content."""

        prompt = self._build_experience_prompt(item, job_context)
        model = self.settings.llm.primary_model

        estimated_tokens = len(prompt) // 4 + 300

        response = await self._make_llm_call(
            prompt=prompt,
            model=model,
            estimated_tokens=estimated_tokens,
            item_id=item.metadata.item_id,
            prompt_type="experience_tailoring"
        )

        return self._extract_content_from_response(response)

    async def _generate_project_content(
        self,
        item: ProjectItem,
        job_context: Dict[str, Any]
    ) -> str:
        """Generate tailored project content."""

        prompt = self._build_project_prompt(item, job_context)
        model = self.settings.llm.primary_model

        estimated_tokens = len(prompt) // 4 + 250

        response = await self._make_llm_call(
            prompt=prompt,
            model=model,
            estimated_tokens=estimated_tokens,
            item_id=item.metadata.item_id,
            prompt_type="project_tailoring"
        )

        return self._extract_content_from_response(response)

    async def _generate_summary_content(
        self,
        item: Item,
        job_context: Dict[str, Any],
        template_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate executive summary content."""

        prompt = self._build_summary_prompt(item, job_context, template_context)
        model = self.settings.llm.secondary_model or self.settings.llm.primary_model

        estimated_tokens = len(prompt) // 4 + 400

        response = await self._make_llm_call(
            prompt=prompt,
            model=model,
            estimated_tokens=estimated_tokens,
            item_id=item.metadata.item_id,
            prompt_type="executive_summary"
        )

        return self._extract_content_from_response(response)

    async def _make_llm_call(
        self,
        prompt: str,
        model: str,
        estimated_tokens: int,
        item_id: str,
        prompt_type: str
    ) -> Any:
        """Make a rate-limited LLM API call with logging."""

        start_time = time.time()

        try:
            # Use rate limiter to make the call
            response = await self.rate_limiter.execute_with_retry(
                self._call_llm_api,
                model=model,
                estimated_tokens=estimated_tokens,
                prompt=prompt,
                model_name=model
            )

            # Calculate actual metrics
            duration = time.time() - start_time
            actual_tokens = self._extract_token_usage(response)

            # Update item metadata
            item_metadata = None
            # Note: We'd need to pass item reference or find another way to update metadata

            # Log the successful call
            call_log = LLMCallLog(
                timestamp=datetime.now().isoformat(),
                model=model,
                prompt_type=prompt_type,
                input_tokens=estimated_tokens // 2,  # Rough split
                output_tokens=actual_tokens - (estimated_tokens // 2),
                total_tokens=actual_tokens,
                duration_seconds=duration,
                success=True,
                session_id=item_id  # Using item_id as session reference
            )

            self.logger.log_llm_call(call_log)

            return response

        except Exception as e:
            duration = time.time() - start_time

            # Log the failed call
            call_log = LLMCallLog(
                timestamp=datetime.now().isoformat(),
                model=model,
                prompt_type=prompt_type,
                input_tokens=estimated_tokens // 2,
                output_tokens=0,
                total_tokens=estimated_tokens // 2,
                duration_seconds=duration,
                success=False,
                error_message=str(e),
                rate_limit_hit=isinstance(e, RateLimitExceeded),
                session_id=item_id
            )

            self.logger.log_llm_call(call_log)
            raise

    async def _call_llm_api(self, prompt: str, model_name: str) -> Any:
        """Make the actual LLM API call."""
        if not self.llm_client:
            raise APIError("No LLM client configured")

        # This would be implemented based on the specific LLM client
        # For now, we'll create a placeholder that works with common interfaces

        try:
            # Try different client interfaces
            if hasattr(self.llm_client, 'chat'):
                # OpenAI-style interface
                response = await self.llm_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                return response

            elif hasattr(self.llm_client, 'generate'):
                # Generic generate interface
                response = await self.llm_client.generate(
                    prompt=prompt,
                    model=model_name,
                    temperature=0.7,
                    max_tokens=1000
                )
                return response

            else:
                # Fallback: assume callable
                response = await self.llm_client(
                    prompt=prompt,
                    model=model_name
                )
                return response

        except Exception as e:
            raise APIError(f"LLM API call failed: {str(e)}") from e

    def _extract_content_from_response(self, response: Any) -> str:
        """Extract content from LLM response."""
        try:
            # Handle different response formats
            if hasattr(response, 'choices') and response.choices:
                # OpenAI-style response
                if hasattr(response.choices[0], 'message'):
                    return response.choices[0].message.content
                elif hasattr(response.choices[0], 'text'):
                    return response.choices[0].text

            elif hasattr(response, 'content'):
                return response.content

            elif hasattr(response, 'text'):
                return response.text

            elif isinstance(response, dict):
                # Dictionary response
                if 'content' in response:
                    return response['content']
                elif 'text' in response:
                    return response['text']
                elif 'choices' in response and response['choices']:
                    choice = response['choices'][0]
                    if 'message' in choice:
                        return choice['message'].get('content', '')
                    elif 'text' in choice:
                        return choice['text']

            elif isinstance(response, str):
                return response

            # Fallback
            return str(response)

        except Exception as e:
            self.logger.error(f"Failed to extract content from response: {e}")
            return ""

    def _extract_token_usage(self, response: Any) -> int:
        """Extract token usage from LLM response."""
        try:
            if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                return response.usage.total_tokens
            elif isinstance(response, dict) and 'usage' in response:
                return response['usage'].get('total_tokens', 0)
            else:
                # Fallback estimation
                content = self._extract_content_from_response(response)
                return len(content) // 4  # Rough estimation
        except:
            return 0

    def _build_qualification_prompt(self, item: QualificationItem, job_context: Dict[str, Any]) -> str:
        """Build prompt for qualification generation."""
        job_skills = job_context.get('required_skills', []) + job_context.get('preferred_skills', [])

        return f"""You are an expert CV writer. Generate a compelling, specific qualification statement that demonstrates expertise relevant to this job.

Job Context:
- Position: {job_context.get('position_title', 'N/A')}
- Company: {job_context.get('company_name', 'N/A')}
- Key Skills Needed: {', '.join(job_skills[:10])}
- Key Responsibilities: {', '.join(job_context.get('responsibilities', [])[:5])}

Original Qualification: {item.original_content}

Instructions:
1. Tailor the qualification to match the job requirements
2. Use specific, measurable language when possible
3. Highlight relevant technologies, methodologies, or achievements
4. Keep it concise (1-2 sentences)
5. Make it compelling and professional

Generate only the tailored qualification statement:"""

    def _build_experience_prompt(self, item: ExperienceItem, job_context: Dict[str, Any]) -> str:
        """Build prompt for experience tailoring."""
        job_skills = job_context.get('required_skills', []) + job_context.get('preferred_skills', [])

        return f"""You are an expert CV writer. Tailor this professional experience to highlight relevance for the target job.

Job Context:
- Position: {job_context.get('position_title', 'N/A')}
- Company: {job_context.get('company_name', 'N/A')}
- Key Skills Needed: {', '.join(job_skills[:10])}
- Key Responsibilities: {', '.join(job_context.get('responsibilities', [])[:5])}

Experience to Tailor:
- Company: {item.company}
- Position: {item.position}
- Duration: {item.duration}
- Original Description: {item.original_content}
- Responsibilities: {', '.join(item.responsibilities)}
- Achievements: {', '.join(item.achievements)}
- Technologies: {', '.join(item.technologies)}

Instructions:
1. Emphasize aspects most relevant to the target job
2. Quantify achievements where possible
3. Highlight transferable skills and technologies
4. Use action verbs and professional language
5. Keep it concise but impactful

Generate the tailored experience description:"""

    def _build_project_prompt(self, item: ProjectItem, job_context: Dict[str, Any]) -> str:
        """Build prompt for project tailoring."""
        job_skills = job_context.get('required_skills', []) + job_context.get('preferred_skills', [])

        return f"""You are an expert CV writer. Tailor this side project to demonstrate skills relevant to the target job.

Job Context:
- Position: {job_context.get('position_title', 'N/A')}
- Company: {job_context.get('company_name', 'N/A')}
- Key Skills Needed: {', '.join(job_skills[:10])}
- Key Responsibilities: {', '.join(job_context.get('responsibilities', [])[:5])}

Project to Tailor:
- Name: {item.name}
- Original Description: {item.original_content}
- Technologies: {', '.join(item.technologies)}
- Achievements: {', '.join(item.achievements)}
- URL: {item.url or 'N/A'}

Instructions:
1. Highlight technologies and skills that match the job requirements
2. Emphasize problem-solving and technical achievements
3. Show initiative and learning ability
4. Quantify impact where possible
5. Keep it professional and concise

Generate the tailored project description:"""

    def _build_summary_prompt(
        self,
        item: Item,
        job_context: Dict[str, Any],
        template_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for executive summary generation."""

        qualifications = template_context.get('qualifications', []) if template_context else []
        experiences = template_context.get('experiences', []) if template_context else []
        projects = template_context.get('projects', []) if template_context else []

        return f"""You are an expert CV writer. Create a compelling executive summary that positions the candidate perfectly for this job.

Job Context:
- Position: {job_context.get('position_title', 'N/A')}
- Company: {job_context.get('company_name', 'N/A')}
- Key Skills Needed: {', '.join(job_context.get('required_skills', [])[:10])}
- Key Responsibilities: {', '.join(job_context.get('responsibilities', [])[:5])}

Candidate Profile:
- Key Qualifications: {', '.join([q[:100] for q in qualifications[:5]])}
- Recent Experience: {', '.join([exp[:100] for exp in experiences[:3]])}
- Notable Projects: {', '.join([proj[:100] for proj in projects[:3]])}

Instructions:
1. Create a 3-4 sentence executive summary
2. Lead with the most relevant qualifications for this specific job
3. Highlight unique value proposition
4. Use confident, professional language
5. Avoid generic statements
6. Make it compelling and memorable

Generate only the executive summary:"""

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total_attempts = self.total_processed + self.total_failed + self.total_rate_limited

        return {
            "total_processed": self.total_processed,
            "total_failed": self.total_failed,
            "total_rate_limited": self.total_rate_limited,
            "total_attempts": total_attempts,
            "success_rate": (self.total_processed / total_attempts * 100) if total_attempts > 0 else 0,
            "rate_limit_rate": (self.total_rate_limited / total_attempts * 100) if total_attempts > 0 else 0,
            "processing_start_time": self.processing_start_time.isoformat() if self.processing_start_time else None
        }

    def reset_stats(self):
        """Reset processing statistics."""
        self.total_processed = 0
        self.total_failed = 0
        self.total_rate_limited = 0
        self.processing_start_time = datetime.now()