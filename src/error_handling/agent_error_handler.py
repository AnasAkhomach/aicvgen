"""Standardized error handling patterns for all agents.

This module provides consistent error handling patterns that all agents should use
to ensure uniform error reporting, logging, and recovery across the system.
"""

from functools import wraps
from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import BaseModel

from src.config.logging_config import get_structured_logger
from src.error_handling.boundaries import CATCHABLE_EXCEPTIONS
from src.error_handling.models import ErrorCategory, ErrorContext, ErrorSeverity

from src.error_handling.exceptions import ValidationError
from src.orchestration.state import GlobalState

from src.utils.decorators import create_async_sync_decorator


logger = get_structured_logger(__name__)


class AgentErrorHandler:
    """Centralized error handling for agents."""

    @staticmethod
    def handle_error(
        message: str,
        category: "ErrorCategory",
        severity: "ErrorSeverity",
        context: Optional["ErrorContext"] = None,
    ):
        """Handles a generic system error, distinct from agent-specific errors."""
        log_extra = {
            "error_category": category.value,
            "error_severity": severity.value,
        }
        if context and context.additional_data:
            log_extra.update(context.additional_data)

        logger.error(message, extra=log_extra)

    @staticmethod
    def handle_validation_error(
        error: ValidationError, agent_type: str, fallback_data: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Handle validation errors consistently across agents."""
        error_msg = f"Input validation failed for {agent_type}: {str(error)}"
        logger.error(error_msg)

        if fallback_data is None:
            fallback_data = {"error": f"{agent_type} failed"}
        elif isinstance(fallback_data, BaseModel):
            fallback_data = fallback_data.model_dump()
        elif not isinstance(fallback_data, dict):
            fallback_data = {"error": str(fallback_data)}

        return {
            "error": error_msg,
            "success": False,
            "fallback_data": fallback_data,
            "metadata": {
                "agent_type": agent_type,
                "validation_error": True,
                "error_type": "ValidationError",
            },
        }

    @staticmethod
    def handle_general_error(
        error: Exception,
        agent_type: str,
        fallback_data: Optional[Any] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle general exceptions consistently across agents."""
        context_msg = f" in {context}" if context else ""
        error_msg = f"{agent_type} error{context_msg}: {str(error)}"
        logger.error(error_msg, exc_info=True)

        if fallback_data is None:
            fallback_data = {"error": f"{agent_type} failed"}
        elif isinstance(fallback_data, BaseModel):
            fallback_data = fallback_data.model_dump()
        elif not isinstance(fallback_data, dict):
            fallback_data = {"error": str(fallback_data)}

        return {
            "error": str(error),
            "success": False,
            "fallback_data": fallback_data,
            "metadata": {
                "agent_type": agent_type,
                "error_type": type(error).__name__,
                "context": context,
            },
        }

    @staticmethod
    def handle_node_error(
        error: Exception,
        agent_type: str,
        state: GlobalState,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle errors in LangGraph node execution."""
        context_msg = f" in {context}" if context else ""
        error_msg = f"{agent_type} Error{context_msg}: {str(error)}"
        logger.error(error_msg, exc_info=True)

        error_list = (
            list(state.get("error_messages", [])) if state.get("error_messages") else []
        )
        error_list.append(error_msg)
        return {"error_messages": error_list}

    @staticmethod
    def create_fallback_data(agent_type: str) -> Dict[str, Any]:
        """Create appropriate fallback data based on agent type."""
        fallback_structures = {
            "parser": {
                "job_description_data": {
                    "error": "Parsing failed",
                    "status": "GENERATION_FAILED",
                    "skills": [],
                    "responsibilities": [],
                    "experience_level": "Unknown",
                },
                "structured_cv": None,
            },
            "content_writer": {
                "updated_item": {
                    "error": "Content generation failed",
                    "status": "GENERATION_FAILED",
                    "content": "[Content generation failed]",
                }
            },
            "research": {
                "research_results": {
                    "error": "Research failed",
                    "company_info": {},
                    "industry_trends": [],
                    "role_insights": {},
                    "skill_requirements": [],
                    "market_data": {},
                },
                "enhanced_job_description": None,
            },
            "quality_assurance": {
                "quality_check_results": {
                    "error": "Quality check failed",
                    "item_checks": [],
                    "section_checks": [],
                    "overall_checks": [],
                    "summary": {
                        "total_items": 0,
                        "passed_items": 0,
                        "warning_items": 0,
                        "failed_items": 0,
                    },
                },
                "updated_structured_cv": None,
            },
        }

        return fallback_structures.get(agent_type, {"error": f"{agent_type} failed"})


def with_error_handling(agent_type: str, context: Optional[str] = None):
    """Decorator to add standardized error handling to agent methods."""

    def create_async_wrapper(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except ValidationError as ve:
                fallback_data = AgentErrorHandler.create_fallback_data(agent_type)
                return AgentErrorHandler.handle_validation_error(
                    ve, agent_type, fallback_data
                )
            except CATCHABLE_EXCEPTIONS as e:
                fallback_data = AgentErrorHandler.create_fallback_data(agent_type)
                return AgentErrorHandler.handle_general_error(
                    e, agent_type, fallback_data, context
                )

        return async_wrapper

    def create_sync_wrapper(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ValidationError as ve:
                fallback_data = AgentErrorHandler.create_fallback_data(agent_type)
                return AgentErrorHandler.handle_validation_error(
                    ve, agent_type, fallback_data
                )
            except CATCHABLE_EXCEPTIONS as e:
                fallback_data = AgentErrorHandler.create_fallback_data(agent_type)
                return AgentErrorHandler.handle_general_error(
                    e, agent_type, fallback_data, context
                )

        return sync_wrapper

    return create_async_sync_decorator(create_async_wrapper, create_sync_wrapper)


def with_node_error_handling(agent_type: str, context: Optional[str] = None):
    """Decorator to add standardized error handling to LangGraph node methods."""

    def create_async_wrapper(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except CATCHABLE_EXCEPTIONS as e:
                state = args[0] if args and isinstance(args[0], dict) else {}
                return AgentErrorHandler.handle_node_error(
                    e, agent_type, state, context
                )

        return async_wrapper

    def create_sync_wrapper(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except CATCHABLE_EXCEPTIONS as e:
                state = args[0] if args and isinstance(args[0], dict) else {}
                return AgentErrorHandler.handle_node_error(
                    e, agent_type, state, context
                )

        return sync_wrapper

    return create_async_sync_decorator(create_async_wrapper, create_sync_wrapper)


class LLMErrorHandler:
    """Specialized error handling for LLM-related operations."""

    @staticmethod
    def handle_llm_response_error(
        error: Exception, agent_type: str, fallback_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle LLM response parsing errors."""
        logger.error("LLM response error in %s: %s", agent_type, str(error))

        return {
            "error": str(error),
            "content": fallback_content or "[Content generation failed]",
            "status": "GENERATION_FAILED",
            "metadata": {"error_type": type(error).__name__, "agent_type": agent_type},
        }

    @staticmethod
    def handle_json_parsing_error(
        error: Exception, raw_response: str, agent_type: str
    ) -> Dict[str, Any]:
        """Handle JSON parsing errors from LLM responses."""
        logger.error(
            f"JSON parsing error in {agent_type}: {str(error)}. "
            f"Raw response: {raw_response[:200]}..."
        )

        return {
            "error": f"JSON parsing failed: {str(error)}",
            "raw_response": raw_response,
            "status": "PARSING_FAILED",
            "metadata": {"error_type": "JSONDecodeError", "agent_type": agent_type},
        }
