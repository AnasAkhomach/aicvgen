"""Standardized error handling patterns for all agents.

This module provides consistent error handling patterns that all agents should use
to ensure uniform error reporting, logging, and recovery across the system.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING
from functools import wraps
from pydantic import BaseModel

from src.models.validation_schemas import ValidationError
from src.orchestration.state import AgentState
from src.models.data_models import ErrorFallbackModel
from src.config.logging_config import get_structured_logger

if TYPE_CHECKING:
    from src.agents.agent_base import AgentResult

logger = get_structured_logger(__name__)


class AgentErrorHandler:
    """Centralized error handling for agents."""

    @staticmethod
    def handle_validation_error(
        error: ValidationError, agent_type: str, fallback_data: Optional[Any] = None
    ) -> "AgentResult":
        """Handle validation errors consistently across agents."""
        from src.agents.agent_base import AgentResult

        error_msg = f"Input validation failed for {agent_type}: {str(error)}"
        logger.error(error_msg)

        if fallback_data is None:
            output_data = ErrorFallbackModel(error=f"{agent_type} failed")
        elif isinstance(fallback_data, BaseModel):
            output_data = fallback_data
        elif isinstance(fallback_data, dict):
            output_data = ErrorFallbackModel(**fallback_data)
        else:
            output_data = ErrorFallbackModel(error=str(fallback_data))

        return AgentResult(
            success=False,
            output_data=output_data,
            confidence_score=0.0,
            error_message=error_msg,
            metadata={
                "agent_type": agent_type,
                "validation_error": True,
                "error_type": "ValidationError",
            },
        )

    @staticmethod
    def handle_general_error(
        error: Exception,
        agent_type: str,
        fallback_data: Optional[Any] = None,
        context: Optional[str] = None,
    ) -> "AgentResult":
        """Handle general exceptions consistently across agents."""
        from src.agents.agent_base import AgentResult

        context_msg = f" in {context}" if context else ""
        error_msg = f"{agent_type} error{context_msg}: {str(error)}"
        logger.error(error_msg, exc_info=True)

        if fallback_data is None:
            output_data = ErrorFallbackModel(error=f"{agent_type} failed")
        elif isinstance(fallback_data, BaseModel):
            output_data = fallback_data
        elif isinstance(fallback_data, dict):
            if "error" not in fallback_data:
                output_data = ErrorFallbackModel(error=str(fallback_data))
            else:
                output_data = ErrorFallbackModel(**fallback_data)
        else:
            output_data = ErrorFallbackModel(error=str(fallback_data))

        return AgentResult(
            success=False,
            output_data=output_data,
            confidence_score=0.0,
            error_message=str(error),
            metadata={
                "agent_type": agent_type,
                "error_type": type(error).__name__,
                "context": context,
            },
        )

    @staticmethod
    def handle_node_error(
        error: Exception,
        agent_type: str,
        state: AgentState,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle errors in LangGraph node execution."""
        context_msg = f" in {context}" if context else ""
        error_msg = f"{agent_type} Error{context_msg}: {str(error)}"
        logger.error(error_msg, exc_info=True)

        error_list = state.get("error_messages", []) or []
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
    from src.utils.decorators import create_async_sync_decorator

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
            except Exception as e:
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
            except Exception as e:
                fallback_data = AgentErrorHandler.create_fallback_data(agent_type)
                return AgentErrorHandler.handle_general_error(
                    e, agent_type, fallback_data, context
                )

        return sync_wrapper

    return create_async_sync_decorator(create_async_wrapper, create_sync_wrapper)


def with_node_error_handling(agent_type: str, context: Optional[str] = None):
    """Decorator to add standardized error handling to LangGraph node methods."""
    from src.utils.decorators import create_async_sync_decorator

    def create_async_wrapper(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
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
            except Exception as e:
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
        logger.error(f"LLM response error in {agent_type}: {str(error)}")

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
