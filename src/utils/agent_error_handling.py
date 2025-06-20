"""Standardized error handling patterns for all agents.

This module provides consistent error handling patterns that all agents should use
to ensure uniform error reporting, logging, and recovery across the system.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING
from functools import wraps
from ..models.validation_schemas import ValidationError
from ..orchestration.state import AgentState

from ..config.logging_config import get_structured_logger

if TYPE_CHECKING:
    from ..agents.agent_base import AgentResult

logger = get_structured_logger(__name__)


class AgentErrorHandler:
    """Centralized error handling for agents."""
    
    @staticmethod
    def handle_validation_error(
        error: ValidationError,
        agent_type: str,
        fallback_data: Optional[Dict[str, Any]] = None
    ) -> "AgentResult":
        """Handle validation errors consistently across agents.
        
        Args:
            error: The validation error that occurred
            agent_type: The type of agent (e.g., 'parser', 'content_writer')
            fallback_data: Optional fallback data to return
            
        Returns:
            AgentResult with error details and fallback data
        """
        # Import at runtime to avoid cyclic import
        from ..agents.agent_base import AgentResult
        
        error_msg = f"Input validation failed for {agent_type}: {str(error)}"
        logger.error(error_msg)
        
        return AgentResult(
            success=False,
            output_data=fallback_data or {},
            confidence_score=0.0,
            error_message=error_msg,
            metadata={
                "agent_type": agent_type,
                "validation_error": True,
                "error_type": "ValidationError"
            },
        )
    
    @staticmethod
    def handle_general_error(
        error: Exception,
        agent_type: str,
        fallback_data: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None
    ) -> "AgentResult":
        """Handle general exceptions consistently across agents.
        
        Args:
            error: The exception that occurred
            agent_type: The type of agent (e.g., 'parser', 'content_writer')
            fallback_data: Optional fallback data to return
            context: Optional context about where the error occurred
            
        Returns:
            AgentResult with error details and fallback data
        """
        # Import at runtime to avoid cyclic import
        from ..agents.agent_base import AgentResult
        
        context_msg = f" in {context}" if context else ""
        error_msg = f"{agent_type} error{context_msg}: {str(error)}"
        logger.error(error_msg, exc_info=True)
        
        return AgentResult(
            success=False,
            output_data=fallback_data or {},
            confidence_score=0.0,
            error_message=str(error),
            metadata={
                "agent_type": agent_type,
                "error_type": type(error).__name__,
                "context": context
            },
        )
    
    @staticmethod
    def handle_node_error(
        error: Exception,
        agent_type: str,
        state: AgentState,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle errors in LangGraph node execution.
        
        Args:
            error: The exception that occurred
            agent_type: The type of agent
            state: The current agent state
            context: Optional context about where the error occurred
            
        Returns:
            Dictionary with error_messages for LangGraph state
        """
        context_msg = f" in {context}" if context else ""
        error_msg = f"{agent_type} Error{context_msg}: {str(error)}"
        logger.error(error_msg, exc_info=True)
        
        error_list = state.error_messages or []
        error_list.append(error_msg)
        return {"error_messages": error_list}
    
    @staticmethod
    def create_fallback_data(agent_type: str) -> Dict[str, Any]:
        """Create appropriate fallback data based on agent type.
        
        Args:
            agent_type: The type of agent
            
        Returns:
            Dictionary with appropriate fallback structure
        """
        fallback_structures = {
            "parser": {
                "job_description_data": {
                    "error": "Parsing failed",
                    "status": "GENERATION_FAILED",
                    "skills": [],
                    "responsibilities": [],
                    "experience_level": "Unknown"
                },
                "structured_cv": None
            },
            "content_writer": {
                "updated_item": {
                    "error": "Content generation failed",
                    "status": "GENERATION_FAILED",
                    "content": "[Content generation failed]"
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
            }
        }
        
        return fallback_structures.get(agent_type, {"error": f"{agent_type} failed"})


def with_error_handling(agent_type: str, context: Optional[str] = None):
    """Decorator to add standardized error handling to agent methods.
    
    Args:
        agent_type: The type of agent (e.g., 'parser', 'content_writer')
        context: Optional context about the method being decorated
        
    Returns:
        Decorated function with error handling
    """
    from .decorators import create_async_sync_decorator
    
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
    """Decorator to add standardized error handling to LangGraph node methods.
    
    Args:
        agent_type: The type of agent
        context: Optional context about the method being decorated
        
    Returns:
        Decorated function with node error handling
    """
    from .decorators import create_async_sync_decorator
    
    def create_async_wrapper(func):
        @wraps(func)
        async def async_wrapper(self, state: AgentState, *args, **kwargs):
            try:
                return await func(self, state, *args, **kwargs)
            except Exception as e:
                return AgentErrorHandler.handle_node_error(
                    e, agent_type, state, context
                )
        return async_wrapper
    
    def create_sync_wrapper(func):
        @wraps(func)
        def sync_wrapper(self, state: AgentState, *args, **kwargs):
            try:
                return func(self, state, *args, **kwargs)
            except Exception as e:
                return AgentErrorHandler.handle_node_error(
                    e, agent_type, state, context
                )
        return sync_wrapper
    
    return create_async_sync_decorator(create_async_wrapper, create_sync_wrapper)


class LLMErrorHandler:
    """Specialized error handling for LLM-related operations."""
    
    @staticmethod
    def handle_llm_response_error(
        error: Exception,
        agent_type: str,
        fallback_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle LLM response parsing errors.
        
        Args:
            error: The error that occurred
            agent_type: The type of agent
            fallback_content: Optional fallback content
            
        Returns:
            Dictionary with error information and fallback content
        """
        logger.error(f"LLM response error in {agent_type}: {str(error)}")
        
        return {
            "error": str(error),
            "content": fallback_content or "[Content generation failed]",
            "status": "GENERATION_FAILED",
            "metadata": {
                "error_type": type(error).__name__,
                "agent_type": agent_type
            }
        }
    
    @staticmethod
    def handle_json_parsing_error(
        error: Exception,
        raw_response: str,
        agent_type: str
    ) -> Dict[str, Any]:
        """Handle JSON parsing errors from LLM responses.
        
        Args:
            error: The JSON parsing error
            raw_response: The raw LLM response that failed to parse
            agent_type: The type of agent
            
        Returns:
            Dictionary with error information and raw response
        """
        logger.error(
            f"JSON parsing error in {agent_type}: {str(error)}. "
            f"Raw response: {raw_response[:200]}..."
        )
        
        return {
            "error": f"JSON parsing failed: {str(error)}",
            "raw_response": raw_response,
            "status": "PARSING_FAILED",
            "metadata": {
                "error_type": "JSONDecodeError",
                "agent_type": agent_type
            }
        }