# Redundancy Elimination Implementation Plan

## Overview
This document outlines the concrete implementation plan to eliminate the identified redundancies in the codebase.

## 1. Agent Execute Logic Consolidation

### Problem
- 13 agents have nearly identical `_execute` method patterns
- Duplicated input validation, progress tracking, content generation, and error handling

### Solution: Create Abstract Base Agent

```python
# src/agents/base/abstract_agent_executor.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from src.core.validation.validator_factory import validate_agent_input
from src.error_handling.exceptions import AgentExecutionError

class AbstractAgentExecutor(ABC):
    """Abstract base class for agent execution with common patterns."""
    
    async def _execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Template method implementing common execution pattern."""
        try:
            # 1. Input validation
            validated_input = self._validate_input(**kwargs)
            
            # 2. Progress tracking
            self.update_progress("main_processing")
            
            # 3. Content generation (delegated to subclass)
            content = await self._generate_content(validated_input)
            
            # 4. CV update logic
            updated_cv = self._update_structured_cv(content, validated_input)
            
            # 5. Final progress update
            self.update_progress("complete")
            
            return {
                "success": True,
                "content": content,
                "structured_cv": updated_cv
            }
            
        except Exception as e:
            return self._handle_execution_error(e)
    
    @abstractmethod
    async def _generate_content(self, validated_input: Dict[str, Any]) -> str:
        """Generate agent-specific content."""
        pass
    
    @abstractmethod
    def _update_structured_cv(self, content: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update structured CV with generated content."""
        pass
    
    def _validate_input(self, **kwargs: Any) -> Dict[str, Any]:
        """Validate input using agent-specific schema."""
        return validate_agent_input(self.__class__.__name__, kwargs)
    
    def _handle_execution_error(self, error: Exception) -> Dict[str, Any]:
        """Standardized error handling."""
        error_msg = f"{self.__class__.__name__} execution failed: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg,
            "error_type": type(error).__name__
        }
```

### Implementation Steps
1. Create `AbstractAgentExecutor` base class
2. Refactor each agent to inherit from base class
3. Move agent-specific logic to abstract methods
4. Remove duplicated code from individual agents

## 2. LLM Interaction Pattern Consolidation

### Problem
- 8+ services have similar LLM interaction patterns
- Duplicated `generate_content` calls and response processing

### Solution: Create LLM Interaction Mixin

```python
# src/services/mixins/llm_interaction_mixin.py
from typing import Any, Dict, Optional
from src.services.llm_service_interface import LLMServiceInterface
from src.templates.content_templates import ContentTemplateManager

class LLMInteractionMixin:
    """Mixin providing standardized LLM interaction patterns."""
    
    llm_service: LLMServiceInterface
    template_manager: ContentTemplateManager
    
    async def generate_templated_content(
        self,
        template_name: str,
        template_data: Dict[str, Any],
        content_type: str = "text",
        **llm_kwargs
    ) -> str:
        """Generate content using template and LLM."""
        # Format template
        formatted_prompt = self.template_manager.format_template(
            template_name, template_data
        )
        
        # Generate content
        response = await self.llm_service.generate_content(
            prompt=formatted_prompt,
            content_type=content_type,
            **llm_kwargs
        )
        
        return response
    
    async def generate_and_parse_json(
        self,
        template_name: str,
        template_data: Dict[str, Any],
        **llm_kwargs
    ) -> Dict[str, Any]:
        """Generate JSON content and parse it."""
        raw_response = await self.generate_templated_content(
            template_name, template_data, content_type="json", **llm_kwargs
        )
        
        return self._extract_json_from_response(raw_response)
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response with robust parsing."""
        # Implementation from llm_cv_parser_service.py
        import json
        import re
        
        # Try direct JSON parsing first
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # Extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Extract JSON from response text
        json_match = re.search(r'{.*}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        raise LLMResponseParsingError(f"No valid JSON found in response: {response[:200]}...")
```

## 3. Error Handling Consolidation

### Problem
- 15+ files have scattered error handling logic
- Inconsistent exception handling patterns

### Solution: Create Error Handling Utilities

```python
# src/error_handling/error_handler_mixin.py
from typing import Any, Dict, Type, Union, Callable
from src.error_handling.exceptions import *
import logging

class ErrorHandlerMixin:
    """Mixin providing standardized error handling patterns."""
    
    logger: logging.Logger
    
    def handle_with_recovery(
        self,
        operation: Callable,
        error_context: str,
        recovery_strategies: Dict[Type[Exception], Callable] = None,
        reraise: bool = True
    ) -> Any:
        """Execute operation with standardized error handling and recovery."""
        try:
            return operation()
        except Exception as e:
            self._log_error(e, error_context)
            
            if recovery_strategies and type(e) in recovery_strategies:
                try:
                    return recovery_strategies[type(e)](e)
                except Exception as recovery_error:
                    self._log_error(recovery_error, f"{error_context} - Recovery failed")
            
            if reraise:
                raise
            
            return self._create_error_response(e, error_context)
    
    def _log_error(self, error: Exception, context: str) -> None:
        """Standardized error logging."""
        self.logger.error(
            f"{context}: {type(error).__name__}: {str(error)}",
            exc_info=True
        )
    
    def _create_error_response(self, error: Exception, context: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "context": context
        }
```

## 4. State Validation Consolidation

### Problem
- 6 components have duplicated state validation logic

### Solution: Create State Validation Utilities

```python
# src/core/validation/state_validator.py
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ValidationError
from src.models.structured_cv import StructuredCV
from src.orchestration.state import AgentState

class StateValidator:
    """Centralized state validation utilities."""
    
    @staticmethod
    def validate_agent_state(state: AgentState) -> List[str]:
        """Validate agent state and return list of errors."""
        errors = []
        
        # Common validation logic from multiple files
        if not state.structured_cv:
            errors.append("StructuredCV is required")
        
        if state.current_section_index < 0:
            errors.append("Current section index must be non-negative")
        
        # Add more validation rules
        return errors
    
    @staticmethod
    def validate_structured_cv(cv: StructuredCV) -> List[str]:
        """Validate StructuredCV and return list of errors."""
        errors = []
        
        try:
            # Use Pydantic validation
            cv.model_validate(cv.model_dump())
        except ValidationError as e:
            errors.extend([str(error) for error in e.errors()])
        
        return errors
    
    @staticmethod
    def validate_input_data(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
        """Validate input data against required fields."""
        errors = []
        
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif not data[field]:
                errors.append(f"Empty value for required field: {field}")
        
        return errors
```

## 5. Configuration Loading Consolidation

### Problem
- 4 modules have duplicated configuration loading patterns

### Solution: Create Configuration Manager

```python
# src/config/config_manager.py
from typing import Any, Dict, Optional, Type, TypeVar
from pathlib import Path
import os
from src.config.settings import AppConfig
from src.error_handling.exceptions import ConfigurationError

T = TypeVar('T')

class ConfigurationManager:
    """Centralized configuration management."""
    
    _instance: Optional['ConfigurationManager'] = None
    _config_cache: Dict[str, Any] = {}
    
    def __new__(cls) -> 'ConfigurationManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_config(
        self,
        config_class: Type[T],
        config_file: Optional[Path] = None,
        env_prefix: str = "",
        cache_key: Optional[str] = None
    ) -> T:
        """Load configuration with caching and environment override."""
        cache_key = cache_key or config_class.__name__
        
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        try:
            # Load from file if provided
            config_data = {}
            if config_file and config_file.exists():
                config_data = self._load_from_file(config_file)
            
            # Override with environment variables
            env_data = self._load_from_environment(env_prefix)
            config_data.update(env_data)
            
            # Create config instance
            config = config_class(**config_data)
            
            # Cache the result
            self._config_cache[cache_key] = config
            
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load {config_class.__name__}: {e}") from e
    
    def _load_from_file(self, config_file: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        # Implementation for JSON/YAML/TOML loading
        pass
    
    def _load_from_environment(self, prefix: str) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_data = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                env_data[config_key] = value
        return env_data
```

## 6. Template Processing Consolidation

### Problem
- 3 components have duplicated template processing logic

### Solution: Enhanced Template Manager

```python
# src/templates/unified_template_manager.py
from typing import Any, Dict, Optional
from pathlib import Path
from src.templates.content_templates import ContentTemplateManager
from src.integration.cv_template_manager_facade import CVTemplateManagerFacade
from src.error_handling.exceptions import TemplateError

class UnifiedTemplateManager:
    """Unified template management with all processing patterns."""
    
    def __init__(self):
        self.content_manager = ContentTemplateManager()
        self.cv_facade = CVTemplateManagerFacade()
    
    def process_template(
        self,
        template_name: str,
        template_data: Dict[str, Any],
        template_type: str = "content",
        output_format: str = "text"
    ) -> str:
        """Unified template processing interface."""
        try:
            if template_type == "cv":
                return self.cv_facade.format_template(template_name, template_data)
            else:
                return self.content_manager.format_template(template_name, template_data)
        except Exception as e:
            raise TemplateError(f"Template processing failed: {e}") from e
    
    def load_template_with_metadata(
        self,
        template_path: Path
    ) -> Dict[str, Any]:
        """Load template with frontmatter metadata."""
        return self.content_manager.load_template_with_metadata(template_path)
    
    def get_available_templates(self, template_type: str = "content") -> List[str]:
        """Get list of available templates."""
        if template_type == "cv":
            return self.cv_facade.get_available_templates()
        else:
            return self.content_manager.get_available_templates()
```

## Implementation Timeline

### Week 1: Foundation
- Create abstract base classes and mixins
- Implement error handling utilities
- Set up configuration manager

### Week 2: Agent Refactoring
- Refactor 5 agents to use AbstractAgentExecutor
- Test and validate functionality

### Week 3: Service Consolidation
- Implement LLM interaction mixin
- Refactor services to use unified patterns
- Consolidate template processing

### Week 4: Validation & Testing
- Implement state validation utilities
- Comprehensive testing of all changes
- Performance validation

## Expected Benefits

1. **Code Reduction**: ~40% reduction in duplicated code
2. **Maintainability**: Single source of truth for common patterns
3. **Consistency**: Standardized error handling and validation
4. **Testing**: Easier to test common functionality
5. **Performance**: Reduced memory footprint and faster loading

## Risk Mitigation

1. **Incremental Implementation**: Refactor one component at a time
2. **Comprehensive Testing**: Maintain test coverage throughout
3. **Backward Compatibility**: Ensure existing interfaces remain functional
4. **Rollback Plan**: Keep original implementations until validation complete