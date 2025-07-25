# Architectural Decoupling Implementation Plan

## Executive Summary

This plan addresses the **High-Level Dependency Map** violations identified in the technical debt audit, focusing on breaking tight coupling between architectural layers and implementing proper separation of concerns.

## Current Architectural Violations

### 1. Container → Services (High Coupling)
- **Problem**: DI container directly manages service lifecycles
- **Impact**: Services tightly bound to container implementation
- **Files Affected**: `src/core/container.py`, `src/services/service_factory.py`

### 2. Agent → LLMService (Direct Dependency)
- **Problem**: All agents directly inject `LLMServiceInterface`
- **Impact**: No abstraction layer, difficult to test and swap implementations
- **Files Affected**: All agent files in `src/agents/`

### 3. State → Workflow (Bidirectional Dependency)
- **Problem**: Circular dependencies between state and workflow components
- **Impact**: Tight coupling, difficult to modify independently
- **Files Affected**: `src/orchestration/state.py`, `src/orchestration/cv_workflow_graph.py`

### 4. UI → Business Logic (Architectural Violation)
- **Problem**: UI layer directly accesses core services and business logic
- **Impact**: Violates layered architecture principles
- **Files Affected**: `src/ui/ui_manager.py`

### 5. Error Handling → Multiple Layers (Scattered Responsibility)
- **Problem**: Error handling logic scattered across 15+ files
- **Impact**: Inconsistent error handling, difficult maintenance
- **Files Affected**: Multiple files across all layers

## Proposed Solutions

### Solution 1: Service Abstraction Layer

**Goal**: Decouple container from direct service management

```python
# src/core/abstractions/service_registry.py
from abc import ABC, abstractmethod
from typing import TypeVar, Type, Optional

T = TypeVar('T')

class ServiceRegistry(ABC):
    """Abstract service registry for dependency injection."""
    
    @abstractmethod
    def register_service(self, service_type: Type[T], implementation: T) -> None:
        """Register a service implementation."""
        pass
    
    @abstractmethod
    def get_service(self, service_type: Type[T]) -> Optional[T]:
        """Retrieve a service by type."""
        pass
    
    @abstractmethod
    def is_registered(self, service_type: Type[T]) -> bool:
        """Check if service is registered."""
        pass

# src/core/abstractions/service_factory_interface.py
class ServiceFactoryInterface(ABC):
    """Abstract factory for creating services."""
    
    @abstractmethod
    def create_llm_service(self) -> 'LLMServiceInterface':
        pass
    
    @abstractmethod
    def create_vector_store_service(self) -> 'VectorStoreServiceInterface':
        pass
```

### Solution 2: Agent Communication Layer

**Goal**: Abstract LLM interactions through a communication layer

```python
# src/agents/abstractions/agent_communication_interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class AgentCommunicationInterface(ABC):
    """Abstract interface for agent-to-service communication."""
    
    @abstractmethod
    async def generate_content(
        self, 
        template_name: str, 
        context: Dict[str, Any],
        response_format: Optional[str] = None
    ) -> str:
        """Generate content using templates and context."""
        pass
    
    @abstractmethod
    async def parse_structured_response(
        self, 
        response: str, 
        target_model: type
    ) -> Any:
        """Parse LLM response into structured data."""
        pass

# src/agents/abstractions/agent_context.py
class AgentContext:
    """Provides context and dependencies for agent execution."""
    
    def __init__(
        self,
        communication: AgentCommunicationInterface,
        template_manager: 'TemplateManagerInterface',
        logger: 'Logger'
    ):
        self.communication = communication
        self.template_manager = template_manager
        self.logger = logger
```

### Solution 3: State-Workflow Separation

**Goal**: Separate state management from workflow orchestration

```python
# src/orchestration/abstractions/state_manager_interface.py
from abc import ABC, abstractmethod
from typing import Optional

class StateManagerInterface(ABC):
    """Abstract interface for state management."""
    
    @abstractmethod
    async def get_state(self, session_id: str) -> Optional['AgentState']:
        """Retrieve state by session ID."""
        pass
    
    @abstractmethod
    async def save_state(self, state: 'AgentState') -> None:
        """Save state changes."""
        pass
    
    @abstractmethod
    async def update_state_field(self, session_id: str, field: str, value: Any) -> None:
        """Update specific state field."""
        pass

# src/orchestration/abstractions/workflow_orchestrator_interface.py
class WorkflowOrchestratorInterface(ABC):
    """Abstract interface for workflow orchestration."""
    
    @abstractmethod
    async def execute_step(self, step_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step."""
        pass
    
    @abstractmethod
    async def get_next_step(self, current_state: 'AgentState') -> Optional[str]:
        """Determine next workflow step."""
        pass
```

### Solution 4: UI Facade Layer

**Goal**: Create a facade between UI and business logic

```python
# src/ui/facades/workflow_facade.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class WorkflowFacadeInterface(ABC):
    """Facade interface for UI-to-business logic communication."""
    
    @abstractmethod
    async def start_workflow(self, cv_text: str, jd_text: str) -> str:
        """Start a new workflow and return session ID."""
        pass
    
    @abstractmethod
    async def get_workflow_status(self, session_id: str) -> Dict[str, Any]:
        """Get workflow status for UI display."""
        pass
    
    @abstractmethod
    async def submit_feedback(self, session_id: str, feedback: Dict[str, Any]) -> bool:
        """Submit user feedback."""
        pass
    
    @abstractmethod
    async def export_results(self, session_id: str, format: str) -> Optional[str]:
        """Export workflow results."""
        pass

# src/ui/facades/workflow_facade_impl.py
class WorkflowFacade(WorkflowFacadeInterface):
    """Concrete implementation of workflow facade."""
    
    def __init__(
        self,
        workflow_manager: 'WorkflowManagerInterface',
        state_manager: 'StateManagerInterface'
    ):
        self.workflow_manager = workflow_manager
        self.state_manager = state_manager
    
    async def start_workflow(self, cv_text: str, jd_text: str) -> str:
        # Delegate to workflow manager without exposing internal details
        return await self.workflow_manager.create_workflow(cv_text, jd_text)
```

### Solution 5: Centralized Error Handling

**Goal**: Implement centralized error handling with proper abstraction

```python
# src/error_handling/abstractions/error_handler_interface.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
from enum import Enum

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorHandlerInterface(ABC):
    """Abstract interface for centralized error handling."""
    
    @abstractmethod
    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recovery_strategy: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Handle error with context and recovery strategy."""
        pass
    
    @abstractmethod
    def register_error_handler(self, error_type: type, handler: Callable) -> None:
        """Register custom error handler for specific error types."""
        pass

# src/error_handling/centralized_error_handler.py
class CentralizedErrorHandler(ErrorHandlerInterface):
    """Centralized error handling implementation."""
    
    def __init__(self, logger: 'Logger'):
        self.logger = logger
        self.error_handlers: Dict[type, Callable] = {}
        self.error_metrics: Dict[str, int] = {}
    
    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recovery_strategy: Optional[Callable] = None
    ) -> Dict[str, Any]:
        # Log error with context
        self._log_error(error, context, severity)
        
        # Update metrics
        self._update_error_metrics(error, severity)
        
        # Try recovery strategy
        if recovery_strategy:
            try:
                return await recovery_strategy(error, context)
            except Exception as recovery_error:
                self._log_error(recovery_error, {**context, "recovery_failed": True})
        
        # Use registered handler if available
        error_type = type(error)
        if error_type in self.error_handlers:
            return await self.error_handlers[error_type](error, context)
        
        # Default error response
        return {
            "success": False,
            "error": str(error),
            "error_type": error_type.__name__,
            "severity": severity.value,
            "context": context
        }
```

## Implementation Timeline

### Phase 1: Foundation (Week 1-2)
1. **Create Abstract Interfaces**
   - Service registry and factory interfaces
   - Agent communication interfaces
   - State and workflow interfaces
   - Error handling interfaces

2. **Implement Core Abstractions**
   - Service registry implementation
   - Agent context and communication layer
   - Centralized error handler

### Phase 2: Decoupling (Week 3-4)
1. **Refactor Container Dependencies**
   - Implement service registry pattern
   - Update container to use abstractions
   - Migrate service factory to use interfaces

2. **Decouple Agent Dependencies**
   - Implement agent communication layer
   - Update all agents to use AgentContext
   - Remove direct LLMService dependencies

### Phase 3: Architectural Separation (Week 5-6)
1. **Separate State and Workflow**
   - Implement state manager interface
   - Create workflow orchestrator abstraction
   - Refactor cv_workflow_graph to use interfaces

2. **Implement UI Facade**
   - Create workflow facade
   - Update UI to use facade instead of direct service calls
   - Remove business logic from UI layer

### Phase 4: Integration and Testing (Week 7-8)
1. **Integration Testing**
   - Test all abstraction layers
   - Verify proper separation of concerns
   - Performance testing

2. **Documentation and Cleanup**
   - Update architecture documentation
   - Remove deprecated direct dependencies
   - Code cleanup and optimization

## Expected Benefits

### 1. **Improved Testability**
- Mock interfaces instead of concrete implementations
- Isolated unit testing for each layer
- Better test coverage and reliability

### 2. **Enhanced Maintainability**
- Clear separation of concerns
- Easier to modify individual components
- Reduced ripple effects from changes

### 3. **Better Scalability**
- Pluggable service implementations
- Easier to add new features
- Support for different deployment configurations

### 4. **Architectural Compliance**
- Proper layered architecture
- Dependency inversion principle
- Single responsibility principle

## Risk Mitigation

### 1. **Backward Compatibility**
- Implement adapters for existing code
- Gradual migration strategy
- Maintain existing APIs during transition

### 2. **Performance Impact**
- Benchmark abstraction overhead
- Optimize critical paths
- Use dependency injection efficiently

### 3. **Complexity Management**
- Clear documentation for each abstraction
- Training for development team
- Code review guidelines

## Success Metrics

1. **Coupling Metrics**
   - Reduce direct dependencies by 80%
   - Achieve proper layered architecture
   - Eliminate circular dependencies

2. **Code Quality**
   - Increase test coverage to 90%+
   - Reduce code duplication by 60%
   - Improve maintainability index

3. **Development Velocity**
   - Faster feature development
   - Easier debugging and troubleshooting
   - Reduced time for architectural changes

This plan provides a systematic approach to addressing the architectural violations while maintaining system functionality and improving overall code quality.