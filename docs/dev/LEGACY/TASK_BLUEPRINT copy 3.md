# TASK_BLUEPRINT.md

## Overview

This document outlines the stabilization and technical debt remediation plan for the `anasakhomach-aicvgen` project. The primary goal is to address critical architectural drift, contract breaches, and code smells identified in the technical audit. The successful completion of these tasks will result in a more robust, maintainable, and scalable codebase, paving the way for future feature development.

This plan prioritizes fixes as follows:
-   **P1 (Critical):** Issues causing runtime failures, data loss, or significant architectural decay.
-   **P2 (High):** Issues that increase complexity, hinder testability, and violate core design principles.
-   **P3 (Medium):** Issues related to code duplication, inconsistencies, and deprecated logic that affect maintainability.

## 1. Core Architecture: Dependency Injection Enforcement (AD-01)

-   **Priority:** P1 (Critical)
-   **Root Cause:** The `AgentLifecycleManager` is the intended central factory for agents, but its use is inconsistent. Many agents bypass it by fetching their own dependencies (e.g., `LLMService`) using global `get_...()` service locators. This practice undermines the DI pattern, creating tight coupling and making testing difficult.
-   **Impacted Modules:**
    -   `src/core/dependency_injection.py`
    -   `src/core/agent_lifecycle_manager.py`
    -   All agent implementations (e.g., `enhanced_content_writer.py`, `parser_agent.py`)
    -   All service implementations (e.g., `llm_service.py`)

### Logic Change: Enforce Constructor Injection via Factories

All agents and services must declare their dependencies in their constructors. The `AgentLifecycleManager` will be the single point of agent creation, and its registered factories will use the `DependencyContainer` to resolve and inject dependencies during instantiation.

#### Before (`src/agents/agent_base.py`):

```python
# src/agents/agent_base.py
from ..services.error_recovery import get_error_recovery_service

class EnhancedAgentBase(ABC):
    def __init__(self, name: str, ...):
        # Dependencies are fetched globally, hiding them.
        self.error_recovery = get_error_recovery_service()
        # ...
```

#### After (`src/agents/agent_base.py`):

```python
# src/agents/agent_base.py
from ..services.error_recovery import ErrorRecoveryService

class EnhancedAgentBase(ABC):
    def __init__(
        self,
        name: str,
        # ... other params
        error_recovery_service: ErrorRecoveryService,
        # ... other dependencies
    ):
        # Dependencies are explicitly injected.
        self.name = name
        self.error_recovery = error_recovery_service
        # ...
```

#### After (`src/core/agent_lifecycle_manager.py` - Example Factory Registration):

```python
# src/core/agent_lifecycle_manager.py
from ..core.dependency_injection import get_container
from ..agents.parser_agent import ParserAgent
from ..services.llm_service import EnhancedLLMService

def register_agents(lifecycle_manager):
    container = get_container()

    def parser_agent_factory() -> ParserAgent:
        # Resolve dependencies from the container
        llm_service = container.get(EnhancedLLMService)
        # Inject dependencies into the constructor
        return ParserAgent(
            name="ParserAgent",
            description="Parses CV and JD.",
            llm_service=llm_service
        )

    # Register the factory with the lifecycle manager
    lifecycle_manager.register_agent_type(
        agent_type="parser",
        factory=parser_agent_factory,
        # ... other pool config
    )
```

### Implementation Checklist

1.  **Modify Constructors:** Update the `__init__` method of all agents and services to accept their dependencies as arguments.
2.  **Update Factories:** Refactor the agent creation factories used by `AgentLifecycleManager`. Inside each factory, use `container.get(DependencyType)` to resolve dependencies and pass them into the agent's constructor.
3.  **Remove Global Getters:** Perform a codebase-wide removal of all `get_...()` service locator calls from within the business logic of components. Replace them with `self.dependency_name`.
4.  **Update `DependencyContainer`:** Ensure all services (like `EnhancedLLMService`, `ErrorRecoveryService`, etc.) are registered as singletons in the `DependencyContainer` so they can be resolved by the factories.

## 2. Models & State Contracts

### 2.1. Fix `AgentState` Contract Breach (CB-01)

-   **Priority:** P1 (Critical)
-   **Root Cause:** The `AgentState` model lacks a field to store the `CVAnalysisResult` object produced by `CVAnalysisAgent`, causing the analysis output to be lost during the workflow.
-   **Impacted Modules:** `src/orchestration/state.py`, `src/agents/cv_analyzer_agent.py`, `src/orchestration/cv_workflow_graph.py`

#### Logic Change: Extend `AgentState`

Add a new optional field to `AgentState` to hold the analysis results.

#### Before (`src/orchestration/state.py`):

```python
# src/orchestration/state.py
class AgentState(BaseModel):
    # ... existing fields
    structured_cv: StructuredCV
    # No field for CV analysis results
```

#### After (`src/orchestration/state.py`):

```python
# src/orchestration/state.py
from typing import Optional
from pydantic import BaseModel
from ..models.data_models import StructuredCV
from ..models.cv_analysis_result import CVAnalysisResult

class AgentState(BaseModel):
    # ... existing fields
    structured_cv: StructuredCV
    cv_analysis_results: Optional[CVAnalysisResult] = None # New field
```

### Implementation Checklist

1.  **Update `AgentState`:** Add `cv_analysis_results: Optional[CVAnalysisResult] = None` to the `AgentState` model in `src/orchestration/state.py`.
2.  **Update `CVAnalysisAgent`:** Ensure the `run_as_node` method of `CVAnalysisAgent` returns a dictionary with the key `cv_analysis_results` holding the `CVAnalysisResult` object.
3.  **Update Workflow Graph:** Add a node for `cv_analyzer_agent` to `cv_workflow_graph.py` and ensure it's correctly wired in the workflow sequence.

### 2.2. Implement Agent Input Validation (CS-03)

-   **Priority:** P2 (High)
-   **Root Cause:** The `validate_agent_input` function is a stub and does not perform any validation, exposing agents to potentially malformed data.
-   **Impacted Modules:** `src/models/validation_schemas.py`, all agent `run_as_node` methods.

#### Logic Change: Define Per-Agent Input Schemas

Create specific Pydantic models for the inputs required by each agent's `run_as_node` method.

#### Before (`src/models/validation_schemas.py`):

```python
# src/models/validation_schemas.py
def validate_agent_input(agent_type: str, input_data: Any) -> Any:
    # Placeholder: No actual validation is performed.
    return input_data
```

#### After (`src/models/validation_schemas.py`):

```python
# src/models/validation_schemas.py
from pydantic import BaseModel, Field, ValidationError
from typing import Any, Dict, Optional
from ..orchestration.state import AgentState
from ..models.data_models import StructuredCV

# Example for ContentWriterAgent
class ContentWriterInput(BaseModel):
    structured_cv: StructuredCV
    current_item_id: str = Field(..., min_length=1)
    research_findings: Optional[Dict[str, Any]] = None

def validate_agent_input(agent_type: str, state: AgentState) -> Any:
    """Validate agent input data against a specific Pydantic model."""
    try:
        if agent_type == "enhanced_content_writer":
            input_data_dict = {
                "structured_cv": state.structured_cv,
                "current_item_id": state.current_item_id,
                "research_findings": state.research_findings
            }
            return ContentWriterInput.model_validate(input_data_dict)
        # ... add validation for other agents
        return state # Return full state if no specific validation
    except ValidationError as e:
        # Re-raise as a custom exception if needed, or handle here
        raise ValueError(f"Input validation failed for {agent_type}: {e}") from e
```

### Implementation Checklist

1.  **Define Input Models:** In `src/models/validation_schemas.py`, create Pydantic models for the inputs of `ParserAgent`, `EnhancedContentWriterAgent`, `ResearchAgent`, and `QualityAssuranceAgent`.
2.  **Update `validate_agent_input`:** Implement the logic in `validate_agent_input` to select the correct model based on `agent_type` and validate the relevant slice of the `AgentState`.
3.  **Integrate Validation:** Call `validate_agent_input` at the beginning of each agent's `run_as_node` method.

## 3. Service Refactoring

### 3.1. Decompose `EnhancedLLMService` (CS-01)

-   **Priority:** P2 (High)
-   **Root Cause:** `EnhancedLLMService` is a "God object" managing API calls, retries, caching, rate limiting, and API key switching, violating the Single Responsibility Principle.
-   **Impacted Modules:** `src/services/llm_service.py`

#### Logic Change: Composition over Monolith

Decompose the service into smaller, focused components. The main service will orchestrate these components.

#### Before (`src/services/llm_service.py`):

```python
# A single large class handles everything
class EnhancedLLMService:
    @retry(...)
    async def generate_content(...):
        # ... checks cache
        # ... waits for rate limit
        # ... makes API call
        # ... handles fallback keys
```

#### After (New Proposed Structure):

```python
# src/services/llm_client.py
class LLMClient:
    """Handles the direct API call to the LLM provider."""
    async def generate(self, prompt: str, model_name: str) -> Any:
        # ... logic to call genai.GenerativeModel(...).generate_content
        pass

# src/services/llm_retry_handler.py
from tenacity import retry
class LLMRetryHandler:
    """Wraps an LLM client with retry logic."""
    def __init__(self, client: LLMClient):
        self.client = client

    @retry(...)
    async def generate_with_retries(self, prompt: str, model_name: str) -> Any:
        return await self.client.generate(prompt, model_name)

# src/services/llm_service.py (Refactored)
class EnhancedLLMService:
    """Orchestrates LLM-related operations."""
    def __init__(self, retry_handler: LLMRetryHandler, cache: AdvancedCache, ...):
        self.retry_handler = retry_handler
        self.cache = cache

    async def generate_content(self, prompt: str, ...) -> LLMResponse:
        cached = self.cache.get(...)
        if cached:
            return cached
        response = await self.retry_handler.generate_with_retries(...)
        self.cache.set(...)
        return response
```

### Implementation Checklist

1.  **Create `LLMClient`:** Create a new class responsible only for making the API call to Gemini.
2.  **Create `LLMRetryHandler`:** Create a class that wraps an `LLMClient` and uses `tenacity` for retries.
3.  **Refactor `EnhancedLLMService`:** Modify the main service to accept the new components via its constructor (DI). Its `generate_content` method will now orchestrate calls to the cache and the retry handler.
4.  **Update `DependencyContainer`:** Adjust the DI container to correctly build and inject this new chain of services.

### 3.2. Make `StateManager` I/O Async (PB-01)

-   **Priority:** P2 (High)
-   **Root Cause:** `StateManager.save_state` uses synchronous file I/O, which blocks the `asyncio` event loop. The primary caller (`SessionManager._periodic_cleanup`) is already `async`.
-   **Impacted Modules:** `src/core/state_manager.py`, `src/services/session_manager.py`

#### Logic Change: Use `asyncio.to_thread`

Convert the `save_state` method to `async` and wrap the blocking I/O call.

#### Before (`src/core/state_manager.py`):

```python
# src/core/state_manager.py
import json
class StateManager:
    def save_state(self):
        # ...
        with open(state_file, "w") as f:
            json.dump(...) # This is a blocking call
```

#### After (`src/core/state_manager.py`):

```python
# src/core/state_manager.py
import json
import asyncio
from .state_manager import EnumEncoder # Assuming EnumEncoder is in the same module

class StateManager:
    async def save_state(self):
        structured_cv = self.get_structured_cv()
        if not structured_cv:
            return None
        # ... logic to create state_file path ...
        state_file = f"data/sessions/{structured_cv.id}/state.json"

        def blocking_io():
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            with open(state_file, "w") as f:
                json.dump(structured_cv.to_dict(), f, indent=2, cls=EnumEncoder)

        await asyncio.to_thread(blocking_io)
        return state_file
```

### Implementation Checklist

1.  **Modify `save_state`:** Change its signature to `async def save_state(self):`. Wrap the `with open(...)` block in a local function and execute it with `await asyncio.to_thread(...)`.
2.  **Modify `load_state`:** Apply the same `async def` and `asyncio.to_thread` pattern for `json.load()`.
3.  **Update `SessionManager`:** Find the call to `save_state` within `SessionManager` (likely in a cleanup task) and add the `await` keyword.

### 3.3. Centralize Duplicated Logic (D-01, D-03)

-   **Priority:** P3 (Medium)
-   **Root Cause:** Logic for rate-limit error detection and fallback content generation is duplicated.
-   **Impacted Modules:** `src/agents/enhanced_content_writer.py`, `src/services/error_recovery.py`, `src/services/rate_limiter.py`, `src/services/llm_service.py`

### Implementation Checklist

1.  **Create `src/utils/error_classification.py`:**
    -   Add a function `is_rate_limit_error(exception: Exception) -> bool`.
    -   Move the string-matching logic into this function.
    -   Refactor `RateLimiter`, `EnhancedLLMService`, and `ErrorRecoveryService` to call this utility.
2.  **Refactor `EnhancedContentWriterAgent`:**
    -   Remove the `_generate_item_fallback_content` method.
    -   In the error handling block of `_process_single_item`, call `self.error_recovery.handle_error(...)`.
    -   Use the `fallback_content` from the returned `RecoveryAction` object.

## 4. Agent Refactoring

### 4.1. Refactor `FormatterAgent` with Jinja2 (CS-02)

-   **Priority:** P2 (High)
-   **Root Cause:** The `FormatterAgent` uses hardcoded f-strings to generate HTML, making the CV layout difficult to maintain.
-   **Impacted Modules:** `src/agents/formatter_agent.py`, `src/templates/`

#### Logic Change: Use Jinja2 Templating Engine

Replace the Python-based HTML generation with a call to render a Jinja2 template.

#### Before (`src/agents/formatter_agent.py`):

```python
# src/agents/formatter_agent.py
class FormatterAgent(EnhancedAgentBase):
    def _format_with_template(self, content_data: ContentData, ...) -> str:
        formatted_text = f"## {content_data.name}\n\n"
        # ... many more lines of f-string concatenation ...
        return formatted_text
```

#### After (`src/agents/formatter_agent.py` and new template file):

**`src/agents/formatter_agent.py`:**

```python
# src/agents/formatter_agent.py
from jinja2 import Environment, FileSystemLoader
from ..models.data_models import StructuredCV

class FormatterAgent(EnhancedAgentBase):
    def __init__(self, ...):
        # ...
        self.jinja_env = Environment(
            loader=FileSystemLoader('src/templates/'),
            autoescape=True
        )

    def _generate_html(self, structured_cv: StructuredCV) -> str:
        template = self.jinja_env.get_template('cv_template.html') # Corrected template name
        css = self._get_template_css('professional') # Get CSS content
        return template.render(cv=structured_cv, styles=css)
```

**`src/templates/cv_template.html` (formerly `pdf_template.html`):**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{ cv.metadata.get('name', 'CV') }}</title>
    <style>
        {{ styles | safe }}
    </style>
</head>
<body>
    <header>
        <h1>{{ cv.metadata.get('name', 'Your Name') | e }}</h1>
        <p class="contact-info">
            {{ cv.metadata.get('email', '') | e }}
            {% if cv.metadata.get('phone') %} | {{ cv.metadata.get('phone') | e }}{% endif %}
            {% if cv.metadata.get('linkedin') %} | <a href="{{ cv.metadata.get('linkedin') | e }}">LinkedIn</a>{% endif %}
        </p>
    </header>

    {% for section in cv.sections %}
    <section class="cv-section">
        <h2>{{ section.name | e }}</h2>
        <hr>
        {# ... rest of the template ... #}
    </section>
    {% endfor %}
</body>
</html>
```

### Implementation Checklist

1.  **Add Dependency:** Add `Jinja2` to `requirements.txt`.
2.  **Rename and Update Template:** Rename `pdf_template.html` to `cv_template.html` and update it to accept a `styles` variable for CSS injection.
3.  **Refactor `FormatterAgent`:**
    -   Update the agent to initialize the Jinja2 environment.
    -   Modify the HTML generation logic to call `template.render()`, passing both the `cv` object and the CSS string from `_get_template_css`.
    -   The `_format_with_template` method should be removed or refactored into the new `_generate_html` logic.

### 4.2. Remove Deprecated `_extract_json_from_response` (DL-01)

-   **Priority:** P3 (Medium)
-   **Root Cause:** The `_extract_json_from_response` method in `EnhancedAgentBase` is deprecated but still exists.
-   **Impacted Modules:** `src/agents/agent_base.py`

### Implementation Checklist

1.  **Code Search:** Perform a global search for `_extract_json_from_response`.
2.  **Refactor Callers:** If any callers are found, refactor them to use `_generate_and_parse_json`.
3.  **Delete Method:** Delete the `_extract_json_from_response` method from `src/agents/agent_base.py`.
4.  **Run Tests:** Execute the test suite to ensure no regressions were introduced.

## Testing Strategy

-   **Dependency Injection (AD-01):**
    -   **Unit:** For each agent/service, write a test that instantiates it with mock dependencies. Verify the dependencies are correctly assigned.
    -   **Integration:** Test that an agent retrieved from `AgentLifecycleManager` has its dependencies correctly resolved and injected by the container.
-   **Service Refactoring (CS-01, PB-01):**
    -   **Unit:** Write separate unit tests for `LLMClient` and `LLMRetryHandler`. Test the `async` `StateManager` methods using `pytest-asyncio`, mocking `asyncio.to_thread`.
    -   **Integration:** Test the `SessionManager` to ensure it correctly `await`s the new `async` `StateManager.save_state` method.
-   **Contract Enforcement (CB-01, CS-03):**
    -   **Unit:** Write `pytest` cases for `validate_agent_input` with both valid and invalid data to ensure `ValueError` is raised correctly.
    -   **Integration:** Create a test that runs a mini-workflow including the `CVAnalysisAgent` and asserts that `state.cv_analysis_results` is correctly populated.
-   **Agent Refactoring (CS-02, DL-01):**
    -   **Unit:** Test the refactored `FormatterAgent` by passing a mock `StructuredCV` and asserting that the rendered HTML output is correct.
    -   **Regression:** The existing test suite should fail if any code still calls the deleted `_extract_json_from_response` method, confirming its successful removal.

---

## 5. Architectural Refinements

### 5.1. Refactor `StateManager` for Single Responsibility (AD-02)

-   **Priority:** P2 (High)
-   **Root Cause:** The `StateManager` class violates the Single Responsibility Principle by mixing persistence logic (saving/loading state) with business logic (modifying the state of the `StructuredCV` object). This creates ambiguity about where state modifications should occur.
-   **Impacted Modules:** `src/core/state_manager.py`, any agents or services that call modification methods on `StateManager`.

#### Logic Change: Isolate Persistence Logic

The `StateManager` must be refactored to be a pure persistence layer. Its sole responsibility is to save and load the `StructuredCV` object. All methods that modify the `StructuredCV` object must be removed. State modifications should only be performed by agents within the workflow.

#### Before (`src/core/state_manager.py`):

```python
# src/core/state_manager.py

class StateManager:
    # ... (save_state, load_state)

    def update_item_content(self, item_id, new_content):
        """Modifies the state of the CV object directly."""
        structured_cv = self.get_structured_cv()
        if not structured_cv:
            # ... error handling
            return False
        # This is business logic that belongs in an agent.
        return structured_cv.update_item_content(item_id, new_content)

    def update_section(self, section_data: Dict[str, Any]) -> bool:
        """Another example of business logic in the persistence layer."""
        # ... logic to find and update a section ...
        pass
```

#### After (`src/core/state_manager.py`):

```python
# src/core/state_manager.py
import asyncio
import json
import os
from typing import Optional

from ..models.data_models import StructuredCV
from .state_manager import EnumEncoder # Assuming EnumEncoder is in the same module

class StateManager:
    """Manages the persistence of the StructuredCV state."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.__structured_cv: Optional[StructuredCV] = None

    async def save_state(self, cv_to_save: StructuredCV):
        """Asynchronously saves the provided StructuredCV object to a file."""
        if not cv_to_save:
            return None

        state_file = f"data/sessions/{cv_to_save.id}/state.json"

        def blocking_io():
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            with open(state_file, "w") as f:
                json.dump(cv_to_save.model_dump(), f, indent=2, cls=EnumEncoder)

        await asyncio.to_thread(blocking_io)
        return state_file

    async def load_state(self, session_id: str) -> Optional[StructuredCV]:
        """Asynchronously loads a StructuredCV object from a file."""
        state_file = f"data/sessions/{session_id}/state.json"

        def blocking_io():
            if not os.path.exists(state_file):
                return None
            with open(state_file, "r") as f:
                data = json.load(f)
            return StructuredCV.model_validate(data)

        return await asyncio.to_thread(blocking_io)

    # All modification methods like update_item_content, update_section, etc., are removed.
```

### Implementation Checklist

1.  **Identify Modification Methods:** List all methods in `StateManager` other than `save_state`, `load_state`, `get_structured_cv`, and `set_structured_cv`. These are candidates for removal.
2.  **Refactor Callers:** For each identified method (e.g., `update_item_content`), find all callers in the codebase.
3.  **Move Logic to Agents:** Move the state modification logic into the appropriate agent(s) that should be responsible for that change. For example, content updates should happen within the `ContentWriterAgent` or in response to user feedback processing.
4.  **Delete Methods:** Once all callers have been refactored, delete the modification methods from `StateManager`.
5.  **Update `save_state` Signature:** Modify `save_state` to accept the `StructuredCV` object to be saved (e.g., `async def save_state(self, cv_to_save: StructuredCV)`), making it a pure function that doesn't rely on `self.__structured_cv`.

## 6. Code Cleanup and Deprecation

### 6.1. Deprecate and Remove `ItemProcessor` (D-04)

-   **Priority:** P3 (Medium)
-   **Root Cause:** The `ItemProcessor` service duplicates prompt-building logic that is now handled more robustly by the `EnhancedContentWriterAgent` and `ContentTemplateManager`. It appears to be a legacy component.
-   **Impacted Modules:** `src/services/item_processor.py`, any modules that import and use it.

### Implementation Checklist

1.  **Identify Callers:** Perform a global search for `ItemProcessor` to find where it is instantiated and used.
2.  **Refactor Callers:** Modify the calling code to use the `EnhancedContentWriterAgent` or `ContentTemplateManager` directly for prompt construction and content generation.
3.  **Delete Files:** Delete the `src/services/item_processor.py` file and its corresponding unit test `tests/unit/test_item_processor_simplified.py`.
4.  **Run Tests:** Execute the full test suite to ensure that the removal did not cause regressions.

### 6.2. Consolidate Session State Initialization (D-02)

-   **Priority:** P3 (Medium)
-   **Root Cause:** Two functions named `initialize_session_state` exist in `src/core/state_helpers.py` and `src/frontend/state_helpers.py`, creating redundancy and potential for inconsistency. The function in `src/core` appears to be dead code.
-   **Impacted Modules:** `src/core/state_helpers.py`, `src/frontend/state_helpers.py`, `src/core/main.py`.

### Implementation Checklist

1.  **Confirm Canonical Version:** Verify that `src/core/main.py` calls the function from `src/frontend/state_helpers.py`, making it the canonical version.
2.  **Delete Redundant File:** Delete the `src/core/state_helpers.py` file.
3.  **Update Imports:** Search the codebase for any imports from `src.core.state_helpers` and update them to point to `src.frontend.state_helpers`.
4.  **Run Application:** Run the Streamlit application to ensure session state initialization still works correctly.

### 6.3. Remove Legacy Code from `ParserAgent` (DL-02)

-   **Priority:** P3 (Medium)
-   **Root Cause:** `src/agents/parser_agent.py` contains large, commented-out blocks of legacy code (e.g., regex-based parsers), which reduce readability and create confusion.
-   **Impacted Modules:** `src/agents/parser_agent.py`.

### Implementation Checklist

1.  **Review Commented Code:** Carefully review the commented-out code blocks in `parser_agent.py`.
2.  **Delete Obsolete Code:** If the code is truly obsolete and has been replaced by the LLM-first parsing approach, delete it permanently from the source file.
3.  **Archive Reference Code (Optional):** If a commented block contains logic that might be useful for a future fallback mechanism, move it to a separate file in a `docs/archive` directory. It should not remain in the active source code.

## 7. Model and State Contract Refinements

### 7.1. Capture Node-Specific Metadata in `AgentState` (CB-02)

-   **Priority:** P2 (High)
-   **Root Cause:** Important metadata from agent node executions, such as the `success` and `confidence` from `CVAnalyzerNodeResult`, is lost because `AgentState` lacks a generic mechanism to store it.
-   **Impacted Modules:** `src/orchestration/state.py`, `src/agents/cv_analyzer_agent.py`.

#### Logic Change: Add a Generic Metadata Store to `AgentState`

Implement the audit's recommendation to add a flexible dictionary to `AgentState` for storing per-node execution metadata. This avoids cluttering the main state schema with node-specific fields.

#### Before (`src/orchestration/state.py`):

```python
# src/orchestration/state.py
class AgentState(BaseModel):
    # ... existing fields
    # No generic place to store node-specific metadata
```

#### After (`src/orchestration/state.py`):

```python
# src/orchestration/state.py
from typing import Dict, Any
from pydantic import Field

class AgentState(BaseModel):
    # ... existing fields
    node_execution_metadata: Dict[str, Any] = Field(default_factory=dict)
```

#### After (`src/agents/cv_analyzer_agent.py`):

```python
# src/agents/cv_analyzer_agent.py
from ..models.cv_analyzer_models import CVAnalyzerNodeResult

class CVAnalyzerAgent(EnhancedAgentBase):
    async def run_as_node(self, state: "AgentState") -> Dict[str, Any]:
        # ... agent logic to produce result: CVAnalyzerNodeResult ...
        result = CVAnalyzerNodeResult(...)

        # Populate the new metadata field in the returned dictionary
        return {
            "cv_analysis_results": result.cv_analysis_results,
            "node_execution_metadata": {
                **state.get("node_execution_metadata", {}), # Preserve existing metadata
                "cv_analyzer": {
                    "success": result.cv_analyzer_success,
                    "confidence": result.cv_analyzer_confidence,
                    "error": result.cv_analyzer_error,
                }
            }
        }
```

### Implementation Checklist

1.  **Update `AgentState`:** Add `node_execution_metadata: Dict[str, Any] = Field(default_factory=dict)` to the `AgentState` model.
2.  **Refactor `CVAnalyzerAgent`:** Modify its `run_as_node` method to return the `node_execution_metadata` dictionary populated with its execution status.
3.  **Review Other Agents:** Identify other agents that produce valuable metadata and refactor them to use this new pattern.

## 8. Naming and Conventions

### 8.1. Standardize Prometheus Metric Naming (NC-01)

-   **Priority:** P3 (Low)
-   **Root Cause:** Prometheus metric variables are named in `UPPER_CASE`, which conflicts with the project's `snake_case` pylint configuration, generating unnecessary linter noise.
-   **Impacted Modules:** `src/services/metrics_exporter.py`, `config/.pylintrc`.

#### Logic Change: Suppress Linter Warning and Document Convention

The most pragmatic solution is to acknowledge that `prometheus-client`'s convention uses `UPPER_CASE` for metric objects. We will keep the `UPPER_CASE` names for clarity and suppress the `pylint` warning for this specific case.

#### Before (`src/services/metrics_exporter.py`):

```python
# src/services/metrics_exporter.py
# This code generates an 'invalid-name' warning from pylint.
WORKFLOW_DURATION_SECONDS = Histogram(...)
LLM_TOKEN_USAGE_TOTAL = Counter(...)
```

#### After (`src/services/metrics_exporter.py`):

```python
# src/services/metrics_exporter.py

# pylint: disable=invalid-name
# The UPPER_CASE naming convention is used here to align with the
# standard practice for prometheus-client metric objects.

WORKFLOW_DURATION_SECONDS = Histogram(...)
LLM_TOKEN_USAGE_TOTAL = Counter(...)
```

### Implementation Checklist

1.  **Add Pylint Disable Comment:** Add `# pylint: disable=invalid-name` at the top of the `src/services/metrics_exporter.py` file or just before the metric definitions.
2.  **Add Explanatory Comment:** Add a comment explaining why the linter warning is being disabled, as shown in the "After" block.
3.  **Verify Linter:** Run `pylint` on the file to confirm that the `invalid-name` warnings for these variables are gone.

---

## Implementation Checklist (Consolidated)

1.  **[DI]** Modify all agent/service constructors to accept dependencies.
2.  **[DI]** Update the `AgentLifecycleManager` factories to use the `DependencyContainer` for injection.
3.  **[DI]** Remove all `get_...()` service locator calls from component logic.
4.  **[Models]** Add `cv_analysis_results: Optional[CVAnalysisResult]` to `AgentState`.
5.  **[Models]** Add `node_execution_metadata: Dict[str, Any]` to `AgentState`.
6.  **[Models]** Define Pydantic input models for each agent in `validation_schemas.py` and implement `validate_agent_input`.
7.  **[Services]** Decompose `EnhancedLLMService` into `LLMClient`, `LLMRetryHandler`, etc.
8.  **[Services]** Refactor `StateManager.save_state` and `load_state` to be `async` using `asyncio.to_thread`.
9.  **[Services]** Create `src/utils/error_classification.py` and centralize rate-limit error detection.
10. **[Services]** Remove state-modifying methods from `StateManager` (`update_item_content`, etc.).
11. **[Agents]** Refactor `EnhancedContentWriterAgent` to use the centralized `ErrorRecoveryService` for fallbacks.
12. **[Agents]** Add `Jinja2` dependency and refactor `FormatterAgent` to use an HTML template file.
13. **[Agents]** Refactor `CVAnalyzerAgent` to populate the new `node_execution_metadata` field.
14. **[Cleanup]** Delete the deprecated `_extract_json_from_response` method from `EnhancedAgentBase`.
15. **[Cleanup]** Delete the legacy `ItemProcessor` service and refactor its callers.
16. **[Cleanup]** Delete the redundant `src/core/state_helpers.py` file.
17. **[Cleanup]** Delete commented-out legacy code from `src/agents/parser_agent.py`.
18. **[Conventions]** Add a `pylint: disable=invalid-name` comment to `src/services/metrics_exporter.py`.

## Testing Strategy (Updated)

-   **Dependency Injection (AD-01):**
    -   **Unit:** For each agent/service, write a test that instantiates it with mock dependencies.
    -   **Integration:** Test the `AgentLifecycleManager`'s ability to create an agent with all dependencies correctly injected.
-   **Service Refactoring (CS-01, PB-01, AD-02):**
    -   **Unit:** Write separate unit tests for `LLMClient`, `LLMRetryHandler`. Test the `async` `StateManager` methods using `pytest-asyncio`. Test that the refactored `StateManager` no longer contains modification methods.
    -   **Integration:** Test the refactored `EnhancedLLMService` orchestration. Test `SessionManager`'s `await` on `StateManager.save_state`.
-   **Contract Enforcement (CB-01, CB-02, CS-03):**
    -   **Unit:** Write `pytest` cases for `validate_agent_input` with both valid and invalid data.
    -   **Integration:** Create a test that runs a mini-workflow including the `CVAnalysisAgent`, asserting that both `state.cv_analysis_results` and `state.node_execution_metadata['cv_analyzer']` are correctly populated.
-   **Agent & Cleanup Refactoring (CS-02, D-04, DL-01, DL-02):**
    -   **Unit:** Test the refactored `FormatterAgent` by passing a mock `StructuredCV` and asserting the rendered HTML is correct.
    -   **Regression:** The test suite should fail if any code still calls deleted methods or imports deleted modules (`ItemProcessor`, old `state_helpers`). This confirms successful removal.
-   **Conventions (NC-01):**
    -   **Static Analysis:** Run `pylint src/services/metrics_exporter.py` and confirm no `invalid-name` errors are reported for the metric objects.

---

## 9. Prompt Management Standardization

-   **Priority:** P2 (High)
-   **Root Cause:** The deprecation of `ItemProcessor` (D-04) removes one source of duplicated prompt logic, but the remaining logic within agents like `EnhancedContentWriterAgent` is not standardized. This leads to prompts being constructed with f-strings directly in the agent code, making them hard to manage, version, and A/B test.
-   **Impacted Modules:** `src/agents/enhanced_content_writer.py`, `src/agents/parser_agent.py`, `src/templates/content_templates.py`

### Logic Change: Centralize Prompt Access via a Manager

All agents requiring LLM prompts must use the `ContentTemplateManager`. This manager will be responsible for loading, caching, and formatting prompts from external files. Agents will request a prompt by a key (e.g., `"resume_role_writer"`), not by a file path. This centralizes prompt engineering and separates it from agent logic.

#### Before (`src/agents/enhanced_content_writer.py`):

```python
# src/agents/enhanced_content_writer.py
class EnhancedContentWriterAgent(EnhancedAgentBase):
    def __init__(self, ...):
        # ...
        # Prompts are loaded directly or constructed inside the agent
        self.content_templates = {
            ContentType.EXPERIENCE: self._load_prompt_template("resume_role_writer"),
            # ...
        }

    def _build_prompt(self, ...):
        # ... complex logic to format the prompt string ...
        return prompt
```

#### After (Agent using the manager):

```python
# src/agents/enhanced_content_writer.py
from ..templates.content_templates import ContentTemplateManager

class EnhancedContentWriterAgent(EnhancedAgentBase):
    def __init__(self, template_manager: ContentTemplateManager, ...):
        # ...
        # The manager is injected as a dependency
        self.template_manager = template_manager

    def _build_prompt(self, template_key: str, context_vars: dict) -> str:
        # Agent logic is simplified to getting and formatting a template
        template = self.template_manager.get_template(template_key)
        if not template:
            # ... handle error ...
            return ""
        return self.template_manager.format_template(template, context_vars)
```

### Implementation Checklist

1.  **Refactor `ContentTemplateManager`:** Ensure `get_template_manager` returns a singleton instance that loads all templates from the `data/prompts` directory at startup.
2.  **Enforce DI for Manager:** Inject `ContentTemplateManager` into the constructors of all agents that build prompts.
3.  **Refactor Agents:** Modify `EnhancedContentWriterAgent`, `ParserAgent`, and any other relevant agents to use `self.template_manager.get_template()` and `self.template_manager.format_template()` instead of their internal prompt-building logic.
4.  **Externalize Prompts:** Ensure all prompts are defined in external `.md` or `.txt` files within the `data/prompts` directory, organized by function (e.g., `parsing/`, `generation/`).

## 10. Standardized Agent Error Handling Pattern

-   **Priority:** P2 (High)
-   **Root Cause:** While an `ErrorRecoveryService` exists, there is no enforced pattern for how agents should use it. This can lead to inconsistent error handling, where some agents might crash while others silently fail.
-   **Impacted Modules:** All agent implementations.

### Logic Change: Standardize `try...except` block in Agents

All primary agent logic within methods like `run_async` or `_process_single_item` must be wrapped in a standardized `try...except` block. Any exception caught should be delegated to the `ErrorRecoveryService` to determine the appropriate `RecoveryAction`.

#### Before (Inconsistent Error Handling):

```python
# Example of inconsistent error handling in an agent
async def some_agent_method(self, ...):
    try:
        # ... logic that might fail ...
        result = await self.llm_service.generate_content(...)
    except Exception as e:
        # Agent decides how to handle the error locally
        logger.error(f"Agent failed: {e}")
        return "Fallback content" # Inconsistent fallback
```

#### After (Standardized Error Handling Pattern):

```python
# Standardized pattern for all agents
async def some_agent_method(self, context: AgentExecutionContext, ...) -> AgentResult:
    try:
        # ... main agent logic ...
        llm_response = await self.llm_service.generate_content(...)
        if not llm_response.success:
            raise AgentExecutionError(f"LLM call failed: {llm_response.error_message}")

        return AgentResult(success=True, output_data=llm_response.content)

    except Exception as e:
        # Delegate error handling to the centralized service
        recovery_action = await self.error_recovery.handle_error(
            exception=e,
            item_id=context.item_id,
            item_type=context.content_type,
            session_id=context.session_id,
            retry_count=context.retry_count,
        )

        # Act on the recommended recovery strategy
        if recovery_action.strategy == RecoveryStrategy.FALLBACK_CONTENT:
            return AgentResult(
                success=True, # The operation succeeded from the workflow's perspective
                output_data=recovery_action.fallback_content,
                metadata={"fallback_used": True}
            )
        else:
            # Re-raise or return an error result to be handled by the orchestrator
            return AgentResult(success=False, error_message=str(e))
```

### Implementation Checklist

1.  **Review `ErrorRecoveryService`:** Ensure the `handle_error` method and `RecoveryAction` model are robust and cover all necessary recovery strategies (retry, fallback, skip).
2.  **Refactor All Agents:** Go through each agent (`ParserAgent`, `EnhancedContentWriterAgent`, etc.) and refactor their primary execution logic to implement the standardized `try...except` pattern shown above.
3.  **Inject `ErrorRecoveryService`:** Ensure `ErrorRecoveryService` is injected into every agent via its constructor as part of the DI enforcement task (AD-01).

## 11. Configuration Management

-   **Priority:** P2 (High)
-   **Root Cause:** Without a strict policy, refactoring can introduce hardcoded values (e.g., model names, timeouts, feature flags), increasing maintenance overhead and making the system brittle.
-   **Impacted Modules:** All modules.

### Logic Change: Enforce Centralized Pydantic-based Configuration

All configurable parameters must be defined in Pydantic models within `src/config/settings.py` and loaded from environment variables (`.env` file). Access to configuration must only occur through the `get_config()` utility. Hardcoded configuration values within agent or service logic are strictly forbidden.

#### Before (Potential hardcoded value):

```python
# some_service.py
class SomeService:
    def __init__(self):
        self.timeout = 30 # Hardcoded value
        self.model = "gemini-1.5-flash" # Hardcoded value
```

#### After (Using centralized config):

```python
# src/config/settings.py
class LLMSettings(BaseModel):
    default_model: str = "gemini-1.5-flash"
    request_timeout: int = 30

class AppConfig(BaseModel):
    llm: LLMSettings = Field(default_factory=LLMSettings)

# some_service.py
from ..config.settings import get_config

class SomeService:
    def __init__(self):
        config = get_config()
        self.timeout = config.llm.request_timeout # Value from config
        self.model = config.llm.default_model # Value from config
```

### Implementation Checklist

1.  **Audit for Hardcoded Values:** During the refactoring process, actively search for and replace any hardcoded configuration values (e.g., model names, timeouts, file paths, API endpoints).
2.  **Extend `settings.py`:** Add any newly identified configurable parameters to the appropriate Pydantic model in `src/config/settings.py`.
3.  **Update `.env.example`:** For every new configuration variable added, add a corresponding entry to the `.env.example` file with a default value and a comment explaining its purpose.
4.  **Code Review Enforcement:** Make adherence to this configuration pattern a mandatory checklist item for all code reviews during the stabilization phase.

## 12. Golden Path Integration Test

-   **Priority:** P1 (Critical)
-   **Root Cause:** The extensive refactoring of core components (DI, services, agents) creates a high risk of breaking the end-to-end data flow of the main CV generation workflow. A dedicated integration test is needed to verify the integrity of the refactored system.
-   **Impacted Modules:** `tests/integration/`

### Logic Change: Create an End-to-End Workflow Test

A new integration test file, `test_golden_path_workflow.py`, will be created. This test will invoke the entire `cv_workflow_graph` from the `parser` node to the `formatter` node, mocking only the external LLM calls to ensure the internal data flow and state transitions are correct.

### Test Implementation Details

-   **Test File:** `tests/integration/test_golden_path_workflow.py`
-   **Input:** A fixed, sample `AgentState` object containing a job description and CV text.
-   **Mocks:**
    -   The `EnhancedLLMService.generate_content` method will be mocked using `pytest-mock`'s `mocker`.
    -   The mock will be configured with `side_effect` to return different, predictable, structured responses based on the prompt's content (e.g., if the prompt is for parsing, return parsed JSON; if for content writing, return enhanced text).
-   **Execution:** The test will call `cv_graph_app.ainvoke(initial_state)`.
-   **Assertions:**
    1.  The final state returned by the workflow should not contain any `error_messages`.
    2.  The `structured_cv` in the final state should contain content that reflects the mocked LLM responses.
    3.  The `research_findings` and `quality_check_results` fields should be populated.
    4.  The `final_output_path` field should contain a non-empty string, indicating the formatter ran successfully.

### Implementation Checklist

1.  **Create Test File:** Create `tests/integration/test_golden_path_workflow.py`.
2.  **Define Mock Responses:** Create realistic mock JSON and text responses for each agent's LLM call (parser, research, content writer, QA).
3.  **Implement Test Case:** Write the `test_full_workflow_success` test case.
    -   Set up the `mocker` to patch `EnhancedLLMService`.
    -   Define the `side_effect` function for the mock.
    -   Define the initial `AgentState`.
    -   Invoke the graph: `final_state = await cv_graph_app.ainvoke(...)`.
    -   Write assertions to validate the final state.
4.  **Add to CI/CD:** Ensure this new test is included in the continuous integration pipeline to catch regressions.

---

## 13. Production Readiness

### 13.1. Implement Resilient Health Check

-   **Priority:** P2 (High)
-   **Root Cause:** The current Docker health check (`curl ... /_stcore/health`) relies on an internal, undocumented Streamlit API. This is fragile and could break with future Streamlit updates. Furthermore, it only confirms the web server is running, not that the application's critical dependencies (LLM, vector store) are functional.
-   **Impacted Modules:** `app.py` (or `src/core/main.py`), `Dockerfile`, `docker-compose.yml`

#### Logic Change: Create a Custom, Dependency-Aware Health Check Endpoint

We will implement a custom `/healthz` endpoint within the Streamlit application. This endpoint will perform a "deep" health check, verifying not only that the application is running but also that it can connect to its critical downstream services.

#### Implementation in `src/core/main.py` (or equivalent entry point):

```python
# src/core/main.py
import streamlit as st
import json
from src.services.llm_service import get_llm_service
from src.services.vector_store_service import get_vector_store_service
from src.utils.exceptions import ConfigurationError

async def handle_health_check():
    """Handles the /healthz endpoint request."""
    health_status = {"status": "healthy", "services": {}}
    status_code = 200

    # 1. Check LLM Service
    try:
        llm_service = get_llm_service()
        # Perform a lightweight, non-blocking validation
        if not await llm_service.validate_api_key():
             raise ConfigurationError("LLM API key is invalid.")
        health_status["services"]["llm_service"] = "ok"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["services"]["llm_service"] = f"failed: {str(e)}"
        status_code = 503

    # 2. Check Vector Store
    try:
        vector_store = get_vector_store_service()
        # The client.heartbeat() method checks the connection.
        vector_store.client.heartbeat()
        health_status["services"]["vector_store"] = "ok"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["services"]["vector_store"] = f"failed: {str(e)}"
        status_code = 503

    # Set response headers and body
    st.status(code=status_code)
    st.json(health_status)

def main():
    # ... existing main logic ...
    # Add routing for the health check
    if st.query_params.get("page") == "healthz":
        asyncio.run(handle_health_check())
        return # Stop further execution for the health check page

    # ... rest of the main function ...
```

#### Updated `Dockerfile` and `docker-compose.yml`:

```yaml
# In Dockerfile or docker-compose.yml
# Change the healthcheck command:
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/?page=healthz || exit 1
```

### Implementation Checklist

1.  **Create Health Check Handler:** Implement the `handle_health_check` async function as shown above in the application's main entry point.
2.  **Add Routing Logic:** Add the routing logic to `main()` to detect and serve the `/healthz` endpoint.
3.  **Update `Dockerfile`:** Modify the `HEALTHCHECK` command in the `Dockerfile` to point to the new endpoint (`http://localhost:8501/?page=healthz`).
4.  **Update `docker-compose.yml`:** Update the `healthcheck` command in `docker-compose.yml` to match the `Dockerfile`'s new command.

## 14. Instrumentation and Monitoring

-   **Priority:** P2 (High)
-   **Root Cause:** While Prometheus metrics exist, they are not consistently applied across all components. As the system is refactored, it's crucial to add instrumentation to monitor the performance and reliability of the new, smaller components.
-   **Impacted Modules:** All refactored services and agents.

### Logic Change: Instrument All Key Operations

All refactored services and key agent operations should be instrumented with the existing Prometheus metrics defined in `src/services/metrics_exporter.py`.

### Implementation Checklist

1.  **`EnhancedLLMService`:**
    -   Inside the refactored `EnhancedLLMService`, use the `LLM_REQUEST_DURATION_SECONDS` histogram to record the duration of each call to the retry handler.
    -   Use the `LLM_REQUESTS_TOTAL` counter to track successes and failures.
    -   Use `LLM_TOKEN_USAGE_TOTAL` to record token counts from successful responses.
2.  **`StateManager`:**
    -   Create a new histogram metric: `STATE_MANAGER_IO_SECONDS = Histogram('aicvgen_state_manager_io_seconds', 'Duration of StateManager I/O operations', ['operation'])`.
    -   In `save_state`, record the duration with `STATE_MANAGER_IO_SECONDS.labels(operation='save').observe(duration)`.
    -   In `load_state`, record the duration with `STATE_MANAGER_IO_SECONDS.labels(operation='load').observe(duration)`.
3.  **Agents:**
    -   In the standardized error handling wrapper (`with_node_error_handling` or similar), wrap the agent's logic with the `AGENT_EXECUTION_DURATION_SECONDS` histogram.
    -   In the `except` block of the wrapper, increment the `AGENT_ERRORS_TOTAL` counter.
4.  **`DependencyContainer`:**
    -   Create a new gauge: `ACTIVE_DEPENDENCIES_GAUGE = Gauge('aicvgen_active_dependencies', 'Number of active dependency instances in the container')`.
    -   Update the gauge whenever an instance is created or disposed.

## 15. Strategic Roadmap (Summary)

This roadmap outlines the phased implementation of the tasks in this blueprint, prioritizing system stability and correctness before comprehensive refactoring and cleanup.

### Phase 1: Foundation & Contracts (P1 Fixes)

-   **Objective:** Stop architectural decay, ensure data integrity, and establish a safety net for further changes.
-   **Key Tasks:**
    -   **Implement DI (AD-01):** Focus on the most critical agents and services first. Establish the factory pattern in `AgentLifecycleManager`.
    -   **Fix State Contracts (CB-01, CB-02):** Update `AgentState` to prevent data loss. This is critical for workflow correctness.
    -   **Implement Golden Path Test (Section 12):** Create the end-to-end integration test *before* major refactoring to serve as a regression safety net.
-   **Goal:** A system that runs without critical data loss and has a reliable test harness.

### Phase 2: Core Refactoring (P2 Fixes)

-   **Objective:** Reduce complexity, improve maintainability, and align the codebase with its intended design principles.
-   **Key Tasks:**
    -   **Decompose `EnhancedLLMService` (CS-01):** Break down the God object into smaller, testable components.
    -   **Refactor `StateManager` (AD-02, PB-01):** Isolate persistence logic and make I/O operations non-blocking.
    -   **Implement Jinja2 Templates (CS-02):** Decouple presentation from logic in the `FormatterAgent`.
    -   **Implement Input Validation (CS-03):** Harden agent contracts by implementing the `validate_agent_input` function.
    -   **Implement Health Check (Section 13.1):** Improve production readiness.
-   **Goal:** A more modular, testable, and performant system architecture.

### Phase 3: Cleanup & Polish (P3 Fixes)

-   **Objective:** Improve the developer experience and long-term code health by eliminating redundancy and inconsistencies.
-   **Key Tasks:**
    -   **Centralize Duplicated Logic (D-01, D-03):** Consolidate error classification and fallback logic.
    -   **Remove Deprecated Code (DL-01, D-04, DL-02):** Delete `_extract_json_from_response`, `ItemProcessor`, and commented-out code.
    -   **Consolidate Session State Init (D-02):** Remove the redundant `initialize_session_state` function.
    -   **Standardize Naming (NC-01):** Address the Prometheus metric naming convention.
    -   **Add Monitoring (Section 14):** Instrument all newly refactored components.
-   **Goal:** A clean, consistent, and observable codebase that is easy for developers to navigate and extend.

---
