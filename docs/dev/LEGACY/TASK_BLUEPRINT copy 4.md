# TASK_BLUEPRINT.md

## Overview

This document outlines the stabilization and technical debt remediation plan for the `anasakhomach-aicvgen` project. The tasks are derived from the `Codebase Analysis_ Technical Debt Audit_ (1).md` and are prioritized to address critical runtime failures, architectural drift, and maintainability issues first. This version incorporates clarifications on the DI container, router logic, and health check implementation.

**Priorities:**
*   **P1 - Critical:** Must-fix issues causing runtime failures, data loss, or significant architectural violations.
*   **P2 - High:** Issues that increase complexity and risk, hindering development velocity.
*   **P3 - Medium/Low:** Clean-up, consistency, and best-practice enforcement.

---

## 1. Core Architecture: Dependency Injection (DI)

### **Task: AD-01 - Enforce Constructor-Based Dependency Injection**

*   **Priority:** P1
*   **Root Cause:** Systemic bypass of the DI framework in favor of a Service Locator pattern (`get_...()` functions). This creates tight coupling, hinders testing, and obscures dependencies.
*   **Impacted Modules:**
    *   `src/agents/agent_base.py`
    *   All agent implementations (e.g., `enhanced_content_writer.py`, `parser_agent.py`)
    *   All service implementations (e.g., `llm_service.py`)
    *   `src/core/dependency_injection.py` (as the source for instantiation)

#### Required Changes

All agents and services must receive their dependencies via their `__init__` constructor. All calls to global `get_...()` functions must be removed from component logic. The `DependencyContainer` in `src/core/dependency_injection.py` will be the central point for component instantiation and dependency resolution.

#### Code Snippets (Example: `EnhancedContentWriterAgent`)

**Before (`src/agents/enhanced_content_writer.py`):**

```python
from ..services.llm_service import get_llm_service
from ..services.error_recovery import get_error_recovery_service

class EnhancedContentWriterAgent(EnhancedAgentBase):
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.llm_service = get_llm_service() # Service Locator Pattern
        self.error_recovery = get_error_recovery_service() # Service Locator Pattern
        # ...

    async def _process_single_item(...):
        # ... logic that uses self.llm_service
        pass
```

**After (`src/agents/enhanced_content_writer.py`):**

```python
from ..services.llm_service import EnhancedLLMService
from ..services.error_recovery import ErrorRecoveryService

class EnhancedContentWriterAgent(EnhancedAgentBase):
    def __init__(self, name: str, description: str, llm_service: EnhancedLLMService, error_recovery_service: ErrorRecoveryService):
        super().__init__(name, description)
        # Constructor Injection
        self.llm_service = llm_service
        self.error_recovery_service = error_recovery_service
        # ...

    async def _process_single_item(...):
        # ... logic that uses self.llm_service
        pass
```

#### Implementation Checklist

1.  [ ] Modify the `__init__` method of `EnhancedAgentBase` to accept `error_recovery_service` and `progress_tracker` as arguments.
2.  [ ] Refactor all agent constructors (e.g., `ParserAgent`, `EnhancedContentWriterAgent`, `QualityAssuranceAgent`, etc.) to accept their service dependencies (like `llm_service`) via the constructor.
3.  [ ] Update the `DependencyContainer` to be responsible for instantiating and injecting these dependencies when creating components.
4.  [ ] Perform a codebase-wide removal of all calls to `get_...()` service locator functions from within the business logic of components.
5.  [ ] Update all corresponding unit tests to pass mocked dependencies during instantiation instead of patching global functions.

#### Tests to Add

*   Add unit tests for each agent's `__init__` method to verify that dependencies are correctly assigned.
*   Update integration tests for agent workflows to inject mock services.

---

## 2. Orchestration & Workflow (`src/orchestration/`)

### **Task: WF-01 - Fix Race Condition in `route_after_qa` Router**

*   **Priority:** P2
*   **Root Cause:** The `route_after_qa` function prioritizes error checking over user feedback. If the `AgentState` contains both an error and a user regeneration request, the user's action is ignored, and the workflow terminates.
*   **Impacted Modules:**
    *   `src/orchestration/cv_workflow_graph.py`

#### Required Changes

Re-prioritize the conditional logic in `route_after_qa` to check for user feedback *before* checking for errors. This ensures user intent is always honored.

#### Code Snippets

**Before (`src/orchestration/cv_workflow_graph.py`):**

```python
async def route_after_qa(state: Dict[str, Any]) -> str:
    agent_state = AgentState.model_validate(state)
    # Priority 1: Check for errors first
    if agent_state.error_messages:
        return "error"
    # Priority 2: Check for user feedback
    if ( agent_state.user_feedback and agent_state.user_feedback.action == UserAction.REGENERATE ):
        return "regenerate"
    # Priority 3: Continue with content generation loop
    return should_continue_generation(state)
```

**After (`src/orchestration/cv_workflow_graph.py`):**

```python
async def route_after_qa(state: Dict[str, Any]) -> str:
    agent_state = AgentState.model_validate(state)
    # Priority 1: Check for user feedback first to ensure user intent is honored.
    if ( agent_state.user_feedback and agent_state.user_feedback.action == UserAction.REGENERATE ):
        logger.info("User requested regeneration, routing to prepare_regeneration")
        return "regenerate"
    # Priority 2: Check for errors if no explicit user action.
    if agent_state.error_messages:
        logger.warning("Errors detected in state, routing to error handler")
        return "error"
    # Priority 3: Continue with the normal content generation loop.
    return should_continue_generation(state)
```

#### Implementation Checklist

1.  [ ] Modify the `route_after_qa` function in `src/orchestration/cv_workflow_graph.py`.
2.  [ ] Move the `if agent_state.user_feedback...` block to be the first check in the function.
3.  [ ] Keep the `if agent_state.error_messages...` check as the second condition.
4.  [ ] Add logging to confirm which path is taken.

#### Tests to Add

*   Update `tests/unit/test_cv_workflow_state_validation.py` with a new test case where the state contains both an error and user feedback, and assert the route is `regenerate`.

---

## 3. Services (`src/services/`)

### **Task: CS-01 - Decompose Monolithic `EnhancedLLMService`**

*   **Priority:** P1
*   **Root Cause:** `EnhancedLLMService` has become a "God Object," violating the Single Responsibility Principle by managing API calls, retries, caching, and rate limiting. This makes it complex and fragile.
*   **Impacted Modules:**
    *   `src/services/llm_service.py`
    *   All components that use `EnhancedLLMService`.

#### Required Changes

Decompose `EnhancedLLMService` into smaller, focused components. The main service will then compose these smaller pieces.

#### Code Snippets

**After (`src/services/llm_client.py` - New File):**

```python
# src/services/llm_client.py
import google.generativeai as genai
from typing import Any

class LLMClient:
    """Handles the direct API call to the LLM provider (Gemini)."""
    def __init__(self, llm_model: genai.GenerativeModel):
        self.llm = llm_model

    async def generate_content(self, prompt: str) -> Any:
        """Directly call the LLM provider's API."""
        if self.llm is None:
            raise ValueError("LLM model is not initialized.")
        return await self.llm.generate_content_async(prompt)
```

**After (`src/services/llm_service.py` - Refactored):**

```python
# src/services/llm_service.py
from tenacity import retry, stop_after_attempt, wait_exponential
from .llm_client import LLMClient
# ... other imports

class EnhancedLLMService:
    def __init__(self, llm_client: LLMClient, rate_limiter, cache, ...):
        self.llm_client = llm_client
        self.rate_limiter = rate_limiter
        self.cache = cache
        # ...

    @retry(stop=stop_after_attempt(3), wait=wait_exponential())
    async def _call_llm_with_retry(self, prompt: str):
        return await self.llm_client.generate_content(prompt)

    async def generate_content(self, prompt: str, ...):
        # 1. Check cache
        cached = self.cache.get(prompt)
        if cached:
            return cached

        # 2. Check rate limit
        await self.rate_limiter.wait_if_needed_async(...)

        # 3. Call with retry logic
        response = await self._call_llm_with_retry(prompt)

        # 4. Cache result and return
        self.cache.set(prompt, response)
        return response
```

#### Implementation Checklist

1.  [ ] Create a new `LLMClient` class in `src/services/llm_client.py` responsible only for making the API call.
2.  [ ] Refactor `EnhancedLLMService` to use composition. It will hold instances of `LLMClient`, `RateLimiter`, and `AdvancedCache`.
3.  [ ] Move the `tenacity.retry` decorator to a private method within `EnhancedLLMService` that wraps the `LLMClient` call.
4.  [ ] Update the `generate_content` method in `EnhancedLLMService` to orchestrate the calls: cache check -> rate limit -> retryable API call -> caching.
5.  [ ] Update the DI container to construct `EnhancedLLMService` with its new dependencies.

#### Tests to Add

*   Unit test for `LLMClient` to verify it makes the API call correctly.
*   Unit test for `EnhancedLLMService` to verify the orchestration logic (e.g., that it calls cache first).

### **Task: D-03 & D-01 - Consolidate Duplicated Logic**

*   **Priority:** P2
*   **Root Cause:** Error classification logic (especially for rate limiting) and fallback content generation are duplicated across multiple agents and services.
*   **Impacted Modules:**
    *   `src/services/rate_limiter.py`
    *   `src/services/llm_service.py`
    *   `src/services/error_recovery.py`
    *   `src/agents/enhanced_content_writer.py`

#### Required Changes

1.  **Error Classification:** Create a centralized utility function for classifying errors.
2.  **Fallback Content:** Remove local fallback logic from agents and ensure they call the central `ErrorRecoveryService`.

#### Code Snippets

**After (`src/utils/error_classification.py` - New File):**

```python
# src/utils/error_classification.py
def is_rate_limit_error(exception: Exception) -> bool:
    """Checks if an exception is a rate-limit error."""
    error_message = str(exception).lower()
    return any(
        keyword in error_message
        for keyword in ["rate limit", "quota exceeded", "429", "resource_exhausted"]
    )
```

**After (`src/agents/enhanced_content_writer.py` - Refactored Error Handling):**

```python
# In an agent's error handling block
# ...
except Exception as e:
    recovery_action = await self.error_recovery_service.handle_error(
        e, context.item_id, context.content_type, context.session_id
    )
    # Use the fallback content provided by the centralized service
    fallback_content = recovery_action.fallback_content
    # ... update item with fallback_content ...
```

#### Implementation Checklist

1.  [ ] Create a new file `src/utils/error_classification.py`.
2.  [ ] Implement `is_rate_limit_error()` and other classification functions as needed.
3.  [ ] Refactor `RateLimiter`, `EnhancedLLMService`, and `ErrorRecoveryService` to use the new utility functions.
4.  [ ] Remove the `_generate_item_fallback_content` method from `EnhancedContentWriterAgent`.
5.  [ ] Ensure the agent's error handling logic correctly calls `error_recovery_service.handle_error()` and uses the returned `RecoveryAction`.

#### Tests to Add

*   Unit tests for `error_classification.py` functions.
*   Unit test for `EnhancedContentWriterAgent` to verify it calls `ErrorRecoveryService` on failure.

### **Task: PB-01 - Make StateManager I/O Asynchronous**

*   **Priority:** P2
*   **Root Cause:** `StateManager` uses synchronous file I/O, which can block the `asyncio` event loop, causing performance bottlenecks.
*   **Impacted Modules:** `src/core/state_manager.py`

#### Required Changes

Convert `save_state` and `load_state` to `async` methods and run the blocking I/O in a separate thread.

#### Code Snippets

**Before (`src/core/state_manager.py`):**

```python
class StateManager:
    def save_state(self):
        # ...
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        # ...

    def load_state(self):
        # ...
        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # ...
```

**After (`src/core/state_manager.py`):**

```python
import asyncio
import json
import os

class StateManager:
    async def save_state(self):
        # ... (logic to get data)
        def blocking_io():
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=2) # data_to_save is the dict

        await asyncio.to_thread(blocking_io)
        # ...

    async def load_state(self):
        # ... (logic to get state_file path)
        def blocking_io():
            if not os.path.exists(state_file):
                return None
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)

        data = await asyncio.to_thread(blocking_io)
        # ... (process loaded data)
```

#### Implementation Checklist

1.  [ ] Add the `async` keyword to `save_state` and `load_state` method signatures.
2.  [ ] Wrap the synchronous `open()`, `json.dump()`, and `json.load()` calls within a helper function.
3.  [ ] Use `await asyncio.to_thread(helper_function)` to execute the blocking I/O.
4.  [ ] Update all callers of `save_state` and `load_state` to `await` the methods.

#### Tests to Add

*   Update `tests/integration/test_state_manager_async.py` to ensure the async methods work correctly.

### **Task: DL-03 - Remove Deprecated `ItemProcessor` Service**

*   **Priority:** P3
*   **Root Cause:** The `ItemProcessor` service (`D-04`) duplicates prompt engineering logic already present in the `EnhancedContentWriterAgent` and is no longer used by any core application components.
*   **Impacted Modules:**
    *   `src/services/item_processor.py`
    *   `tests/integration/test_item_processor_integration.py`
    *   `tests/unit/test_item_processor_simplified.py`

#### Required Changes

Safely delete the `ItemProcessor` service and its associated tests.

#### Implementation Checklist

1.  [ ] Delete the file `src/services/item_processor.py`.
2.  [ ] Delete the file `tests/integration/test_item_processor_integration.py`.
3.  [ ] Delete the file `tests/unit/test_item_processor_simplified.py`.
4.  [ ] Run the full test suite to ensure no regressions were introduced.

---

## 4. Agents (`src/agents/`)

### **Task: CS-02 - Refactor FormatterAgent with Jinja2**

*   **Priority:** P1
*   **Root Cause:** `FormatterAgent` uses brittle, hardcoded f-strings to generate HTML, making it difficult to maintain and modify the CV layout.
*   **Impacted Modules:**
    *   `src/agents/formatter_agent.py`
    *   `src/templates/` (new template file will be added)

#### Required Changes

Replace the Python-based HTML generation with a Jinja2 template.

#### Code Snippets

**Before (`src/agents/formatter_agent.py`):**

```python
class FormatterAgent(EnhancedAgentBase):
    def _format_with_template(self, content_data: ContentData, ...) -> str:
        html = "<html>..."
        if content_data.summary:
            html += f"<h2>Summary</h2><p>{content_data.summary}</p>"
        # ... many more if/elif/f-string statements ...
        return html
```

**After (`src/templates/pdf_template.html` - Hardened):**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{ cv.metadata.name or 'CV' }}</title>
    <!-- CSS is injected by WeasyPrint -->
</head>
<body>
    <header>
        <h1>{{ cv.metadata.name or 'Your Name' | e }}</h1>
        <p class="contact-info">
            {# Defensive: Only show if present, escape all #}
            {% if cv.metadata.email %}{{ cv.metadata.email | e }}{% endif %}
            {% if cv.metadata.phone %} | {{ cv.metadata.phone | e }}{% endif %}
            {% if cv.metadata.linkedin %} | <a href="{{ cv.metadata.linkedin | e }}">LinkedIn</a>{% endif %}
        </p>
    </header>

    {% for section in cv.sections %}
    <section class="cv-section">
        <h2>{{ section.name | e }}</h2>
        <hr>
        {# ... Jinja2 logic for items and subsections ... #}
    </section>
    {% endfor %}
</body>
</html>
```

**After (`src/agents/formatter_agent.py` - Refactored):**

```python
from jinja2 import Environment, FileSystemLoader

class FormatterAgent(EnhancedAgentBase):
    def __init__(self, ...):
        # ...
        template_dir = "src/templates" # Path should be configured
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=True
        )

    async def _format_with_template(self, structured_cv, template_name: str) -> str:
        template = self.jinja_env.get_template(template_name)
        return template.render(cv=structured_cv)
```

#### Implementation Checklist

1.  [ ] Add `Jinja2` to `requirements.txt`.
2.  [ ] Harden the existing `src/templates/pdf_template.html` to handle missing data defensively using `| e` (escape) and `if` checks.
3.  [ ] Modify `FormatterAgent` to initialize a `jinja2.Environment`.
4.  [ ] Replace the f-string logic in `_format_with_template` with a call to `template.render()`, passing the `StructuredCV` object.
5.  [ ] Ensure the agent's `run` method correctly calls the new formatting logic.

#### Tests to Add

*   Add unit tests (`test_pdf_template_hardening.py`) to render the Jinja2 template with various incomplete `StructuredCV` objects to ensure it doesn't crash.

---

## 5. Data Models & Workflow (`src/models/`, `src/orchestration/`)

### **Task: CB-01 & CB-02 - Fix Data Contract Breaches in AgentState**

*   **Priority:** P1
*   **Root Cause:** `AgentState` is missing fields to store the outputs from `CVAnalysisAgent` and `CVAnalyzerAgent`, causing critical data to be lost during the workflow.
*   **Impacted Modules:**
    *   `src/orchestration/state.py`
    *   `src/orchestration/cv_workflow_graph.py`
    *   `src/agents/cv_analyzer_agent.py`

#### Required Changes

1.  Add `cv_analysis_results` to `AgentState` to store the output of the `CVAnalysisAgent`.
2.  Add a flexible `node_execution_metadata` dictionary to `AgentState` to capture metadata from any node.
3.  Update the workflow graph to include a node for `CVAnalysisAgent` and ensure it populates the new state fields.

#### Code Snippets

**Before (`src/orchestration/state.py`):**

```python
class AgentState(BaseModel):
    # ... existing fields ...
    # No field for cv_analysis_results
```

**After (`src/orchestration/state.py`):**

```python
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from ..models.cv_analysis_result import CVAnalysisResult

class AgentState(BaseModel):
    # ... existing fields ...

    # CB-01 Fix: Add field to store analysis results
    cv_analysis_results: Optional[CVAnalysisResult] = None

    # CB-02 Fix: Add generic field for node-specific metadata
    node_execution_metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
```

**After (`src/agents/cv_analyzer_agent.py` - Node Update):**

```python
# In CVAnalyzerAgent.run_as_node method
async def run_as_node(self, state: "AgentState") -> Dict[str, Any]:
    # ... agent logic ...
    analysis_result = CVAnalysisResult(...)
    node_meta = {'success': True, 'confidence': 0.9}

    # Return a dictionary with keys matching AgentState fields
    return {
        "cv_analysis_results": analysis_result,
        "node_execution_metadata": {
            **state.get("node_execution_metadata", {}),
            "cv_analyzer": node_meta
        }
    }
```

#### Implementation Checklist

1.  [ ] Add `cv_analysis_results: Optional[CVAnalysisResult] = None` to `AgentState` in `src/orchestration/state.py`.
2.  [ ] Add `node_execution_metadata: Dict[str, Any] = Field(default_factory=dict)` to `AgentState`.
3.  [ ] In `cv_workflow_graph.py`, add a new node for the `CVAnalysisAgent`.
4.  [ ] Ensure the `CVAnalysisAgent` node function returns a dictionary with the `cv_analysis_results` key.
5.  [ ] Update `CVAnalyzerAgent` to populate `node_execution_metadata`.

#### Tests to Add

*   Unit test `test_orchestration_state.py` to confirm that `AgentState` can store `CVAnalysisResult`.
*   Integration test `test_agent_state_alignment.py` to verify that the `CVAnalysisAgent` node correctly updates the `AgentState`.

---

## Implementation Checklist (Summary)

1.  **[P1]** Refactor all agents and services to use constructor-based Dependency Injection (`AD-01`).
2.  **[P1]** Decompose the monolithic `EnhancedLLMService` into smaller, single-responsibility classes (`CS-01`).
3.  **[P1]** Fix `AgentState` data contracts by adding `cv_analysis_results` and `node_execution_metadata` fields (`CB-01`, `CB-02`).
4.  **[P1]** Refactor the `FormatterAgent` to use Jinja2 for HTML templating instead of f-strings (`CS-02`).
5.  **[P2]** Fix the race condition in the `route_after_qa` workflow router (`WF-01`).
6.  **[P2]** Centralize duplicated rate-limit error classification logic into `src/utils/error_classification.py` (`D-03`).
7.  **[P2]** Consolidate all fallback content generation into the central `ErrorRecoveryService` (`D-01`).
8.  **[P2]** Refactor `StateManager` to use `asyncio.to_thread` for non-blocking file I/O (`PB-01`).
9.  **[P2]** Implement concrete Pydantic models for agent inputs in `validation_schemas.py` (`CS-03`).
10. **[P2]** Complete the removal of the deprecated `_extract_json_from_response` method (`DL-01`).
11. **[P3]** Remove the unused `ItemProcessor` service and its associated tests (`DL-03`).
12. **[P3]** Perform final cleanup of dead code, commented-out logic, and naming inconsistencies (`DL-02`, `D-02`, `NC-01`).

---

## Testing Strategy

1.  **Baseline Tests:** Before starting any refactoring, ensure the existing test suite (`unit/`, `integration/`) is passing. This creates a safety net. If coverage is low for a component to be refactored, add baseline tests that capture its current behavior.
2.  **Test-Driven Refactoring:** For each task, add or modify tests *first* to reflect the desired outcome.
    *   **DI Refactoring (AD-01):** Modify agent unit tests to inject mock services via the constructor. The tests will fail until the agent's `__init__` is updated.
    *   **Service Decompositon (CS-01):** Write new unit tests for the smaller components (`LLMClient`, etc.) before refactoring the main service.
    *   **Contract Fixes (CB-01):** Add a test that invokes the workflow and asserts that `state.cv_analysis_results` is correctly populated. This test will fail until the state and workflow are updated.
3.  **New Tests:** Each task in this blueprint specifies "Tests to Add." These must be implemented as part of the task's definition of done.
4.  **Regression Testing:** After each major refactoring (especially P1 tasks), run the entire test suite (`unit`, `integration`, `e2e`) to ensure no existing functionality has been broken.

---

## Critical Gaps & Questions

*All critical questions have been resolved based on the provided clarifications. The current plan is actionable.*

---

---

## 6. Data Models & Validation (`src/models/`)

### **Task: CS-03 - Implement Concrete Agent Input Validation**

*   **Priority:** P2
*   **Root Cause:** The `validate_agent_input` function in `src/models/validation_schemas.py` is a placeholder and does not perform any real validation. This exposes agents to potentially malformed or incomplete data from the `AgentState`.
*   **Impacted Modules:**
    *   `src/models/validation_schemas.py`
    *   All agent `run_as_node` methods that should be validating their inputs.

#### Required Changes

Define specific Pydantic models for the input requirements of each agent. Update the `validate_agent_input` function to use these models for validation based on the `agent_type`.

#### Code Snippets

**Before (`src/models/validation_schemas.py`):**

```python
def validate_agent_input(agent_type: str, state: AgentState) -> Any:
    """
    Validates agent input data.
    NOTE: This is a placeholder. Specific schemas needed for each agent.
    """
    # No actual validation is performed.
    return state
```

**After (`src/models/validation_schemas.py`):**

```python
from pydantic import BaseModel, Field
from typing import Optional
from ..orchestration.state import AgentState
from ..models.data_models import StructuredCV, JobDescriptionData, ResearchFindings

# Input schema for the Content Writer Agent
class ContentWriterAgentInput(BaseModel):
    structured_cv: StructuredCV
    current_item_id: str = Field(..., min_length=1)
    research_findings: Optional[ResearchFindings] = None

# Input schema for the Parser Agent
class ParserAgentInput(BaseModel):
    cv_text: str
    job_description_data: JobDescriptionData

# ... other agent input models ...

def validate_agent_input(agent_type: str, state: AgentState) -> Any:
    """Validate agent input data against a specific Pydantic model."""
    try:
        if agent_type == "content_writer":
            return ContentWriterAgentInput.model_validate({
                "structured_cv": state.structured_cv,
                "current_item_id": state.current_item_id,
                "research_findings": state.research_findings,
            })
        elif agent_type == "parser":
            return ParserAgentInput.model_validate({
                "cv_text": state.cv_text,
                "job_description_data": state.job_description_data,
            })
        # ... add cases for other agents ...
        else:
            # Default behavior: no specific validation, return original state
            return state
    except ValidationError as e:
        logger.error("Validation error for agent '%s': %s", agent_type, e)
        # Re-raise as a standard ValueError to be handled by the agent's error handler
        raise ValueError(f"Input validation failed for {agent_type}: {e}") from e
```

#### Implementation Checklist

1.  [ ] Define a specific Pydantic input model for each agent (`ParserAgentInput`, `ContentWriterAgentInput`, `ResearchAgentInput`, `QualityAssuranceAgentInput`) in `src/models/validation_schemas.py`.
2.  [ ] Each model should only include the slice of `AgentState` that the agent strictly requires to perform its task.
3.  [ ] Update the `validate_agent_input` function to be a dispatcher that selects the correct Pydantic model based on the `agent_type`.
4.  [ ] Call `validate_agent_input` at the beginning of each agent's `run_as_node` method.

#### Tests to Add

*   Add unit tests in `tests/unit/test_validation_schemas.py` for each new Pydantic input model.
*   Test that `validate_agent_input` raises a `ValueError` when required data is missing from the state.

---

## 7. Agent Base Class (`src/agents/`)

### **Task: DL-01 - Remove Deprecated `_extract_json_from_response` Method**

*   **Priority:** P2
*   **Root Cause:** The `_extract_json_from_response` method in `EnhancedAgentBase` is explicitly marked as deprecated and superseded by the more robust `_generate_and_parse_json`. Its presence creates code clutter and risks accidental use of suboptimal logic.
*   **Impacted Modules:**
    *   `src/agents/agent_base.py`
    *   Any agent that might still be calling the deprecated method.

#### Required Changes

Perform a codebase-wide search for any remaining usages of `_extract_json_from_response` and refactor them to use `_generate_and_parse_json`. After confirming there are no more callers, delete the deprecated method.

#### Code Snippets

**Before (`src/agents/agent_base.py`):**

```python
class EnhancedAgentBase(ABC):
    # ...
    async def _generate_and_parse_json(...) -> Dict[str, Any]:
        """
        Generates content from the LLM and robustly parses the JSON output.
        """
        # ... modern, robust implementation ...

    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON content from LLM response...
        This method is kept for backward compatibility...
        """
        # ... old, brittle implementation ...
```

**After (`src/agents/agent_base.py`):**

```python
class EnhancedAgentBase(ABC):
    # ...
    async def _generate_and_parse_json(...) -> Dict[str, Any]:
        """
        Generates content from the LLM and robustly parses the JSON output.
        """
        # ... modern, robust implementation ...

    # The _extract_json_from_response method is completely removed.
```

#### Implementation Checklist

1.  [ ] Use a global search (e.g., `grep` or IDE search) to find all calls to `_extract_json_from_response` in the codebase.
2.  [ ] For any found usages, refactor the code to use the modern `_generate_and_parse_json` async method. This may require making the calling method `async`.
3.  [ ] Once all calls are migrated, delete the `_extract_json_from_response` method from `src/agents/agent_base.py`.
4.  [ ] Run the full test suite to ensure no regressions.

#### Tests to Add

*   Ensure that tests previously covering `_extract_json_from_response` are either removed or updated to test `_generate_and_parse_json`.

---

## 8. Core Services (`src/core/`)

### **Task: AD-02 - Refactor Ambiguous `StateManager`**

*   **Priority:** P2
*   **Root Cause:** The `StateManager` class mixes persistence logic (save/load) with business logic (state modification methods like `update_item_content`). This violates the Single Responsibility Principle and blurs architectural layers.
*   **Impacted Modules:**
    *   `src/core/state_manager.py`
    *   Any component that calls the modification methods on `StateManager`.

#### Required Changes

Refactor `StateManager` to be a pure persistence service. Remove all methods that modify the `StructuredCV` object. Its public API should be limited to `load_state`, `save_state`, `get_structured_cv`, and `set_structured_cv`.

#### Code Snippets

**Before (`src/core/state_manager.py`):**

```python
class StateManager:
    # ...
    def update_item_content(self, item_id: str, new_content: str):
        # ... logic to find and modify the item in self.__structured_cv ...

    def update_section(self, section_id: str, new_data: dict):
        # ... logic to find and modify the section ...

    def save_state(self):
        # ...
```

**After (`src/core/state_manager.py`):**

```python
class StateManager:
    """
    Persistence layer for StructuredCV state. Handles only save/load operations.
    """
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.__structured_cv = None
        # ...

    def set_structured_cv(self, structured_cv: StructuredCV):
        self.__structured_cv = structured_cv

    def get_structured_cv(self) -> Optional[StructuredCV]:
        return self.__structured_cv

    async def save_state(self):
        # Pure persistence logic using asyncio.to_thread
        # ...

    async def load_state(self) -> Optional[StructuredCV]:
        # Pure persistence logic using asyncio.to_thread
        # ...

    # Methods like update_item_content and update_section are REMOVED.
```

#### Implementation Checklist

1.  [ ] Identify all callers of modification methods on `StateManager` (e.g., `update_item_content`).
2.  [ ] Refactor these callers to modify the `StructuredCV` object directly within the agent logic.
3.  [ ] Remove `update_item_content`, `update_section`, and any other state-modifying methods from `StateManager`.
4.  [ ] Ensure the `StateManager`'s public API is limited to `set_structured_cv`, `get_structured_cv`, `save_state`, and `load_state`.

#### Tests to Add

*   Update unit tests for `StateManager` to remove tests for the deleted methods.
*   Add integration tests to verify that state modifications made by agents are correctly persisted by the refactored `StateManager`.

---

## 9. Cleanup & Consistency

### **Task: NC-01 & DL-02 - Final Code Cleanup**

*   **Priority:** P3
*   **Root Cause:** Minor inconsistencies and deprecated code remain, increasing cognitive load and maintenance overhead.
    *   `NC-01`: Prometheus metric variables are in `UPPER_CASE`, violating the project's `snake_case` linting rule.
    *   `DL-02`: Large blocks of commented-out code exist in `src/agents/parser_agent.py`.
*   **Impacted Modules:**
    *   `src/services/metrics_exporter.py`
    *   `src/agents/parser_agent.py`
    *   `config/.pylintrc`

#### Required Changes

1.  **Prometheus Metrics:** Add a comment explaining the convention deviation and disable the specific `pylint` warning for those lines. This acknowledges the external library's convention without sacrificing linter enforcement elsewhere.
2.  **Commented Code:** Review and delete all obsolete, commented-out code blocks.

#### Code Snippets

**After (`src/services/metrics_exporter.py`):**

```python
from prometheus_client import Counter, Histogram

# pylint: disable=invalid-name
# UPPER_CASE is used here to conform to the standard prometheus_client library convention for metric objects.
WORKFLOW_DURATION_SECONDS = Histogram(...)
WORKFLOW_ERRORS_TOTAL = Counter(...)
# pylint: enable=invalid-name
```

#### Implementation Checklist

1.  [ ] In `src/services/metrics_exporter.py`, wrap the Prometheus metric definitions with `pylint: disable/enable=invalid-name` comments.
2.  [ ] Review the commented-out code in `src/agents/parser_agent.py`. If it is obsolete, delete it. If it's a useful reference, move it to a separate documentation file.
3.  [ ] Run `pylint` to confirm the `invalid-name` warnings for metrics are suppressed and that no new errors have been introduced.

#### Tests to Add

*   No new tests are required for this cleanup task. The goal is to have a cleaner codebase and a passing `pylint` run.

---

---

## 9. Code Cleanup and Consistency

### **Task: D-02 - Consolidate Session State Initialization**

*   **Priority:** P3
*   **Root Cause:** The function `initialize_session_state` is duplicated in `src/core/state_helpers.py` and `src/frontend/state_helpers.py`. This violates the DRY principle, creates confusion about the source of truth, and risks divergent logic if one file is updated but the other is not.
*   **Impacted Modules:**
    *   `src/core/state_helpers.py` (to be deleted)
    *   `src/frontend/state_helpers.py` (to be kept)
    *   `src/core/main.py` (or any other module that imports this function)

#### Required Changes

The duplicated logic must be consolidated into a single, canonical function. The version in `src/frontend/state_helpers.py` is the correct one to keep as it directly relates to UI state and is called from the application's entry point. The file `src/core/state_helpers.py` should be deleted.

#### Code Snippets

**Before (Two redundant files):**

```python
# In src/core/state_helpers.py
import streamlit as st
def initialize_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = "..."
    # ... other initializations ...

# In src/frontend/state_helpers.py
import streamlit as st
def initialize_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = "..."
    # ... nearly identical initializations ...
```

**After (Single source of truth):**

```python
# src/core/state_helpers.py is DELETED.

# The version in src/frontend/state_helpers.py is kept as the canonical implementation.

# In src/core/main.py (or other callers)
# The import is updated to point to the correct location.
from ..frontend.state_helpers import initialize_session_state

def main():
    # ...
    initialize_session_state()
    # ...
```

#### Implementation Checklist

1.  [ ] Identify `src/frontend/state_helpers.py` as the canonical source for the `initialize_session_state` function.
2.  [ ] Delete the file `src/core/state_helpers.py`.
3.  [ ] Perform a global search for any imports from `src.core.state_helpers` and update them to import from `src.frontend.state_helpers` instead.
4.  [ ] Run the application and relevant tests to ensure that session state is still initialized correctly on startup.

#### Tests to Add

*   Add a unit test that mocks `streamlit` and verifies that `main.py` calls the `initialize_session_state` function from `src.frontend.state_helpers` exactly once.

### **Task: NC-01 & DL-02 - Final Code Cleanup**

*   **Priority:** P3
*   **Root Cause:** Minor inconsistencies and deprecated code remain, increasing cognitive load and maintenance overhead.
    *   `NC-01`: Prometheus metric variables are in `UPPER_CASE`, violating the project's `snake_case` linting rule.
    *   `DL-02`: Large blocks of commented-out code exist in `src/agents/parser_agent.py`.
*   **Impacted Modules:**
    *   `src/services/metrics_exporter.py`
    *   `src/agents/parser_agent.py`
    *   `config/.pylintrc`

#### Required Changes

1.  **Prometheus Metrics:** Add a comment explaining the convention deviation and disable the specific `pylint` warning for those lines. This acknowledges the external library's convention without sacrificing linter enforcement elsewhere.
2.  **Commented Code:** Review and delete all obsolete, commented-out code blocks.

#### Code Snippets

**After (`src/services/metrics_exporter.py`):**

```python
from prometheus_client import Counter, Histogram

# pylint: disable=invalid-name
# UPPER_CASE is used here to conform to the standard prometheus_client library convention for metric objects.
WORKFLOW_DURATION_SECONDS = Histogram(...)
WORKFLOW_ERRORS_TOTAL = Counter(...)
# pylint: enable=invalid-name
```

#### Implementation Checklist

1.  [ ] In `src/services/metrics_exporter.py`, wrap the Prometheus metric definitions with `pylint: disable/enable=invalid-name` comments.
2.  [ ] Review the commented-out code in `src/agents/parser_agent.py`. If it is obsolete, delete it. If it's a useful reference, move it to a separate documentation file.
3.  [ ] Run `pylint` to confirm the `invalid-name` warnings for metrics are suppressed and that no new errors have been introduced.

#### Tests to Add

*   No new tests are required for this cleanup task. The goal is to have a cleaner codebase and a passing `pylint` run.

---

---

## TODO: Unrelated Unit Test Failures (Post-MVP Triage Required)

The following unit tests are currently failing or erroring, but are unrelated to the session state initialization refactor. Review each for relevance: keep if the feature is still in scope, update if the interface/logic changed, or delete if deprecated.

- tests/unit/test_agent_error_handling.py: multiple failures (validation, general error, async/sync error handling)
- tests/unit/test_api_key_management.py: multiple failures (API key priority, fallback, switching)
- tests/unit/test_application_startup.py: test_initialize_llm_service_success
- tests/unit/test_centralized_json_parsing.py: test_parser_agent_uses_centralized_method, test_research_agent_uses_centralized_method
- tests/unit/test_cleaning_agent.py: multiple errors (error handler, execution)
- tests/unit/test_consolidated_caching.py: all tests error
- tests/unit/test_cv_analyzer_agent.py: all tests error
- tests/unit/test_di_constructor_injection.py: test_enhanced_content_writer_agent_constructor_injection
- tests/unit/test_di_container_agent_resolution.py: test_di_container_resolves_enhanced_content_writer_agent
- tests/unit/test_enhanced_content_writer.py: most tests error or fail
- tests/unit/test_enhanced_content_writer_refactored.py: most tests error
- tests/unit/test_executor_usage.py: all tests fail
- tests/unit/test_formatter_agent.py: most tests error
- tests/unit/test_formatter_agent_contract.py: test_formatter_agent_run_returns_pydantic_model (error)

Action: Triage and resolve or remove as appropriate for production readiness.
