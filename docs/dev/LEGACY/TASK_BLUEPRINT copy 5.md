# TASK_BLUEPRINT.md

## Overview

This document outlines the strategic refactoring plan for the `aicvgen` codebase. The primary objective is to remediate identified technical debt, reinforce architectural integrity, and improve performance and maintainability. The plan is structured around a prioritized set of tasks targeting critical contract breaches, architectural drift, and code quality issues. All development work for this phase must strictly adhere to the specifications laid out in this blueprint.

## 1. Models & Data Contracts (`AgentState`, `data_models.py`)

### Task M-01: Normalize `AgentState` and Eliminate Redundancy

-   **Priority**: P2
-   **Root Cause**: `CS-03` - Redundant state fields (`cv_text`, `start_from_scratch`) in `AgentState` create two sources of truth, as this information is also stored within `structured_cv.metadata`.
-   **Impacted Modules**:
    -   `src/orchestration/state.py`
    -   `src/core/state_helpers.py`
    -   `src/agents/parser_agent.py`

#### Required Changes

1.  **Modify `AgentState`**: Remove the `cv_text` and `start_from_scratch` fields from the `AgentState` Pydantic model. The single source of truth for this initial data will be `state.structured_cv.metadata.extra`.
2.  **Update `create_initial_agent_state`**: Refactor this helper to store `cv_text` and `start_from_scratch` exclusively within the `structured_cv.metadata.extra` dictionary upon state creation.
3.  **Update `ParserAgent`**: Modify the `ParserAgent`'s `run_as_node` method to read `cv_text` and `start_from_scratch` from `state.structured_cv.metadata.extra`.

#### Code Snippets

**Before (`src/orchestration/state.py`)**:
```python
class AgentState(BaseModel):
    # ...
    structured_cv: StructuredCV
    # ...
    cv_text: Optional[str] = None
    start_from_scratch: Optional[bool] = None
    # ...
```

**After (`src/orchestration/state.py`)**:
```python
class AgentState(BaseModel):
    # ...
    structured_cv: StructuredCV
    # ...
    # cv_text and start_from_scratch are REMOVED from this level.
    # They now reside exclusively in structured_cv.metadata.extra.
```

**Before (`src/core/state_helpers.py`)**:
```python
def create_initial_agent_state() -> AgentState:
    # ...
    initial_state = AgentState(
        structured_cv=structured_cv,
        job_description_data=job_description_data,
        cv_text=cv_text, # <-- REDUNDANT
        start_from_scratch=start_from_scratch, # <-- REDUNDANT
        # ...
    )
    return initial_state
```

**After (`src/core/state_helpers.py`)**:
```python
def create_initial_agent_state() -> AgentState:
    from ..models.data_models import MetadataModel
    # ...
    cv_text = st.session_state.get("cv_text_input", "")
    start_from_scratch = st.session_state.get("start_from_scratch_input", False)

    metadata = MetadataModel(
        extra={
            "original_cv_text": cv_text,
            "start_from_scratch": start_from_scratch
        }
    )
    structured_cv = StructuredCV(metadata=metadata)

    initial_state = AgentState(
        structured_cv=structured_cv,
        job_description_data=job_description_data
        # cv_text and start_from_scratch are no longer top-level fields
    )
    return initial_state
```

#### Implementation Checklist

-   \[ ] Remove `cv_text` and `start_from_scratch` from `AgentState` in `src/orchestration/state.py`.
-   \[ ] Update `create_initial_agent_state` in `src/core/state_helpers.py` to populate `structured_cv.metadata.extra`.
-   \[ ] Update `ParserAgent.run_as_node` to read initial data from `state.structured_cv.metadata.extra`.
-   \[ ] Run all tests related to state initialization and parsing to ensure no regressions.

## 2. Services (`EnhancedLLMService`, `dependency_injection.py`, `ContentTemplateManager`)

### Task S-01: Centralize `EnhancedLLMService` Instantiation via DI

-   **Priority**: P1
-   **Root Cause**: `DU-01`, `AD-02` - The complex initialization logic for `EnhancedLLMService` is duplicated in multiple modules, and the DI container is bypassed.
-   **Impacted Modules**:
    -   `src/core/dependency_injection.py`
    -   `src/core/application_startup.py`
    -   `src/frontend/callbacks.py`
    -   `src/services/llm_service.py`

#### Required Changes

1.  **Create a Factory**: Define a single factory function, `build_llm_service`, in `src/core/dependency_injection.py` that encapsulates all logic for creating an `EnhancedLLMService` instance.
2.  **Register as Singleton**: Register `EnhancedLLMService` as a singleton in the DI container using the new factory.
3.  **Refactor Call Sites**: Modify `application_startup.py` and `callbacks.py` to retrieve the `EnhancedLLMService` instance from the DI container instead of creating it manually.

#### Code Snippets

**In `src/core/dependency_injection.py` (New Code)**:
```python
def build_llm_service(container: "DependencyContainer", user_api_key: Optional[str] = None) -> EnhancedLLMService:
    """Factory to build the EnhancedLLMService with all its dependencies."""
    settings = container.get("settings", AppConfig)
    rate_limiter = container.get("rate_limiter", RateLimiter)
    # ... resolve other dependencies like cache, retry_handler, etc. ...

    # This logic is now centralized here
    llm_client = LLMClient(genai.GenerativeModel(settings.llm_settings.default_model))
    retry_handler = LLMRetryHandler(llm_client, is_retryable_error)
    cache = get_advanced_cache()

    return EnhancedLLMService(
        settings=settings,
        llm_client=llm_client,
        llm_retry_handler=retry_handler,
        cache=cache,
        rate_limiter=rate_limiter,
        user_api_key=user_api_key,
        # ... other dependencies ...
    )

def register_core_services(container: "DependencyContainer"):
    """Register all core services."""
    container.register_singleton(
        "EnhancedLLMService",
        EnhancedLLMService,
        factory=lambda: build_llm_service(container) # Factory for general use
    )
    # ... other service registrations ...
```

**In `src/frontend/callbacks.py` (Refactored)**:
```python
def handle_api_key_validation():
    # ...
    from src.core.dependency_injection import get_container, build_llm_service

    user_api_key = st.session_state.get("user_gemini_api_key", "")
    container = get_container()

    # Create a temporary, user-specific instance for validation without polluting the singleton
    llm_service_for_validation = build_llm_service(container, user_api_key=user_api_key)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    is_valid = loop.run_until_complete(llm_service_for_validation.validate_api_key())
    # ...
```

#### Implementation Checklist

-   \[ ] Implement `build_llm_service` factory in `src/core/dependency_injection.py`.
-   \[ ] Register `EnhancedLLMService` as a singleton in the DI container.
-   \[ ] Remove manual instantiation from `application_startup.py`.
-   \[ ] Refactor `handle_api_key_validation` to use the factory for validation.
-   \[ ] Add a unit test to verify that `EnhancedLLMService` is resolved as a singleton.

### Task S-02: Implement In-Memory Caching for Prompt Templates

-   **Priority**: P2
-   **Root Cause**: `PB-02` - Static assets like prompt templates are repeatedly loaded from disk, adding unnecessary I/O latency.
-   **Impacted Modules**:
    -   `src/templates/content_templates.py`
    -   All agents that load prompts (`ParserAgent`, `CVAnalyzerAgent`, etc.)

#### Required Changes

1.  **Enhance `ContentTemplateManager`**: Modify the `ContentTemplateManager` to pre-load all templates from the `data/prompts` directory into an in-memory dictionary (e.g., `self._template_cache`) at application startup.
2.  **Singleton Registration**: Ensure `ContentTemplateManager` is registered as a singleton in the DI container.
3.  **Refactor Agents**: Update all agents to retrieve prompt templates from the `ContentTemplateManager` singleton instance instead of reading them directly from the file system.

#### Implementation Checklist

-   \[ ] Modify `ContentTemplateManager.__init__` to scan the prompts directory and load all templates into a cache.
-   \[ ] Ensure `ContentTemplateManager` is registered as a singleton in `dependency_injection.py`.
-   \[ ] Refactor all agents to get templates via `template_manager.get_template("template_key")`.
-   \[ ] Add a unit test to verify that templates are served from the cache after initial load.

## 3. Agents & Orchestration

### Task A-01: Enforce `run_as_node` Return Contract

-   **Priority**: P1
-   **Root Cause**: `CB-01` - Agents violate the `run_as_node` contract by returning full `AgentState` objects or custom Pydantic models instead of a `Dict` slice of the updated state.
-   **Impacted Modules**:
    -   `src/agents/quality_assurance_agent.py`
    -   `src/agents/formatter_agent.py`
    -   `src/agents/cv_analyzer_agent.py`
    -   `src/agents/cleaning_agent.py`
    -   `src/utils/node_validation.py`

#### Required Changes

1.  **Refactor All Agents**: Modify the `run_as_node` method in every agent to strictly return a dictionary where keys are valid `AgentState` field names.
2.  **Use Pydantic Models for Internal Logic**: Agents should use their specific Pydantic models (e.g., `CVAnalyzerNodeResult`) internally but must convert the final output to a dictionary before returning from `run_as_node`.
3.  **Retain `validate_node_output` Decorator**: The decorator in `node_validation.py` will be kept as a critical runtime safety check to enforce this contract automatically and prevent state corruption.

#### Code Snippets

**Before (`src/agents/cv_analyzer_agent.py`)**:
```python
# ...
from ..models.cv_analyzer_models import CVAnalyzerNodeResult

class CVAnalyzerAgent(EnhancedAgentBase):
    # ...
    async def run_as_node(self, state) -> "CVAnalyzerNodeResult":
        # ...
        # Returns a custom Pydantic model, violating the contract
        return CVAnalyzerNodeResult(
            cv_analysis_results=cv_analysis,
            # ...
        )
```

**After (`src/agents/cv_analyzer_agent.py`)**:
```python
# ...
from ..models.cv_analyzer_models import CVAnalyzerNodeResult

class CVAnalyzerAgent(EnhancedAgentBase):
    # ...
    async def run_as_node(self, state: AgentState) -> Dict[str, Any]:
        # ...
        # Internal logic can still use the Pydantic model for clarity
        node_result = CVAnalyzerNodeResult(
            cv_analysis_results=cv_analysis,
            # ...
        )

        # Returns a dictionary slice, fulfilling the contract
        return {"cv_analysis_results": node_result.cv_analysis_results}
```

#### Implementation Checklist

-   \[ ] Refactor `QualityAssuranceAgent.run_as_node` to return a `Dict`.
-   \[ ] Refactor `FormatterAgent.run_as_node` to return a `Dict`.
-   \[ ] Refactor `CVAnalyzerAgent.run_as_node` to return a `Dict`.
-   \[ ] Refactor `CleaningAgent.run_as_node` to return a `Dict`.
-   \[ ] Add unit tests for each agent to verify the return type of `run_as_node`.
-   \[ ] Ensure `@validate_node_output` decorator is applied to all relevant agent nodes and is functioning as a runtime check.

### Task A-02: Eliminate Blocking I/O in Async Agents

-   **Priority**: P1
-   **Root Cause**: `PB-01` - Synchronous `open()` calls are used inside `async` agent methods, blocking the event loop. This will be partially mitigated by `S-02`, but this task ensures any remaining file I/O is handled correctly.
-   **Impacted Modules**:
    -   `src/agents/cv_analyzer_agent.py`
    -   `src/agents/parser_agent.py`
    -   `src/agents/enhanced_content_writer.py`

#### Required Changes

1.  **Isolate I/O**: For any file I/O that cannot be eliminated by `S-02`, create a synchronous helper function for the operation.
2.  **Offload to Thread**: In all `async` methods, use `asyncio.to_thread()` to call the synchronous helper. This offloads the blocking operation from the main event loop. Using `asyncio.to_thread` is sufficient; `aiofiles` is not required for this phase.

#### Code Snippets

**Before (`src/agents/cv_analyzer_agent.py`)**:
```python
class CVAnalyzerAgent(EnhancedAgentBase):
    async def analyze_cv(self, input_data, context=None):
        # ...
        # BLOCKING I/O in an async method
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        # ...
```

**After (if not using Template Manager)**:
```python
import asyncio

def _read_file_sync(path: str) -> str:
    """Synchronous helper to read a file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

class CVAnalyzerAgent(EnhancedAgentBase):
    async def analyze_cv(self, input_data, context=None):
        # ...
        # Non-blocking I/O using asyncio.to_thread
        prompt_template = await asyncio.to_thread(_read_file_sync, prompt_path)
        # ...
```

#### Implementation Checklist

-   \[ ] Audit all `async def` methods in `src/agents/` for file I/O.
-   \[ ] Prioritize replacing file reads with calls to the `ContentTemplateManager` (Task `S-02`).
-   \[ ] For any remaining, necessary file I/O, refactor to use `await asyncio.to_thread(sync_helper, path)`.
-   \[ ] Add a unit test using `unittest.mock.patch` on `asyncio.to_thread` to verify it's being called where appropriate.

## 4. Frontend & UI-State

### Task F-01: Decouple Workflow Control from `st.session_state`

-   **Priority**: P2
-   **Root Cause**: `AD-01` - The UI layer uses `st.session_state` for workflow control flags, tightly coupling the backend to Streamlit.
-   **Impacted Modules**:
    -   `src/core/main.py`
    -   `src/frontend/callbacks.py`
    -   `src/frontend/ui_components.py`

#### Required Changes

1.  **Remove Control Flags**: Eliminate `run_workflow` and `workflow_result` from `st.session_state`.
2.  **Centralize Workflow Invocation**: The "Generate" button will directly call a single function (e.g., `start_cv_generation`) that handles state creation and background thread execution.
3.  **State-Driven UI**: The UI will render based on the contents of `st.session_state.agent_state` and a new simple `st.session_state.is_processing` flag.
4.  **Error Propagation**: The pattern of catching exceptions in the background thread and storing them in `st.session_state.workflow_error` will be retained as it is a sound practice for this architecture. The main UI thread will check this key to display errors.

#### Code Snippets

**After (`src/frontend/callbacks.py`)**:
```python
import threading
from ..core.state_helpers import create_initial_agent_state
from ..orchestration.cv_workflow_graph import cv_graph_app

def start_cv_generation():
    """Initializes and starts the CV generation workflow in a background thread."""
    initial_state = create_initial_agent_state()
    st.session_state.agent_state = initial_state
    st.session_state.is_processing = True
    st.session_state.workflow_error = None # Clear previous errors

    thread = threading.Thread(
        target=_execute_workflow_in_thread,
        args=(initial_state,),
        daemon=True
    )
    thread.start()
    st.rerun()

def _execute_workflow_in_thread(initial_state: AgentState):
    """Target for the background thread. Handles execution and state updates."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        final_state = loop.run_until_complete(cv_graph_app.ainvoke(initial_state))
        st.session_state.agent_state = final_state
    except Exception as e:
        st.session_state.workflow_error = e
    finally:
        st.session_state.is_processing = False
```

#### Implementation Checklist

-   \[ ] Remove `run_workflow` and `workflow_result` from `frontend/state_helpers.py`.
-   \[ ] Add `is_processing` and `workflow_error` to `frontend/state_helpers.py`.
-   \[ ] Implement the `start_cv_generation` and `_execute_workflow_in_thread` functions in `callbacks.py`.
-   \[ ] Refactor the "Generate" button in `ui_components.py` to call `start_cv_generation`.
-   \[ ] Update `main.py` to display a spinner based on `st.session_state.is_processing` and to check for/display errors from `st.session_state.workflow_error`.

## Implementation Checklist

### Priority 1: Critical Stabilization

-   \[ ] **S-01**: Centralize `EnhancedLLMService` instantiation in the DI container.
-   \[ ] **A-01**: Refactor all `run_as_node` methods to return a `Dict[str, Any]` and retain the `@validate_node_output` decorator.
-   \[ ] **A-02**: Replace all blocking I/O calls in `async` agent methods with `asyncio.to_thread` or by using the new Template Manager.

### Priority 2: Architectural Realignment

-   \[ ] **F-01**: Decouple workflow control logic from `st.session_state`.
-   \[ ] **M-01**: Normalize `AgentState` by removing redundant fields.
-   \[ ] **S-02**: Implement in-memory caching for prompt templates in `ContentTemplateManager`.
-   \[ ] **AD-03**: Consolidate all raw text parsing into `ParserAgent`.
-   \[ ] **AD-02**: Ensure all services are consistently retrieved from the DI container.

### Priority 3: Code Hygiene and Maintainability

-   \[ ] **CS-01**: Refactor long methods in `FormatterAgent` and `ParserAgent`.
-   \[ ] **CS-02**: Replace all magic numbers with named constants from `settings.py`.
-   \[ ] **CB-02/CB-03**: Fix all Pydantic contract violations for `AgentResult` and `StructuredCV.metadata`.
-   \[ ] **DL-01/DL-02**: Remove all deprecated methods and legacy compatibility wrappers.
-   \[ ] **NI-01**: Standardize field names for `JobDescriptionData`.

## Testing Strategy

1.  **Run All Existing Tests**: Establish a passing baseline.
2.  **Add Contract-Enforcement Tests**: For each agent, add a test to verify `run_as_node` returns a `dict` with valid `AgentState` keys.
3.  **Add I/O Offloading Tests**: Add a test that mocks `asyncio.to_thread` to verify it's called for any remaining file I/O in `async` methods.
4.  **Add DI Singleton Tests**: Add tests to verify that `EnhancedLLMService` and `ContentTemplateManager` are resolved as singletons.
5.  **Add Template Manager Cache Test**: Add a test that calls `get_template` twice and asserts that the file system is only read from once.
6.  **Integration Test for UI State Flow**: Add a test to simulate a UI button click, which calls `start_cv_generation`, and verify that `st.session_state.is_processing` and `st.session_state.agent_state` are updated correctly.
7.  **Regression Testing**: After each major refactoring task (P1, P2), run the entire test suite to ensure no existing functionality has been broken.

---

## 5. Agent & Orchestration (Continued)

### Task A-03: Clarify Agent Responsibilities (Parser vs. Cleaner)

-   **Priority**: P2
-   **Root Cause**: `AD-03` - The responsibility of parsing raw text into structured data has leaked from `ParserAgent` into `CleaningAgent`.
-   **Impacted Modules**:
    -   `src/agents/parser_agent.py`
    -   `src/agents/cleaning_agent.py`
    -   `src/agents/enhanced_content_writer.py`

#### Required Changes

1.  **Consolidate Parsing in `ParserAgent`**: Move any logic that extracts structure from raw text (e.g., regex for skills) from `CleaningAgent` into `ParserAgent`. The `_parse_big_10_skills` method in `ParserAgent` should be the single source of truth for this task.
2.  **Refocus `CleaningAgent`**: The `CleaningAgent` should only operate on *already structured* data. Its role is to refine, correct, and standardize data within Pydantic models (e.g., fixing typos, standardizing date formats, removing artifacts like "Here is the list:"), not to perform initial extraction.
3.  **Decouple `EnhancedContentWriterAgent`**: Remove the direct call from `EnhancedContentWriterAgent` to `parser_agent._parse_big_10_skills`. The "Big 10 Skills" should be generated by a dedicated node in the workflow (e.g., a `generate_skills_node` that uses the `ParserAgent`) and passed to the writer via the `AgentState`.

#### Code Snippets

**Before (`src/agents/cleaning_agent.py`)**:
```python
class CleaningAgent(EnhancedAgentBase):
    # ...
    async def _clean_big_10_skills(self, raw_output: str, context: AgentExecutionContext) -> List[str]:
        # This method uses regex and other parsing logic, which belongs in ParserAgent.
        skills = []
        numbered_pattern = r"\d+\.\s*([^\n]+)"
        numbered_matches = re.findall(numbered_pattern, raw_output)
        # ... more parsing logic ...
        return cleaned_skills[:10]
```

**After (`src/agents/cleaning_agent.py`)**:
```python
class CleaningAgent(EnhancedAgentBase):
    # ...
    async def _clean_skills_list(self, skills: List[str], context: AgentExecutionContext) -> List[str]:
        # This method now assumes it receives a list of strings (structured data).
        # Its job is to refine this list.
        cleaned_skills = []
        for skill in skills:
            # Example cleaning: standardize capitalization, remove trailing punctuation.
            cleaned_skill = skill.strip().title().rstrip('.,;')
            if cleaned_skill:
                cleaned_skills.append(cleaned_skill)
        return cleaned_skills

    async def run_async(self, input_data: Any, context: AgentExecutionContext) -> AgentResult:
        # ... logic to route to the correct cleaning method based on input type ...
        if isinstance(input_data, list):
            cleaned_data = await self._clean_skills_list(input_data, context)
        # ...
```

#### Implementation Checklist

-   \[ ] Audit `CleaningAgent` and move all raw text parsing logic to `ParserAgent`.
-   \[ ] Refactor `CleaningAgent` methods to operate on structured Pydantic models or lists/dicts.
-   \[ ] Remove the dependency of `EnhancedContentWriterAgent` on `ParserAgent`'s internal methods.
-   \[ ] Update the workflow graph (`cv_workflow_graph.py`) to ensure the `ParserAgent` runs before the `CleaningAgent` for relevant tasks.

## 6. Code Hygiene and Maintainability

### Task C-01: Refactor "God Methods"

-   **Priority**: P3
-   **Root Cause**: `CS-01` - Methods like `FormatterAgent._format_with_template` are excessively long, handle too many concerns, and are difficult to maintain.
-   **Impacted Modules**:
    -   `src/agents/formatter_agent.py`
    -   `src/agents/parser_agent.py`

#### Required Changes

1.  **Decompose `_format_with_template`**: Break down the single large method in `FormatterAgent` into smaller, private helper methods, each responsible for formatting a single CV section (e.g., `_format_experience_section`, `_format_qualifications_section`).
2.  **Decompose `ParserAgent.run_as_node`**: Refactor the main `run_as_node` logic into smaller helper methods for each distinct step: `_parse_job_description`, `_process_cv_path`.

#### Code Snippets

**Before (`src/agents/formatter_agent.py`)**:
```python
class FormatterAgent(EnhancedAgentBase):
    def _format_with_template(self, content_data: ContentData, ...) -> str:
        formatted_text = "# Tailored CV\n\n"
        # ... 300 lines of nested if/elif statements for every section ...
        if hasattr(content_data, "summary") and content_data.summary:
            # ... summary formatting logic ...
        if hasattr(content_data, "experience_bullets") and content_data.experience_bullets:
            # ... experience formatting logic ...
        # ... and so on for all other sections ...
        return formatted_text
```

**After (`src/agents/formatter_agent.py`)**:
```python
class FormatterAgent(EnhancedAgentBase):
    def _format_with_template(self, content_data: ContentData, ...) -> str:
        parts = ["# Tailored CV"]
        parts.append(self._format_header(content_data))
        parts.append(self._format_summary_section(content_data))
        parts.append(self._format_experience_section(content_data))
        # ... call other helper methods ...
        return "\n\n".join(filter(None, parts))

    def _format_header(self, content_data: ContentData) -> str:
        # ... logic for formatting the header ...
        return header_string

    def _format_summary_section(self, content_data: ContentData) -> str:
        # ... logic for formatting the summary ...
        return summary_string

    # ... one private method per section ...
```

#### Implementation Checklist

-   \[ ] Refactor `FormatterAgent._format_with_template` into multiple private helper methods.
-   \[ ] Refactor `ParserAgent.run_as_node` into smaller, more focused private helper methods.
-   \[ ] Ensure all existing unit tests for these agents still pass after refactoring.

### Task C-02: Remove Deprecated Logic and Centralize Constants

-   **Priority**: P3
-   **Root Cause**: `DL-01`, `DL-02`, `CS-02` - The codebase contains dead code, legacy wrappers, and hardcoded "magic numbers".
-   **Impacted Modules**:
    -   `src/agents/parser_agent.py`
    -   `src/agents/cleaning_agent.py`
    -   `src/config/settings.py`

#### Required Changes

1.  **Remove Dead Code**: Delete the `_convert_parsing_result_to_structured_cv` method from `ParserAgent`.
2.  **Remove Legacy Wrappers**: Identify all callers of the synchronous `parse_cv_text` wrapper, refactor them to be `async` and call `parse_cv_with_llm` directly, then delete the wrapper.
3.  **Centralize Constants**: Move all magic numbers (e.g., `10` for skills, `5` for bullet points) to named constants in `src/config/settings.py`. Update all agent code to reference these constants.

#### Code Snippets

**In `src/config/settings.py` (New Code)**:
```python
@dataclass
class OutputConfig:
    # ...
    max_skills_count: int = 10
    max_bullet_points_per_role: int = 5
    max_bullet_points_per_project: int = 3
    min_skill_length: int = 2
```

**In `src/agents/cleaning_agent.py` (Refactored)**:
```python
from ..config.settings import get_config

class CleaningAgent(EnhancedAgentBase):
    def __init__(self, ...):
        # ...
        self.settings = get_config()

    async def _clean_big_10_skills(self, ...):
        # ...
        # Use the constant from settings instead of a magic number
        return cleaned_skills[:self.settings.output.max_skills_count]
```

#### Implementation Checklist

-   \[ ] Delete the `_convert_parsing_result_to_structured_cv` method from `ParserAgent`.
-   \[ ] Find and refactor all callers of `ParserAgent.parse_cv_text`, then delete the method.
-   \[ ] Add constants for all magic numbers to `src/config/settings.py`.
-   \[ ] Replace all hardcoded numbers in the codebase with references to the new constants.
-   \[ ] Run the full test suite to ensure no behavior has changed unintentionally.

## Critical Gaps & Questions

This section is considered resolved based on the provided clarifications. The recommendations have been integrated into the task blueprints above. Key decisions are:
1.  **Prompt Caching**: Will be implemented by enhancing the existing `ContentTemplateManager` as a singleton service.
2.  **I/O Offloading**: `asyncio.to_thread` will be used as the primary strategy.
3.  **Node Validation**: The `@validate_node_output` decorator will be retained as a runtime safety check.
4.  **Error Propagation**: The current method of using `st.session_state.workflow_error` is confirmed as the correct approach for this architecture.

---

## 6. Code Hygiene and Maintainability (Continued)

### Task C-03: Enforce Pydantic Model Contracts

-   **Priority**: P3
-   **Root Cause**: `CB-02`, `CB-03` - Agents and utility functions inconsistently interact with Pydantic models, sometimes treating them as dictionaries or returning incorrect types, which undermines the data contract layer.
-   **Impacted Modules**:
    -   `src/agents/agent_base.py`
    -   `src/agents/cv_analyzer_agent.py`
    -   `src/agents/research_agent.py`
    -   `src/agents/cv_conversion_utils.py`

#### Required Changes

1.  **Validate `AgentResult.output_data`**: Audit all `run_async` methods in every agent. Ensure the `output_data` field of the returned `AgentResult` is always an instance of a Pydantic `BaseModel`, as required by the validator in `agent_base.py`.
2.  **Standardize `StructuredCV.metadata` Access**: Correct all instances where `structured_cv.metadata` is accessed like a dictionary (e.g., `metadata['name']`). All custom or dynamic data must be stored in and accessed via `structured_cv.metadata.extra['key']`. Pre-defined fields like `item_id` or `company` should be accessed via `structured_cv.metadata.item_id`.

#### Code Snippets

**`AgentResult.output_data` Violation (Before)**:
```python
# In an agent's run_async method
class SomeAgent(EnhancedAgentBase):
    async def run_async(self, ...) -> AgentResult:
        # ...
        raw_dict_output = {"data": "some_value"}
        return AgentResult(
            success=True,
            output_data=raw_dict_output # <-- VIOLATION: This is a dict, not a BaseModel
        )
```

**`AgentResult.output_data` Fix (After)**:```python
# Define a Pydantic model for the output
class SomeAgentOutput(BaseModel):
    data: str

# In the agent's run_async method
class SomeAgent(EnhancedAgentBase):
    async def run_async(self, ...) -> AgentResult:
        # ...
        pydantic_output = SomeAgentOutput(data="some_value")
        return AgentResult(
            success=True,
            output_data=pydantic_output # <-- CORRECT: This is a Pydantic model instance
        )
```

**`metadata` Access Violation (Before)**:
```python
# In cv_conversion_utils.py or other modules
def some_function(structured_cv: StructuredCV, ...):
    # ...
    # VIOLATION: Direct dict-like access on a Pydantic model instance
    structured_cv.metadata['name'] = personal_info.name
```

**`metadata` Access Fix (After)**:
```python
# In cv_conversion_utils.py or other modules
def some_function(structured_cv: StructuredCV, ...):
    # ...
    # CORRECT: Accessing the 'extra' dictionary within the MetadataModel
    structured_cv.metadata.extra['name'] = personal_info.name
```

#### Implementation Checklist

-   \[ ] Review all `run_async` methods and ensure their `AgentResult` return values comply with the Pydantic validator.
-   \[ ] Create simple Pydantic output models for agents that currently return raw dictionaries.
-   \[ ] Grep the codebase for `structured_cv.metadata[` and refactor all instances to use `structured_cv.metadata.extra[`.
-   \[ ] Add a unit test that specifically tries to create an `AgentResult` with a raw dictionary to confirm the Pydantic validator raises a `TypeError`.

### Task C-04: Standardize Naming and Data Access

-   **Priority**: P3
-   **Root Cause**: `NI-01` - Inconsistent field names are used when accessing `JobDescriptionData` (e.g., `job_title` vs. `title`), reducing readability and increasing the risk of errors.
-   **Impacted Modules**:
    -   `src/models/data_models.py`
    -   `src/agents/enhanced_content_writer.py`
    -   Any other module interacting with `JobDescriptionData`.

#### Required Changes

1.  **Normalize `JobDescriptionData`**: Review the `JobDescriptionData` Pydantic model and establish a single, consistent name for each piece of information (e.g., consistently use `job_title`, not `title`). Remove any redundant or ambiguous fields.
2.  **Refactor All Accessors**: Search the entire codebase for usages of `JobDescriptionData` and update all attribute access to use the newly standardized field names.

#### Code Snippets

**In `src/models/data_models.py` (Standardized)**:
```python
class JobDescriptionData(BaseModel):
    """A structured representation of a parsed job description."""
    raw_text: str
    job_title: Optional[str] = None # <-- Standardized name
    company_name: Optional[str] = None # <-- Standardized name
    # ... other fields ...
```

**In an agent (Refactored)**:
```python
# ...
def some_method(self, job_data: JobDescriptionData):
    # CORRECT: Using the standardized field name
    title = job_data.job_title

    # INCORRECT (to be removed):
    # title = job_data.title
```

#### Implementation Checklist

-   \[ ] Review and finalize the field names in the `JobDescriptionData` model in `src/models/data_models.py`.
-   \[ ] Globally search for all instances where `JobDescriptionData` objects are accessed.
-   \[ ] Update all access patterns to use the standardized field names.
-   \[ ] Run the test suite, paying close attention to tests involving job description parsing and content generation, to catch any `AttributeError` regressions.

---

## 5. Agent & Orchestration (Continued)

### Task A-04: Simplify Agent-Level Error Handling and Centralize Recovery

-   **Priority**: P2
-   **Root Cause**: `DU-02` - The base agent method `execute_with_context` contains a complex retry loop and error handling logic, which overlaps with the responsibilities of the `ErrorRecoveryService` and the LangGraph orchestrator.
-   **Impacted Modules**:
    -   `src/agents/agent_base.py`
    -   `src/orchestration/cv_workflow_graph.py` (and its nodes)
    -   `src/services/error_recovery.py`

#### Required Changes

1.  **Simplify `EnhancedAgentBase.execute_with_context`**: This method should be removed or significantly simplified. Its primary role should be to invoke the agent's core logic (`run_async`) and wrap the result, not to manage retries. The complex `while` loop for retries should be eliminated from the base agent.
2.  **Delegate Retry Logic to Orchestrator**: Leverage LangGraph's built-in capabilities for retrying nodes upon failure. The agent node itself should fail by raising an exception. The graph will be configured to catch this and decide whether to retry the node.
3.  **Centralize Recovery Decisions**: The `ErrorRecoveryService` should be invoked from the orchestration layer, not from within the base agent. When an agent node fails, the graph can route to a dedicated `error_handler_node` which uses the `ErrorRecoveryService` to decide the next step (e.g., retry, skip, or terminate).
4.  **Agents Should Fail Fast**: Agent methods (`run_async`, `run_as_node`) should re-raise exceptions after logging them, rather than catching them and returning a failure `AgentResult`. This allows the orchestrator to have full control over the execution flow.

#### Code Snippets

**Before (`src/agents/agent_base.py`)**:
```python
class EnhancedAgentBase(ABC):
    async def execute_with_context(self, input_data: Any, context: AgentExecutionContext, max_retries: int = 3) -> AgentResult:
        # ...
        retry_count = 0
        while retry_count <= max_retries: # <-- Complex retry logic inside the agent
            try:
                result = await self.run_async(input_data, context)
                # ...
                return result
            except Exception as e:
                # ...
                # Agent decides on recovery action
                recovery_action = await self.error_recovery_service.handle_error(...)
                if recovery_action.strategy.value == "retry":
                    retry_count += 1
                    continue # <-- Agent manages its own retry loop
                # ...
                return AgentResult(...) # Returns a failure object
```

**After (Conceptual)**:
The `execute_with_context` method is removed. The agent's `run_as_node` becomes the primary entry point for the orchestrator.

**In `src/agents/parser_agent.py` (Refactored `run_as_node`)**:
```python
class ParserAgent(EnhancedAgentBase):
    async def run_as_node(self, state: AgentState) -> Dict[str, Any]:
        try:
            # ... core agent logic ...
            job_data = await self._parse_job_description(...)
            structured_cv = await self._parse_cv_with_llm(...)
            return {"job_description_data": job_data, "structured_cv": structured_cv}
        except Exception as e:
            logger.error(f"ParserAgent failed: {e}", exc_info=True)
            # Fail fast by raising the exception for the orchestrator to handle
            raise AgentExecutionError(agent_name="ParserAgent", message=str(e)) from e
```

**In `src/orchestration/cv_workflow_graph.py` (Conceptual)**:
```python
# ...
# LangGraph can be configured to retry nodes on specific exceptions
graph_with_retries = some_graph.with_config(
    {"recursion_limit": 5},
    retry_on=(AgentExecutionError,)
)

# Or, a dedicated error handling node can be used
def error_handler_node(state: AgentState) -> Dict[str, Any]:
    # ... logic to inspect state.error_messages ...
    # ... call error_recovery_service ...
    # ... decide on next step (e.g., route back to the failing node or end) ...
    return {"user_feedback": ...} # Or other state updates

workflow.add_node("error_handler", error_handler_node)
# ... graph edges to route to the error handler ...
```

#### Implementation Checklist

-   \[ ] Remove the `execute_with_context` method from `EnhancedAgentBase`.
-   \[ ] Refactor all agent `run_as_node` methods to use a single `try...except` block that re-raises a specific `AgentExecutionError`.
-   \[ ] Remove the direct dependency on `ErrorRecoveryService` from within the agents' primary execution path.
-   \[ ] Configure the LangGraph workflow in `cv_workflow_graph.py` to handle `AgentExecutionError`, either through built-in retries or by routing to a dedicated error-handling node.
-   \[ ] Add integration tests to simulate an agent failure and verify that the graph-level retry/recovery mechanism is triggered correctly.