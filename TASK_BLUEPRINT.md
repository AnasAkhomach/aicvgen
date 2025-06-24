# TASK_BLUEPRINT: AI CV Generator Refactoring & Remediation (Revision 2)

## Overview

This document outlines the prioritized tasks for refactoring the `anasakhomach-aicvgen` codebase. The primary mission is to remediate critical technical debt, harden the system against failures, and enforce architectural consistency. This revision incorporates clarifications on the existing architecture.

**Architectural Principles to Enforce:**
- **Fail-fast & Recover Gracefully:** Detect and handle errors at the point of failure.
- **Single Source of Truth:** Unify configuration, data models, and utility functions.
- **Strict Contracts:** Enforce data contracts between components using Pydantic models.
- **Clear Separation of Concerns:** Isolate business logic (agents), orchestration (LangGraph), and services.
- **Asynchronous Best Practices:** Eliminate blocking I/O in asynchronous contexts.

---

## State Management Architecture

Based on review, the following state management architecture is validated and will be adhered to:
1.  **UI State (`st.session_state`):** Holds raw user inputs (e.g., text from `st.text_area`) and simple UI flags.
2.  **Workflow State (`AgentState`):** A canonical Pydantic model (`src/orchestration/state.py`) that serves as the single source of truth for a workflow run.
3.  **State Initialization:** The `create_initial_agent_state` function in `src/utils/state_utils.py` is the designated factory for creating the `AgentState` object from UI inputs at the start of a workflow.
4.  **Data Flow:**
    -   A `start_cv_generation` callback in `src/frontend/callbacks.py` creates an `AgentState` object.
    -   This `AgentState` object is passed to a background thread for processing by the LangGraph workflow.
    -   Upon completion, the background thread writes the final, mutated `AgentState` object back into `st.session_state.agent_state`.
    -   The UI re-renders based on the contents of the updated `st.session_state.agent_state`.

This pattern is accepted as a pragmatic approach for managing state in a Streamlit application with background processing.

---

## 1. Services Layer (`src/services/`)

### **Task S-01 (P1): Fortify LLM JSON Parsing Service**

-   **Root Cause:** The `llm_cv_parser_service.py` performs unsafe `json.loads()` on raw LLM responses, which are not guaranteed to be valid JSON. This is the direct cause of the `json.decoder.JSONDecodeError` that crashes the workflow.
-   **Impacted Modules:** `src/services/llm_cv_parser_service.py`, `src/agents/parser_agent.py`.
-   **Required Changes:**
    1.  The `_generate_and_parse_json` method in `llm_cv_parser_service.py` must be made resilient.
    2.  It must validate that the LLM response is a non-empty string before attempting to parse.
    3.  It must robustly extract the JSON object from the raw response, handling common markdown code fences (e.g., ` ```json ... ``` `).
    4.  The `json.loads()` call must be wrapped in a `try...except json.JSONDecodeError` block.
    5.  On parsing failure, the method must log the malformed response and raise a specific `LLMResponseParsingError`, which the calling agent can handle.

-   **Code Implementation (`src/services/llm_cv_parser_service.py`):**

    **Before:**
    ```python
    # In LLMCVParserService._generate_and_parse_json
    async def _generate_and_parse_json(
        self, prompt: str, session_id: Optional[str], trace_id: Optional[str]
    ) -> Any:
        llm_response = await self.llm_service.generate(
            prompt=prompt,
            session_id=session_id,
            trace_id=trace_id,
        )
        return json.loads(llm_response.content) # Unsafe call
    ```

    **After:**
    ```python
    import re
    import json
    from ..utils.exceptions import LLMResponseParsingError

    # In LLMCVParserService._generate_and_parse_json
    async def _generate_and_parse_json(
        self, prompt: str, session_id: Optional[str], trace_id: Optional[str]
    ) -> Any:
        """Generates content and robustly parses the JSON output."""
        llm_response = await self.llm_service.generate(
            prompt=prompt,
            session_id=session_id,
            trace_id=trace_id,
        )

        raw_text = llm_response.content
        if not raw_text or not isinstance(raw_text, str):
            raise LLMResponseParsingError("Received empty or non-string response from LLM.", raw_response=str(raw_text))

        # Use regex to find JSON within ```json ... ``` code blocks
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback for raw JSON or JSON embedded in text
            start_index = raw_text.find("{")
            end_index = raw_text.rfind("}") + 1
            if start_index == -1 or end_index == 0:
                raise LLMResponseParsingError("No valid JSON object found in the LLM response.", raw_response=raw_text)
            json_str = raw_text[start_index:end_index]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON from LLM response: %s", e)
            logger.debug("Malformed JSON string: %s", json_str)
            raise LLMResponseParsingError(
                f"Could not parse JSON from LLM response. Error: {e}", raw_response=raw_text
            ) from e
    ```

-   **Implementation Checklist:**
    -   [ ] Update `_generate_and_parse_json` in `llm_cv_parser_service.py` with the robust implementation.
    -   [ ] Ensure `parser_agent.py` catches `LLMResponseParsingError` and handles it by updating the `AgentState` with an error message instead of crashing.
    -   [ ] Add comprehensive logging for parsing failures, including the raw response payload.

-   **Testing:**
    -   [ ] Create a unit test for `_generate_and_parse_json` that mocks `llm_service.generate`.
    -   [ ] Test case 1: Mock response is valid JSON.
    -   [ ] Test case 2: Mock response is JSON wrapped in ` ```json ... ``` `.
    -   [ ] Test case 3: Mock response is an empty string. Assert `LLMResponseParsingError` is raised.
    -   [ ] Test case 4: Mock response is malformed JSON. Assert `LLMResponseParsingError` is raised.

---

## 2. Orchestration Layer (`src/orchestration/`)

### **Task O-01 (P1): Implement Workflow Resilience to Node Failures**

-   **Root Cause:** An unhandled exception in any node (e.g., `parser_node`) causes the entire LangGraph workflow to crash, leaving the application in a hung state.
-   **Impacted Modules:** `src/orchestration/cv_workflow_graph.py`.
-   **Required Changes:**
    1.  Introduce a centralized `error_handler_node` in the graph.
    2.  Modify all conditional routing functions (e.g., `should_continue_generation`) to first check for the presence of messages in `state.error_messages`.
    3.  If errors exist, the router must direct the workflow to the `error_handler_node`.
    4.  The `error_handler_node` will log the final state, perform any necessary cleanup, and transition the graph to `END`. This prevents the exception from propagating out of the `cv_graph_app.ainvoke` call.

-   **Code Implementation (`src/orchestration/cv_workflow_graph.py`):**

    **Before (Router):**
    ```python
    def should_continue_generation(state: Dict[str, Any]) -> str:
        agent_state = AgentState.model_validate(state)
        if agent_state.content_generation_queue:
            return "continue"
        return "complete"
    ```

    **After (Router):**
    ```python
    def should_continue_generation(state: Dict[str, Any]) -> str:
        """Router function to determine if content generation loop should continue."""
        agent_state = AgentState.model_validate(state)

        # Priority 1: Check for errors first.
        if agent_state.error_messages:
            logger.error("Errors detected in state, routing to error_handler_node.")
            return "error"

        # Priority 2: Check for user feedback for regeneration.
        if agent_state.user_feedback and agent_state.user_feedback.action == UserAction.REGENERATE:
             logger.info("User requested regeneration, routing to prepare_regeneration.")
             return "regenerate"

        # Continue with normal workflow.
        if agent_state.content_generation_queue:
            logger.info("Content generation queue has items, continuing loop.")
            return "continue"

        logger.info("Content generation queue is empty, completing workflow.")
        return "complete"
    ```

    **Graph Definition:**
    ```python
    # In build_cv_workflow_graph()

    # Add the error handler node
    workflow.add_node("error_handler", error_handler_node)

    # Add conditional edge from a processing node (e.g., qa_node)
    workflow.add_conditional_edges(
        "qa", # Source node
        should_continue_generation, # Router function
        {
            "error": "error_handler", # New path for errors
            "regenerate": "prepare_regeneration",
            "continue": "pop_next_item",
            "complete": "formatter",
        },
    )

    # Ensure the error handler is a terminal node
    workflow.add_edge("error_handler", END)
    ```

-   **Implementation Checklist:**
    -   [ ] Implement the `error_handler_node` function in `cv_workflow_graph.py`. It should log the final state and prepare for graceful termination.
    -   [ ] Update all conditional routing functions to check `state.error_messages` first and route to the error handler.
    -   [ ] Update the graph definition to include the new node and conditional edges.

-   **Testing:**
    -   [ ] Add an integration test for the workflow graph.
    -   [ ] Mock the `parser_agent` to raise an `AgentExecutionError`.
    -   [ ] Invoke the graph and assert that the final node executed was `error_handler_node`.
    -   [ ] Assert that the `ainvoke` call completes without raising an exception.

---

## 3. Data Models (`src/models/`)

### **Task M-01 (P2): Consolidate Agent-Specific Output Models**

-   **Root Cause:** Proliferation of agent-specific Pydantic models (e.g., `formatter_agent_models.py`, `cv_analyzer_models.py`) violates the DRY principle and increases coupling.
-   **Impacted Modules:** `src/models/`, all agent implementations.
-   **Required Changes:**
    1.  Create a new central module `src/models/agent_output_models.py`.
    2.  Define canonical output models within this new module for each agent (e.g., `ParserAgentOutput`, `EnhancedContentWriterOutput`).
    3.  Refactor all agents' `run_async` methods to return `AgentResult` where the `output_data` field is an instance of one of these new canonical models.
    4.  Delete the old, redundant model files (`formatter_agent_models.py`, `quality_assurance_agent_models.py`, etc.).

-   **Code Implementation (`src/models/agent_output_models.py`):**
    ```python
    # In src/models/agent_output_models.py
    from typing import Optional, List, Dict, Any
    from pydantic import BaseModel, Field
    from .data_models import JobDescriptionData, StructuredCV
    from .quality_assurance_agent_models import QualityAssuranceResult
    from .cv_analysis_result import CVAnalysisResult
    from .cv_analyzer_models import BasicCVInfo

    class ParserAgentOutput(BaseModel):
        """Standardized output for the ParserAgent."""
        job_description_data: Optional[JobDescriptionData] = Field(default=None)
        structured_cv: Optional[StructuredCV] = Field(default=None)

    class CVAnalyzerAgentOutput(BaseModel):
        """Standardized output for the CVAnalyzerAgent."""
        analysis_results: CVAnalysisResult
        extracted_data: BasicCVInfo # Fallback data

    # ... define other canonical output models ...
    ```

-   **Implementation Checklist:**
    -   [ ] Create `src/models/agent_output_models.py`.
    -   [ ] Define all required output models in the new file.
    -   [ ] Refactor `ParserAgent` to use `ParserAgentOutput`.
    -   [ ] Refactor `CVAnalyzerAgent` to use `CVAnalyzerAgentOutput`.
    -   [ ] Continue for all other agents.
    -   [ ] Delete the now-unused model files.

-   **Testing:**
    -   [ ] Update unit tests for each agent to assert that the `run_async` method returns an `AgentResult` containing the correct Pydantic output model.

---

## 4. Configuration & Code Hygiene

### **Task C-01 (P3): Unify Linter Configuration and Remove Orphaned Files**

-   **Root Cause:** The project contains two conflicting `.pylintrc` files and several orphaned/backup files, creating ambiguity and risk.
-   **Impacted Files:** `/.pylintrc`, `/config/.pylintrc`, `/emergency_fix.py`, `/userinput.py`, `/src/core/dependency_injection.py.backup`.
-   **Required Changes:**
    1.  Delete `config/.pylintrc`.
    2.  Review the root `.pylintrc` and ensure `py-version` is set to `3.11` to match the `Dockerfile`.
    3.  Delete `emergency_fix.py`.
    4.  Delete `userinput.py`.
    5.  Delete `src/core/dependency_injection.py.backup`.

-   **Implementation Checklist:**
    -   [ ] Delete `/config/.pylintrc`.
    -   [ ] Verify `py-version=3.11` in the root `/.pylintrc`.
    -   [ ] Delete the three specified orphaned/backup files.
    -   [ ] Run `pylint src/` from the root directory to confirm the configuration is working as expected.

-   **Testing:**
    -   N/A (This is a code hygiene task, validated by CI pipeline).

---

## Implementation Checklist

1.  [ ] **Branch:** Create a new feature branch `refactor/architecture-remediation-rev2`.
2.  [ ] **Task S-01:** Implement robust JSON parsing in `llm_cv_parser_service`.
3.  [ ] **Task O-01:** Implement error handling and resilience in the `cv_workflow_graph`.
4.  [ ] **Testing (P1):** Add unit and integration tests for the P1 fixes (S-01, O-01).
5.  [ ] **Task M-01:** Consolidate agent-specific output models into `agent_output_models.py`.
6.  [ ] **Task C-01:** Remove redundant configurations and orphaned files.
7.  [ ] **Code Review:** Submit a pull request for review.
8.  [ ] **Merge:** Merge the changes into the main development branch after approval.

## Testing Strategy

-   **Unit Tests:** Each refactored component must have corresponding unit tests. For services and agents, mock all external dependencies (e.g., `llm_service`). Test both "happy path" and failure scenarios (e.g., invalid input, exceptions from dependencies).
-   **Integration Tests:** Add integration tests for the `cv_workflow_graph`. These tests should:
    -   Run the full graph with mocked agents.
    -   Simulate an agent failure (e.g., `parser_node` raising an exception) and assert that the graph routes to the `error_handler_node` and terminates gracefully.
    -   Validate the end-to-end data flow through the state object.
-   **Code Coverage:** The CI pipeline should be configured to report on code coverage, with a target of >80% for all new and refactored code.

## Clarifications & Deferred Tasks

-   **Async Task Execution:** The `asyncio.new_event_loop()` pattern in `callbacks.py` is a valid approach. The critical issue is the unhandled exception from the graph, which is addressed by **Task O-01**. The recommendation to use `run_coroutine_threadsafe` (from `BE-04` in the audit) is a potential future optimization and is deferred.
-   **LLM Service API Key Coupling (Deferred):** The `application_startup` service reads the API key directly from `st.session_state`, creating a coupling between the core and the frontend. While this is a valid architectural concern, it is out of scope for this immediate refactoring phase. A future task should be created to pass the API key as an explicit argument during application initialization to improve decoupling and testability.

---

Excellent. The clarifications confirm the core architectural patterns and allow for a more precise and focused remediation plan. The existing plan is solid, but based on a deeper analysis of the interplay between `llm_service.py` and `llm_retry_handler.py`, a critical asynchronous contract violation has been identified. This needs to be prioritized alongside the other P1 tasks.

Here is the updated `TASK_BLUEPRINT.md` with the new P1 task and architectural notes.

---
*Continuing from the previous plan...*

## 5. Architectural Notes & Validated Patterns

This section codifies the architectural decisions confirmed during the audit and clarification phase. All new and refactored code must adhere to these patterns.

### 5.1. State Management Architecture

-   **UI State (`st.session_state`):** This is the primary mechanism for preserving state across Streamlit reruns. It should **only** hold:
    -   Raw user inputs (e.g., `job_description_input`, `cv_text_input`).
    -   Simple UI flags (e.g., `is_processing`, `api_key_validated`).
    -   The final, complete `AgentState` object, which is written back by the background workflow thread.
-   **Workflow State (`AgentState`):** The canonical Pydantic model found in `src/orchestration/state.py` is the **single source of truth** for any given workflow execution. It is immutable during an agent's execution; agents receive it as input and return a dictionary of the fields they have changed.
-   **Data Flow:**
    1.  The `start_cv_generation` callback in `src/frontend/callbacks.py` is the entry point.
    2.  It uses the `create_initial_agent_state` factory (`src/utils/state_utils.py`) to construct the initial `AgentState` from UI inputs.
    3.  This `AgentState` instance is passed to the background workflow thread.
    4.  The workflow thread executes the LangGraph, which mutates the state.
    5.  Upon completion (or graceful failure), the thread writes the final `AgentState` object back to `st.session_state.agent_state`.
    6.  The Streamlit UI re-renders based on the new data in `st.session_state.agent_state`.

### 5.2. Asynchronous Task Execution

-   The pattern of creating a new `asyncio` event loop within a dedicated `threading.Thread` (as seen in `src/frontend/callbacks.py`) is the accepted method for integrating the asynchronous LangGraph workflow with the synchronous Streamlit framework.
-   The root cause of UI hanging is not the threading model but the unhandled exceptions from the workflow. **Task O-01** is the designated solution for this problem.

---

## 6. Services Layer (`src/services/`) - Additional P1 Task

### **Task S-02 (P1): Fix Asynchronous Call Contract in `EnhancedLLMService`**

-   **Root Cause:** `EnhancedLLMService._make_llm_api_call` is a synchronous method (`def`) that incorrectly calls an asynchronous method (`llm_retry_handler.generate_content`) without using `await`. This returns a coroutine object instead of the expected result. The subsequent wrapping in a `ThreadPoolExecutor` (`run_in_executor`) is an anti-pattern for an already-asynchronous function, masking the underlying issue and causing unpredictable behavior.
-   **Impacted Modules:** `src/services/llm_service.py`, `src/services/llm_retry_handler.py`.
-   **Required Changes:**
    1.  Remove the `ThreadPoolExecutor` from the `EnhancedLLMService` constructor. It is not needed for `async` SDK calls.
    2.  Refactor the `generate_content` method to directly `await` the retryable call and wrap it with `asyncio.wait_for` for timeout management.
    3.  Eliminate the unnecessary `_generate_with_timeout` and `_make_llm_api_call` helper methods. The logic should be consolidated within `generate_content` and a new, correctly awaited `_call_llm_with_retry` method.

-   **Code Implementation (`src/services/llm_service.py`):**

    **Before:**
    ```python
    # In EnhancedLLMService
    def __init__(...):
        # ...
        self.executor = concurrent.futures.ThreadPoolExecutor(...)

    async def _generate_with_timeout(...):
        # ... uses run_in_executor with a synchronous wrapper ...
        executor_task = loop.run_in_executor(
            self.executor, self._make_llm_api_call, prompt
        )
        result = await asyncio.wait_for(executor_task, ...)
        return result

    def _make_llm_api_call(self, prompt: str) -> Any:
        # BUG: This is not awaited! It returns a coroutine.
        response = self.llm_retry_handler.generate_content(prompt)
        return response
    ```

    **After:**
    ```python
    # In EnhancedLLMService
    def __init__(self, settings, llm_client, llm_retry_handler, cache, timeout, ...):
        # ...
        # self.executor is removed
        self.llm_retry_handler = llm_retry_handler
        self.timeout = timeout
        # ...

    async def _call_llm_with_retry(self, prompt: str, trace_id: str) -> Any:
        """Correctly awaits the async retry handler."""
        logger.info("Calling LLM via retry handler.", trace_id=trace_id)
        # The retry logic is encapsulated in the handler
        return await self.llm_retry_handler.generate_content(prompt)

    async def generate_content(self, prompt: str, ...) -> LLMResponse:
        """Main entry point for generating content with timeout and retries."""
        # ... (cache check, rate limiting logic remains the same) ...
        try:
            # Directly await the retryable call and wrap with a timeout
            response = await asyncio.wait_for(
                self._call_llm_with_retry(prompt, trace_id),
                timeout=self.timeout
            )
            # ... (process successful response) ...
        except asyncio.TimeoutError as e:
            logger.error(
                "LLM request timed out after %s seconds", self.timeout,
                extra={"trace_id": trace_id, "prompt_length": len(prompt)}
            )
            raise OperationTimeoutError(
                f"LLM request timed out after {self.timeout} seconds"
            ) from e
        except Exception as e:
            # ... (general error handling and fallback logic) ...
            return await self._handle_error_with_fallback(...)
    ```

-   **Implementation Checklist:**
    -   [ ] Remove `self.executor` from `EnhancedLLMService`.
    -   [ ] Remove the `_generate_with_timeout` and `_make_llm_api_call` methods.
    -   [ ] Implement the new `_call_llm_with_retry` `async` method.
    -   [ ] Refactor `generate_content` to use `asyncio.wait_for` on `_call_llm_with_retry`.

-   **Testing:**
    -   [ ] Create a unit test for `EnhancedLLMService.generate_content`.
    -   [ ] Mock `llm_retry_handler.generate_content` to be an `AsyncMock`.
    -   [ ] Test Case 1 (Success): Assert `llm_retry_handler.generate_content` is awaited.
    -   [ ] Test Case 2 (Timeout): Mock the handler to `asyncio.sleep` longer than the timeout and assert that `OperationTimeoutError` is raised.

## Updated Implementation Checklist

1.  [ ] **Branch:** Create a new feature branch `refactor/architecture-remediation-rev2`.
2.  [ ] **Task S-01:** Implement robust JSON parsing in `llm_cv_parser_service`.
3.  [ ] **Task S-02:** Fix the asynchronous call contract in `EnhancedLLMService`.
4.  [ ] **Task O-01:** Implement error handling and resilience in the `cv_workflow_graph`.
5.  [ ] **Testing (P1):** Add unit and integration tests for all P1 fixes (S-01, S-02, O-01).
6.  [ ] **Task M-01:** Consolidate agent-specific output models into `agent_output_models.py`.
7.  [ ] **Task C-01:** Remove redundant configurations and orphaned files.
8.  [ ] **Code Review:** Submit a pull request for review.
9.  [ ] **Merge:** Merge the changes into the main development branch after approval.

---

The plan is comprehensive but can be improved by adding concrete tasks for the remaining architectural drift issues identified in the audit. Specifically, the proliferation of error-handling utilities, the inconsistent use of the dependency injection container, and misplaced factory logic in the `ParserAgent` must be addressed to fully stabilize the architecture.

Here is the continuation of the `TASK_BLUEPRINT.md` with these additional tasks.

---

## 7. Core Architecture & Dependency Injection (`src/core/`)

### **Task DI-01 (P2): Centralize Dependency Injection Container**

-   **Root Cause:** The `DependencyContainer` is instantiated in multiple places (e.g., `application_startup.py`, `cv_workflow_graph.py`), which undermines the singleton pattern and leads to inconsistent service instances across the application. This is a critical architectural flaw.
-   **Impacted Modules:** `src/core/dependency_injection.py`, `src/core/application_startup.py`, `src/orchestration/cv_workflow_graph.py`.
-   **Required Changes:**
    1.  Establish `application_startup.py` as the **single source of truth** for container instantiation and configuration.
    2.  The `get_container()` function in `dependency_injection.py` must be refactored to manage and return a single, global instance of the container, preventing re-initialization.
    3.  All other parts of the application, especially `cv_workflow_graph.py`, must be refactored to call the global `get_container()` function instead of creating their own instances.

-   **Code Implementation (`src/core/dependency_injection.py`):**

    **Before (potential for multiple instances):**
    ```python
    # In dependency_injection.py
    _global_container: Optional[DependencyContainer] = None
    # ...
    def get_container(session_id: Optional[str] = None) -> "DependencyContainer":
        global _global_container
        # ... logic that might re-create _global_container
        if _global_container is None:
            _global_container = DependencyContainer(session_id)
        return _global_container

    # In cv_workflow_graph.py
    _CONTAINER = None
    def get_workflow_container():
        global _CONTAINER
        if _CONTAINER is None:
            _CONTAINER = DependencyContainer() # Creates a new, separate container
            configure_container(_CONTAINER)
        return _CONTAINER
    ```

    **After (guaranteed singleton):**
    ```python
    # In dependency_injection.py
    _global_container: Optional[DependencyContainer] = None
    _container_lock = threading.Lock()

    def get_container() -> "DependencyContainer":
        """Get the global dependency container. Initializes it on first call."""
        global _global_container
        if _global_container is None:
            with _container_lock:
                if _global_container is None:
                    _global_container = DependencyContainer()
                    # It's crucial that configure_container is called only once,
                    # ideally at application startup.
        return _global_container

    # In application_startup.py
    def initialize_application(...):
        # ...
        container = get_container()
        if not container.get_registrations(): # Check if not already configured
             configure_container(container)
        # ...

    # In cv_workflow_graph.py
    def get_workflow_agents():
        # ...
        container = get_container() # Use the global instance
        # ...
    ```

-   **Implementation Checklist:**
    -   [ ] Refactor `get_container()` in `dependency_injection.py` to be a thread-safe singleton provider.
    -   [ ] Remove the local container instantiation logic from `cv_workflow_graph.py` and have it use the global `get_container()`.
    -   [ ] Ensure `application_startup.py` is the sole location for calling `configure_container`.

-   **Testing:**
    -   [ ] Write a unit test that calls `get_container()` from two different mock modules and asserts that the returned object `id()` is the same.
    -   [ ] Verify that integration tests for the workflow still pass, ensuring agents receive their dependencies correctly from the single container.

---

## 8. Utilities Layer (`src/utils/`)

### **Task U-01 (P2): Consolidate Error Handling Framework**

-   **Root Cause:** The codebase has at least six different modules related to error handling (`agent_error_handling.py`, `error_boundaries.py`, `error_classification.py`, `error_handling.py`, `error_utils.py`, `exceptions.py`). This fragmentation leads to redundancy, inconsistency, and high maintenance overhead.
-   **Impacted Modules:** The entire `src/utils/` directory, and any module that imports from these error handlers.
-   **Required Changes:**
    1.  **Designate Canonical Modules:**
        -   `src/utils/exceptions.py`: Will remain the single source of truth for all custom exception class definitions (`AicvgenError`, `LLMResponseParsingError`, etc.).
        -   `src/utils/error_classification.py`: Will contain all functions for classifying errors (e.g., `is_rate_limit_error`).
        -   `src/utils/error_handling.py`: Will be the **new canonical module** for the `ErrorHandler` class, decorators (`@handle_errors`), and context managers (`ErrorBoundary`).
    2.  **Consolidate Logic:**
        -   Merge the logic from `agent_error_handling.py`, `error_boundaries.py`, and `error_utils.py` into the refactored `error_handling.py`.
    3.  **Refactor Call Sites:** Update all `import` statements across the codebase to point to the new canonical modules.
    4.  **Deprecate and Delete:** After refactoring, delete the redundant files: `agent_error_handling.py`, `error_boundaries.py`, and `error_utils.py`.

-   **Implementation Checklist:**
    -   [ ] Refactor `src/utils/error_handling.py` to be the central error handling service.
    -   [ ] Move relevant functions and classes from other error modules into `error_handling.py`.
    -   [ ] Globally search and replace imports to point to the new canonical locations.
    -   [ ] Delete the now-empty/redundant error utility files.

-   **Testing:**
    -   [ ] Existing tests that rely on error handling should be updated and must pass.
    -   [ ] Add new unit tests for the consolidated `ErrorHandler` class to ensure all previous functionalities are preserved.

---

## 9. Agents Layer (`src/agents/`)

### **Task A-01 (P3): Refactor `create_empty_cv_structure` out of `ParserAgent`**

-   **Root Cause:** The `ParserAgent` contains logic for creating an empty `StructuredCV` when the "start from scratch" option is used. This violates the Single Responsibility Principle, as an agent's job is to *process* data, not *create* initial data structures.
-   **Impacted Modules:** `src/agents/parser_agent.py`, `src/models/data_models.py`.
-   **Required Changes:**
    1.  Move the `create_empty_cv_structure` logic from `ParserAgent` to a more appropriate location. The best practice is a static factory method on the `StructuredCV` model itself.
    2.  Refactor the `ParserAgent` and any other call sites to use `StructuredCV.create_empty(...)`.

-   **Code Implementation:**

    **In `src/models/data_models.py`:**
    ```python
    # Inside the StructuredCV class
    @staticmethod
    def create_empty(job_data: Optional[JobDescriptionData] = None) -> "StructuredCV":
        """
        Creates an empty CV structure for the "Start from Scratch" option.
        """
        # ... (The logic from parser_agent.create_empty_cv_structure goes here) ...
        structured_cv = StructuredCV()
        # ...
        return structured_cv
    ```

    **In `src/agents/parser_agent.py`:**
    ```python
    # In _process_cv method
    async def _process_cv(self, state: AgentState, job_data: JobDescriptionData) -> StructuredCV:
        # ...
        if start_from_scratch:
            logger.info("Creating empty CV for 'Start from Scratch'.")
            # Use the new factory method
            return StructuredCV.create_empty(job_data)
        # ...
    ```

-   **Implementation Checklist:**
    -   [ ] Add the `create_empty` static method to the `StructuredCV` model in `data_models.py`.
    -   [ ] Copy the implementation logic from `parser_agent.py` into the new method.
    -   [ ] Delete the `create_empty_cv_structure` method from `ParserAgent`.
    -   [ ] Update the `_process_cv` method in `ParserAgent` to call `StructuredCV.create_empty()`.

-   **Testing:**
    -   [ ] Create a unit test for `StructuredCV.create_empty` to ensure it produces a valid, empty CV structure.
    -   [ ] Update the integration test for the parser agent's "start from scratch" path to ensure it still functions correctly.

---

