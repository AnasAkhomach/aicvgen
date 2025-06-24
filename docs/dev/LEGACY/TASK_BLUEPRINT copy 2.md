# TASK_BLUEPRINT.md: anasakhomach-aicvgen Stabilization

## 1. Overview

This document outlines the prioritized tasks for remediating technical debt and stabilizing the `anasakhomach-aicvgen` system. The plan is divided into three phases:

*   **P1 (Stabilization):** Fix all critical runtime errors and contract breaches causing immediate failures.
*   **P2 (Consolidation & Refactoring):** Address architectural drift and code duplication to improve maintainability and reduce fragility.
*   **P3 (Structural Improvement):** Implement deeper changes to enforce best practices and prevent future technical debt.

Each task includes a root cause analysis, impacted components, and a detailed implementation plan with testing requirements.

---

## 2. Per-Component Remediation Plan

### 2.1. Services (`src/services/llm_service.py`)

#### Task 2.1.1: Consolidate Caching & Retry Logic (DP-01)

*   **Priority:** `P2`
*   **Root Cause:** Two independent caching systems (`AdvancedCache` class and `_response_cache` dict) and two retry-condition functions (`is_transient_error` and `_should_retry_exception`) exist, creating redundancy, complexity, and risk of inconsistent behavior.
*   **Impacted Modules:** `src/services/llm_service.py`
*   **Team Consensus:** The team agrees to consolidate on `AdvancedCache` due to its superior features (Persistence, TTL, LRU Eviction, Statistics).

*   **Required Changes:**
    1.  Remove the legacy `_response_cache` dictionary and its helper functions (`get_cached_response`, `set_cached_response`).
    2.  Refactor `generate_content` to use only the `self.cache` (`AdvancedCache` instance).
    3.  Consolidate `is_transient_error` and `_should_retry_exception` into a single, robust method within the `EnhancedLLMService` class.

*   **Code Snippets (Before/After):**

    *   **Before (Duplicated Caching):**
        ```python
        # src/services/llm_service.py

        # ...
        _response_cache = {} # Legacy cache

        class EnhancedLLMService:
            def __init__(self):
                self.cache = AdvancedCache() # Modern cache

            async def generate_content(self, prompt: str, ...):
                # Checks legacy cache
                cached = get_cached_response(...)
                if cached:
                    return cached

                # Checks modern cache
                cached = self.cache.get(...)
                if cached:
                    return cached

                # ... generation logic ...
        ```

    *   **After (Consolidated Caching):**
        ```python
        # src/services/llm_service.py

        class EnhancedLLMService:
            def __init__(self):
                self.cache = AdvancedCache()

            async def generate_content(self, prompt: str, ...):
                # Only checks the single, modern cache instance
                cache_key = self.cache._generate_cache_key(prompt, self.model_name)
                cached = self.cache.get(cache_key)
                if cached:
                    # ... return cached response

                # ... generation logic ...
        ```

*   **Implementation Checklist:**
    1.  [ ] Delete the global `_response_cache` dictionary from `llm_service.py`.
    2.  [ ] Delete the `get_cached_response` and `set_cached_response` helper functions.
    3.  [ ] Refactor the `generate_content` method to remove calls to the old cache functions.
    4.  [ ] Consolidate `is_transient_error` and `_should_retry_exception` into a private method `_is_retryable_error`.
    5.  [ ] Update the `@retry` decorator to use the new `_is_retryable_error` method.
    6.  [ ] Run all related unit tests to ensure no regressions.

*   **Tests to Add:**
    *   Unit test in `tests/unit/test_llm_service.py` to verify that a cached response is returned correctly from `AdvancedCache`.
    *   Unit test to confirm that `_response_cache` no longer exists and calls to it raise an error.
    *   Unit test for `_is_retryable_error` with various exception types.

---

#### Task 2.1.2: Fix Incorrect Executor Usage (AD-02)

*   **Priority:** `P1`
*   **Root Cause:** The `_generate_with_timeout` method calls `loop.run_in_executor(None, ...)` instead of `loop.run_in_executor(self.executor, ...)`, bypassing the configured thread pool and its settings (e.g., `max_workers`). This leads to unpredictable concurrency.
*   **Impacted Modules:** `src/services/llm_service.py`

*   **Required Changes:** Ensure the managed `self.executor` is used for all background blocking calls.

*   **Code Snippets (Before/After):**

    *   **Before:**
        ```python
        # src/services/llm_service.py in _generate_with_timeout

        loop = asyncio.get_event_loop()
        executor_task = loop.run_in_executor(None, self._make_llm_api_call, prompt)
        ```

    *   **After:**
        ```python
        # src/services/llm_service.py in _generate_with_timeout

        loop = asyncio.get_event_loop()
        # Correctly use the instance's configured executor
        executor_task = loop.run_in_executor(self.executor, self._make_llm_api_call, prompt)
        ```

*   **Implementation Checklist:**
    1.  [ ] Locate the `loop.run_in_executor(None, ...)` call in `llm_service.py`.
    2.  [ ] Change `None` to `self.executor`.
    3.  [ ] Verify that `self.executor` is correctly initialized in the `__init__` method.

*   **Tests to Add:**
    *   In `tests/unit/test_llm_service.py`, patch `loop.run_in_executor` and assert it's called with `self.executor` as the first argument.

---

### 2.2. Agents

#### Task 2.2.1: Fix `await` Misuse in `CVAnalyzerAgent` (CS-01)

*   **Priority:** `P1`
*   **Root Cause:** The `analyze_cv` method in `cv_analyzer_agent.py` uses the `await` keyword but is defined as a synchronous function (`def` instead of `async def`), which will cause a `SyntaxError` or `RuntimeError`.
*   **Impacted Modules:** `src/agents/cv_analyzer_agent.py`

*   **Required Changes:** Convert `analyze_cv` to an `async` function and ensure its callers `await` it.

*   **Code Snippets (Before/After):**

    *   **Before:**
        ```python
        # src/agents/cv_analyzer_agent.py

        def analyze_cv(self, input_data, context=None):
            # ...
            extracted_data = await self._generate_and_parse_json(prompt=prompt) # This will fail
            # ...
        ```

    *   **After:**
        ```python
        # src/agents/cv_analyzer_agent.py

        async def analyze_cv(self, input_data, context=None):
            # ...
            extracted_data = await self._generate_and_parse_json(prompt=prompt) # This is now valid
            # ...

        async def run_async(self, input_data: Any, context: "AgentExecutionContext") -> "AgentResult":
            # ...
            # Ensure the call to analyze_cv is awaited
            result = await self.analyze_cv(input_data, context)
            # ...
        ```

*   **Implementation Checklist:**
    1.  [ ] Change the signature of `analyze_cv` to `async def analyze_cv(...)`.
    2.  [ ] Review all call sites of `analyze_cv` (e.g., within `run_async`) and ensure they use `await`.
    3.  [ ] Add a regression test based on `test_await_issue.py` to the formal test suite.

*   **Tests to Add:**
    *   Unit test in `tests/unit/test_cv_analyzer_agent.py` that calls `run_async` and verifies it executes without a `SyntaxError` or `RuntimeError`.

---

#### Task 2.2.2: Fix Non-Existent Error Handler Call in `CleaningAgent` (CS-02)

*   **Priority:** `P1`
*   **Root Cause:** `cleaning_agent.py` calls `AgentErrorHandler.handle_agent_error`, a method which does not exist. This breaks the `try...except` block, masking the original error.
*   **Impacted Modules:** `src/agents/cleaning_agent.py`, `src/utils/agent_error_handling.py`

*   **Required Changes:** Identify the correct error handling method in `AgentErrorHandler` (likely `handle_general_error`) and update the call site in `CleaningAgent`.

*   **Code Snippets (Before/After):**

    *   **Before:**
        ```python
        # src/agents/cleaning_agent.py

        except Exception as e:
            # ...
            error_result = AgentErrorHandler.handle_agent_error(e, "CleaningAgent", context)
        ```

    *   **After (assuming correct method is `handle_general_error`):**
        ```python
        # src/agents/cleaning_agent.py

        except Exception as e:
            # ...
            fallback_data = AgentErrorHandler.create_fallback_data("cleaning")
            error_result = AgentErrorHandler.handle_general_error(
                e, "CleaningAgent", fallback_data, context
            )
        ```

*   **Implementation Checklist:**
    1.  [ ] Inspect `src/utils/agent_error_handling.py` to confirm the correct public method for handling general agent errors.
    2.  [ ] Replace the incorrect call in `src/agents/cleaning_agent.py`.
    3.  [ ] Ensure the new method call provides all required arguments (e.g., `fallback_data`).

*   **Tests to Add:**
    *   Unit test in `tests/unit/test_agent_error_handling.py` for `CleaningAgent` that mocks an exception within `run_async` and asserts that the correct `AgentErrorHandler` method is called.

---

#### Task 2.2.3: Fix Incorrect Error Handling in `FormatterAgent` (CB-01)

*   **Priority:** `P1`
*   **Root Cause:** The `except` block in `FormatterAgent`'s `run_as_node` method incorrectly handles the dictionary returned by `AgentErrorHandler.handle_node_error`. It attempts to access a `.error_message` attribute on a `dict`, causing an `AttributeError`.
*   **Impacted Modules:** `src/agents/formatter_agent.py`

*   **Required Changes:** Modify the `except` block in `FormatterAgent` to correctly access the `error_messages` list from the dictionary returned by the error handler.

*   **Code Snippets (Before/After):**

    *   **Before (Conceptual):**
        ```python
        # src/agents/formatter_agent.py

        async def run_as_node(self, state: AgentState, ...):
            try:
                # ... main logic ...
            except Exception as e:
                error_result = AgentErrorHandler.handle_node_error(e, "FormatterAgent", state)
                # This is the bug: error_result is a dict like {'error_messages': [...]}, not an object.
                error_list = state.error_messages or []
                error_list.append(f"Formatter error: {error_result.error_message}") # This will fail
                return {"error_messages": error_list}
        ```

    *   **After:**
        ```python
        # src/agents/formatter_agent.py

        async def run_as_node(self, state: AgentState, ...):
            try:
                # ... main logic ...
            except Exception as e:
                # AgentErrorHandler returns a dictionary to update the state.
                error_update_dict = AgentErrorHandler.handle_node_error(e, "FormatterAgent", state)
                return error_update_dict # Simply return the dictionary
        ```

*   **Implementation Checklist:**
    1.  [ ] Locate the `try...except` block in the `run_as_node` method of `formatter_agent.py`.
    2.  [ ] Modify the `except` block to correctly handle the dictionary returned by `handle_node_error`.
    3.  [ ] The simplest fix is to directly `return` the dictionary from the error handler, as it's already in the correct format for updating the LangGraph state.

*   **Tests to Add:**
    *   Add a unit test for `FormatterAgent` that mocks an exception during its execution and verifies that the returned dictionary contains the correct `error_messages` key and that no `AttributeError` is raised.

---

#### Task 2.2.4: Refactor `EnhancedContentWriterAgent` Parsing Logic (AD-01, CS-03)

*   **Priority:** `P2`
*   **Root Cause:** `EnhancedContentWriterAgent` is responsible for parsing raw CV text, which violates the Single Responsibility Principle. This logic belongs in the `ParserAgent`.
*   **Impacted Modules:** `src/agents/enhanced_content_writer.py`, `src/agents/parser_agent.py`, `src/orchestration/cv_workflow_graph.py`
*   **Data Contract:** `ParserAgent` must populate the `subsections` list of the "Professional Experience" `Section` in `StructuredCV`. Each `Subsection` should have its `.name` as "Title at Company (Date Range)" and its `.items` list populated with `Item` objects of type `BULLET_POINT`. No Pydantic model changes are needed.

*   **Implementation Checklist:**
    1.  [ ] **Analyze:** Identify the parsing logic in `EnhancedContentWriterAgent._parse_cv_text_to_content_item` and its helpers.
    2.  [ ] **Move Logic:** Copy these parsing methods into `src/agents/parser_agent.py`.
    3.  [ ] **Refactor `ParserAgent`:** In `ParserAgent.run_as_node`, after the initial CV parsing, call the new methods to populate the `StructuredCV.sections` list with the correct `Subsection` and `Item` objects for professional experience.
    4.  [ ] **Refactor `EnhancedContentWriterAgent`:** Remove the parsing methods (`_parse_cv_text_to_content_item`, `_extract_company_from_cv`). Update its logic to directly consume the `StructuredCV` object from the state, assuming the experience is already fully structured.
    5.  [ ] **Delete:** Remove the now-redundant parsing methods from `EnhancedContentWriterAgent`.

*   **Tests to Add:**
    *   Add unit tests to `tests/unit/test_parser_agent_refactored.py` to verify the new parsing capabilities. The tests should assert that the `StructuredCV` object produced by the parser contains correctly structured `Subsection` and `Item` objects.
    *   Add unit tests to `tests/unit/test_enhanced_content_writer.py` to verify it correctly consumes the structured data from a mock `AgentState`.
    *   Update integration tests in `tests/integration/test_agent_workflow_integration.py` to reflect the new data flow.

---


### 2.3. Data Models & Contracts

#### Task 2.3.1: Fix `section.name` vs. `section.title` Inconsistency (NI-01)

*   **Priority:** `P1`
*   **Root Cause:** A component is attempting to access `section.title` on a `Section` object, but the Pydantic model in `data_models.py` defines the attribute as `name`. This causes a runtime `AttributeError`.
*   **Impacted Modules:** Multiple agent files, `src/models/data_models.py`, `src/templates/pdf_template.html`.

*   **Required Changes:** Perform a global search and replace to standardize on `section.name`.

*   **Implementation Checklist:**
    1.  [ ] Use an IDE or command-line tool to find all occurrences of `.title` on objects that are likely `Section` instances.
    2.  [ ] Replace all incorrect `.title` access with `.name`.
    3.  [ ] **Crucially, inspect `src/templates/pdf_template.html`** and any other Jinja2 templates for `{{ section.title }}` and change it to `{{ section.name }}`.

*   **Tests to Add:**
    *   In `tests/unit/test_agent_state_contracts.py`, add a test that creates a `Section` object and asserts `hasattr(section, 'name')` is true and `hasattr(section, 'title')` is false. This will prevent regressions.

---

#### Task 2.3.2: Enforce Pydantic Models over `Dict[str, Any]` (CB-02)

*   **Priority:** `P3`
*   **Root Cause:** Widespread use of generic dictionaries instead of strongly-typed Pydantic models for data exchange between agents. This undermines type safety and leads to contract breaches and runtime errors.
*   **Impacted Modules:** All agent and service files.

*   **Required Changes:** Systematically refactor function signatures and return types to use specific Pydantic models (e.g., `JobDescriptionData`, `StructuredCV`) instead of `Dict[str, Any]`.

*   **Implementation Checklist:**
    1.  [ ] **Prioritize:** Start with the most critical data structures passed between agents in the `cv_workflow_graph`.
    2.  [ ] **Refactor:** For each targeted data structure, update the function signatures in the agent that consumes it.
    3.  [ ] **Update Logic:** Change dictionary key lookups (`.get("key")`) to Pydantic attribute access (`.key`).
    4.  [ ] **Use `.model_dump()`:** Only use `.model_dump()` at the boundaries where serialization is explicitly required (e.g., saving to a file, returning from a graph node if necessary).

*   **Tests to Add:**
    *   Add unit tests that pass Pydantic models to the refactored methods and assert correct behavior.
    *   Add tests using `pytest.raises(ValidationError)` to ensure that invalid data structures are rejected by the refactored methods.

---

### 2.4. Root Directory Cleanup

#### Task 2.4.1: Migrate Orphaned Debugging Scripts (DL-01)

*   **Priority:** `P3`
*   **Root Cause:** The root directory is cluttered with numerous `test_*.py` scripts created for ad-hoc debugging. These scripts contain valuable test cases but are not part of the formal test suite, making them hard to maintain and run.
*   **Impacted Modules:** Root directory, `tests/` directory.

*   **Implementation Checklist:**
    1.  [ ] **Review:** Go through each `test_*.py` script in the root directory.
    2.  [ ] **Extract:** Identify the core test case or scenario each script was designed to solve (e.g., `test_await_issue.py` was for `CS-01`).
    3.  [ ] **Migrate:** Rewrite the test case using `pytest` conventions and place it in the appropriate file under `tests/unit/` or `tests/integration/`.
    4.  [ ] **Delete:** Once the test case is successfully migrated and passing in the formal suite, delete the original script from the root directory.
    5.  [ ] **Update Runner:** Ensure `run_tests.py` or the CI script is configured to run the new tests.

*   **Tests to Add:**
    *   The primary goal is to migrate existing, informal tests into the formal `pytest` suite, thereby increasing overall test coverage and creating regression tests for the P1/P2 fixes.

---

## 3. Implementation Checklist

| Priority | Task ID | Description                                     | Status      |
| :------- | :------ | :---------------------------------------------- | :---------- |
| **P1**   | CS-01   | Fix `await` misuse in `CVAnalyzerAgent`         | `[ ] To Do` |
| **P1**   | CS-02   | Fix non-existent error handler call in `CleaningAgent` | `[ ] To Do` |
| **P1**   | CB-01   | Fix incorrect error handling in `FormatterAgent`| `[ ] To Do` |
| **P1**   | NI-01   | Fix `section.name` vs. `section.title`          | `[ ] To Do` |
| **P1**   | AD-02   | Fix incorrect `run_in_executor` usage         | `[ ] To Do` |
| **P2**   | DP-01   | Consolidate Caching in `EnhancedLLMService`     | `[ ] To Do` |
| **P2**   | AD-01   | Refactor parsing logic out of `ContentWriter`   | `[ ] To Do` |
| **P3**   | CB-02   | Enforce Pydantic Models over `Dict[str, Any]` | `[ ] To Do` |
| **P3**   | DL-01   | Migrate and delete orphaned debug scripts       | `[ ] To Do` |

---

## 4. Testing Strategy

1.  **Unit Testing:** Each fix must be accompanied by new or updated unit tests that specifically target the bug or refactoring. For example, the `await` fix (`CS-01`) must have a test that proves the async call chain works.
2.  **Integration Testing:** After P1 fixes are complete, run the full `test_agent_workflow_integration.py` to ensure the agents still work together correctly. This test must be updated to reflect the new data contracts from the refactoring tasks.
3.  **Regression Testing:** The test cases from the orphaned debug scripts (e.g., `test_await_issue.py`) must be converted into permanent regression tests in the `pytest` suite. This ensures that the critical bugs they were designed to catch do not reappear.
4.  **Static Analysis:** After applying fixes, run `pylint` using the provided `.pylintrc` configuration. The goal is to have zero new `E` (Error) or `W` (Warning) messages related to the changed code.

---

## 5. Critical Gaps & Questions

*   No critical gaps or questions remain at this time. The provided clarifications have been integrated into the plan.

---

### 2.5. LLM Service (`src/services/llm_service.py`) - Continued

#### Task 2.5.1: Clarify API Key Management Strategy (NI-02)

*   **Priority:** `P2`
*   **Root Cause:** The `EnhancedLLMService` class uses multiple, similarly named attributes for API keys (`user_api_key`, `primary_api_key`, `fallback_api_key`, `current_api_key`), creating ambiguity about which key is active and making the logic for key selection and fallback complex and error-prone.
*   **Impacted Modules:** `src/services/llm_service.py`

*   **Required Changes:**
    1.  Introduce a dedicated dataclass or Pydantic model (`APIKeyManager`) to encapsulate all keys and the selection logic.
    2.  Refactor `EnhancedLLMService` to use this manager, simplifying its state.
    3.  The service should have a single, clear property like `active_key` that is managed by the new component.

*   **Code Snippets (Before/After):**

    *   **Before (Ambiguous Attributes):**
        ```python
        # src/services/llm_service.py
        class EnhancedLLMService:
            def __init__(self, user_api_key: Optional[str] = None):
                self.settings = get_config()
                self.user_api_key = user_api_key
                self.primary_api_key = self.settings.llm.gemini_api_key_primary
                self.fallback_api_key = self.settings.llm.gemini_api_key_fallback
                self.current_api_key = self._select_active_key() # Complex logic here
                # ...
        ```

    *   **After (Centralized Key Management):**
        ```python
        # src/services/llm_service.py
        from pydantic import BaseModel, Field

        class APIKeyManager(BaseModel):
            user: Optional[str] = None
            primary: Optional[str] = Field(default_factory=lambda: get_config().llm.gemini_api_key_primary)
            fallback: Optional[str] = Field(default_factory=lambda: get_config().llm.gemini_api_key_fallback)

            @property
            def active_key(self) -> str:
                if self.user:
                    return self.user
                if self.primary:
                    return self.primary
                if self.fallback:
                    return self.fallback
                raise ConfigurationError("No API key available.")

            def switch_to_fallback(self) -> bool:
                if self.fallback and self.active_key != self.fallback:
                    self.user = None # Invalidate user/primary keys
                    self.primary = None
                    return True
                return False

        class EnhancedLLMService:
            def __init__(self, user_api_key: Optional[str] = None):
                self.key_manager = APIKeyManager(user=user_api_key)
                self.llm = self._configure_model()

            def _configure_model(self):
                genai.configure(api_key=self.key_manager.active_key)
                return genai.GenerativeModel(self.settings.llm_settings.default_model)
        ```

*   **Implementation Checklist:**
    1.  [ ] Define the `APIKeyManager` Pydantic model inside `llm_service.py`.
    2.  [ ] Refactor `EnhancedLLMService.__init__` to use `APIKeyManager`.
    3.  [ ] Replace all direct references to `self.user_api_key`, `self.primary_api_key`, etc., with `self.key_manager.active_key`.
    4.  [ ] Update the fallback logic (`_switch_to_fallback_key`) to use `self.key_manager.switch_to_fallback()`.

*   **Tests to Add:**
    *   Unit tests for `APIKeyManager` to verify `active_key` property logic with different key combinations.
    *   Unit test for `EnhancedLLMService` to confirm it correctly uses the key from `APIKeyManager`.

---

#### Task 2.5.2: Fix Asynchronous Contract of `generate_content` (CB-03)

*   **Priority:** `P1`
*   **Root Cause:** The `_generate_with_timeout` method is synchronous but is called from the `async` `generate_content` method in a way that suggests it's a coroutine. The use of `loop.run_in_executor` returns a `Future`, which is not directly awaitable in the same way as a native coroutine. This leads to `TypeError: object NoneType can't be awaited` when errors occur inside the executor thread.
*   **Impacted Modules:** `src/services/llm_service.py`

*   **Required Changes:** Refactor `_generate_with_timeout` to be a proper `async` method. The blocking I/O call (`_make_llm_api_call`) should be the only part delegated to `run_in_executor`.

*   **Code Snippets (Before/After):**

    *   **Before:**
        ```python
        # src/services/llm_service.py

        # This is a synchronous function
        def _generate_with_timeout(self, prompt: str, ...):
            # ... logic ...
            response = self._make_llm_api_call(prompt)
            # ...
            return response

        async def generate_content(self, prompt: str, ...):
            # ...
            # Incorrectly awaiting a function that is not a coroutine
            response = await self._generate_with_timeout(prompt, ...)
            # ...
        ```

    *   **After:**
        ```python
        # src/services/llm_service.py

        # This is now a proper async function
        async def _generate_with_timeout(self, prompt: str, ...) -> Any:
            loop = asyncio.get_event_loop()
            try:
                # The blocking call is correctly run in an executor and awaited
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor, self._make_llm_api_call, prompt
                    ),
                    timeout=self.timeout
                )
                return response
            except asyncio.TimeoutError:
                logger.error("LLM request timed out after %s seconds", self.timeout)
                raise TimeoutError(f"LLM request timed out after {self.timeout} seconds")

        async def generate_content(self, prompt: str, ...):
            # ...
            # The call is now correct because _generate_with_timeout is a coroutine
            response = await self._generate_with_timeout(prompt, ...)
            # ...
        ```

*   **Implementation Checklist:**
    1.  [ ] Change the signature of `_generate_with_timeout` to `async def`.
    2.  [ ] Move the `loop.run_in_executor` call inside `_generate_with_timeout`.
    3.  [ ] Wrap the executor call with `asyncio.wait_for` to handle timeouts correctly.
    4.  [ ] Ensure that `generate_content` continues to `await _generate_with_timeout`.
    5.  [ ] Consolidate the retry logic (`@retry`) onto the `generate_content` method, as it is the public entry point.

*   **Tests to Add:**
    *   Re-purpose the logic from `test_await_issue.py` and other debug scripts into a formal unit test in `tests/unit/test_llm_service_comprehensive.py`.
    *   The test should mock `_make_llm_api_call` to simulate success, failure, and timeout scenarios, asserting that `generate_content` returns a valid `LLMResponse` or raises the correct exception.

---

### 2.6. UI & Application Entry Point

#### Task 2.6.1: Implement Fail-Fast Gemini API Key Validation (CF-01)

*   **Priority:** `P1`
*   **Root Cause:** The application only discovers an invalid API key deep within the workflow execution, leading to a poor user experience and wasted processing. Configuration errors should be detected at the earliest possible moment.
*   **Impacted Modules:** `src/frontend/ui_components.py`, `src/services/llm_service.py`

*   **Required Changes:**
    1.  Add a lightweight validation method to `EnhancedLLMService`, like `validate_api_key()`, which performs a simple, low-cost API call (e.g., listing models).
    2.  In the Streamlit UI (`ui_components.py`), after a user enters an API key, immediately call this validation method.
    3.  Provide immediate feedback to the user (`st.success` or `st.error`) and prevent the "Generate" button from being enabled if the key is invalid.

*   **Implementation Checklist:**
    1.  [ ] **Service Method:** Add `async def validate_api_key(self)` to `EnhancedLLMService`. Inside, make a simple call like `genai.list_models()`. Wrap it in a `try...except` block to catch authentication errors. Return `True` on success, `False` on failure.
    2.  [ ] **UI Logic:** In `src/frontend/ui_components.py`, within `display_sidebar`, when a new `user_api_key` is entered, trigger a callback.
    3.  [ ] **Callback:** This callback will instantiate the `EnhancedLLMService` with the new key and call `validate_api_key()`.
    4.  [ ] **State Update:** Store the validation result (e.g., `st.session_state.api_key_validated = True/False`).
    5.  [ ] **UI Feedback:** Use the session state flag to show an appropriate `st.success` or `st.error` message in the sidebar.
    6.  [ ] **Disable Button:** The "Generate Tailored CV" button's `disabled` property should check `st.session_state.api_key_validated`.

*   **Tests to Add:**
    *   Unit test for `EnhancedLLMService.validate_api_key` that mocks `genai.list_models` to return success and to raise an exception.
    *   Integration test for the Streamlit UI that simulates entering a valid and invalid key and asserts that the correct UI feedback is shown and the button state is updated.

---

### 2.7. Orchestration & State Contracts

#### Task 2.7.1: Enforce State Validation at Node Boundaries (CB-04)

*   **Priority:** `P3`
*   **Root Cause:** Agents in the `cv_workflow_graph` implicitly trust the state they receive. If a preceding agent fails to populate a required field, a downstream agent will fail with a confusing `AttributeError` or `KeyError`.
*   **Impacted Modules:** `src/orchestration/cv_workflow_graph.py`, `src/models/validation_schemas.py`

*   **Required Changes:** Introduce validation at the entry point of each LangGraph node. A decorator is a clean way to implement this. This ensures each node fails fast if its required inputs are not present in the `AgentState`.

*   **Code Snippets (Proposed Decorator):**
    ```python
    # src/models/validation_schemas.py (or a new validation utility file)
    from functools import wraps
    from pydantic import ValidationError

    def validate_node_input(required_fields: list[str]):
        def decorator(node_func):
            @wraps(node_func)
            async def wrapper(state: AgentState):
                missing_fields = []
                for field_path in required_fields:
                    # Simple check for top-level fields
                    if '.' not in field_path:
                        if not getattr(state, field_path, None):
                            missing_fields.append(field_path)
                    # Add logic for nested fields if needed

                if missing_fields:
                    error_msg = f"Node '{node_func.__name__}' failed precondition: Missing required state fields: {missing_fields}"
                    logger.error(error_msg)
                    return {"error_messages": state.error_messages + [error_msg]}

                return await node_func(state)
            return wrapper
        return decorator

    # src/orchestration/cv_workflow_graph.py
    @validate_node_input(required_fields=["structured_cv", "job_description_data", "research_findings"])
    async def content_writer_node(state: AgentState) -> Dict[str, Any]:
        # ... node logic ...
    ```

*   **Implementation Checklist:**
    1.  [ ] Create the `validate_node_input` decorator.
    2.  [ ] For each node in `cv_workflow_graph.py`, identify its essential input fields from the `AgentState`.
    3.  [ ] Apply the decorator to each node function with its list of required fields.
    4.  [ ] Ensure the graph's error handling correctly routes to an end state when the decorator returns an error message.

*   **Tests to Add:**
    *   Unit test for the `validate_node_input` decorator itself.
    *   For each node (e.g., `test_content_writer_node`), add a test case where the input `AgentState` is missing a required field and assert that the node returns an `error_messages` dictionary without executing its main logic.

---

## 3. Implementation Checklist (Updated)

| Priority | Task ID | Description                                     | Status      |
| :------- | :------ | :---------------------------------------------- | :---------- |
| **P1**   | CS-01   | Fix `await` misuse in `CVAnalyzerAgent`         | `[ ] To Do` |
| **P1**   | CS-02   | Fix non-existent error handler call in `CleaningAgent` | `[ ] To Do` |
| **P1**   | CB-01   | Fix incorrect error handling in `FormatterAgent`| `[ ] To Do` |
| **P1**   | CB-03   | Fix Asynchronous Contract of `generate_content` | `[ ] To Do` |
| **P1**   | CF-01   | Implement Fail-Fast Gemini API Key Validation   | `[ ] To Do` |
| **P1**   | NI-01   | Fix `section.name` vs. `section.title`          | `[ ] To Do` |
| **P1**   | AD-02   | Fix incorrect `run_in_executor` usage         | `[ ] To Do` |
| **P2**   | DP-01   | Consolidate Caching in `EnhancedLLMService`     | `[ ] To Do` |
| **P2**   | NI-02   | Clarify API Key Management Strategy in `llm_service` | `[ ] To Do` |
| **P2**   | AD-01   | Refactor parsing logic out of `ContentWriter`   | `[ ] To Do` |
| **P3**   | CB-02   | Enforce Pydantic Models over `Dict[str, Any]` | `[ ] To Do` |
| **P3**   | CB-04   | Enforce State Validation at Node Boundaries     | `[ ] To Do` |
| **P3**   | DL-01   | Migrate and delete orphaned debug scripts       | `[ ] To Do` |

---

## 4. Testing Strategy (Updated)

1.  **Unit Testing:** Each fix must be accompanied by new or updated unit tests that specifically target the bug or refactoring. For example, the `await` fix (`CS-01`) must have a test that proves the async call chain works.
2.  **Integration Testing:** After P1 fixes are complete, run the full `test_agent_workflow_integration.py` to ensure the agents still work together correctly. This test must be updated to reflect the new data contracts from the refactoring tasks.
3.  **Regression Testing:** The test cases from the orphaned debug scripts (e.g., `test_await_issue.py`) must be converted into permanent regression tests in the `pytest` suite. This ensures that the critical bugs they were designed to catch do not reappear.
4.  **Static Analysis:** After applying fixes, run `pylint` using the provided `.pylintrc` configuration. The goal is to have zero new `E` (Error) or `W` (Warning) messages related to the changed code.

---

## 5. Critical Gaps & Questions

*   No critical gaps or questions remain at this time. The provided clarifications have been integrated into the plan.

---
### 2.8. Vector Store Services

#### Task 2.8.1: Consolidate Vector Store Implementations (DP-02)

*   **Priority:** `P2`
*   **Root Cause:** The codebase contains two separate vector store service implementations: `src/services/vector_db.py` (appears to be FAISS-based) and `src/services/vector_store_service.py` (ChromaDB-based). This creates architectural ambiguity, code duplication, and potential data fragmentation. The official stack specifies `ChromaDB`.
*   **Impacted Modules:** `src/services/vector_db.py`, `src/services/vector_store_service.py`, `src/agents/research_agent.py`, any other component using vector search.

*   **Required Changes:**
    1.  Designate `vector_store_service.py` (ChromaDB) as the single source of truth for vector operations.
    2.  Migrate any unique, valuable functionality from the legacy `vector_db.py` into the consolidated service.
    3.  Refactor all components, particularly `ResearchAgent`, to exclusively use the `get_vector_store_service()`.
    4.  Deprecate and delete `src/services/vector_db.py`.

*   **Implementation Checklist:**
    1.  [ ] Audit `ResearchAgent` and other potential consumers to identify all calls to the legacy `vector_db.py`.
    2.  [ ] Ensure the public API of `VectorStoreService` (ChromaDB) supports all required operations (e.g., `add_item`, `search`).
    3.  [ ] Refactor all call sites to use `VectorStoreService`.
    4.  [ ] Delete the legacy `src/services/vector_db.py` file.
    5.  [ ] Update `tests/unit/test_vector_store_service.py` to cover all required functionality and remove any tests related to the old service.

*   **Tests to Add:**
    *   Integration test for `ResearchAgent` confirming it interacts correctly with the `VectorStoreService`.
    *   Unit tests for `VectorStoreService` to ensure its public methods behave as expected.

---

### 2.9. Utilities & Core Logic

#### Task 2.9.1: Standardize Error Handling Strategy (AD-03)

*   **Priority:** `P2`
*   **Root Cause:** The existence of multiple error handling modules (`error_handling.py`, `agent_error_handling.py`, `error_boundaries.py`) and inconsistent application (evidenced by `CS-02` and `CB-01`) creates a fragile and hard-to-debug system.
*   **Impacted Modules:** All agent files, `src/utils/*`.

*   **Required Changes:**
    1.  Consolidate all agent-specific error handling logic into `src/utils/agent_error_handling.py`.
    2.  Mandate the use of the `@with_node_error_handling` decorator or the `AgentErrorHandler.handle_node_error` static method for all `run_as_node` methods in agents.
    3.  Deprecate general-purpose handlers in `error_handling.py` if they are superseded by the more specific agent handlers.

*   **Implementation Checklist:**
    1.  [ ] Review the contents of `error_handling.py`, `agent_error_handling.py`, and `error_boundaries.py`.
    2.  [ ] Define a clear hierarchy: `agent_error_handling.py` for agent-level errors, `error_boundaries.py` for UI errors, and `exceptions.py` for custom exception types.
    3.  [ ] Audit every agent's `run_as_node` method. Remove custom `try...except` blocks and replace them with the `@with_node_error_handling` decorator or a direct call to `AgentErrorHandler.handle_node_error`.
    4.  [ ] Ensure all error handlers correctly populate the `error_messages` field in the `AgentState` without throwing secondary exceptions.

*   **Tests to Add:**
    *   Add unit tests for the `@with_node_error_handling` decorator to `tests/unit/test_agent_error_handling.py`.
    *   For each agent, add a test case that mocks a failure and asserts that the `error_messages` field in the state is populated correctly.

---

### 2.10. Data Models (`src/models/`)

#### Task 2.10.1: Consolidate Data Model Definitions (DP-03)

*   **Priority:** `P3`
*   **Root Cause:** Core data contracts are split between `data_models.py` and `validation_schemas.py`, increasing cognitive load and the risk of definitions drifting apart.
*   **Impacted Modules:** `src/models/data_models.py`, `src/models/validation_schemas.py`, and all files that import from them.

*   **Required Changes:**
    1.  `data_models.py` should be the single source of truth for all Pydantic models that define the application's core state (e.g., `StructuredCV`, `JobDescriptionData`, `Section`, `Item`, `AgentState`).
    2.  `validation_schemas.py` should be reserved for schemas that validate external data or raw, transient structures (e.g., raw LLM JSON output before it's mapped to a core model).
    3.  Move any core models from `validation_schemas.py` to `data_models.py` and update all imports.

*   **Implementation Checklist:**
    1.  [ ] Analyze both model files to identify which Pydantic models represent core, persistent application state.
    2.  [ ] Move the identified core models to `data_models.py`.
    3.  [ ] Perform a global search and replace to update all import statements across the codebase to point to the new location of the moved models.
    4.  [ ] Verify that `validation_schemas.py` now only contains schemas for validation purposes.

*   **Tests to Add:**
    *   No new tests are strictly required, but running the existing test suite is critical to catch any broken imports or contract changes resulting from the file consolidation.

---

## 3. Implementation Checklist (Updated)

| Priority | Task ID | Description                                     | Status      |
| :------- | :------ | :---------------------------------------------- | :---------- |
| **P1**   | CS-01   | Fix `await` misuse in `CVAnalyzerAgent`         | `[ ] To Do` |
| **P1**   | CS-02   | Fix non-existent error handler call in `CleaningAgent` | `[ ] To Do` |
| **P1**   | CB-01   | Fix incorrect error handling in `FormatterAgent`| `[ ] To Do` |
| **P1**   | CB-03   | Fix Asynchronous Contract of `generate_content` | `[ ] To Do` |
| **P1**   | CF-01   | Implement Fail-Fast Gemini API Key Validation   | `[ ] To Do` |
| **P1**   | NI-01   | Fix `section.name` vs. `section.title`          | `[ ] To Do` |
| **P1**   | AD-02   | Fix incorrect `run_in_executor` usage         | `[ ] To Do` |
| **P2**   | DP-01   | Consolidate Caching in `EnhancedLLMService`     | `[ ] To Do` |
| **P2**   | DP-02   | Consolidate Vector Store Services             | `[ ] To Do` |
| **P2**   | NI-02   | Clarify API Key Management Strategy in `llm_service` | `[ ] To Do` |
| **P2**   | AD-01   | Refactor parsing logic out of `ContentWriter`   | `[ ] To Do` |
| **P2**   | AD-03   | Standardize Error Handling Strategy             | `[ ] To Do` |
| **P3**   | CB-02   | Enforce Pydantic Models over `Dict[str, Any]` | `[ ] To Do` |
| **P3**   | CB-04   | Enforce State Validation at Node Boundaries     | `[ ] To Do` |
| **P3**   | DP-03   | Consolidate Data Model Definitions            | `[ ] To Do` |
| **P3**   | DL-01   | Migrate and delete orphaned debug scripts       | `[ ] To Do` |

---

## 4. Testing Strategy (Updated)

1.  **Unit Testing:** Each fix must be accompanied by new or updated unit tests that specifically target the bug or refactoring. For example, the `await` fix (`CS-01`) must have a test that proves the async call chain works.
2.  **Integration Testing:** After P1 fixes are complete, run the full `test_agent_workflow_integration.py` to ensure the agents still work together correctly. This test must be updated to reflect the new data contracts from the refactoring tasks.
3.  **Regression Testing:** The test cases from the orphaned debug scripts (e.g., `test_await_issue.py`) must be converted into permanent regression tests in the `pytest` suite. This ensures that the critical bugs they were designed to catch do not reappear.
4.  **Static Analysis:** After applying fixes, run `pylint` using the provided `.pylintrc` configuration. The goal is to have zero new `E` (Error) or `W` (Warning) messages related to the changed code.

---

## 5. Critical Gaps & Questions

*   No critical gaps or questions remain at this time. The provided clarifications have been integrated into the plan.

---

### 2.11. Core Logic & Application Lifecycle

#### Task 2.11.1: Formalize Application Startup and Service Initialization (AD-04)

*   **Priority:** `P3`
*   **Root Cause:** The application's startup sequence is implicit, with service initialization (e.g., `get_llm_service`, `get_vector_store_service`) occurring on-demand when first called. This can lead to unpredictable initialization order and makes it difficult to manage the lifecycle of services, especially within Streamlit's execution model.
*   **Impacted Modules:** `app.py`, `src/core/main.py`, `src/services/*`

*   **Required Changes:**
    1.  Create a dedicated, idempotent `initialize_app()` function within `src/core/main.py`.
    2.  This function will be responsible for loading configuration, setting up logging, and pre-initializing all critical singleton services.
    3.  Use `st.session_state` to ensure this initialization runs only once per user session.
    4.  Refactor `main()` to call `initialize_app()` at the very beginning.

*   **Code Snippets (Proposed Structure):**
    ```python
    # src/core/main.py
    import streamlit as st
    from ..config.settings import get_config
    from ..config.logging_config import setup_logging
    from ..services.llm_service import get_llm_service
    from ..services.vector_store_service import get_vector_store_service
    # ... other imports

    def initialize_app():
        """Initializes all critical application components. Idempotent."""
        if "app_initialized" in st.session_state:
            return

        # 1. Load configuration
        get_config()

        # 2. Setup logging
        setup_logging()

        # 3. Initialize services (fail-fast)
        get_vector_store_service()
        get_llm_service() # Initial call without user key

        # 4. Initialize session state for UI
        initialize_session_state() # From frontend.state_helpers

        st.session_state.app_initialized = True
        logger.info("Application initialized successfully for session: %s", st.session_state.session_id)

    def main():
        """Main Streamlit application controller."""
        try:
            # Run initialization at the start of every script run.
            # The logic inside ensures it only executes once.
            initialize_app()

            # ... rest of the main application logic ...

        except Exception as e:
            # ... error handling ...
    ```

*   **Implementation Checklist:**
    1.  [ ] Create the `initialize_app()` function in `src/core/main.py`.
    2.  [ ] Move initialization calls for config, logging, and services into this function.
    3.  [ ] Add the `if "app_initialized" in st.session_state: return` guard.
    4.  [ ] Call `initialize_app()` as the first line in the `main()` function.
    5.  [ ] Remove redundant initialization calls from other parts of the code if they are now handled by `initialize_app`.

*   **Tests to Add:**
    *   An integration test that runs the `main` function and asserts that `st.session_state.app_initialized` is set to `True`.
    *   Patch the service getters (e.g., `get_llm_service`) and assert they are called exactly once during the initialization phase of a session.

---

#### Task 2.11.2: Stabilize `EnhancedLLMService` Singleton Lifecycle (AD-03)

*   **Priority:** `P3`
*   **Root Cause:** The `get_llm_service()` function re-initializes the `EnhancedLLMService` instance whenever a new `user_api_key` is provided. In a complex application, different components could hold references to different instances (one with the old key, one with the new), leading to inconsistent behavior, especially with caching and rate limiting.
*   **Impacted Modules:** `src/services/llm_service.py`, `src/frontend/ui_components.py`, all consumers of the LLM service.

*   **Required Changes:** Tie the lifecycle of the `EnhancedLLMService` instance to Streamlit's session state. This ensures there is only one active instance per session, and it is correctly replaced when the user provides a new API key.

*   **Code Snippets (Before/After):**

    *   **Before (Global Singleton):**
        ```python
        # src/services/llm_service.py
        _llm_service_instance = None

        def get_llm_service(user_api_key: Optional[str] = None) -> EnhancedLLMService:
            global _llm_service_instance
            if _llm_service_instance is None or (user_api_key and _llm_service_instance.user_api_key != user_api_key):
                _llm_service_instance = EnhancedLLMService(user_api_key=user_api_key)
            return _llm_service_instance
        ```

    *   **After (Session-Scoped Singleton):**
        ```python
        # src/services/llm_service.py
        import streamlit as st

        def get_llm_service(user_api_key: Optional[str] = None) -> EnhancedLLMService:
            """Gets the LLM service instance for the current Streamlit session."""
            # Use a consistent key for session state
            _service_key = "llm_service_instance"

            # Get current instance or its key from session state
            current_instance = st.session_state.get(_service_key)
            current_key = getattr(current_instance, 'user_api_key', None) if current_instance else None

            # Re-initialize if the instance doesn't exist or the key has changed
            if current_instance is None or (user_api_key and user_api_key != current_key):
                instance = EnhancedLLMService(user_api_key=user_api_key)
                st.session_state[_service_key] = instance
                logger.info("LLM service (re)initialized in session state.")
                return instance

            return current_instance
        ```

*   **Implementation Checklist:**
    1.  [ ] Refactor the `get_llm_service` function as shown above to use `st.session_state`.
    2.  [ ] Ensure all calls to `get_llm_service` throughout the application do not store the returned instance long-term; they should always call the getter to ensure they have the current, valid instance for the session.
    3.  [ ] Verify the UI callback for setting the API key correctly calls `get_llm_service(new_key)` to trigger the re-initialization.

*   **Tests to Add:**
    *   Unit test for `get_llm_service` that mocks `st.session_state` and verifies that a new instance is created only when the API key changes or when no instance exists.

---

### 2.12. Code Quality

#### Task 2.12.1: Remediate High-Priority Pylint Violations (CQ-01)

*   **Priority:** `P2`
*   **Root Cause:** The codebase has accumulated various static analysis violations that, while not all being runtime errors, increase technical debt, reduce readability, and can hide subtle bugs.
*   **Impacted Modules:** Entire `src/` directory.

*   **Required Changes:** Perform a focused pass to fix all high-priority `pylint` errors and warnings (`E` and `W` codes).

*   **Implementation Checklist:**
    1.  [ ] **Run Analysis:** Execute `pylint src/` from the project root to generate a fresh report.
    2.  [ ] **Fix Errors (`E` codes):** Prioritize and fix all reported errors. These often point to genuine bugs (e.g., `no-member`, `not-callable`).
    3.  [ ] **Fix Warnings (`W` codes):** Address key warnings like `unused-variable`, `redefined-builtin`, and `unreachable-code`.
    4.  [ ] **Review Conventions (`C` codes):** Address convention violations like `invalid-name` and `missing-docstring` for all new and modified code. A full pass on existing code is lower priority but recommended.
    5.  [ ] **CI Integration:** Configure the CI/CD pipeline to run `pylint` on every commit or pull request and fail the build if the score drops below a defined threshold or new high-priority errors are introduced.

*   **Tests to Add:**
    *   This task is about code quality, not new functionality. The "test" is a clean `pylint` report. The existing unit and integration test suite will serve as a regression safety net to ensure that fixes do not break existing functionality.

---

## 3. Implementation Checklist (Final)

| Priority | Task ID | Description                                     | Status      |
| :------- | :------ | :---------------------------------------------- | :---------- |
| **P1**   | CS-01   | Fix `await` misuse in `CVAnalyzerAgent`         | `[ ] To Do` |
| **P1**   | CS-02   | Fix non-existent error handler call in `CleaningAgent` | `[ ] To Do` |
| **P1**   | CB-01   | Fix incorrect error handling in `FormatterAgent`| `[ ] To Do` |
| **P1**   | CB-03   | Fix Asynchronous Contract of `generate_content` | `[ ] To Do` |
| **P1**   | CF-01   | Implement Fail-Fast Gemini API Key Validation   | `[ ] To Do` |
| **P1**   | NI-01   | Fix `section.name` vs. `section.title`          | `[ ] To Do` |
| **P1**   | AD-02   | Fix incorrect `run_in_executor` usage         | `[ ] To Do` |
| **P2**   | DP-01   | Consolidate Caching in `EnhancedLLMService`     | `[ ] To Do` |
| **P2**   | DP-02   | Consolidate Vector Store Services             | `[ ] To Do` |
| **P2**   | NI-02   | Clarify API Key Management Strategy in `llm_service` | `[ ] To Do` |
| **P2**   | AD-01   | Refactor parsing logic out of `ContentWriter`   | `[ ] To Do` |
| **P2**   | AD-03   | Standardize Error Handling Strategy             | `[ ] To Do` |
| **P2**   | CQ-01   | Remediate High-Priority Pylint Violations       | `[ ] To Do` |
| **P3**   | AD-04   | Formalize Application Startup and Service Init  | `[ ] To Do` |
| **P3**   | AD-03   | Stabilize `EnhancedLLMService` Singleton Lifecycle | `[ ] To Do` |
| **P3**   | CB-02   | Enforce Pydantic Models over `Dict[str, Any]` | `[ ] To Do` |
| **P3**   | CB-04   | Enforce State Validation at Node Boundaries     | `[ ] To Do` |
| **P3**   | DP-03   | Consolidate Data Model Definitions            | `[ ] To Do` |
| **P3**   | DL-01   | Migrate and delete orphaned debug scripts       | `[ ] To Do` |

---

## 4. Testing Strategy (Final)

1.  **Unit Testing:** Each fix must be accompanied by new or updated unit tests that specifically target the bug or refactoring. For example, the `await` fix (`CS-01`) must have a test that proves the async call chain works.
2.  **Integration Testing:** After P1 fixes are complete, run the full `test_agent_workflow_integration.py` to ensure the agents still work together correctly. This test must be updated to reflect the new data contracts from the refactoring tasks.
3.  **Regression Testing:** The test cases from the orphaned debug scripts (e.g., `test_await_issue.py`) must be converted into permanent regression tests in the `pytest` suite. This ensures that the critical bugs they were designed to catch do not reappear.
4.  **Static Analysis:** After applying fixes, run `pylint` using the provided `.pylintrc` configuration. The goal is to have zero new `E` (Error) or `W` (Warning) messages related to the changed code.

---

## 5. Critical Gaps & Questions

*   No critical gaps or questions remain at this time. The provided clarifications have been integrated into the plan.

---


### 2.2. Agents (Continued)

#### Task 2.2.4: Refactor `EnhancedContentWriterAgent` Parsing Logic (AD-01, CS-03) - *Updated*

*   **Priority:** `P2`
*   **Root Cause:** `EnhancedContentWriterAgent` is responsible for parsing raw CV text, which violates the Single Responsibility Principle. This logic belongs in the `ParserAgent`. The primary parsing method, `_parse_cv_text_to_content_item`, is also over 100 lines long (a "Long Method" code smell), making it difficult to test and maintain.
*   **Impacted Modules:** `src/agents/enhanced_content_writer.py`, `src/agents/parser_agent.py`, `src/orchestration/cv_workflow_graph.py`
*   **Data Contract:** `ParserAgent` must populate the `subsections` list of the "Professional Experience" `Section` in `StructuredCV`. Each `Subsection` should have its `.name` as "Title at Company (Date Range)" and its `.items` list populated with `Item` objects of type `BULLET_POINT`. No Pydantic model changes are needed.

*   **Implementation Checklist:**
    1.  [ ] **Refactor First (CS-03):** Before moving, refactor the `_parse_cv_text_to_content_item` method within `EnhancedContentWriterAgent` into smaller, single-purpose, and more testable helper functions (e.g., `_extract_roles_from_lines`, `_extract_accomplishments_for_role`). This isolates the logic and makes the next steps easier.
    2.  [ ] **Analyze:** Identify the full set of parsing methods and the exact data structures they produce.
    3.  [ ] **Move Logic:** Copy the newly refactored helper methods from `EnhancedContentWriterAgent` to `src/agents/parser_agent.py`.
    4.  [ ] **Refactor `ParserAgent`:** In `ParserAgent.run_as_node`, after the initial CV parsing, call the new methods to populate the `StructuredCV.sections` list with the correct `Subsection` and `Item` objects for professional experience.
    5.  [ ] **Refactor `EnhancedContentWriterAgent`:** Remove the parsing methods. Update its logic to directly consume the `StructuredCV` object from the state, assuming the experience is already fully structured.
    6.  [ ] **Delete:** Remove the now-redundant parsing methods from `EnhancedContentWriterAgent`.

*   **Tests to Add:**
    *   Add unit tests to `tests/unit/test_parser_agent_refactored.py` to verify the new parsing capabilities. The tests should assert that the `StructuredCV` object produced by the parser contains correctly structured `Subsection` and `Item` objects.
    *   Add unit tests to `tests/unit/test_enhanced_content_writer.py` to verify it correctly consumes the structured data from a mock `AgentState`.
    *   Update integration tests in `tests/integration/test_agent_workflow_integration.py` to reflect the new data flow.

---

### 2.13. Templates & Output Generation

#### Task 2.13.1: Harden PDF Generation Template (CB-05)

*   **Priority:** `P3`
*   **Root Cause:** The Jinja2 template (`pdf_template.html`) directly accesses nested data (e.g., `{{ cv.metadata.get('name') }}`) without robust fallbacks. If a piece of data is missing from the `StructuredCV` object, the PDF generation could fail or render incorrectly.
*   **Impacted Modules:** `src/templates/pdf_template.html`

*   **Required Changes:** Update the Jinja2 template to use filters like `default` to provide fallback values for optional data, preventing rendering errors.

*   **Code Snippets (Before/After):**

    *   **Before (Brittle Access):**
        ```html
        <!-- src/templates/pdf_template.html -->
        <h1>{{ cv.metadata.get('name') }}</h1>
        <p class="contact-info">
            {{ cv.metadata.get('email') }} | {{ cv.metadata.get('phone') }}
        </p>
        ```

    *   **After (Robust Access):**
        ```html
        <!-- src/templates/pdf_template.html -->
        <h1>{{ cv.metadata.get('name', 'Your Name') }}</h1>
        <p class="contact-info">
            {% if cv.metadata.get('email') %}{{ cv.metadata.get('email') }}{% endif %}
            {% if cv.metadata.get('phone') %} | {{ cv.metadata.get('phone') }}{% endif %}
        </p>
        ```
        *Note: Using `if` checks is often safer than `| default('')` for layout purposes.*

*   **Implementation Checklist:**
    1.  [ ] Audit `pdf_template.html` for all data access points (e.g., `{{ variable.access }}`).
    2.  [ ] For each optional field, add a Jinja2 `if` condition or a `| default('...')` filter.
    3.  [ ] Prioritize fields that are most likely to be missing, such as optional contact info or metadata.
    4.  [ ] Ensure loops (`{% for ... %}`) are robust and don't fail on empty lists.

*   **Tests to Add:**
    *   In `tests/unit/test_pdf_generation.py`, add test cases that pass a `StructuredCV` object with missing optional data (e.g., no phone number, no GitHub link) to the `FormatterAgent` and assert that PDF generation still completes successfully without errors.

---

### 2.14. Orchestration & State Contracts (Continued)

#### Task 2.14.1: Audit and Enforce Agent Output Contracts (CB-06)

*   **Priority:** `P2`
*   **Root Cause:** While nodes receive the full `AgentState`, they are expected to return only a dictionary containing the fields they have modified. There is no enforcement of this contract, creating a risk that an agent could accidentally return a key that is not a valid `AgentState` field, which LangGraph would silently ignore, leading to data loss.
*   **Impacted Modules:** All agent `run_as_node` methods.

*   **Required Changes:**
    1.  Create a lightweight decorator or utility function that validates the keys of the dictionary returned by each `run_as_node` method.
    2.  The validator should check that every key in the returned dictionary is a valid field name in the `AgentState` Pydantic model.
    3.  If an invalid key is found, it should log a `CRITICAL` error, as this indicates a developer error and a contract breach.

*   **Code Snippets (Proposed Decorator):**
    ```python
    # src/models/validation_schemas.py (or a new validation utility file)
    from functools import wraps

    def validate_node_output(node_func):
        @wraps(node_func)
        async def wrapper(state: AgentState):
            output_dict = await node_func(state)

            # Validate the output dictionary
            valid_keys = set(AgentState.model_fields.keys())
            returned_keys = set(output_dict.keys())

            invalid_keys = returned_keys - valid_keys
            if invalid_keys:
                logger.critical(
                    "Node '%s' returned invalid keys not in AgentState: %s",
                    node_func.__name__, invalid_keys
                )
                # Optionally, filter out invalid keys to prevent data loss
                # output_dict = {k: v for k, v in output_dict.items() if k in valid_keys}

            return output_dict
        return wrapper

    # src/orchestration/cv_workflow_graph.py
    @validate_node_output
    async def parser_node(state: AgentState) -> Dict[str, Any]:
        # ... node logic ...
    ```

*   **Implementation Checklist:**
    1.  [ ] Implement the `validate_node_output` decorator.
    2.  [ ] Apply this decorator to every agent node function defined in `cv_workflow_graph.py`.
    3.  [ ] Run the full integration test suite to ensure no valid data is being accidentally flagged.

*   **Tests to Add:**
    *   Unit test for the `validate_node_output` decorator. One test should pass a valid dictionary, and another should pass a dictionary with an invalid key and assert that a critical error is logged.

---

## 3. Implementation Checklist (Final)

| Priority | Task ID | Description                                     | Status      |
| :------- | :------ | :---------------------------------------------- | :---------- |
| **P1**   | CS-01   | Fix `await` misuse in `CVAnalyzerAgent`         | `[ ] To Do` |
| **P1**   | CS-02   | Fix non-existent error handler call in `CleaningAgent` | `[ ] To Do` |
| **P1**   | CB-01   | Fix incorrect error handling in `FormatterAgent`| `[ ] To Do` |
| **P1**   | CB-03   | Fix Asynchronous Contract of `generate_content` | `[ ] To Do` |
| **P1**   | CF-01   | Implement Fail-Fast Gemini API Key Validation   | `[ ] To Do` |
| **P1**   | NI-01   | Fix `section.name` vs. `section.title`          | `[ ] To Do` |
| **P1**   | AD-02   | Fix incorrect `run_in_executor` usage         | `[ ] To Do` |
| **P2**   | DP-01   | Consolidate Caching in `EnhancedLLMService`     | `[ ] To Do` |
| **P2**   | DP-02   | Consolidate Vector Store Services             | `[ ] To Do` |
| **P2**   | NI-02   | Clarify API Key Management Strategy in `llm_service` | `[ ] To Do` |
| **P2**   | AD-01   | Refactor parsing logic out of `ContentWriter`   | `[ ] To Do` |
| **P2**   | AD-03   | Standardize Error Handling Strategy             | `[ ] To Do` |
| **P2**   | CB-06   | Audit and Enforce Agent Output Contracts      | `[ ] To Do` |
| **P2**   | CQ-01   | Remediate High-Priority Pylint Violations       | `[ ] To Do` |
| **P3**   | AD-04   | Formalize Application Startup and Service Init  | `[ ] To Do` |
| **P3**   | AD-03   | Stabilize `EnhancedLLMService` Singleton Lifecycle | `[ ] To Do` |
| **P3**   | CB-02   | Enforce Pydantic Models over `Dict[str, Any]` | `[ ] To Do` |
| **P3**   | CB-04   | Enforce State Validation at Node Boundaries     | `[ ] To Do` |
| **P3**   | CB-05   | Harden PDF Generation Template                  | `[ ] To Do` |
| **P3**   | DP-03   | Consolidate Data Model Definitions            | `[ ] To Do` |
| **P3**   | DL-01   | Migrate and delete orphaned debug scripts       | `[ ] To Do` |

---

## 4. Testing Strategy (Final)

1.  **Unit Testing:** Each fix must be accompanied by new or updated unit tests that specifically target the bug or refactoring. For example, the `await` fix (`CS-01`) must have a test that proves the async call chain works.
2.  **Integration Testing:** After P1 fixes are complete, run the full `test_agent_workflow_integration.py` to ensure the agents still work together correctly. This test must be updated to reflect the new data contracts from the refactoring tasks.
3.  **Regression Testing:** The test cases from the orphaned debug scripts (e.g., `test_await_issue.py`) must be converted into permanent regression tests in the `pytest` suite. This ensures that the critical bugs they were designed to catch do not reappear.
4.  **Static Analysis:** After applying fixes, run `pylint` using the provided `.pylintrc` configuration. The goal is to have zero new `E` (Error) or `W` (Warning) messages related to the changed code.

---

## 5. Critical Gaps & Questions

*   No critical gaps or questions remain at this time. The provided clarifications have been integrated into the plan.

---

### 2.15. Observability & Monitoring

#### Task 2.15.1: Standardize Workflow Tracing (OB-01)

*   **Priority:** `P3`
*   **Root Cause:** While the `AgentState` contains a `trace_id`, its propagation and use in logging are not consistently enforced. This makes it difficult to trace a single user request through the entire workflow, a weakness highlighted by the proliferation of ad-hoc debugging scripts (`DL-01`).
*   **Impacted Modules:** All agent `run_as_node` methods, `src/services/llm_service.py`, `src/config/logging_config.py`.

*   **Required Changes:**
    1.  **Propagate `trace_id`:** Ensure the `trace_id` from `AgentState` is passed to all key service calls, especially `llm_service.generate_content`.
    2.  **Structured Logging:** Enhance the logging configuration to automatically include the `trace_id` in log records when it's available. This can be done via a `logging.Filter` or by passing it in the `extra` dictionary.
    3.  **Node Entry/Exit Logging:** All `run_as_node` methods should begin with a log message indicating entry and end with a message indicating exit, both including the `trace_id`.

*   **Code Snippets (Proposed Logging Pattern):**
    ```python
    # src/orchestration/cv_workflow_graph.py

    async def parser_node(state: AgentState) -> Dict[str, Any]:
        logger.info(
            "Entering parser_node",
            extra={"trace_id": state.trace_id}
        )
        # ... logic ...
        result = await parser_agent.run_as_node(state)
        logger.info(
            "Exiting parser_node",
            extra={"trace_id": state.trace_id, "output_keys": list(result.keys())}
        )
        return result

    # src/services/llm_service.py
    async def generate_content(self, prompt: str, ..., trace_id: Optional[str] = None):
        # ...
        logger.info(
            "Generating LLM content",
            extra={"trace_id": trace_id, "prompt_length": len(prompt)}
        )
        # ...
    ```

*   **Implementation Checklist:**
    1.  [ ] Modify the `generate_content` signature in `EnhancedLLMService` to accept an optional `trace_id`.
    2.  [ ] Update all calls to `generate_content` within agents to pass `state.trace_id`.
    3.  [ ] Add entry and exit log statements with the `trace_id` to every node function in `cv_workflow_graph.py`.
    4.  [ ] Review `logging_config.py` to ensure the `JsonFormatter` is configured to correctly handle and display fields passed in the `extra` dictionary.

*   **Tests to Add:**
    *   Update integration tests to pass a mock `AgentState` with a known `trace_id`.
    *   Capture log output during the test run (e.g., using `caplog` fixture in `pytest`) and assert that the `trace_id` is present in logs from different agents and services involved in the workflow.

---

## 3. Implementation Checklist (Final)

| Priority | Task ID | Description                                     | Status      |
| :------- | :------ | :---------------------------------------------- | :---------- |
| **P1**   | CS-01   | Fix `await` misuse in `CVAnalyzerAgent`         | `[ ] To Do` |
| **P1**   | CS-02   | Fix non-existent error handler call in `CleaningAgent` | `[ ] To Do` |
| **P1**   | CB-01   | Fix incorrect error handling in `FormatterAgent`| `[ ] To Do` |
| **P1**   | CB-03   | Fix Asynchronous Contract of `generate_content` | `[ ] To Do` |
| **P1**   | CF-01   | Implement Fail-Fast Gemini API Key Validation   | `[ ] To Do` |
| **P1**   | NI-01   | Fix `section.name` vs. `section.title`          | `[ ] To Do` |
| **P1**   | AD-02   | Fix incorrect `run_in_executor` usage         | `[ ] To Do` |
| **P2**   | DP-01   | Consolidate Caching in `EnhancedLLMService`     | `[ ] To Do` |
| **P2**   | DP-02   | Consolidate Vector Store Services             | `[ ] To Do` |
| **P2**   | NI-02   | Clarify API Key Management Strategy in `llm_service` | `[ ] To Do` |
| **P2**   | AD-01   | Refactor parsing logic out of `ContentWriter`   | `[ ] To Do` |
| **P2**   | AD-03   | Standardize Error Handling Strategy             | `[ ] To Do` |
| **P2**   | CB-06   | Audit and Enforce Agent Output Contracts      | `[ ] To Do` |
| **P2**   | CQ-01   | Remediate High-Priority Pylint Violations       | `[ ] To Do` |
| **P3**   | AD-04   | Formalize Application Startup and Service Init  | `[ ] To Do` |
| **P3**   | AD-03   | Stabilize `EnhancedLLMService` Singleton Lifecycle | `[ ] To Do` |
| **P3**   | CB-02   | Enforce Pydantic Models over `Dict[str, Any]` | `[ ] To Do` |
| **P3**   | CB-04   | Enforce State Validation at Node Boundaries     | `[ ] To Do` |
| **P3**   | CB-05   | Harden PDF Generation Template                  | `[ ] To Do` |
| **P3**   | DP-03   | Consolidate Data Model Definitions            | `[ ] To Do` |
| **P3**   | OB-01   | Standardize Workflow Tracing                  | `[ ] To Do` |
| **P3**   | DL-01   | Migrate and delete orphaned debug scripts       | `[ ] To Do` |

---

## 4. Testing Strategy (Final)

1.  **Unit Testing:** Each fix must be accompanied by new or updated unit tests that specifically target the bug or refactoring. For example, the `await` fix (`CS-01`) must have a test that proves the async call chain works.
2.  **Integration Testing:** After P1 fixes are complete, run the full `test_agent_workflow_integration.py` to ensure the agents still work together correctly. This test must be updated to reflect the new data contracts from the refactoring tasks.
3.  **Regression Testing:** The test cases from the orphaned debug scripts (e.g., `test_await_issue.py`) must be converted into permanent regression tests in the `pytest` suite. This ensures that the critical bugs they were designed to catch do not reappear.
4.  **Static Analysis:** After applying fixes, run `pylint` using the provided `.pylintrc` configuration. The goal is to have zero new `E` (Error) or `W` (Warning) messages related to the changed code.

---

## 5. Critical Gaps & Questions

*   No critical gaps or questions remain at this time. The provided clarifications have been integrated into the plan.

