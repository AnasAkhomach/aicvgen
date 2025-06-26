# TASK_BLUEPRINT.md

## Overview

This document outlines the prioritized tasks for refactoring the `anasakhomach-aicvgen` codebase. The primary mission is to remediate critical technical debt, harden the system against failures, and enforce architectural consistency as identified in the codebase audits. This plan serves as the single source of truth for the implementation team.

**Architectural Principles to Enforce:**
- **Fail-fast & Recover Gracefully:** Detect configuration errors and contract breaches at startup or as early as possible.
- **Single Source of Truth:** Unify configuration, data models, and dependency management.
- **Strict Contracts:** Enforce data contracts between components using Pydantic models.
- **Clear Separation of Concerns:** Isolate business logic (agents), orchestration (LangGraph), and services.
- **Asynchronous Best Practices:** Eliminate blocking I/O in asynchronous contexts and ensure contract integrity.

---

## 1. Configuration & Deployment (P1)

### **Task CD-01 (P1): Fix Critical Logging Failure due to Volume Mount Drift**

-   **Root Cause:** `docker-compose.yml` mounts a legacy `./logs` directory, while the refactored application writes all runtime data, including logs, to the `./instance` directory. This causes all logs to be ephemeral and lost on container restart, breaking observability. (Audit ID: `LOG-01`)
-   **Impacted Modules:** `docker-compose.yml`, `README.md`.
-   **Required Changes:** Correct the volume mount in `docker-compose.yml` to persist the entire `instance` directory, which aligns the deployment environment with the application's architecture.

-   **Code Implementation (`docker-compose.yml`):**

    **Before:**
    ```yaml
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
    ```

    **After:**
    ```yaml
    volumes:
      - ./instance:/app/instance
    # The ./data and ./config mounts are removed as they are copied
    # into the image by the Dockerfile and should not be mounted
    # from the host in a production-like environment.
    # The instance directory is the single source of persistent runtime data.
    ```
-   **Implementation Checklist:**
    -   [ ] Remove the `- ./logs:/app/logs` and `- ./config:/app/config` lines from `docker-compose.yml`.
    -   [ ] Add the line `- ./instance:/app/instance` to `docker-compose.yml`.
    -   [ ] Update the `docker run` command in `README.md` to use `-v "%cd%/instance:/app/instance"`.

-   **Testing:**
    -   [ ] Run `docker-compose up`.
    -   [ ] Trigger an action in the application that generates a log.
    -   [ ] Verify that logs appear in the `./instance/logs/` directory on the host machine.
    -   [ ] Run `docker-compose down` and `docker-compose up` again to ensure logs persist.

---

## 2. Services Layer (`src/services/`)

### **Task S-01 (P1): Fix Asynchronous Call Contract in `EnhancedLLMService`**

-   **Root Cause:** `EnhancedLLMService._make_llm_api_call` is a synchronous method (`def`) that incorrectly calls an asynchronous method (`llm_retry_handler.generate_content`) without using `await`. This is an anti-pattern that returns a coroutine instead of a result, masked by a `ThreadPoolExecutor`. (Audit ID: `CB-01`, `S-02`)
-   **Impacted Modules:** `src/services/llm_service.py`.
-   **Required Changes:** Refactor `EnhancedLLMService` to be fully asynchronous. Remove the `ThreadPoolExecutor` and synchronous wrappers. Use `asyncio.wait_for` for timeouts and directly `await` the retry handler.

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
        # ...

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

    async def _call_llm_with_retry(self, prompt: str, **kwargs) -> Any:
        """Correctly awaits the async retry handler."""
        logger.info("Calling LLM via retry handler.", extra=kwargs)
        return await self.llm_retry_handler.generate_content(prompt)

    async def generate_content(self, prompt: str, **kwargs) -> LLMResponse:
        """Main entry point for generating content with timeout and retries."""
        # ... (cache check, rate limiting logic remains the same) ...
        try:
            # Directly await the retryable call and wrap with a timeout
            response = await asyncio.wait_for(
                self._call_llm_with_retry(prompt, **kwargs),
                timeout=self.timeout
            )
            # ... (process successful response) ...
        except asyncio.TimeoutError as e:
            # ... (handle timeout) ...
            raise OperationTimeoutError(f"LLM request timed out after {self.timeout} seconds") from e
        # ...
    ```

-   **Implementation Checklist:**
    -   [ ] Remove `self.executor` from `EnhancedLLMService` constructor.
    -   [ ] Remove the `_generate_with_timeout` and `_make_llm_api_call` methods.
    -   [ ] Implement the new `async def _call_llm_with_retry`.
    -   [ ] Refactor `generate_content` to use `asyncio.wait_for` on the new async helper.

-   **Testing:**
    -   [ ] Add/update unit test for `EnhancedLLMService.generate_content`.
    -   [ ] Mock `llm_retry_handler.generate_content` as an `AsyncMock`.
    -   [ ] Test success path, asserting the mock was awaited.
    -   [ ] Test timeout path, asserting `OperationTimeoutError` is raised.

### **Task S-02 (P1): Fortify LLM JSON Parsing Service**

-   **Root Cause:** `llm_cv_parser_service.py` performs unsafe `json.loads()` on raw LLM responses, which are not guaranteed to be valid JSON, causing `JSONDecodeError` crashes that halt the entire workflow. (Audit ID: `CR-01`, `S-01`)
-   **Impacted Modules:** `src/services/llm_cv_parser_service.py`, `src/agents/parser_agent.py`.
-   **Required Changes:** Make `_generate_and_parse_json` resilient by validating the response, handling markdown code fences, wrapping the `json.loads()` call in a `try...except` block, and raising a specific `LLMResponseParsingError` on failure.

-   **Code Implementation (`src/services/llm_cv_parser_service.py`):**

    **Before:**
    ```python
    # In LLMCVParserService._generate_and_parse_json
    llm_response = await self.llm_service.generate(...)
    return json.loads(llm_response.content) # Unsafe call
    ```

    **After:**
    ```python
    import re
    import json
    from ..utils.exceptions import LLMResponseParsingError

    # In LLMCVParserService._generate_and_parse_json
    async def _generate_and_parse_json(...) -> Any:
        llm_response = await self.llm_service.generate(...)
        raw_text = llm_response.content
        if not raw_text or not isinstance(raw_text, str):
            raise LLMResponseParsingError("Received empty or non-string response from LLM.", raw_response=str(raw_text))

        json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            start_index = raw_text.find("{")
            end_index = raw_text.rfind("}") + 1
            if start_index == -1 or end_index == 0:
                raise LLMResponseParsingError("No valid JSON object found in the LLM response.", raw_response=raw_text)
            json_str = raw_text[start_index:end_index]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON from LLM response: %s", e)
            raise LLMResponseParsingError(f"Could not parse JSON. Error: {e}", raw_response=raw_text) from e
    ```

-   **Implementation Checklist:**
    -   [ ] Update `_generate_and_parse_json` in `llm_cv_parser_service.py`.
    -   [ ] Ensure `parser_agent.py` catches `LLMResponseParsingError` and updates `AgentState` with an error message instead of crashing.

-   **Testing:**
    -   [ ] Add unit test for `_generate_and_parse_json`.
    -   [ ] Test cases: valid JSON, markdown-wrapped JSON, empty string, malformed JSON.

---

## 3. Agents & Core Logic (P2)

### **Task A-01 (P1): Fix Broken Imports for Error Handling**

-   **Root Cause:** The error handling framework was consolidated into `src/error_handling/`, but `import` statements in several agents were not updated, causing `E0401: Unable to import` errors. This is a critical build-breaking issue. (Audit ID: `AD-02`)
-   **Impacted Modules:** `src/agents/cleaning_agent.py`, `src/agents/formatter_agent.py`, etc.
-   **Required Changes:** Globally search for all imports from old `src/utils/` error modules and replace them with the correct paths to `src/error_handling/`.

-   **Code Implementation (Example in `src/agents/formatter_agent.py`):**

    **Before:**
    ```python
    from ..utils.agent_error_handling import with_node_error_handling # Broken import
    from ..utils.error_utils import handle_errors # Broken import
    ```

    **After:**
    ```python
    from ..error_handling.agent_error_handler import with_node_error_handling # Correct import
    from ..error_handling.decorators import handle_errors # Correct import
    ```

-   **Implementation Checklist:**
    -   [ ] Search codebase for imports from `src/utils/error*` and `src/utils/agent_error*`.
    -   [ ] Replace all found imports with the correct path from `src/error_handling/`.
    -   [ ] Run `pylint src/` to confirm all `E0401` import errors are resolved.

-   **Testing:**
    -   [ ] The primary validation is a clean `pylint` run.
    -   [ ] CI pipeline should be configured to fail on `pylint` errors.

### **Task A-02 (P2): Enforce Singleton DI Container Usage**

-   **Root Cause:** The "DI Schism". A central, singleton `DependencyContainer` exists but is not used universally. `src/agents/specialized_agents.py` manually instantiates agents, bypassing the container and creating duplicate service instances. This also leads to broken constructor calls. (Audit IDs: `AD-02`, `AD-03`, `CONTRACT-01`)
-   **Impacted Modules:** `src/agents/specialized_agents.py`, `src/core/dependency_injection.py`, `src/core/application_startup.py`.
-   **Required Changes:**
    -   **Phase 1 (Immediate Fix):** Refactor the factory functions in `specialized_agents.py` to delegate agent creation to the central DI container. This immediately resolves the runtime errors caused by incorrect constructor signatures.
    -   **Phase 2 (Future Task):** Eliminate the factory functions entirely, refactoring all call sites to request agents directly from the container.
    -   **Immediate Action:**
        1.  Refactor `get_container()` in `dependency_injection.py` to be a thread-safe singleton provider.
        2.  Refactor all factory functions in `specialized_agents.py` to retrieve agent instances from the global container via `get_container()`.
        3.  Remove all legacy calls to the deleted `register_agents` method.

-   **Code Implementation (`src/core/dependency_injection.py`):**
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
        return _global_container
    ```-   **Code Implementation (`src/agents/specialized_agents.py`):**

    **Before:**
    ```python
    def create_quality_assurance_agent(llm_service, error_recovery_service, progress_tracker) -> "QualityAssuranceAgent":
        # BUG: Incorrect signature, causes TypeError
        return QualityAssuranceAgent(error_recovery_service=error_recovery_service, ...)
    ```

    **After:**
    ```python
    from ..core.dependency_injection import get_container

    def create_quality_assurance_agent() -> "QualityAssuranceAgent":
        """Factory that correctly gets a QA agent from the central DI container."""
        container = get_container()
        # The DI container knows the correct constructor signature.
        # A session_id must be provided by the calling context.
        session_id = get_current_session_id() # Assumes a helper function exists
        return container.get_by_name('qa_agent', session_id=session_id)
    ```
-   **Implementation Checklist:**
    -   [ ] Update `get_container()` to be a thread-safe singleton.
    -   [ ] Refactor all factory functions in `specialized_agents.py` to use `get_container()`. This fixes `CONTRACT-01`.
    -   [ ] Remove any calls to the non-existent `container.register_agents()`.
    -   [ ] The calling context (e.g., Streamlit callbacks) must be responsible for providing the `session_id`.

-   **Testing:**
    -   [ ] Unit test for `get_container()` to assert it always returns the same object ID.
    -   [ ] Integration tests should be updated to verify agents receive their dependencies correctly from the single container.

### **Task A-03 (P2): Consolidate CV Structuring Utilities**

-   **Root Cause:** Redundant utility modules (`cv_structure_utils.py`, `cv_conversion_utils.py`) and misplaced factory logic (`create_empty_cv_structure` in `ParserAgent`) exist after refactoring. (Audit ID: `DP-01`, `BE-03`)
-   **Impacted Modules:** `src/agents/parser_agent.py`, `src/agents/cv_structure_utils.py`, `src/agents/cv_conversion_utils.py`, `src/models/data_models.py`.
-   **Required Changes:**
    1.  **Create New Module:** Create a new, consolidated utility module, e.g., `src/utils/cv_data_factory.py`.
    2.  **Move & Merge:** Move the `create_empty_cv_structure` logic from `ParserAgent` to a static method `StructuredCV.create_empty()` on the model itself. Move all functions from `cv_structure_utils.py` and `cv_conversion_utils.py` into the new factory module. Perform a diff to resolve any overlaps.
    3.  **Refactor Imports:** Globally search and replace all `import` statements that point to the old files to use the new `cv_data_factory.py` module and the `StructuredCV.create_empty()` static method.
    4.  **Delete Old Files:** Once all dependencies are updated, delete `cv_structure_utils.py` and `cv_conversion_utils.py`.

-   **Code Implementation (`src/models/data_models.py`):**
    ```python
    # Inside the StructuredCV class
    @staticmethod
    def create_empty(job_data: Optional["JobDescriptionData"] = None) -> "StructuredCV":
        """Creates an empty CV structure for the "Start from Scratch" option."""
        # ... (The logic from parser_agent.create_empty_cv_structure goes here) ...
        structured_cv = StructuredCV()
        # ...
        return structured_cv
    ```
-   **Implementation Checklist:**
    -   [ ] Add the `create_empty` static method to the `StructuredCV` model.
    -   [ ] Create `src/utils/cv_data_factory.py` and consolidate functions from old utils.
    -   [ ] Refactor `ParserAgent` to call `StructuredCV.create_empty()`.
    -   [ ] Refactor all other imports to use the new factory module.
    -   [ ] Delete the old utility files.

-   **Testing:**
    -   [ ] Add a unit test for `StructuredCV.create_empty`.
    -   [ ] Update `ParserAgent` tests to reflect the change.

---

## 4. Data Models (P2)

### **Task M-01 (P2): Finalize Agent Output Model Consolidation**

-   **Root Cause:** The `src/models/agent_output_models.py` file was created, but agents were not refactored to use it. Numerous deprecated, agent-specific model files (`cleaning_agent_models.py`, etc.) still exist, violating the DRY principle. (Audit ID: `DM-01`, `CONTRACT-02`)
-   **Impacted Modules:** `src/models/`, all agent implementations (`src/agents/*.py`).
-   **Required Changes:**
    1.  Systematically review all agent `run_async` methods.
    2.  Refactor them to return `AgentResult` where the `output_data` field is an instance of a canonical model from `src/models/agent_output_models.py`.
    3.  Update all call sites and unit tests to expect the new canonical models.
    4.  Delete all deprecated agent-specific model files from `src/models/`.

-   **Implementation Checklist:**
    -   [ ] Audit all files in `src/agents/`.
    -   [ ] For each agent, identify its current output model.
    -   [ ] Ensure a canonical equivalent exists in `agent_output_models.py`.
    -   [ ] Refactor the agent's `run_async` method to use the canonical model.
    -   [ ] After all agents are migrated, delete the old model files (e.g., `src/models/cv_analyzer_models.py`).

-   **Testing:**
    -   [ ] All existing agent unit tests must be updated to assert the new, canonical output model type.

---

## 5. Code Hygiene and Documentation (P3)

### **Task H-01 (P3): Remediate Pervasive Code Smells**

-   **Root Cause:** Widespread use of `except Exception`, `logging-fstring-interpolation`, and inconsistent environment variable naming increases maintenance cost and masks potential bugs. (Audit ID: `CS-01`, `CS-02`, `NAMING-01`)
-   **Impacted Modules:** Entire codebase.
-   **Required Changes:**
    1.  **Exceptions:** Globally search for `except Exception` and replace with specific exception types (e.g., `except (KeyError, TypeError, json.JSONDecodeError)`).
    2.  **Logging:** Globally search for `logger.info(f"...")` (and other levels) and refactor to use lazy formatting: `logger.info("Message: %s", var)`.
    3.  **Config Naming:** Standardize on `SESSION_TIMEOUT_SECONDS` in both `.env.example` and `docker-compose.yml`.

-   **Implementation Checklist:**
    -   [ ] Perform a global search-and-replace for logging f-strings.
    -   [ ] Systematically review and refactor each `except Exception` block.
    -   [ ] Update `docker-compose.yml` to use `SESSION_TIMEOUT_SECONDS`.

-   **Testing:**
    -   [ ] `pylint` should report fewer `broad-except` and `logging-fstring-interpolation` warnings.
    -   [ ] Manually verify the session timeout setting is correctly applied from the environment.

### **Task H-02 (P3): Update Project Documentation**

-   **Root Cause:** The `README.md` contains an outdated project structure diagram that does not reflect the refactored architecture (e.g., omits `instance/` and `error_handling/`). (Audit ID: `AD-01`)
-   **Impacted Modules:** `README.md`.
-   **Required Changes:** Update the project structure diagram and text in `README.md` to accurately reflect the current layout.

-   **Implementation Checklist:**
    -   [ ] Generate a new directory tree diagram.
    -   [ ] Replace the old diagram in `README.md`.
    -   [ ] Update the descriptions for `instance/`, `src/frontend/`, and `src/error_handling/`.

-   **Testing:**
    -   [ ] Manual review of the updated `README.md` for accuracy.

## Implementation Checklist

1.  [ ] **Branch:** Create a new feature branch `refactor/architectural-remediation`.
2.  [ ] **P1 - Critical Fixes:**
    -   [ ] **Task CD-01:** Fix `docker-compose.yml` volume mount.
    -   [ ] **Task S-01:** Fix async contract in `EnhancedLLMService`.
    -   [ ] **Task S-02:** Harden JSON parsing in `llm_cv_parser_service.py`.
    -   [ ] **Task A-01:** Fix all broken `error_handling` imports.
3.  [ ] **P2 - Architectural Refactoring:**
    -   [ ] **Task A-02:** Enforce singleton DI container usage and fix agent factories.
    -   [ ] **Task M-01:** Consolidate all agent output models.
    -   [ ] **Task A-03:** Consolidate CV structuring utilities.
4.  [ ] **P3 - Code Hygiene:**
    -   [ ] **Task H-01:** Remediate code smells (exceptions, logging, config naming).
    -   [ ] **Task H-02:** Update `README.md` documentation.
5.  [ ] **Testing:** Run all unit and integration tests. Add new tests as specified in each task.
6.  [ ] **Linting:** Run `pylint src/ tests/` and ensure a clean report with no new errors.
7.  [ ] **Code Review:** Submit a pull request for review and merge after approval.

## Testing Strategy

-   **Unit Tests:** Each refactored component must have corresponding unit tests. Mocks should be used for external dependencies (`llm_service`, `chromadb`). Test both happy paths and failure scenarios (e.g., invalid input, exceptions from dependencies).
-   **Integration Tests:**
    -   Expand `test_cv_workflow_error_handling.py` to simulate `LLMResponseParsingError` and verify the workflow routes to the error handler and terminates gracefully.
    -   Add integration tests for the application startup sequence to validate DI container configuration and service readiness.
-   **Deployment Test:** Create a simple script to run after `docker-compose up` that verifies a test log message can be written and read from the mounted `instance/` volume.
-   **CI/CD Pipeline:** The CI pipeline must be configured to run `pylint` and `pytest` on every pull request. The build must fail if any `pylint` errors (E-codes) are present.

## Architectural Decisions

-   **Session ID for DI:** The `session_id` is established by the entry point of the workflow (e.g., a Streamlit callback in `src/frontend/callbacks.py`). This ID is then passed to the `CVWorkflowGraph` instance, which uses it for all subsequent requests to the DI container to retrieve the correct session-scoped agent instances. This ensures state isolation between concurrent user sessions.
-   **Agent Factories in `specialized_agents.py`:** A two-phase approach will be used.
    -   **Phase 1 (This Blueprint):** The factory functions will be immediately refactored to delegate agent creation to the central DI container (`get_container()`). This resolves critical runtime bugs related to incorrect constructor signatures.
    -   **Phase 2 (Future Task):** The factory functions will be eliminated entirely. All call sites will be updated to request agents directly from the DI container.
-   **Consolidation of CV Utilities:** `cv_structure_utils.py` and `cv_conversion_utils.py` will be consolidated. A new module (e.g., `src/utils/cv_data_factory.py`) will be created. All functions from both old files will be merged into the new module, imports will be refactored globally, and the old files will be deleted. The `create_empty_cv_structure` logic will be moved to a static method on the `StructuredCV` model.

---

## 6. Frontend Integration (P2)

### **Task FI-01 (P2): Integrate Streamlit UI with the EnhancedCVIntegration Layer**

-   **Root Cause:** The `EnhancedCVIntegration` layer has been implemented as a central facade for the application's backend services and workflows, but it is not yet consumed by the Streamlit UI (`app.py`). This leaves the application's frontend disconnected from the refactored, robust backend, preventing end-to-end functionality.
-   **Impacted Modules:** `app.py`, `src/frontend/callbacks.py`, `src/integration/enhanced_cv_system.py`.
-   **Required Changes:** Refactor the Streamlit callbacks in `app.py` and/or `src/frontend/callbacks.py` to instantiate and use `EnhancedCVIntegration` to execute all CV generation workflows. This will replace any direct, legacy calls to agents or services from the UI layer, fully adopting the intended architectural pattern.

-   **Implementation Checklist:**
    -   [ ] Identify all Streamlit UI actions (e.g., button clicks) that are intended to trigger a CV generation or processing workflow.
    -   [ ] In the corresponding callback functions, remove any existing logic that directly instantiates or calls agents/services.
    -   [ ] Add a call to `get_enhanced_cv_integration()` to get the central integration instance.
    -   [ ] Construct the appropriate `AgentState` object from the user's input collected by the Streamlit UI.
    -   [ ] Execute the workflow by calling `await integration.execute_workflow(workflow_type, agent_state, session_id)`.
    -   [ ] Update the UI to properly handle the results from the workflow, displaying either the final generated document path or any error messages returned.

-   **Testing:**
    -   [ ] Perform manual End-to-End (E2E) testing by running the Streamlit application (`streamlit run app.py`).
    -   [ ] Trigger a full CV generation workflow from the UI and verify that the process completes successfully and the final document is generated and downloadable.
    -   [ ] Test failure scenarios by providing invalid or incomplete input and confirming that user-friendly error messages (returned from the workflow) are displayed in the UI.

---

