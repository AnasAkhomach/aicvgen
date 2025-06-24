# TASK_BLUEPRINT: `aicvgen` Refactoring & Remediation (Revision 2)

This document outlines the prioritized tasks for refactoring the `aicvgen` codebase to address technical debt, enforce architectural consistency, and improve maintainability as identified in `Codebase Audit and Review_.md`. This revision incorporates feedback to defer specific post-MVP refactoring tasks.

## Overview

The primary goals of this refactoring phase are:
1.  **Stabilize the Core**: Eliminate critical duplication and contract breaches that risk runtime stability and data integrity.
2.  **Enforce Architectural Boundaries**: Refactor components to respect the Single Responsibility Principle and ensure logic resides in the correct layer.
3.  **Improve Maintainability**: Consolidate redundant logic, remove dead code, and enforce consistent coding and configuration patterns across the application.

---

## **Priority 1: Critical Stability & Consistency Fixes**

### **Task C-1: Unify State and Configuration (AD-01, CONF-01)**

-   **Root Cause**: Critical duplication of state initialization logic (`state_helpers.py`) and logging configuration (`logging_config.py`, `logging_config_simple.py`), creating multiple sources of truth.
-   **Priority**: **P1 (Critical)**
-   **Impacted Modules**:
    -   `src/core/state_helpers.py` (to be deleted)
    -   `src/frontend/state_helpers.py` (to be deleted)
    -   `src/config/logging_config.py` (to be refactored)
    -   `src/config/logging_config_simple.py` (to be deleted)
    -   `src/config/logging_config.py.backup` (to be deleted)
    -   `src/core/application_startup.py` (to be updated)
    -   `src/frontend/callbacks.py` (to be updated)

#### **Required Changes**

1.  **Centralize State Helpers**:
    -   Create a new canonical module: `src/utils/state_utils.py`.
    -   Merge the logic from `src/core/state_helpers.py` and `src/frontend/state_helpers.py` into this new file.
    -   Delete the two redundant `state_helpers.py` files.
    -   Update all imports to point to the new utility.

    **Before (`src/frontend/callbacks.py`):**
    ```python
    from ..core.state_helpers import create_initial_agent_state
    ```

    **After (`src/utils/state_utils.py` is created and `src/frontend/callbacks.py` is updated):**
    ```python
    # In src/frontend/callbacks.py
    from ..utils.state_utils import create_initial_agent_state
    ```

2.  **Unify Logging Configuration**:
    -   Refactor `src/config/logging_config.py` to be environment-aware. It should read an environment variable (e.g., `APP_ENV`) to apply the correct logging format (e.g., simple for `development`, structured JSON for `production`).
    -   Delete `src/config/logging_config_simple.py` and `src/config/logging_config.py.backup`.
    -   Ensure `application_startup.py` calls the single, unified `setup_logging()` function.

    **After (`src/config/logging_config.py`):**
    ```python
    import os
    import logging
    from pythonjsonlogger import jsonlogger

    def setup_logging():
        app_env = os.getenv("APP_ENV", "development")
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Clear existing handlers to prevent duplicate logging
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        handler = logging.StreamHandler()

        if app_env == "production":
            # Structured JSON logging for production
            formatter = jsonlogger.JsonFormatter(
                '%(asctime)s %(name)s %(levelname)s %(message)s'
            )
        else:
            # Simple console logging for development
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        logging.info(f"Logging initialized for '{app_env}' environment at level {log_level}.")

    # ... other logging helpers ...
    ```

#### **Implementation Checklist**

-   [ ] Create `src/utils/state_utils.py` and move `create_initial_agent_state` logic into it.
-   [ ] Update `src/frontend/callbacks.py` and any other call sites to import from `src/utils/state_utils.py`.
-   [ ] Delete `src/core/state_helpers.py` and `src/frontend/state_helpers.py`.
-   [ ] Refactor `src/config/logging_config.py` to be environment-aware.
-   [ ] Delete `src/config/logging_config_simple.py` and `src/config/logging_config.py.backup`.
-   [ ] Verify `application_startup.py` uses the unified logging setup.

#### **Tests to Add**

-   **Unit Test**: `test_create_initial_agent_state` in `tests/unit/test_state_utils.py`.
-   **Unit Test**: `test_setup_logging_dev_mode` and `test_setup_logging_prod_mode` in `tests/unit/test_logging_config.py`.

### **Task C-2: Enforce Strict Agent Return Contracts (CB-01)**

-   **Root Cause**: `CVAnalyzerAgent.analyze_cv` returns different Pydantic models on success vs. failure, violating its implicit contract and risking `AttributeError` in the orchestration layer.
-   **Priority**: **P1 (Critical)**
-   **Impacted Modules**:
    -   `src/agents/cv_analyzer_agent.py`

#### **Required Changes**

-   Refactor `analyze_cv` to *always* return the expected data shape on failure and raise a custom exception for error conditions. The `run_async` method will catch this and format a proper `AgentResult`.

    **Before (`src/agents/cv_analyzer_agent.py`):**
    ```python
    # In analyze_cv method
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM response during CV analysis: {e}")
        # ...
        return fallback_extraction # This is a BasicCVInfo model
    ```

    **After (`src/agents/cv_analyzer_agent.py`):**
    ```python
    from ..utils.exceptions import AgentExecutionError, LLMResponseParsingError
    from ..models.agent_output_models import CVAnalyzerAgentOutput

    # In analyze_cv method
    # ...
    except json.JSONDecodeError as e:
        logger.error("Failed to decode JSON from LLM response: %s", e)
        # Raise a specific exception instead of returning a different type
        raise LLMResponseParsingError("Could not parse JSON from LLM response.") from e
    except Exception as e:
        logger.error("An unexpected error occurred during CV analysis: %s", e)
        raise AgentExecutionError(agent_name="CVAnalyzerAgent", message=str(e)) from e

    # In run_async method
    async def run_async(
        self, input_data: Any, context: "AgentExecutionContext"
    ) -> "AgentResult":
        try:
            # ...
            analysis_result_dict = await self.analyze_cv(input_data, context)
            output_data = CVAnalyzerAgentOutput(analysis_results=analysis_result_dict)
            return AgentResult(success=True, output_data=output_data)
        except (LLMResponseParsingError, AgentExecutionError) as e:
            logger.error("CV analysis failed: %s", e)
            return AgentResult(
                success=False,
                output_data=CVAnalyzerAgentOutput(analysis_results={}), # Return empty model on failure
                error_message=str(e),
                confidence_score=0.0
            )
    ```

#### **Implementation Checklist**

-   [ ] Create custom exceptions in `src/utils/exceptions.py` if they don't exist (e.g., `LLMResponseParsingError`).
-   [ ] Modify `cv_analyzer_agent.py` to raise exceptions on parsing/execution failure.
-   [ ] Update the `run_async` method to catch these exceptions and return a consistent `AgentResult` object with `success=False`.

#### **Tests to Add**

-   **Unit Test**: `test_analyze_cv_json_parsing_failure` to assert that `LLMResponseParsingError` is raised.
-   **Unit Test**: `test_run_async_failure_path` to assert it returns `AgentResult(success=False)` and an empty `CVAnalyzerAgentOutput`.

### **Task C-3: Align Linter and Runtime Environments (CONF-03)**

-   **Root Cause**: Mismatch between Python version in `.pylintrc` (3.13) and `Dockerfile` (3.11).
-   **Priority**: **P1 (Critical)**
-   **Impacted Modules**:
    -   `.pylintrc`

#### **Required Changes**

-   Update `.pylintrc` to match the `Dockerfile`'s Python version to ensure static analysis is accurate.

    **Before (`.pylintrc`):**
    ```ini
    py-version=3.13
    ```

    **After (`.pylintrc`):**
    ```ini
    py-version=3.11
    ```

#### **Implementation Checklist**

-   [ ] Edit the `py-version` line in `.pylintrc`.
-   [ ] Run `pylint` locally to ensure no new errors are introduced due to the version change.

#### **Tests to Add**

-   N/A (Configuration change, verified by CI pipeline).

---

## **Priority 2: Architectural Refinements**

### **Task A-2: Encapsulate Parsing Logic within ParserAgent (AD-02)**

-   **Root Cause**: CV parsing and structuring logic has leaked from `ParserAgent` into external utility modules (`cv_conversion_utils.py`, `cv_structure_utils.py`), breaking encapsulation.
-   **Priority**: **P2 (High)**
-   **Impacted Modules**:
    -   `src/agents/parser_agent.py`
    -   `src/agents/cv_conversion_utils.py` (to be deleted)
    -   `src/agents/cv_structure_utils.py` (to be deleted)

#### **Required Changes**

-   Move all data conversion and structuring logic from the two utility modules into `ParserAgent` as private helper methods. The functions from `cv_structure_utils.py` will serve as lower-level helpers called by the higher-level conversion logic from `cv_conversion_utils.py`, all within `ParserAgent`.
-   Refactor `ParserAgent.run_as_node` to be the single entry point that produces a fully validated `StructuredCV` object.
-   Delete the now-redundant utility files.

    **Before (`parser_agent.py`):**
    ```python
    from .cv_conversion_utils import convert_parsing_result_to_structured_cv

    class ParserAgent(EnhancedAgentBase):
        async def parse_cv_with_llm(self, cv_text: str, job_data: JobDescriptionData) -> StructuredCV:
            # ...
            parsing_result = await self.llm_cv_parser_service.parse_cv_with_llm(...)
            # External call to finish the job
            structured_cv = convert_parsing_result_to_structured_cv(parsing_result, ...)
            return structured_cv
    ```

    **After (`parser_agent.py`):**
    ```python
    class ParserAgent(EnhancedAgentBase):
        # ...
        def _determine_section_content_type(self, section_name: str) -> str:
            # Logic from cv_structure_utils.py
            # ...

        def _convert_parsing_result_to_structured_cv(self, ...) -> StructuredCV:
            # All logic from cv_conversion_utils.py is now here
            # It can call other private helpers like _determine_section_content_type
            # ...

        async def run_as_node(self, state: AgentState) -> Dict[str, Any]:
            # ...
            parsing_result = await self.llm_cv_parser_service.parse_cv_with_llm(...)
            # Internal call to a private method
            final_cv = self._convert_parsing_result_to_structured_cv(parsing_result, ...)
            return {"structured_cv": final_cv, ...}
    ```

#### **Implementation Checklist**

-   [ ] Copy methods from `cv_conversion_utils.py` and `cv_structure_utils.py` into `parser_agent.py` as private methods.
-   [ ] Refactor `ParserAgent` to call these new private methods.
-   [ ] Ensure `ParserAgent`'s public methods return fully validated `StructuredCV` or `JobDescriptionData` objects.
-   [ ] Delete `src/agents/cv_conversion_utils.py` and `src/agents/cv_structure_utils.py`.

#### **Tests to Add**

-   **Unit Test**: `test_parser_agent_full_conversion` to verify it correctly transforms raw text into a complete `StructuredCV`.

### **Task A-3: Consolidate Dependency Injection (CONF-02)**

-   **Root Cause**: The DI container has multiple, overlapping registration methods, creating an ambiguous and duplicative API.
-   **Priority**: **P2 (Medium)**
-   **Impacted Modules**:
    -   `src/core/dependency_injection.py`
    -   `src/core/application_startup.py`

#### **Required Changes**

-   Consolidate all registration logic into a single, idempotent `configure_container()` function. This function will be the single entry point for setting up all services and agents.
-   Remove the redundant `register_agents`, `register_services`, and `register_agents_and_services` methods.

    **Before (`dependency_injection.py`):**
    ```python
    class DependencyContainer:
        # ...
        def register_agents(self) -> None:
            # ...

        def register_agents_and_services(self):
            # ...

    def register_core_services(container: "DependencyContainer"):
        # ...
    ```

    **After (`dependency_injection.py`):**
    ```python
    def configure_container(container: "DependencyContainer", settings: "Settings"):
        """A single, idempotent function to configure the entire DI container."""
        # Register core services
        container.register_singleton("settings", Settings, factory=lambda: settings)
        # ... other services

        # Register LLM Service
        container.register_singleton(
            "EnhancedLLMService",
            EnhancedLLMService,
            factory=lambda: build_llm_service(container)
        )

        # Register all agents
        # ... parser agent registration
        # ... content writer agent registration
        # ... etc.
        logger.info("Dependency container configured.")

    class DependencyContainer:
        # No more register_* methods
        # ...
    ```
    **After (`application_startup.py`):**
    ```python
    from ..core.dependency_injection import get_container, configure_container
    from ..config.settings import get_config

    def initialize_application(user_api_key: str = ""):
        # ...
        container = get_container()
        settings = get_config()
        # Configure the container only if it hasn't been configured yet
        if not container.get_registrations().get("settings"):
             configure_container(container, settings)
        # ...
    ```

#### **Implementation Checklist**

-   [ ] Create the `configure_container` function in `dependency_injection.py`.
-   [ ] Move all registration logic from various methods into `configure_container`.
-   [ ] Delete the old `register_*` methods from `DependencyContainer`.
-   [ ] Update `application_startup.py` to call `configure_container` once.
-   [ ] Delete `src/core/dependency_injection.py.backup`.

#### **Tests to Add**

-   **Unit Test**: `test_configure_container` to verify all expected services and agents are registered correctly.

### **Task A-4: Centralize LLM JSON Parsing Logic (DUP-01, PERF-01)**

-   **Root Cause**: Duplication of the LLM JSON parsing logic in `EnhancedAgentBase` and `ResearchAgent`. The performance concern (`PERF-01`) is acknowledged but deferred.
-   **Priority**: **P2 (Medium)**
-   **Impacted Modules**:
    -   `src/agents/agent_base.py`
    -   `src/agents/research_agent.py`

#### **Required Changes**

1.  **Centralize Parsing**: Ensure the `_generate_and_parse_json` method in `EnhancedAgentBase` is the single, canonical implementation.
2.  **Refactor `ResearchAgent`**: Modify `ResearchAgent` to remove its local implementation and call the base class method via `super()`.

    **Before (`research_agent.py`):**
    ```python
    class ResearchAgent(EnhancedAgentBase):
        # ...
        async def analyze_job_description(self, ...):
            # ...
            # Duplicated logic to call LLM and parse JSON
            analysis = await self._generate_and_parse_json(...)
            return analysis
    ```

    **After (`research_agent.py`):**
    ```python
    class ResearchAgent(EnhancedAgentBase):
        # ...
        async def analyze_job_description(self, ...):
            # ...
            # Calls the unified method from the base class
            analysis = await super()._generate_and_parse_json(
                prompt=prompt,
                session_id=...,
                trace_id=...
            )
            return analysis
    ```

#### **Implementation Checklist**

-   [ ] Review `ResearchAgent`'s parsing logic to ensure no unique behavior is lost.
-   [ ] Refactor `ResearchAgent` to call `super()._generate_and_parse_json()`.
-   [ ] Remove the redundant private parsing method from `ResearchAgent`.

#### **Tests to Add**

-   **Unit Test**: `test_research_agent_uses_base_parser` to ensure it calls the super method.

---

## **Priority 3: Code Hygiene and Cleanup**

### **Task H-1: Standardize Data Model Identifiers (NAM-01)**

-   **Root Cause**: Inconsistent naming for the same logical ID (`id` vs `item_id`) across different Pydantic models.
-   **Priority**: **P3 (Low)**
-   **Impacted Modules**:
    -   `src/models/data_models.py`
    -   `src/models/quality_assurance_agent_models.py`
    -   Any code that references `item.id`.

#### **Required Changes**

-   Standardize on the more descriptive `item_id` and ensure type consistency (`UUID`).

    **Before (`data_models.py`):**
    ```python
    class Item(BaseModel):
        id: UUID = Field(default_factory=uuid4)
        # ...
    ```

    **After (`data_models.py`):**
    ```python
    class Item(BaseModel):
        item_id: UUID = Field(default_factory=uuid4, alias='id') # Use alias for backward compatibility if needed
        # ...
        class Config:
            populate_by_name = True
    ```

    **After (`quality_assurance_agent_models.py`):**
    ```python
    class ItemQualityResultModel(BaseModel):
        item_id: UUID # Was item_id: str
        # ...
    ```

#### **Implementation Checklist**

-   [ ] Rename `id` to `item_id` in the `Item` model. Consider using Pydantic's `alias` feature for a phased migration if a global find-and-replace is too risky.
-   [ ] Update `ItemQualityResultModel` to use `item_id: UUID`.
-   [ ] Use a global find-and-replace to update all references from `.id` to `.item_id` on `Item` objects.

#### **Tests to Add**

-   Run all existing tests to catch any missed renames.

### **Task H-2: Prune Deprecated and Orphaned Files (DEP-01)**

-   **Root Cause**: Obsolete scripts and backup files clutter the repository, increasing cognitive load.
-   **Priority**: **P3 (Low)**
-   **Impacted Modules**:
    -   `emergency_fix.py`
    -   `userinput.py`
    -   `scripts/migrate_logs.py`
    -   `scripts/optimization_demo.py`
    -   `*.backup` files

#### **Required Changes**

-   **Archive**: Move useful but non-production scripts like `optimization_demo.py` to a new `tools/` directory.
-   **Delete**: Remove confirmed obsolete files like `emergency_fix.py`, `userinput.py`, `migrate_logs.py`, and all `.backup` files from Git.

#### **Implementation Checklist**

-   [ ] Create a `tools/` directory at the project root.
-   [ ] `git mv scripts/optimization_demo.py tools/optimization_demo.py`
-   [ ] `git rm emergency_fix.py`
-   [ ] `git rm userinput.py`
-   [ ] `git rm scripts/migrate_logs.py`
-   [ ] `git rm src/core/dependency_injection.py.backup`
-   [ ] `git rm src/config/logging_config.py.backup`

---

## **Deferred Tasks (Post-MVP)**

### **Task A-1: Decompose Overloaded Content Writer Agent (SRP-01)**

-   **Status**: **DEFERRED**. This task will be addressed post-MVP to accelerate the delivery of core refactoring goals.
-   **Root Cause**: `EnhancedContentWriterAgent` violates the Single Responsibility Principle by handling content generation for multiple, distinct CV sections.
-   **Priority**: P2 (High)
-   **Impacted Modules**:
    -   `src/agents/enhanced_content_writer.py`
    -   `src/core/dependency_injection.py`
    -   `src/orchestration/cv_workflow_graph.py`
-   **High-Level Plan**:
    1.  Refactor `EnhancedContentWriterAgent` into an abstract base class `BaseContentWriterAgent`.
    2.  Create new, smaller, specialized agent classes that inherit from the base class (e.g., `ExperienceWriterAgent`, `QualificationWriterAgent`).
    3.  Update the DI container and the `cv_workflow_graph` to use the new specialized agents.

---

## **Implementation Checklist (Master)**

| Priority | Task ID | Summary                                       | Status        |
| :------- | :------ | :-------------------------------------------- | :------------ |
| P1       | C-1     | Unify State Helpers and Logging Config        | `[ ] TO DO`   |
| P1       | C-2     | Enforce Strict Agent Return Contracts         | `[ ] TO DO`   |
| P1       | C-3     | Align Linter and Runtime Environments         | `[ ] TO DO`   |
| P2       | A-2     | Encapsulate Parsing Logic in ParserAgent      | `[ ] TO DO`   |
| P2       | A-3     | Consolidate Dependency Injection              | `[ ] TO DO`   |
| P2       | A-4     | Centralize LLM JSON Parsing Logic             | `[ ] TO DO`   |
| P3       | H-1     | Standardize Data Model Identifiers (`item_id`)| `[ ] TO DO`   |
| P3       | H-2     | Prune Deprecated and Orphaned Files           | `[ ] TO DO`   |
| Post-MVP | A-1     | Decompose Overloaded Content Writer Agent     | `[ ] DEFERRED`|

---

## **Testing Strategy**

1.  **Pre-Refactor**: Ensure all existing `pytest` unit and integration tests are passing to establish a baseline.
2.  **During Refactor**: For each task, add or update unit tests to validate the specific changes. For example, when fixing a contract breach, add a test that explicitly checks the failure path for the correct exception and return type.
3.  **Post-Refactor**:
    -   Run the full `pytest` suite to catch any regressions.
    -   Manually run the Streamlit application and perform an end-to-end test of the main CV generation workflow to ensure no user-facing functionality was broken.
    -   Run `pylint` across the codebase to confirm the new configuration is working and to catch any new linting errors.

---

This revision adds a new "Architectural Principles" section for high-level guidance and formally adds the deferred performance task to the plan.

---

## **Architectural Principles**

This refactoring effort will be guided by the following core principles:

1.  **Fail-fast Configuration**: The application must detect and report misconfigurations (e.g., missing API keys) at startup, not during runtime.
2.  **Single Source of Truth**: All duplicated logic, especially for state, configuration, and common utilities, must be consolidated into a single, canonical module.
3.  **Constructor-based Dependency Injection**: All services and agents must receive their dependencies through their `__init__` constructor. In-method instantiation of dependencies is forbidden.
4.  **Strict Contract Adherence**: All functions and agent nodes must strictly adhere to their declared input and output data contracts (Pydantic models or `AgentState` fields). Errors should be signaled via exceptions, not by returning alternative data shapes.
5.  **Isolate Responsibilities**: Logic must reside in the appropriate layer. Agents are responsible for business logic, services for external interactions, and utilities for shared, stateless functions.

---
*Existing P1, P2, P3 tasks remain as defined in the previous response. The following section is updated.*
---

## **Deferred Tasks (Post-MVP)**

### **Task A-1: Decompose Overloaded Content Writer Agent (SRP-01)**

-   **Status**: **DEFERRED**. This task will be addressed post-MVP to accelerate the delivery of core refactoring goals.
-   **Root Cause**: `EnhancedContentWriterAgent` violates the Single Responsibility Principle by handling content generation for multiple, distinct CV sections.
-   **Priority**: P2 (High)
-   **Impacted Modules**:
    -   `src/agents/enhanced_content_writer.py`
    -   `src/core/dependency_injection.py`
    -   `src/orchestration/cv_workflow_graph.py`
-   **High-Level Plan**:
    1.  Refactor `EnhancedContentWriterAgent` into an abstract base class `BaseContentWriterAgent`.
    2.  Create new, smaller, specialized agent classes that inherit from the base class (e.g., `ExperienceWriterAgent`, `QualificationWriterAgent`).
    3.  Update the DI container and the `cv_workflow_graph` to use the new specialized agents.

### **Task P-1: Address Blocking I/O in Async Context (PERF-01)**

-   **Status**: **DEFERRED**. The performance impact is currently negligible. This task will be addressed if the operation becomes a bottleneck.
-   **Root Cause**: The `_generate_and_parse_json` method in `agent_base.py` uses a synchronous `re.search` call within an `async` function, which can block the event loop.
-   **Priority**: P3 (Low)
-   **Impacted Modules**:
    -   `src/agents/agent_base.py`
-   **High-Level Plan**:
    1.  Wrap the `re.search` call and subsequent processing in `asyncio.to_thread` to run it in a separate thread, preventing it from blocking the main event loop.

    **After (`src/agents/agent_base.py`):**
    ```python
    import asyncio
    # ...
    async def _generate_and_parse_json(self, ...):
        # ...
        raw_text = llm_response.content

        def _parse_json_sync(text: str) -> Dict[str, Any]:
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # ... fallback logic ...
                json_str = text[start_index:end_index]
            return json.loads(json_str)

        try:
            # Run the synchronous parsing logic in a separate thread
            parsed_data = await asyncio.to_thread(_parse_json_sync, raw_text)
            return parsed_data
        except json.JSONDecodeError as e:
            # ... handle error ...
    ```

---

## **Implementation Checklist (Master)**

| Priority | Task ID | Summary                                       | Status        |
| :------- | :------ | :-------------------------------------------- | :------------ |
| P1       | C-1     | Unify State Helpers and Logging Config        | `[ ] TO DO`   |
| P1       | C-2     | Enforce Strict Agent Return Contracts         | `[ ] TO DO`   |
| P1       | C-3     | Align Linter and Runtime Environments         | `[ ] TO DO`   |
| P2       | A-2     | Encapsulate Parsing Logic in ParserAgent      | `[ ] TO DO`   |
| P2       | A-3     | Consolidate Dependency Injection              | `[ ] TO DO`   |
| P2       | A-4     | Centralize LLM JSON Parsing Logic             | `[ ] TO DO`   |
| P3       | H-1     | Standardize Data Model Identifiers (`item_id`)| `[ ] TO DO`   |
| P3       | H-2     | Prune Deprecated and Orphaned Files           | `[ ] TO DO`   |
| Post-MVP | A-1     | Decompose Overloaded Content Writer Agent     | `[ ] DEFERRED`|
| Post-MVP | P-1     | Address Blocking Regex Call in Async Context  | `[ ] DEFERRED`|

---

This final revision adds a "Verification" section to each task, providing explicit steps to confirm that the changes have been implemented correctly and have not introduced regressions. This makes the blueprint more robust and actionable for the development team.

---

## **Architectural Principles**

This refactoring effort will be guided by the following core principles:

1.  **Fail-fast Configuration**: The application must detect and report misconfigurations (e.g., missing API keys) at startup, not during runtime.
2.  **Single Source of Truth**: All duplicated logic, especially for state, configuration, and common utilities, must be consolidated into a single, canonical module.
3.  **Constructor-based Dependency Injection**: All services and agents must receive their dependencies through their `__init__` constructor. In-method instantiation of dependencies is forbidden.
4.  **Strict Contract Adherence**: All functions and agent nodes must strictly adhere to their declared input and output data contracts (Pydantic models or `AgentState` fields). Errors should be signaled via exceptions, not by returning alternative data shapes.
5.  **Isolate Responsibilities**: Logic must reside in the appropriate layer. Agents are responsible for business logic, services for external interactions, and utilities for shared, stateless functions.

---

## **Priority 1: Critical Stability & Consistency Fixes**

### **Task C-1: Unify State and Configuration (AD-01, CONF-01)**

-   **Root Cause**: Critical duplication of state initialization logic (`state_helpers.py`) and logging configuration (`logging_config.py`, `logging_config_simple.py`), creating multiple sources of truth.
-   **Priority**: **P1 (Critical)**
-   **Impacted Modules**:
    -   `src/core/state_helpers.py` (to be deleted)
    -   `src/frontend/state_helpers.py` (to be deleted)
    -   `src/config/logging_config.py` (to be refactored)
    -   `src/config/logging_config_simple.py` (to be deleted)
    -   `src/config/logging_config.py.backup` (to be deleted)
    -   `src/core/application_startup.py` (to be updated)
    -   `src/frontend/callbacks.py` (to be updated)

#### **Required Changes**

1.  **Centralize State Helpers**:
    -   Create a new canonical module: `src/utils/state_utils.py`.
    -   Merge the logic from `src/core/state_helpers.py` and `src/frontend/state_helpers.py` into this new file.
    -   Delete the two redundant `state_helpers.py` files.
    -   Update all imports to point to the new utility.

2.  **Unify Logging Configuration**:
    -   Refactor `src/config/logging_config.py` to be environment-aware. It should read an environment variable (e.g., `APP_ENV`) to apply the correct logging format (e.g., simple for `development`, structured JSON for `production`).
    -   Delete `src/config/logging_config_simple.py` and `src/config/logging_config.py.backup`.
    -   Ensure `application_startup.py` calls the single, unified `setup_logging()` function.

#### **Implementation Checklist**

-   [ ] Create `src/utils/state_utils.py` and move `create_initial_agent_state` logic into it.
-   [ ] Update `src/frontend/callbacks.py` and any other call sites to import from `src/utils/state_utils.py`.
-   [ ] Delete `src/core/state_helpers.py` and `src/frontend/state_helpers.py`.
-   [ ] Refactor `src/config/logging_config.py` to be environment-aware.
-   [ ] Delete `src/config/logging_config_simple.py` and `src/config/logging_config.py.backup`.
-   [ ] Verify `application_startup.py` uses the unified logging setup.

#### **Tests to Add**

-   **Unit Test**: `test_create_initial_agent_state` in `tests/unit/test_state_utils.py`.
-   **Unit Test**: `test_setup_logging_dev_mode` and `test_setup_logging_prod_mode` in `tests/unit/test_logging_config.py`.

#### **Verification**

-   [ ] Run the application locally with `APP_ENV=development` and confirm logs are in the simple format.
-   [ ] Run the application locally with `APP_ENV=production` and confirm logs are in JSON format.
-   [ ] Confirm the application starts and a new CV generation can be initiated, proving the new state helper is being called correctly.

### **Task C-2: Enforce Strict Agent Return Contracts (CB-01)**

-   **Root Cause**: `CVAnalyzerAgent.analyze_cv` returns different Pydantic models on success vs. failure, violating its implicit contract and risking `AttributeError` in the orchestration layer.
-   **Priority**: **P1 (Critical)**
-   **Impacted Modules**:
    -   `src/agents/cv_analyzer_agent.py`

#### **Required Changes**

-   Refactor `analyze_cv` to *always* return the expected data shape on failure and raise a custom exception for error conditions. The `run_async` method will catch this and format a proper `AgentResult`.

#### **Implementation Checklist**

-   [ ] Create custom exceptions in `src/utils/exceptions.py` (e.g., `LLMResponseParsingError`).
-   [ ] Modify `cv_analyzer_agent.py`'s `analyze_cv` method to raise exceptions on parsing/execution failure.
-   [ ] Update the `run_async` method to catch these exceptions and return a consistent `AgentResult` object with `success=False` and an empty `CVAnalyzerAgentOutput`.

#### **Tests to Add**

-   **Unit Test**: `test_analyze_cv_json_parsing_failure` to assert that `LLMResponseParsingError` is raised.
-   **Unit Test**: `test_run_async_failure_path` to assert it returns `AgentResult(success=False)` and an empty `CVAnalyzerAgentOutput`.

#### **Verification**

-   [ ] Run the workflow with an input that causes the CV Analyzer to fail (e.g., malformed CV text).
-   [ ] Confirm the application does not crash and instead gracefully handles the error, showing a message in the UI.

### **Task C-3: Align Linter and Runtime Environments (CONF-03)**

-   **Root Cause**: Mismatch between Python version in `.pylintrc` (3.13) and `Dockerfile` (3.11).
-   **Priority**: **P1 (Critical)**
-   **Impacted Modules**:
    -   `.pylintrc`

#### **Required Changes**

-   Update `.pylintrc` to match the `Dockerfile`'s Python version to ensure static analysis is accurate.

#### **Implementation Checklist**

-   [ ] Edit the `py-version` line in `.pylintrc` from `3.13` to `3.11`.
-   [ ] Run `pylint src/` locally to ensure no new errors are introduced due to the version change.

#### **Tests to Add**

-   N/A (Configuration change, verified by CI pipeline).

#### **Verification**

-   [ ] The CI pipeline's `pylint` step should pass without errors related to Python version.

---

## **Priority 2: Architectural Refinements**

### **Task A-2: Encapsulate Parsing Logic within ParserAgent (AD-02)**

-   **Root Cause**: CV parsing and structuring logic has leaked from `ParserAgent` into external utility modules (`cv_conversion_utils.py`, `cv_structure_utils.py`), breaking encapsulation.
-   **Priority**: **P2 (High)**
-   **Impacted Modules**:
    -   `src/agents/parser_agent.py`
    -   `src/agents/cv_conversion_utils.py` (to be deleted)
    -   `src/agents/cv_structure_utils.py` (to be deleted)

#### **Required Changes**

-   Move all data conversion and structuring logic from the two utility modules into `ParserAgent` as private helper methods.
-   Refactor `ParserAgent.run_as_node` to be the single entry point that produces a fully validated `StructuredCV` object.
-   Delete the now-redundant utility files.

#### **Implementation Checklist**

-   [ ] Copy methods from `cv_conversion_utils.py` and `cv_structure_utils.py` into `parser_agent.py` as private methods (e.g., `_convert_parsing_result_to_structured_cv`).
-   [ ] Refactor `ParserAgent` to call these new private methods.
-   [ ] Ensure `ParserAgent`'s public methods return fully validated `StructuredCV` or `JobDescriptionData` objects.
-   [ ] Delete `src/agents/cv_conversion_utils.py` and `src/agents/cv_structure_utils.py`.

#### **Tests to Add**

-   **Unit Test**: `test_parser_agent_full_conversion` to verify it correctly transforms raw text into a complete `StructuredCV`.

#### **Verification**

-   [ ] Run the main CV generation workflow and confirm that the CV is parsed and structured correctly in the UI.
-   [ ] Verify that a `git status` shows the two utility files have been deleted.

### **Task A-3: Consolidate Dependency Injection (CONF-02)**

-   **Root Cause**: The DI container has multiple, overlapping registration methods, creating an ambiguous and duplicative API.
-   **Priority**: **P2 (Medium)**
-   **Impacted Modules**:
    -   `src/core/dependency_injection.py`
    -   `src/core/application_startup.py`

#### **Required Changes**

-   Consolidate all registration logic into a single, idempotent `configure_container()` function.
-   Remove the redundant `register_agents`, `register_services`, and `register_agents_and_services` methods.

#### **Implementation Checklist**

-   [ ] Create the `configure_container` function in `dependency_injection.py`.
-   [ ] Move all registration logic from various methods into `configure_container`.
-   [ ] Delete the old `register_*` methods from `DependencyContainer`.
-   [ ] Update `application_startup.py` to call `configure_container` once.
-   [ ] Delete `src/core/dependency_injection.py.backup`.

#### **Tests to Add**

-   **Unit Test**: `test_configure_container` to verify all expected services and agents are registered correctly.

#### **Verification**

-   [ ] Start the application and confirm there are no startup errors related to dependency resolution.
-   [ ] All agent and service functionality should work as before.

### **Task A-4: Centralize LLM JSON Parsing Logic (DUP-01, PERF-01)**

-   **Root Cause**: Duplication of the LLM JSON parsing logic in `EnhancedAgentBase` and `ResearchAgent`. The performance concern (`PERF-01`) is acknowledged but deferred.
-   **Priority**: **P2 (Medium)**
-   **Impacted Modules**:
    -   `src/agents/agent_base.py`
    -   `src/agents/research_agent.py`

#### **Required Changes**

-   Ensure the `_generate_and_parse_json` method in `EnhancedAgentBase` is the single, canonical implementation.
-   Refactor `ResearchAgent` to remove its local implementation and call the base class method via `super()`.

#### **Implementation Checklist**

-   [ ] Review `ResearchAgent`'s parsing logic to ensure no unique behavior is lost.
-   [ ] Refactor `ResearchAgent` to call `super()._generate_and_parse_json()`.
-   [ ] Remove the redundant private parsing method from `ResearchAgent`.

#### **Tests to Add**

-   **Unit Test**: `test_research_agent_uses_base_parser` to ensure it calls the super method.

#### **Verification**

-   [ ] Run the workflow and confirm that the `research_node` executes successfully, proving it can still parse LLM responses via the base class method.

---

## **Priority 3: Code Hygiene and Cleanup**

### **Task H-1: Standardize Data Model Identifiers (NAM-01)**

-   **Root Cause**: Inconsistent naming for the same logical ID (`id` vs `item_id`) across different Pydantic models.
-   **Priority**: **P3 (Low)**
-   **Impacted Modules**:
    -   `src/models/data_models.py`
    -   `src/models/quality_assurance_agent_models.py`
    -   Any code that references `item.id`.

#### **Required Changes**

-   Standardize on the more descriptive `item_id` and ensure type consistency (`UUID`).

#### **Implementation Checklist**

-   [ ] Rename `id` to `item_id` in the `Item` model in `data_models.py`.
-   [ ] Update `ItemQualityResultModel` in `quality_assurance_agent_models.py` to use `item_id: UUID`.
-   [ ] Use a global find-and-replace to update all references from `.id` to `.item_id` on `Item` objects.

#### **Tests to Add**

-   Run all existing tests to catch any missed renames.

#### **Verification**

-   [ ] The entire test suite (`pytest`) must pass.
-   [ ] Manually verify that user actions in the "Review & Edit" tab (Accept/Regenerate), which rely on `item_id`, function correctly.

### **Task H-2: Prune Deprecated and Orphaned Files (DEP-01)**

-   **Root Cause**: Obsolete scripts and backup files clutter the repository, increasing cognitive load.
-   **Priority**: **P3 (Low)**
-   **Impacted Modules**:
    -   `emergency_fix.py`, `userinput.py`, `scripts/migrate_logs.py`, `scripts/optimization_demo.py`, `*.backup` files

#### **Required Changes**

-   **Archive**: Move useful but non-production scripts like `optimization_demo.py` to a new `tools/` directory.
-   **Delete**: Remove confirmed obsolete files from the Git repository.

#### **Implementation Checklist**

-   [ ] Create a `tools/` directory at the project root.
-   [ ] `git mv scripts/optimization_demo.py tools/optimization_demo.py`
-   [ ] `git rm emergency_fix.py`
-   [ ] `git rm userinput.py`
-   [ ] `git rm scripts/migrate_logs.py`
-   [ ] `git rm src/core/dependency_injection.py.backup`
-   [ ] `git rm src/config/logging_config.py.backup`

#### **Tests to Add**

-   N/A (File system cleanup).

#### **Verification**

-   [ ] Confirm the deleted files are no longer present in the repository.
-   [ ] Confirm the application still builds and runs correctly after the file removals.