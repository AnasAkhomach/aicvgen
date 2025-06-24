# TASK_BLUEPRINT.md

## Overview

This blueprint outlines the stabilization tasks required to resolve critical runtime failures in the `anasakhomach-aicvgen` system. The primary objective is to fix the `TypeError: object NoneType can't be used in 'await' expression` (ERR-01) by implementing a fail-fast strategy for service configuration. This involves ensuring the `EnhancedLLMService` crashes loudly and immediately if the `GEMINI_API_KEY` is not configured, preventing the propagation of `None` objects into the application's workflow. The secondary objective is to resolve data contract breaches (CB-01) between agents and the orchestrator to ensure predictable state transitions.

---

## 1. Service Layer: Fail-Fast LLM Service Initialization (P1)

### 1.1. Root Cause & Priority

-   **ID:** `ERR-01`
-   **Priority:** P1 (Critical Blocker)
-   **Cause:** The `get_llm_service()` singleton factory in `src/services/llm_service.py` silently catches initialization errors (e.g., missing `GEMINI_API_KEY`) and returns `None`. This `None` object is injected as a dependency into agents, causing a `TypeError` when an `await` operation is attempted on it. This violates the "fail-fast" and "crash loud" principles.

### 1.2. Impacted Modules

-   `src/utils/exceptions.py` (New)
-   `src/services/llm_service.py`
-   `app.py` (or main Streamlit entry point)

### 1.3. Implementation Plan

#### 1.3.1. Create a Specific `ConfigurationError` Exception

A dedicated exception makes error handling explicit and robust.

-   **File:** `src/utils/exceptions.py`
-   **Logic:** Add a new custom exception class.

```python
# src/utils/exceptions.py

class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass

class ConfigurationError(Exception):
    """Raised for critical configuration errors that prevent startup."""
    pass
```

#### 1.3.2. Refactor `EnhancedLLMService` to Fail Fast

Modify the service constructor to raise the new `ConfigurationError` with a clear, actionable message.

-   **File:** `src/services/llm_service.py`
-   **Class:** `EnhancedLLMService`
-   **Method:** `__init__`

**Before:**
```python
# src/services/llm_service.py

class EnhancedLLMService:
    def __init__(self, user_api_key: Optional[str] = None):
        # ...
        api_key = self.user_api_key or self.primary_api_key or self.fallback_api_key
        if not api_key:
            raise ValueError(
                "No Gemini API key found. Please provide your API key or "
                "set GEMINI_API_KEY environment variable."
            )
        # ...
```

**After:**
```python
# src/services/llm_service.py

from src.utils.exceptions import ConfigurationError # Add this import

class EnhancedLLMService:
    def __init__(self, user_api_key: Optional[str] = None):
        # ...
        api_key = self.user_api_key or self.primary_api_key or self.fallback_api_key
        if not api_key:
            # Fail fast with a clear, actionable error message.
            raise ConfigurationError(
                "CRITICAL: Gemini API key is not configured. "
                "Please set the GEMINI_API_KEY in your .env file or provide it in the UI. "
                "Application cannot start without a valid API key."
            )
        # ...
```

#### 1.3.3. Remove Silent Error Handling in `get_llm_service`

Modify the singleton factory to let the `ConfigurationError` propagate, ensuring the application halts immediately on misconfiguration.

-   **File:** `src/services/llm_service.py`
-   **Function:** `get_llm_service`

**Before:**
```python
# src/services/llm_service.py

def get_llm_service(user_api_key: Optional[str] = None) -> Optional[EnhancedLLMService]:
    """Get the global LLM service instance (singleton)."""
    global _llm_service_instance
    if _llm_service_instance is None:
        try:
            _llm_service_instance = EnhancedLLMService(user_api_key=user_api_key)
            logger.info("LLM service singleton created.")
        except Exception as e:
            logger.error(
                f"Failed to initialize EnhancedLLMService: {e}", exc_info=True
            )
            return None # This is the root cause of the NoneType bug
    return _llm_service_instance
```

**After:**```python
# src/services/llm_service.py

def get_llm_service(user_api_key: Optional[str] = None) -> EnhancedLLMService:
    """Get the global LLM service instance (singleton)."""
    global _llm_service_instance
    if _llm_service_instance is None:
        # The constructor now raises ConfigurationError on failure.
        # This critical error should halt the application, so no try/except is needed.
        _llm_service_instance = EnhancedLLMService(user_api_key=user_api_key)
        logger.info("LLM service singleton created.")
    # Update the user_api_key if a new one is provided on subsequent calls
    elif user_api_key and _llm_service_instance.user_api_key != user_api_key:
        _llm_service_instance = EnhancedLLMService(user_api_key=user_api_key)
        logger.info("LLM service re-initialized with new user-provided API key.")

    return _llm_service_instance
```

#### 1.3.4. Update UI to Handle `ConfigurationError` Gracefully

The Streamlit UI must catch the `ConfigurationError` and display a helpful message to the user.

-   **File:** `app.py`
-   **Logic:** Wrap the service initialization in a `try...except` block. Add an API key input field to the UI.

```python
# app.py

import streamlit as st
from src.utils.exceptions import ConfigurationError
from src.services.llm_service import get_llm_service
from src.frontend.ui_components import render_main_ui

def main():
    st.set_page_config(page_title="AI CV Generator", layout="wide")
    st.title("AI CV Generator")

    # Add API Key input to sidebar
    api_key_input = st.sidebar.text_input(
        "Gemini API Key", type="password", help="Your API key is used only for this session."
    )

    try:
        # Initialize service. This will now fail fast if no key is found
        # in .env and the user hasn't provided one.
        get_llm_service(user_api_key=api_key_input)

        # If initialization is successful, render the rest of the UI
        render_main_ui()

    except ConfigurationError as e:
        st.error(f"**Configuration Error:**\n\n{e}")
        st.warning("Please provide your Gemini API key in the sidebar to proceed.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.stop()

if __name__ == "__main__":
    main()
```

### 1.4. Implementation Checklist

1.  [ ] Create `src/utils/exceptions.py` and define the `ConfigurationError` class.
2.  [ ] Update `src/services/llm_service.py` to import `ConfigurationError`.
3.  [ ] Modify `EnhancedLLMService.__init__` to raise `ConfigurationError` instead of `ValueError`.
4.  [ ] Refactor `get_llm_service` to remove the `try...except` block and update its docstring/type hints.
5.  [ ] Update `app.py` to include a sidebar input for the API key.
6.  [ ] Wrap the service initialization and UI rendering in `app.py` with a `try...except ConfigurationError` block to display a user-friendly error.

### 1.5. Testing Strategy

-   **Unit Test:** Create `tests/unit/test_llm_service.py`.
    -   Write a test that unsets the `GEMINI_API_KEY` environment variable.
    -   Assert that calling `EnhancedLLMService()` raises `pytest.raises(ConfigurationError)`.
-   **Integration Test:**
    -   Manually run `streamlit run app.py` without a `.env` file or a configured API key.
    -   **Expected:** The Streamlit app loads and displays only the `st.error` message about the missing API key.
    -   **Not Expected:** The app shows a blank screen or a terminal traceback with `TypeError`.
-   **E2E Test:**
    -   Run the app, provide a valid API key in the UI, and execute a full CV generation workflow to ensure the fix did not introduce regressions.

---

## 2. Orchestration: Enforce Agent-State Data Contracts (P2)

### 2.1. Root Cause & Priority

-   **ID:** `CB-01`, `CB-02`, `CB-03`, `CB-04`, `CB-05`
-   **Priority:** P2 (High)
-   **Cause:** Agent `run_as_node` methods return dictionaries with keys that do not match the field names in the `AgentState` Pydantic model (`src/orchestration/state.py`). This causes LangGraph to either lose data or write it to the wrong state attribute, leading to downstream `NoneType` errors when other nodes access the state.

### 2.2. Impacted Modules

-   `src/agents/parser_agent.py`
-   `src/agents/research_agent.py`
-   `src/agents/quality_assurance_agent.py`
-   `src/orchestration/cv_workflow_graph.py`
-   `src/orchestration/state.py` (as the source of truth)

### 2.3. Implementation Plan

#### 2.3.1. Align `ParserAgent` Output with `AgentState`

The agent returns `structured_cv` and `job_description_data`, which is correct. The issue is that its formal `AgentIO` schema is incorrect and misleading. The primary action is to ensure the node keys in the graph match the `AgentState` fields.

-   **File:** `src/agents/parser_agent.py`
-   **Method:** `run_as_node`
-   **Action:** Ensure the returned dictionary keys are identical to `AgentState` field names.

**Before (Conceptual):**
```python
# src/agents/parser_agent.py
class ParserAgent(EnhancedAgentBase):
    async def run_as_node(self, state: AgentState) -> dict:
        # ... logic to parse cv and job description ...
        # Incorrect key "parsed_data" might be used, or schema is wrong.
        return {"parsed_data": {"cv": structured_cv, "job": job_description_data}}
```

**After (Enforced):**
```python
# src/agents/parser_agent.py
class ParserAgent(EnhancedAgentBase):
    async def run_as_node(self, state: AgentState) -> dict:
        # ... logic to parse cv and job description ...
        structured_cv = await self.parse_cv_text(state.cv_text)
        job_description_data = await self.parse_job_description(state.raw_job_description)

        # Return a dictionary with keys that EXACTLY match AgentState fields
        return {
            "structured_cv": structured_cv,
            "job_description_data": job_description_data
        }
```

#### 2.3.2. Align Other Agents (Research, QA)

Apply the same principle to all other agents identified in the audit.

-   **ResearchAgent (`CB-03`):**
    -   **File:** `src/agents/research_agent.py`
    -   **Action:** Modify `run_as_node` to return `{"research_findings": ...}` instead of `{"research_results": ...}`. Ensure it also returns `content_matches` if that logic exists, or remove it from the contract.
-   **QualityAssuranceAgent (`CB-04`):**
    -   **File:** `src/agents/quality_assurance_agent.py`
    -   **Action:** Modify `run_as_node` to return `{"quality_check_results": ...}` instead of `{"cv_analysis_results": ...}`.

### 2.4. Implementation Checklist

1.  [ ] Review `src/orchestration/state.py` to confirm the exact field names of `AgentState`.
2.  [ ] In `src/agents/parser_agent.py`, verify `run_as_node` returns a dictionary with keys `"structured_cv"` and `"job_description_data"`.
3.  [ ] In `src/agents/research_agent.py`, modify `run_as_node` to return a dictionary with the key `"research_findings"`.
4.  [ ] In `src/agents/quality_assurance_agent.py`, modify `run_as_node` to return a dictionary with the key `"quality_check_results"`.
5.  [ ] For each agent, update its `AgentIO` Pydantic schema in `src/models/data_models.py` to accurately reflect the true inputs and outputs, making them reliable contracts again.

### 2.5. Testing Strategy

-   **Integration Test:** Enhance `tests/integration/test_agent_workflow_integration.py`.
    -   Create a test that runs the entire `cv_workflow_graph` from start to finish.
    -   After the graph execution, inspect the final `AgentState` object.
    -   Assert that fields like `state.research_findings` and `state.quality_check_results` are not `None` and contain the expected data structures. This directly verifies that data is no longer being lost between nodes.

---

## Critical Gaps & Questions

1.  **API Key Persistence:** The proposed UI solution with `st.sidebar.text_input` is session-based. Should the application offer to save the user-provided API key to the `.env` file for future sessions? This has security implications and should be clarified.
2.  **Global Configuration Validation:** The fail-fast pattern is being applied to the LLM service. A follow-up task should be created to audit other components (e.g., vector store connections, external API clients) for similar configuration dependencies and apply this robust error-handling pattern globally.
3.  **AgentIO Schema Enforcement:** Simply updating the schemas (`AgentIO`) is a good first step. A future P3 task could involve writing a meta-test that programmatically inspects each agent's `run_as_node` signature and compares its return annotation to its `AgentIO.output_schema` to prevent future contract drift.

---

---

## 3. Service Layer: Fail-Fast Vector Store Initialization (P1)

### 3.1. Root Cause & Priority

-   **ID:** `ERR-04` (New)
-   **Priority:** P1 (Critical Blocker)
-   **Cause:** Based on the "Config Validation" directive, other critical services like the vector store must also adopt the fail-fast pattern. A failure to connect to ChromaDB (e.g., invalid host, non-existent persistent path) could allow a `None` client to be injected into agents, leading to runtime errors during retrieval operations.

### 3.2. Impacted Modules

-   `src/services/vector_store_service.py` (Assumed module)
-   `src/utils/exceptions.py` (Used)
-   `app.py`

### 3.3. Implementation Plan

#### 3.3.1. Refactor `VectorStoreService` to Connect on Init

The service constructor should eagerly attempt to connect to the ChromaDB backend and raise a `ConfigurationError` on failure.

-   **File:** `src/services/vector_store_service.py` (or equivalent)
-   **Class:** `VectorStoreService`
-   **Method:** `__init__`

**Before (Conceptual):**
```python
# src/services/vector_store_service.py
import chromadb

class VectorStoreService:
    def __init__(self, settings):
        self.settings = settings
        self.client = None # Lazy initialization

    def get_client(self):
        if self.client is None:
            # Connection happens on first use, hiding startup errors
            self.client = chromadb.PersistentClient(path=self.settings.db.path)
        return self.client
```

**After (Enforced):**
```python
# src/services/vector_store_service.py
import chromadb
from src.utils.exceptions import ConfigurationError

class VectorStoreService:
    def __init__(self, settings):
        self.settings = settings
        self.client = self._connect()

    def _connect(self) -> chromadb.Client:
        """Eagerly connect to ChromaDB and fail fast."""
        try:
            # Attempt to connect immediately during initialization.
            client = chromadb.PersistentClient(path=self.settings.db.path)
            # Ping the server to ensure the connection is live.
            client.heartbeat()
            logger.info("Successfully connected to ChromaDB at %s", self.settings.db.path)
            return client
        except Exception as e:
            # Any exception during connection is a critical configuration error.
            raise ConfigurationError(
                f"CRITICAL: Failed to connect to Vector Store (ChromaDB) at "
                f"path '{self.settings.db.path}'. Please check the path and permissions. "
                f"Error: {e}"
            ) from e

    def get_client(self) -> chromadb.Client:
        return self.client
```

#### 3.3.2. Implement a Fail-Fast Singleton Getter

Create a singleton factory for the `VectorStoreService` that allows the `ConfigurationError` to propagate, halting the application.

-   **File:** `src/services/vector_store_service.py`
-   **Function:** `get_vector_store_service`

```python
# src/services/vector_store_service.py

_vector_store_instance = None

def get_vector_store_service() -> VectorStoreService:
    """Get the global VectorStoreService instance (singleton)."""
    global _vector_store_instance
    if _vector_store_instance is None:
        # This will raise ConfigurationError if connection fails.
        _vector_store_instance = VectorStoreService(settings=get_config())
    return _vector_store_instance
```

#### 3.3.3. Update UI to Validate All Critical Services

Modify the main application entry point to initialize and validate all critical services at startup.

-   **File:** `app.py`
-   **Logic:** Add the vector store initialization to the main `try...except` block.

**Before:**
```python
# app.py
try:
    get_llm_service(user_api_key=api_key_input)
    render_main_ui()
except ConfigurationError as e:
    st.error(f"**Configuration Error:**\n\n{e}")
```

**After:**
```python
# app.py
import streamlit as st
from src.utils.exceptions import ConfigurationError
from src.services.llm_service import get_llm_service
from src.services.vector_store_service import get_vector_store_service
from src.frontend.ui_components import render_main_ui

def main():
    # ... (st.set_page_config, title, api_key_input) ...

    try:
        # Initialize all critical services at startup.
        # Each will raise ConfigurationError on failure.
        get_llm_service(user_api_key=api_key_input)
        get_vector_store_service()

        # If all services initialize successfully, render the UI.
        render_main_ui()

    except ConfigurationError as e:
        # A single catch block handles all critical startup failures.
        st.error(f"**Application Startup Failed:**\n\n{e}")
        st.warning("Please check your configuration (.env file, database paths) and restart.")
        st.stop()
    # ... (other exception handling) ...

if __name__ == "__main__":
    main()
```

### 3.4. Implementation Checklist

1.  [ ] Create `src/services/vector_store_service.py` if it does not exist.
2.  [ ] Implement the `VectorStoreService` class with an eager `_connect` method in its `__init__`.
3.  [ ] Ensure the `_connect` method raises `ConfigurationError` on any connection failure.
4.  [ ] Implement the `get_vector_store_service` singleton factory.
5.  [ ] Update `app.py` to call `get_vector_store_service()` within the main startup `try...except` block.

### 3.5. Testing Strategy

-   **Unit Test:** Create `tests/unit/test_vector_store_service.py`.
    -   Use `unittest.mock` to patch `chromadb.PersistentClient`.
    -   Configure the mock to raise an exception (e.g., `ValueError`, `FileNotFoundError`).
    -   Assert that `VectorStoreService()` raises `pytest.raises(ConfigurationError)`.
-   **Integration Test:**
    -   Run `streamlit run app.py` with an invalid ChromaDB path in the configuration.
    -   **Expected:** The app loads and displays the `st.error` message about the vector store connection failure.
    -   **Not Expected:** The app crashes with an unhandled exception later in the workflow.

---

---

## 4. Code Quality: Static Analysis Error Remediation (P1)

### 4.1. Root Cause & Priority

-   **ID:** `ERR-02`, `ERR-03`
-   **Priority:** P1 (Critical Blocker)
-   **Cause:** The codebase contains Pylint errors of type `import-error` and `no-member`. These are not stylistic warnings but indicators of code that is guaranteed to crash at runtime. Their presence suggests gaps in the local development setup or CI/CD quality gates.

### 4.2. Impacted Modules

-   `src/agents/specialized_agents.py`
-   `src/agents/formatter_agent.py`
-   (Other modules as identified by a full `pylint --errors-only` scan)

### 4.3. Implementation Plan

#### 4.3.1. Fix `E0401: import-error`

This fatal error prevents modules from being loaded, causing the application to fail at startup.

-   **File:** `src/agents/specialized_agents.py`
-   **Action:** Correct the invalid import path for agent exceptions.

**Before:**
```python
# src/agents/specialized_agents.py
# This path is incorrect and will fail.
from aicvgen.src.exceptions.agent_exceptions import ContentGenerationError
```

**After:**
```python
# src/agents/specialized_agents.py
# Assuming the custom exception is defined in the centralized exceptions module.
from src.utils.exceptions import AgentError # Or a more specific exception if defined
```

#### 4.3.2. Fix `E1101: no-member`

This error occurs when accessing an attribute on an object that does not have it (e.g., accessing a `dict` with dot notation like an object). This results in a fatal `AttributeError` at runtime.

-   **File:** `src/agents/formatter_agent.py`
-   **Action:** Use dictionary-safe access methods (`.get()`) instead of attribute access on dictionaries.

**Before:**
```python
# src/agents/formatter_agent.py
def some_method(self, result: dict):
    if result.error_message: # This will raise AttributeError if result is a dict
        logger.error(result.error_message)
```

**After:**
```python
# src/agents/formatter_agent.py
def some_method(self, result: dict):
    error_message = result.get("error_message") # Use .get() for safe access
    if error_message:
        logger.error(error_message)
```

### 4.4. Implementation Checklist

1.  [ ] Run `pylint src --errors-only` from the project root to get a complete list of all critical errors.
2.  [ ] In `src/agents/specialized_agents.py`, correct the `import-error` to point to the valid exceptions module (likely `src/utils/exceptions.py`).
3.  [ ] In `src/agents/formatter_agent.py` and any other file with a `no-member` error on a dictionary, change attribute access (`my_dict.key`) to dictionary access (`my_dict['key']` or `my_dict.get('key')`).
4.  [ ] Re-run `pylint src --errors-only` to confirm that no critical errors remain.

### 4.5. Testing Strategy

-   **Static Analysis:** The primary verification is a clean Pylint scan. This should be integrated into the CI pipeline as a mandatory check.
-   **CI Gate:** Add a step to the CI workflow (e.g., GitHub Actions) that executes `pylint src --errors-only --fail-on=E`. This will automatically prevent new critical static analysis errors from being merged.
-   **Smoke Test:** After fixing the errors, run the application and execute a basic workflow to ensure the fixes work as expected and have not introduced regressions.

---

## 5. Architectural Refactor: Decompose Sub-Orchestrator (P3)

### 5.1. Root Cause & Priority

-   **ID:** `AD-01`
-   **Priority:** P3 (Architectural Debt)
-   **Cause:** The `ContentOptimizationAgent` acts as a "sub-orchestrator" by internally instantiating and calling other agents (`EnhancedContentWriterAgent`). This hides workflow logic, bypasses the main orchestrator's control and observability, and makes the system difficult to debug and manage. This task is deferred post-MVP stabilization but is documented here as a critical next step.

### 5.2. Impacted Modules

-   `src/agents/specialized_agents.py` (To be simplified/removed)
-   `src/orchestration/cv_workflow_graph.py` (To be expanded)
-   `src/agents/enhanced_content_writer.py` (To be used as a standard node)

### 5.3. Implementation Plan

#### 5.3.1. Expose Iteration Logic to the Main Graph

The core of the refactor is to move the looping logic from the agent into the LangGraph definition itself.

-   **File:** `src/orchestration/cv_workflow_graph.py`
-   **Action:** Add nodes and conditional edges to manage the iteration over CV sections that require content generation.

**Before (Conceptual Anti-Pattern):**
```python
# src/orchestration/cv_workflow_graph.py
workflow.add_node("research_node", research_agent.run_as_node)
# A single, opaque node that hides a complex internal loop
workflow.add_node("content_optimization_node", content_optimization_agent.run_as_node)
workflow.add_node("qa_node", qa_agent.run_as_node)

workflow.add_edge("research_node", "content_optimization_node")
workflow.add_edge("content_optimization_node", "qa_node")
```

**After (Refactored Graph):**
```python
# src/orchestration/cv_workflow_graph.py

# A new state field to track the iteration
class AgentState(TypedDict):
    # ... other fields
    sections_to_write: list[str]
    written_sections: list[dict]

# A simple content writer node
workflow.add_node("content_writer_node", enhanced_content_writer.run_as_node)

# A conditional router to manage the loop
def should_continue_writing(state: AgentState) -> str:
    """Determines if there are more sections to write."""
    if state.get("sections_to_write") and len(state["sections_to_write"]) > 0:
        return "continue_writing"
    return "writing_complete"

# The graph manages the iteration explicitly
workflow.add_node("check_writing_status_node", should_continue_writing)

# Define the explicit workflow loop
workflow.add_edge("research_node", "check_writing_status_node")
workflow.add_conditional_edges(
    "check_writing_status_node",
    should_continue_writing,
    {
        "continue_writing": "content_writer_node",
        "writing_complete": "qa_node", # Move to QA after all writing is done
    },
)
# After writing a section, check again if more sections are left
workflow.add_edge("content_writer_node", "check_writing_status_node")
```

### 5.4. Implementation Checklist

1.  [ ] Add `sections_to_write: list[str]` and `written_sections: list[dict]` to the `AgentState` model in `src/orchestration/state.py`.
2.  [ ] Create a node or function (e.g., `prepare_writing_tasks`) that populates `sections_to_write` in the state after the parsing/research phase.
3.  [ ] In `cv_workflow_graph.py`, add the `content_writer_node` using the existing `EnhancedContentWriterAgent`.
4.  [ ] Implement the conditional router function `should_continue_writing`.
5.  [ ] Add the conditional router node to the graph.
6.  [ ] Rewire the graph edges to create the explicit writing loop as described in the "After" snippet.
7.  [ ] Deprecate or remove the `ContentOptimizationAgent` from `src/agents/specialized_agents.py`, as its logic is now in the graph.

### 5.5. Testing Strategy

-   **Integration Test:** The primary test for this refactor.
    -   Run the full workflow integration test.
    -   **Expected:** The final generated CV should be identical to the one produced before the refactor.
    -   **Verification:** Inspect the LangGraph logs or traces. You should now see multiple calls to `content_writer_node` and `check_writing_status_node`, providing full visibility into the content generation process. A failure in writing the "Executive Summary" will now clearly be attributed to a specific run of the `content_writer_node`.

---

---

## 6. Code Quality: Centralize JSON Parsing Logic (P3)

### 6.1. Root Cause & Priority

-   **ID:** `DUP-01`
-   **Priority:** P3 (Maintainability)
-   **Cause:** Multiple agents (`ParserAgent`, `ResearchAgent`, `EnhancedContentWriterAgent`) implement bespoke, fragile, and duplicated logic for extracting JSON from raw LLM text responses. This violates the DRY (Don't Repeat Yourself) principle and creates a high maintenance burden. A single bug fix or enhancement to the parsing logic must be manually replicated across the codebase.

### 6.2. Impacted Modules

-   `src/agents/agent_base.py` (Location of the centralized utility)
-   `src/agents/parser_agent.py` (To be refactored)
-   `src/agents/research_agent.py` (To be refactored)
-   `src/agents/enhanced_content_writer.py` (To be refactored)

### 6.3. Implementation Plan

#### 6.3.1. Enhance the `_generate_and_parse_json` Utility

The `EnhancedAgentBase` class should provide a single, robust utility for all agents to use. This method will handle LLM calls and reliably extract JSON, even from responses containing markdown code fences or conversational boilerplate.

-   **File:** `src/agents/agent_base.py`
-   **Class:** `EnhancedAgentBase`
-   **Method:** `_generate_and_parse_json`

```python
# src/agents/agent_base.py
import re
import json
from typing import Any, Dict

class EnhancedAgentBase:
    # ... existing methods ...

    async def _generate_and_parse_json(
        self, prompt: str, session_id: str, trace_id: str
    ) -> Dict[str, Any]:
        """
        Generates content from the LLM and robustly parses the JSON output.
        Handles common LLM response formats like markdown code blocks.
        """
        llm_response = await self.llm_service.generate_content(
            prompt=prompt, session_id=session_id, trace_id=trace_id
        )

        if not llm_response.success:
            raise AgentError(f"LLM generation failed: {llm_response.error_message}")

        raw_text = llm_response.content

        # Regex to find JSON within ```json ... ``` code blocks
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback for raw JSON or JSON embedded in text
            start_index = raw_text.find('{')
            end_index = raw_text.rfind('}') + 1
            if start_index == -1 or end_index == 0:
                raise ValueError("No valid JSON object found in the LLM response.")
            json_str = raw_text[start_index:end_index]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON from LLM response: %s", e)
            logger.debug("Malformed JSON string: %s", json_str)
            raise ValueError(f"Could not parse JSON from LLM response. Error: {e}")
```

#### 6.3.2. Refactor Agents to Use the Centralized Utility

Update all agents that currently perform manual JSON parsing to call the new base class method.

-   **File:** `src/agents/parser_agent.py`
-   **Method:** `_parse_job_description_with_llm` (and similar methods)

**Before:**
```python
# src/agents/parser_agent.py
async def _parse_job_description_with_llm(self, job_text: str) -> dict:
    # ...
    response = await self.llm_service.generate_content(...)
    content = response.content
    # Manual, fragile parsing
    start = content.find('{')
    end = content.rfind('}') + 1
    json_data = json.loads(content[start:end])
    return json_data
```

**After:**
```python
# src/agents/parser_agent.py
async def _parse_job_description_with_llm(self, job_text: str, trace_id: str) -> dict:
    # ...
    prompt = self._create_job_description_prompt(job_text)

    # Delegate to the robust, centralized utility
    parsed_json = await self._generate_and_parse_json(
        prompt=prompt, session_id="parser_session", trace_id=trace_id
    )
    return parsed_json
```

### 6.4. Implementation Checklist

1.  [ ] Implement or enhance the `_generate_and_parse_json` method in `src/agents/agent_base.py` as specified.
2.  [ ] Refactor `ParserAgent` to remove all manual JSON parsing and use `self._generate_and_parse_json`.
3.  [ ] Refactor `ResearchAgent` to remove all manual JSON parsing and use `self._generate_and_parse_json`.
4.  [ ] Refactor `EnhancedContentWriterAgent` to remove its `_extract_json_from_response` helper and use `self._generate_and_parse_json`.
5.  [ ] Review all other agents for any remaining manual JSON parsing logic and refactor them.

### 6.5. Testing Strategy

-   **Unit Test:** Create `tests/unit/test_agent_base.py`.
    -   Write specific tests for `_generate_and_parse_json`.
    -   Mock the `llm_service.generate_content` call to return various raw text formats:
        -   A string containing only a JSON object.
        -   A string with a JSON object wrapped in ```json ... ```.
        -   A string with conversational text before and after a JSON object.
        -   A string with malformed JSON (to test error handling).
    -   Assert that the method correctly extracts and parses the JSON in each case or raises the appropriate `ValueError`.
-   **Regression Testing:** The existing integration tests for the refactored agents must continue to pass. This ensures that their core functionality remains intact after the internal implementation change.

---

## 7. Code Pruning: Remove Deprecated and Redundant Code (P3)

### 7.1. Root Cause & Priority

-   **ID:** `DL-01`, `DL-02`
-   **Priority:** P3 (Maintainability)
-   **Cause:** The codebase is cluttered with dead code, including obsolete import-fixing scripts (`fix_*.py`) and a non-functional, traditional HTML/JS frontend. This code increases cognitive load for developers, inflates the codebase, and creates confusion about the project's true architecture.

### 7.2. Impacted Modules

-   Project root (`/`)
-   `src/frontend/templates/`
-   `src/frontend/static/`

### 7.3. Implementation Plan

This task involves the direct removal of unused files and directories.

#### 7.3.1. Remove Obsolete `fix_*.py` Scripts

These scripts were one-time utilities for a past refactoring and serve no ongoing purpose.

-   **Location:** Project root directory (`/`)
-   **Action:** Delete the following files.

```bash
git rm fix_agents_imports.py
git rm fix_all_imports.py
git rm fix_imports.py
```

#### 7.3.2. Remove Redundant Traditional Web Frontend

The application's functional UI is built with Streamlit (`app.py`). The legacy HTML, CSS, and JavaScript files are unused and should be removed.

-   **Location:** `src/frontend/`
-   **Action:** Delete the following directories and their contents.

```bash
git rm -r src/frontend/templates/
git rm -r src/frontend/static/
```

### 7.4. Implementation Checklist

1.  [ ] Execute `git rm` to delete `fix_agents_imports.py`.
2.  [ ] Execute `git rm` to delete `fix_all_imports.py`.
3.  [ ] Execute `git rm` to delete `fix_imports.py`.
4.  [ ] Confirm that the traditional web frontend is not used by any active part of the application.
5.  [ ] Execute `git rm -r` to delete the `src/frontend/templates` directory.
6.  [ ] Execute `git rm -r` to delete the `src/frontend/static` directory.
7.  [ ] Commit the deletions with a clear message (e.g., "refactor: Remove deprecated fix scripts and legacy frontend").

### 7.5. Testing Strategy

-   **Smoke Test:** After deleting the files, start the Streamlit application (`streamlit run app.py`) and run a full end-to-end workflow.
-   **Regression Testing:** Execute the entire test suite (`pytest`).
-   **Expected:** The application and all tests should function identically to before the file deletions. This confirms that the removed code was indeed dead and not a hidden dependency.


---

---

## 8. Code Quality: Refactor Complex `parse_cv_text` Function (P3)

### 8.1. Root Cause & Priority

-   **ID:** `SMELL-01`
-   **Priority:** P3 (Maintainability)
-   **Cause:** The `parse_cv_text` method in `src/agents/parser_agent.py` is a "Long Method" code smell. It has too many responsibilities, including extracting contact information, identifying sections, parsing subsections, and handling bullet points. This high complexity makes the function difficult to read, test, and safely modify, increasing the risk of introducing regressions.

### 8.2. Impacted Modules

-   `src/agents/parser_agent.py`
-   `tests/unit/test_parser_agent.py` (New tests required)

### 8.3. Implementation Plan

The plan is to decompose the monolithic `parse_cv_text` method into a set of smaller, single-responsibility private helper methods. The main method will become a simple orchestrator that calls these helpers.

#### 8.3.1. Decompose Logic into Private Helper Methods

-   **File:** `src/agents/parser_agent.py`
-   **Class:** `ParserAgent`
-   **Action:** Create new private methods for each distinct parsing task.

**Before (Conceptual):**
```python
# src/agents/parser_agent.py
class ParserAgent(EnhancedAgentBase):
    async def parse_cv_text(self, cv_text: str) -> StructuredCV:
        # 1. Regex logic to find name, email, phone...
        # ... many lines of code ...

        # 2. Regex logic to identify top-level sections...
        # ... many lines of code ...

        # 3. Logic to loop through sections and parse bullet points...
        # ... many lines of code ...

        # 4. Assemble the entire StructuredCV object here...
        # ... many lines of code ...

        return structured_cv
```

**After (Refactored Structure):**
```python
# src/agents/parser_agent.py
class ParserAgent(EnhancedAgentBase):

    def _parse_contact_info(self, cv_text: str) -> PersonalInfo:
        """Extracts name, email, phone, etc., from the CV text."""
        # ... focused logic for contact info parsing ...
        return PersonalInfo(...)

    def _parse_sections(self, cv_text: str) -> list[dict]:
        """Splits the CV text into logical sections based on headers."""
        # ... focused logic for section splitting ...
        return sections_data

    def _parse_section_content(self, section_text: str) -> list[str]:
        """Parses bullet points or content from a single section's text."""
        # ... focused logic for parsing bullet points ...
        return bullet_points

    async def parse_cv_text(self, cv_text: str) -> StructuredCV:
        """
        Orchestrates the parsing of raw CV text into a StructuredCV object.
        """
        # 1. Call helper to parse contact info
        personal_info = self._parse_contact_info(cv_text)

        # 2. Call helper to split text into sections
        raw_sections = self._parse_sections(cv_text)

        # 3. Loop and call helpers to process each section
        parsed_sections = []
        for raw_section in raw_sections:
            content = self._parse_section_content(raw_section['text'])
            # Potentially enhance with LLM if needed
            # enhanced_content = await self._enhance_section_with_llm(content)
            parsed_sections.append(
                Section(title=raw_section['title'], items=content)
            )

        # 4. Assemble the final object
        structured_cv = StructuredCV(
            personal_info=personal_info,
            sections=parsed_sections
        )

        return structured_cv
```

### 8.4. Implementation Checklist

1.  [ ] In `src/agents/parser_agent.py`, identify the distinct logical blocks within the original `parse_cv_text` method.
2.  [ ] Create a new private method `_parse_contact_info` and move the contact info parsing logic into it.
3.  [ ] Create a new private method `_parse_sections` and move the section identification/splitting logic into it.
4.  [ ] Create a new private method `_parse_section_content` and move the bullet point parsing logic into it.
5.  [ ] Refactor the main `parse_cv_text` method to be a clean orchestrator that calls the new helper methods in the correct sequence.
6.  [ ] Ensure the final `StructuredCV` object is correctly assembled from the outputs of the helper methods.

### 8.5. Testing Strategy

-   **Unit Tests:** This refactor requires new, focused unit tests.
    -   In `tests/unit/test_parser_agent.py`, add a new test class `TestParserAgentHelpers`.
    -   Write a specific test for `_parse_contact_info` with sample CV headers and assert that the correct `PersonalInfo` object is returned.
    -   Write a specific test for `_parse_sections` with a multi-section CV string and assert that the sections are split correctly with the right titles.
    -   Write a specific test for `_parse_section_content` with sample section text and assert that the bullet points are correctly extracted.
-   **Regression Testing:** The existing integration test for `ParserAgent.run_as_node` must continue to pass. This ensures the overall functionality of the agent is unchanged despite the internal refactoring.

---

---

## 9. Service Layer: Centralize Service Resilience Patterns (P3)

### 9.1. Root Cause & Priority

-   **ID:** `SMELL-02` (New, related to Service Layer Cohesion)
-   **Priority:** P3 (Architectural Debt)
-   **Cause:** Resilience logic (retries, rate limiting) is duplicated across multiple services, notably `EnhancedLLMService` and `ItemProcessor`. This layered retry approach leads to unpredictable delays, masks the true source of failures, and can cause a "thundering herd" problem against external APIs. The Single Responsibility Principle is violated.

### 9.2. Impacted Modules

-   `src/services/llm_service.py` (Becomes the single source of truth for LLM resilience)
-   `src/services/item_processor.py` (To be simplified)
-   `src/agents/*` (Any agent using `ItemProcessor`)

### 9.3. Implementation Plan

The plan is to make `EnhancedLLMService` solely responsible for handling all resilience related to the Gemini API. Other services like `ItemProcessor` will be simplified to trust that the LLM service is already resilient.

#### 9.3.1. Enhance `EnhancedLLMService` with Tenacity

Ensure the LLM service uses the `tenacity` library to handle transient errors like rate limiting or temporary network issues. Fatal errors like `ConfigurationError` should not be retried.

-   **File:** `src/services/llm_service.py`
-   **Class:** `EnhancedLLMService`
-   **Method:** `generate_content`

```python
# src/services/llm_service.py
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions as google_exceptions
from src.utils.exceptions import ConfigurationError

# Define what errors are transient and should be retried
def is_transient_error(e: Exception) -> bool:
    """Return True if the exception is a transient network or rate limit error."""
    return isinstance(e, (
        google_exceptions.ResourceExhausted, # Rate limit
        google_exceptions.ServiceUnavailable, # Temporary server issue
        TimeoutError
    ))

class EnhancedLLMService:
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(Exception), # Retry on general exceptions...
        retry_error_callback=lambda retry_state: retry_state.outcome.result() # Return last result on failure
    )
    async def generate_content(self, prompt: str, ...) -> LLMResponse:
        """
        Generates content using the Gemini model. Retries are handled by tenacity.
        It will NOT retry on non-transient errors like ConfigurationError.
        """
        # The try-except block inside this method should be for logging or
        # wrapping exceptions, not for retrying.
        try:
            # ... (logic to call _make_llm_api_call via a timeout wrapper) ...
            # The core API call logic remains the same.
            response = await self._generate_with_timeout(...)
            return LLMResponse(success=True, content=response.text, ...)
        except ConfigurationError:
            # Do not retry on fatal config errors. Re-raise immediately.
            raise
        except Exception as e:
            if not is_transient_error(e):
                # If it's not a transient error, stop retrying by raising it.
                # Or handle it and return a failed LLMResponse.
                logger.error("Non-retriable error during LLM call: %s", e)
                return LLMResponse(success=False, error_message=str(e))
            # For transient errors, re-raise to let tenacity handle the retry.
            raise e
```

#### 9.3.2. Simplify `ItemProcessor`

Remove all retry and rate-limiting logic from `ItemProcessor`. It should now assume that any call to `llm_service.generate_content` has already been handled with appropriate resilience.

-   **File:** `src/services/item_processor.py`
-   **Class:** `ItemProcessor`
-   **Method:** `process_item`

**Before (Conceptual):**
```python
# src/services/item_processor.py
class ItemProcessor:
    async def process_item(self, item):
        for i in range(MAX_RETRIES): # Internal retry loop
            try:
                response = await self.llm_service.generate_content(...)
                if response.success:
                    return response
            except Exception:
                await asyncio.sleep(i * 2) # Internal backoff
        return None # Failed after retries
```

**After (Simplified):**
```python
# src/services/item_processor.py
class ItemProcessor:
    async def process_item(self, item):
        # No more retry loop. Trust the LLM service to be resilient.
        prompt = self._create_prompt_for_item(item)
        response = await self.llm_service.generate_content(prompt=prompt)

        if not response.success:
            logger.error("Item processing failed for item '%s' after LLM service retries.", item)
            # Handle the final failure, but do not retry here.
            return None

        return response.content
```

### 9.4. Implementation Checklist

1.  [ ] Review and implement the `@retry` decorator from `tenacity` on `EnhancedLLMService.generate_content`.
2.  [ ] Define a helper function or logic within the service to distinguish between transient errors (like rate limits) and fatal errors (like invalid arguments or auth failure).
3.  [ ] Ensure that fatal errors are not retried.
4.  [ ] Go to `src/services/item_processor.py` and remove any custom `try...except` blocks, `for` loops, or `asyncio.sleep` calls that are intended for retrying LLM calls.
5.  [ ] Refactor `ItemProcessor` to make a single call to `llm_service.generate_content` and handle the final success or failure state.

### 9.5. Testing Strategy

-   **Unit Test:** In `tests/unit/test_llm_service.py`:
    -   Mock the underlying `_make_llm_api_call` method.
    -   Configure the mock to raise a transient error (e.g., `google_exceptions.ResourceExhausted`) multiple times before succeeding.
    -   Assert that `generate_content` is called multiple times (verifying the retry) and eventually returns a successful response.
    -   Configure the mock to raise a fatal error (e.g., `ConfigurationError`).
    -   Assert that `generate_content` is only called once and re-raises the fatal error.
-   **Integration Test:** In `tests/integration/test_item_processor.py`:
    -   Run a test that uses the real `ItemProcessor`.
    -   Mock the `EnhancedLLMService` to return a failed `LLMResponse`.
    -   Assert that the `ItemProcessor` correctly handles the final failure without attempting its own retries.

---

## 10. Consolidated Implementation Roadmap

This provides a step-by-step checklist for a developer to execute the tasks outlined in this blueprint in a logical order, prioritizing stability (P1), then correctness (P2), then maintainability (P3).

### Phase 1: Critical Stabilization (P1 Tasks)

*Goal: Make the application start reliably and eliminate all known crashers.*

1.  **Setup:** Run `pylint src --errors-only` to get a baseline of critical static analysis errors.
2.  **Fix Imports:** Correct the `E0401: import-error` in `src/agents/specialized_agents.py` (Task 4.3.1).
3.  **Fix Attribute Access:** Correct all `E1101: no-member` errors across the codebase, such as in `src/agents/formatter_agent.py` (Task 4.3.2).
4.  **Create Custom Exception:** Create `src/utils/exceptions.py` and define `ConfigurationError` (Task 1.3.1).
5.  **Implement Fail-Fast LLM Service:**
    -   Modify `EnhancedLLMService.__init__` to raise `ConfigurationError` on a missing API key (Task 1.3.2).
    -   Remove the silent `try...except` from the `get_llm_service` singleton factory (Task 1.3.3).
6.  **Implement Fail-Fast Vector Store Service:**
    -   Refactor `VectorStoreService` to connect on `__init__` and raise `ConfigurationError` on failure (Task 3.3.1).
    -   Implement the `get_vector_store_service` singleton (Task 3.3.2).
7.  **Update UI for Fail-Fast:**
    -   Add the API key input field to the Streamlit UI in `app.py` (Task 1.3.4).
    -   Wrap all service initializations (`get_llm_service`, `get_vector_store_service`) in a single `try...except ConfigurationError` block in `app.py` (Task 3.3.3).
8.  **Verification:** Run the app without configuration. It must display a clean error message in the UI and not crash. Run `pylint src --errors-only`; it must report no errors.

### Phase 2: Data Contract Enforcement (P2 Tasks)

*Goal: Ensure data flows predictably through the workflow, eliminating data loss between nodes.*

1.  **Align Agent Outputs:**
    -   Modify `ParserAgent.run_as_node` to ensure its return keys match `AgentState` fields (Task 2.3.1).
    -   Modify `ResearchAgent.run_as_node` to return `{"research_findings": ...}` (Task 2.3.2).
    -   Modify `QualityAssuranceAgent.run_as_node` to return `{"quality_check_results": ...}` (Task 2.3.2).
2.  **Update Agent Schemas:** For each agent modified above, update its corresponding `AgentIO` schema in `src/models/data_models.py` to match the actual inputs and outputs (Task 2.4, Step 5).
3.  **Verification:** Run the full end-to-end integration test. Inspect the final state and assert that no data fields are unexpectedly `None`.

### Phase 3: Architectural and Code Quality Refactoring (P3 Tasks)

*Goal: Pay down technical debt to improve long-term maintainability.*

1.  **Remove Dead Code:**
    -   Delete the `fix_*.py` scripts from the root directory (Task 7.3.1).
    -   Delete the `src/frontend/templates` and `src/frontend/static` directories (Task 7.3.2).
2.  **Centralize JSON Parsing:**
    -   Enhance the `_generate_and_parse_json` utility in `EnhancedAgentBase` (Task 6.3.1).
    -   Refactor `ParserAgent`, `ResearchAgent`, and other agents to use this centralized utility (Task 6.3.2).
3.  **Refactor Complex Function:**
    -   Decompose the `parse_cv_text` method in `ParserAgent` into smaller, single-responsibility helper methods (`_parse_contact_info`, etc.) (Task 8.3.1).
    -   Create new, focused unit tests for each new helper method (Task 8.5).
4.  **Centralize Resilience Logic:**
    -   Implement the `tenacity` `@retry` decorator on `EnhancedLLMService.generate_content` (Task 9.3.1).
    -   Simplify `ItemProcessor` by removing its internal retry logic (Task 9.3.2).
5.  **Decompose Sub-Orchestrator (Major Refactor):**
    -   Modify `AgentState` to support iteration (Task 5.4, Step 1).
    -   Implement the explicit writing loop in `cv_workflow_graph.py` using a conditional router (Task 5.3.1).
    -   Deprecate and remove the `ContentOptimizationAgent` (Task 5.4, Step 7).
6.  **Final Verification:** Run the entire test suite (`pytest`) and perform a full manual end-to-end test to ensure no regressions were introduced during the refactoring phase.

---

---

## 11. Process: Formalize CI Quality Gates (P2)

### 11.1. Root Cause & Priority

-   **ID:** `PROC-01` (New)
-   **Priority:** P2 (High)
-   **Cause:** The presence of P1-level static analysis errors and broken tests in the codebase indicates that automated quality gates are either missing, not configured correctly, or not being enforced. A robust CI pipeline is essential to prevent regressions and maintain codebase health.

### 11.2. Impacted Modules

-   This is a process and infrastructure task, primarily impacting the CI/CD configuration file.
-   **File:** `/.github/workflows/ci.yml` (or equivalent for other CI providers).

### 11.3. Implementation Plan

The plan is to create or enhance a CI workflow that automatically runs on every pull request and push to the main branch. This workflow will act as a non-negotiable quality gate.

#### 11.3.1. Define the CI Workflow for GitHub Actions

Create a YAML file that defines jobs for linting and testing. The workflow must be configured to fail the pull request check if any job fails.

-   **File:** `/.github/workflows/ci.yml`

```yaml
# /.github/workflows/ci.yml
name: Python Application CI

on:
  push:
    branches: [ "main", "develop" ]
  pull_request:
    branches: [ "main", "develop" ]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pylint
    - name: Run Pylint
      run: |
        # Fail the build if any Error (E) or Fatal (F) messages are found.
        pylint src --errors-only --fail-on=E,F

  test:
    runs-on: ubuntu-latest
    needs: lint # Run tests only if linting passes
    env:
      # Use GitHub Secrets to provide a test API key for integration tests
      GEMINI_API_KEY: ${{ secrets.TEST_GEMINI_API_KEY }}
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run Pytest
      run: |
        # Run all unit and integration tests.
        # --cov generates a coverage report. Fails if coverage is below 70%.
        pytest --cov=src --cov-report=xml --cov-fail-under=70

```

### 11.4. Implementation Checklist

1.  [ ] Create the `.github/workflows` directory in the project root if it doesn't exist.
2.  [ ] Create the `ci.yml` file with the content specified above.
3.  [ ] Go to the repository settings on GitHub. Under "Secrets and variables" -> "Actions", add a new repository secret named `TEST_GEMINI_API_KEY` with a valid API key to be used exclusively for running the CI tests.
4.  [ ] Configure the main branch's protection rules in GitHub to require the "Python Application CI" check to pass before merging is allowed.

### 11.5. Testing Strategy

-   **Testing the CI Workflow:**
    -   Create a new branch and open a pull request with a deliberate Pylint error (e.g., an unused import that `pylint --errors-only` won't catch, but a full `pylint src` would, or a simple syntax error).
    -   **Expected:** The "lint" job in the GitHub Actions tab should fail, blocking the PR.
    -   Create another pull request with a failing unit test.
    -   **Expected:** The "test" job should fail, blocking the PR.
    -   Finally, create a pull request with clean code that passes all checks.
    -   **Expected:** All jobs should pass, and the PR should be mergeable.

---

## 12. Architectural Refactor: Clarify State Management Source of Truth (P3)

### 12.1. Root Cause & Priority

-   **ID:** `SMELL-03` (New, from State Management Complexity)
-   **Priority:** P3 (Architectural Debt)
-   **Cause:** The application's state is managed across three disconnected layers (Streamlit `session_state`, LangGraph `AgentState`, and filesystem `StateManager`), creating ambiguity and a high risk of data synchronization issues. There is no clear "source of truth" during the application's lifecycle.

### 12.2. Impacted Modules

-   `src/frontend/state_helpers.py` (or equivalent UI logic)
-   `src/orchestration/state.py`
-   `src/core/state_manager.py`

### 12.3. Implementation Plan

The plan is to establish clear boundaries and define the role of each state layer, ensuring a unidirectional and unambiguous flow of data.

1.  **Streamlit `st.session_state`:** Its **only** role is to hold raw UI component values (e.g., text from a `st.text_area`, boolean from a `st.checkbox`) and track UI-specific status (e.g., `processing_started`). It should **not** hold complex data objects like `StructuredCV`.

2.  **LangGraph `AgentState`:** This is the **single source of truth** for a single, end-to-end workflow execution. It is created once at the beginning of a workflow from the raw UI state and is destroyed or archived after the workflow completes. Agents should only ever read from and write to this object.

3.  **`StateManager`:** Its **only** role is persistence. It is used to save the final, successful `AgentState` or `StructuredCV` to the disk at the *end* of a workflow, or to load a previous state at the *beginning* of a new session. It should not be called randomly from within the workflow.

#### 12.3.1. Refactor State Creation Logic

Ensure there is a single, clean function responsible for transitioning from UI state to workflow state.

-   **File:** `src/frontend/state_helpers.py`
-   **Function:** `create_agent_state_from_ui`

```python
# src/frontend/state_helpers.py
from src.orchestration.state import AgentState

def create_initial_agent_state() -> AgentState:
    """
    Creates the initial AgentState object from the raw UI session_state.
    This is the primary entry point for starting a workflow.
    """
    # Reads directly from st.session_state
    cv_text = st.session_state.get('cv_text_input', '')
    job_description = st.session_state.get('job_description_input', '')

    # All other fields in AgentState should have sensible defaults.
    initial_state = AgentState(
        cv_text=cv_text,
        raw_job_description=job_description,
        # Initialize other fields as empty/default
        structured_cv=None,
        job_description_data=None,
        research_findings=[],
        quality_check_results=None,
        final_output_path=None,
        error_messages=[]
    )
    return initial_state
```

### 12.4. Implementation Checklist

1.  [ ] Review all uses of `st.session_state` and remove any storage of complex Pydantic models. Store only raw user input.
2.  [ ] Refactor or create the `create_initial_agent_state` function to be the single point of conversion from UI state to `AgentState`.
3.  [ ] Review all agents and services. Ensure that the `StateManager` is only called from the main application logic (`app.py` or the orchestrator controller) to save/load state, not from within an agent's `run_as_node` method.
4.  [ ] Add developer documentation (docstrings or a short `README.md` in `src/core`) explaining the three state layers and their designated roles.

### 12.5. Testing Strategy

-   **Code Review:** The primary validation for this task is a manual code review to confirm that the state management boundaries are being respected.
-   **Regression Testing:** The existing integration tests must pass. A workflow that successfully saves a CV at the end confirms that the `StateManager` is still functioning correctly in its designated role.

---

---

## 13. Code Quality: Fix Function Call Signature Mismatches (P1)

### 13.1. Root Cause & Priority

-   **ID:** `ERR-05` (New, from `E1121` in Pylint report)
-   **Priority:** P1 (Critical Blocker)
-   **Cause:** The static analysis report identified a `E1121: too-many-function-args` error in `src/agents/enhanced_content_writer.py`. This indicates a method is being called with more arguments than its definition allows. This is a guaranteed `TypeError` at runtime and will crash the workflow.

### 13.2. Impacted Modules

-   `src/agents/enhanced_content_writer.py`
-   Any module that calls the method with the incorrect signature.

### 13.3. Implementation Plan

The plan is to identify the specific method call and its definition and align them. This usually happens when a method signature is changed, but not all call sites are updated.

#### 13.3.1. Correct the Method Call

-   **File:** `src/agents/enhanced_content_writer.py`
-   **Action:** Locate the method call flagged by Pylint and remove the extraneous arguments.

**Example Scenario:** Assume a static method was changed.

**Method Definition (Hypothetical):**
```python
# src/agents/enhanced_content_writer.py
class EnhancedContentWriterAgent(EnhancedAgentBase):
    @staticmethod
    def _create_prompt(section_title: str, job_keywords: list[str]) -> str:
        # The method only accepts two arguments.
        # ... prompt creation logic ...
        return prompt
```

**Incorrect Call Site (Before):**
```python
# src/agents/enhanced_content_writer.py
class EnhancedContentWriterAgent(EnhancedAgentBase):
    async def run_as_node(self, state: AgentState) -> dict:
        # ...
        # Pylint E1121: This call has too many arguments (3 given, 2 expected).
        prompt = self._create_prompt(
            current_section.title,
            state.job_description_data["keywords"],
            "some_extra_argument_that_was_removed" # This argument is the error
        )
        # ...
```

**Corrected Call Site (After):**
```python
# src/agents/enhanced_content_writer.py
class EnhancedContentWriterAgent(EnhancedAgentBase):
    async def run_as_node(self, state: AgentState) -> dict:
        # ...
        # Corrected call with the right number of arguments.
        prompt = self._create_prompt(
            current_section.title,
            state.job_description_data["keywords"]
        )
        # ...
```

### 13.4. Implementation Checklist

1.  [ ] Run `pylint src` and find the exact line number for the `E1121: too-many-function-args` error in `src/agents/enhanced_content_writer.py`.
2.  [ ] Examine the method definition and the call site to identify the mismatched arguments.
3.  [ ] Correct the method call by removing or modifying the arguments to match the function's signature.
4.  [ ] Re-run `pylint src --errors-only` to confirm the `E1121` error is resolved.

### 13.5. Testing Strategy

-   **Static Analysis:** The primary verification is a clean Pylint scan.
-   **Unit/Integration Test:** The existing tests for `EnhancedContentWriterAgent` should be run. If the incorrect call was in a code path covered by tests, a previously failing test should now pass. If the code path was not covered, this highlights a gap that should be filled with a new unit test for the `run_as_node` method.

---

## 14. Final Verification and Stabilization Sign-off

This section outlines the final steps to verify that all stabilization tasks are complete and the system is ready to exit the stabilization phase. This serves as a final quality gate before resuming feature development.

### 14.1. Verification Checklist

1.  [ ] **All P1/P2 Tasks Complete:** Confirm that every implementation checklist item from Tasks 1, 2, 3, 4, 11, and 13 in this blueprint has been completed and committed.
2.  [ ] **Clean Static Analysis:** Run `pylint src --errors-only --fail-on=E,F` one last time from the `develop` or release candidate branch. The command must exit with a status code of 0 and produce no output.
3.  [ ] **Full Test Suite Pass:** Execute the entire test suite (`pytest`) in an environment configured with a valid test API key. All tests must pass, and code coverage should meet the established threshold (e.g., 70%).
4.  [ ] **Successful End-to-End Manual Test:**
    -   Delete any existing `.env` file.
    -   Run `streamlit run app.py`.
    -   **Assert:** The application loads and shows only the "Configuration Error" message for the missing API key.
    -   Provide a valid API key in the UI sidebar.
    -   **Assert:** The full UI loads correctly.
    -   Input a sample CV and job description and run a full generation workflow.
    -   **Assert:** The workflow completes successfully, and a valid PDF is generated without any errors in the terminal.
5.  [ ] **CI Pipeline Green:** All checks on the final pull request (from `develop` to `main`) must be green. This includes the `lint` and `test` jobs defined in the CI workflow.

### 14.2. Sign-off and Next Steps

-   Once all verification steps are complete, the stabilization branch (`develop` or `feature/stabilization`) can be merged into `main`.
-   A new git tag (e.g., `v1.0.0-stable`) should be created from the `main` branch to mark this stable baseline.
-   The project can now officially move to implementing the P3 architectural improvements (Tasks 5, 6, 8, 9, 12) or begin work on new features, with high confidence in the stability of the underlying system.

---

---

## 15. Dependency Management: Upgrade Core Libraries (P3)

### 15.1. Root Cause & Priority

-   **ID:** `TECH-DEBT-01` (New)
-   **Priority:** P3 (Maintainability, Future-proofing)
-   **Cause:** The project's `requirements.txt` pins `google-generativeai==0.8.5`, which is a deprecated version. Relying on obsolete libraries poses a significant long-term risk, including lack of security updates, no access to new features (e.g., improved async support), and eventual incompatibility with other packages.

### 15.2. Impacted Modules

-   This is a project-wide change.
-   `requirements.txt`
-   `src/services/llm_service.py` (Will require significant updates to adapt to the new library's API).
-   All other modules that may directly or indirectly depend on the LLM service's response objects.

### 15.3. Implementation Plan

The plan is to upgrade to the modern `google-generativeai` library and manage dependencies formally using `pip-tools`.

#### 15.3.1. Introduce `pip-tools` for Dependency Management

Instead of manually editing `requirements.txt`, we will manage dependencies in a `requirements.in` file and compile it. This ensures deterministic builds and simplifies future upgrades.

-   **Action:** Create `requirements.in` and `dev-requirements.in` files.

**`requirements.in`:**
```
# Application Dependencies
streamlit
pydantic
langgraph
chromadb
google-generativeai>=1.0.0 # Specify the new version
weasyprint
jinja2
tenacity
asyncio
```

**`dev-requirements.in`:**
```
-r requirements.in
# Development & Testing Dependencies
pytest
pytest-cov
pylint
pip-tools
```

-   **Action:** Generate the `requirements.txt` file.

```bash
# Install pip-tools first
pip install pip-tools

# Compile the requirements
pip-compile requirements.in -o requirements.txt
pip-compile dev-requirements.in -o dev-requirements.txt
```

#### 15.3.2. Refactor `EnhancedLLMService` for the New SDK

The new `google-generativeai` SDK has a different API, particularly for async calls. The service must be updated accordingly.

-   **File:** `src/services/llm_service.py`
-   **Action:** Update the API calls.

**Before (Using Deprecated SDK):**
```python
# src/services/llm_service.py
# This synchronous call was being run in an executor
response = self.llm.generate_content(prompt)
```

**After (Using Modern SDK with Native Async):**
```python
# src/services/llm_service.py
class EnhancedLLMService:
    # ... __init__ ...

    async def _make_llm_api_call(self, prompt: str) -> Any:
        """Makes the native async LLM API call."""
        # The new SDK has a native generate_content_async method
        response = await self.llm.generate_content_async(prompt)
        return response

    async def _generate_with_timeout(self, prompt: str, ...) -> Any:
        """Generate content with a timeout using native async."""
        try:
            # No more need for run_in_executor, simplifying the code
            return await asyncio.wait_for(
                self._make_llm_api_call(prompt),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            # ... error handling ...
```

### 15.4. Implementation Checklist

1.  [ ] Create the `requirements.in` and `dev-requirements.in` files.
2.  [ ] Run `pip-compile` to generate the locked `requirements.txt` and `dev-requirements.txt` files.
3.  [ ] Update the CI workflow (`ci.yml`) to install dependencies using `pip install -r dev-requirements.txt`.
4.  [ ] Refactor `EnhancedLLMService` in `src/services/llm_service.py` to use the new SDK's API, especially the native `generate_content_async` method.
5.  [ ] Remove the `run_in_executor` logic from `_generate_with_timeout` and replace it with a direct `await` on the new async method, wrapped in `asyncio.wait_for`.
6.  [ ] Run the full test suite to catch any regressions or breaking changes from the SDK upgrade.

### 15.5. Testing Strategy

-   **Full Regression Testing:** This is the most critical part of this task. The entire test suite (`pytest`) must be executed.
-   **Manual E2E Testing:** A full manual workflow test is required to visually and functionally confirm that the output of the new SDK is handled correctly by the rest of the application. Pay close attention to the structure of the response object, as it may have changed.

---

## 16. Documentation: Update Project README (P3)

### 16.1. Root Cause & Priority

-   **ID:** `DOC-01` (New)
-   **Priority:** P3 (Maintainability)
-   **Cause:** The project's `README.md` is likely outdated or minimal. Clear, concise documentation is essential for developer onboarding, project maintenance, and clarifying the purpose and architecture of the system.

### 16.2. Impacted Modules

-   `README.md` (Project root)

### 16.3. Implementation Plan

The plan is to restructure and update the `README.md` to include essential information for anyone interacting with the project.

#### 16.3.1. Proposed `README.md` Structure

```markdown
# AI CV Generator (`anasakhomach-aicvgen`)

A multi-agent system built with LangGraph and Streamlit to automatically generate and tailor CVs based on a job description.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Core Technologies](#core-technologies)
- [Setup and Installation](#setup-and-installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Running Tests](#running-tests)

## Architecture Overview

This project uses a multi-agent, state-driven architecture orchestrated by **LangGraph**. The primary components are:
- **UI (Streamlit):** The user-facing interface for inputting data.
- **Orchestrator (LangGraph):** Manages the state and flow of the CV generation process through a defined graph of nodes.
- **Agents:** Specialized Python classes responsible for discrete tasks (e.g., parsing, research, writing, QA).
- **Services:** Shared utilities that provide core functionality like LLM interaction (`EnhancedLLMService`) and vector storage (`VectorStoreService`).
- **State (`AgentState`):** A Pydantic model that serves as the single source of truth during a workflow execution.

## Core Technologies

- **UI:** Streamlit
- **Orchestration:** LangGraph
- **LLM:** Google Gemini
- **Vector Store:** ChromaDB
- **Data Modeling:** Pydantic
- **PDF Generation:** WeasyPrint + Jinja2

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/anasakhomach-aicvgen.git
    cd anasakhomach-aicvgen
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    This project uses `pip-tools` for dependency management. Install from the compiled `dev-requirements.txt` file.
    ```bash
    pip install -r dev-requirements.txt
    ```

## Configuration

The application requires a Google Gemini API key.

1.  **Create a `.env` file** in the project root by copying the example:
    ```bash
    cp .env.example .env
    ```

2.  **Edit the `.env` file** and add your API key:
    ```
    GEMINI_API_KEY="your-api-key-here"
    ```
    Alternatively, you can provide the key via the Streamlit UI at runtime.

## Running the Application

To start the Streamlit UI, run the following command from the project root:
```bash
streamlit run app.py
```

## Running Tests

To run the full suite of unit and integration tests, use `pytest`:
```bash
pytest
```

To run tests with code coverage reporting:
```bash
pytest --cov=src
```
```

### 16.4. Implementation Checklist

1.  [ ] Create a `.env.example` file for users to copy.
2.  [ ] Review the proposed `README.md` structure.
3.  [ ] Write or update the "Architecture Overview" section to reflect the current design.
4.  [ ] Update the "Setup and Installation" instructions to include the use of `dev-requirements.txt`.
5.  [ ] Add the "Configuration" section explaining the `.env` file.
6.  [ ] Verify that the commands in "Running the Application" and "Running Tests" are correct.
7.  [ ] Replace the placeholder `README.md` with the new, updated content.

---

The blueprint is now comprehensive, covering all phases from immediate critical bug fixes to long-term architectural improvements and process formalization. No further tasks are required for the stabilization phase. The existing 16 sections provide a complete and actionable roadmap for taking the `anasakhomach-aicvgen` project from a fragile prototype to a stable, maintainable, and well-documented MVP baseline.