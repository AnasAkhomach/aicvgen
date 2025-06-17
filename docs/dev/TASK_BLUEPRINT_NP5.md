# **TASK_BLUEPRINT.md**

## **1. Overview & Technical Strategy**

This technical blueprint outlines the Minimum Viable Product (MVP) for remediating the critical technical debt identified in the `aicvgen` codebase audit. The primary goals are to enhance security, stabilize the architecture, and improve user experience by addressing the most impactful issues first.

The strategy is divided into two main phases:

1.  **Critical Security Remediation:** Immediately address the hard-coded API key vulnerability to secure the application.
2.  **Architectural Unification & Performance Fix:** Formally complete the incomplete architectural refactoring by removing the legacy UI-to-backend interaction pattern, enforcing strict data contracts with Pydantic models, and fixing the blocking UI operations.

Successfully executing this plan will resolve the audit's highest-priority findings (SV-01, AD-01, CB-01, PERF-01) and establish a stable, secure, and maintainable foundation for future development.

---

## **Phase 1: Critical Security Remediation (SV-01)**

### **Task/Feature Addressed**

Remove the hard-coded Google API key from the repository to mitigate a critical security vulnerability. Ensure the application exclusively uses environment variables for sensitive credentials.

### **Affected Component(s)**

*   `docs/user/playwrite_test_var.txt`: Contains the hard-coded key.
*   `src/config/settings.py`: Must be verified to load the key from the environment.
*   `.env.example`: Must be the single source of truth for environment variable examples.
*   All components that rely on the Gemini API key.

### **Detailed Implementation Steps**

1.  **Immediate Action (Outside of Code):** The exposed API key `AIzaSyAkFX7p9KvqEmWPHWvB-L6uhBwt_5QbDoY` must be considered **compromised**. **Revoke this key immediately** in the Google Cloud Platform console to prevent any potential misuse.
2.  **Remove Hard-coded Key:** Delete the entire `api key: {{...}}` line from `docs/user/playwrite_test_var.txt`.
3.  **Verify Environment Loading:** In `src/config/settings.py`, within the `LLMConfig` dataclass, ensure the `gemini_api_key_primary` field is loaded **only** from the environment. The current implementation is correct and requires no changes.

    ```python
    # src/config/settings.py -> LLMConfig

    # This is correct. Do not change.
    gemini_api_key_primary: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    # ...

    # The __post_init__ correctly raises an error if the key is not found.
    # This is also correct.
    def __post_init__(self):
        if not self.gemini_api_key_primary and not self.gemini_api_key_fallback:
            raise ValueError(
                "At least one Gemini API key is required. "
                "Please set GEMINI_API_KEY or GEMINI_API_KEY_FALLBACK environment variables..."
            )
    ```

4.  **Update `.env.example`:** Modify the `.env.example` file to remove the exposed key and provide a clear placeholder.

    ```diff
    # .env.example

    # =============================================================================
    # LLM API Configuration
    # =============================================================================

    # Primary Gemini API Key (Required)
    # Get your API key from: https://aistudio.google.com/app/apikey
    -GEMINI_API_KEY=AIzaSyA7o8aq_BthwDiJfzmpMFuWmPOTK97B-Lg
    +GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```

5.  **Follow-up Security Note:** While purging the key from the Git history is out of scope for this MVP task, it remains a critical security best practice to prevent historical compromise. This should be scheduled as a high-priority follow-up action.

### **Testing Considerations**

*   **Unit Test:** Create a new test file `tests/unit/test_config_security.py` to verify that `AppConfig` raises a `ValueError` if the `GEMINI_API_KEY` environment variable is not set.
*   **Manual Test:** After implementing the changes, delete your local `.env` file. Run the application and confirm that it fails on startup with a clear error message asking for the API key.

---

## **Phase 2: Architectural Unification & Performance Fix (AD-01, CB-01, PERF-01)**

This phase addresses the core architectural drift by removing the legacy UI pattern, enforcing strict data contracts, and fixing the blocking UI.

### **Sub-task 2.1: Deprecate Legacy UI Interaction Pattern**

**Task/Feature Addressed:** Unify the codebase around the modern `src/frontend/` and LangGraph-based architecture. This resolves the architectural duality (AD-01) and removes dead code (DL-01).

**Affected Component(s):**

*   `src/core/callbacks.py` (To be deleted)
*   `src/core/ui_components.py` (To be deleted)
*   `src/core/enhanced_orchestrator.py` (To be deleted)
*   `src/core/main.py` (To be refactored)
*   `app.py` (To be verified)

**Detailed Implementation Steps:**

1.  **Confirm Entry Point:** The main entry point is `app.py`, which correctly calls `main` from `src/core/main.py`. This remains unchanged.
2.  **Refactor `src/core/main.py`:** This file is the central controller. It must be refactored to *only* use components from `src/frontend/`.
    *   Remove any imports from `src/core/callbacks.py` and `src.core.ui_components.py`.
    *   Ensure the main application loop uses the modern, non-blocking `handle_workflow_execution` from `src/frontend/callbacks.py`.

    ```python
    # src/core/main.py (Conceptual Refactoring)

    # ... other imports
    # REMOVE: from src.core.ui_components import ...
    # REMOVE: from src.core.callbacks import ...

    # ADD/VERIFY these imports:
    from src.frontend.ui_components import (
        display_sidebar,
        display_input_form,
        display_review_and_edit_tab,
        display_export_tab
    )
    from src.frontend.state_helpers import initialize_session_state
    from src.frontend.callbacks import handle_workflow_execution # The modern, non-blocking handler
    # ...

    def main():
        # ...
        initialize_session_state()
        display_sidebar(st.session_state.agent_state)
        # ...

        if st.session_state.get('run_workflow'):
            st.session_state.processing = True
            st.session_state.run_workflow = False

            trace_id = str(uuid.uuid4())

            # This is the key change: use the modern, non-blocking handler
            handle_workflow_execution(trace_id)

        # ... Render tabs using src.frontend.ui_components
        tab1, tab2, tab3 = st.tabs(["📝 Input & Generate", "✏️ Review & Edit", "📄 Export"])

        with tab1:
            display_input_form(st.session_state.agent_state)

        with tab2:
            display_review_and_edit_tab(st.session_state.agent_state)

        with tab3:
            display_export_tab(st.session_state.agent_state)
        # ...
    ```

3.  **Delete Legacy Components:** After confirming that `src/core/main.py` and all other parts of the application no longer reference them, delete the following files:
    *   `src/core/callbacks.py`
    *   `src/core/ui_components.py`
    *   `src/core/enhanced_orchestrator.py` (This is replaced by the LangGraph `cv_graph_app`).

### **Sub-task 2.2: Enforce Pydantic Models as the API Contract**

**Task/Feature Addressed:** Eliminate API contract brittleness (CB-01) by enforcing the use of Pydantic models (`AgentState`) for all data flow into and between agents.

**Affected Component(s):**

*   All agent implementations in `src/agents/`.
*   Specifically, `src/agents/enhanced_content_writer.py` and `src/agents/quality_assurance_agent.py`.

**Detailed Implementation Steps:**

1.  **Establish `AgentState` as the Contract:** The `run_as_node` method for every agent already accepts `state: AgentState`. This contract must be strictly trusted. No agent should receive raw dictionaries.
2.  **Remove Defensive `isinstance` Checks:** Go through all agent implementations and remove code that defensively checks if an input is a `dict` versus a Pydantic model. The data entering the agent `run_as_node` method is guaranteed to be a validated `AgentState` object by LangGraph.

    **Example Refactoring in `src/agents/enhanced_content_writer.py`:**

    ```python
    # In a helper method called by run_as_node

    # --- BEFORE (Defensive Coding) ---
    # job_data = input_data.get("job_description_data", {})
    # if isinstance(job_data, str):
    #     logger.warning("DATA STRUCTURE MISMATCH: job_description_data is a string...")
    #     # ... conversion logic ...
    # elif isinstance(job_data, dict):
    #     # ... validation logic ...

    # --- AFTER (Trusting the Contract) ---
    # The run_as_node method receives the state directly.
    # No validation is needed here because it's handled at the graph's edge.

    # In run_as_node(self, state: AgentState) -> dict:
    job_data = state.job_description_data
    structured_cv = state.structured_cv

    # The agent can now directly access attributes with confidence, e.g.:
    job_title = job_data.title
    required_skills = job_data.skills
    # No more .get() with defaults or isinstance checks are needed.
    ```

3.  **Update Agent Logic:** Refactor the internal logic of all agents to directly access attributes from the Pydantic models within the `AgentState` (e.g., `state.structured_cv.sections`, `state.job_description_data.skills`).

### **Sub-task 2.3: Implement Non-Blocking Workflow Execution**

**Task/Feature Addressed:** Fix the critical UI performance bottleneck (PERF-01) where the Streamlit interface freezes during CV generation.

**Affected Component(s):**

*   `src/frontend/callbacks.py`
*   `src/core/main.py` (the main application loop)

**Detailed Implementation Steps:**

1.  **Refactor `handle_workflow_execution` for Threading:** Modify the function in `src/frontend/callbacks.py` to start the async workflow in a separate thread, allowing the UI to remain responsive.

    ```python
    # src/frontend/callbacks.py

    import streamlit as st
    import asyncio
    import threading
    from src.orchestration.cv_workflow_graph import cv_graph_app
    from src.orchestration.state import AgentState

    def _execute_workflow_in_thread(initial_state_dict: dict, trace_id: str):
        """The target function for the background thread."""
        try:
            # Each thread needs its own event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run the async workflow
            final_state_dict = loop.run_until_complete(
                cv_graph_app.ainvoke(initial_state_dict, {"configurable": {"trace_id": trace_id}})
            )

            # Store the result in session_state for the main thread to pick up
            st.session_state.workflow_result = final_state_dict
        except Exception as e:
            st.session_state.workflow_error = e
        finally:
            st.session_state.processing = False

    def handle_workflow_execution(trace_id: str):
        """
        Starts the CV generation workflow in a separate thread to avoid
        blocking the Streamlit UI.
        """
        if 'agent_state' not in st.session_state or st.session_state.agent_state is None:
            from src.core.state_helpers import create_agent_state_from_ui
            st.session_state.agent_state = create_agent_state_from_ui()

        initial_state = st.session_state.agent_state
        initial_state.trace_id = trace_id

        # Clear previous results/errors
        st.session_state.workflow_result = None
        st.session_state.workflow_error = None

        # Run the workflow in a background thread
        thread = threading.Thread(
            target=_execute_workflow_in_thread,
            args=(initial_state.model_dump(), trace_id)
        )
        thread.start()

        # The UI will now remain responsive. The main loop will check for the result.
    ```

2.  **Update the Main Application Loop (`src/core/main.py`):** Modify the main loop to handle the results from the background thread without blocking.

    ```python
    # src/core/main.py

    def main():
        # ... (initialization and sidebar display) ...

        # --- Start of new logic to handle thread results ---
        if st.session_state.get('processing'):
            st.spinner("Processing your CV... Please wait.")
            # The UI is not blocked here. You could add a progress bar that updates.

        if 'workflow_result' in st.session_state and st.session_state.workflow_result:
            # Workflow is done, process the result
            final_state_dict = st.session_state.workflow_result
            st.session_state.agent_state = AgentState.model_validate(final_state_dict)
            st.session_state.workflow_result = None # Clear the result
            st.success("CV Generation Complete!")
            st.rerun() # Rerun to display the new state in the 'Review & Edit' tab

        if 'workflow_error' in st.session_state and st.session_state.workflow_error:
            # Workflow failed
            error = st.session_state.workflow_error
            st.error(f"An error occurred during CV generation: {error}")
            st.session_state.workflow_error = None # Clear the error
        # --- End of new logic ---

        if st.session_state.get('run_workflow'):
            st.session_state.processing = True
            st.session_state.run_workflow = False
            trace_id = str(uuid.uuid4())
            handle_workflow_execution(trace_id)
            st.rerun() # Rerun to show the spinner immediately

        # ... (rest of the UI tab rendering) ...
    ```

### **Testing Strategy for Phase 2**

*   **Unit Tests:**
    *   Create unit tests for the refactored agents to ensure they correctly process `AgentState` objects without defensive checks.
    *   Write a unit test for the new `_execute_workflow_in_thread` function to verify it correctly runs the async loop and stores results in `st.session_state`.
*   **Integration Tests:**
    *   Update `tests/integration/test_complete_workflow_integration.py` to reflect the new non-blocking UI flow.
    *   Add a new integration test to simulate a user clicking "Generate", verify the spinner appears, and then verify the results are displayed correctly once the background thread completes.
*   **E2E Tests:**
    *   The existing E2E tests in `tests/e2e/` will be critical for regression testing. They must be run to ensure that the architectural changes have not broken the end-to-end functionality.

---

---

## **Phase 3: Code Hygiene and Maintainability**

This phase focuses on improving the long-term health, readability, and maintainability of the codebase by addressing lower-priority but still important technical debt items.

### **Sub-task 3.1: Consolidate Configuration Variables (AD-02)**

**Task/Feature Addressed:** Remove ambiguity and redundancy from the environment configuration by consolidating conflicting variables.

**Affected Component(s):**

*   `.env.example`: The source of the redundant variables.
*   `src/config/settings.py`: The consumer of these variables; may need minor adjustments.
*   `src/config/environment.py`: An alternative location where configuration might be consumed.

**Detailed Implementation Steps:**

1.  **Audit and Consolidate in `.env.example`:** Edit the `.env.example` file to remove duplicates and establish a single source of truth for each setting.

    ```diff
    # .env.example

    # ...
    # Debug mode (set to false in production)
    -ENABLE_DEBUG_MODE=false
    -DEBUG=true
    +DEBUG_MODE=false

    # Session timeout in seconds (default: 1 hour)
    -SESSION_TIMEOUT=3600
    -SESSION_TIMEOUT_MINUTES=60
    +SESSION_TIMEOUT_SECONDS=3600

    # Maximum concurrent requests
    MAX_CONCURRENT_REQUESTS=10

    # ...
    # RATE LIMITING AND PERFORMANCE
    # ...
    # Rate limiting (requests and tokens per minute)
    -RATE_LIMIT_RPM=60
    -MAX_REQUESTS_PER_MINUTE=30
    +LLM_REQUESTS_PER_MINUTE=30
    -MAX_TOKENS_PER_MINUTE=6000
    +LLM_TOKENS_PER_MINUTE=60000 # Correcting likely typo from audit

    # ...
    ```

2.  **Verify Configuration Loading:** Review `src/config/settings.py` and `src/config/environment.py` to ensure they load the consolidated variable names (e.g., `SESSION_TIMEOUT_SECONDS`, `LLM_REQUESTS_PER_MINUTE`). Adjust the code to use these new canonical keys.

### **Sub-task 3.2: Refactor Duplicated LLM Interaction Logic (DUP-01)**

**Task/Feature Addressed:** Abstract the common pattern of calling an LLM and parsing its JSON output into a shared utility to reduce code duplication.

**Affected Component(s):**

*   `src/agents/cv_analyzer_agent.py`
*   `src/agents/parser_agent.py`
*   `src/agents/agent_base.py` or a new `src/utils/llm_utils.py` file.

**Detailed Implementation Steps:**

1.  **Create a Shared Utility:** The best location for this is in `src/agents/agent_base.py` as a protected method on `EnhancedAgentBase`, since it's a common agent task.

    ```python
    # src/agents/agent_base.py -> Add to EnhancedAgentBase class

    from src.services.llm import LLMResponse
    from src.utils.exceptions import LLMResponseParsingError
    import json

    # ... inside EnhancedAgentBase class ...

    async def _generate_and_parse_json(
        self,
        prompt: str,
        context: 'AgentExecutionContext',
        llm_service: 'EnhancedLLMService'
    ) -> Dict[str, Any]:
        """
        Generates content from the LLM and parses the JSON response.
        Handles common markdown fence issues and parsing errors.
        """
        try:
            # Generate content using the provided LLM service
            llm_response: LLMResponse = await llm_service.generate_content(
                prompt,
                session_id=context.session_id,
                item_id=context.item_id
            )

            if not llm_response.success or not llm_response.content:
                raise LLMResponseParsingError("LLM response was empty or unsuccessful.")

            # Extract JSON from the raw response (handles ```json ... ```)
            json_string = llm_response.content.strip()
            if json_string.startswith("```json"):
                json_string = json_string[len("```json"):].strip()
            if json_string.endswith("```"):
                json_string = json_string[:-len("```")].strip()

            # Parse the JSON string
            return json.loads(json_string)

        except json.JSONDecodeError as e:
            self.logger.error("Failed to decode JSON from LLM response", error=str(e), raw_response=llm_response.content)
            raise LLMResponseParsingError(f"LLM response was not valid JSON: {e}", raw_response=llm_response.content) from e
        except Exception as e:
            self.logger.error("An unexpected error occurred during LLM interaction", error=str(e))
            raise AgentExecutionError(self.name, f"LLM interaction failed: {e}")

    ```

2.  **Refactor Agents:** Update `CVAnalyzerAgent` and `ParserAgent` to use this new utility method.

    **Example for `ParserAgent`:**

    ```python
    # src/agents/parser_agent.py

    # ... inside _parse_job_description_with_llm method ...

    # --- BEFORE ---
    # response = await self.llm.generate_content(prompt, trace_id=trace_id)
    # raw_response_content = response.content
    # json_start = raw_response_content.find('{')
    # # ... manual JSON extraction and parsing ...
    # parsed_data = json.loads(json_str)

    # --- AFTER ---
    # Assuming 'context' is available or can be constructed
    parsed_data = await self._generate_and_parse_json(prompt, context, self.llm)

    # ... continue with Pydantic validation ...
    ```

### **Sub-task 3.3: Apply Performance Optimization Frameworks (PERF-02)**

**Task/Feature Addressed:** Connect the existing, powerful performance optimization frameworks to the agents doing the actual work.

**Affected Component(s):**

*   `src/agents/enhanced_content_writer.py`
*   `src/agents/cv_analyzer_agent.py`
*   `src/agents/parser_agent.py`
*   `src/agents/quality_assurance_agent.py`
*   `src/agents/research_agent.py`

**Detailed Implementation Steps:**

1.  **Apply `@optimize_async` Decorator:** Go to each agent file and apply the decorator to the primary `run_as_node` method. This single change will activate the performance monitoring, adaptive concurrency, and other features of the `AsyncOptimizer`.

    ```python
    # Example for src/agents/enhanced_content_writer.py

    from src.core.async_optimizer import optimize_async

    class EnhancedContentWriterAgent(EnhancedAgentBase):
        # ... existing methods ...

        @optimize_async("content_generation") # Add this decorator
        async def run_as_node(self, state: AgentState) -> dict:
            # ... existing implementation ...
    ```

2.  **Repeat for All Agents:** Apply the decorator to the `run_as_node` method of all other key agents, using an appropriate `operation_type` string:
    *   `ParserAgent`: `@optimize_async("parsing")`
    *   `ResearchAgent`: `@optimize_async("research")`
    *   `QualityAssuranceAgent`: `@optimize_async("qa_check")`
    *   `FormatterAgent`: `@optimize_async("formatting")`

### **Testing Considerations for Phase 3**

*   **Configuration:** After refactoring, run the application with different settings in a local `.env` file to ensure the new consolidated variables are being correctly interpreted.
*   **Duplication Refactor:** Unit tests for `CVAnalyzerAgent` and `ParserAgent` should be updated to mock the new `_generate_and_parse_json` method instead of the raw LLM service call.
*   **Performance Decorators:** No new tests are strictly required, but running existing integration tests will now generate performance metrics that can be monitored in logs, confirming the framework is active.

---

## **Final Blueprint Summary**

This MVP blueprint prioritizes security and stability.

*   **Phase 1** secures the application by removing the hard-coded API key.
*   **Phase 2** resolves the core architectural problems by unifying the codebase on the modern LangGraph pattern, enforcing strict data contracts, and fixing the freezing UI.
*   **Phase 3** improves code hygiene and activates the application's powerful but underutilized performance frameworks.

Completing these phases will result in a secure, stable, and maintainable application, ready for reliable future development. ready for reliable future development.

---

