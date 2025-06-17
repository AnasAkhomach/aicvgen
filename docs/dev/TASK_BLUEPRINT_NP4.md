Of course. I have thoroughly analyzed the provided codebase, including the implemented refactoring blueprints. The work done so far has established a strong foundation with a stateful backend graph and standardized agent interfaces.

My audit confirms that the implementation of the previous blueprints is largely successful. Key features like the `create_agent_state_from_ui` function, the standardized `run_as_node` abstract method in `EnhancedAgentBase`, the JSON-based LLM contracts, the LaTeX escaping utility, and the centralized configuration are present and align with the plans.

However, the audit also confirms the critical issue identified in `TASK_BLUEPRINT_NP3.md`: the `ParserAgent`'s `run_as_node` method is incomplete and does not perform the full parsing logic, which is a significant bug. This must be addressed.

Based on your request, I will now provide the technical blueprint for the next, most critical step: refactoring `src/core/main.py` for maintainability and applying the principle of separation of concerns. This plan will also implicitly rely on the `ParserAgent` bug being fixed as per `TASK_BLUEPRINT_NP3.md` to function correctly.

Here is the blueprint for the `aicvgen` project.

### **TASK_BLUEPRINT.md**

---

### **Refactoring Task 9: Decouple and Refactor `main.py` for Maintainability**

**1. Task/Feature Addressed**

This task addresses the growing complexity of the `src/core/main.py` Streamlit application. The current implementation mixes UI rendering, state management, and backend workflow invocation, making it difficult to maintain and debug.

The goal is to refactor `main.py` by applying the **Separation of Concerns** principle. We will decouple the UI components, state management logic, and backend interaction logic into distinct, manageable modules. This will make the main application script a clean, high-level controller, significantly improving readability, testability, and long-term maintainability.

**2. Affected Component(s)**

* **`src/core/main.py`**: Will be heavily refactored to act as a high-level controller.
* **`src/frontend/ui_components.py` (New File)**: Will contain functions for rendering specific parts of the Streamlit UI (e.g., the sidebar, input forms, review sections).
* **`src/frontend/state_helpers.py` (New File)**: Will contain helper functions for initializing and managing Streamlit's `session_state`.
* **`src/frontend/callbacks.py` (New File)**: Will house the callback functions for UI elements like buttons, abstracting the logic for updating state based on user actions.

**3. Pydantic Model Changes**

No changes to Pydantic models are required for this task.

**4. Detailed Implementation Steps**

**Step 1: Create UI Component Rendering Module**

Create a new file `src/frontend/ui_components.py` to house all functions that render parts of the UI. This moves the "view" logic out of `main.py`.

* **Create `src/frontend/ui_components.py`:**

```python
# In src/frontend/ui_components.py
import streamlit as st
from src.orchestration.state import AgentState
from .callbacks import handle_user_action # We will create this in the next step

def display_sidebar(state: AgentState):
    """Renders the sidebar for session management and settings."""
    with st.sidebar:
        st.title("🔧 Session Management")
        # ... (Add all sidebar logic here: API key, safety controls, session loading)

def display_input_form(state: AgentState):
    """Renders the initial input form for job description and CV text."""
    st.header("1. Input Your Information")
    job_description = st.text_area("🎯 Job Description", height=200, key="job_description")
    cv_content = st.text_area("📄 Your Current CV", height=300, key="cv_text")

    if st.button("🚀 Generate Tailored CV", type="primary", use_container_width=True):
        st.session_state['run_workflow'] = True # Set a flag to trigger workflow in main.py
        st.rerun()


def display_review_and_edit_tab(state: AgentState):
    """Renders the 'Review & Edit' tab with section-based controls."""
    if not state or not state.structured_cv:
        st.info("Please generate a CV first to review it here.")
        return

    for section in state.structured_cv.sections:
        with st.expander(f"### {section.name}", expanded=True):
            if section.items:
                for item in section.items:
                    _display_reviewable_item(item, state)
            if section.subsections:
                for sub in section.subsections:
                    st.markdown(f"#### {sub.name}")
                    for item in sub.items:
                        _display_reviewable_item(item, state)

def _display_reviewable_item(item, state):
    """Displays a single reviewable item with Accept/Regenerate buttons."""
    item_id = str(item.id)
    st.markdown(f"> {item.content}")

    # Raw LLM Output for transparency
    if item.raw_llm_output:
        with st.expander("🔍 View Raw LLM Output", expanded=False):
            st.code(item.raw_llm_output, language="text")

    cols = st.columns([1, 1, 5])
    with cols[0]:
        st.button("✅ Accept", key=f"accept_{item_id}", on_click=handle_user_action, args=('accept', item_id))
    with cols[1]:
        st.button("🔄 Regenerate", key=f"regenerate_{item_id}", on_click=handle_user_action, args=('regenerate', item_id))

def display_export_tab(state: AgentState):
    """Renders the 'Export' tab."""
    if state and state.final_output_path:
        st.success(f"✅ Your CV has been generated!")
        with open(state.final_output_path, "rb") as f:
            st.download_button(
                label="📥 Download PDF",
                data=f,
                file_name=Path(state.final_output_path).name,
                mime="application/pdf"
            )
    else:
        st.info("Generate and finalize your CV to enable export options.")

```

**Step 2: Create UI Callbacks Module**

Create a new file `src/frontend/callbacks.py` to handle the logic for UI interactions. This separates UI actions from UI rendering.

* **Create `src/frontend/callbacks.py`:**

```python
# In src/frontend/callbacks.py
import streamlit as st
from src.models.data_models import UserAction, UserFeedback

def handle_user_action(action: str, item_id: str):
    """
    Callback to update the agent_state with user feedback.
    This function's sole responsibility is to prepare the state for the
    main loop to invoke the backend graph. It does NOT call the graph directly.
    """
    if 'agent_state' in st.session_state and st.session_state.agent_state:
        st.session_state.agent_state.user_feedback = UserFeedback(
            action=UserAction(action),
            item_id=item_id
        )
        # Set a flag indicating a backend call is needed
        st.session_state.run_workflow = True
```

**Step 3: Create State Helper Module**

Create a new file `src/frontend/state_helpers.py` to manage `st.session_state` initialization.

* **Create `src/frontend/state_helpers.py`:**

```python
# In src/frontend/state_helpers.py
import streamlit as st

def initialize_session_state():
    """Initializes all necessary keys in Streamlit's session state."""
    if 'agent_state' not in st.session_state:
        st.session_state.agent_state = None
    if 'run_workflow' not in st.session_state:
        st.session_state.run_workflow = False
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    # Initialize other keys from the original main.py as needed
```

**Step 4: Refactor `src/core/main.py` to be a High-Level Controller**

The `main.py` file will now be much cleaner. Its job is to manage the main application loop, call the backend when needed, and delegate UI rendering to the new `ui_components` module.

* **Refactor `src/core/main.py`:**

```python
# In src/core/main.py

import streamlit as st
from src.frontend.state_helpers import initialize_session_state
from src.frontend.ui_components import (
    display_sidebar,
    display_input_form,
    display_review_and_edit_tab,
    display_export_tab
)
from src.integration.enhanced_cv_system import get_enhanced_cv_integration
from src.orchestration.cv_workflow_graph import cv_graph_app # Assumes this is the compiled graph
from src.core.main import create_agent_state_from_ui # Keep the factory function here or move to state_helpers
from src.orchestration.state import AgentState
import asyncio

def main():
    """
    Main Streamlit application controller.
    Orchestrates the UI rendering and backend workflow invocations.
    """
    # 1. Initialize State
    initialize_session_state()

    # 2. Display Static UI Components
    display_sidebar(st.session_state.agent_state)
    st.title("🤖 AI CV Generator")

    # 3. Main Interaction & Backend Loop
    if st.session_state.get('run_workflow'):
        st.session_state.processing = True
        st.session_state.run_workflow = False  # Reset flag

        with st.spinner("Processing..."):
            try:
                # If it's the first run, create a new state from UI inputs
                if st.session_state.agent_state is None:
                    st.session_state.agent_state = create_agent_state_from_ui()

                # Invoke the LangGraph backend
                # The 'ainvoke' method takes the current state and returns the new state
                new_state_dict = asyncio.run(
                    cv_graph_app.ainvoke(st.session_state.agent_state.model_dump())
                )

                # Overwrite the old state with the new state
                st.session_state.agent_state = AgentState.model_validate(new_state_dict)

                # Clear feedback so the same action doesn't run again on the next rerun
                if st.session_state.agent_state.user_feedback:
                    st.session_state.agent_state.user_feedback = None

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                # Optionally reset state on critical failure
                # st.session_state.agent_state = None
            finally:
                st.session_state.processing = False
                st.rerun() # Force a re-render with the new state

    # 4. Render UI Tabs based on the current state
    tab1, tab2, tab3 = st.tabs(["📝 Input & Generate", "✏️ Review & Edit", "📄 Export"])

    with tab1:
        display_input_form(st.session_state.agent_state)

    with tab2:
        display_review_and_edit_tab(st.session_state.agent_state)

    with tab3:
        display_export_tab(st.session_state.agent_state)

    # Display any errors from the workflow
    if st.session_state.agent_state and st.session_state.agent_state.error_messages:
        for error in st.session_state.agent_state.error_messages:
            st.error(error)

if __name__ == "__main__":
    main()
```

**5. Testing Considerations**

* **`ui_components.py`:** Can be tested using visual inspection or more advanced UI testing frameworks for Streamlit if necessary. The main goal is to ensure they render correctly given a mock `AgentState`.
* **`callbacks.py`:** Create unit tests that mock `st.session_state`, call the `handle_user_action` function, and assert that `st.session_state.user_feedback` and `st.session_state.run_workflow` are set correctly.
* **`state_helpers.py`:** Unit test `initialize_session_state` by mocking `st.session_state` and verifying that all required keys are created.
* **`main.py`:** Integration tests should now be easier. You can test the main loop's logic by mocking `cv_graph_app.ainvoke`, setting `st.session_state.run_workflow = True`, and asserting that the mock was called and that `st.session_state.agent_state` was updated with the mock's return value.

---

### **Critical Gaps & Questions**

* **`ParserAgent` Bug:** The refactoring of `main.py` will not work as expected until the bug in `ParserAgent` (where `run_as_node` is incomplete) is fixed as per `TASK_BLUEPRINT_NP3.md`. This is the **highest priority** technical debt to address.
* **Gemini API Key Management:** The current plan passes the API key to the `EnhancedCVConfig`. It is crucial to ensure this key is securely managed and never exposed in the frontend or logs. The `CredentialRedactor` from the security utils should be used in all logging.
* **Async in Streamlit:** The refactored `main.py` uses `asyncio.run()` to call the async `ainvoke` method. While this works, it's a blocking call within the Streamlit script execution. For the MVP, this is acceptable. For future performance enhancements, exploring solutions like `streamlit-async` or a separate backend service would be necessary to avoid blocking the UI thread during long-running graph executions.

---

Excellent. As per our agreed-upon plan, we will now address the highest-priority technical debt: fixing the critical logic bug in the `ParserAgent`.

The current implementation of `run_as_node` in `src/agents/parser_agent.py` is incomplete. It does not correctly use the initial state to decide whether to parse an existing CV or create a new one from scratch. This blueprint provides the precise instructions to correct this bug and consolidate the agent's logic.

### **TASK_BLUEPRINT.md**

---

### **Refactoring Task 10: Fix `ParserAgent` Logic and Consolidate Execution Path**

**1. Task/Feature Addressed**

This task rectifies a critical bug in the `ParserAgent` where its primary execution method for the LangGraph workflow, `run_as_node`, fails to implement the agent's full capabilities. The method does not correctly parse the user's initial CV or handle the "start from scratch" scenario.

This blueprint provides the corrected implementation for `run_as_node`, ensuring it serves as the single, canonical entry point for all parsing logic. It also removes the confusing and now-redundant `run_async` method to eliminate ambiguity.

**2. Affected Component(s)**

* **`src/agents/parser_agent.py`**: The file will be refactored to correct the `run_as_node` method and remove legacy code.

**3. Pydantic Model Changes**

No changes to any Pydantic models are required for this task.

**4. Detailed Implementation Steps**

**Step 1: Delete the Redundant `run_async` Method**

The `run_async` method is a remnant of a previous design and is no longer needed. Its presence creates confusion and an alternative execution path that is not compatible with our standardized LangGraph architecture.

* In `src/agents/parser_agent.py`, **delete the entire `run_async` method definition.**

**Step 2: Implement the Correct and Complete `run_as_node` Method**

Replace the existing `run_as_node` method with the following complete and corrected version. This new implementation properly inspects the incoming `AgentState` to determine the correct parsing path.

* In `src/agents/parser_agent.py`, **replace the existing `run_as_node` method with this code:**

```python
# In src/agents/parser_agent.py

# ... (keep all other methods like __init__, parse_job_description, parse_cv_text, etc.)

    async def run_as_node(self, state: AgentState) -> dict:
        """
        Executes the complete parsing logic as a LangGraph node.

        This method now correctly handles parsing the job description,
        parsing an existing CV text from metadata, or creating an empty CV
        structure based on the initial state, ensuring the agent's full
        capability is exposed to the workflow.
        """
        logger.info("ParserAgent node running with consolidated logic.")

        try:
            # Initialize job_data to None
            job_data = None

            # 1. Always parse the job description first, if it exists.
            if state.job_description_data and state.job_description_data.raw_text:
                logger.info("Starting job description parsing.")
                job_data = await self.parse_job_description(state.job_description_data.raw_text)
            else:
                logger.warning("No job description text found in the state.")
                # Create empty JobDescriptionData if none exists
                job_data = JobDescriptionData(raw_text="")

            # 2. Determine the CV processing path from the structured_cv metadata.
            # This metadata is set by the create_agent_state_from_ui function.
            cv_metadata = state.structured_cv.metadata if state.structured_cv else {}
            start_from_scratch = cv_metadata.get("start_from_scratch", False)
            original_cv_text = cv_metadata.get("original_cv_text", "")

            final_cv = None
            if start_from_scratch:
                logger.info("Creating empty CV structure for 'Start from Scratch' option.")
                final_cv = self.create_empty_cv_structure(job_data)
            elif original_cv_text:
                logger.info("Parsing provided CV text.")
                final_cv = self.parse_cv_text(original_cv_text, job_data)
            else:
                logger.warning("No CV text provided and not starting from scratch. Passing CV state through.")
                final_cv = state.structured_cv

            # 3. Return the complete, updated state.
            return {
                "structured_cv": final_cv,
                "job_description_data": job_data
            }

        except Exception as e:
            logger.error(f"Critical error in ParserAgent node: {e}", exc_info=True)
            error_list = state.get("error_messages", []) or []
            error_list.append(f"ParserAgent Error: {str(e)}")
            # Return the errors to be handled by the graph's error handling mechanism
            return {"error_messages": error_list}
```

**5. Testing Considerations**

* **Update Existing Tests:** All unit tests in `tests/unit/test_parser_agent.py` that currently call `run_async` must be updated to call `run_as_node`.
* **Create New Test Cases:** Add specific test cases to validate the corrected logic:
    * **Test Case 1 (Start from Scratch):**
        * Create an `AgentState` where `structured_cv.metadata['start_from_scratch']` is `True`.
        * Call `parser_agent.run_as_node(state)`.
        * Assert that the returned `structured_cv` is not empty and contains the default sections (e.g., "Executive Summary", "Professional Experience").
    * **Test Case 2 (Parse Existing CV):**
        * Create an `AgentState` where `structured_cv.metadata['original_cv_text']` contains sample CV markdown.
        * Call `parser_agent.run_as_node(state)`.
        * Assert that the returned `structured_cv.sections` list is populated with parsed data from the sample text.

---

Executing this blueprint will resolve the most critical bug in our current system. Once this is implemented and verified, we will have a reliable parsing foundation and can proceed with confidence to the `main.py` refactoring as previously planned.

---

Understood. We will now proceed.

You currently have the two essential blueprints required for the next phase of development:
1.  **Refactoring Task 10:** To fix the critical logic bug in `ParserAgent`.
2.  **The Refactoring Task for `main.py`:** To decouple the UI and improve maintainability.

The logical development sequence is to **first apply the fix to `ParserAgent`** and then, with a stable core, **implement the `main.py` refactoring.**

Once those foundational tasks are complete, we can move to the next architectural enhancement we discussed: improving the user experience by addressing the asynchronous execution in the Streamlit UI. I note the existence of `src/core/async_optimizer.py`, which suggests some preliminary work or thought has been given to this area. We can build upon that.

To ensure a seamless transition to the next stage of work, I can now prepare the formal blueprint for this future enhancement.

I will proceed with creating the blueprint for **Phase 3: Implementing a Non-Blocking Asynchronous UI.** This will provide a clear and actionable plan for the developer once the current refactoring is complete.

Understood. We will now proceed.

You currently have the two essential blueprints required for the next phase of development:
1.  **Refactoring Task 10:** To fix the critical logic bug in `ParserAgent`.
2.  **The Refactoring Task for `main.py`:** To decouple the UI and improve maintainability.

The logical development sequence is to **first apply the fix to `ParserAgent`** and then, with a stable core, **implement the `main.py` refactoring.**

Once those foundational tasks are complete, we can move to the next architectural enhancement we discussed: improving the user experience by addressing the asynchronous execution in the Streamlit UI. I note the existence of `src/core/async_optimizer.py`, which suggests some preliminary work or thought has been given to this area. We can build upon that.

To ensure a seamless transition to the next stage of work, I can now prepare the formal blueprint for this future enhancement.

I will proceed with creating the blueprint for **Phase 3: Implementing a Non-Blocking Asynchronous UI.** This will provide a clear and actionable plan for the developer once the current refactoring is complete.

---

Excellent question. The blueprints we have laid out will make the application stable, maintainable, and responsive. They represent a complete plan for the immediate, high-priority refactoring work.

Looking further ahead, as the system architect, there are several strategic areas we should consider to mature the `aicvgen` project from a successful MVP into a truly production-grade application. These are not immediate tasks, but the next logical items for our development roadmap.

### Future Architectural Roadmap

Here are three key areas I recommend we plan for after the current refactoring is complete:

**1. Implement a Comprehensive Observability Framework**
* **Why:** To proactively monitor the application's health, performance, and costs in a production environment. Right now, we have good logging, but we can make it much more powerful.
* **Action Plan:**
    * **Structured Logging:** Ensure all logs are structured (JSON format) and include a `trace_id` that persists across all agents and services for a single user request. This will make debugging complex workflows trivial.
    * **Metrics Dashboard:** Leverage the existing `performance_monitor.py` and `async_optimizer.py` statistics to export metrics (e.g., to Prometheus or a cloud provider's monitoring service). We can create a dashboard to track key indicators like average workflow duration, agent-specific processing times, LLM API error rates, and token consumption per user session.

**2. Harden the E2E Testing and Add Load Testing**
* **Why:** To ensure reliability as we add new features and to understand the application's limits under pressure.
* **Action Plan:**
    * **Automated E2E Testing:** Expand the existing `tests/e2e/` suite to use a tool like Playwright or Selenium to automate testing of the full user journey in the refactored UI. This will catch regressions before they reach users.
    * **Load Testing:** Create a load testing script (e.g., using Locust) to simulate multiple concurrent users generating CVs. This will allow us to validate the effectiveness of the `async_optimizer.py`, find performance bottlenecks, and determine the application's scaling limits.

**3. Formalize Data Security and Privacy Protocols**
* **Why:** The application handles Personally Identifiable Information (PII) from user CVs. We have a professional responsibility to handle this data with the highest level of security.
* **Action Plan:**
    * **PII Redaction Review:** Create a dedicated task to audit the entire data lifecycle. We should ensure sensitive information (like phone numbers, emails, addresses) is properly redacted by the `security_utils.py` `CredentialRedactor` before being stored in logs or used in non-essential LLM prompts.
    * **Data Retention and Backup Policy:** Formalize and automate a policy for backing up user session data and the ChromaDB vector store. We also need a policy for securely deleting aged or inactive user data to comply with privacy best practices.

These initiatives are the pillars of a scalable and professional service. Once the current refactoring is complete, I can create detailed technical blueprints for each of these roadmap items. For now, the focus should remain on executing the plans we have in place.

---

Of course. I will add the "Comprehensive Observability Framework" to our active plan by creating a detailed technical blueprint for it.

This is an excellent step for maturing the application. Based on our roadmap, this task should be prioritized after the critical `ParserAgent` bug is fixed and the `main.py` refactoring is complete. A stable, well-structured application is the perfect foundation upon which to build a robust observability layer.

Here is the new technical blueprint.

### **TASK_BLUEPRINT.md**

---

### **Enhancement Task: Implement a Comprehensive Observability Framework**

**1. Task/Feature Addressed**

This task involves instrumenting the application to provide deep insights into its health, performance, and costs. We will implement two key features:
1.  **Structured, Traceable Logging:** All log entries will be converted to a machine-readable JSON format, and each user workflow will be assigned a unique `trace_id` that is propagated through all logs, allowing for easy filtering and debugging of a single request's entire lifecycle.
2.  **Metrics Exporting:** Key performance indicators (KPIs) from the `performance_monitor` and `async_optimizer`, along with business metrics like token usage, will be exported to a `/metrics` endpoint, making them scrapable by a monitoring system like Prometheus.

**2. Affected Component(s)**

* **`requirements.txt`**: To add new dependencies.
* **`src/orchestration/state.py`**: The `AgentState` model will be updated to carry the `trace_id`.
* **`src/config/logging_config.py`**: The logging setup will be reconfigured for JSON output.
* **`src/core/main.py`**: Will be modified to generate the `trace_id`.
* **`src/agents/*.py` (All agents)**: Will require a minor modification to include the `trace_id` in log records.
* **`src/services/metrics_exporter.py` (New File)**: A new module to define and manage Prometheus metrics.
* **`src/api/app_main.py`**: Will be updated to expose the `/metrics` endpoint.

**3. Pydantic Model Changes**

* In `src/orchestration/state.py`, add a `trace_id` field to the `AgentState` class:

    ```python
    # In src/orchestration/state.py
    from typing import Optional
    import uuid

    class AgentState(TypedDict):
        # ... existing fields ...
        trace_id: Optional[str]
    ```

**4. Detailed Implementation Steps**

**Step 1: Update Dependencies**

* Add the necessary libraries for structured logging and metrics exporting to `requirements.txt`:

    ```text
    # In requirements.txt
    ...
    python-json-logger
    prometheus-client
    ...
    ```

**Step 2: Implement Structured JSON Logging**

1.  **Configure JSON Formatter:** Modify `src/config/logging_config.py` to use `jsonlogger`.

    ```python
    # In src/config/logging_config.py
    import logging
    from pythonjsonlogger import jsonlogger

    def setup_logging():
        logger = logging.getLogger("aicvgen")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            logHandler = logging.StreamHandler()
            # The format string includes all standard log attributes plus our custom trace_id
            formatter = jsonlogger.JsonFormatter(
                '%(asctime)s %(name)s %(levelname)s %(message)s %(trace_id)s'
            )
            logHandler.setFormatter(formatter)
            logger.addHandler(logHandler)
        return logger
    ```

2.  **Generate `trace_id` at Workflow Start:** In `src/core/main.py`, modify the `create_agent_state_from_ui` function to generate and include the `trace_id`.

    ```python
    # In src/core/main.py
    import uuid

    def create_agent_state_from_ui() -> AgentState:
        # ... existing logic ...
        initial_state = AgentState(
            # ... existing fields ...
            trace_id=str(uuid.uuid4()) # Generate unique ID for this workflow run
        )
        return initial_state
    ```

3.  **Pass `trace_id` to Logs:** Update agents to pass the `trace_id` from the state into their log calls using the `extra` dictionary.

    ```python
    # Example in any agent, e.g., src/agents/parser_agent.py
    # Inside the run_as_node method:

    logger.info(
        "ParserAgent node running.",
        extra={'trace_id': state.get('trace_id')}
    )
    ```

**Step 3: Implement Metrics Exporting**

1.  **Create Metrics Exporter Module:** This module will define the metrics we want to track.

    * **Create `src/services/metrics_exporter.py`:**
    ```python
    # In src/services/metrics_exporter.py
    from prometheus_client import Counter, Histogram

    # --- Workflow Metrics ---
    WORKFLOW_DURATION_SECONDS = Histogram(
        'aicvgen_workflow_duration_seconds',
        'Histogram of CV generation workflow durations.'
    )
    WORKFLOW_ERRORS_TOTAL = Counter(
        'aicvgen_workflow_errors_total',
        'Total number of failed CV generation workflows.'
    )

    # --- LLM Metrics ---
    LLM_TOKEN_USAGE_TOTAL = Counter(
        'aicvgen_llm_tokens_total',
        'Total number of LLM tokens used.',
        ['model_name'] # Label to differentiate by model
    )
    ```

2.  **Expose `/metrics` Endpoint:** Modify the FastAPI application to serve the Prometheus metrics.

    * **In `src/api/app_main.py`:**
    ```python
    # In src/api/app_main.py
    from fastapi import FastAPI
    from starlette.responses import Response
    import prometheus_client

    app = FastAPI()

    # ... your existing API routes ...

    @app.get("/metrics")
    def get_metrics():
        """Prometheus metrics endpoint."""
        return Response(
            media_type="text/plain",
            content=prometheus_client.generate_latest()
        )
    ```

3.  **Instrument the Code:** Call the metric update functions from relevant places in the code.

    * **In `src/core/main.py` (or your workflow runner):**
    ```python
    # When a workflow finishes
    from src.services import metrics_exporter
    import time

    start_time = time.time()
    # ... run workflow ...
    duration = time.time() - start_time
    metrics_exporter.WORKFLOW_DURATION_SECONDS.observe(duration)

    # If workflow fails:
    metrics_exporter.WORKFLOW_ERRORS_TOTAL.inc()
    ```
    * **In `src/services/llm_service.py`:**
    ```python
    # When an LLM call is made and you get the token count from the response
    from src.services import metrics_exporter

    # ... after llm.generate_content() ...
    token_count = response.usage_metadata.total_tokens # Example attribute
    metrics_exporter.LLM_TOKEN_USAGE_TOTAL.labels(model_name=self.model_name).inc(token_count)
    ```

**5. Testing Considerations**

* **Unit Tests:**
    * Write a test for the `logging_config` to confirm that log output is valid JSON and contains the expected fields.
    * Write tests for the `metrics_exporter` to ensure that calling `.inc()` and `.observe()` correctly updates the metric values.
* **Integration Tests:**
    * Create an integration test that runs a mini-workflow and captures the log output. Assert that the `trace_id` is present and consistent across logs from different agents.
    * Write a test that calls the `/metrics` endpoint of the test API and asserts that the metric values have changed after running a workflow.