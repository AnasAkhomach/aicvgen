---

Of course. We will now proceed to the Priority 3 tasks, which are focused on improving the long-term health and maintainability of the codebase. Here is the blueprint for the first of these tasks.

### **TASK\_BLUEPRINT.md**

---

### **Refactoring Task 6: Refactor Hardcoded Configurations**

**1. Task/Feature Addressed**

Throughout the codebase, there are hardcoded values such as LLM model names, prompt file names, and other magic strings. This practice makes the system rigid and difficult to configure. For example, changing the LLM model for an agent requires finding and modifying its source code directly.

This refactoring task will centralize these values into the existing Pydantic-based configuration system (`src/config/settings.py`), making the application more flexible, easier to maintain, and configurable without code changes.

**2. Affected Component(s)**

* **`src/config/settings.py`**: This file will be updated with new Pydantic models to hold centralized configurations for LLMs and prompts.
* **`src/agents/*.py`**: All agents will be refactored to read configuration values from the central settings object instead of using hardcoded strings. We will use `ParserAgent` as a concrete example.
* **`src/services/llm_service.py`**: This service will be updated to use the model name and parameters from the configuration.

**3. Pydantic Model Changes**

Yes, the models within `src/config/settings.py` will be expanded to create a structured and validated configuration schema.

* **Modify `src/config/settings.py` to include new models:**

```python
# In src/config/settings.py
from pydantic import BaseModel, Field, DirectoryPath
from typing import Dict, List, Optional
import os

# ... other existing imports

class LLMSettings(BaseModel):
    """Configuration for Large Language Model services."""
    default_model: str = "gemini-1.5-flash"
    default_temperature: float = 0.7
    max_tokens: int = 4096

class PromptSettings(BaseModel):
    """Configuration for prompt templates, mapping a key to a filename."""
    job_description_parser: str = "job_description_parsing_prompt.md"
    resume_role_writer: str = "resume_role_prompt.md"
    project_writer: str = "side_project_prompt.md"
    key_qualifications_writer: str = "key_qualifications_prompt.md"
    executive_summary_writer: str = "executive_summary_prompt.md"
    # ... add other prompts as needed

class Settings(BaseModel):
    """Main application settings."""
    app_name: str = "AI CV Generator"
    # ... existing settings ...

    # NEW: Add structured configuration sections
    llm: LLMSettings = Field(default_factory=LLMSettings)
    prompts: PromptSettings = Field(default_factory=PromptSettings)

    # Path to the prompts directory
    prompts_dir: DirectoryPath = Field(default_factory=lambda: os.path.join(os.path.dirname(__file__), "..", "data", "prompts"))

    def get_prompt_path(self, prompt_key: str) -> str:
        """
        Constructs the full path to a prompt file using a key from PromptSettings.
        """
        prompt_filename = getattr(self.prompts, prompt_key, None)
        if not prompt_filename:
            raise ValueError(f"Prompt key '{prompt_key}' not found in settings.")

        path = os.path.join(self.prompts_dir, prompt_filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt file does not exist at path: {path}")
        return path

# ... existing get_config() function ...
```

**4. Detailed Implementation Steps**

**Step 1: Refactor the `LLMService` to Use Central Config**

The service that directly interacts with the LLM API should be the first to be updated.

* **Modify `src/services/llm_service.py`:**

```python
# In src/services/llm_service.py
from src.config.settings import get_config

class LLMService:
    def __init__(self):
        # The service now reads its parameters from the central config
        self.settings = get_config()
        self.model_name = self.settings.llm.default_model
        # ... initialize the generativeai client ...

    async def generate_content(self, prompt: str, temperature: Optional[float] = None) -> LLMResponse:
        # Uses model_name and other parameters from the config
        generation_config = GenerationConfig(
            temperature=temperature if temperature is not None else self.settings.llm.default_temperature,
            max_output_tokens=self.settings.llm.max_tokens,
        )
        # ... logic to call the LLM API using self.model_name and generation_config ...
```

**Step 2: Refactor the `ParserAgent` to Remove Hardcoded Prompt Key**

The agent will now use the new `get_prompt_path` helper, which relies on the centralized `PromptSettings`.

* **Modify `src/agents/parser_agent.py`:**

```python
# In src/agents/parser_agent.py
from src.config.settings import get_config

class ParserAgent(EnhancedAgentBase):
    def __init__(self, name: str, description: str, llm_service=None):
        self.name = name
        self.description = description
        self.llm = llm_service or get_llm_service()
        # The agent gets a handle to the global config
        self.settings = get_config()

    async def parse_job_description(self, raw_text: str) -> JobDescriptionData:
        # ...
        try:
            # --- BEFORE (Illustrative) ---
            # prompt_path = self.settings.get_prompt_path("job_description_parsing_prompt") # Key is hardcoded

            # --- AFTER ---
            # The agent now uses the config to find the correct prompt path for its purpose.
            # This decouples the agent's logic from the specific filename.
            prompt_path = self.settings.get_prompt_path("job_description_parser")

            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
            prompt = prompt_template.format(raw_text=raw_text)

            # ... rest of the method ...

```

**Step 3: Apply This Pattern to All Other Agents**

Systematically go through all other agent classes (`EnhancedContentWriterAgent`, etc.) and service files.
1.  Find any hardcoded string, path, or parameter.
2.  If it's a configurable value, add a corresponding field to the appropriate model in `src/config/settings.py`.
3.  Refactor the agent/service to read that value from the `get_config()` instance instead of using the hardcoded value.

**5. Testing Considerations**

* **Updating Existing Tests:** Unit tests that mock agents or services may need to be updated. If a test previously expected a hardcoded value to be used, it must now be adjusted to mock the `get_config()` call and provide the expected value through the mocked settings object.
* **Configuration Testing:** It would be beneficial to add a new test file, `tests/unit/test_settings.py`.
    * **Test Case 1 (Defaults):** Test that calling `get_config()` without any environment variables loads the default values correctly (e.g., `assert settings.llm.default_model == "gemini-1.5-flash"`).
    * **Test Case 2 (Environment Override):** Write a test that sets an environment variable (e.g., `os.environ['AICVGEN_LLM_DEFAULT_MODEL'] = 'gemini-pro'`) before calling `get_config()` and asserts that the setting is correctly overridden. (Note: Pydantic V2 automatically handles environment variable overrides if prefixed correctly).

---
Completing this refactoring will make the `aicvgen` application significantly more professional and adaptable. It separates the "what" (configuration) from the "how" (code logic), which is a cornerstone of good software design.

We have one final task in our plan: **"Isolate and Test the PDF Generation Pipeline."** Please confirm, and I will create the final blueprint.

---

Of course. Here is the final blueprint in our comprehensive refactoring plan. This task will provide a significant boost to your development and debugging workflow for the PDF output.

### **TASK\_BLUEPRINT.md**

---

### **Refactoring Task 7: Isolate and Test the PDF Generation Pipeline**

**1. Task/Feature Addressed**

Currently, debugging the final PDF output (e.g., fixing a layout issue, changing a font, or adjusting template logic) is a slow and cumbersome process. It requires running the entire, time-consuming agentic workflow just to generate a single PDF.

This task involves creating a dedicated, isolated test script (a "test harness") that decouples the final formatting and rendering step from the main workflow. This will allow for rapid, iterative development and debugging of the PDF template and styles.

**2. Affected Component(s)**

* **`tests/integration/test_pdf_pipeline.py`**: A new test file will be created to house the isolated test script.
* **`tests/e2e/test_data/`**: We may create a helper file here to store a rich, sample `StructuredCV` object for testing, or construct it directly in the test script.
* **`src/agents/formatter_agent.py`**: The logic of this agent will be directly invoked by our new test script.

**3. Pydantic Model Changes**

No changes to any Pydantic models are required for this task. We will be using the existing `StructuredCV` and `AgentState` models.

**4. Detailed Implementation Steps**

This task is focused on creating a new development utility in the form of a test script.

**Step 1: Create a Sample `StructuredCV` Data Factory**

To test the pipeline, we need realistic data. We will create a helper function that generates a rich `StructuredCV` object on demand. This can live directly in the new test file or in a shared test data module.

* **Create a new test file: `tests/integration/test_pdf_pipeline.py`**
* **Add the following code to the new file:**

```python
# In tests/integration/test_pdf_pipeline.py
import pytest
from src.models.data_models import StructuredCV, Section, Subsection, Item, ItemType, ItemStatus
from src.orchestration.state import AgentState
from src.agents.formatter_agent import FormatterAgent
# You may need to mock or import other dependencies for agent instantiation
from src.services.llm_service import LLMService # Assuming it's a dependency

def create_sample_structured_cv() -> StructuredCV:
    """Creates a rich, sample StructuredCV object for testing the PDF pipeline."""
    cv = StructuredCV(
        metadata={
            "name": "John 'Johnny' Doe",
            "email": "john.doe@email.com",
            "phone": "+1 (555) 123-4567",
            "linkedin": "linkedin.com/in/johndoe-dev",
            "github": "github.com/johndoe-dev"
        }
    )

    # Executive Summary Section
    summary_section = Section(name="Executive Summary", content_type="DYNAMIC", order=0)
    summary_section.items.append(Item(content="A results-oriented developer with 5+ years of experience in Python & Cloud services, specializing in building scalable AI applications. Eager to tackle new challenges in data-driven environments."))
    cv.sections.append(summary_section)

    # Professional Experience Section
    exp_section = Section(name="Professional Experience", content_type="DYNAMIC", order=1)
    exp1 = Subsection(name="Senior AI Engineer at TechCorp Inc. (Jan 2022 - Present)")
    exp1.items.append(Item(content="Led the 'Project_Phoenix' team, delivering a 30% increase in processing efficiency."))
    exp1.items.append(Item(content="Developed a new API for R&D, handling 100% of internal data requests with >99.9% uptime."))
    exp_section.subsections.append(exp1)
    cv.sections.append(exp_section)

    # Key Qualifications Section with special characters
    qual_section = Section(name="Key Qualifications", content_type="DYNAMIC", order=2)
    qual_section.items.append(Item(content="Python & Django"))
    qual_section.items.append(Item(content="C# Programming"))
    qual_section.items.append(Item(content="Cloud Services (AWS & Azure)"))
    qual_section.items.append(Item(content="Data Analysis (Pandas & SQL)"))
    cv.sections.append(qual_section)

    return cv
```

**Step 2: Create the PDF Pipeline Test Harness**

Now, add the test function that uses this sample data to run *only* the `FormatterAgent`.

* **Add the following test function to `tests/integration/test_pdf_pipeline.py`:**

```python
# In tests/integration/test_pdf_pipeline.py
import os
import asyncio

@pytest.mark.asyncio
async def test_run_pdf_generation_pipeline_in_isolation():
    """
    This test runs the FormatterAgent in isolation to rapidly generate a PDF.
    It's designed as a utility for quick visual debugging of the PDF template.
    """
    # 1. ARRANGE: Create dependencies and initial state
    # Mock dependencies if they make network calls or are not needed for formatting
    mock_llm_service = LLMService() # Or a proper mock
    formatter_agent = FormatterAgent(
        name="FormatterAgent",
        description="Formats the final CV.",
        llm_service=mock_llm_service
    )

    sample_cv = create_sample_structured_cv()
    initial_state = AgentState(structured_cv=sample_cv)

    # 2. ACT: Run only the formatter agent's node logic
    result_state = await formatter_agent.run_as_node(initial_state)

    # 3. ASSERT: Check for success
    assert "error_messages" not in result_state or not result_state["error_messages"]
    assert "final_output_path" in result_state

    output_path = result_state["final_output_path"]
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 1000  # Assert file is not empty

    print(f"\nPDF successfully generated at: {output_path}")
    print("This file can now be opened for visual inspection.")

    # Optional: Automatically open the file for immediate visual feedback
    # Note: This part is for developer convenience and might be commented out
    # in a CI/CD environment.
    # import subprocess, sys
    # if sys.platform == "win32":
    #     os.startfile(output_path)
    # else:
    #     subprocess.call(["open", output_path])

```

**Step 3: Run the Test Harness**

You can now run this specific test from your terminal whenever you need to check the PDF output.

```bash
# Example using pytest
pytest tests/integration/test_pdf_pipeline.py
```

**5. Testing Considerations**

* **This script *is* the test.** Its primary purpose is to serve as a development utility. It provides an immediate feedback loop for visual changes.
* **CI/CD Environments:** The part of the script that automatically opens the generated PDF should be commented out or placed behind a flag when running in a non-interactive environment like a GitHub Action.
* **Data Variations:** You can easily create multiple data factory functions (e.g., `create_cv_with_many_sections()`, `create_cv_with_no_experience()`) to test how the PDF template handles different edge cases.

---
By implementing this test harness, you have concluded our comprehensive refactoring plan. You now have a more robust, reliable, and maintainable system, along with powerful tools for future development and debugging. Congratulations on completing this architectural review and planning phase.

---

Of course. It is an excellent idea to be thorough and ensure nothing is left to chance. Creating a specific blueprint for **TD-008** will transform the general policy from Blueprint 2 into a concrete, verifiable cleanup and refactoring task.

This also allows us to address a subtle but critical risk discovered in the `parser_agent.py` file: the standardized `run_as_node` method is actually *less complete* than the legacy `run` method, meaning the graph is not currently using the agent to its full potential. This blueprint will rectify that.

Here is the detailed blueprint to address inconsistent agent method naming.

### **TASK\_BLUEPRINT.md**

---

### **Refactoring Task 8: Consolidate Agent Execution Methods**

**1. Task/Feature Addressed**

This task directly addresses the technical debt item **TD-008**. The audit identified that agent classes contain multiple, inconsistently named execution methods (e.g., `run`, `run_async`, `_legacy_run_implementation`, `run_as_node`). This creates confusion and, more critically, has led to a situation where the official `run_as_node` method for the `ParserAgent` contains less functionality than its legacy counterparts.

The goal is to refactor all agents to have a single, canonical `run_as_node` method that contains the complete and correct logic for that agent's role in the graph. All legacy methods will be removed to eliminate ambiguity and ensure the workflow is using the agents as intended.

**2. Affected Component(s)**

* **`src/agents/parser_agent.py`**: This is the primary target and example, as it contains all four legacy and current method names.
* **`src/agents/*`**: A systematic review of all other agent files is required to apply the same consolidation pattern.

**3. Pydantic Model Changes**

No changes to any Pydantic models are required for this task.

**4. Detailed Implementation Steps**

**Step 1: Full Logic Consolidation in `ParserAgent`**

We will refactor `ParserAgent` to ensure its `run_as_node` method correctly implements the full logic present in the legacy `run` method, which includes parsing both the job description and the CV text.

* **Analysis of `src/agents/parser_agent.py`:**
    * `run()` and `_legacy_run_implementation()`: Contain the complete logic for parsing `job_description`, `cv_text`, or handling `start_from_scratch`.
    * `run_as_node()`: The current implementation *only* parses the job description and passes the `structured_cv` through untouched. **This is a critical bug.**
    * `run_async()`: Appears to be a previous iteration of an async interface.

* **Refactor `src/agents/parser_agent.py`:**

    **BEFORE (Conceptual Structure):**
    ```python
    class ParserAgent(EnhancedAgentBase):
        # ... init ...
        async def run(self, input: dict) -> Dict[str, Any]:
            # This method has the FULL logic for JD, CV, and scratch...
            pass

        async def _legacy_run_implementation(self, input: dict) -> Dict[str, Any]:
            # Redundant copy of run()
            pass

        async def run_async(self, input_data: Any, context: 'AgentExecutionContext') -> 'AgentResult':
            # An older async implementation...
            pass

        async def run_as_node(self, state: AgentState) -> dict:
            # INCOMPLETE LOGIC! Only handles job_description.
            logger.info("ParserAgent node running.")
            job_data = await self.parse_job_description(state.job_description_data.raw_text)
            return {
                "structured_cv": state.structured_cv, # Passes CV through without parsing!
                "job_description_data": job_data
            }
    ```

    **AFTER (Consolidated and Corrected Logic):**
    ```python
    # In src/agents/parser_agent.py
    # Delete the old run(), _legacy_run_implementation(), and run_async() methods.
    # Replace them with this single, complete implementation.

    class ParserAgent(EnhancedAgentBase):
        # ... __init__ and other methods like parse_job_description remain ...

        async def run_as_node(self, state: AgentState) -> dict:
            """
            Executes the complete parsing logic as a LangGraph node.

            This method now correctly handles parsing the job description,
            parsing an existing CV text, or creating an empty CV structure,
            ensuring the agent's full capability is exposed to the workflow.
            """
            logger.info("ParserAgent node running with consolidated logic.")

            try:
                # 1. Parse job description (if text exists)
                job_desc_text = state.job_description_data.raw_text
                job_data = await self.parse_job_description(job_desc_text)

                # 2. Determine CV processing path from the state
                cv_text = state.structured_cv.metadata.get("original_cv_text", "")
                from_scratch = state.structured_cv.metadata.get("start_from_scratch", False)

                final_cv = None
                if cv_text:
                    # Parse the provided CV text
                    logger.info("Parsing CV text...")
                    final_cv = self.parse_cv_text(cv_text, job_data)
                elif from_scratch:
                    # Create an empty CV structure
                    logger.info("Creating empty CV structure from scratch...")
                    final_cv = self.create_empty_cv_structure(job_data)
                else:
                    # Pass through the CV if no action is specified
                    final_cv = state.structured_cv

                return {
                    "structured_cv": final_cv,
                    "job_description_data": job_data
                }

            except Exception as e:
                logger.error(f"Error in ParserAgent node: {e}", exc_info=True)
                error_list = state.get("error_messages", [])
                error_list.append(f"ParserAgent Error: {e}")
                return {"error_messages": error_list}
    ```

**Step 2: Audit and Refactor All Other Agents**

Systematically apply the consolidation principle to every other agent in `src/agents/`.
1.  **Search:** For each agent file, search for method definitions: `run`, `run_async`, `_legacy`.
2.  **Analyze:** Determine if the logic in these legacy methods is redundant or if it contains functionality missing from the agent's `run_as_node` method.
3.  **Consolidate:** Merge any necessary logic into the `run_as_node` method.
4.  **Delete:** Remove the now-redundant legacy methods (`run`, `run_async`, etc.). The only public-facing execution method for the graph should be `run_as_node`.

**5. Testing Considerations**

* **Update Unit Tests:** The unit tests for each agent, especially `tests/unit/test_parser_agent.py`, must be thoroughly reviewed and updated.
    * All test cases must be modified to call `agent.run_as_node(test_state)` instead of the legacy methods.
    * **Crucially, new test cases must be added for the `ParserAgent`** to verify the newly consolidated logic:
        * A test case where the input `AgentState` contains `cv_text`, asserting that `parse_cv_text` is called and the returned `structured_cv` is correctly parsed.
        * A test case where the input `AgentState` has `start_from_scratch=True`, asserting that `create_empty_cv_structure` is called.
* **Regression Testing:** After refactoring all agents, run the full integration test suite to ensure that consolidating the logic has not introduced any regressions in the end-to-end workflow.

---
This blueprint ensures that the inconsistent naming is resolved and, more importantly, corrects a critical bug where the `ParserAgent` was not performing its full duties within the main workflow. This leaves nothing to chance and makes the system's behavior far more predictable.