# Refactoring Plan: Agent Architecture with LCEL

## 1. Objective

This report details a concrete plan to refactor the project's agent architecture, moving from a manual, imperative style to a declarative approach using the **LangChain Expression Language (LCEL)**. This change will significantly reduce boilerplate code, improve maintainability, and align the project more closely with modern `LangChain` best practices.

We will use the `KeyQualificationsWriterAgent` as a representative example for this refactoring.

## 2. Analysis of the Current `KeyQualificationsWriterAgent`

The current implementation in `src/agents/key_qualifications_writer_agent.py` follows a manual process:

1.  **Input Validation:** A dedicated `_validate_inputs` method checks the incoming data.
2.  **Manual Prompt Construction:** It fetches a template string from `ContentTemplateManager` and then manually formats it with the required data.
3.  **Direct LLM Service Call:** It directly calls the `llm_service.generate_content` method.
4.  **Manual Output Parsing:** It takes the raw string output from the LLM and manually splits it by newlines and strips characters to create a list of qualifications.
5.  **Complex `_execute` Method:** The core logic is contained within a lengthy `_execute` method with extensive error handling.

This approach, while functional, is verbose and tightly couples the agent to the specific implementations of the template and LLM services.

## 3. Proposed Refactoring with LCEL

The agent can be re-implemented as a declarative chain using LCEL. This involves three main components: a Prompt Template, the Model, and an Output Parser.

### 3.1. Step 1: Define the Prompt Template

Instead of manually loading and formatting a string, we will use `ChatPromptTemplate`. This object is more structured and integrates directly into the LCEL chain.

```python
# In a new file, e.g., src/agents/prompts/key_qualifications_prompts.py

from langchain_core.prompts import ChatPromptTemplate

KEY_QUALIFICATIONS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert CV writer. Your task is to generate a list of key qualifications based on the provided job description and a summary of the candidate's CV. Focus on aligning the candidate's skills with the job requirements. Return a list of qualifications, with each qualification on a new line.",
        ),
        (
            "human",
            """
            **Job Description:**
            {main_job_description_raw}

            **Candidate's CV Summary:**
            {my_talents}
            """,
        ),
    ]
)
```

### 3.2. Step 2: Define the Output Parser

To eliminate manual string splitting, we'll use `StrOutputParser` combined with a simple function to handle the list conversion. For more complex outputs, a `PydanticOutputParser` or `JsonOutputParser` could be used.

```python
# In the refactored agent file or a shared parser module

from langchain_core.output_parsers import StrOutputParser

def parse_qualifications_list(text: str) -> list[str]:
    """Parses newline-separated qualifications into a list."""
    return [line.strip() for line in text.strip().split('\n') if line.strip()]

# The chain will look like: ... | StrOutputParser() | parse_qualifications_list
```

### 3.3. Step 3: Rebuild the Agent with the LCEL Chain

The `KeyQualificationsWriterAgent` class will be significantly simplified. The core logic will be encapsulated in an LCEL chain.

```python
# In the refactored src/agents/key_qualifications_writer_agent.py

from typing import Any, Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI # Or any other LangChain model
from langchain_core.runnables import Runnable

from src.agents.agent_base import AgentBase
from src.error_handling.exceptions import AgentExecutionError
# Import the new prompt and parser function
from .prompts.key_qualifications_prompts import KEY_QUALIFICATIONS_PROMPT
from .parsers import parse_qualifications_list # Assuming a new parser module

class KeyQualificationsWriterAgent(AgentBase):
    """Agent for generating tailored Key Qualifications content using LCEL."""

    def __init__(self, settings: dict, session_id: str):
        super().__init__(
            name="KeyQualificationsWriter",
            description="Generates tailored Key Qualifications for the CV.",
            session_id=session_id,
            settings=settings,
        )
        # The agent now directly holds the LCEL chain
        self.chain = self._build_chain()

    def _build_chain(self) -> Runnable:
        """Builds the LCEL chain for the agent."""
        llm = ChatGoogleGenerativeAI(
            model=self.settings.get("model_name", "gemini-pro"),
            temperature=self.settings.get("temperature_content_generation", 0.7),
        )
        
        # The entire logic is now a declarative chain
        chain = (
            KEY_QUALIFICATIONS_PROMPT
            | llm
            | StrOutputParser()
            | parse_qualifications_list
        )
        return chain

    async def _execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes the LCEL chain with inputs from the state.
        Returns a dictionary compatible with the LangGraph state.
        """
        self.update_progress(10, "Generating Key Qualifications with LCEL chain.")
        
        try:
            # The core logic is now a single, clean invocation
            generated_qualifications = await self.chain.ainvoke(kwargs)
            
            self.update_progress(100, "LCEL chain execution complete.")
            
            # Return the result in the format expected by the graph
            return {
                "generated_key_qualifications": generated_qualifications,
                "current_item_id": "key_qualifications_section"
            }
        except Exception as e:
            logger.error(f"LCEL chain invocation failed in {self.name}: {e}", exc_info=True)
            raise AgentExecutionError(agent_name=self.name, message=str(e)) from e

```

## 4. Benefits of the New Approach

*   **Simplicity & Readability:** The agent's logic is now expressed as a clear, declarative chain. The complex, multi-step process in the original `_execute` method is replaced by a single `chain.ainvoke()` call.
*   **Reduced Boilerplate:** All manual prompt formatting and output parsing code is eliminated.
*   **Maintainability:** Modifying the prompt, the model, or the output format is now a matter of changing a single component in the chain, rather than rewriting imperative code.
*   **`LangChain` Integration:** The agent is now a native `LangChain` "Runnable." This makes it trivial to add other `LangChain` features like streaming, batching, and more complex routing directly to the chain.
*   **Testability:** The chain can be tested independently of the agent, and each component (prompt, model, parser) can be unit-tested separately.

## 5. Integration with `LangGraph`

This refactoring is fully compatible with the existing `LangGraph` workflow. The `AgentBase.run_as_node` method will continue to work as before, as it simply extracts inputs from the `AgentState` and passes them as `**kwargs` to the agent's `_execute` method. The refactored agent still returns a dictionary, which `LangGraph` will use to update the state.

The key change is internal to the agent, replacing complex imperative code with a simple, powerful, and maintainable LCEL chain.

## 6. Bonus: Adding LangSmith for Observability

The current architecture is perfectly suited for LangSmith, LangChain's native observability platform. Integrating it requires minimal effort and provides invaluable debugging and tracing capabilities for your `LangGraph` workflows.

### Step 1: Configure Environment Variables

Add the following variables to your `.env` file. This is the only step required to enable tracing.

```bash
# =============================================================================
# LANGSMITH OBSERVABILITY
# =============================================================================

# Set to "true" to enable LangSmith tracing
export LANGCHAIN_TRACING_V2="true"

# Your LangSmith API key (get from https://smith.langchain.com/settings)
export LANGCHAIN_API_KEY="YOUR_LANGSMITH_API_KEY_HERE"

# (Optional) Name of the project in LangSmith to group your runs
export LANGCHAIN_PROJECT="aicvgen-dev"

# (Optional) The API endpoint for LangSmith
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
```

### Step 2: Run and Observe

No code changes are necessary. Once the environment variables are set, `LangGraph` will automatically detect them and begin sending detailed traces of your graph executions to your LangSmith project. You will be able to visualize the entire workflow, inspect the inputs and outputs of each node, and debug performance issues.

### Step 3 (Recommended): Enhance Traces with Metadata

To make traces easier to filter and analyze, you can add custom metadata to your graph invocations. Your existing code structure makes this simple.

**Example in `cv_workflow_graph.py`:**

```python
# Inside a method like trigger_workflow_step or invoke

config = {
    "configurable": {
        "thread_id": self.session_id,
        # Add any other metadata for better filtering in LangSmith
        "user_id": "some_user_identifier", # If you have user management
        "run_type": "cv_generation_pass_1",
    }
}

# When you call the graph
stream = self.app.astream(state_dict, config=config)
```
The `thread_id` you already use is excellent for grouping steps within a session. Adding more context like `user_id` or `run_type` can further improve your debugging capabilities.