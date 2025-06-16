# **TASK\_BLUEPRINT.md**

## **Task/Feature Addressed: Implement Granular, Item-by-Item Processing Workflow**

This technical blueprint provides the definitive specification for refactoring the anasakhomach-aicvgen application's core workflow. The primary objective is to transition from a monolithic, one-shot generation process to a dynamic, iterative, and stateful system managed by `LangGraph`. This will implement the core user-in-the-loop experience for the Minimum Viable Product (MVP), enabling users to review and regenerate individual components of their CV.

This plan directly addresses the functional requirements for granular control over "Professional Experience" and "Project Experience" sections (REQ-FUNC-GEN-3, REQ-FUNC-GEN-4) and the user interface requirement for hybrid control (REQ-FUNC-UI-2) as outlined in the project's governing Software Requirements Specification.

### **Overall Technical Strategy**

The existing system architecture is prepared for this functional evolution. The current `EnhancedOrchestrator` will be simplified to a thin wrapper, with the core orchestration logic being implemented within the `LangGraph` application defined in `src/orchestration/cv_workflow_graph.py`.

The architectural pattern for this implementation will be a **Stateful Backend, Stateless Frontend**. This model is crucial for integrating a persistent state machine like LangGraph with a framework like Streamlit, which reruns its script on every user interaction.

1.  **Backend (LangGraph):** The `cv_graph_app` will be implemented as a stateful, long-running state machine. It will be the single source of truth for all business logic, agent execution, and state transitions. The state will be explicitly managed via the `AgentState` Pydantic model.
2.  **Frontend (Streamlit):** The Streamlit UI will function as a pure "view" layer. It will be completely stateless between user interactions. Its primary responsibility is to render the current `AgentState` and capture user actions.
3.  **The Interaction Loop:** The integration will be governed by a strict, predictable loop:
    *   **State In:** The Streamlit UI renders based on the current `AgentState` stored in `st.session_state`.
    *   **UI Action:** The user clicks a button (e.g., "Accept" or "Regenerate"). The button's `on_click` callback populates the `user_feedback` field within the `AgentState` object in `st.session_state`.
    *   **State Out:** The main script loop detects the presence of `user_feedback`, invokes the LangGraph application (`cv_graph_app.ainvoke`) with the entire current `AgentState`, and receives a new, complete `AgentState` object as the result.
    *   **Re-render:** The script overwrites the old state in `st.session_state` with the new one and calls `st.rerun()`, causing the UI to redraw itself based on the updated state.

This "State In -> UI Action -> State Out -> Re-render" pattern is the core architectural solution for managing state with Streamlit and making the data flow explicit and unidirectional.

### **Component-Level Implementation Plan**

This section details the required modifications for each affected component.

#### **1. Pydantic Model & State Management Changes**

This task formalizes the data contracts for UI-to-backend communication and adapts the central state model to manage the iterative workflow.

*   **Task/Feature Addressed:** Define data contracts for user interaction and update the central state model for iterative processing.
*   **Affected Component(s):** `src/models/data_models.py`, `src/orchestration/state.py`.
*   **Pydantic Model Changes (`src/models/data_models.py`):**
    *   The following models must be added to create a strict, validated API contract between the Streamlit UI and the LangGraph backend.

    ```python
    from enum import Enum
    from typing import Optional
    from pydantic import BaseModel

    class UserAction(str, Enum):
        """Enumeration for user actions in the UI."""
        ACCEPT = "accept"
        REGENERATE = "regenerate"

    class UserFeedback(BaseModel):
        """User feedback for item review."""
        action: UserAction
        item_id: str
        feedback_text: Optional[str] = None
    ```

*   **AgentState Refactoring (`src/orchestration/state.py`):**
    *   The `AgentState` model must be refactored to its definitive form to manage the granular workflow. The existing `AgentState` will be updated to include the following fields:

    ```python
    from typing import Any, Dict, List, Optional
    from pydantic import BaseModel, Field
    from src.models.data_models import JobDescriptionData, StructuredCV, UserFeedback

    class AgentState(BaseModel):
        """
        Represents the complete, centralized state of the CV generation workflow
        for LangGraph orchestration.
        """
        # Core Data Models
        structured_cv: StructuredCV
        job_description_data: JobDescriptionData

        # Workflow Control for Granular Processing
        current_section_key: Optional[str] = None
        items_to_process_queue: List[str] = Field(default_factory=list)
        current_item_id: Optional[str] = None
        is_initial_generation: bool = True

        # User Feedback for Regeneration
        user_feedback: Optional[UserFeedback] = None

        # Agent Outputs & Finalization
        research_findings: Optional[Dict] = None
        final_output_path: Optional[str] = None
        error_messages: List[str] = Field(default_factory=list)

        class Config:
            arbitrary_types_allowed = True
    ```

#### **2. Agent Logic Modifications: Granular Processing**

This task refactors the `EnhancedContentWriterAgent` to align with the Single Responsibility Principle, making it more testable and reusable within the LangGraph framework.

*   **Task/Feature Addressed:** Refactor the content writer agent to operate on a single CV item.
*   **Affected Component(s):** `src/agents/enhanced_content_writer.py`.
*   **`run_as_node` Refactoring:**
    *   The existing `run_as_node` method will be modified. Its logic must now focus exclusively on processing the single item referenced by `state.current_item_id`. It must no longer iterate over the entire `StructuredCV`.
*   **New Helper Method (`_build_single_item_prompt`):**
    *   A new private method, `_build_single_item_prompt`, must be created within the `EnhancedContentWriterAgent`.
    *   **Inputs:** This method will accept the target item, its parent section and subsection, the `job_description_data`, and any `user_feedback` from the `AgentState`.
    *   **Logic:** It will construct a highly specific, contextual LLM prompt tailored to generating content for only that single item.
    *   **Example Prompt Structure Snippet:**

      > You are an expert CV writer. Your task is to generate content for a single item in a CV.
      > The section is: 'Professional Experience'
      > The subsection is: 'Senior Software Engineer at TechCorp'
      > The original content of the item is: 'Developed Python applications.'
      > The target job description keywords are: FastAPI, Microservices, AWS, Docker, Kubernetes
      > Incorporate the following user feedback: 'Focus more on the DevOps aspects of this role.'
      > Please generate the new, improved content for this single item. Respond with only the generated text.

#### **3. LangGraph Workflow Implementation**

This task involves designing and implementing the complete state machine that orchestrates the iterative CV generation process, making the workflow explicit and robust.

*   **Task/Feature Addressed:** Implement the full state machine graph for the iterative workflow.
*   **Affected Component(s):** `src/orchestration/cv_workflow_graph.py`.
*   **Node Definitions:** The graph will be composed of the following asynchronous nodes:
    *   `parser_node`: Parses the initial CV and job description text, populating the initial `AgentState`.
    *   `research_node`: Enriches the state with research findings from the `ResearchAgent`.
    *   `generate_skills_node`: Generates the "Big 10" skills and populates the Key Qualifications section of the `StructuredCV`.
    *   `process_next_item_node`: Pops the next item ID from `items_to_process_queue` and sets it as `current_item_id`. It also clears `user_feedback` to prevent re-triggering.
    *   `content_writer_node`: Invokes the refactored `EnhancedContentWriterAgent` to process the `current_item_id`.
    *   `qa_node`: Invokes the `QualityAssuranceAgent` to inspect the newly generated content and add any warnings to the item's metadata.
    *   `prepare_next_section_node`: Identifies the next section from a predefined `WORKFLOW_SEQUENCE`, populates the `items_to_process_queue` with item IDs from that new section, and clears `current_item_id`.
    *   `formatter_node`: Invokes the `FormatterAgent` to generate the final PDF output.
*   **Conditional Routing Logic (`route_after_review`):** The core of the graph's intelligence lies in this conditional edge, which is invoked after the `qa_node`. Its logic is defined by the state of `user_feedback` and `items_to_process_queue`.

| Current State                                                           | User Action  | Next Node              | Rationale                                                                |
| :---------------------------------------------------------------------- | :----------- | :--------------------- | :----------------------------------------------------------------------- |
| `user_feedback.action` is `REGENERATE`                                  | Regenerate   | `content_writer`       | The user wants to retry generation for the current item.                 |
| `user_feedback.action` is `ACCEPT` AND `items_to_process_queue` is NOT empty | Accept       | `process_next_item`    | The current section is not finished; process the next item in the queue. |
| `user_feedback.action` is `ACCEPT` AND `items_to_process_queue` is empty AND next section exists | Accept       | `prepare_next_section` | The current section is finished; prepare and move to the next section.   |
| `user_feedback.action` is `ACCEPT` AND `items_to_process_queue` is empty AND NO next section | Accept       | `formatter`            | All sections are complete; proceed to final PDF generation.              |

*   **Graph Assembly (`cv_workflow_graph.py`):** The engineer will assemble the graph using `langgraph.graph.StateGraph`.

    ```python
    from langgraph.graph import StateGraph, END
    from src.orchestration.state import AgentState

    #... (import all node functions)...

    def build_cv_workflow_graph() -> StateGraph:
        workflow = StateGraph(AgentState)

        # Add all nodes
        workflow.add_node("parser", parser_node)
        workflow.add_node("generate_skills", generate_skills_node)
        workflow.add_node("process_next_item", process_next_item_node)
        workflow.add_node("content_writer", content_writer_node)
        workflow.add_node("qa", qa_node)
        workflow.add_node("prepare_next_section", prepare_next_section_node)
        workflow.add_node("formatter", formatter_node)

        # Define workflow edges
        workflow.set_entry_point("parser")
        workflow.add_edge("parser", "generate_skills")
        workflow.add_edge("generate_skills", "process_next_item")
        workflow.add_edge("process_next_item", "content_writer")
        workflow.add_edge("prepare_next_section", "process_next_item")
        workflow.add_edge("content_writer", "qa")
        workflow.add_edge("formatter", END)

        # Add the conditional routing logic
        workflow.add_conditional_edges(
            "qa",
            route_after_review,
            {
                "content_writer": "content_writer",
                "process_next_item": "process_next_item",
                "prepare_next_section": "prepare_next_section",
                "formatter": "formatter",
                END: END
            }
        )

        return workflow.compile()

    cv_graph_app = build_cv_workflow_graph()
    ```

#### **4. UI Interaction Model (Streamlit)**

This task implements the stateless view layer that enables user interaction with the LangGraph backend.

*   **Task/Feature Addressed:** Implement the Streamlit UI to facilitate the user-in-the-loop workflow.
*   **Affected Component(s):** `src/core/main.py`.
*   **Detailed Implementation Steps:**
    1.  **State Initialization:** The `main` function in `src/core/main.py` must initialize `st.session_state.agent_state` to `None` if it does not exist.
    2.  **UI Rendering Functions:** Create modular functions to render the `StructuredCV` from the `agent_state`. A key function will be `display_regenerative_item`, which renders a card for each job role or project.
    3.  **`display_regenerative_item` with Controls:** This function will render each item with its own "Accept" and "Regenerate" buttons. The `key` for each button must be unique (e.g., `f"accept_{item_id}"`). The `on_click` parameter will be bound to a callback function.
    4.  **`handle_user_action` Callback:** This callback function's sole responsibility is to populate `st.session_state.agent_state.user_feedback` with the appropriate `UserFeedback` object. It must not invoke the graph directly.
    5.  **Main Application Loop:** The main body of the script will contain the core logic that orchestrates the UI-backend interaction.

        ```python
        # src/core/main.py
        import streamlit as st
        from src.orchestration.cv_workflow_graph import cv_graph_app
        from src.orchestration.state import AgentState
        from src.models.data_models import UserAction, UserFeedback
        #... other imports

        def handle_user_action(action: str, item_id: str):
            """Callback to update the state with user feedback."""
            if st.session_state.agent_state:
                st.session_state.agent_state.user_feedback = UserFeedback(
                    action=UserAction(action),
                    item_id=item_id,
                )

        def main():
            if 'agent_state' not in st.session_state:
                st.session_state.agent_state = None

            # --- Main Interaction Loop ---
            if st.session_state.agent_state and st.session_state.agent_state.user_feedback:
                with st.spinner("Processing your request..."):
                    # Invoke the graph with the current state (which includes user feedback)
                    new_state_dict = cv_graph_app.invoke(st.session_state.agent_state.model_dump())

                    # Overwrite the session state with the new state from the graph
                    st.session_state.agent_state = AgentState.model_validate(new_state_dict)

                    # Clear the feedback so this block doesn't run again on the next rerun
                    st.session_state.agent_state.user_feedback = None

                    # Trigger an immediate re-render to show the updated content
                    st.rerun()

            # --- UI Rendering Code ---
            # ... (Tabs, input forms, etc.)...
            # The "Review & Edit" tab will call rendering functions that use
            # `st.session_state.agent_state` to display the CV and the
            # `handle_user_action` callback for buttons.
        ```

### **Implementation & Testing Plan**

#### **Detailed Implementation Steps**

1.  **Implement Pydantic Models:** Create/update `UserAction` and `UserFeedback` in `src/models/data_models.py`.
2.  **Refactor AgentState:** Update `src/orchestration/state.py` with the new fields for workflow control.
3.  **Refactor EnhancedContentWriterAgent:** Modify `run_as_node` and add the `_build_single_item_prompt` helper method in `src/agents/enhanced_content_writer.py`.
4.  **Build LangGraph Workflow:** Implement all nodes and the `route_after_review` conditional edge in `src/orchestration/cv_workflow_graph.py`.
5.  **Implement Streamlit UI:** Refactor `src/core/main.py` to include the main interaction loop, rendering functions, and callbacks as specified.

#### **Testing Considerations**

*   **Unit Tests:**
    *   Test the `route_after_review` function with various mock `AgentState` configurations to assert it returns the correct next node name.
    *   Test the refactored `EnhancedContentWriterAgent.run_as_node` to verify it only modifies the content of the item specified by `current_item_id`.
    *   Test the `handle_user_action` callback to ensure it correctly populates `user_feedback` in `st.session_state`.
*   **E2E Integration Test:**
    *   An E2E test must be created to simulate a full user session. This test will programmatically:
        1.  Submit initial data to trigger the first graph invocation.
        2.  Verify the first reviewable section appears.
        3.  Simulate a user clicking "Accept".
        4.  Verify the workflow advances to the next item in the same section.
        5.  Simulate a user clicking "Regenerate".
        6.  Verify the content of *only that item* changes.
        7.  Simulate accepting all items in a section.
        8.  Verify the workflow transitions to the next section.
    *   This test is critical for validating the entire state machine's logic.

### **Critical Gaps & Questions**

*   **Potential Challenges & Critical Considerations:**
    *   **State Immutability:** It is critical that the Streamlit UI code **never** modifies the `st.session_state.agent_state` object directly. All state changes must originate from the new state object returned by the `cv_graph_app.invoke` call. This must be strictly enforced.
    *   **Performance:** Invoking the entire graph on every click is acceptable for the MVP. However, the engineer should be aware that future performance optimization may require more targeted graph invocations or caching strategies.
*   **Assumptions Made:**
    *   The `ParserAgent` can successfully segment the input CV into a `StructuredCV` model with unique, UUID-tagged items for each role and project.
    *   The `FormatterAgent` is capable of rendering a final `StructuredCV` object into a PDF file.
*   **Critical Gaps & Questions to Address:**
    *   **Initial Generation Flow:** The blueprint clarifies that the initial "Generate" button click invokes the graph, which runs until it pauses for the first user review. The returned state is then rendered.
    *   **Error Display in UI:** How should non-fatal errors accumulated in `state.error_messages` be presented to the user?
        *   **Recommendation:** Implement a dedicated `st.expander("View Processing Warnings")` in the Streamlit UI that displays any messages from the `error_messages` list. This keeps the main UI clean while providing transparency.

---

Excellent. The recommendation for handling errors in the UI is approved. We will now proceed with the next task in the development plan, which is a core value-add feature for the user.

The next task is **Task 3.2: Implement "Big 10" Skills Generation**.

This blueprint details the implementation of this feature, which involves analyzing the job description and CV to automatically generate a list of the top 10 most relevant skills. This provides immediate, high-value content for the user and streamlines the CV tailoring process.

Here is the detailed technical blueprint for this implementation.

# **TASK_BLUEPRINT.md**

## **Task/Feature Addressed: Task 3.2 - Implement "Big 10" Skills Generation**

This blueprint details the implementation of the "Big 10" skills generation feature. The system will analyze the job description and the user's base CV to identify and present the top 10 most relevant skills or qualifications. This feature directly addresses a core value proposition of the MVP: automatically highlighting a candidate's key strengths in relation to a specific job.

---

### **Overall Technical Strategy**

The implementation will be encapsulated within a new, dedicated method on the `EnhancedContentWriterAgent`. This method, `generate_big_10_skills`, will orchestrate a two-step LLM chain to ensure both high-relevance and clean formatting:

1.  **Generation:** The first LLM call will use the `key_qualifications_prompt.md` to generate a raw, potentially verbose list of key qualifications based on the job description and a summary of the user's existing skills.
2.  **Cleaning & Structuring:** The second LLM call will use a cleaning prompt (e.g., `clean_skill_list_prompt.md`) to parse the raw output from the first step into a clean, structured list of exactly 10 skills.

The resulting clean list of skills and the initial raw LLM output will be stored in new, dedicated fields within the `StructuredCV` Pydantic model. This entire process will be integrated into the LangGraph workflow as a new, self-contained node (`generate_skills_node`) that runs immediately after the initial `parser_node`.

---

### **1. Pydantic Model Changes**

To store the generated skills and maintain transparency, the `StructuredCV` model must be extended.

*   **Affected Component(s):**
    *   `src/models/data_models.py`

*   **Pydantic Model Changes:**
    Modify the `StructuredCV` model to include fields for the processed skills list and the raw LLM output from the initial generation step.

    ```python
    # src/models/data_models.py
    # ... (other imports)
    from typing import List, Optional, Dict, Any
    from pydantic import BaseModel, Field
    from uuid import UUID

    class StructuredCV(BaseModel):
        """The main data model representing the entire CV structure."""
        id: UUID = Field(default_factory=uuid4)
        sections: List['Section'] = Field(default_factory=list)
        metadata: Dict[str, Any] = Field(default_factory=dict)

        # --- NEWLY ADDED FIELDS (Task 3.2) ---
        big_10_skills: List[str] = Field(
            default_factory=list,
            description="A clean list of the top 10 generated key qualifications."
        )
        big_10_skills_raw_output: Optional[str] = Field(
            None,
            description="The raw, uncleaned output from the LLM for the key qualifications generation."
        )
        # --- END OF NEW FIELDS ---

        # ... (existing methods like find_item_by_id) ...
    ```

*   **Rationale for Changes:**
    *   `big_10_skills`: Provides a structured, clean list of strings that can be easily rendered in the UI and used by other agents.
    *   `big_10_skills_raw_output`: Fulfills requirement `REQ-FUNC-UI-6` to store raw LLM output for user transparency and debugging purposes.

---

### **2. LLM Prompt Usage**

This feature will utilize two existing prompts in a chained sequence. A prerequisite is to rename one of the prompts for clarity.

*   **Affected Component(s):**
    *   `data/prompts/key_qualifications_prompt.md`
    *   `data/prompts/clean_big_6_prompt.md` -> Rename to `data/prompts/clean_skill_list_prompt.md`

*   **LLM Prompt Definitions:**

    1.  **Prerequisite:** Rename `data/prompts/clean_big_6_prompt.md` to `data/prompts/clean_skill_list_prompt.md` to reflect its general purpose.
    2.  **Generation Prompt (`key_qualifications_prompt.md`):** Used to generate the initial set of skills.
        *   **Context Variables:** `{main_job_description_raw}`, `{my_talents}`
    3.  **Cleaning Prompt (`clean_skill_list_prompt.md`):** Used to parse the potentially messy output of the first call.
        *   **Context Variables:** `{raw_response}`

---

### **3. Agent Logic Modifications (`EnhancedContentWriterAgent`)**

A new method will be added to `EnhancedContentWriterAgent` to handle this specific task.

*   **Affected Component(s):**
    *   `src/agents/enhanced_content_writer.py`

*   **Agent Logic Modifications:**
    Implement the `generate_big_10_skills` method. This is a self-contained utility function that performs both the generation and cleaning steps.

    ```python
    # src/agents/enhanced_content_writer.py
    import re

    class EnhancedContentWriterAgent(EnhancedAgentBase):
        # ... (existing __init__ and other methods) ...

        async def generate_big_10_skills(self, job_description: str, my_talents: str = "") -> Dict[str, Any]:
            """
            Generates the "Big 10" skills using a two-step LLM chain (generate then clean).
            Returns a dictionary with the clean skills list and the raw LLM output.
            """
            try:
                # === Step 1: Generate Raw Skills ===
                generation_template = self._load_prompt_template("key_qualifications_prompt")
                generation_prompt = generation_template.format(
                    main_job_description_raw=job_description,
                    my_talents=my_talents or "Professional with diverse technical and analytical skills"
                )

                logger.info("Generating raw 'Big 10' skills...")
                raw_response = await self.llm_service.generate_content(
                    prompt=generation_prompt,
                    content_type=ContentType.SKILLS
                )

                if not raw_response.success or not raw_response.content.strip():
                    raise ValueError(f"LLM returned an empty or failed response for skills generation: {raw_response.error_message}")

                raw_skills_output = raw_response.content

                # === Step 2: Clean the Raw Output ===
                cleaning_template = self._load_prompt_template("clean_skill_list_prompt")
                cleaning_prompt = cleaning_template.format(raw_response=raw_skills_output)

                logger.info("Cleaning generated skills...")
                cleaned_response = await self.llm_service.generate_content(
                    prompt=cleaning_prompt,
                    content_type=ContentType.SKILLS
                )

                if not cleaned_response.success:
                    raise ValueError(f"LLM cleaning call failed: {cleaned_response.error_message}")

                # === Step 3: Parse and Finalize ===
                skills_list = self._parse_big_10_skills(cleaned_response.content)

                logger.info(f"Successfully generated and cleaned {len(skills_list)} skills.")

                return {
                    "skills": skills_list,
                    "raw_llm_output": raw_skills_output,
                    "success": True,
                    "error": None
                }

            except Exception as e:
                logger.error(f"Error in generate_big_10_skills: {e}", exc_info=True)
                return {"skills": [], "raw_llm_output": "", "success": False, "error": str(e)}

        def _parse_big_10_skills(self, llm_response: str) -> List[str]:
            """
            Robustly parses the LLM response to extract a list of skills.
            Ensures exactly 10 skills are returned by truncating or padding.
            """
            lines = [line.strip().lstrip('-•* ').strip() for line in llm_response.split('\n') if line.strip()]
            cleaned_skills = [re.sub(r'^\d+\.\s*', '', line) for line in lines]
            final_skills = [skill for skill in cleaned_skills if skill and len(skill) > 2]

            if len(final_skills) > 10:
                return final_skills[:10]
            elif len(final_skills) < 10:
                padding = [f"Placeholder Skill {i+1}" for i in range(10 - len(final_skills))]
                return final_skills + padding
            return final_skills
    ```

---

### **4. LangGraph Workflow Integration**

A new node will be added to the graph to orchestrate the "Big 10" skills generation.

*   **Affected Component(s):**
    *   `src/orchestration/cv_workflow_graph.py`

*   **Orchestrator/Workflow Changes:**

    1.  **Define `generate_skills_node`:** Create a new async node function that calls the agent method and updates the state.

        ```python
        # src/orchestration/cv_workflow_graph.py
        from src.models.data_models import Item, ItemStatus, ItemType

        async def generate_skills_node(state: AgentState) -> dict:
            """Generates the 'Big 10' skills and updates the CV state."""
            logger.info("--- Executing Node: generate_skills_node ---")

            my_talents = ", ".join([item.content for section in state.structured_cv.sections if section.name == "Key Qualifications" for item in section.items])

            result = await content_writer_agent.generate_big_10_skills(
                job_description=state.job_description_data.raw_text,
                my_talents=my_talents
            )

            if result["success"]:
                updated_cv = state.structured_cv.model_copy(deep=True)
                updated_cv.big_10_skills = result["skills"]
                updated_cv.big_10_skills_raw_output = result["raw_llm_output"]

                qual_section = updated_cv.get_section_by_name("Key Qualifications")
                if not qual_section:
                    qual_section = Section(name="Key Qualifications", content_type="DYNAMIC", order=1)
                    updated_cv.sections.insert(1, qual_section)

                qual_section.items = [Item(content=skill, status=ItemStatus.GENERATED, item_type=ItemType.KEY_QUALIFICATION) for skill in result["skills"]]

                return {"structured_cv": updated_cv}
            else:
                return {"error_messages": state.error_messages + [f"Skills generation failed: {result['error']}"]}
        ```

    2.  **Update Graph Edges:** Insert the new node into the workflow sequence right after the parser.

        ```python
        # In build_cv_workflow_graph() function:

        workflow.add_node("generate_skills", generate_skills_node)

        # Update edges
        workflow.add_edge("parser", "generate_skills")
        workflow.add_edge("generate_skills", "process_next_item") # Or next relevant node
        ```

---

### **5. Testing Considerations**

*   **Unit Tests:**
    *   Test `EnhancedContentWriterAgent.generate_big_10_skills` by mocking the two LLM calls. Verify that it correctly handles a clean response, a messy response, and an error response.
    *   Test the `_parse_big_10_skills` helper method with various inputs (more/less than 10 skills, numbered lists, bullet points) to ensure it always returns a list of 10 strings.
*   **Integration Tests:**
    *   Test the `parser` -> `generate_skills` graph sequence. Verify that the `AgentState` passed out of `generate_skills` contains a `structured_cv` with the `big_10_skills` and `big_10_skills_raw_output` fields populated.
*   **E2E Testing:**
    *   Run a full workflow and assert that the "Key Qualifications" section in the UI displays a list of 10 skills after the initial processing step.

---

### **Critical Gaps & Questions**

*   No critical gaps are identified for this task. The plan is self-contained and builds directly on the established architecture. The two-step generate-and-clean process within a single agent method provides a good balance of encapsulation and robustness for this feature.

---

Excellent. The blueprint for the "Big 10" skills generation is approved. The system is now ready for the next critical feature in the MVP plan: generating the final, exportable document.

We will now proceed with **Task 3.3: Implement PDF Output Generation**.

This blueprint details the implementation of this feature, which provides users with a tangible, professional-grade artifact. It also incorporates a code hygiene task identified in the previous planning audit to ensure consistency.

Here is the detailed technical blueprint for this implementation.

# **TASK_BLUEPRINT.md**

## **Task/Feature Addressed: Task 3.3 - Implement PDF Output Generation**

This blueprint provides the technical specification for implementing PDF export functionality. It builds upon the completion of the foundational tasks and the granular processing workflow. This task will also incorporate the action items identified in the previous planning audit, ensuring code hygiene and consistency.

---

### **Part 0: Prerequisite Code Hygiene**

Before beginning the primary task, the following code cleanup, identified in a prior planning audit, must be completed to ensure a clean and logical codebase.

*   **Task 0.1: Rename Cleaning Prompt**
    *   **Action:** In the file system, rename `data/prompts/clean_big_6_prompt.md` to `data/prompts/clean_skill_list_prompt.md`.
    *   **Reason:** The prompt's function is to clean a list of skills, and its name should reflect its purpose generically, not a specific number of items.
    *   **Impact:** Update the reference to this filename within the `EnhancedContentWriterAgent` in the `generate_big_10_skills` method.

---

### **Part 1: PDF Output Generation Implementation**

### **Overall Technical Strategy**

The core of this feature will be implemented within the `FormatterAgent`. The agent will use the **Jinja2** templating engine to populate a professional HTML template with data from the final, accepted `StructuredCV` object. This rendered HTML, along with a dedicated CSS stylesheet for formatting, will then be converted into a PDF file using the **WeasyPrint** library. The `FormatterAgent` will be integrated as the final node in the LangGraph workflow, triggered after all content sections have been accepted by the user. The path to the generated PDF will be stored in the `AgentState`, making it available for download in the Streamlit UI.

---

### **1. Pydantic Model & State Management**

No changes are required for the `AgentState` or other Pydantic models. The existing `final_output_path: Optional[str]` field in `AgentState` is sufficient to store the result of this task.

---

### **2. New Components: HTML Template and CSS**

New files for templating the PDF output are required.

*   **Affected Component(s):**
    *   `src/templates/pdf_template.html` (New File)
    *   `src/frontend/static/css/pdf_styles.css` (New File)

*   **HTML Template (`pdf_template.html`):**
    *   This file will define the structure of the CV using HTML tags and Jinja2 templating syntax.

    ```html
    <!-- src/templates/pdf_template.html -->
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{{ cv.metadata.get('name', 'CV') }}</title>
        <!-- The CSS will be injected by WeasyPrint, no need for a <link> tag that works in a browser -->
    </head>
    <body>
        <header>
            <h1>{{ cv.metadata.get('name', 'Your Name') }}</h1>
            <p class="contact-info">
                {% if cv.metadata.get('email') %}{{ cv.metadata.get('email') }}{% endif %}
                {% if cv.metadata.get('phone') %} | {{ cv.metadata.get('phone') }}{% endif %}
                {% if cv.metadata.get('linkedin') %} | <a href="{{ cv.metadata.get('linkedin') }}">LinkedIn</a>{% endif %}
            </p>
        </header>

        {% for section in cv.sections %}
        <section class="cv-section">
            <h2>{{ section.name }}</h2>
            <hr>
            {% if section.items %}
                {% if section.name == 'Key Qualifications' %}
                    <p class="skills">
                        {% for item in section.items %}{{ item.content }}{% if not loop.last %} | {% endif %}{% endfor %}
                    </p>
                {% else %}
                    <ul>
                    {% for item in section.items %}
                        <li>{{ item.content }}</li>
                    {% endfor %}
                    </ul>
                {% endif %}
            {% endif %}
            {% if section.subsections %}
                {% for sub in section.subsections %}
                <div class="subsection">
                    <h3>{{ sub.name }}</h3>
                    {% if sub.metadata %}
                    <p class="metadata">
                        {% if sub.metadata.get('company') %}{{ sub.metadata.get('company') }}{% endif %}
                        {% if sub.metadata.get('duration') %} | {{ sub.metadata.get('duration') }}{% endif %}
                    </p>
                    {% endif %}
                    <ul>
                    {% for item in sub.items %}
                        <li>{{ item.content }}</li>
                    {% endfor %}
                    </ul>
                </div>
                {% endfor %}
            {% endif %}
        </section>
        {% endfor %}
    </body>
    </html>
    ```

*   **CSS Stylesheet (`pdf_styles.css`):**
    *   This file will contain professional styling for the PDF (e.g., fonts, margins, colors). It should be a clean, single-column layout suitable for professional CVs.

---

### **3. Agent Logic Modification (`FormatterAgent`)**

The `FormatterAgent` will be updated to perform the HTML rendering and PDF conversion.

*   **Affected Component(s):**
    *   `src/agents/formatter_agent.py`

*   **Agent Logic Modifications:**

    1.  **Import necessary libraries:** `jinja2` and `weasyprint`.
    2.  **Implement `run_as_node`:** This method will now orchestrate the PDF generation. It must handle the case where `WeasyPrint` system dependencies might be missing.

    ```python
    # src/agents/formatter_agent.py
    import os
    from jinja2 import Environment, FileSystemLoader
    from src.orchestration.state import AgentState
    from src.config.settings import get_config
    from src.config.logging_config import get_structured_logger
    from src.agents.agent_base import EnhancedAgentBase

    logger = get_structured_logger(__name__)

    try:
        from weasyprint import HTML, CSS
        WEASYPRINT_AVAILABLE = True
    except (ImportError, OSError) as e:
        WEASYPRINT_AVAILABLE = False
        logger.warning(f"WeasyPrint not available: {e}. PDF generation will be disabled, falling back to HTML.")

    class FormatterAgent(EnhancedAgentBase):
        async def run_as_node(self, state: AgentState) -> dict:
            """
            Takes the final StructuredCV from the state and renders it as a PDF or HTML.
            """
            logger.info("FormatterAgent: Starting output generation.")
            cv_data = state.structured_cv
            if not cv_data:
                return {"error_messages": state.error_messages + ["FormatterAgent: No CV data found in state."]}

            try:
                config = get_config()
                template_dir = config.project_root / "src" / "templates"
                static_dir = config.project_root / "src" / "frontend" / "static"
                output_dir = config.project_root / "data" / "output"
                output_dir.mkdir(exist_ok=True)

                # 1. Set up Jinja2 environment
                env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
                template = env.get_template("pdf_template.html")

                # 2. Render HTML from template
                html_out = template.render(cv=cv_data)

                # 3. Generate PDF using WeasyPrint (if available) or fallback to HTML
                if WEASYPRINT_AVAILABLE:
                    css_path = static_dir / "css" / "pdf_styles.css"
                    css = CSS(css_path) if css_path.exists() else None
                    pdf_bytes = HTML(string=html_out, base_url=str(template_dir)).write_pdf(stylesheets=[css] if css else None)

                    output_filename = f"CV_{state.structured_cv.id}.pdf"
                    output_path = output_dir / output_filename
                    with open(output_path, "wb") as f:
                        f.write(pdf_bytes)
                    logger.info(f"FormatterAgent: PDF successfully generated at {output_path}")
                else:
                    # Fallback to saving the HTML file
                    output_filename = f"CV_{state.structured_cv.id}.html"
                    output_path = output_dir / output_filename
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(html_out)
                    logger.warning(f"FormatterAgent: Fallback HTML successfully generated at {output_path}")

                return {"final_output_path": str(output_path)}

            except Exception as e:
                logger.error(f"FormatterAgent failed: {e}", exc_info=True)
                return {"error_messages": (state.error_messages or []) + [f"Output generation failed: {e}"]}
    ```

---

### **4. LangGraph Workflow Integration**

The `formatter_node` must be correctly placed at the end of the workflow.

*   **Affected Component(s):**
    *   `src/orchestration/cv_workflow_graph.py`

*   **Orchestrator/Workflow Changes:**
    *   The conditional router, `route_after_review`, should already be configured to transition to the `formatter` node when all sections are complete.
    *   Ensure the `formatter` node is defined and correctly added to the graph, with its final edge pointing to `END`.

---

### **5. Testing Considerations**

*   **Unit Test:** Test `FormatterAgent.run_as_node` by providing a mock `AgentState`. Assert that a file is created at the expected output path and that the returned dictionary contains the correct `final_output_path`. Test both the PDF and HTML fallback scenarios by mocking `WEASYPRINT_AVAILABLE`.
*   **Dependency Management:** The engineer must ensure that `weasyprint` and its system-level dependencies (like Pango, Cairo) are correctly listed in `requirements.txt` and documented in the `README.md` and `Dockerfile` for seamless setup.

---

### **Critical Gaps & Questions**

*   No critical gaps are identified. The plan is robust and incorporates the key architectural decisions from the audit (advisory QA, research consumption). The fallback to HTML is a critical mitigation for environments where `WeasyPrint` dependencies are not installed.

---

Excellent. The PDF generation feature is now planned. The next logical step in the development plan is to implement the user-facing transparency feature that was previously deferred.

This leads us to **Task 3.4: Implement Raw LLM Output Display**.

This blueprint details the implementation of this feature, which is crucial for user trust and debugging. It ensures that the raw, unedited text from the language model is captured and can be viewed by the user.

Here is the detailed technical blueprint for this implementation.

# **TASK_BLUEPRINT.md**

## **Task/Feature Addressed: Task 3.4 - Implement Raw LLM Output Display**

This blueprint provides the technical specification for implementing the display of raw LLM outputs. This feature directly addresses `REQ-FUNC-UI-6`, which requires that the system provides transparency into the AI's generation process, allowing users to see the original text produced by the language model before any cleaning or formatting.

---

### **Overall Technical Strategy**

The implementation is straightforward and focuses on ensuring data is captured and then rendered.

1.  **Capture:** The `EnhancedContentWriterAgent` will be modified to ensure that after every successful LLM call, the raw text content of the response is stored in the `raw_llm_output` field of the corresponding `Item` Pydantic model within the `StructuredCV`.
2.  **Render:** The Streamlit UI function responsible for rendering a reviewable item (`display_regenerative_item` in `src/core/main.py`) will be updated to include a conditional `st.expander`. This expander will only appear if the `raw_llm_output` field for an item contains data, and it will display the raw text within a code block.

---

### **1. Pydantic Model Changes**

This task relies on a field that should already exist from previous architectural work. This step is a verification.

*   **Affected Component(s):**
    *   `src/models/data_models.py`

*   **Pydantic Model Changes:**
    *   **Verification:** Confirm that the `Item` Pydantic model contains the `raw_llm_output: Optional[str]` field. This field is essential for storing the raw LLM response. No changes should be necessary if Task 3.2 was implemented correctly.

    ```python
    # src/models/data_models.py
    class Item(BaseModel):
        # ... (existing fields)
        raw_llm_output: Optional[str] = Field(default=None, description="Raw LLM output for this item for transparency and debugging")
        # ... (other fields)
    ```

---

### **2. Agent Logic Modification (`EnhancedContentWriterAgent`)**

The agent must be updated to correctly populate the `raw_llm_output` field.

*   **Affected Component(s):**
    *   `src/agents/enhanced_content_writer.py`

*   **Agent Logic Modifications:**
    *   Update the `run_as_node` method. After a successful LLM call, assign the raw response text from the `LLMResponse` object to the `target_item.raw_llm_output` field. In case of an LLM failure, populate this field with the error message for debugging.

    ```python
    # src/agents/enhanced_content_writer.py
    from src.orchestration.state import AgentState
    from src.models.data_models import ItemStatus, Item

    class EnhancedContentWriterAgent(EnhancedAgentBase):
        # ... (other methods are unchanged) ...

        async def run_as_node(self, state: AgentState) -> dict:
            # ... (logic to find target_item is unchanged) ...
            try:
                # ... (logic to build prompt is unchanged) ...

                llm_response = await self.llm_service.generate_content(...)

                if llm_response.success:
                    target_item.content = llm_response.content.strip()
                    # --- TASK 3.4 IMPLEMENTATION ---
                    # Store the raw, unadulterated text from the LLM for transparency.
                    target_item.raw_llm_output = llm_response.content
                    # --- END TASK 3.4 ---
                    target_item.status = ItemStatus.GENERATED
                else:
                    target_item.status = ItemStatus.GENERATION_FAILED
                    target_item.content = "Error: Could not generate content."
                    # --- TASK 3.4 IMPLEMENTATION (Failure Case) ---
                    target_item.raw_llm_output = f"LLM_ERROR: {llm_response.error_message or 'Unknown LLM error.'}"
                    # --- END TASK 3.4 ---

                return {"structured_cv": updated_cv}

            except Exception as e:
                # ... (error handling) ...
    ```

---

### **3. UI Changes (`src/core/main.py`)**

The UI must be updated to display the captured raw output.

*   **Affected Component(s):**
    *   `src/core/main.py`

*   **UI Changes:**
    *   Modify the `display_regenerative_item` function. After rendering the main content of an item, add a conditional `st.expander` that displays the `raw_llm_output` if it exists.

    ```python
    # src/core/main.py
    import streamlit as st

    def display_regenerative_item(item_data: Union[Section, Subsection], item_id: str):
        with st.container(border=True):
            # ... (logic to render header and item content) ...

            # --- TASK 3.4 IMPLEMENTATION ---
            # Check the first item in the group (section or subsection) as a proxy
            # for the raw output of the whole generative step.
            first_item = item_data.items[0] if item_data.items else None
            if first_item and first_item.raw_llm_output:
                 with st.expander("🔍 View Raw LLM Output"):
                    st.code(first_item.raw_llm_output, language="text")
            # --- END TASK 3.4 ---

            # ... (logic for accept/regenerate buttons) ...
    ```

---

### **4. Testing Considerations**

*   **Unit Test for Agent:** Update the unit test for `EnhancedContentWriterAgent.run_as_node`. Mock an LLM response and assert that the `raw_llm_output` field of the target item in the returned `structured_cv` is populated with the exact content from the mocked response. Also test the failure case.
*   **E2E Test:** The E2E test should be updated to verify that after a generation step, the "View Raw LLM Output" expander appears in the UI and contains the expected mock text from the E2E test fixture.

---

### **Critical Gaps & Questions**

*   No critical gaps are identified. The implementation is self-contained and leverages existing data structures and component interactions.

This completes the blueprint for the raw output display feature. The project is now ready for the final, and most user-facing, implementation task of the core MVP.

---

Excellent. The raw LLM output display feature is now planned. This completes the backend and data-centric tasks for the MVP's core functionality.

The final and most critical task in this phase is to build the user interface that brings the entire granular, iterative workflow to life.

We will now proceed with **Task 3.5: Implement Streamlit UI for Hybrid, Granular Control**. This blueprint incorporates the approved UX enhancements, such as automatic tab switching and dynamic status indicators, to ensure an intuitive user experience.

Here is the detailed technical blueprint for this implementation.

# **TASK_BLUEPRINT.md**

## **Task/Feature Addressed: Task 3.5 - Implement Streamlit UI for Hybrid, Granular Control**

This blueprint provides the definitive technical specification for implementing the Streamlit user interface to support the granular, item-by-item workflow orchestrated by LangGraph. The UI will render the `StructuredCV` from the `AgentState` and provide users with "Accept" and "Regenerate" controls for each major content item (e.g., each role in "Professional Experience"). This revision incorporates a more refined UX flow, including automatic tab switching and dynamic status indicators.

---

### **Overall Technical Strategy**

The Streamlit UI in `src/core/main.py` will be architected as a pure "view" of the `AgentState` stored in `st.session_state`. The application will follow a strict, unidirectional data flow:

1.  **Render State:** The UI will always render based on the current `st.session_state.agent_state`.
2.  **User Action:** User interactions (e.g., clicking "Regenerate" on a job role) will trigger an `on_click` callback.
3.  **Update State:** The *only* responsibility of the callback function is to update the `user_feedback` field within `st.session_state.agent_state`. It will not call any backend logic directly.
4.  **Invoke Graph:** After the callback completes, the main script loop resumes. It will detect the updated `user_feedback`, invoke the compiled LangGraph application (`cv_graph_app`) with the entire current state, and receive a new, updated state object.
5.  **Overwrite & Rerun:** The script will overwrite `st.session_state.agent_state` with the new state returned by the graph and then call `st.rerun()` to restart the script from the top, causing the UI to re-render with the latest content.

In this MVP, a regenerative "item" will correspond to an entire `Subsection` (e.g., one job role or one project) or a `Section` if it contains direct items (like Key Qualifications).

---

### **1. Pydantic Model & State Management**

No changes are required. This task will utilize the `AgentState` and `UserFeedback` models defined in the blueprint for Task 3.1.

---

### **2. UI Implementation (`src/core/main.py`)**

*   **Affected Component(s):**
    *   `src/core/main.py`

*   **Detailed Implementation Steps:**

    1.  **State Initialization:** Ensure the main function initializes `st.session_state` with an `agent_state` and an `active_tab` key.

        ```python
        # src/core/main.py
        def main():
            if 'agent_state' not in st.session_state:
                st.session_state.agent_state = None
            if 'active_tab' not in st.session_state:
                st.session_state.active_tab = "Input"
        ```

    2.  **"Generate Tailored CV" Button Logic (Enhanced UX):** This button initiates the entire process and must guide the user to the next logical step.

        ```python
        # In the "Input & Generate" tab
        if st.button("🚀 Generate Tailored CV", ...):
            # ... (validation and initial state creation) ...
            with st.spinner("Analyzing inputs and generating first section..."):
                initial_state_dict = cv_graph_app.invoke(initial_state.model_dump())
                st.session_state.agent_state = AgentState.model_validate(initial_state_dict)

                # --- UX ENHANCEMENT ---
                # Automatically switch the user to the review tab
                st.session_state.active_tab = "Review"
                st.rerun()
        ```

    3.  **Implement the Main Application Loop (with Dynamic Spinner):** This is the core logic that reacts to user feedback.

        ```python
        # src/core/main.py (at the top of the main function)
        if st.session_state.agent_state and st.session_state.agent_state.user_feedback:
            current_state = st.session_state.agent_state

            # --- UX ENHANCEMENT: Dynamic Spinner Text ---
            spinner_text = "Processing your request..."
            if current_state.user_feedback.action == UserAction.ACCEPT:
                spinner_text = "Accepting content and preparing next item..."
            elif current_state.user_feedback.action == UserAction.REGENERATE:
                 spinner_text = "Regenerating content..."

            with st.spinner(spinner_text):
                # Invoke the graph with the current state
                new_state_dict = cv_graph_app.invoke(current_state.model_dump())
                st.session_state.agent_state = AgentState.model_validate(new_state_dict)

                # Clear the feedback to prevent re-triggering
                st.session_state.agent_state.user_feedback = None

                # --- UX ENHANCEMENT: Handle Workflow Completion ---
                if new_state_dict.get("final_output_path"):
                    st.session_state.active_tab = "Export"

                st.rerun()
        ```

    4.  **Implement `display_regenerative_item` with Visual Status:** Update the rendering function to provide clear visual cues about the state of each item. An "item" here is a `Section` or `Subsection` object.

        ```python
        # src/core/main.py
        from src.models.data_models import UserAction, UserFeedback, ItemStatus, Section, Subsection
        from typing import Union

        def handle_user_action(action: str, item_id: str):
            if st.session_state.agent_state:
                st.session_state.agent_state.user_feedback = UserFeedback(action=UserAction(action), item_id=item_id)

        def display_regenerative_item(item_data: Union[Section, Subsection], item_id: str):
            is_accepted = all(item.status == ItemStatus.USER_ACCEPTED for item in item_data.items)

            with st.container(border=True):
                header_text = f"**{item_data.name}**" + ("  ✅" if is_accepted else "")
                st.markdown(header_text)

                for bullet in item_data.items:
                    st.markdown(f"- {bullet.content}")

                if item_data.items and item_data.items[0].raw_llm_output:
                     with st.expander("🔍 View Raw LLM Output"):
                        st.code(item_data.items[0].raw_llm_output, language="text")

                if item_data.metadata.get('qa_status') == 'warning':
                    issues = "\n- ".join(item_data.metadata.get('qa_issues', []))
                    st.warning(f"⚠️ **Quality Alert:**\n- {issues}", icon="⚠️")

                if not is_accepted:
                    cols = st.columns([1, 1, 4])
                    cols[0].button("✅ Accept", key=f"accept_{item_id}", on_click=handle_user_action, args=("accept", item_id))
                    cols[1].button("🔄 Regenerate", key=f"regenerate_{item_id}", on_click=handle_user_action, args=("regenerate", item_id))
                else:
                    st.success("Accepted.")
        ```

    5.  **Implement `display_export_options` and Error Display:** The UI needs a place to show the final download link and any processing errors.

        ```python
        # src/core/main.py
        import os

        def display_export_options(agent_state: Optional[AgentState]):
            if not agent_state: return

            # APPROVED: Implement dedicated expander for errors.
            if agent_state.error_messages:
                with st.expander("⚠️ View Processing Warnings", expanded=True):
                    for error in agent_state.error_messages:
                        st.warning(error)

            final_path = agent_state.final_output_path
            if final_path and os.path.exists(final_path):
                with open(final_path, "rb") as file:
                    st.download_button(
                        label="📄 Download Your CV",
                        data=file,
                        file_name=os.path.basename(final_path),
                        mime="application/pdf",
                        use_container_width=True
                    )
            else:
                st.info("Your final CV document will be available for download here once the review process is complete.")
        ```

---

### **3. Testing Considerations**

*   **Component Tests:**
    *   Test the `handle_user_action` callback by creating a mock `st.session_state` and asserting that `st.session_state.agent_state.user_feedback` is correctly populated.
    *   Test the `display_regenerative_item` function by passing an "accepted" item and asserting that the control buttons are not rendered.
*   **E2E Test for UX Flow:**
    *   Enhance the E2E test to validate the tab-switching behavior.
        1.  Start on the "Input" tab.
        2.  Click "Generate Tailored CV".
        3.  Assert that the UI now displays the "Review & Edit" content.
        4.  Sequentially click "Accept" on all generated items.
        5.  After the final "Accept", assert that the UI now displays the "Export" content and a download button is visible.

---

### **Conclusion**

This blueprint provides a complete and user-centric plan for the Streamlit UI. It maintains the robust architectural pattern while adding specific, high-impact UX enhancements that will make the application significantly more intuitive. This concludes the core feature implementation phase of the MVP. The system is now ready for the final phase of comprehensive testing and deployment preparation.

---

Excellent. The UI blueprint for Task 3.5 is approved. All core features for the MVP have now been planned.

We will now proceed with the final phase of the project: **Phase 3: "Smart Agent" Logic, Fallbacks, and Full SRS Alignment**. This phase focuses on enhancing the system's intelligence and resilience, moving it from a functional prototype to a reliable tool.

The first task in this phase is **Task 4.1: Implement "Smart Agent" Logic with Robust Fallbacks**. Per the latest strategic directive, this will be combined with **Task 4.2** to focus on a centralized, robust retry mechanism within the `LLMService` rather than complex, deterministic fallbacks in each agent.

Here is the detailed technical blueprint for this implementation.

# **TASK_BLUEPRINT.md**

## **Task/Feature Addressed: Task 4.1 & 4.2 - Implement Robust Error Handling in `LLMService` with Tenacity**

This blueprint consolidates Tasks 4.1 and 4.2 into a single, focused effort. The goal is to make the system resilient to transient API errors by implementing a robust retry mechanism directly within the `LLMService`, which is the root of most potential failures. Per the strategic directive, we will **not** implement complex, deterministic fallbacks (like regex parsing) in the MVP. Instead, if all retries fail, the agent will gracefully terminate its operation and report a clear error to the user.

---

### **Overall Technical Strategy**

The entire implementation will be centralized in `src/services/llm.py`. We will use the `tenacity` library to wrap the core LLM API call in a decorator that automatically handles retries with exponential backoff. This will be configured to retry only on specific, transient error types (e.g., rate limits, server errors, timeouts) while failing immediately on non-retryable errors (e.g., authentication failure). The `generate_content` method will be updated to catch the final `RetryError` from `tenacity` and return a structured `LLMResponse` object indicating failure, which will then be propagated up to the UI.

---

### **1. Dependency Management**

*   **Affected Component(s):** `requirements.txt`
*   **Action:** Confirm that `tenacity>=8.2.0` is present. No changes are required as it's already listed.

---

### **2. `LLMService` Refactoring for Resilience**

*   **Affected Component(s):**
    *   `src/services/llm.py`
    *   All agents that call `llm_service.generate_content()` (as they will now need to handle a `success=False` response).

*   **Detailed Implementation Steps:**

    1.  **Define Retry-able Exceptions:** At the top of `llm.py`, define a tuple of exceptions that warrant a retry, based on the approved clarification.

        ```python
        # src/services/llm.py
        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
        try:
            from google.api_core import exceptions as google_exceptions
            RETRYABLE_EXCEPTIONS = (
                google_exceptions.ResourceExhausted,  # For 429 Rate Limit Exceeded
                google_exceptions.ServiceUnavailable, # For 503 Service Unavailable
                google_exceptions.InternalServerError, # For 500 Internal Server Error
                google_exceptions.DeadlineExceeded,   # For timeouts
                TimeoutError,
                ConnectionError,
            )
        except ImportError:
            RETRYABLE_EXCEPTIONS = (TimeoutError, ConnectionError)
        ```

    2.  **Refactor `_make_llm_api_call` with `@retry`:** Create a private, decorated method that contains *only* the direct API call.

        ```python
        # src/services/llm.py

        class EnhancedLLMService:
            # ... (__init__ and other methods) ...

            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=60),
                retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
                reraise=True # IMPORTANT: Reraise the exception if all retries fail
            )
            async def _make_llm_api_call(self, prompt: str) -> Any:
                """A new private method that contains only the direct API call logic."""
                logger.info("Making LLM API call...")
                response = await self.llm.generate_content_async(prompt)
                if not hasattr(response, 'text') or not response.text:
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                        raise ValueError(f"Content blocked by API safety filters: {response.prompt_feedback.block_reason.name}")
                    raise ValueError("LLM returned an empty or invalid response.")
                return response
        ```

    3.  **Refactor `generate_content` to Handle Final Failure:** This method now orchestrates the call and handles the case where all retries have been exhausted.

        ```python
        # src/services/llm.py

        class EnhancedLLMService:
            async def generate_content(self, prompt: str, ...) -> LLMResponse:
                # ... (caching logic remains the same) ...
                try:
                    response = await self._make_llm_api_call(prompt)
                    raw_text = response.text
                    return LLMResponse(content=raw_text, raw_response_text=raw_text, success=True, ...)

                except Exception as e:
                    # ALL retries have failed, or it was a non-retryable error.
                    # Report failure gracefully.
                    processing_time = time.time() - start_time
                    logger.error(f"LLM call failed after all retries: {e}", exc_info=True)
                    return LLMResponse(
                        content="",
                        raw_response_text=f"ERROR: {str(e)}",
                        success=False,
                        error_message=f"The AI service failed after multiple retries. Please try again later. Error: {type(e).__name__}"
                    )
        ```

---

### **3. Agent Logic Modification (All Agents)**

All agents must now be updated to handle the `success=False` case from the `LLMResponse`.

*   **Affected Component(s):** `ParserAgent`, `EnhancedContentWriterAgent`, `ResearchAgent`, etc.

*   **Agent Logic Modifications:**

    ```python
    # Example in EnhancedContentWriterAgent.run_as_node

    class EnhancedContentWriterAgent(EnhancedAgentBase):
        async def run_as_node(self, state: AgentState) -> dict:
            # ... (logic to find target_item and build prompt) ...

            llm_response = await self.llm_service.generate_content(prompt)

            if not llm_response.success:
                # LLM service failed after all retries. Propagate the error.
                logger.error(f"ContentWriterAgent giving up on item {state.current_item_id} due to LLM failure.")
                error_list = state.error_messages or []
                error_list.append(llm_response.error_message)
                return {"error_messages": error_list}

            # ... (continue with normal logic using llm_response.content) ...
    ```

*   **Rationale:** This change simplifies agent logic significantly. Agents no longer need their own `try...except` blocks for LLM calls. They simply check the `success` flag on the returned `LLMResponse` and act accordingly.

---

### **4. Testing Considerations**

*   **Unit Tests for `LLMService`:**
    *   **Retry-able Error Test:** Mock `llm.generate_content_async` to raise `google_exceptions.ResourceExhausted` twice, then succeed. Assert that the underlying mock was called 3 times.
    *   **Non-Retry-able Error Test:** Mock `llm.generate_content_async` to raise `ValueError` (e.g., invalid API key error). Assert that the mock was only called *once* and the exception was re-raised immediately.
    *   **Exhausted Retries Test:** Mock the call to always raise `TimeoutError`. Call `generate_content` (not the private method) and assert that it returns an `LLMResponse` object with `success=False` and a descriptive `error_message`.

---

### **Conclusion**

This blueprint provides a robust, centralized approach to handling API resilience. By implementing this, the system will be significantly more stable and capable of weathering transient network and service issues, a key non-functional requirement for the MVP.

---

Excellent. The robust error handling mechanism is now planned. The next logical step in the development plan is to integrate the remaining agents into the workflow to enhance its intelligence.

We will now proceed with **Task 4.3: Integrate Remaining MVP Agents (QA, Research) into LangGraph Workflow**.

This blueprint outlines the integration of the `ResearchAgent` and `QualityAssuranceAgent` into the main LangGraph workflow. This step is crucial for enriching the context before content generation and for validating the output afterward. This plan incorporates the "Advisory QA" model and "Research Consumption" strategies identified during previous planning.

Here is the detailed technical blueprint for this implementation.

# **TASK_BLUEPRINT.md**

## **Task/Feature Addressed: Task 4.3 - Integrate Remaining MVP Agents (QA, Research) into LangGraph Workflow**

This blueprint outlines the integration of the `ResearchAgent` and `QualityAssuranceAgent` into the main LangGraph workflow. This step enhances the intelligence of the system by enriching the context before content generation and validating the output after generation. This plan incorporates the "Advisory QA" model and "Research Consumption" strategies.

---

### **Overall Technical Strategy**

The integration will happen by defining new nodes and updating the graph's topology in `src/orchestration/cv_workflow_graph.py`.

1.  **Research Integration:** A `research` node will be inserted immediately after the initial `parser` node. It will use the parsed job description and CV to perform its analysis and populate the `research_findings` field in the `AgentState`. This ensures all subsequent generative agents have access to this enriched context.
2.  **Quality Assurance Integration:** A `qa` node will be inserted after every generative step (e.g., after `content_writer`). It will inspect the newly generated content (identified by `current_item_id`), check it against quality criteria, and add its findings to the metadata of the corresponding `Item` in the `StructuredCV`. The QA agent will be "advisory" and will not trigger automatic regeneration loops; it only provides feedback to the user.

The `run_as_node` methods for both agents will be implemented to conform to the LangGraph standard of accepting the `AgentState` and returning a dictionary of the updated state fields.

---

### **Part 1: `ResearchAgent` Integration & Consumption**

*   **Affected Component(s):**
    *   `src/agents/research_agent.py`
    *   `src/agents/enhanced_content_writer.py`
    *   `src/orchestration/cv_workflow_graph.py`
    *   `src/orchestration/state.py`

*   **Detailed Implementation Steps:**

    1.  **Update `AgentState`:** Ensure the `AgentState` model in `src/orchestration/state.py` includes the `research_findings` field.
        ```python
        # src/orchestration/state.py
        class AgentState(BaseModel):
            # ...
            research_findings: Optional[Dict[str, Any]] = None
            # ...
        ```

    2.  **Implement `research_agent.run_as_node`:** This method will be the agent's entry point from the graph.
        ```python
        # src/agents/research_agent.py
        class ResearchAgent(EnhancedAgentBase):
            async def run_as_node(self, state: AgentState) -> dict:
                logger.info("--- Executing Node: ResearchAgent ---")
                if not state.job_description_data:
                    logger.warning("ResearchAgent: No job description data found in state.")
                    return {}
                try:
                    # In a real async agent, self.run would be async. Here we simulate.
                    input_data = {
                        "job_description_data": state.job_description_data.model_dump(),
                        "structured_cv": state.structured_cv.model_dump()
                    }
                    findings = await self.run_async(input_data, None) # Assuming run_async exists

                    logger.info("ResearchAgent completed successfully.")
                    return {"research_findings": findings.output_data}
                except Exception as e:
                    return {"error_messages": (state.error_messages or []) + [f"Research failed: {e}"]}
        ```

    3.  **Modify `ContentWriterAgent` to Consume Findings:** Update the prompt-building logic in `EnhancedContentWriterAgent` to incorporate `research_findings` from the state.
        ```python
        # src/agents/enhanced_content_writer.py
        class EnhancedContentWriterAgent(EnhancedAgentBase):
            def _build_single_item_prompt(self, item, section, subsection, job_data, feedback, research_findings):
                # ... build prompt string ...
                if research_findings:
                    company_values = research_findings.get("company_values", [])
                    prompt += f"\n\n--- CRITICAL CONTEXT ---\nCompany Values: {', '.join(company_values)}"
                return prompt
        ```

    4.  **Update Graph Topology:** Insert the `research` node into `cv_workflow_graph.py` after the `parser` node.
        ```python
        # src/orchestration/cv_workflow_graph.py
        workflow.add_node("research", research_node)
        workflow.add_edge("parser", "research")
        workflow.add_edge("research", "generate_skills")
        ```

*   **Testing Considerations:**
    *   Write a unit test for `ResearchAgent.run_as_node` to ensure it returns a dictionary with the `research_findings` key.
    *   Write a unit test for `EnhancedContentWriterAgent._build_single_item_prompt` and pass a mock `research_findings` dictionary to assert that the returned prompt string contains the contextual information.

---

### **Part 2: "Advisory" `QualityAssuranceAgent` Integration**

*   **Affected Component(s):**
    *   `src/agents/quality_assurance_agent.py`
    *   `src/orchestration/cv_workflow_graph.py`
    *   `src/core/main.py` (UI rendering)

*   **Detailed Implementation Steps:**

    1.  **Define `Item` Metadata Convention:** Establish the convention that the `QAAgent` will add the following keys to an `Item`'s `metadata` dict:
        *   `qa_status`: "passed" | "warning" | "failed"
        *   `qa_issues`: `List[str]`

    2.  **Implement `qa_agent.run_as_node` as an Annotator:** Refactor the agent's node function to only inspect and annotate metadata. It **must not** alter `item.content`.
        ```python
        # src/agents/quality_assurance_agent.py
        class QualityAssuranceAgent(EnhancedAgentBase):
            async def run_as_node(self, state: AgentState) -> dict:
                logger.info(f"--- Executing Node: QAAgent (Item: {state.current_item_id}) ---")
                if not state.current_item_id:
                    return {}
                updated_cv = state.structured_cv.model_copy(deep=True)
                item, _, _ = updated_cv.find_item_by_id(state.current_item_id)
                if not item:
                    return {"error_messages": (state.error_messages or []) + [f"QA failed: Item {state.current_item_id} not found."]}

                issues = []
                # Example Check 1: Content Length
                if len(item.content.split()) < 10:
                    issues.append("Content may be too short. Consider adding more detail.")

                # Example Check 2: Action Verbs
                action_verbs = ["developed", "led", "managed", "optimized", "implemented"]
                if not any(item.content.lower().lstrip().startswith(verb) for verb in action_verbs):
                    issues.append("Consider starting with a strong action verb (e.g., Developed, Managed).")

                # Annotate metadata
                item.metadata['qa_status'] = 'warning' if issues else 'passed'
                item.metadata['qa_issues'] = issues

                return {"structured_cv": updated_cv}
        ```

    3.  **Update Graph Topology:** Place the `qa` node immediately after the `content_writer` node. The conditional routing edge will now originate from `qa`.
        ```python
        # src/orchestration/cv_workflow_graph.py
        workflow.add_node("qa", qa_node)
        workflow.add_edge("content_writer", "qa")
        workflow.add_conditional_edges("qa", route_after_review, {...})
        ```

    4.  **Update Streamlit UI:** Modify `src/core/main.py` to read and display the QA metadata as a warning.
        ```python
        # src/core/main.py
        def display_regenerative_item(...):
            # ... (rendering logic) ...
            if item_data.metadata.get('qa_status') == 'warning':
                issues = "\n- ".join(item_data.metadata.get('qa_issues', []))
                st.warning(f"⚠️ **Quality Alert:**\n- {issues}", icon="⚠️")
        ```

*   **Testing Considerations:**
    *   **Unit Test `QAAgent`:** Pass a state object with a `current_item_id` and specific content. Assert that the returned `structured_cv` has the correct `qa_status` and `qa_issues` in the target item's metadata, and that the `item.content` remains unchanged.
    *   **E2E Test:** Run a full workflow. After a piece of content is generated, verify that a "Quality Alert" warning message appears in the UI if the content violates a QA rule.

---

### **Critical Gaps & Questions**

*   No critical gaps are identified. The plan is robust and incorporates the key architectural decisions from the audit (advisory QA, research consumption). This completes the planning for the "smart agent" integration phase. The next step is the final codebase standardization.

---

Excellent. The integration plan for the `ResearchAgent` and `QualityAssuranceAgent` is approved.

We will now proceed with the final task of this phase, **Task 4.4: Finalize LangGraph-Compatible Agent Interfaces**. This is a crucial refactoring step to ensure long-term maintainability and consistency across the entire agentic system. It will formalize the contracts that agents must adhere to, making the system more predictable and easier to debug.

Here is the detailed technical blueprint for this task.

# **TASK_BLUEPRINT.md**

## **Task/Feature Addressed: Task 4.4 - Finalize LangGraph-Compatible Agent Interfaces**

This task is a final, crucial refactoring and standardization step. The goal is to ensure that every agent in the system rigorously adheres to the standard LangGraph-compatible interface. This will involve reviewing and modifying each agent's primary execution method (`run_as_node`) to guarantee it exclusively interacts with the rest of the system via the `AgentState` object. This creates a clean, predictable, and maintainable architecture.

---

### **Overall Technical Strategy**

The strategy involves a comprehensive audit and refactoring of all agent classes in `src/agents/`. Each agent's `run_as_node` method will be standardized to follow a strict pattern:

1.  **Input:** The method signature must be `async def run_as_node(self, state: AgentState) -> dict`. It will receive the *entire* current state of the workflow.
2.  **Processing:** The agent will read all necessary data directly from the input `state` object (e.g., `state.structured_cv`, `state.current_item_id`, `state.research_findings`). It will perform its core logic based on this data.
3.  **State Immutability:** The agent **must not** modify the input `state` object directly. It must work on copies of any complex objects it needs to change (e.g., `updated_cv = state.structured_cv.model_copy(deep=True)`).
4.  **Output:** The method must return a dictionary containing *only* the fields of the `AgentState` that it has created or modified. LangGraph will be responsible for merging this dictionary back into the main state.

This task also involves removing any legacy synchronous `run` methods or ensuring they are clearly marked as deprecated and not used by the core LangGraph workflow.

---

### **1. Agent Interface Standardization**

*   **Affected Component(s):**
    *   `src/agents/agent_base.py`
    *   `src/agents/parser_agent.py`
    *   `src/agents/research_agent.py`
    *   `src/agents/enhanced_content_writer.py`
    *   `src/agents/quality_assurance_agent.py`
    *   `src/agents/formatter_agent.py`
    *   `src/agents/cleaning_agent.py` (if it exists, otherwise this is a check)

*   **Detailed Implementation Steps:**

    1.  **Define Abstract `run_as_node` in `EnhancedAgentBase`:** Formalize the contract by adding an abstract `run_as_node` method to the base class.

        ```python
        # src/agents/agent_base.py
        from abc import abstractmethod
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from src.orchestration.state import AgentState

        class EnhancedAgentBase(ABC):
            # ... (existing methods) ...

            @abstractmethod
            async def run_as_node(self, state: "AgentState") -> dict:
                """
                Standard LangGraph node interface for all agents.
                This method must be implemented by all subclasses.
                """
                raise NotImplementedError("This agent must implement the run_as_node method.")
        ```

    2.  **Refactor All Agent Implementations:** Audit every agent file listed above and ensure its `run_as_node` method conforms to the `async def run_as_node(self, state: AgentState) -> dict:` signature and behavior.
        *   **Input Handling:** All inputs must be derived from the `state` object.
        *   **Output Handling:** The `return` statement must be a dictionary of `AgentState` fields.
        *   **Immutability:** All modifications to complex state objects must be done on a `model_copy(deep=True)`.

    3.  **Deprecate Old `run` Methods:** Locate any old synchronous `run` methods. Add a `DeprecationWarning` and a docstring indicating that `run_as_node` should be used instead. For the MVP, these methods can be kept for backward compatibility in existing tests, but they should not be called by the main workflow.

        ```python
        import warnings

        def run(self, input_data: Any) -> Any:
            """
            DEPRECATED: This method is for legacy compatibility only.
            The LangGraph workflow uses run_as_node.
            """
            warnings.warn(f"The 'run' method on {self.name} is deprecated.", DeprecationWarning)
            # For now, we can keep the implementation for old tests.
            pass
        ```

---

### **2. Example: Refactoring `ParserAgent` (Final Check)**

This example confirms the final, correct structure for the `ParserAgent`.

*   **Affected Component(s):** `src/agents/parser_agent.py`

*   **Refactoring Logic:**

    ```python
    # src/agents/parser_agent.py
    from src.orchestration.state import AgentState
    from typing import Dict

    class ParserAgent(EnhancedAgentBase):
        async def run_as_node(self, state: AgentState) -> Dict:
            """Parses raw text from the state and populates structured data fields."""
            logger.info("ParserAgent node running.")
            try:
                # 1. Read input directly from the state object
                raw_jd = state.job_description_data.raw_text
                raw_cv = state.structured_cv.metadata.get("original_cv_text", "")

                # 2. Perform core logic
                parsed_jd = await self.parse_job_description(raw_jd)
                parsed_cv = self.parse_cv_text(raw_cv, parsed_jd)

                # 3. Return a dictionary of the updated state fields
                return {
                    "job_description_data": parsed_jd,
                    "structured_cv": parsed_cv,
                }
            except Exception as e:
                logger.error(f"Error in ParserAgent node: {e}", exc_info=True)
                return {"error_messages": state.error_messages + [f"Parsing failed: {e}"]}
    ```

---

### **3. Testing Considerations**

*   **Refactor Existing Unit Tests:** All existing agent unit tests must be refactored. Instead of calling `agent.run(mock_input)`, they should now:
    1.  Create a mock `AgentState` object containing the necessary input data.
    2.  Call `await agent.run_as_node(mock_state)`.
    3.  Assert that the returned dictionary contains the expected keys and that the values are correct.

*   **New Unit Tests for Immutability:**
    *   For each agent, write a test that passes a state object to `run_as_node`.
    *   After the call, assert that the *original* state object passed into the function has **not** been changed. This verifies that the agent is correctly working on a copy.

*   **Integration Tests:** The existing LangGraph integration tests are already aligned with this pattern and will serve as the primary validation that the refactored agents are working correctly within the graph. They must all pass after the agent refactoring.

---

### **Conclusion**

With the completion of this task, the agentic architecture of the MVP will be fully standardized and robust. All agents will communicate via a single, predictable contract (`AgentState`), making the system easier to debug, maintain, and extend. This concludes the planning for the "Smart Agent" phase. The project is now ready to move into the final phase of comprehensive testing and deployment preparation.

---

Excellent. The architectural hardening and smart agent integration phase is now fully planned.

We will now proceed with the final phase of the project: **Phase 4: Comprehensive Testing & Deployment Preparation**. This phase is critical to ensure the application is reliable, performant, and ready for users. I will combine the blueprints for **Tasks 5.1 & 5.2 (Testing)**, **Task 5.3 (Performance)**, and **Tasks 5.4 & 5.5 (Documentation & Deployment)** into a single, comprehensive handoff document for the implementation team.

Here is the final technical blueprint for the project.

# **TASK_BLUEPRINT.md (Final Phase)**

## **Tasks Addressed: 5.1 & 5.2 (Testing), 5.3 (Performance Tuning), 5.4 (Documentation), & 5.5 (Deployment Prep)**

This final blueprint details the concluding phase of MVP development. It covers the creation of a comprehensive testing suite, performance optimization, the creation of user and developer documentation, and the finalization of the containerization setup for a reproducible deployment.

---

### **Part 1: Comprehensive Testing & NFR Validation (Tasks 5.1 & 5.2)**

**Overall Strategy:** The testing strategy is multi-layered to ensure quality from the unit level to the full user experience.

1.  **Unit Testing (`tests/unit/`):** Focus on isolating and testing individual components, especially agent logic, helper functions, and data models. Mocking will be used extensively to isolate dependencies (e.g., LLM API calls, file system access).
2.  **Integration Testing (`tests/integration/`):** Test the interactions between components. The primary focus will be on testing short sequences of the LangGraph workflow to ensure that state transitions and data handoffs between nodes are correct.
3.  **Deterministic E2E Testing (`tests/e2e/`):** Use `pytest` with `asyncio` support to simulate the full user workflow from start to finish. These tests will run against a **fully mocked LLM service** that returns predictable, pre-defined responses from files. This ensures the tests are fast and reliable for the CI/CD pipeline.
4.  **Live API Quality Monitoring (`tests/live_api/`):** A separate, small suite of tests will be created to make real calls to the Gemini API. These tests are **not part of the standard CI/CD run**. They will be used for manual validation or scheduled, non-blocking monitoring to check for prompt drift or breaking changes in the live API.

*   **Affected Component(s):**
    *   `/tests/` (entire directory)
    *   `pytest.ini`

*   **Detailed Implementation Steps:**

    1.  **Create Test Data Fixtures:** In `tests/e2e/test_data/`, create subdirectories for each distinct E2E test scenario (e.g., `scenario_happy_path_swe/`). Populate these with `input_cv.txt`, `input_jd.txt`, and mock LLM response files.
    2.  **Implement Mock LLM Fixture:** In `tests/e2e/conftest.py`, create a `pytest` fixture (`mock_e2e_llm_service`) that provides a mocked `EnhancedLLMService`. This mock will load its responses from the test data files based on the input prompt it receives.
    3.  **Write "Happy Path" E2E Test:** In `tests/e2e/test_complete_cv_generation.py`, write a test that uses the mock service to validate the entire workflow from initial state to final output path. Assert that no errors occurred and that the final state is as expected (e.g., `big_10_skills` is populated).
    4.  **Write Live API Quality Test:** In a new directory `tests/live_api/`, create `test_live_llm_quality.py`. Add a test marked with `@pytest.mark.live_api` that makes a real call to the Gemini API and performs "soft" assertions on the quality of the response (e.g., `len(response.content) > 50`).
    5.  **Configure `pytest.ini`:** Create this file at the root to register the `live_api` marker. This prevents `pytest` from issuing warnings about unknown markers.

        ```ini
        # pytest.ini
        [pytest]
        markers =
            live_api: marks tests that call the live Gemini API (slow, non-deterministic)
        ```
    6.  **CI/CD Configuration:** Ensure the CI/CD script is updated to run tests with `pytest -m "not live_api"` to exclude the live, non-deterministic tests from the automated pipeline.

---

### **Part 2: Performance Tuning and Optimization (Task 5.3)**

**Overall Strategy:** The optimization strategy will be data-driven, focusing on three main areas: profiling to find bottlenecks, optimizing LLM call patterns via caching, and ensuring all I/O is non-blocking.

*   **Affected Component(s):**
    *   `scripts/profiling_runner.py` (New File)
    *   `src/services/llm.py`

*   **Detailed Implementation Steps:**

    1.  **Create a Profiling Script:** Create a new script, `scripts/profiling_runner.py`, that programmatically runs a full E2E workflow and generates a performance profile using `cProfile`. This allows for repeatable analysis.
    2.  **Implement LLM Caching:** This is the highest-impact optimization. Modify `src/services/llm.py` to include an in-memory caching layer.

        ```python
        # src/services/llm.py
        import functools
        import hashlib

        @functools.lru_cache(maxsize=128)
        def get_cached_response(cache_key: str):
            """This function is a placeholder for a more robust cache, but leverages lru_cache."""
            return None # In a real implementation, this would interact with a cache store.

        class EnhancedLLMService:
            async def generate_content(self, prompt: str, ...):
                cache_key = hashlib.md5(prompt.encode()).hexdigest()
                cached = get_cached_response(cache_key)
                if cached:
                    return cached

                # ... (actual API call logic) ...
                # On success, store the result. This part needs a real cache implementation.
                # For now, we rely on the conceptual nature of the cache.
                return llm_response
        ```
        *Note: The engineer should implement a simple dictionary-based cache if a more complex one (like `cachetools`) is not desired for the MVP.*

    3.  **Audit Asynchronous Execution:** Conduct a final review of the codebase for any blocking I/O calls within `async` functions (e.g., `time.sleep`, synchronous `open()`). Replace them with their `asyncio` equivalents (`asyncio.sleep`, `aiofiles`).

---

### **Part 3: Documentation and Deployment Prep (Tasks 5.4 & 5.5)**

**Overall Strategy:** Create comprehensive documentation for users and developers and finalize the containerization setup for a reproducible deployment.

*   **Affected Component(s):**
    *   `README.md` (Update)
    *   `/docs/` (New Directory & Files)
    *   `Dockerfile`

*   **Detailed Implementation Steps:**

    1.  **Create Documentation Suite:**
        *   Create the `/docs` directory.
        *   Create `docs/user_guide.md` explaining the workflow from input to export.
        *   Create `docs/developer_guide.md` covering setup, testing (`pytest -m "not live_api"`), and contribution guidelines.
        *   Create `docs/architecture.md` with a high-level flowchart of the LangGraph workflow and an explanation of the `AgentState`.
    2.  **Update `README.md`:** Significantly update the README to reflect the final MVP features, provide clear setup instructions (prioritizing Docker), and link to the detailed documentation in the `/docs` directory.
    3.  **Finalize `Dockerfile`:**
        *   **Verify Dependencies:** Confirm the `apt-get install` command includes all necessary libraries for `WeasyPrint` (`libpango-1.0-0`, `libcairo2`, etc.). This was validated in a previous step.
        *   **Security Best Practices:** Confirm the application runs as a **non-root user** (`aicvgen`) and that no secrets are hardcoded in the image.
        *   **Entrypoint:** Verify the `CMD ["python", "run_app.py"]` is correct.

---

### **Testing Considerations for Final Phase**

*   **Benchmarking:** Use the `scripts/profiling_runner.py` script to run benchmarks before and after the caching implementation in Part 2 to quantify the performance improvements.
*   **Deployment Testing:** On a clean machine with only Docker installed, run `docker-compose up --build`. Access the running container and perform a full E2E test to validate the entire production-like environment.

---

### **Conclusion of Planning**

This blueprint marks the completion of the technical planning phase for the AI CV Generator MVP. All identified bugs have been addressed with architectural solutions, and all features have been planned with detailed, actionable steps. The project is now fully specified and ready for the final implementation, testing, and documentation push.

---

