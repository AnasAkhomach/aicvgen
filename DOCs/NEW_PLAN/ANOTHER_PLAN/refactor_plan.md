# Refactoring Plan: Supervisor-Orchestrated Subgraph Architecture

This document outlines the plan to refactor the existing `CVWorkflowGraph` into a more modular, supervisor-orchestrated architecture using LangGraph subgraphs. This approach will improve maintainability, readability, and allow for more complex, stateful interactions within each CV section.

## 1. High-Level Architecture

The new architecture will consist of a main **Supervisor Graph** that orchestrates the execution of several **Section Subgraphs**. 

- **Supervisor Graph**: Manages the overall workflow, routing control to the appropriate subgraph based on the current state. It does not perform any content generation itself.
- **Section Subgraphs**: Each subgraph is responsible for generating a specific section of the CV (e.g., Key Qualifications, Professional Experience). It encapsulates the logic for writing, quality assurance, and handling user feedback for that section.

## 2. State Management (`state.py`)

The existing `AgentState` is largely sufficient. The following fields will be key to the new architecture:

- `current_section_index`: Used by the supervisor to determine which subgraph to invoke next.
- `user_feedback`: Used within each subgraph to decide whether to regenerate content or proceed.
- `error_messages`: Used by the supervisor for global error handling.

No major changes are anticipated for `state.py` at this stage.

## 3. Refactoring `cv_workflow_graph.py`

This file will see the most significant changes.

### 3.1. New Methods for Subgraph Construction

For each section in `WORKFLOW_SEQUENCE`, a dedicated method will be created to build and compile its subgraph. For example:

- `_build_key_qualifications_subgraph()`
- `_build_professional_experience_subgraph()`
- `_build_projects_subgraph()`
- `_build_executive_summary_subgraph()`

Each of these methods will return a compiled `StateGraph`.

**Internal Subgraph Flow:**
1.  **`START`**: Entry point.
2.  **`{section}_writer`**: The existing node for writing content (e.g., `key_qualifications_writer_node`).
3.  **`qa`**: The existing `qa_node` for quality checks.
4.  **`await_user_feedback`**: A node that pauses for user input (initially simulated).
5.  **Conditional Edge**: Based on `state.user_feedback`:
    - If `REGENERATE`: Go back to the `{section}_writer` node.
    - If `CONTINUE` (or no feedback): Proceed to `END`.
6.  **`END`**: Exit point of the subgraph.

### 3.2. Supervisor Node

A new `supervisor_node` method will be implemented. Its logic will be:

1.  Check for errors in `state.error_messages`. If any, route to `error_handler`.
2.  Check if `state.current_section_index` has reached the end of `WORKFLOW_SEQUENCE`. If so, route to `formatter`.
3.  Otherwise, get the next section name from `WORKFLOW_SEQUENCE` and route to the corresponding subgraph (e.g., `key_qualifications_subgraph`).

### 3.3. Main Graph Construction (`_build_graph`)

The main graph will be simplified to wire the components together:

1.  **Add Nodes**:
    - Initial processing nodes (`process_job_description`, `process_cv`, etc.).
    - The `supervisor_node`.
    - The compiled subgraphs, added as nodes (e.g., `workflow.add_node("key_qualifications_subgraph", self._build_key_qualifications_subgraph())`).
    - Final nodes (`formatter`, `error_handler`).

2.  **Define Edges**:
    - `START` -> `process_job_description`.
    - `process_job_description` -> `process_cv` -> `research` -> `cv_analysis`.
    - `cv_analysis` -> `supervisor`.
    - Add a conditional edge from the `supervisor` to each subgraph and the `formatter`.
    - Each subgraph node will have an edge back to the `supervisor` to create the main loop.
    - `formatter` -> `END`.
    - `error_handler` -> `END`.

## 4. Implementation Steps

1.  **Implement Subgraph Builders**: Create the `_build_*_subgraph` methods in `cv_workflow_graph.py`.
2.  **Implement Supervisor**: Create the `supervisor_node` and the feedback/decision logic (`await_user_feedback_node`, `decide_next_step_in_subgraph`).
3.  **Refactor `_build_graph`**: Re-wire the main graph according to the new architecture.
4.  **Update State Transitions**: Ensure that `current_section_index` is incremented correctly after a subgraph successfully completes and before control is returned to the supervisor.
5.  **Testing**: Thoroughly test the new graph to ensure the routing logic is correct and state is managed properly across the supervisor and subgraphs.