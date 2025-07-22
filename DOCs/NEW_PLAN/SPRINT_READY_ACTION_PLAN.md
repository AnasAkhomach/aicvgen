# Technical Debt Remediation: Sprint-Ready Action Plan

## Introduction

This document provides a sprint-ready backlog of development tickets to systematically address the architectural technical debt identified in the `FINAL_CONSOLIDATED_ARCHITECTURAL_AUDIT.md`. The work is broken down into five main epics, each corresponding to a core pillar of the refactoring plan.

---

## Epic 1: State Management Refactoring (`STATE-MGT`)

*Objective: Refactor the state management system from a mutable Pydantic `BaseModel` to a composable, immutable `TypedDict` architecture.*

---

### **Work Item ID:** `STATE-MGT-01`
**Task Title:** Convert `AgentState` from `BaseModel` to `TypedDict`

**Acceptance Criteria (AC):**
1.  The `AgentState` class in `src/orchestration/state.py` is redefined using `typing_extensions.TypedDict`.
2.  All methods (e.g., `add_error_message`, `set_current_item`) are removed from the `AgentState` definition.
3.  The application runs without type errors related to the old `BaseModel` methods.
4.  All nodes in the graph that previously called methods on the state object are identified and marked for refactoring (to be completed in subsequent tickets).

**Technical Implementation Notes:**
-   Locate `src/orchestration/state.py`.
-   Change the class definition from `class AgentState(BaseModel):` to `class AgentState(TypedDict):`.
-   Delete all `def` blocks within the class.
-   Run a project-wide search for method calls like `.add_error_message(` or `.set_current_item(` to identify all locations that need to be updated to use the new declarative update pattern.

---

### **Work Item ID:** `STATE-MGT-02`
**Task Title:** Decompose Monolithic State into `GlobalState` and Composable Subgraph States

**Acceptance Criteria (AC):**
1.  A new `GlobalState(TypedDict)` is defined in `src/orchestration/state.py`, containing fields common to the entire workflow (e.g., `structured_cv`, `session_id`).
2.  For each subgraph (e.g., Key Qualifications), a new `TypedDict` state is created that inherits from `GlobalState` (e.g., `class KeyQualificationsState(GlobalState):`).
3.  Each subgraph state definition includes only the *additional* fields it is responsible for creating or modifying.
4.  The main graph definition in `main_graph.py` is updated to use `GlobalState` as its schema.
5.  Each subgraph definition is updated to use its new, specific state schema.

**Technical Implementation Notes:**
-   This task directly follows `STATE-MGT-01`.
-   Identify the fields in the original `AgentState` that are universally required. Move these into the new `GlobalState`.
-   For each subgraph (Key Qualifications, Professional Experience, etc.), create a new `TypedDict` that inherits from `GlobalState` and adds the specific output fields for that graph (e.g., `generated_key_qualifications: Optional[List[str]]`).

---

### **Work Item ID:** `STATE-MGT-03`
**Task Title:** Implement Declarative List Updates with `typing.Annotated`

**Acceptance Criteria (AC):**
1.  The `error_messages` field in `GlobalState` is redefined using `Annotated[List[str], operator.add]`.
2.  Any other list-based fields that accumulate data over the workflow are similarly updated.
3.  All nodes that previously appended to these lists are refactored to return only the *new* item(s) in a list.
4.  The graph correctly and automatically appends the returned items to the state list.

**Technical Implementation Notes:**
-   Import `Annotated` from `typing` and `operator` from the standard library.
-   Change the type hint for `error_messages` in `GlobalState`.
-   Search for all nodes that returned a manually concatenated list for `error_messages` and simplify their return statement to `return {"error_messages": ["new error message"]}`.

---

## Epic 2: CV Structure Refactoring (`CV-STRUCT`)

*Objective: Decouple the CV's structure from the agent logic by making it driven by a template file.*

---

### **Work Item ID:** `CV-STRUCT-01`
**Task Title:** Implement `CVTemplateLoaderService`

**Acceptance Criteria (AC):**
1.  A new file exists at `src/services/cv_template_loader_service.py`.
2.  The file contains a `CVTemplateLoaderService` class with a method `load_from_markdown(file_path: str) -> StructuredCV`.
3.  The method successfully parses a Markdown file, creating `Section` and `Subsection` objects based on `##` and `###` headers.
4.  The `items` list for all created sections and subsections is initialized as empty.
5.  The service raises a clear error if the template file is not found or is malformed.

**Technical Implementation Notes:**
-   Use Python's `re` module for parsing headers. A simple regex like `r'^##\s+(.*)'` can capture section titles.
-   The service should be stateless.
-   Ensure the returned `StructuredCV` object passes Pydantic validation.

---

### **Work Item ID:** `CV-STRUCT-02`
**Task Title:** Refactor Application Startup to be Template-Driven

**Acceptance Criteria (AC):**
1.  The application's main entry point (e.g., `app.py`) is modified to use the `CVTemplateLoaderService` at startup.
2.  The `LangGraph` workflow is now invoked with an initial state that contains a pre-populated `structured_cv` "skeleton" and the raw `cv_text`.
3.  The `UserCVParserAgent` and its corresponding `cv_parser_node` are completely removed from the main workflow graph.

**Technical Implementation Notes:**
-   The startup sequence must be: 1. Load template to create skeleton. 2. Load raw CV text. 3. Create initial state dictionary. 4. Invoke graph.
-   This change will likely break all writer agents until `CV-STRUCT-03` is completed. This is expected.

---

### **Work Item ID:** `CV-STRUCT-03`
**Task Title:** Refactor Writer Agents to Populate the CV Skeleton

**Acceptance Criteria (AC):**
1.  Every writer agent (Key Qualifications, Professional Experience, etc.) is refactored.
2.  Each agent's `_execute` method now finds its target section within the `structured_cv` skeleton received from the state.
3.  Each agent generates content and populates the `items` list of its target section.
4.  Each agent returns the *entire, modified `structured_cv` object* in its output dictionary to ensure immutable state updates.
5.  An agent raises a configuration error if its target section is not found in the provided skeleton.

**Technical Implementation Notes:**
-   The core logic change is from "create and return content" to "find, populate, and return the whole structure."
-   Use `return {"structured_cv": cv_skeleton.copy(deep=True)}` to ensure the state update is immutable.

---

## Epic 3: Agent & Utility Refactoring (`AGENT-UTIL`)

*Objective: Refactor agents to be lean, declarative LCEL chains and replace custom utilities with library-based solutions.*

---

### **Work Item ID:** `AGENT-UTIL-01`
**Task Title:** Deprecate Custom JSON and Retry Utilities

**Acceptance Criteria (AC):**
1.  The file `src/utils/json_utils.py` is deleted.
2.  The file `src/error_handling/classification.py` is deleted.
3.  All imports from these files are removed from the project.

**Technical Implementation Notes:**
-   This is a preparatory cleanup step. The logic will be replaced in the following tickets.

---

### **Work Item ID:** `AGENT-UTIL-02`
**Task Title:** Refactor Agents to use LCEL and `PydanticOutputParser`

**Acceptance Criteria (AC):**
1.  At least one "writer" agent (e.g., `KeyQualificationsWriterAgent`) is fully refactored to use an LCEL chain.
2.  The agent's logic is composed of a `PromptTemplate`, the LLM, and a `PydanticOutputParser`.
3.  A Pydantic model is defined for the expected JSON structure of the agent's output.
4.  The agent's complex `_execute` method is replaced with a simple `chain.ainvoke()` call.
5.  The custom JSON parsing logic is no longer used by the refactored agent.

**Technical Implementation Notes:**
-   Follow the detailed plan in `REFACTORING_PLAN_LCEL.md`.
-   The `PydanticOutputParser` is key here, as it will handle both the parsing and validation of the LLM's output, replacing `json_utils.py`.
-   This ticket can be templated and repeated for every agent in the system.

---

### **Work Item ID:** `AGENT-UTIL-03`
**Task Title:** Refactor `tenacity` Retry Logic

**Acceptance Criteria (AC):**
1.  The `tenacity` `@retry` decorators are updated to use explicit exception types (`retry_if_exception_type`) instead of string matching.
2.  Custom predicates are used for transient errors that don't have a specific exception type (e.g., checking for "503" in an error message).
3.  The logic from the old `classification.py` file is now fully replaced by the new, more robust `tenacity` configuration.

**Technical Implementation Notes:**
-   Import specific exception classes from your API client libraries (e.g., `google.api_core.exceptions.ResourceExhausted`).
-   Use the `|` operator to combine multiple retry conditions (e.g., `retry_if_exception_type(TypeError) | retry_if_exception_type(ValueError)`).

---

## Epic 4: Workflow & DI Refactoring (`WORKFLOW-DI`)

*Objective: Decouple the monolithic workflow graph and implement a true Dependency Injection container.*

---

### **Work Item ID:** `WORKFLOW-DI-01`
**Task Title:** Restructure Orchestration Layer into Modular Directories

**Acceptance Criteria (AC):**
1.  New directories `src/orchestration/graphs` and `src/orchestration/nodes` are created.
2.  Node logic from `CVWorkflowGraph` is moved into standalone functions in the `/nodes` directory, separated by concern (e.g., `parsing_nodes.py`, `content_nodes.py`).
3.  The `CVWorkflowGraph` class is deleted.
4.  A new `main_graph.py` file is created to declaratively assemble the graph from the imported node functions.

**Technical Implementation Notes:**
-   This is a significant structural change. It's best to do this in a dedicated branch.
-   The new node functions will be broken until the DI container is implemented in the next ticket, as they won't have access to their agent dependencies.

---

### **Work Item ID:** `WORKFLOW-DI-02`
**Task Title:** Implement and Integrate Dependency Injection Container

**Acceptance Criteria (AC):**
1.  The `src/core/container.py` file is populated with providers for all services and agents.
2.  Services (e.g., `LLMService`) are configured as `providers.Singleton`.
3.  Agents are configured as `providers.Factory`.
4.  All refactored node functions now retrieve their dependencies from the DI container at runtime (e.g., `agent = Container.key_qualifications_writer_agent()`).
5.  The main `app.py` is responsible for creating and wiring the container at startup.
6.  All global `get_...()` service locator functions are removed.

**Technical Implementation Notes:**
-   This ticket fixes the broken state left by `WORKFLOW-DI-01`.
-   The key is to ensure no component instantiates its own dependencies. Everything should come from the container.

---

## Epic 5: Observability (`OBSERVE`)

*Objective: Integrate LangSmith for best-in-class debugging and tracing.*

---

### **Work Item ID:** `OBSERVE-01`
**Task Title:** Configure and Enable LangSmith Tracing

**Acceptance Criteria (AC):**
1.  The `.env` file is updated with the required LangSmith environment variables (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`).
2.  When the application is run, traces for the `LangGraph` workflow appear in the specified LangSmith project.
3.  The traces correctly show the execution flow between all nodes in the graph.
4.  (Optional but recommended) The `configurable` dictionary in the graph invocation is updated to include additional metadata like `user_id` or `run_type` for easier filtering.

**Technical Implementation Notes:**
-   This is a configuration-only change and requires no code modification to enable.
-   Refer to the LangSmith documentation for obtaining an API key.
-   The `thread_id` already present in your `configurable` config is a great start and will automatically correlate runs within a session.
