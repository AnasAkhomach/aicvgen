# Technical Debt Remediation: Sprint-Ready Action Plan

This document provides a sprint-ready backlog of development tickets to systematically resolve the technical debt identified in the `COMPREHENSIVE_AUDIT_REPORT.md`. Each ticket is designed to be an atomic work item that can be picked up by a developer.

---

### **Work Item ID:** `REM-AGENT-001`
**Task Title:** Refactor Writer Agents to Separate Content Generation from State Updates

**Acceptance Criteria (AC):**
1.  All LCEL-based "writer" agents (e.g., `KeyQualificationsWriterAgent`, `ProfessionalExperienceWriterAgent`) are modified to return only the raw generated content (e.g., a dictionary like `{"generated_key_qualifications": [...]}`).
2.  The writer agents no longer contain any logic to find, access, or modify the `structured_cv` object from the state.
3.  Dedicated "updater" agents (e.g., `ProfessionalExperienceUpdaterAgent`) exist for each content type and are responsible for merging the generated content into the `structured_cv` object.
4.  The main graph in `src/orchestration/graphs/main_graph.py` is updated to include a two-node sequence (`writer_node` -> `updater_node`) for each content generation subgraph.
5.  The application runs end-to-end, successfully populating all CV sections using the new two-step process.

**Technical Implementation Notes:**
-   Reference `TD-AGENT-001` from the audit report.
-   In each writer agent (e.g., `KeyQualificationsWriterAgent`), simplify the `_execute` method to only call `self.chain.ainvoke()` and return its raw output in a dictionary. Remove all code that iterates through `structured_cv.sections`.
-   Create new updater agents where they don't exist, modeling them after `KeyQualificationsUpdaterAgent`. Their sole responsibility is to take the generated content from the state and place it correctly within the `structured_cv` object.
-   Modify the `build_*_subgraph` functions in `main_graph.py`. Each subgraph will now have at least two nodes in its main path.

---

### **Work Item ID:** `REM-AGENT-002`
**Task Title:** Refactor `JobDescriptionParserAgent` to Use Dependency Injection

**Acceptance Criteria (AC):**
1.  The `JobDescriptionParserAgent` no longer instantiates `LLMCVParserService` within its `__init__` method.
2.  The `__init__` method of `JobDescriptionParserAgent` is modified to accept an instance of `LLMCVParserService` as an argument.
3.  The DI container in `src/core/container.py` is updated to correctly construct and inject `LLMCVParserService` into the `JobDescriptionParserAgent` provider.
4.  A provider for `LLMCVParserService` is created in the container if one does not already exist.

**Technical Implementation Notes:**
-   Reference `TD-AGENT-002` from the audit report.
-   In `src/agents/job_description_parser_agent.py`, change `__init__` to `def __init__(self, llm_cv_parser_service: LLMCVParserService, ...):`. Remove the line `self.llm_cv_parser_service = LLMCVParserService(...)`.
-   In `src/core/container.py`, first define a provider for the service: `llm_cv_parser_service = providers.Factory(LLMCVParserService, ...)`.
-   Then, update the agent's provider to use the new service provider: `job_description_parser_agent = providers.Factory(JobDescriptionParserAgent, llm_cv_parser_service=llm_cv_parser_service, ...)`.

---

### **Work Item ID:** `REM-AGENT-003`
**Task Title:** Standardize `ProjectsWriterAgent` by Inheriting from `AgentBase`

**Acceptance Criteria (AC):**
1.  The class signature of `ProjectsWriterAgent` in `src/agents/projects_writer_agent.py` is changed to `class ProjectsWriterAgent(AgentBase):`.
2.  The `__init__` method of `ProjectsWriterAgent` calls `super().__init__(...)` with the appropriate `name` and `description`.
3.  The agent's execution logic is contained within an `async def _execute(self, **kwargs: Any) -> dict[str, Any]:` method, matching the `AgentBase` contract.

**Technical Implementation Notes:**
-   Reference `TD-AGENT-003` from the audit report.
-   This is a straightforward refactoring to ensure consistency. Ensure the `_execute` method correctly handles input validation and returns a dictionary compatible with the graph state.

---

### **Work Item ID:** `REM-AGENT-004`
**Task Title:** Centralize Pydantic Model Validation with a Decorator

**Acceptance Criteria (AC):**
1.  A new decorator (e.g., `@ensure_pydantic_model`) is created in `src/utils/node_validation.py`.
2.  This decorator accepts a key (e.g., `"structured_cv"`) and a Pydantic model class (e.g., `StructuredCV`) as arguments.
3.  The decorator's logic inspects the decorated function's arguments, finds the specified key, and if its value is a dictionary, it converts it into an instance of the provided Pydantic model.
4.  The manual `dict`-to-Pydantic conversion logic is removed from `CVAnalyzerAgent`, `QualityAssuranceAgent`, and `KeyQualificationsUpdaterAgent`.
5.  The new decorator is applied to the `_execute` methods of the affected agents.

**Technical Implementation Notes:**
-   Reference `TD-AGENT-004` from the audit report.
-   The decorator can be implemented using `functools.wraps`. It will need to handle both `*args` and `**kwargs` to locate the target key in the function's arguments.
-   Example usage: `@ensure_pydantic_model("structured_cv", StructuredCV)`

---

### **Work Item ID:** `REM-SVC-001`
**Task Title:** Refactor `CVTemplateLoaderService` to be an Injectable Instance

**Acceptance Criteria (AC):**
1.  All `@classmethod` decorators are removed from `CVTemplateLoaderService` in `src/services/cv_template_loader_service.py`.
2.  The method signatures are updated from `(cls, ...)` to `(self, ...)`.
3.  A `providers.Singleton` for `CVTemplateLoaderService` is added to the DI container in `src/core/container.py`.
4.  The `WorkflowManager` is modified to receive the `CVTemplateLoaderService` via dependency injection in its `__init__` method.
5.  The call in `WorkflowManager.create_new_workflow` is updated from `CVTemplateLoaderService.load_from_markdown(...)` to `self.cv_template_loader_service.load_from_markdown(...)`.

**Technical Implementation Notes:**
-   Reference `TD-SVC-001` from the audit report.
-   This change aligns the service with the rest of the application's DI-managed services, improving testability and consistency.

---

### **Work Item ID:** `REM-SVC-002`
**Task Title:** Abstract `LLMClient` to be Provider-Agnostic

**Acceptance Criteria (AC):**
1.  A new abstract base class or interface, `LLMClientInterface`, is created, defining methods like `async generate(...)`.
2.  The existing `LLMClient` class is renamed to `GeminiClient` and is modified to implement `LLMClientInterface`.
3.  The `GeminiClient` is refactored to be instantiated with its API key in its constructor, removing reliance on global `genai.configure` calls.
4.  The DI container is updated to provide a `providers.Factory` for `GeminiClient` where the `LLMClientInterface` is required.

**Technical Implementation Notes:**
-   Reference `TD-SVC-002` from the audit report.
-   This is a foundational change for future flexibility. The goal is to decouple the rest of the application from any specific LLM provider's library.
-   Consider creating a new directory `src/services/llm/` to house the interface and its implementations.

---

### **Work Item ID:** `REM-SVC-003`
**Task Title:** Centralize Duplicated LLM JSON Parsing Logic

**Acceptance Criteria (AC):**
1.  A new file, `src/utils/json_utils.py`, is created.
2.  A new function, `parse_llm_json_response(raw_text: str) -> dict`, is created in the new file.
3.  The robust JSON parsing logic (handling markdown code blocks and cleaning) is moved from `LLMCVParserService` and `ResearchAgent` into this new central function.
4.  Both `LLMCVParserService` and `ResearchAgent` are refactored to call the new utility function, removing the duplicated code.

**Technical Implementation Notes:**
-   Reference `TD-SVC-003` from the audit report.
-   Ensure the new utility function is pure and has no side effects. It should take a string and return a dictionary or raise a specific parsing exception.

---

### **Work Item ID:** `REM-CORE-001`
**Task Title:** Refactor Agent `session_id` Provisioning to Use Explicit Injection

**Acceptance Criteria (AC):**
1.  The global state mechanism (`_current_session_id`, `set_session_id`) is removed from `src/core/container.py`.
2.  The agent providers in the container are modified to accept `session_id` as a runtime argument (e.g., `providers.Factory(MyAgent, session_id=providers.Argument(str))`).
3.  The graph assembly logic in `src/orchestration/graphs/main_graph.py` is updated to retrieve the `session_id` from the state and pass it explicitly when creating an agent instance for a node.

**Technical Implementation Notes:**
-   Reference `TD-CORE-001` from the audit report.
-   This change makes agent dependencies explicit. The code that binds agents to nodes (e.g., using `functools.partial`) will need to be updated to pass the `session_id` during the binding process.

---

### **Work Item ID:** `REM-CORE-002`
**Task Title:** Delete Obsolete `ContentAggregator` Module

**Acceptance Criteria (AC):**
1.  The import statement `from .content_aggregator import ContentAggregator` is removed from `src/core/__init__.py`.
2.  The file `src/core/content_aggregator.py` is deleted.
3.  The associated test file `tests/unit/test_content_aggregator_refactored.py` is deleted.

**Technical Implementation Notes:**
-   Reference `TD-CORE-002` from the audit report.
-   This is a straightforward cleanup task. A global search for "ContentAggregator" can provide a final verification that no active code uses it before deletion.

---

### **Work Item ID:** `REM-ORCH-001`
**Task Title:** Simplify Graph Assembly and Dependency Injection Logic

**Acceptance Criteria (AC):**
1.  The use of `functools.partial` for injecting agents into nodes in `main_graph.py` is removed.
2.  A simpler, more explicit mechanism (like a factory class or higher-order function) is used to configure nodes with their agent dependencies *before* the graph is assembled.
3.  The custom `WorkflowGraphWrapper` class is removed. The `WorkflowManager` now calls the compiled graph's `ainvoke` method directly.
4.  The `create_cv_workflow_graph_with_di` function is refactored or renamed to clearly separate the concerns of DI and graph assembly.

**Technical Implementation Notes:**
-   Reference `TD-ORCH-001` from the audit report.
-   The goal is to make the `build_main_workflow_graph` function a pure, declarative assembly of pre-configured nodes. The dependency injection should happen at a higher level, with the results being passed into the build function.

---

### **Work Item ID:** `REM-ORCH-002`
**Task Title:** Refactor Graph Nodes to be Thin Wrappers

**Acceptance Criteria (AC):**
1.  All node functions in `src/orchestration/nodes/` are refactored to remove complex logic (e.g., `try...except` blocks, manual state merging).
2.  Each node's primary responsibility is to call the `run_as_node` method of its corresponding agent, passing the `state`.
3.  The agent's `_execute` method is responsible for returning a dictionary containing *only the new or modified state fields*.
4.  The node function directly returns the dictionary received from the agent's `run_as_node` method.

**Technical Implementation Notes:**
-   Reference `TD-ORCH-002` from the audit report.
-   This change pushes business logic and error handling into the agents, where it belongs, and makes the graph definition cleaner and more focused on orchestration. The `AgentBase.run_as_node` method should already handle the necessary input extraction.
