# Comprehensive Architectural Refactoring Plan for aicvgen

## 1. Executive Summary

This document presents a unified and comprehensive architectural refactoring plan for the `aicvgen` project. It synthesizes the findings from four previous analyses (`STATE_REFACTORING_PLAN.md`, `TEMPLATE_DRIVEN_REFACTORING_PLAN.md`, `REFACTORING_PLAN_LCEL.md`, and `WORKFLOW_REFACTORING_PLAN.md`) and addresses the final, overarching issue of dependency management.

The goal is to evolve `aicvgen` from its current state into a truly modular, flexible, and production-grade application by systematically adopting modern LangGraph design patterns. This plan provides a clear, step-by-step roadmap for this transformation.

## 2. The Four Pillars of Refactoring

Our new architecture will be built upon four core pillars, each addressing a specific anti-pattern in the current implementation.

### Pillar 1: State Management - From Mutable Class to Immutable `TypedDict`

- **Anti-Pattern:** The current `AgentState` is a monolithic, mutable Pydantic `BaseModel` that encourages imperative state updates.
- **Remediation (from `STATE_REFACTORING_PLAN.md`):**
    1.  **Convert to `TypedDict`:** Change `AgentState` to be a `TypedDict` to enforce its role as a pure data container.
    2.  **Decompose State:** Break the monolithic state into a `GlobalState` and smaller, composable `TypedDict`s for each subgraph (e.g., `KeyQualificationsState(GlobalState)`).
    3.  **Use `Annotated` for Updates:** Use `typing.Annotated` to declaratively manage list updates (e.g., for `error_messages`), simplifying node logic.

### Pillar 2: CV Structure - From Agent-Defined to Template-Driven

- **Anti-Pattern:** Parser agents currently define the structure of the final CV, making the layout rigid and hard to change.
- **Remediation (from `TEMPLATE_DRIVEN_REFACTORING_PLAN.md`):**
    1.  **Implement `CVTemplateLoaderService`:** Create a service to parse a Markdown template file and generate a `StructuredCV` object that serves as a "skeleton."
    2.  **Modify Startup Sequence:** The application will now start by loading this skeleton *before* invoking the LangGraph workflow.
    3.  **Redefine Agent Roles:** Writer agents will no longer create sections. Their new role is to find their designated section within the skeleton and populate it with content. The original `UserCVParserAgent` will be deprecated.

### Pillar 3: Agent Architecture - From "Fat Agents" to "Lean Chains"

- **Anti-Pattern:** Agents are "fat," containing complex, imperative logic for prompt formatting, LLM calls, and output parsing.
- **Remediation (from `REFACTORING_PLAN_LCEL.md`):**
    1.  **Adopt LCEL:** Refactor every agent to be a thin wrapper around a LangChain Expression Language (LCEL) chain.
    2.  **Compose with Primitives:** Each chain will be a declarative composition of `PromptTemplate`, the LLM, and an `OutputParser`.
    3.  **Benefits:** This makes agents simpler, more readable, easier to test, and tightly integrated with the LangChain ecosystem (e.g., for streaming, logging).

### Pillar 4: Workflow Orchestration - From Monolithic to Modular

- **Anti-Pattern:** The `CVWorkflowGraph` class is a monolithic "God Object" that contains all node definitions, routing logic, and dependency management.
- **Remediation (from `WORKFLOW_REFACTORING_PLAN.md`):**
    1.  **Restructure Files:** Create a new, modular directory structure (`src/orchestration/graphs/`, `src/orchestration/nodes/`).
    2.  **Extract Nodes:** Convert all node methods from the `CVWorkflowGraph` class into standalone functions within the new `nodes` modules.
    3.  **Rebuild Graph Declaratively:** In a new `main_graph.py`, assemble the workflow by importing and connecting the standalone node functions.

## 3. The Final Piece: True Dependency Injection

- **Overarching Anti-Pattern:** The project suffers from a "Service Locator" pattern and lacks true dependency injection. The old `CVWorkflowGraph` class acted as a manual constructor for all its dependencies.
- **Remediation:**
    1.  **Fully Utilize `dependency-injector`:** The existing `src/core/container.py` must become the single source of truth for object construction.
    2.  **Define All Providers:** Create `providers.Singleton` for all services (`LLMService`, `CVTemplateLoaderService`, etc.) and `providers.Factory` for all agents in the container.
    3.  **Inject into Nodes:** Node functions should not instantiate anything. They should retrieve their required dependencies (e.g., an agent instance) directly from the container at the start of their execution.
    4.  **Wire at Startup:** The main `app.py` will be responsible for creating the container and wiring it with configuration, making it available to the rest of the application.

## 4. Suggested Implementation Order

To ensure a smooth transition, the refactoring should be done in the following order, as each step builds upon the last:

1.  **State Management (Pillar 1):** Start by refactoring the state object. A clean state model is the foundation for everything else.
2.  **Template-Driven Structure (Pillar 2):** Implement the template loader and change the startup sequence. This defines *what* the agents will be working on.
3.  **Agent Architecture (Pillar 3):** Refactor the agents to use LCEL. This changes *how* the agents work.
4.  **Workflow Orchestration (Pillar 4):** Break apart the monolithic graph class into the new modular structure.
5.  **Dependency Injection (Final Piece):** With everything else in place, implement the DI container to cleanly manage the construction and provision of all components.

By completing this comprehensive refactoring, the `aicvgen` project will be transformed into a highly modular, maintainable, and scalable system that fully leverages the power and elegance of the LangGraph framework.
