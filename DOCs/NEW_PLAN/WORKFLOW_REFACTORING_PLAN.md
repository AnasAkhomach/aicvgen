# Workflow Refactoring Plan: Aligning with `langgraph-course`

## 1. Objective

This document outlines a strategic plan to refactor the `aicvgen` project's workflow orchestration layer. The goal is to align our architecture with the modular and maintainable patterns demonstrated in the `emarco177/langgraph-course` repository. This will involve decoupling the monolithic `cv_workflow_graph.py` into a more organized structure, improving readability, and adopting advanced agentic patterns.

## 2. Architectural Comparison and Motivation

The `langgraph-course` repository, particularly its advanced projects like `agentic-rag`, promotes a decoupled architecture where graph components are separated into logical modules.

| Feature | `langgraph-course` (Target Architecture) | `aicvgen` (Current Architecture) |
| :--- | :--- | :--- |
| **Graph Definition** | Modular: separate files for state, nodes, and graph assembly. | Monolithic: `cv_workflow_graph.py` contains all logic. |
| **Nodes** | Standalone functions in a dedicated `nodes.py` file. | Methods within the `CVWorkflowGraph` class. |
| **Agent Logic** | Composed within the graph definition, using nodes as building blocks. | Encapsulated in separate agent classes, called by node methods. |

By adopting the target architecture, we will make the workflow easier to understand, test, and extend.

## 3. Step-by-Step Refactoring Plan

### Phase 1: Restructure the Orchestration Layer

The first step is to create a new directory structure within `src/orchestration` that promotes modularity.

1.  **Create New Directories and Files:**
    *   `src/orchestration/graphs/`: This new directory will house all graph and subgraph definitions.
        *   `__init__.py`
        *   `main_graph.py`: For the primary workflow assembly.
        *   `subgraphs.py`: For building the various subgraphs (Key Qualifications, etc.).
    *   `src/orchestration/nodes/`: This new directory will contain all node functions, separated by concern.
        *   `__init__.py`
        *   `parsing_nodes.py`: For `jd_parser_node`, `cv_parser_node`.
        *   `content_nodes.py`: For all writer and updater nodes.
        *   `utility_nodes.py`: For `supervisor_node`, `formatter_node`, `error_handler_node`, etc.
    *   `src/orchestration/state.py`: This file already exists and is correctly placed. No move is needed.

### Phase 2: Decouple Nodes from the Graph Class

The core of the refactoring is to move the node logic from methods inside the `CVWorkflowGraph` class into standalone, importable functions in the newly created `nodes` files.

**Example Transformation (`jd_parser_node`):**

*   **Current Location:** `src/orchestration/cv_workflow_graph.py` (as a class method)
*   **New Location:** `src/orchestration/nodes/parsing_nodes.py` (as a standalone function)

**Implementation Detail:**
The original methods relied on `self` to access agent instances (e.g., `self.job_description_parser_agent`). The new standalone functions will need access to these agents. The recommended approach is to use a dependency injection container to provide the necessary agent instances to the node functions when the graph is built.

**Before (in `cv_workflow_graph.py`):**
```python
class CVWorkflowGraph:
    # ...
    async def jd_parser_node(self, state: AgentState) -> Dict[str, Any]:
        # ... logic using self.job_description_parser_agent ...
```

**After (in `src/orchestration/nodes/parsing_nodes.py`):**
```python
from src.orchestration.state import AgentState
from src.container import Container # Assuming a DI container
from src.utils.node_validation import validate_node_output

@validate_node_output
async def jd_parser_node(state: AgentState) -> dict[str, Any]:
    """Parses the job description from the state."""
    jd_parser_agent = Container.job_description_parser_agent()
    # The agent is now retrieved from the container, not from 'self'
    result = await jd_parser_agent.run_as_node(state)
    return result
```
This process will be repeated for every node in the original `CVWorkflowGraph` class.

### Phase 3: Rebuild the Graph in a Modular Way

With the nodes refactored into separate modules, the graph can be reassembled in `src/orchestration/graphs/main_graph.py`. This file will be much cleaner and will focus solely on the structure and flow of the workflow.

**Example (in `src/orchestration/graphs/main_graph.py`):**
```python
from langgraph.graph import StateGraph, END
from src.orchestration.state import AgentState
from src.orchestration.nodes import parsing_nodes, content_nodes, utility_nodes
from .subgraphs import build_key_qualifications_subgraph # Import subgraph builders

def build_main_workflow() -> StateGraph:
    """Builds and returns the main CV generation workflow graph."""
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("jd_parser", parsing_nodes.jd_parser_node)
    workflow.add_node("cv_parser", parsing_nodes.cv_parser_node)
    workflow.add_node("supervisor", utility_nodes.supervisor_node)
    # ... add all other nodes

    # 2. Add Subgraphs
    kq_subgraph = build_key_qualifications_subgraph()
    workflow.add_node("key_qualifications_subgraph", kq_subgraph)
    # ... add all other subgraphs

    # 3. Define Edges
    workflow.set_entry_point("jd_parser")
    workflow.add_edge("jd_parser", "cv_parser")
    # ... define all other edges and conditional routes

    return workflow.compile()

# Global compiled app instance
app = build_main_workflow()
```

### Phase 4: Adopt Advanced Agentic Patterns

The `langgraph-course` repository provides excellent templates for more advanced agent behaviors. We should plan to refactor our existing agents to incorporate these patterns.

1.  **`ResearchAgent` -> ReAct Pattern:**
    *   **Objective:** Refactor the `ResearchAgent` to use the Reason-Act (ReAct) pattern.
    *   **Steps:**
        1.  Define a set of tools for the agent (e.g., `web_search`, `read_file`).
        2.  Create a new `StateGraph` for the ReAct agent.
        3.  The graph will loop: the LLM **reasons** which tool to use, the agent **acts** by executing the tool, and the result is fed back into the loop until the research is complete.

2.  **Writer Agents -> Reflection/Self-Correction Pattern:**
    *   **Objective:** Enhance the writer agents (e.g., `ProfessionalExperienceWriterAgent`) with a self-correction loop.
    *   **Steps:**
        1.  After the initial content is generated, add a `reflection_node`. This node uses an LLM to critique the generated text based on quality criteria.
        2.  Add a `revision_node` that takes the original text and the critique as input and generates an improved version.
        3.  This creates a subgraph: `generate` -> `reflect` -> `revise`.

## 4. Conclusion

This refactoring represents a significant architectural improvement. It will result in a more modular, maintainable, and scalable workflow. By separating concerns and adopting advanced agentic patterns, we will align the `aicvgen` project with the best practices of modern, production-grade AI engineering.
