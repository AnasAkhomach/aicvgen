# Refactoring Plan: Template-Driven CV Structure

## 1. Executive Summary

This document addresses a critical architectural flaw in the `aicvgen` project's design: the dynamic, agent-driven creation of the `StructuredCV` object. The current implementation, where parser agents define the document's structure, is brittle, inflexible, and violates the Single Responsibility Principle.

The proposed solution is to refactor the system to a **"Template-Driven Structure"** pattern. In this new architecture, a CV template file (e.g., in Markdown format) will serve as the single source of truth for the document's schema. Agents will no longer create the structure; they will intelligently *populate* a pre-defined skeleton loaded from the template. This change will make the system significantly more robust, flexible, and maintainable.

## 2. The Core Problem: The "Parser-as-Definer" Anti-Pattern

- **Current Workflow:** `Raw CV Text` -> `UserCVParserAgent` -> `StructuredCV`
- **The Flaw:** The parser agent is responsible for both extracting information and defining the output document's structure. This is incorrect. Changing the CV layout requires changing the agent's Python code.

## 3. The Solution: The "Template-Driven Structure" Pattern

- **New Workflow:**
  1.  **Load Template:** A new service reads a `.md` template file and creates a `StructuredCV` skeleton (all sections defined, but with empty content).
  2.  **Ingest Data:** The user's raw CV text is read.
  3.  **Fill Skeleton:** The `AgentState` is initialized with the skeleton and raw text. The `LangGraph` workflow then executes, with each agent's sole purpose being to find its designated section in the skeleton and populate it.

## 4. Detailed Refactoring Plan

### Step 1: Implement the `CVTemplateLoaderService`

A new, dedicated service is required to handle the parsing of template files.

- **Action:** Create a new file: `src/services/cv_template_loader_service.py`.
- **Responsibilities:**
    -   Define a method `load_from_markdown(file_path: str) -> StructuredCV`.
    -   This method will read a Markdown file and parse its headers (`##` for sections, `###` for subsections) to build a `StructuredCV` object.
    -   All `items` lists within the created `Section` and `Subsection` objects will be initialized as empty.
- **Example Implementation:**
  ```python
  # In src/services/cv_template_loader_service.py
  import re
  from src.models.cv_models import StructuredCV, Section, Subsection, Item

  class CVTemplateLoaderService:
      def load_from_markdown(self, file_path: str) -> StructuredCV:
          """Parses a markdown file to create a structured CV skeleton."""
          with open(file_path, 'r', encoding='utf-8') as f:
              content = f.read()
          # Logic to parse headers and create Section/Subsection objects
          # with empty 'items' lists.
          # ...
          return StructuredCV(sections=parsed_sections)
  ```

### Step 2: Modify the Application's Startup Sequence

The application's entry point must be updated to use the new service *before* the `LangGraph` workflow is invoked.

- **Action:** Modify the startup logic (e.g., in `app.py` or `src/core/main.py`).
- **New Sequence:**
  1.  Instantiate `CVTemplateLoaderService`.
  2.  Call `load_from_markdown()` with the path to the chosen template file to get the `cv_skeleton`.
  3.  Read the user's raw CV text into a `cv_text` variable.
  4.  Create the initial `AgentState` dictionary, pre-populating it:
      ```python
      initial_state = {
          "structured_cv": cv_skeleton,
          "cv_text": cv_text,
          # ... other initial fields like session_id
      }
      ```
  5.  Invoke the `LangGraph` workflow with this `initial_state`.

### Step 3: Deprecate the `UserCVParserAgent`

The original parser agent's primary role is now obsolete.

- **Action:** Remove the `cv_parser_node` from the `main_graph.py` definition. The responsibility of parsing is now distributed among the specialized writer agents, who will parse the `cv_text` in the context of the specific section they are trying to fill.

### Step 4: Refactor All Writer Agents

This is the most critical implementation change. All agents that generate content must be updated to work with the pre-existing skeleton.

- **Action:** Modify the `_execute` method of every writer agent (e.g., `KeyQualificationsWriterAgent`, `ProfessionalExperienceWriterAgent`).
- **New Agent Logic:**
  1.  **Receive State:** The agent receives the `AgentState` containing the `structured_cv` skeleton and the `raw_cv_text`.
  2.  **Find Target Section:** The agent iterates through `state.structured_cv.sections` to find the specific section it is responsible for (e.g., where `section.name == "Key Qualifications"`). It should raise a configuration error if the section is not found in the template.
  3.  **Generate Content:** The agent uses its LLM to generate content (e.g., a list of qualification strings), using the `raw_cv_text` and other data as context.
  4.  **Populate Skeleton:** The agent creates `Item` objects from the generated content and populates the `items` list of the `target_section` it found in step 2.
  5.  **Return Updated State:** The agent returns a dictionary containing the *entire, updated `structured_cv` object*. This ensures the change is propagated immutably through the graph.

  **Example Snippet for a Writer Agent:**
  ```python
  # New logic inside an agent's _execute method
  cv_skeleton = state.get("structured_cv")
  target_section = find_section_by_name(cv_skeleton, "Professional Experience")
  # ... (handle case where section is not found)

  generated_content = await self._generate_experience_items(...)
  target_section.items = [Item(**item_data) for item_data in generated_content]

  return {"structured_cv": cv_skeleton.copy(deep=True)}
  ```

## 5. Benefits of This Refactoring

- **Decoupling:** The CV's structure (data) is now completely decoupled from the content generation logic (code).
- **Flexibility:** To change the CV layout, you now simply edit a Markdown template file. No code changes are needed.
- **Maintainability:** Agents become simpler and more focused. Their single responsibility is to fill a specific, predefined part of a structure.
- **Robustness:** The workflow starts with a known, predictable structure, eliminating an entire class of potential runtime errors caused by inconsistent LLM parsing.
