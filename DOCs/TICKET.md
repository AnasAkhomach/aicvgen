  Work Item ID: `AGENT-DI-REMEDIATION-06`

  Task Title: Create a Generic Node Factory to Eliminate Repetitive Node Logic

  Objective: To abstract the common logic of (1) preparing agent inputs from state, (2) calling an agent, and (3) updating the state with the agent's
  output into a reusable factory. This will radically simplify all writer nodes in content_nodes.py, making them declarative and easy to maintain.

  Acceptance Criteria (AC):
   1. A new file, src/orchestration/factories.py, is created and contains the AgentNodeFactory class.
   2. A new file, src/orchestration/node_helpers.py, is created and contains all the "mapper" and "updater" functions.
   3. The content_nodes.py file is refactored to be extremely minimal. It should contain no complex logic, only the instantiation of the
      AgentNodeFactory for each writer node.
   4. All the complex data preparation and state update logic currently in content_nodes.py is successfully moved into the appropriate mapper and
      updater functions in node_helpers.py.
   5. The main graph continues to function identically to the previous version.

  ---

  Technical Implementation Plan

  This plan involves creating two new files and refactoring one existing file.

  Step 1: Create the `node_helpers.py` File (The "Unique Logic")

   * What to do: Create a new file at src/orchestration/node_helpers.py. This file will contain the small, focused functions that handle the unique
     data transformations for each node.

   * Example Implementation for the "Projects" node:

    1     # In src/orchestration/node_helpers.py
    2
    3     from src.orchestration.state import GlobalState
    4     from src.models.cv_models import StructuredCV
    5     from src.agents.projects_writer_agent import ProjectsWriterAgentInput, ProjectLLMOutput
    6     from src.utils.cv_data_factory import get_item_by_id, update_item_by_id
    7
    8     # 1. The MAPPER function
    9     def map_state_to_projects_input(state: GlobalState) -> ProjectsWriterAgentInput:
   10         """Takes the full graph state and maps it to the specific input model for the Projects agent."""
   11         structured_cv = state["structured_cv"]
   12         current_item_id = state["current_item_id"]
   13         project_item = get_item_by_id(structured_cv, current_item_id)
   14
   15         # All the logic for extracting data and preparing strings goes here
   16         key_qualifications_str = ... # Logic to extract from structured_cv
   17         professional_experience_str = ... # Logic to extract from structured_cv
   18
   19         return ProjectsWriterAgentInput(
   20             job_description=state["job_description_data"].raw_text,
   21             project_item=project_item.model_dump(),
   22             key_qualifications=key_qualifications_str,
   23             professional_experience=professional_experience_str,
   24             research_findings=state.get("research_findings")
   25         )
   26
   27     # 2. The UPDATER function
   28     def update_cv_with_project_data(cv: StructuredCV, agent_output: dict, item_id: str) -> StructuredCV:
   29         """Takes the agent's output and updates the CV state."""
   30         generated_data: ProjectLLMOutput = agent_output["generated_project_content"]
   31
   32         # Logic to format the bullet points or description
   33         content_text = "\n".join([f"• {bullet}" for bullet in generated_data.bullet_points])
   34
   35         return update_item_by_id(
   36             cv,
   37             item_id,
   38             {"content": content_text, "status": "completed"}
   39         )
   40
   41     # --- You will create a mapper and an updater for EACH writer agent ---
   42     # e.g., map_state_to_experience_input, update_cv_with_experience_data, etc.

  Step 2: Create the `factories.py` File (The "Boilerplate Logic")

   * What to do: Create a new file at src/orchestration/factories.py. This will contain our generic factory class.

   * Implementation:

    1     # In src/orchestration/factories.py
    2
    3     from typing import Callable, Awaitable
    4     from src.orchestration.state import GlobalState
    5     from src.agents.agent_base import AgentBase # Or a more specific base if you have one
    6
    7     class AgentNodeFactory:
    8         def __init__(
    9             self,
   10             agent: AgentBase,
   11             input_mapper: Callable[[GlobalState], BaseModel], # Takes state, returns Pydantic model
   12             output_updater: Callable[[StructuredCV, dict, str], StructuredCV] # Takes CV, agent output, item_id, returns updated CV
   13         ):
   14             self.agent = agent
   15             self.input_mapper = input_mapper
   16             self.output_updater = output_updater
   17
   18         async def create_node(self, state: GlobalState) -> dict:
   19             """This is the generic node logic that will be executed for every writer."""
   20             # 1. Use the specific mapper to prepare the agent's input
   21             agent_input = self.input_mapper(state)
   22
   23             # 2. Call the agent with the prepared input
   24             agent_output = await self.agent._execute(**agent_input.model_dump())
   25
   26             # 3. Use the specific updater to modify the CV
   27             current_cv = state["structured_cv"]
   28             item_id_to_update = state["current_item_id"]
   29             updated_cv = self.output_updater(current_cv, agent_output, item_id_to_update)
   30
   31             # 4. Return the updated state dictionary
   32             return {"structured_cv": updated_cv}

  Step 3: Radically Simplify `content_nodes.py`

   * What to do: Delete all the complex, repetitive code from content_nodes.py. It will now become a simple, declarative file that just wires together
     the components from the other two files.

   * The New `content_nodes.py`:

    1     # The NEW, clean content_nodes.py
    2
    3     from .factories import AgentNodeFactory
    4     from .node_helpers import (
    5         map_state_to_projects_input,
    6         update_cv_with_project_data,
    7         map_state_to_experience_input,
    8         update_cv_with_experience_data,
    9         # ... import all your other mappers and updaters
   10     )
   11     from ..core.container import get_container # Or your DI mechanism
   12
   13     # Assume 'container' is your DI container instance
   14     container = get_container()
   15
   16     # --- Create the Professional Experience Node ---
   17     experience_factory = AgentNodeFactory(
   18         agent=container.professional_experience_writer_agent(),
   19         input_mapper=map_state_to_experience_input,
   20         output_updater=update_cv_with_experience_data
   21     )
   22     professional_experience_writer_node = experience_factory.create_node
   23
   24     # --- Create the Projects Node ---
   25     projects_factory = AgentNodeFactory(
   26         agent=container.projects_writer_agent(),
   27         input_mapper=map_state_to_projects_input,
   28         output_updater=update_cv_with_project_data
   29     )
   30     projects_writer_node = projects_factory.create_node
   31
   32     # --- Create the Executive Summary Node ---
   33     # ... and so on for the other writer nodes
   34
   35     # The qa_node and other non-writer nodes that have unique logic
   36     # can remain as standalone functions for now.