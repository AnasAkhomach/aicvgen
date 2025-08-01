  Corrected Remediation Plan: Integrating the CV Parser as a Workflow Node

  ---

  Work Item ID: REM-FIX-01-REVISED
  Task Title: Modernize UserCVParserAgent and Define it as a Workflow Node

  Acceptance Criteria (AC):
   1. The UserCVParserAgent is refactored to be a standard, modern agent that uses the LLMCVParserService for its core logic.
   2. A new node function, user_cv_parser_node, is created in src/orchestration/nodes/parsing_nodes.py.
   3. This new node is responsible for calling the UserCVParserAgent, taking the raw cv_text from the GlobalState, and writing the resulting
      StructuredCV object back into the GlobalState.
   4. The agent and the new node are fully unit-tested.

  Technical Implementation Notes:

  1. Modernize the Agent (as before)
   * File: src/agents/user_cv_parser_agent.py
   * Action: Refactor the agent to be a lightweight wrapper around LLMCVParserService. This part of the plan remains the same.

  2. Create the New Workflow Node
   * File: src/orchestration/nodes/parsing_nodes.py
   * Action: Add a new node function that will execute the agent.

    1 # In src/orchestration/nodes/parsing_nodes.py
    2
    3 from src.agents.user_cv_parser_agent import UserCVParserAgent
    4 from src.orchestration.state import GlobalState
    5
    6 # ... (other node functions)
    7
    8 async def user_cv_parser_node(state: GlobalState, agent: UserCVParserAgent) -> GlobalState:
    9     """
   10     Parses the raw CV text from the state and updates the state
   11     with the structured CV object.
   12     """
   13     cv_text = state['cv_text']
   14
   15     structured_cv = await agent.run(cv_text=cv_text)
   16
   17     # Update the state with the result and return the modified state
   18     state['structured_cv'] = structured_cv
   19     return state

  ---

  Work Item ID: REM-FIX-02-REVISED
  Task Title: Integrate CV Parser Node into the Main Workflow Graph

  Acceptance Criteria (AC):
   1. The UserCVParserAgent is correctly registered as a provider in the DI container (main_container.py).
   2. The main_graph.py is updated to include the new user_cv_parser_node.
   3. The user_cv_parser_node is now the first node to run after the workflow starts.
   4. The jd_parser_node now runs after the user_cv_parser_node.
   5. The CvGenerationFacade.start_cv_generation method is simplified to only pass the raw text into the initial state of the workflow, without
      any pre-parsing.

  Technical Implementation Notes:

  1. Update DI Container and Graph Definitions
   * File: src/core/containers/main_container.py
       * Action: Ensure the user_cv_parser_agent provider is active.
   * File: src/orchestration/graphs/main_graph.py
       * Action: Import the new user_cv_parser_node. Add it to the create_node_functions dictionary. Add the new node to the graph itself.

  2. Re-wire the Main Graph
   * File: src/orchestration/graphs/main_graph.py
   * Action: Modify the graph's edges to reflect the new, correct flow.

  New Workflow Logic:

    1 # In build_main_workflow_graph function:
    2
    3     # ... (add the new node)
    4     workflow.add_node("USER_CV_PARSER", node_functions["user_cv_parser_node"])
    5
    6     # The entry point is now the CV Parser
    7     workflow.set_entry_point("USER_CV_PARSER")
    8
    9     # Define the new sequential processing chain
   10     workflow.add_edge("USER_CV_PARSER", WorkflowNodes.JD_PARSER.value)
   11     workflow.add_edge(WorkflowNodes.JD_PARSER.value, "INITIALIZE_SUPERVISOR")
   12     # ... (the rest of the graph follows)

  3. Simplify the Facade
   * File: src/core/facades/cv_generation_facade.py
   * Action: The facade no longer needs to know about the parser agent at all. Its job is now even simpler.

    1 # In CvGenerationFacade class
    2
    3 def __init__(self, workflow_manager: WorkflowManager, ...):
    4     # The user_cv_parser_agent is NO LONGER injected here.
    5     self._workflow_manager = workflow_manager
    6     # ...
    7
    8 def start_cv_generation(self, cv_content: str, job_description: str, ...) -> str:
    9     """
   10     Initializes the workflow with RAW text. The graph itself will handle parsing.
   11     """
   12     initial_state = {
   13         "cv_text": cv_content, # Pass the raw text directly
   14         "job_description_data": JobDescriptionData(raw_text=job_description), # Start with a partial object
   15         "structured_cv": None, # It starts as None! The node will fill it.
   16         # ... other initial state fields
   17     }
   18     session_id = self._workflow_manager.start_workflow(initial_state)
   19
   20     # ... rest of the method
