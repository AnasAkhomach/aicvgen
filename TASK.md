Here is the definitive, sprint-ready action plan to fix the workflow.

  ---


  Phase 1: Correct the Core State Propagation
  (Goal: Fix the central state management loop to correctly handle updates, which will make the graph functional again.)


  Work Item ID: LG-FIX-01
  Task Title: Correct State Update Logic in trigger_workflow_step


  Acceptance Criteria (AC):
   1. The astream loop within trigger_workflow_step in src/orchestration/cv_workflow_graph.py is refactored.
   2. The logic that incorrectly attempts to create a new AgentState (e.g., state = AgentState(**node_result)) is completely removed.
   3. It is replaced with the correct state update pattern: state = state.model_copy(update=update_dict), where update_dict is the dictionary
      of changes received from the stream.
   4. The application log must clearly show the state being progressively and correctly updated after each node executes.
   5. The 'NoneType' object has no attribute 'model_dump_json' error is eliminated.


  Technical Implementation Notes:
   * File: src/orchestration/cv_workflow_graph.py
   * Method: trigger_workflow_step
   * Guidance: This is the most critical ticket. The astream method yields dictionaries in the format {node_name: update_dict}. You must
     extract the update_dict from this structure and use it to update the existing state object for the next step. This is the core principle
     of state management in LangGraph.

  ---


  Phase 2: Ensure Functional Compliance
  (Goal: Ensure all components adhere to the stateless, functional paradigm required by LangGraph.)


  Work Item ID: LG-FIX-02
  Task Title: Audit and Refactor All Graph Nodes to Return Dictionaries


  Acceptance Criteria (AC):
   1. A complete audit of every node function in cv_workflow_graph.py is performed.
   2. Any node that is not returning a Dict[str, Any] is refactored to do so.
   3. All instances of state.model_copy(update=...) are removed from all node functions.
   4. The code for all nodes compiles without type errors.


  Technical Implementation Notes:
   * File: src/orchestration/cv_workflow_graph.py
   * Guidance: This is a cleanup and verification step. Go through every node and ensure it follows the functional paradigm. This will prevent
     future state-related bugs.

  ---


  Work Item ID: LG-FIX-03
  Task Title: Audit and Refactor All Agent Implementations to be Stateless


  Acceptance Criteria (AC):
   1. A complete audit of src/agents/agent_base.py and all agent classes in src/agents/ is performed.
   2. The run_as_node method in AgentBase and all implementing classes is confirmed to return Dict[str, Any].
   3. Agents no longer return AgentResult objects or AgentState objects.


  Technical Implementation Notes:
   * Files: src/agents/agent_base.py and all files in src/agents/.
   * Guidance: This ensures the agents are fully decoupled from the workflow's state management, making them more modular and easier to test.

  ---


  Phase 3: Validation and Hardening
  (Goal: Verify the fix and add tests to prevent future regressions.)


  Work Item ID: LG-FIX-04
  Task Title: Create Unit Test for trigger_workflow_step State Propagation


  Acceptance Criteria (AC):
   1. A new unit test file is created for cv_workflow_graph.py.
   2. A specific unit test is created for the trigger_workflow_step method.
   3. This test uses unittest.mock to create a mock app.astream that yields a predefined sequence of update dictionaries.
   4. The test asserts that the state object within trigger_workflow_step is correctly and progressively updated after each yielded dictionary.


  Technical Implementation Notes:
   * File: tests/unit/test_cv_workflow_graph.py (or a new file).
   * Guidance: This test is critical for preventing this class of bug from ever happening again. It will validate the core state management
     loop in isolation.


  ---


  Work Item ID: LG-FIX-05
  Task Title: Full End-to-End Workflow Validation


  Acceptance Criteria (AC):
   1. The entire, refactored workflow executes successfully from a clean state without any errors.
   2. A resumed workflow (where parsing nodes are skipped) executes successfully without any errors.
   3. All Pydantic and NoneType errors are eliminated from all possible execution paths.
   4. All existing integration tests are updated to reflect the refactored architecture and are passing.


  Technical Implementation Notes:
   * Files: src/orchestration/cv_workflow_graph.py, tests/
   * Guidance: This is the final sign-off. The development team must rigorously test all user flows and edge cases to ensure the refactoring
     is complete and correct.