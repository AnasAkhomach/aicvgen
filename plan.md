  Here's the proposed architecture and the plan to implement it:

  Proposed Architecture: Supervisor-Orchestrated Subgraphs


  This approach leverages LangGraph's capabilities for modularity, reusability, and clear state management.


   1. Main Graph (Orchestrator):
       * Handles the overall flow: Parsing -> Research -> Content Generation (via Supervisor) -> Formatting.
       * Contains a central supervisor_node that decides which content generation subgraph to invoke next.


   2. Supervisor Node:
       * A single node in the main graph responsible for routing.
       * It inspects the AgentState to determine:
           * Which content section is next in the sequence (Key Qualifications, Professional Experience, Projects, Executive Summary).
           * If all content sections are done, route to the formatter_node.
           * If there are errors, route to the error_handler_node.


   3. Content Generation Subgraphs (e.g., `key_qualifications_subgraph`, `professional_experience_subgraph`):
       * Each major content section (Key Qualifications, Professional Experience, Projects, Executive Summary) will have its own dedicated
         subgraph.
       * Nodes within a Subgraph:
           * Writer Node: Invokes the specific content writer agent (e.g., KeyQualificationsWriterAgent).
           * QA Node: Invokes the QualityAssuranceAgent to review the generated content.
           * User Feedback Node (Placeholder): This node will check the AgentState.user_feedback field. In a real Streamlit integration,
             this is where the backend would pause and wait for UI input. For now, we'll simulate its behavior based on the user_feedback
             in the state.
       * Conditional Edges within a Subgraph:
           * If user feedback is "regenerate" or QA fails: Loop back to the Writer Node.
           * If user feedback is "approve" and QA passes: Exit the subgraph (return to the Supervisor).

  Workflow Breakdown with Nodes and Edges:

  Main Graph (`CVWorkflowGraph`):


   * Entry Point: jd_parser
   * Nodes:
       * jd_parser_node: Parses Job Description.
       * cv_parser_node: Parses User CV.
       * research_node: Conducts research (company values, etc.).
       * supervisor_node: Decides the next content section to generate or if formatting is next.
       * key_qualifications_subgraph: Invokes the subgraph for Key Qualifications.
       * professional_experience_subgraph: Invokes the subgraph for Professional Experience.
       * projects_subgraph: Invokes the subgraph for Project Experience.
       * executive_summary_subgraph: Invokes the subgraph for Executive Summary.
       * formatter_node: Generates the final formatted CV.
       * error_handler_node: Centralized error handling.
   * Edges:
       * jd_parser -> cv_parser
       * cv_parser -> research
       * research -> supervisor_node
       * Conditional Edges from `supervisor_node`:
           * To key_qualifications_subgraph (if Key Qualifications is next)
           * To professional_experience_subgraph (if Professional Experience is next)
           * To projects_subgraph (if Projects is next)
           * To executive_summary_subgraph (if Executive Summary is next)
           * To formatter_node (if all content sections are done)
           * To error_handler_node (if errors detected)
       * key_qualifications_subgraph -> supervisor_node (after completion)
       * professional_experience_subgraph -> supervisor_node (after completion)
       * projects_subgraph -> supervisor_node (after completion)
       * executive_summary_subgraph -> supervisor_node (after completion)
       * formatter_node -> END
       * error_handler_node -> END

  Example Subgraph (`key_qualifications_subgraph`):


   * Entry Point: generate_key_qualifications
   * Nodes:
       * generate_key_qualifications_node: Calls KeyQualificationsWriterAgent.
       * qa_key_qualifications_node: Calls QualityAssuranceAgent.
       * await_user_feedback_key_qualifications_node: Checks AgentState.user_feedback.
   * Edges:
       * generate_key_qualifications_node -> qa_key_qualifications_node
       * qa_key_qualifications_node -> await_user_feedback_key_qualifications_node
       * Conditional Edges from `await_user_feedback_key_qualifications_node`:
           * If user_feedback.action == REGENERATE or qa_key_qualifications_node indicates issues: -> generate_key_qualifications_node
             (loop back for retry)
           * If user_feedback.action == APPROVE and qa_key_qualifications_node passes: -> END (exit subgraph)

  Implementation Steps:


   1. Define New Agent Classes:
       * Create KeyQualificationsWriterAgent, ProfessionalExperienceWriterAgent, ProjectsWriterAgent, and ExecutiveSummaryWriterAgent in
         src/agents/. These will extend AgentBase and use EnhancedLLMService.

   2. Update Dependency Injection Container:
       * Modify src/core/container.py to register these new agent classes.


   3. Refactor `src/orchestration/cv_workflow_graph.py`:
       * Implement the supervisor_node function.
       * Create separate functions for each content generation subgraph (e.g., _build_key_qualifications_subgraph).
       * Modify the _build_graph method to integrate the supervisor and call these subgraphs.
       * Adjust existing nodes like generate_skills_node (which currently populates key qualifications) to fit into the new subgraph
         structure. The generate_skills_node might become part of the key_qualifications_subgraph or its logic will be moved into the
         KeyQualificationsWriterAgent.


   4. Update `src/orchestration/state.py`:
       * Ensure AgentState can accommodate any new state variables needed for the supervisor or subgraphs (e.g., current_section_index for
         the supervisor to track progress).

   5. Create/Update Content Templates:
       * Ensure prompt templates exist for each content type in src/data/prompts/ or src/templates/.


   6. Testing Strategy:
       * Unit Tests: For each new agent, mock LLM calls and verify input validation, prompt construction, and output parsing.
       * Integration Tests: Test the subgraphs independently to ensure their internal loops and conditional logic work. Then, test the main
         graph, simulating AgentState updates (including user_feedback) to verify the overall flow.