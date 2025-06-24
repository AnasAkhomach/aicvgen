

# **TASK\_BLUEPRINT.md**

## **Task/Feature Addressed: Implement Granular, Item-by-Item Processing Workflow**

This technical blueprint provides the definitive specification for refactoring the anasakhomach-aicvgen application's core workflow. The primary objective is to transition from a monolithic, one-shot generation process to a dynamic, iterative, and stateful system managed by LangGraph. This will implement the core user-in-the-loop experience for the Minimum Viable Product (MVP), enabling users to review and regenerate individual components of their CV.

This plan directly addresses the functional requirements for granular control over "Professional Experience" and "Project Experience" sections (REQ-FUNC-GEN-3, REQ-FUNC-GEN-4) and the user interface requirement for hybrid control (REQ-FUNC-UI-2) as outlined in the project's governing Software Requirements Specification.1

### **Overall Technical Strategy**

The existing system architecture, having completed its initial hardening phase, is now prepared for this functional evolution.1 The current

EnhancedOrchestrator is a simple wrapper, and the true orchestration logic will now be built within the LangGraph application defined in src/orchestration/cv\_workflow\_graph.py.1

The architectural pattern for this implementation will be a **Stateful Backend, Stateless Frontend**. This model is crucial for integrating a persistent state machine like LangGraph with a framework like Streamlit, which reruns its script on every user interaction.

1. **Backend (LangGraph):** The cv\_graph\_app will be implemented as a stateful, long-running state machine. It will be the single source of truth for all business logic, agent execution, and state transitions. The state will be explicitly managed via the AgentState Pydantic model.  
2. **Frontend (Streamlit):** The Streamlit UI will function as a pure "view" layer. It will be completely stateless between user interactions. The UI's primary responsibility is to render the current AgentState and capture user actions.  
3. **The Interaction Loop:** The integration will be governed by a strict, predictable loop:  
   * **State In:** The Streamlit UI renders based on the current AgentState stored in st.session\_state.  
   * **UI Action:** The user clicks a button (e.g., "Accept" or "Regenerate"). The button's on\_click callback populates the user\_feedback field within the AgentState object in st.session\_state.  
   * **State Out:** The main script loop detects the presence of user\_feedback, invokes the LangGraph application (cv\_graph\_app.ainvoke) with the entire current AgentState, and receives a new, complete AgentState object as the result.  
   * **Re-render:** The script overwrites the old state in st.session\_state with the new one and calls st.rerun(), causing the UI to redraw itself based on the updated state.

This "State In \-\> UI Action \-\> State Out \-\> Re-render" pattern is the core architectural solution. It resolves the debugging and state management challenges observed in the project's earlier phases by making the data flow explicit and unidirectional.1

The final deliverable will be a robust, interactive CV generation experience where the user is guided through each section of their CV, with the ability to accept or regenerate content for each professional role and project individually, aligning with the conceptual workflow.1

### **Component-Level Implementation Plan**

This section details the required modifications for each affected component.

#### **1\. Pydantic Model & State Management Changes**

This task formalizes the data contracts for UI-to-backend communication and adapts the central state model to manage the iterative workflow. This addresses the historical data model inconsistencies that previously led to agent failures.1

* **Task/Feature Addressed:** Define data contracts for user interaction and update the central state model for iterative processing.  
* **Affected Component(s):** src/models/data\_models.py, src/orchestration/state.py.1  
* **Pydantic Model Changes (src/models/data\_models.py):**  
  * The following models must be added to create a strict, validated API contract between the Streamlit UI and the LangGraph backend.

Python  
from enum import Enum  
from typing import Optional  
from pydantic import BaseModel

class UserAction(str, Enum):  
    """Enumeration for user actions in the UI."""  
    ACCEPT \= "accept"  
    REGENERATE \= "regenerate"

class UserFeedback(BaseModel):  
    """User feedback for item review."""  
    action: UserAction  
    item\_id: str  
    feedback\_text: Optional\[str\] \= None

* **AgentState Refactoring (src/orchestration/state.py):**  
  * The AgentState model must be refactored to its definitive form to manage the granular workflow. The existing AgentState will be updated to include the following fields 1:

Python  
from typing import Any, Dict, List, Optional  
from pydantic import BaseModel, Field  
from src.models.data\_models import JobDescriptionData, StructuredCV, UserFeedback

class AgentState(BaseModel):  
    """  
    Represents the complete, centralized state of the CV generation workflow  
    for LangGraph orchestration.  
    """  
    \# Core Data Models  
    structured\_cv: StructuredCV  
    job\_description\_data: JobDescriptionData

    \# Workflow Control for Granular Processing  
    current\_section\_key: Optional\[str\] \= None  
    items\_to\_process\_queue: List\[str\] \= Field(default\_factory=list)  
    current\_item\_id: Optional\[str\] \= None  
    is\_initial\_generation: bool \= True

    \# User Feedback for Regeneration  
    user\_feedback: Optional\[UserFeedback\] \= None

    \# Agent Outputs & Finalization  
    research\_findings: Optional\] \= None  
    final\_output\_path: Optional\[str\] \= None  
    error\_messages: List\[str\] \= Field(default\_factory=list)

    class Config:  
        arbitrary\_types\_allowed \= True

#### **2\. Agent Logic Modifications: Granular Processing**

This task refactors the EnhancedContentWriterAgent to align with the Single Responsibility Principle, making it more testable and reusable within the LangGraph framework.

* **Task/Feature Addressed:** Refactor the content writer agent to operate on a single CV item.  
* **Affected Component(s):** src/agents/enhanced\_content\_writer.py.1  
* **run\_as\_node Refactoring:**  
  * The existing run\_as\_node method will be modified. Its logic will now focus exclusively on processing the single item referenced by state.current\_item\_id. It will no longer iterate over the entire CV structure.  
* **New Helper Method (\_build\_single\_item\_prompt):**  
  * A new private method, \_build\_single\_item\_prompt, must be created within the EnhancedContentWriterAgent.  
  * **Inputs:** This method will accept the target\_item, its parent section and subsection, the job\_description\_data, and any user\_feedback from the AgentState.  
  * **Logic:** It will construct a highly specific, contextual LLM prompt tailored to generating content for only that single item.  
  * **Example Prompt Structure Snippet:**

You are an expert CV writer. Your task is to generate content for a single item in a CV.The section is: 'Professional Experience'The subsection is: 'Senior Software Engineer at TechCorp'The original content of the item is: 'Developed Python applications.'The target job description keywords are: FastAPI, Microservices, AWS, Docker, KubernetesIncorporate the following user feedback: 'Focus more on the DevOps aspects of this role.'Please generate the new, improved content for this single item. Respond with only the generated text.

#### **3\. LangGraph Workflow Implementation**

This task involves designing and implementing the complete state machine that orchestrates the iterative CV generation process, making the workflow explicit and robust.

* **Task/Feature Addressed:** Implement the full state machine graph for the iterative workflow.  
* **Affected Component(s):** src/orchestration/cv\_workflow\_graph.py.1  
* **Node Definitions:** The graph will be composed of the following asynchronous nodes:  
  * parser\_node: Parses the initial CV and job description text, populating the initial AgentState.  
  * research\_node: Enriches the state with research findings from the ResearchAgent.  
  * generate\_skills\_node: Generates the "Big 10" skills and populates the Key Qualifications section of the StructuredCV.  
  * process\_next\_item\_node: Pops the next item ID from items\_to\_process\_queue and sets it as current\_item\_id. It also clears user\_feedback to prevent re-triggering.  
  * content\_writer\_node: Invokes the refactored EnhancedContentWriterAgent to process the current\_item\_id.  
  * qa\_node: Invokes the QualityAssuranceAgent to inspect the newly generated content and add any warnings to the item's metadata.  
  * prepare\_next\_section\_node: Identifies the next section from a predefined WORKFLOW\_SEQUENCE, populates the items\_to\_process\_queue with item IDs from that new section, and clears current\_item\_id.  
  * formatter\_node: Invokes the FormatterAgent to generate the final PDF output.  
* **Conditional Routing Logic (route\_after\_review):** The core of the graph's intelligence lies in this conditional edge, which is invoked after the qa\_node. Its logic is defined by the state of user\_feedback and items\_to\_process\_queue.

| Current State | User Action | Next Node | Rationale |
| :---- | :---- | :---- | :---- |
| user\_feedback.action is REGENERATE | Regenerate | content\_writer | The user wants to retry generation for the current item. |
| user\_feedback.action is ACCEPT AND items\_to\_process\_queue is NOT empty | Accept | process\_next\_item | The current section is not finished; process the next item in the queue. |
| user\_feedback.action is ACCEPT AND items\_to\_process\_queue is empty AND next section exists | Accept | prepare\_next\_section | The current section is finished; prepare and move to the next section. |
| user\_feedback.action is ACCEPT AND items\_to\_process\_queue is empty AND NO next section | Accept | formatter | All sections are complete; proceed to final PDF generation. |

* **Graph Assembly (cv\_workflow\_graph.py):** The engineer will assemble the graph using langgraph.graph.StateGraph.  
  Python  
  from langgraph.graph import StateGraph, END  
  from src.orchestration.state import AgentState

  \#... (import all node functions)...

  def build\_cv\_workflow\_graph() \-\> StateGraph:  
      workflow \= StateGraph(AgentState)

      \# Add all nodes  
      workflow.add\_node("parser", parser\_node)  
      workflow.add\_node("generate\_skills", generate\_skills\_node)  
      workflow.add\_node("process\_next\_item", process\_next\_item\_node)  
      workflow.add\_node("content\_writer", content\_writer\_node)  
      workflow.add\_node("qa", qa\_node)  
      workflow.add\_node("prepare\_next\_section", prepare\_next\_section\_node)  
      workflow.add\_node("formatter", formatter\_node)

      \# Define workflow edges  
      workflow.set\_entry\_point("parser")  
      workflow.add\_edge("parser", "generate\_skills")  
      workflow.add\_edge("generate\_skills", "process\_next\_item")  
      workflow.add\_edge("process\_next\_item", "content\_writer")  
      workflow.add\_edge("prepare\_next\_section", "process\_next\_item")  
      workflow.add\_edge("content\_writer", "qa")  
      workflow.add\_edge("formatter", END)

      \# Add the conditional routing logic  
      workflow.add\_conditional\_edges(  
          "qa",  
          route\_after\_review,  
          {  
              "content\_writer": "content\_writer",  
              "process\_next\_item": "process\_next\_item",  
              "prepare\_next\_section": "prepare\_next\_section",  
              "formatter": "formatter",  
              END: END  
          }  
      )

      return workflow.compile()

  cv\_graph\_app \= build\_cv\_workflow\_graph()

#### **4\. UI Interaction Model (Streamlit)**

This task implements the stateless view layer that enables user interaction with the LangGraph backend.

* **Task/Feature Addressed:** Implement the Streamlit UI to facilitate the user-in-the-loop workflow.  
* **Affected Component(s):** src/core/main.py.  
* **Detailed Implementation Steps:**  
  1. **State Initialization:** The main function in src/core/main.py must initialize st.session\_state.agent\_state to None if it does not exist.  
  2. **UI Rendering Functions:** Create modular functions to render the StructuredCV from the agent\_state. A key function will be display\_regenerative\_item, which renders a card for each job role or project.  
  3. **display\_regenerative\_item with Controls:** This function will render each item with its own "Accept" and "Regenerate" buttons. The key for each button must be unique (e.g., f"accept\_{item\_id}"). The on\_click parameter will be bound to a callback function.  
  4. **handle\_user\_action Callback:** This callback function's sole responsibility is to populate st.session\_state.agent\_state.user\_feedback with the appropriate UserFeedback object. It should not invoke the graph directly.  
  5. **Main Application Loop:** The main body of the script will contain the core logic that orchestrates the UI-backend interaction.  
     Python  
     \# src/core/main.py  
     import streamlit as st  
     from src.orchestration.cv\_workflow\_graph import cv\_graph\_app  
     from src.orchestration.state import AgentState  
     from src.models.data\_models import UserAction, UserFeedback  
     \#... other imports

     def handle\_user\_action(action: str, item\_id: str):  
         """Callback to update the state with user feedback."""  
         if st.session\_state.agent\_state:  
             st.session\_state.agent\_state.user\_feedback \= UserFeedback(  
                 action=UserAction(action),  
                 item\_id=item\_id,  
             )

     def main():  
         if 'agent\_state' not in st.session\_state:  
             st.session\_state.agent\_state \= None

         \# \--- Main Interaction Loop \---  
         if st.session\_state.agent\_state and st.session\_state.agent\_state.user\_feedback:  
             with st.spinner("Processing your request..."):  
                 \# Invoke the graph with the current state (which includes user feedback)  
                 new\_state\_dict \= cv\_graph\_app.invoke(st.session\_state.agent\_state.model\_dump())

                 \# Overwrite the session state with the new state from the graph  
                 st.session\_state.agent\_state \= AgentState.model\_validate(new\_state\_dict)

                 \# Clear the feedback so this block doesn't run again on the next rerun  
                 st.session\_state.agent\_state.user\_feedback \= None

                 \# Trigger an immediate re-render to show the updated content  
                 st.rerun()

         \# \--- UI Rendering Code \---  
         \#... (Tabs, input forms, etc.)...  
         \# The "Review & Edit" tab will call rendering functions that use  
         \# \`st.session\_state.agent\_state\` to display the CV and the  
         \# \`handle\_user\_action\` callback for buttons.

### **Implementation & Testing Plan**

#### **Detailed Implementation Steps**

1. **Implement Pydantic Models:** Create/update UserAction and UserFeedback in src/models/data\_models.py.  
2. **Refactor AgentState:** Update src/orchestration/state.py with the new fields for workflow control.  
3. **Refactor EnhancedContentWriterAgent:** Modify run\_as\_node and add the \_build\_single\_item\_prompt helper method in src/agents/enhanced\_content\_writer.py.  
4. **Build LangGraph Workflow:** Implement all nodes and the route\_after\_review conditional edge in src/orchestration/cv\_workflow\_graph.py.  
5. **Implement Streamlit UI:** Refactor src/core/main.py to include the main interaction loop, rendering functions, and callbacks as specified.

#### **Testing Considerations**

* **Unit Tests:**  
  * Test the route\_after\_review function with various mock AgentState configurations to assert it returns the correct next node name.  
  * Test the refactored EnhancedContentWriterAgent.run\_as\_node to verify it only modifies the content of the item specified by current\_item\_id.  
  * Test the handle\_user\_action callback to ensure it correctly populates user\_feedback in st.session\_state.  
* **E2E Integration Test:**  
  * An E2E test must be created to simulate a full user session. This test will programmatically:  
    1. Submit initial data to trigger the first graph invocation.  
    2. Verify the first reviewable section appears.  
    3. Simulate a user clicking "Accept".  
    4. Verify the workflow advances to the next item in the same section.  
    5. Simulate a user clicking "Regenerate".  
    6. Verify the content of *only that item* changes.  
    7. Simulate accepting all items in a section.  
    8. Verify the workflow transitions to the next section.  
  * This test is critical for validating the entire state machine's logic.

### **Critical Gaps & Questions**

* **Potential Challenges & Critical Considerations:**  
  * **State Immutability:** It is critical that the Streamlit UI code **never** modifies the st.session\_state.agent\_state object directly. All state changes must originate from the new state object returned by the cv\_graph\_app.invoke call. This must be strictly enforced.  
  * **Performance:** Invoking the entire graph on every click is acceptable for the MVP. However, the engineer should be aware that future performance optimization may require more targeted graph invocations or caching strategies.  
* **Assumptions Made:**  
  * The ParserAgent can successfully segment the input CV into a StructuredCV model with unique, UUID-tagged items for each role and project.  
  * The FormatterAgent is capable of rendering a final StructuredCV object into a PDF file.1  
* **Critical Gaps & Questions to Address:**  
  * **Initial Generation Flow:** The blueprint clarifies that the initial "Generate" button click invokes the graph, which runs until it pauses for the first user review. The returned state is then rendered.  
  * **Error Display in UI:** How should non-fatal errors accumulated in state.error\_messages be presented to the user?  
    * **Recommendation:** Implement a dedicated st.expander("View Processing Warnings") in the Streamlit UI that displays any messages from the error\_messages list. This keeps the main UI clean while providing transparency.

#### **Works cited**

1. Comceptual Workflow.txt