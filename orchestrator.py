# orchestrator.py
import json # Import json
from state_manager import WorkflowState, WorkflowStage, VectorStoreConfig, JobDescriptionData, ContentData, AgentIO, SkillEntry, ExperienceEntry, CVData
from parser_agent import ParserAgent
from template_renderer import TemplateRenderer
from vector_store_agent import VectorStoreAgent
from vector_db import VectorDB
from content_writer_agent import ContentWriterAgent
from research_agent import ResearchAgent
from cv_analyzer_agent import CVAnalyzerAgent
from tools_agent import ToolsAgent
from formatter_agent import FormatterAgent
from quality_assurance_agent import QualityAssuranceAgent
from uuid import uuid4
from llm import LLM

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Any, Dict, List

class Orchestrator:
    """
    Orchestrates the CV tailoring workflow using LangGraph.
    """
    def __init__(self, parser_agent: ParserAgent, template_renderer: TemplateRenderer, vector_store_agent: VectorStoreAgent, content_writer_agent: ContentWriterAgent, research_agent: ResearchAgent, cv_analyzer_agent: CVAnalyzerAgent, tools_agent: ToolsAgent, formatter_agent: FormatterAgent, quality_assurance_agent: QualityAssuranceAgent, llm: LLM):
        self.parser_agent = parser_agent
        self.template_renderer = template_renderer
        self.vector_store_agent = vector_store_agent
        self.content_writer_agent = content_writer_agent
        self.research_agent = research_agent
        self.cv_analyzer_agent = cv_analyzer_agent
        self.tools_agent = tools_agent
        self.formatter_agent = formatter_agent
        self.quality_assurance_agent = quality_assurance_agent
        self.llm = llm
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def handle_feedback_node(self, state: WorkflowState) -> WorkflowState:
        """
        LangGraph node to handle user feedback for generated content.
        """
        print("Executing: handle_feedback")
        state["current_stage"] = {"stage_name": "Feedback Handling", "description": "Processing user feedback", "is_completed": False}

        # Simulate feedback collection (replace with actual feedback interface integration)
        generated_content = state.get("generated_content", {})
        feedback = state.get("review_status", "pending")
        user_feedback = state.get("review_feedback", "")

        if feedback == "reject":
            print("User rejected the content. Regenerating...")
            # Use user feedback to refine content generation (if applicable)
            content_writer_input = {
                "job_description_data": state.get("job_description", {}),
                "relevant_experiences": state.get("relevant_experiences", []),
                "research_results": state.get("research_results", {}),
                "user_cv_data": state.get("user_cv", {})
            }
            try:
                regenerated_content = self.content_writer_agent.generate_batch(content_writer_input, batch_type="experience_bullet")
                state["generated_content"] = dict(regenerated_content)
                print("Content regenerated successfully.")
            except Exception as e:
                print(f"Error regenerating content: {e}")
                state["current_stage"] = {"stage_name": "Feedback Handling Failed", "description": f"Error: {e}", "is_completed": True}
                return state

        state["current_stage"]["is_completed"] = True
        print("Completed: handle_feedback")
        return state

    def _build_graph(self) -> StateGraph:
        """
        Builds the LangGraph workflow with checkpointing.
        Includes human review step and conditional routing.
        """
        workflow = StateGraph(WorkflowState)

        # Define nodes for each agent/step
        workflow.add_node("parse_job_description", self.parse_job_description_node)
        workflow.add_node("analyze_cv", self.analyze_cv_node)
        workflow.add_node("add_experiences_to_vector_store", self.add_experiences_to_vector_store_node)
        workflow.add_node("search_vector_store", self.search_vector_store_node)
        workflow.add_node("run_research", self.run_research_node)
        workflow.add_node("generate_content", self.generate_content_node)
        workflow.add_node("format_cv", self.format_cv_node)
        workflow.add_node("run_quality_assurance", self.run_quality_assurance_node)
        workflow.add_node("human_review", self.human_review_node) # Add human_review node
        workflow.add_node("handle_feedback", self.handle_feedback_node) # Add feedback handling node
        workflow.add_node("render_cv", self.render_cv_node)

        # Define edges (transitions between nodes)
        workflow.add_edge("parse_job_description", "analyze_cv")
        workflow.add_edge("analyze_cv", "add_experiences_to_vector_store")
        workflow.add_edge("add_experiences_to_vector_store", "search_vector_store")
        workflow.add_edge("search_vector_store", "run_research")
        workflow.add_edge("run_research", "generate_content")
        workflow.add_edge("generate_content", "format_cv")
        workflow.add_edge("format_cv", "run_quality_assurance")
        workflow.add_edge("run_quality_assurance", "human_review")

        # Add conditional edge from human_review
        workflow.add_conditional_edges(
            "human_review", # The starting node for conditional transitions
            self._decide_next_step_after_review, # The function that determines the next node
            {
                "approve": "render_cv", # If review_status is "approve", go to render_cv
                "reject": "handle_feedback", # If review_status is "reject", go to handle_feedback
            }
        )

        workflow.add_edge("handle_feedback", "generate_content") # Loop back to content generation after feedback
        workflow.add_edge("render_cv", END) # End the workflow after rendering

        # Set the entry point
        workflow.set_entry_point("parse_job_description")

        # Compile the graph with the checkpointer
        return workflow.compile(checkpointer=self.checkpointer)

    # Define the conditional logic function
    def _decide_next_step_after_review(self, state: WorkflowState) -> str:
        """
        Determines the next step after human review based on the review_status.
        """
        print("Deciding next step after human review...")
        review_status = state.get("review_status")

        if review_status == "approve":
            print("Review approved. Proceeding to rendering.")
            return "render_cv"
        elif review_status == "reject":
            print("Review rejected. Looping back to content generation for refinement.")
            return "handle_feedback"
        else:
            print(f"Unknown review status: {review_status}. Proceeding to rendering by default.")
            return "render_cv"

    def parse_job_description_node(self, state: WorkflowState) -> WorkflowState:
        """
        LangGraph node to parse the job description.
        """
        print("Executing: parse_job_description")
        state["current_stage"] = {"stage_name": "Parsing", "description": "Extracting data from the job description", "is_completed": False}

        job_description_text = state.get("job_description", {}).get("raw_text", "")
        if not job_description_text:
             print("Error: Job description not found in state.")
             return state

        try:
            job_description_data = self.parser_agent.run({"job_description": job_description_text})
        except Exception as e:
            print(f"Error running parser agent: {e}")
            state["current_stage"] = {"stage_name": "Parsing Failed", "description": f"Error: {e}", "is_completed": True}
            return state

        state["job_description"] = {
             "raw_text": job_description_text,
             "skills": job_description_data.skills,
             "experience_level": job_description_data.experience_level,
             "responsibilities": job_description_data.responsibilities,
             "industry_terms": job_description_data.industry_terms,
             "company_values": job_description_data.company_values
        }

        state["current_stage"]["is_completed"] = True
        print("Completed: parse_job_description")
        return state

    def analyze_cv_node(self, state: WorkflowState) -> WorkflowState:
        """
        LangGraph node to analyze the user's CV.
        """
        print("Executing: analyze_cv")
        state["current_stage"] = {"stage_name": "CV Analysis", "description": "Extracting data from user CV", "is_completed": False}

        user_cv_data = state.get("user_cv")
        job_description_data = state.get("job_description")

        if not user_cv_data or not user_cv_data.get("raw_text"):
            print("Error: User CV data not found in state.")
            state["current_stage"] = {"stage_name": "CV Analysis Failed", "description": "Error: User CV data not found.", "is_completed": True}
            if state.get("user_cv") is None:
                 state["user_cv"] = CVData(raw_text="", experiences=[], summary="", skills=[], education=[], projects=[])
            else:
                state["user_cv"]["experiences"] = state["user_cv"].get("experiences", [])
                state["user_cv"]["summary"] = state["user_cv"].get("summary", "")
                state["user_cv"]["skills"] = state["user_cv"].get("skills", [])
                state["user_cv"]["education"] = state["user_cv"].get("education", [])
                state["user_cv"]["projects"] = state["user_cv"].get("projects", [])

            return state

        try:
            extracted_cv_data = self.cv_analyzer_agent.run({"user_cv": user_cv_data, "job_description": job_description_data})

            state["user_cv"]["summary"] = extracted_cv_data.get("summary", "")
            state["user_cv"]["experiences"] = extracted_cv_data.get("experiences", [])
            state["user_cv"]["skills"] = extracted_cv_data.get("skills", [])
            state["user_cv"]["education"] = extracted_cv_data.get("education", [])
            state["user_cv"]["projects"] = extracted_cv_data.get("projects", [])

            state["extracted_skills"] = state["user_cv"]["skills"]

            state["current_stage"]["is_completed"] = True
            print("Completed: analyze_cv")

        except Exception as e:
            print(f"Error running CV analyzer agent: {e}")
            state["current_stage"] = {"stage_name": "CV Analysis Failed", "description": f"Error: {e}", "is_completed": True}
            if state.get("user_cv") is None:
                 state["user_cv"] = CVData(raw_text=user_cv_data.get("raw_text", ""), experiences=[], summary=f"Error analyzing CV: {e}", skills=[], education=[], projects=[])
            else:
                state["user_cv"]["summary"] = f"Error analyzing CV: {e}"
                state["user_cv"]["experiences"] = state["user_cv"].get("experiences", [])
                state["user_cv"]["skills"] = state["user_cv"].get("skills", [])
                state["user_cv"]["education"] = state["user_cv"].get("education", [])
                state["user_cv"]["projects"] = state["user_cv"].get("projects", [])

            state["extracted_skills"] = state["user_cv"]["skills"]

        return state

    def add_experiences_to_vector_store_node(self, state: WorkflowState) -> WorkflowState:
        """
        LangGraph node to add user experiences to the vector store.
        Now uses experiences extracted by the CVAnalyzerAgent.
        """
        print("Executing: add_experiences_to_vector_store")
        state["current_stage"] = {"stage_name": "Vector Store Update", "description": "Adding user experiences to vector store", "is_completed": False}

        user_experiences_for_embedding = state.get("user_cv", {}).get("experiences", [])

        if not user_experiences_for_embedding:
            print("No user experiences found in state to add to vector store.")
            state["current_stage"]["is_completed"] = True
            print("Completed: add_experiences_to_vector_store (no items added)")
            return state

        try:
            for item_text in user_experiences_for_embedding:
                 self.vector_store_agent.run_add_item(ExperienceEntry(text=item_text), text=item_text)
            print(f"Added {len(user_experiences_for_embedding)} items to vector store.")
        except Exception as e:
            print(f"Error adding items to vector store: {e}")
            state["current_stage"] = {"stage_name": "Vector Store Update Failed", "description": f"Error: {e}", "is_completed": True}
            return state

        state["current_stage"]["is_completed"] = True
        print("Completed: add_experiences_to_vector_store")
        return state

    def search_vector_store_node(self, state: WorkflowState) -> WorkflowState:
        """
        LangGraph node to search the vector store for relevant experiences.
        Now uses skills and responsibilities from the parsed job description and potentially extracted CV skills.
        """
        print("Executing: search_vector_store")
        state["current_stage"] = {"stage_name": "Vector Store Search", "description": "Searching for relevant experiences", "is_completed": False}

        job_description_data = state.get("job_description", {})
        user_cv_data = state.get("user_cv", {}) # Get user_cv data

        search_query_parts = []
        if job_description_data.get("skills"):
            search_query_parts.extend(job_description_data["skills"])
        if job_description_data.get("responsibilities"):
            search_query_parts.extend(job_description_data["responsibilities"])

        search_query = ". ".join(search_query_parts)

        relevant_experiences = []
        if search_query:
            try:
                search_results = self.vector_store_agent.search(query=search_query, k=5)
                relevant_experiences = [result.text for result in search_results if hasattr(result, 'text')]
                print(f"Found {len(relevant_experiences)} relevant experiences.")
                print(f"Relevant Experiences: {relevant_experiences}")
            except Exception as e:
                print(f"Error searching vector store: {e}")
                state["current_stage"] = {"stage_name": "Vector Store Search Failed", "description": f"Error: {e}", "is_completed": True}
                state["relevant_experiences"] = []
                return state

        state["relevant_experiences"] = relevant_experiences

        state["current_stage"]["is_completed"] = True
        print("Completed: search_vector_store")
        return state

    def run_research_node(self, state: WorkflowState) -> WorkflowState:
        """
        LangGraph node to run the ResearchAgent.
        """
        print("Executing: run_research")
        state["current_stage"] = {"stage_name": "Research", "description": "Gathering job-related information", "is_completed": False}

        job_description_data = state.get("job_description", {})

        try:
            research_results = self.research_agent.run({"job_description_data": job_description_data})
            state["research_results"] = research_results
            state["current_stage"]["is_completed"] = True
            print("Completed: run_research")

        except Exception as e:
            print(f"Error running ResearchAgent: {e}")
            state["current_stage"] = {"stage_name": "Research Failed", "description": f"Error: {e}", "is_completed": True}
            state["research_results"] = {}

        return state

    def generate_content_node(self, state: WorkflowState) -> WorkflowState:
        """
        LangGraph node for content generation using the ContentWriterAgent.
        Now uses relevant experiences, research results, and extracted CV data from the state.
        """
        print("Executing: generate_content")
        state["current_stage"] = {"stage_name": "Content Generation", "description": "Creating content for the CV", "is_completed": False}

        job_description_data = state.get("job_description", {})
        relevant_experiences = state.get("relevant_experiences", [])
        research_results = state.get("research_results", {})
        user_cv_data = state.get("user_cv", {}) # Get user_cv_data

        content_writer_input = {
            "job_description_data": job_description_data,
            "relevant_experiences": relevant_experiences,
            "research_results": research_results,
            "user_cv_data": user_cv_data
        }

        try:
            generated_content: ContentData = self.content_writer_agent.run(content_writer_input)

            state["generated_content"] = dict(generated_content)

            state["current_stage"]["is_completed"] = True
            print("Completed: generate_content")

        except Exception as e:
            print(f"Error running ContentWriterAgent: {e}")
            state["current_stage"] = {"stage_name": "Content Generation Failed", "description": f"Error: {e}", "is_completed": True}

        return state

    def format_cv_node(self, state: WorkflowState) -> WorkflowState:
        """
        LangGraph node for formatting the generated CV content using the FormatterAgent.
        """
        print("Executing: format_cv")
        state["current_stage"] = {"stage_name": "Formatting", "description": "Formatting the CV content", "is_completed": False}

        generated_content_dict = state.get("generated_content", {}) # Get generated content
        generated_content = ContentData(**generated_content_dict)

        format_specifications = {"template_type": "markdown", "style": "professional"}

        try:
            formatted_cv_text = self.formatter_agent.run({
                "content_data": generated_content,
                "format_specifications": format_specifications
            })

            state["formatted_cv_text"] = formatted_cv_text

            state["current_stage"]["is_completed"] = True
            print("Completed: format_cv")

        except Exception as e:
            print(f"Error running FormatterAgent: {e}")
            state["current_stage"] = {"stage_name": "Formatting Failed", "description": f"Error: {e}", "is_completed": True}
            state["formatted_cv_text"] = "Error during formatting."

        return state

    def run_quality_assurance_node(self, state: WorkflowState) -> WorkflowState:
        """
        LangGraph node for running quality assurance checks on the formatted CV.
        """
        print("Executing: run_quality_assurance")
        state["current_stage"] = {"stage_name": "Quality Assurance", "description": "Performing quality checks on the CV", "is_completed": False}

        formatted_cv_text = state.get("formatted_cv_text", "")
        job_description_data = state.get("job_description", {}) # Pass job description for context

        if not formatted_cv_text:
            print("Error: Formatted CV text not found in state for quality assurance.")
            state["current_stage"] = {"stage_name": "Quality Assurance Failed", "description": "Error: Formatted CV text not found.", "is_completed": True}
            if "quality_assurance_results" not in state or state["quality_assurance_results"] is None:
                 state["quality_assurance_results"] = {}
            return state

        try:
            quality_results = self.quality_assurance_agent.run({
                "formatted_cv_text": formatted_cv_text,
                "job_description": job_description_data
            })

            state["quality_assurance_results"] = quality_results

            state["current_stage"]["is_completed"] = True
            print("Completed: run_quality_assurance")

        except Exception as e:
            print(f"Error running QualityAssuranceAgent: {e}")
            state["current_stage"] = {"stage_name": "Quality Assurance Failed", "description": f"Error: {e}", "is_completed": True}
            if "quality_assurance_results" not in state or state["quality_assurance_results"] is None:
                 state["quality_assurance_results"] = {}

        return state

    def human_review_node(self, state: WorkflowState) -> WorkflowState:
        """
        LangGraph node for human review of the tailored CV.
        This node simulates waiting for human input and updating the state.
        """
        print("Executing: human_review")
        state["current_stage"] = {"stage_name": "Human Review", "description": "Waiting for human review and feedback", "is_completed": False}

        print("Simulating human review... (Assume review is pending external input)")
        
        quality_results = state.get("quality_assurance_results", {})
        is_quality_ok = quality_results.get("is_quality_ok", True)
        feedback = quality_results.get("feedback", "")
        suggestions = quality_results.get("suggestions", [])

        simulated_review_status = "approve"
        simulated_review_feedback = "Looks good!"

        if not is_quality_ok:
            simulated_review_status = "reject"
            simulated_review_feedback = f"Issues found during QA: {feedback}. Suggestions: {', '.join(suggestions)}"
            print(f"Simulating rejection due to QA issues: {simulated_review_feedback}")
        else:
             print("Simulating approval.")

        state["review_status"] = simulated_review_status
        state["review_feedback"] = simulated_review_feedback

        state["current_stage"]["is_completed"] = True
        print("Completed: human_review")

        return state

    def render_cv_node(self, state: WorkflowState) -> WorkflowState:
        """
        LangGraph node for rendering the CV.
        Now uses the formatted CV text from the state.
        This node is reached after human review approves the CV.
        """
        print("Executing: render_cv")
        state["current_stage"] = {"stage_name": "Rendering", "description": "Rendering the CV", "is_completed": False}

        formatted_cv_text = state.get("formatted_cv_text", "")

        if not formatted_cv_text:
            print("Error: Formatted CV text not found in state for rendering.")
            state["current_stage"] = {"stage_name": "Rendering Failed", "description": "Error: Formatted CV text not found.", "is_completed": True}
            state["rendered_cv"] = "Rendering failed: No formatted content."
            return state

        try:
            state["rendered_cv"] = formatted_cv_text

            state["current_stage"]["is_completed"] = True
            print("Completed: render_cv")

        except Exception as e:
            print(f"Error during final rendering step: {e}")
            state["current_stage"] = {"stage_name": "Rendering Failed", "description": f"Error: {e}", "is_completed": True}
            state["rendered_cv"] = f"Error during final rendering: {e}"

        return state

    def run_workflow(self, job_description: str, user_cv_data: CVData, workflow_id: str = None) -> str:
        """
        Runs the CV tailoring workflow using the LangGraph graph.

        Args:
            job_description: The raw job description text.
            user_cv_data: CVData object containing user CV data.
            workflow_id: Optional ID to resume a previous workflow run.

        Returns:
            The rendered CV as a string from the final state.
        """
        current_workflow_id = workflow_id if workflow_id else str(uuid4())
        print(f"Starting or resuming workflow with ID: {current_workflow_id}")

        initial_user_cv_state = dict(user_cv_data) if isinstance(user_cv_data, CVData) else user_cv_data
        initial_user_cv_state.setdefault("raw_text", "")
        initial_user_cv_state.setdefault("experiences", [])
        initial_user_cv_state.setdefault("summary", "")
        initial_user_cv_state.setdefault("skills", [])
        initial_user_cv_state.setdefault("education", [])
        initial_user_cv_state.setdefault("projects", [])

        initial_state: WorkflowState = {
            "job_description": {"raw_text": job_description},
            "user_cv": initial_user_cv_state,
            "extracted_skills": {}, 
            "generated_content": {},
            "formatted_cv_text": "", 
            "rendered_cv": "", 
            "feedback": [],
            "revision_history": [],
            "current_stage": {"stage_name": "Initialized", "description": "Workflow initialized", "is_completed": True},
            "workflow_id": current_workflow_id,
            "relevant_experiences": [],
            "research_results": {}, 
            "quality_assurance_results": {}, 
            "review_status": "pending", 
            "review_feedback": ""
        }

        try:
            print(
                "--- Running workflow with invoke (state will be saved automatically) ---"
            )
            final_state = self.graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": current_workflow_id}},
            )

            print("--- Final State ---")
            print("Workflow completed")
            print("End")

            return final_state.get("rendered_cv", 'Rendering failed or did not produce output.')

        except Exception as e:
             print(f"--- Workflow encountered an error ---")
             print(f"Error: {e}")
             try:
                 last_state = self.graph.get_state(thread_id=current_workflow_id)
                 print(f"--- Last saved state (before error) ---")
                 if last_state and hasattr(last_state, 'values'):
                     print(last_state.values)
                 else:
                     print("Could not retrieve last saved state or it is not in expected format.")
             except Exception as state_e:
                 print(f"Could not retrieve last saved state: {state_e}")

             return f"Workflow failed with error: {e}"
