"""
Orchestrator module for AI CV Generator application.

This module manages the overall workflow of the CV tailoring process, coordinating the actions
of various specialized agents to generate a tailored CV based on job description analysis.
It implements both section-level and item-level control as per SDD v1.3.
"""

# Standard imports
import json
import os
import logging
import time
import traceback
import uuid

# Third-party imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Set, Dict, List, Any, Optional
from unittest.mock import MagicMock

# Local imports
from src.core.state_manager import (
    StructuredCV,
    WorkflowState,
    WorkflowStage,
    VectorStoreConfig,
    JobDescriptionData,
    ContentData,
    AgentIO,
    SkillEntry,
    ExperienceEntry,
    CVData,
)
from src.agents.parser_agent import ParserAgent
from src.utils.template_renderer import TemplateRenderer
from src.agents.vector_store_agent import VectorStoreAgent
from src.services.vector_db import VectorDB
from src.agents.content_writer_agent import ContentWriterAgent
from src.agents.research_agent import ResearchAgent
from src.agents.cv_analyzer_agent import CVAnalyzerAgent
from src.agents.tools_agent import ToolsAgent
from src.agents.formatter_agent import FormatterAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.services.llm import LLM

# Setup logging
logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main orchestrator for the CV tailoring workflow.

    This class coordinates the various specialized agents in the system,
    managing the flow of data between them and the overall state of the
    CV tailoring process. It supports section-level operations for content
    regeneration and user feedback.

    Attributes:
        parser_agent: Agent for parsing job descriptions and CVs
        template_renderer: Agent for rendering CV templates
        vector_store_agent: Agent for working with vector embeddings
        content_writer_agent: Agent for generating tailored CV content
        research_agent: Agent for researching job requirements
        cv_analyzer_agent: Agent for analyzing CVs
        tools_agent: Agent providing utility functions
        formatter_agent: Agent for formatting CV content
        quality_assurance_agent: Agent for quality checks
        llm: Language model for text generation
    """

    def __init__(
        self,
        parser_agent,
        template_renderer,
        vector_store_agent,
        content_writer_agent,
        research_agent,
        cv_analyzer_agent,
        tools_agent,
        formatter_agent,
        quality_assurance_agent,
        llm=None,
    ):
        """
        Initialize the Orchestrator with all required agents.

        Args:
            parser_agent: Agent for parsing job descriptions
            template_renderer: Agent for rendering CV templates
            vector_store_agent: Agent for managing vector store
            content_writer_agent: Agent for generating CV content
            research_agent: Agent for research about job and industry
            cv_analyzer_agent: Agent for analyzing existing CVs
            tools_agent: Agent for providing utility tools
            formatter_agent: Agent for formatting CV content
            quality_assurance_agent: Agent for QA checks
            llm: LLM instance (optional)
        """
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

    def run_workflow(
        self,
        job_description,
        user_cv_data,
        workflow_id=None,
        user_feedback=None,
        regenerate_item_ids=None,
        start_from_scratch=False,
    ):
        """
        Run the CV tailoring workflow end-to-end.

        Args:
            job_description: Raw job description text
            user_cv_data: User's original CV data (can be raw text or structured)
            workflow_id: Optional ID for resuming an existing workflow
            user_feedback: Optional feedback on previously generated content
            regenerate_item_ids: Optional list of item IDs to regenerate
            start_from_scratch: Whether to start from scratch (ignore user's CV content)

        Returns:
            Dictionary with workflow results including the tailored CV
        """
        try:
            # 1. Initialize workflow state
            if workflow_id:
                # Load existing workflow state
                logger.info(f"Resuming workflow {workflow_id}")
                # TODO: Implement state loading
                state = {"workflow_id": workflow_id}
            else:
                # Create new workflow ID
                workflow_id = str(uuid.uuid4())
                logger.info(f"Starting new workflow {workflow_id}")
                state = {
                    "workflow_id": workflow_id,
                    "start_time": time.time(),
                    "status": "initializing",
                    "job_description": job_description,
                    "user_cv_data": user_cv_data,
                    "start_from_scratch": start_from_scratch,
                }

            # 2. Parsing stage
            state = self.run_parsing_stage(state)
            if "error" in state:
                return {"status": "error", "error": state["error"]}

            # 3. Research stage
            state = self.run_research_stage(state)
            if "error" in state:
                return {"status": "error", "error": state["error"]}

            # 4. Content generation stage
            state = self.run_content_generation_stage(state)
            if "error" in state:
                return {"status": "error", "error": state["error"]}

            # 5. Quality assurance stage
            state = self.run_quality_assurance_stage(state)
            if "error" in state:
                return {"status": "error", "error": state["error"]}

            # 6. Formatting
            formatted_result = self.run_formatting_stage(state["structured_cv"])
            if "error" in formatted_result:
                return {"status": "error", "error": formatted_result["error"]}

            # 7. Complete the workflow and return results
            state["status"] = "completed"
            state["end_time"] = time.time()
            state["duration"] = state["end_time"] - state["start_time"]

            result = {
                "status": "success",
                "workflow_id": workflow_id,
                "structured_cv": state["structured_cv"],
                "formatted_cv": formatted_result["formatted_cv_text"],
                "duration": state["duration"],
            }

            return result

        except Exception as e:
            logger.error(f"Error in workflow: {str(e)}\n{traceback.format_exc()}")
            return {
                "status": "error",
                "error": f"Workflow error: {str(e)}",
                "workflow_id": workflow_id,
            }

    def run_parsing_stage(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the parsing stage of the workflow.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state
        """
        try:
            logger.info(f"Starting parsing stage for workflow {state['workflow_id']}")

            # Parse job description and CV
            parse_result = self.parser_agent.run(
                {
                    "job_description": state["job_description"],
                    "cv_text": state["user_cv_data"],
                    "start_from_scratch": state.get("start_from_scratch", False),
                }
            )

            # Update state with parsing results
            state["parsed_job_description"] = parse_result["job_description_data"]
            state["structured_cv"] = parse_result["structured_cv"]
            state["stage"] = "parsed"

            logger.info("Parsing stage completed successfully")
            return state

        except Exception as e:
            logger.error(f"Error in parsing stage: {str(e)}")
            state["error"] = f"Parsing error: {str(e)}"
            return state

    def run_research_stage(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the research stage of the workflow.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state
        """
        try:
            logger.info(f"Starting research stage for workflow {state['workflow_id']}")

            # Run the research agent
            research_results = self.research_agent.run(
                {"job_description_data": state["parsed_job_description"]}
            )

            # Update state with research results
            state["research_results"] = research_results
            state["stage"] = "researched"

            return state

        except Exception as e:
            logger.error(f"Error in research stage: {str(e)}")
            state["error"] = f"Research error: {str(e)}"
            return state

    def run_content_generation_stage(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the content generation stage of the workflow.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state
        """
        try:
            logger.info(f"Starting content generation for workflow {state['workflow_id']}")

            # Check if we need to regenerate specific items
            regenerate_item_ids = state.get("regenerate_item_ids", [])

            if regenerate_item_ids:
                logger.info(f"Regenerating {len(regenerate_item_ids)} items")

                # Call the content writer agent to regenerate specific items
                structured_cv = self.content_writer_agent.run(
                    {
                        "structured_cv": state["structured_cv"],
                        "job_description_data": state["parsed_job_description"],
                        "research_results": state["research_results"],
                        "regenerate_item_ids": regenerate_item_ids,
                    }
                )

                # Update state with regenerated content
                state["structured_cv"] = structured_cv

            else:
                logger.info("Generating initial content")

                # Call the content writer agent for full content generation
                structured_cv = self.content_writer_agent.run(
                    {
                        "structured_cv": state["structured_cv"],
                        "job_description_data": state["parsed_job_description"],
                        "research_results": state["research_results"],
                        "regenerate_item_ids": [],
                    }
                )

                # Update state with generated content
                state["structured_cv"] = structured_cv

            state["stage"] = "content_generated"
            return state

        except Exception as e:
            logger.error(f"Error in content generation stage: {str(e)}")
            state["error"] = f"Content generation error: {str(e)}"
            return state

    def run_quality_assurance_stage(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the quality assurance stage of the workflow.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state
        """
        try:
            # Run QA checks on the generated content
            qa_result = self.quality_assurance_agent.run(
                {
                    "structured_cv": state["structured_cv"],
                    "job_description_data": state["parsed_job_description"],
                }
            )

            # Update state with QA results
            state["quality_check_results"] = qa_result.get("quality_check_results")
            state["structured_cv"] = qa_result.get("updated_structured_cv", state["structured_cv"])
            state["stage"] = "quality_checked"

            return state

        except Exception as e:
            logger.error(f"Error in quality assurance stage: {str(e)}")
            state["error"] = f"Quality assurance error: {str(e)}"
            return state

    def run_formatting_stage(self, structured_cv: StructuredCV) -> Dict[str, Any]:
        """
        Run the formatting stage.

        Args:
            structured_cv: The structured CV data.

        Returns:
            The formatted CV.
        """
        print("\n" + "=" * 50)
        print("STARTING FORMATTING STAGE")
        print("=" * 50)

        try:
            # Validate the input structured_cv
            if not structured_cv:
                print("Error: Empty structured_cv provided to formatter")
                return {
                    "formatted_cv_text": "# Error: No CV data available",
                    "error": "Empty CV data",
                }

            # Log the structure for debugging
            print(f"StructuredCV has {len(structured_cv.sections)} sections")
            for section in structured_cv.sections:
                print(
                    f"  Section: {section.name} ({len(section.items)} items, "
                    f"{len(section.subsections)} subsections)"
                )

            # Convert to ContentData for compatibility with formatter
            content_data = structured_cv.to_content_data()

            # Debug the content_data
            print(f"ContentData keys: {list(content_data.keys())}")

            # Sanity check the converted data
            if not content_data:
                print("Warning: StructuredCV.to_content_data() returned empty ContentData")
            else:
                # Check for key sections that should have content
                key_sections = ["summary", "skills_section", "experience_bullets"]
                for section in key_sections:
                    if section not in content_data or not content_data[section]:
                        print(f"Warning: ContentData is missing content for '{section}'")

            # Run the formatter agent
            print("Calling formatter_agent.run()...")
            result = self.formatter_agent.run({"content_data": content_data, "format_specs": {}})

            if not result:
                print("Error: formatter_agent.run() returned None")
                return {
                    "formatted_cv_text": "# Error: Formatter failed",
                    "error": "Formatter returned None",
                }

            print(f"Formatter result keys: {list(result.keys())}")

            if "formatted_cv_text" in result:
                # Validate the formatted text
                formatted_text = result["formatted_cv_text"]
                if not formatted_text:
                    print("Warning: Formatter returned empty formatted_cv_text")
                    formatted_text = "# Error: Empty formatted text"

                # Check for truncated content
                if "..." in formatted_text:
                    print("Warning: Possible truncated content in formatted text")

                # Store the formatted text in the state
                print("Formatting stage completed successfully")
                return {"formatted_cv_text": formatted_text}
            else:
                # There was an issue, return a fallback simple format
                print(
                    "Warning: Formatter did not return formatted_cv_text, using fallback formatting"
                )
                markdown = "# Tailored CV\n\n"

                # Add a summary if available
                summary = content_data.get("summary", "")
                if summary:
                    markdown += "## Professional Profile\n\n"
                    markdown += f"{summary}\n\n"
                    markdown += "---\n\n"

                # Add key qualifications if available
                skills = content_data.get("skills_section", "")
                if skills:
                    markdown += "## Key Qualifications\n\n"
                    markdown += f"{skills}\n\n"
                    markdown += "---\n\n"

                # Add professional experience if available
                experience = content_data.get("experience_bullets", [])
                if experience:
                    markdown += "## Professional Experience\n\n"
                    for exp in experience:
                        if isinstance(exp, dict):
                            position = exp.get("position", "")
                            if position:
                                markdown += f"### {position}\n\n"

                            # Add company info if available
                            company_info = []
                            if exp.get("company"):
                                company_info.append(exp["company"])
                            if exp.get("location"):
                                company_info.append(exp["location"])
                            if exp.get("period"):
                                company_info.append(exp["period"])

                            if company_info:
                                markdown += f"*{' | '.join(company_info)}*\n\n"

                            bullets = exp.get("bullets", [])
                            for bullet in bullets:
                                # Make sure bullet points are properly formatted
                                if bullet and not bullet.endswith((".", "!", "?")):
                                    bullet += "."
                                markdown += f"* {bullet}\n"

                            markdown += "\n"
                        else:
                            markdown += f"* {exp}\n"

                    markdown += "---\n\n"

                # Add projects if available
                projects = content_data.get("projects", [])
                if projects:
                    markdown += "## Project Experience\n\n"
                    for project in projects:
                        if isinstance(project, dict):
                            if project.get("name"):
                                tech_info = ""
                                if project.get("technologies"):
                                    if isinstance(project["technologies"], list):
                                        tech_info = ", ".join(project["technologies"])
                                    else:
                                        tech_info = str(project["technologies"])

                                if tech_info:
                                    markdown += f"### {project['name']} | {tech_info}\n\n"
                                else:
                                    markdown += f"### {project['name']}\n\n"

                            if project.get("description"):
                                markdown += f"{project['description']}\n\n"

                            for bullet in project.get("bullets", []):
                                if bullet and not bullet.endswith((".", "!", "?")):
                                    bullet += "."
                                markdown += f"* {bullet}\n"

                            markdown += "\n"
                        else:
                            markdown += f"* {project}\n"

                    markdown += "---\n\n"

                # Add education if available
                education = content_data.get("education", [])
                if education:
                    markdown += "## Education\n\n"
                    for edu in education:
                        if isinstance(edu, dict):
                            if edu.get("degree"):
                                edu_header = [edu["degree"]]
                                if edu.get("institution"):
                                    edu_header.append(edu["institution"])
                                if edu.get("location"):
                                    edu_header.append(edu["location"])

                                markdown += f"### {' | '.join(edu_header)}\n\n"

                                if edu.get("period"):
                                    markdown += f"*{edu['period']}*\n\n"

                                for detail in edu.get("details", []):
                                    markdown += f"* {detail}\n"

                                markdown += "\n"
                        else:
                            markdown += f"* {edu}\n"

                    markdown += "---\n\n"

                # Add certifications if available
                certifications = content_data.get("certifications", [])
                if certifications:
                    markdown += "## Certifications\n\n"
                    for cert in certifications:
                        if isinstance(cert, dict):
                            cert_text = cert.get("name", "")
                            if cert.get("url"):
                                cert_text = f"[{cert_text}]({cert.get('url')})"
                            if cert.get("issuer"):
                                cert_text += f" - {cert.get('issuer')}"
                            if cert.get("date"):
                                cert_text += f", {cert.get('date')}"

                            markdown += f"* {cert_text}\n"
                        else:
                            markdown += f"* {cert}\n"

                    markdown += "---\n\n"

                # Add languages if available
                languages = content_data.get("languages", [])
                if languages:
                    markdown += "## Languages\n\n"
                    for lang in languages:
                        if isinstance(lang, dict):
                            lang_text = lang.get("name", "")
                            if lang.get("level"):
                                lang_text += f" ({lang.get('level')})"

                            markdown += f"* {lang_text}\n"
                        else:
                            markdown += f"* {lang}\n"

                return {"formatted_cv_text": markdown}

        except Exception as e:
            traceback = traceback
            logger.error(f"Error in formatting stage: {str(e)}\n{traceback.format_exc()}")
            return {
                "formatted_cv_text": "# Error occurred during formatting\n\nPlease try again.",
                "error": f"Formatting error: {str(e)}",
            }

    def process_user_feedback(self, workflow_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user feedback on generated content.

        Args:
            workflow_id: The ID of the workflow.
            feedback: Dictionary containing feedback data including:
                - accepted_items: List of item IDs that are accepted
                - rejected_items: List of item IDs that need regeneration
                - edited_items: Dict mapping item IDs to edited content
                - feedback_comments: Dict mapping item IDs to feedback text

        Returns:
            Dict with status and potentially regenerated content
        """
        try:
            logger.info(f"Processing user feedback for workflow {workflow_id}")

            # Update the state based on feedback
            # TODO: Implement proper state management
            state = {"workflow_id": workflow_id}

            # Items to regenerate
            regenerate_item_ids = feedback.get("rejected_items", [])

            if regenerate_item_ids:
                # Call run_workflow with regeneration list
                return self.run_workflow(
                    job_description="",  # Will be loaded from state
                    user_cv_data="",  # Will be loaded from state
                    workflow_id=workflow_id,
                    regenerate_item_ids=regenerate_item_ids,
                )

            return {"status": "feedback_processed", "workflow_id": workflow_id}

        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return {"status": "error", "error": f"Feedback processing error: {str(e)}"}

    def _get_items_for_sections(self, workflow_id: str, section_names: List[str]) -> List[str]:
        """
        Get all item IDs for specified sections.

        Args:
            workflow_id: The workflow ID
            section_names: List of section names to get items for

        Returns:
            List of item IDs belonging to the specified sections
        """
        # This is a placeholder implementation
        # TODO: Implement proper state management to retrieve items by section
        return []

    def parse_job_description_node(self, job_description):
        """LangGraph node for parsing job descriptions."""
        # Simple wrapper around parser_agent for use in LangGraph
        return self.parser_agent.run({"job_description": job_description})

    def run_research_node(self, parsed_job_data):
        """LangGraph node for research."""
        # Simple wrapper around research_agent for use in LangGraph
        return self.research_agent.run({"job_description_data": parsed_job_data})

    def search_vector_store_node(self, parsed_job_data):
        """LangGraph node for vector store search."""
        # Simple wrapper around vector_store_agent for use in LangGraph
        return self.vector_store_agent.run({"query": parsed_job_data})
