from agent_base import AgentBase
from llm import LLM
from state_manager import (
    JobDescriptionData, 
    ContentData, 
    AgentIO, 
    ExperienceEntry, 
    CVData,
    StructuredCV,
    Section,
    Subsection,
    Item,
    ItemStatus,
    ItemType
)
from typing import List, Dict, Any, Optional
import json
from tools_agent import ToolsAgent
import logging
import os
import re
import time
import traceback

# For progress updates in Streamlit UI
try:
    import streamlit as st
except ImportError:
    st = None

# Set up logging with enhanced format
logger = logging.getLogger(__name__)

def update_progress(message):
    """
    Updates the progress message in the Streamlit UI if available.
    
    Args:
        message: The progress message to display
    """
    # Check if we're running in a Streamlit context with progress placeholder
    if st and hasattr(st, 'session_state') and 'progress_placeholder' in st.session_state:
        try:
            # Update the progress message
            st.session_state.current_generation_stage = message
            st.session_state.progress_placeholder.info(f"🔄 {message}")
        except Exception as e:
            # Silent fail if Streamlit is not properly initialized
            pass

def log_execution_time(func):
    """
    Decorator that logs execution time of functions.
    
    Args:
        func: The function to be decorated
        
    Returns:
        Wrapper function that times and logs execution
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info("%s completed in %.2fs", func.__name__, end_time - start_time)
            return result
        except Exception as e:
            end_time = time.time()
            logger.error("%s failed after %.2fs: %s\n%s", 
                         func.__name__, end_time - start_time, str(e), 
                         traceback.format_exc())
            raise
    return wrapper

class PromptLoader:
    """
    Utility class for loading prompt templates from the prompts_folder directory.
    This fulfills REQ-FUNC-GEN-1 from the SRS.
    """
   
    def __init__(self, prompts_dir="prompts_folder"):
        """
        Initialize the PromptLoader.
        
        Args:
            prompts_dir: Path to the directory containing prompt templates.
        """
        self.prompts_dir = prompts_dir
        self.cache = {}  # Cache loaded prompts to avoid repeated disk reads
        
    def load_prompt(self, prompt_name):
        """
        Load a prompt template from a file.
        
        Args:
            prompt_name: Name of the prompt file (with or without .md extension)
        
        Returns:
            The prompt template text, or a default prompt if the file is not found.
        """
        # Check if prompt is already cached
        if prompt_name in self.cache:
            return self.cache[prompt_name]
        
        # Add .md extension if not present
        if not prompt_name.endswith('.md'):
            prompt_name = f"{prompt_name}.md"
        
        # Construct full path
        prompt_path = os.path.join(self.prompts_dir, prompt_name)
        
        try:
            # Try to read the prompt file
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
                
            # Cache the prompt
            self.cache[prompt_name] = prompt_text
            logger.info("Successfully loaded prompt: %s", prompt_name)
            return prompt_text
            
        except (IOError, UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.error("Error loading prompt %s: %s", prompt_name, str(e))
            return "Generate professional CV content based on the job requirements."

    def get_key_qualifications_prompt(self, job_data, research_results):
        """
        Get the specialized prompt for Key Qualifications section.
        
        Args:
            job_data: Job description data
            research_results: Research results from ResearchAgent
            
        Returns:
            Formatted prompt for Key Qualifications
        """
        template = self.load_prompt("key_qualifications_prompt")
        
        # Extract relevant data for prompt substitution
        main_jd = job_data.get("raw_text", "") if isinstance(job_data, dict) else getattr(job_data, "raw_text", "")
        
        # Extract any additional data from research_results
        talents = []
        if research_results and "job_requirements_analysis" in research_results:
            analysis = research_results["job_requirements_analysis"]
            if "core_technical_skills" in analysis:
                talents.extend(analysis["core_technical_skills"][:3])
            if "soft_skills" in analysis:
                talents.extend(analysis["soft_skills"][:2])
        
        # For MVP, we'll use a simplified version with less substitution
        formatted_prompt = template.replace("{{main_job_description_raw}}", main_jd)
        
        # Replace other placeholders with defaults if they're in the template
        placeholders = [
            "{{similar_job_description_raw_1}}", 
            "{{similar_job_description_raw_2}}",
            "{{similar_job_description_raw_3}}",
            "{{similar_job_description_raw_4}}",
            "{{my_talents}}"
        ]
        replacement_values = [
            "Not provided", 
            "Not provided",
            "Not provided",
            "Not provided",
            ", ".join(talents) if talents else "Not specified"
        ]
        
        for placeholder, value in zip(placeholders, replacement_values):
            formatted_prompt = formatted_prompt.replace(placeholder, value)
            
        return formatted_prompt
    
    def get_executive_summary_prompt(self, job_data, cv_data, research_results):
        """
        Get the specialized prompt for Executive Summary section.
        
        Args:
            job_data: Job description data
            cv_data: CV data or content generated so far
            research_results: Research results from ResearchAgent
            
        Returns:
            Formatted prompt for Executive Summary
        """
        template = self.load_prompt("executive_summary_prompt")
        
        # Extract job description information
        main_jd = job_data.get("raw_text", "") if isinstance(job_data, dict) else getattr(job_data, "raw_text", "")
        
        # Enhanced prompt with proper placeholder replacements
        formatted_prompt = template
        
        # Extract key skills and experience from research results
        skills = []
        experience = ""
        projects = []
        
        if research_results and "job_requirements_analysis" in research_results:
            analysis = research_results["job_requirements_analysis"]
            if "core_technical_skills" in analysis:
                skills = analysis["core_technical_skills"][:5]
        
        # Replace placeholders in the executive summary prompt
        formatted_prompt = formatted_prompt.replace("{{big_6}}", ", ".join(skills))
        formatted_prompt = formatted_prompt.replace("{{professional_experience}}", experience)
        formatted_prompt = formatted_prompt.replace("{{side_projects}}", ", ".join(projects))
        
        return formatted_prompt
    
    def get_resume_role_prompt(self, job_data, role_data, research_results):
        """
        Get the specialized prompt for resume roles/experience.
        
        Args:
            job_data: Job description data
            role_data: Data about the specific role to tailor
            research_results: Research results from ResearchAgent
            
        Returns:
            Formatted prompt for resume role content
        """
        template = self.load_prompt("resume_role_prompt")
        
        # Extract skills from job data for proper target skills formatting
        skills = []
        if isinstance(job_data, dict) and "skills" in job_data:
            skills = job_data["skills"]
        elif hasattr(job_data, "skills"):
            skills = job_data.skills
        
        # Properly replace <skills> tags with actual skills from job description
        skills_section = "<skills>\n" + "\n".join(skills) + "\n</skills>"
        formatted_prompt = template.replace("<skills>\n{{Target Skills}}\n</skills>", skills_section)
        
        return formatted_prompt
    
    def get_side_project_prompt(self, job_data, project_data, research_results):
        """
        Get the specialized prompt for side projects.
        
        Args:
            job_data: Job description data
            project_data: Data about the specific project to tailor
            research_results: Research results from ResearchAgent
            
        Returns:
            Formatted prompt for side project content
        """
        template = self.load_prompt("side_project_prompt")
        
        # Extract job description info for better tailoring
        job_description = job_data.get("raw_text", "") if isinstance(job_data, dict) else getattr(job_data, "raw_text", "")
        skills_needed = []
        
        if isinstance(job_data, dict) and "skills" in job_data:
            skills_needed = job_data["skills"]
        elif hasattr(job_data, "skills"):
            skills_needed = job_data.skills
            
        # Add job description details to the prompt
        formatted_prompt = template
        formatted_prompt += f"\n\nTarget Job Skills: {', '.join(skills_needed)}\n"
        formatted_prompt += f"\nJob Context: {job_description[:500]}..."
        
        return formatted_prompt

class ContentWriterAgent(AgentBase):
    """
    Agent responsible for generating tailored CV content based on job requirements 
    and user experiences. Updated to work with StructuredCV data model.
    """
    def __init__(self, name: str, description: str, llm: LLM, tools_agent: ToolsAgent):
        """
        Initializes the ContentWriterAgent.

        Args:
            name: The name of the agent.
            description: A description of the agent.
            llm: The LLM instance to use for content generation.
            tools_agent: The ToolsAgent instance for content processing.
        """
        super().__init__(
            name=name,
            description=description,
            input_schema=AgentIO(
                input={
                    "job_description_data": Dict[str, Any],
                    "structured_cv": StructuredCV,
                    "regenerate_item_ids": List[str],
                    "research_results": Dict[str, Any],
                },
                output=StructuredCV, # The output is now a StructuredCV object
                description="Generates content for a StructuredCV based on job description, research, and feedback.",
            ),
            output_schema=AgentIO(
                input={
                     "job_description_data": Dict[str, Any],
                    "structured_cv": StructuredCV,
                    "regenerate_item_ids": List[str],
                     "research_results": Dict[str, Any],
                },
                output=StructuredCV,
                description="Updated StructuredCV with generated content.",
            ),
        )
        self.llm = llm
        self.tools_agent = tools_agent # Store ToolsAgent instance
        self.prompt_loader = PromptLoader()  # Initialize the prompt loader
        logger.info("Initialized %s with LLM and Tools Agent", name)

    @log_execution_time
    def run(self, input_data: Dict[str, Any]) -> StructuredCV:
        """
        Legacy method for compatibility with existing code. 
        Delegates to the newer execute method.
        """
        structured_cv = input_data.get("structured_cv")
        job_description_data = input_data.get("job_description_data")
        research_results = input_data.get("research_results", {})
        regenerate_item_ids = input_data.get("regenerate_item_ids", [])
        
        # First, ensure STATIC sections are respected
        self._ensure_static_sections_preserved(structured_cv)
        
        if regenerate_item_ids and len(regenerate_item_ids) > 0:
            return self._regenerate_specific_items(
                structured_cv=structured_cv,
                item_ids=regenerate_item_ids,
                job_description_data=job_description_data,
                research_results=research_results
            )
        else:
            return self._generate_all_dynamic_content(
                structured_cv=structured_cv,
                job_description_data=job_description_data,
                research_results=research_results
            )
    
    def _ensure_static_sections_preserved(self, structured_cv: StructuredCV):
        """
        Makes sure all STATIC sections are preserved and never regenerated.
        
        Args:
            structured_cv: The CV structure to check and protect
        """
        if not structured_cv:
            return
            
        print("\n🔒 Ensuring STATIC sections remain unchanged...")
        update_progress("Preserving static sections...")
        
        # Explicitly define sections that should always be STATIC
        always_static_sections = [
            "Education", "Certifications", "Languages", "Personal Information", 
            "Hobbies", "Additional Information", "References", "Contact Information",
            "**Education:**", "**Certifications:**", "**Languages:**"
        ]
        
        static_sections = []
        for section in structured_cv.sections:
            # Check if this is a section that should always be static
            if section.name in always_static_sections:
                print(f"  Enforcing STATIC status for section: {section.name}")
                section.content_type = "STATIC"
            
            # Process STATIC sections
            if section.content_type == "STATIC":
                static_sections.append(section.name)
                
                # Make sure all items in this section are also marked as STATIC
                for item in section.items:
                    if item.status != ItemStatus.STATIC:
                        print(f"  Protecting item in '{section.name}' as STATIC")
                        item.status = ItemStatus.STATIC
                        
                # Also check subsections
                for subsection in section.subsections:
                    for item in subsection.items:
                        if item.status != ItemStatus.STATIC:
                            print(f"  Protecting item in '{section.name}/{subsection.name}' as STATIC")
                            item.status = ItemStatus.STATIC
        
        if static_sections:
            print(f"  Protected sections: {', '.join(static_sections)}")
        
    @log_execution_time
    def execute(self, task_type: str, **kwargs) -> StructuredCV:
        """
        Execute a specific content generation task at the section level.
        
        Args:
            task_type: Type of task to execute ("generate_section" or "regenerate_section")
            **kwargs: Additional arguments depending on task_type:
                For generate_section:
                    - section_name: Name of the section to generate
                    - structured_cv: Current StructuredCV object
                    - research_results: Research findings for relevance
                For regenerate_section:
                    - section_id: ID of the section to regenerate
                    - structured_cv: Current StructuredCV object
                    - research_results: Research findings for relevance
            
        Returns:
            Updated StructuredCV with generated content
        """
        structured_cv = kwargs.get("structured_cv")
        research_results = kwargs.get("research_results", {})
        
        # First, ensure STATIC sections are respected
        self._ensure_static_sections_preserved(structured_cv)
        
        # Extract or create job description data from structured_cv metadata
        job_description_data = {}
        if "main_jd_text" in structured_cv.metadata:
            job_description_data["raw_text"] = structured_cv.metadata.get("main_jd_text", "")
            
        # Extract job focus information once for efficiency
        job_focus = self._extract_job_focus(job_description_data, research_results)
        
        if task_type == "generate_section":
            section_name = kwargs.get("section_name")
            if not section_name:
                logger.error("Section name required for generate_section task")
                return structured_cv
                
            section = structured_cv.get_section_by_name(section_name)
            if not section:
                logger.error(f"Section '{section_name}' not found in CV")
                return structured_cv
                
            # Skip generation for STATIC sections
            if section.content_type == "STATIC":
                print(f"🔒 Skipping generation for STATIC section: {section_name}")
                return structured_cv
                
            logger.info(f"Generating content for section: {section_name}")
            return self._generate_section_content(
                section=section,
                structured_cv=structured_cv,
                job_description_data=job_description_data,
                research_results=research_results,
                job_focus=job_focus
            )
            
        elif task_type == "regenerate_section":
            section_id = kwargs.get("section_id")
            if not section_id:
                logger.error("Section ID required for regenerate_section task")
                return structured_cv
                
            section = structured_cv.find_section_by_id(section_id)
            if not section:
                logger.error(f"Section with ID '{section_id}' not found in CV")
                return structured_cv
                
            # Skip regeneration for STATIC sections
            if section.content_type == "STATIC":
                print(f"🔒 Skipping regeneration for STATIC section: {section.name}")
                return structured_cv
                
            logger.info(f"Regenerating content for section: {section.name}")
            return self._generate_section_content(
                section=section,
                structured_cv=structured_cv,
                job_description_data=job_description_data,
                research_results=research_results,
                job_focus=job_focus,
                is_regeneration=True
            )
            
        else:
            logger.error(f"Unknown task type: {task_type}")
            return structured_cv
            
    @log_execution_time
    def _generate_section_content(
        self,
        section: Section,
        structured_cv: StructuredCV,
        job_description_data: Dict[str, Any],
        research_results: Dict[str, Any],
        job_focus: Dict[str, Any],
        is_regeneration: bool = False
    ) -> StructuredCV:
        """
        Generate content for a specific section.
        
        Args:
            section: The section to generate content for
            structured_cv: The current structured CV
            job_description_data: Job description data
            research_results: Research results data
            job_focus: Extracted job focus information
            is_regeneration: Whether this is a regeneration request
            
        Returns:
            Updated StructuredCV with generated content
        """
        # Skip generation completely if this is a STATIC section
        if section.content_type == "STATIC":
            print(f"🔒 Skipping generation for STATIC section: {section.name}")
            return structured_cv
            
        # For items explicitly marked as STATIC, respect their status
        # Check both direct items and items in subsections
        static_item_count = 0
        for item in section.items:
            if item.status == ItemStatus.STATIC:
                static_item_count += 1
                
        for subsection in section.subsections:
            for item in subsection.items:
                if item.status == ItemStatus.STATIC:
                    static_item_count += 1
                    
        if static_item_count > 0:
            print(f"🔒 Respecting {static_item_count} STATIC items in section: {section.name}")
            
        section_name = section.name.lower()
        
        # For Project Experience or Experience sections with many items, 
        # use the split generation approach to avoid resource exhaustion
        if "project" in section_name or (
            "experience" in section_name and 
            sum(len(subsection.items) for subsection in section.subsections) > 4
        ):
            logger.info(f"Using split generation for complex section: {section.name}")
            print(f"🔄 Using batched generation for complex section: {section.name}")
            self._split_generation_task(
                section=section,
                structured_cv=structured_cv,
                job_description_data=job_description_data,
                research_results=research_results,
                job_focus=job_focus
            )
        else:
            # Use the standard approach for simpler sections
            if "key qualifications" in section_name or "core competencies" in section_name or "skills" in section_name:
                print(f"🔄 Generating Key Qualifications content...")
                self._generate_key_qualifications(
                    section=section,
                    job_description_data=job_description_data,
                    research_results=research_results,
                    structured_cv=structured_cv,
                    job_focus=job_focus
                )
                
            elif "experience" in section_name and "project" not in section_name:
                print(f"🔄 Generating Professional Experience content...")
                self._generate_experience_content(
                    section=section,
                    job_description_data=job_description_data,
                    research_results=research_results,
                    structured_cv=structured_cv,
                    job_focus=job_focus
                )
                
            elif "project" in section_name:
                print(f"🔄 Generating Project Experience content...")
                self._generate_projects_content(
                    section=section,
                    job_description_data=job_description_data,
                    research_results=research_results,
                    structured_cv=structured_cv,
                    job_focus=job_focus
                )
                
            elif "summary" in section_name or "profile" in section_name:
                print(f"🔄 Generating Executive Summary content...")
                self._generate_summary_content(
                    section=section,
                    job_description_data=job_description_data,
                    research_results=research_results,
                    structured_cv=structured_cv,
                    job_focus=job_focus
                )
                
            else:
                logger.warning(f"Unsupported section type for content generation: {section.name}")
            
        # Update section status after generation
        section.status = ItemStatus.GENERATED
            
        return structured_cv
    
    @log_execution_time
    def _regenerate_specific_items(
        self, 
        structured_cv: StructuredCV, 
        item_ids: List[str],
        job_description_data: Dict[str, Any],
        research_results: Dict[str, Any]
    ) -> StructuredCV:
        """
        Regenerates specific items or sections in the StructuredCV.

        Args:
            structured_cv: The current CV structure
            item_ids: List of item IDs or section IDs to regenerate
            job_description_data: Parsed job description data
            research_results: Research findings

        Returns:
            The updated StructuredCV with regenerated content.
        """
        print("\n🔄 Regenerating specific items requested by user...")
        update_progress("Preparing for regeneration...")
        
        # First, ensure all static sections are preserved
        self._ensure_static_sections_preserved(structured_cv)
        
        job_focus = self._extract_job_focus(job_description_data, research_results)
        
        section_ids = []
        item_ids_only = []
        
        # First separate section IDs from item IDs
        for item_id in item_ids:
            section = structured_cv.find_section_by_id(item_id)
            if section:
                section_ids.append(item_id)
            else:
                item_ids_only.append(item_id)
                
        # Process sections first (using the batched approach internally)
        if section_ids:
            print(f"🔄 Processing {len(section_ids)} sections for regeneration")
            update_progress(f"Processing {len(section_ids)} sections for regeneration")
            
        for i, section_id in enumerate(section_ids):
            section = structured_cv.find_section_by_id(section_id)
            if not section:
                continue
                
            # Skip STATIC sections
            if section.content_type == "STATIC":
                print(f"🔒 Skipping STATIC section: {section.name}")
                update_progress(f"Skipping static section: {section.name}")
                continue
                
            print(f"🔄 Regenerating section: {section.name} ({i+1}/{len(section_ids)})")
            update_progress(f"Regenerating section: {section.name} ({i+1}/{len(section_ids)})")
            logger.info(f"Regenerating entire section: {section.name}")
            # Use the section-level generation method which now includes batching
            self._generate_section_content(
                section=section,
                structured_cv=structured_cv,
                job_description_data=job_description_data,
                research_results=research_results,
                job_focus=job_focus,
                is_regeneration=True
            )
        
        # Process individual items with delays between to avoid resource exhaustion
        if item_ids_only:
            print(f"🔄 Regenerating {len(item_ids_only)} individual items in batches")
            update_progress(f"Regenerating {len(item_ids_only)} individual items")
            
            # Filter out items from STATIC sections
            filtered_item_ids = []
            for item_id in item_ids_only:
                item, section, subsection = structured_cv.find_item_by_id(item_id)
                if not item:
                    continue
                    
                if item.status == ItemStatus.STATIC or (section and section.content_type == "STATIC"):
                    print(f"🔒 Skipping STATIC item: {item_id}")
                    continue
                    
                filtered_item_ids.append(item_id)
                
            logger.info(f"Regenerating {len(filtered_item_ids)} non-static items in batches")
            
            for i, item_id in enumerate(filtered_item_ids):
                # Find item, section, subsection
                item, section, subsection = structured_cv.find_item_by_id(item_id)
                if not item:
                    logger.warning(f"Item with ID {item_id} not found in StructuredCV")
                    continue
                
                # Skip if this item is in a STATIC section or is itself STATIC
                if item.status == ItemStatus.STATIC or (section and section.content_type == "STATIC"):
                    continue
                
                print(f"🔄 Regenerating item {i+1}/{len(filtered_item_ids)} in section {section.name}")
                update_progress(f"Regenerating item {i+1}/{len(filtered_item_ids)} in {section.name}")
                logger.info(f"Regenerating item {i+1}/{len(filtered_item_ids)} in section {section.name}")
                
                try:
                    # Generate content based on item type and context
                    generated_content = self._generate_item_content(
                        item=item,
                        section=section,
                        subsection=subsection,
                        job_description_data=job_description_data,
                        research_results=research_results,
                        structured_cv=structured_cv,
                        job_focus=job_focus
                    )
                    
                    # Update item with generated content
                    item.content = generated_content
                    item.status = ItemStatus.GENERATED
                    
                except Exception as e:
                    logger.error(f"Error regenerating item {item_id}: {str(e)}")
                    item.content = "Content generation failed. Please try regenerating individually."
                    item.status = ItemStatus.GENERATED  # Mark as generated to avoid repeated errors
                
                # Add a small delay between items to avoid rate limiting
                if i < len(filtered_item_ids) - 1:
                    time.sleep(0.5)
        
        print("✅ Regeneration completed")
        return structured_cv
  
    @log_execution_time
    def _generate_all_dynamic_content(
        self,
        structured_cv: StructuredCV,
        job_description_data: Dict[str, Any],
        research_results: Dict[str, Any]
    ) -> StructuredCV:
        """
        Generate content for all dynamic sections in the CV.

        Args:
            structured_cv: The structured CV to generate content for
            job_description_data: Parsed job description data
            research_results: Research findings for relevance

        Returns:
            Updated structured CV with generated content
        """
        # First, make sure all essential sections exist
        self._ensure_dynamic_essential_sections(structured_cv)
        
        # Next, ensure all static sections are preserved
        self._ensure_static_sections_preserved(structured_cv)
        
        # Update the progress indicator
        update_progress("Analyzing job requirements...")
        
        # Extract job focus once for efficiency
        job_focus = self._extract_job_focus(job_description_data, research_results)
        
        # Generate content section by section, in a specific order for best results
        logger.info("Generating content for all dynamic sections")
        print("\n🔄 Generating tailored CV content...")
        
        # Track sections to process
        sections_to_process = []
        
        # 1. First identify Key Qualifications
        key_quals_section = structured_cv.get_section_by_name("Key Qualifications")
        if key_quals_section and key_quals_section.content_type == "DYNAMIC":
            sections_to_process.append(("Key Qualifications", key_quals_section))
        
        # 2. Identify Professional Experience
        experience_section = structured_cv.get_section_by_name("Professional Experience")
        if experience_section and experience_section.content_type == "DYNAMIC":
            sections_to_process.append(("Professional Experience", experience_section))
        
        # 3. Identify Project Experience
        projects_section = structured_cv.get_section_by_name("Project Experience")
        if projects_section and projects_section.content_type == "DYNAMIC":
            sections_to_process.append(("Project Experience", projects_section))
        
        # 4. Identify Executive Summary (last, since it depends on other content)
        summary_section = structured_cv.get_section_by_name("Executive Summary")
        if summary_section and summary_section.content_type == "DYNAMIC":
            sections_to_process.append(("Executive Summary", summary_section))
            
        # Process each section with a delay between to avoid resource exhaustion
        for i, (section_name, section) in enumerate(sections_to_process):
            # Update progress in the UI
            update_progress(f"Generating {section_name} section ({i+1}/{len(sections_to_process)})")
            
            print(f"🔄 Generating {section_name} section ({i+1}/{len(sections_to_process)})")
            logger.info(f"Generating {section_name} section ({i+1}/{len(sections_to_process)})")
            
            try:
                self._generate_section_content(
                    section=section,
                    structured_cv=structured_cv,
                    job_description_data=job_description_data,
                    research_results=research_results,
                    job_focus=job_focus
                )
                
                # Add a delay between section generation to avoid rate limiting
                if i < len(sections_to_process) - 1:
                    logger.info(f"Pausing briefly before generating next section")
                    time.sleep(1.0)  # Longer delay between sections
                    
            except Exception as e:
                logger.error(f"Error generating {section_name} section: {str(e)}")
                # Continue with the next section even if this one fails
        
        # Update completion status
        update_progress("CV content generation completed!")
        print("✅ CV content generation completed")
        return structured_cv
    
    def _ensure_dynamic_essential_sections(self, structured_cv: StructuredCV):
        """
        Ensures essential sections exist and are marked as DYNAMIC.
        
        Args:
            structured_cv: The CV structure to check and modify
        """
        print("\n🔄 Ensuring essential sections are properly configured...")
        
        # Essential section names (including alternative names)
        essential_section_names = [
            "Professional profile:", "Executive Summary", "Summary", "About Me", 
            "Key Qualifications", "**Key Qualifications:**", "Skills", "Core Competencies"
        ]
        
        # Loop through all sections and ensure essential ones are marked as DYNAMIC
        for section in structured_cv.sections:
            if section.name in essential_section_names:
                print(f"  Setting section '{section.name}' to DYNAMIC (was: {section.content_type})")
                section.content_type = "DYNAMIC"
                
                # For existing items in these sections, mark them for regeneration
                for item in section.items:
                    if item.status == ItemStatus.STATIC:
                        print(f"    Setting item in '{section.name}' to TO_REGENERATE (was: {item.status})")
                        item.status = ItemStatus.TO_REGENERATE
    
    @log_execution_time
    def _split_generation_task(
        self,
        section: Section,
        structured_cv: StructuredCV,
        job_description_data: Dict[str, Any],
        research_results: Dict[str, Any],
        job_focus: Dict[str, Any]
    ) -> None:
        """
        Split content generation into batches to avoid resource exhaustion.
        
        Args:
            section: The section to generate content for
            structured_cv: The full CV structure
            job_description_data: Parsed job description data
            research_results: Research findings
            job_focus: Key focus areas from the job
        """
        # Skip if this is a STATIC section
        if section.content_type == "STATIC":
            print(f"🔒 Skipping generation for STATIC section: {section.name}")
            update_progress(f"Skipping static section: {section.name}")
            return
            
        # Process each item/subsection independently with delays between calls
        if section.items:
            logger.info(f"Processing {len(section.items)} items in section {section.name} individually")
            print(f"🔄 Processing {len(section.items)} items in {section.name} section...")
            update_progress(f"Processing {section.name} items...")
            
            # Count how many items need generation
            items_to_generate = [item for item in section.items 
                               if (item.status == ItemStatus.INITIAL or item.status == ItemStatus.TO_REGENERATE)
                               and item.status != ItemStatus.STATIC]  # Skip STATIC items
            
            logger.info(f"Found {len(items_to_generate)} items needing generation in {section.name}")
            
            for i, item in enumerate(items_to_generate):
                try:
                    # Skip STATIC items
                    if item.status == ItemStatus.STATIC:
                        print(f"🔒 Skipping STATIC item {i+1} in section {section.name}")
                        continue
                        
                    # Update progress
                    update_progress(f"Generating item {i+1}/{len(items_to_generate)} in {section.name}...")
                    
                    # Generate content for this specific item only
                    content = self._generate_item_content(
                        item=item,
                        section=section,
                        subsection=None,
                        job_description_data=job_description_data,
                        research_results=research_results,
                        structured_cv=structured_cv,
                        job_focus=job_focus
                    )
                    
                    # Update item with generated content
                    item.content = content
                    item.status = ItemStatus.GENERATED
                    
                    # Add a longer delay between requests to respect rate limits
                    if i < len(items_to_generate) - 1:
                        logger.info(f"Pausing after item {i+1}/{len(items_to_generate)}")
                        # LLM class now has rate limiting, so we don't need additional delays here
                    
                except Exception as e:
                    logger.error(f"Error in isolated generation for item in {section.name}: {str(e)}")
                    item.content = "Content generation failed. Please try regenerating with a simpler prompt."
                    item.status = ItemStatus.GENERATED
        
        # Process subsections if any
        for subsection_index, subsection in enumerate(section.subsections):
            logger.info(f"Processing subsection {subsection.name} items individually")
            print(f"🔄 Processing {subsection.name} subsection...")
            update_progress(f"Processing {subsection.name} in {section.name}...")
            
            # Count items needing generation in this subsection
            items_to_generate = [item for item in subsection.items 
                               if (item.status == ItemStatus.INITIAL or item.status == ItemStatus.TO_REGENERATE)
                               and item.status != ItemStatus.STATIC]  # Skip STATIC items
            
            logger.info(f"Found {len(items_to_generate)} items needing generation in subsection {subsection.name}")
            
            for i, item in enumerate(items_to_generate):
                try:
                    # Skip STATIC items
                    if item.status == ItemStatus.STATIC:
                        print(f"🔒 Skipping STATIC item {i+1} in subsection {subsection.name}")
                        continue
                        
                    # Update progress
                    update_progress(f"Generating item {i+1}/{len(items_to_generate)} in {subsection.name}...")
                    
                    # Generate content for this specific item only
                    content = self._generate_item_content(
                        item=item,
                        section=section,
                        subsection=subsection,
                        job_description_data=job_description_data,
                        research_results=research_results,
                        structured_cv=structured_cv,
                        job_focus=job_focus
                    )
                    
                    # Update item with generated content
                    item.content = content
                    item.status = ItemStatus.GENERATED
                    
                    # LLM class now has rate limiting, so we don't need additional delays here
                    
                except Exception as e:
                    logger.error(f"Error in isolated generation for item in {subsection.name}: {str(e)}")
                    item.content = "Content generation failed. Please try regenerating with a simpler prompt."
                    item.status = ItemStatus.GENERATED
            
            # If we have more subsections to process, add a small wait
            if subsection_index < len(section.subsections) - 1:
                logger.info(f"Finished subsection {subsection_index+1}/{len(section.subsections)}")
                # LLM class now has rate limiting, so we don't need additional delays here

    def _extract_job_focus(self, job_description_data: Dict[str, Any], research_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts key focus areas from the job description for better tailoring.
        
        Args:
            job_description_data: Parsed job description data
            research_results: Research findings
            
        Returns:
            Dictionary of job focus information
        """
        job_focus = {
            "primary_skills": [],
            "primary_responsibilities": [],
            "industry_context": [],
            "company_values": [],
            "job_title": "",
            "required_experience": "",
            "education_requirements": [],
            "key_technologies": []
        }
        
        # Initialize variables to avoid linter errors
        skills = []
        responsibilities = []
        industry_terms = []
        company_values = []
        
        # Extract job title if available
        if hasattr(job_description_data, "get") and callable(job_description_data.get):
            # It's a dictionary-like object
            job_focus["job_title"] = job_description_data.get("title", "")
            skills = job_description_data.get("skills", [])
            responsibilities = job_description_data.get("responsibilities", [])
            industry_terms = job_description_data.get("industry_terms", [])
            company_values = job_description_data.get("company_values", [])
            
            # Try to extract additional useful information
            job_focus["required_experience"] = job_description_data.get("required_experience", "")
            job_focus["education_requirements"] = job_description_data.get("education_requirements", [])
            
            # Extract technologies from job description
            if "technologies" in job_description_data:
                job_focus["key_technologies"] = job_description_data.get("technologies", [])
            elif "technology_stack" in job_description_data:
                job_focus["key_technologies"] = job_description_data.get("technology_stack", [])
            
        elif hasattr(job_description_data, "skills"):
            # It's a JobDescriptionData object
            if hasattr(job_description_data, "title"):
                job_focus["job_title"] = getattr(job_description_data, "title", "")
            
            skills = getattr(job_description_data, "skills", [])
            responsibilities = getattr(job_description_data, "responsibilities", [])
            industry_terms = getattr(job_description_data, "industry_terms", [])
            company_values = getattr(job_description_data, "company_values", [])
            
            # Try to extract additional useful information
            if hasattr(job_description_data, "required_experience"):
                job_focus["required_experience"] = getattr(job_description_data, "required_experience", "")
            if hasattr(job_description_data, "education_requirements"):
                job_focus["education_requirements"] = getattr(job_description_data, "education_requirements", [])
            
            # Extract technologies from job description
            if hasattr(job_description_data, "technologies"):
                job_focus["key_technologies"] = getattr(job_description_data, "technologies", [])
            elif hasattr(job_description_data, "technology_stack"):
                job_focus["key_technologies"] = getattr(job_description_data, "technology_stack", [])
        
        # If raw text is available, try to extract more information using common patterns
        raw_text = ""
        if hasattr(job_description_data, "get") and callable(job_description_data.get):
            raw_text = job_description_data.get("raw_text", "")
        elif hasattr(job_description_data, "raw_text"):
            raw_text = getattr(job_description_data, "raw_text", "")
        
        if raw_text and not job_focus["job_title"]:
            # Try to extract job title using common patterns
            title_patterns = [
                r"(?:job title|position):\s*([^\n]+)",
                r"(?:we are hiring|hiring for)[^\n]*?(?:a|an)\s+([^\n,\.]+)",
                r"(?:^|\n\s*)([a-zA-Z\s]+?(?:engineer|developer|designer|manager|specialist|analyst|consultant|advisor|administrator|technician|architect|lead|director|officer|assistant|coordinator|associate))[^\n]*?(?:\n|$)"
            ]
            
            for pattern in title_patterns:
                match = re.search(pattern, raw_text, re.IGNORECASE)
                if match:
                    job_focus["job_title"] = match.group(1).strip()
                    break
        
        # Process technologies from raw text if not found previously
        if raw_text and not job_focus["key_technologies"]:
            # Look for technology stack section and common tech keywords
            tech_patterns = [
                r"(?:technology stack|tech stack|technologies):\s*([^\n\.]+)",
                r"(?:familiarity|experience|proficiency) with\s+([^\n\.]+)"
            ]
            
            common_techs = [
                "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Ruby", "PHP", "Go", "Rust", "Swift",
                "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL", "Oracle", "Redis", "Cassandra",
                r"React", r"Angular", r"Vue", r"Node\.js", r"Express", r"Django", r"Flask", r"Spring", r"Rails",
                r"AWS", r"Azure", r"GCP", "Docker", "Kubernetes", "Terraform", "Jenkins", "Git", "GitHub", "GitLab",
                "Machine Learning", "AI", "Deep Learning", "NLP", "Computer Vision", "TensorFlow", "PyTorch", "Keras",
                "REST", "GraphQL", "gRPC", "Microservices", "Serverless"
            ]
            
            # Try to extract from explicit technology sections
            for pattern in tech_patterns:
                match = re.search(pattern, raw_text, re.IGNORECASE)
                if match:
                    tech_text = match.group(1).strip()
                    tech_list = [t.strip() for t in re.split(r'[,;/]', tech_text)]
                    job_focus["key_technologies"].extend(tech_list)
            
            # Also try to find common technologies mentioned anywhere
            if not job_focus["key_technologies"]:
                for tech in common_techs:
                    if re.search(r'\b' + tech + r'\b', raw_text, re.IGNORECASE):
                        job_focus["key_technologies"].append(tech.replace("\\", ""))
        
        # Prioritize skills and responsibilities
        if skills:
            # Sort skills by importance (could be inferred from order or frequency)
            sorted_skills = skills
            # Take the top 5-8 skills (or all if fewer)
            job_focus["primary_skills"] = sorted_skills[:min(8, len(sorted_skills))]
        
        if responsibilities:
            # Sort responsibilities by importance
            sorted_responsibilities = responsibilities
            # Take the top 4-5 responsibilities (or all if fewer)
            job_focus["primary_responsibilities"] = sorted_responsibilities[:min(5, len(sorted_responsibilities))]
        
        # Add industry context
        if industry_terms:
            job_focus["industry_context"] = industry_terms[:min(5, len(industry_terms))]
        
        # Add company values
        if company_values:
            job_focus["company_values"] = company_values[:min(3, len(company_values))]
        
        # Add insights from research if available
        if research_results:
            # Add core technical skills from research
            if "job_requirements_analysis" in research_results:
                analysis = research_results["job_requirements_analysis"]
                
                # Add technical skills
                if "core_technical_skills" in analysis and isinstance(analysis["core_technical_skills"], list):
                    for skill in analysis["core_technical_skills"][:5]:  # Add top 5 technical skills
                        if skill not in job_focus["primary_skills"]:
                            job_focus["primary_skills"].append(skill)
                
                # Also look for alternative field names
                technical_skill_keys = [
                    "Core technical skills required", 
                    "technical_skills",
                    "key_skills",
                    "required_skills"
                ]
                
                for key in technical_skill_keys:
                    if key in analysis and isinstance(analysis[key], list) and analysis[key]:
                        for skill in analysis[key][:3]:  # Add top 3 from each alternative field
                            if skill not in job_focus["primary_skills"]:
                                job_focus["primary_skills"].append(skill)
            
                # Add soft skills from research
                soft_skill_keys = [
                    "Soft skills that would be valuable",
                    "soft_skills",
                    "interpersonal_skills"
                ]
                
                for key in soft_skill_keys:
                    if key in analysis and isinstance(analysis[key], list) and analysis[key]:
                        for skill in analysis[key][:2]:  # Add top 2 soft skills from each field
                            if skill not in job_focus["primary_skills"]:
                                job_focus["primary_skills"].append(skill)
                                
                # Add responsibilities if available in research
                responsibility_keys = [
                    "key_responsibilities",
                    "primary_duties",
                    "job_duties",
                    "main_tasks"
                ]
                
                for key in responsibility_keys:
                    if key in analysis and isinstance(analysis[key], list) and analysis[key]:
                        for resp in analysis[key][:3]:  # Add top 3 responsibilities from each field
                            if resp not in job_focus["primary_responsibilities"]:
                                job_focus["primary_responsibilities"].append(resp)
                                
                # Look for industry context and company culture
                if "industry_context" in analysis and isinstance(analysis["industry_context"], (list, str)):
                    if isinstance(analysis["industry_context"], list):
                        for context in analysis["industry_context"][:3]:
                            if context not in job_focus["industry_context"]:
                                job_focus["industry_context"].append(context)
                    else:
                        job_focus["industry_context"].append(analysis["industry_context"])
                
                # Look for company values and culture information
                company_value_keys = [
                    "company_values",
                    "company_culture",
                    "workplace_culture",
                    "cultural_fit"
                ]
                
                for key in company_value_keys:
                    if key in analysis and isinstance(analysis[key], (list, str)):
                        if isinstance(analysis[key], list):
                            for value in analysis[key][:2]:
                                if value not in job_focus["company_values"]:
                                    job_focus["company_values"].append(value)
                        else:
                            job_focus["company_values"].append(analysis[key])
        
        # Remove any duplicates that might have been added from multiple sources
        for key, value in job_focus.items():
            if isinstance(value, list):
                # Keep unique values while preserving order
                unique_values = []
                for v in value:
                    if v and v not in unique_values:
                        unique_values.append(v)
                job_focus[key] = unique_values
        
        return job_focus

    def _generate_key_qualifications(
        self, 
        section: Section, 
        job_description_data: Dict[str, Any], 
        research_results: Dict[str, Any],
        structured_cv: StructuredCV,
        job_focus: Dict[str, Any]
    ):
        """
        Generates content for the Key Qualifications section.
        
        Args:
            section: The Key Qualifications section
            job_description_data: Parsed job description data
            research_results: Research findings
            structured_cv: The full CV structure
            job_focus: Key focus areas from the job
        """
        logger.info("Generating Key Qualifications section")
        update_progress("Generating Key Qualifications...")
        
        # Ensure we have at least 6 key qualifications
        existing_items_count = len([item for item in section.items if item.status != ItemStatus.TO_REGENERATE])
        items_to_generate = max(0, 6 - existing_items_count)
        
        # Add more items if needed
        for _ in range(items_to_generate):
            section.items.append(Item(
                content="",
                status=ItemStatus.TO_REGENERATE,
                item_type=ItemType.KEY_QUAL
            ))
        
        # Generate content for each key qualification, prioritizing job focus
        for i, item in enumerate(section.items):
            # Skip STATIC items
            if item.status == ItemStatus.STATIC:
                print(f"  🔒 Preserving STATIC key qualification #{i+1}")
                continue
                
            if item.status == ItemStatus.INITIAL or item.status == ItemStatus.TO_REGENERATE:
                # Update progress
                update_progress(f"Generating key qualification #{i+1}")
                
                # Generate content for this item
                prompt = self._build_key_qual_prompt_enhanced(
                    job_description_data,
                    research_results,
                    section,
                    item.user_feedback,
                    job_focus,
                    i  # Pass index to help with differentiation
                )
                
                try:
                    # Call the LLM to generate content
                    response = self.llm.generate_content(prompt)
                    
                    # Process and clean the response
                    cleaned_content = self._clean_generated_content(response, item.item_type)
                    
                    # Update item with generated content
                    item.content = cleaned_content
                    item.status = ItemStatus.GENERATED
                    
                except (TimeoutError, ValueError, TypeError) as e:
                    logger.error("Error generating key qualification %d: %s\nFull traceback:\n%s", 
                                 i, str(e), traceback.format_exc())
                    item.content = f"Error generating content: {str(e)}"
                    item.status = ItemStatus.GENERATED  # Mark as generated anyway to avoid repeated errors

    @log_execution_time
    def _generate_experience_content(
        self, 
        section: Section, 
        job_description_data: Dict[str, Any], 
        research_results: Dict[str, Any],
        structured_cv: StructuredCV,
        job_focus: Dict[str, Any]
    ):
        """
        Generates content for the Professional Experience section with better job focus.
        
        Args:
            section: The Professional Experience section
            job_description_data: Parsed job description data
            research_results: Research findings
            structured_cv: The full CV structure
            job_focus: Key focus areas from the job
        """
        logger.info("Generating Professional Experience section")
        update_progress("Generating Professional Experience...")
        
        # Process subsections (experience roles)
        for sub_idx, subsection in enumerate(section.subsections):
            update_progress(f"Processing experience role: {subsection.name}")
            
            # Generate enhanced content for each bullet point
            for i, item in enumerate(subsection.items):
                # Skip STATIC items
                if item.status == ItemStatus.STATIC:
                    print(f"  🔒 Preserving STATIC experience bullet in {subsection.name}")
                    continue
                
                if item.status == ItemStatus.INITIAL or item.status == ItemStatus.TO_REGENERATE:
                    # Update progress
                    update_progress(f"Generating bullet point #{i+1} for {subsection.name}")
                    
                    # Generate content for this item with enhanced job focus
                    prompt = self._build_bullet_point_prompt_enhanced(
                        job_description_data,
                        research_results,
                        section,
                        subsection,
                        item.user_feedback,
                        job_focus
                    )
                    
                    try:
                        # Call the LLM to generate content
                        response = self.llm.generate_content(prompt)
                        
                        # Process and clean the response
                        cleaned_content = self._clean_generated_content(response, item.item_type)
                        
                        # Update item with generated content
                        item.content = cleaned_content
                        item.status = ItemStatus.GENERATED
                        
                    except Exception as e:
                        logger.error(f"Error generating experience bullet for {subsection.name}: {str(e)}\nFull traceback:\n{traceback.format_exc()}")
                        item.content = f"Error generating content: {str(e)}"
                        item.status = ItemStatus.GENERATED  # Mark as generated anyway to avoid repeated errors

    @log_execution_time
    def _generate_summary_content(
        self, 
        section: Section, 
        job_description_data: Dict[str, Any], 
        research_results: Dict[str, Any],
        structured_cv: StructuredCV,
        job_focus: Dict[str, Any]
    ):
        """
        Generates content for the Professional Summary section.
        
        Args:
            section: The Professional Summary section
            job_description_data: Parsed job description data
            research_results: Research findings
            structured_cv: The full CV structure
            job_focus: Key focus areas from the job
        """
        logger.info("Generating Professional Summary section")
        update_progress("Generating Executive Summary...")
        
        # Ensure we have at least one summary item
        if not section.items:
            section.items.append(Item(
                content="",
                status=ItemStatus.TO_REGENERATE,
                item_type=ItemType.SUMMARY_PARAGRAPH
            ))
        
        # Find the generated key qualifications and experience to inform the summary
        key_quals = []
        key_experiences = []
        
        # Get key qualifications
        key_quals_section = structured_cv.get_section_by_name("Key Qualifications") or structured_cv.get_section_by_name("**Key Qualifications:**")
        if key_quals_section:
            for item in key_quals_section.items:
                if item.content and item.status == ItemStatus.GENERATED:
                    key_quals.append(item.content)
        
        # Get key experiences
        exp_section = structured_cv.get_section_by_name("Professional Experience") or structured_cv.get_section_by_name("**Professional experience:**")
        if exp_section:
            for subsection in exp_section.subsections:
                for item in subsection.items:
                    if item.content and item.status == ItemStatus.GENERATED:
                        key_experiences.append(item.content)
        
        # Generate summary content
        for i, item in enumerate(section.items):
            # Skip STATIC items
            if item.status == ItemStatus.STATIC:
                print(f"  🔒 Preserving STATIC summary #{i+1}")
                continue
                
            if item.status == ItemStatus.INITIAL or item.status == ItemStatus.TO_REGENERATE:
                # Update progress
                update_progress("Creating executive summary paragraph")
                
                # Generate summary with awareness of other generated content
                prompt = self._build_summary_prompt_enhanced(
                    job_description_data,
                    research_results,
                    section,
                    item.user_feedback,
                    job_focus,
                    key_quals,
                    key_experiences
                )
                
                try:
                    # Call the LLM to generate content
                    response = self.llm.generate_content(prompt)
                    
                    # Process and clean the response
                    cleaned_content = self._clean_generated_content(response, item.item_type)
                    
                    # Update item with generated content
                    item.content = cleaned_content
                    item.status = ItemStatus.GENERATED
                    
                except Exception as e:
                    logger.error(f"Error generating summary: {str(e)}\nFull traceback:\n{traceback.format_exc()}")
                    item.content = f"Error generating content: {str(e)}"
                    item.status = ItemStatus.GENERATED  # Mark as generated anyway to avoid repeated errors

    @log_execution_time
    def _generate_projects_content(
        self, 
        section: Section, 
        job_description_data: Dict[str, Any], 
        research_results: Dict[str, Any],
        structured_cv: StructuredCV,
        job_focus: Dict[str, Any]
    ):
        """
        Generates content for the Professional Side Projects section.
        
        Args:
            section: The Projects section
            job_description_data: Parsed job description data
            research_results: Research findings
            job_focus: Key focus areas from the job
        """
        # Process each item/subsection independently with delays between calls
        if section.items:
            logger.info(f"Processing {len(section.items)} items in section {section.name} individually")
            print(f"🔄 Processing {len(section.items)} items in {section.name} section...")
            
            # Count how many items need generation
            items_to_generate = [item for item in section.items 
                               if (item.status == ItemStatus.INITIAL or item.status == ItemStatus.TO_REGENERATE)
                               and item.status != ItemStatus.STATIC]  # Skip STATIC items
            
            logger.info(f"Found {len(items_to_generate)} items needing generation in {section.name}")
            
            for i, item in enumerate(items_to_generate):
                try:
                    # Skip STATIC items
                    if item.status == ItemStatus.STATIC:
                        print(f"🔒 Skipping STATIC item {i+1} in section {section.name}")
                        continue
                        
                    # Generate content for this specific item only
                    content = self._generate_item_content(
                        item=item,
                        section=section,
                        subsection=None,
                        job_description_data=job_description_data,
                        research_results=research_results,
                        structured_cv=structured_cv,
                        job_focus=job_focus
                    )
                    
                    # Update item with generated content
                    item.content = content
                    item.status = ItemStatus.GENERATED
                    
                    # Add a longer delay between requests to respect rate limits
                    if i < len(items_to_generate) - 1:
                        logger.info(f"Pausing after item {i+1}/{len(items_to_generate)}")
                        # LLM class now has rate limiting, so we don't need additional delays here
                    
                except Exception as e:
                    logger.error(f"Error in isolated generation for item in {section.name}: {str(e)}")
                    item.content = "Content generation failed. Please try regenerating with a simpler prompt."
                    item.status = ItemStatus.GENERATED
        
        # Process subsections if any
        for subsection_index, subsection in enumerate(section.subsections):
            logger.info(f"Processing subsection {subsection.name} items individually")
            print(f"🔄 Processing {subsection.name} subsection...")
            
            # Count items needing generation in this subsection
            items_to_generate = [item for item in subsection.items 
                               if (item.status == ItemStatus.INITIAL or item.status == ItemStatus.TO_REGENERATE)
                               and item.status != ItemStatus.STATIC]  # Skip STATIC items
            
            logger.info(f"Found {len(items_to_generate)} items needing generation in subsection {subsection.name}")
            
            for i, item in enumerate(items_to_generate):
                try:
                    # Skip STATIC items
                    if item.status == ItemStatus.STATIC:
                        print(f"🔒 Skipping STATIC item {i+1} in subsection {subsection.name}")
                        continue
                        
                    # Generate content for this specific item only
                    content = self._generate_item_content(
                        item=item,
                        section=section,
                        subsection=subsection,
                        job_description_data=job_description_data,
                        research_results=research_results,
                        structured_cv=structured_cv,
                        job_focus=job_focus
                    )
                    
                    # Update item with generated content
                    item.content = content
                    item.status = ItemStatus.GENERATED
                    
                    # LLM class now has rate limiting, so we don't need additional delays here
                    
                except Exception as e:
                    logger.error(f"Error in isolated generation for item in {subsection.name}: {str(e)}")
                    item.content = "Content generation failed. Please try regenerating with a simpler prompt."
                    item.status = ItemStatus.GENERATED
            
            # If we have more subsections to process, add a small wait
            if subsection_index < len(section.subsections) - 1:
                logger.info(f"Finished subsection {subsection_index+1}/{len(section.subsections)}")
                # LLM class now has rate limiting, so we don't need additional delays here

    def _clean_generated_content(self, content: str, item_type: ItemType) -> str:
        """
        Cleans and formats the generated content from the LLM.

        Args:
            content: The raw content from the LLM
            item_type: The type of item being generated

        Returns:
            The cleaned content as a string
        """
        if not content:
            return "Content not available. Please try regenerating."
            
        # Check for error messages that indicate resource exhaustion
        error_indicators = [
            "resourceexhausted", 
            "resource exhausted",
            "context limit",
            "token limit",
            "model encountered an issue"
        ]
        
        for indicator in error_indicators:
            if indicator.lower() in content.lower():
                return "Content generation failed. Please try regenerating with a simpler prompt."
        
        # Remove any code block markers
        content = re.sub(r'```[a-z]*\n', '', content)
        content = re.sub(r'```', '', content)
        
        # Remove any JSON syntax if present
        content = re.sub(r'^\s*\{|\}\s*$', '', content)
        
        # Remove common acknowledgment phrases
        acknowledgments = [
            r'I understand.*', 
            r'Here\'s a tailored.*',
            r'I\'ll generate.*',
            r'Here\'s a bullet point.*',
            r'Here are.*skills.*',
            r'Based on the.*',
            r'New Qualification:.*',
            r'Here\'s a concise.*',
            r'Looking at.*'
        ]
        for ack in acknowledgments:
            content = re.sub(ack, '', content, flags=re.IGNORECASE)
        
        # Clean based on item type
        if item_type == ItemType.BULLET_POINT:
            # Extract bullet points from the content
            bullet_points = re.findall(r'\*\s*(.*?)(?:\n|$)', content)
            if bullet_points:
                # Take only the first bullet point
                return bullet_points[0].strip()
            else:
                # No bullet found, just cleanup whitespace and limit length
                content = content.strip()
                # Truncate if too long
                if len(content) > 150:
                    content = content[:147] + "..."
                return content
            
        elif item_type == ItemType.KEY_QUAL:
            # For key qualifications, extract the first clear qualification
            # First check for a specific bullet point pattern        
            bullet_match = re.search(r'\*\s*([^*\n]+)', content)
            if bullet_match:
                return bullet_match.group(1).strip()
            
            # Check for lines that look like qualifications
            qual_lines = [
                line.strip() for line in content.split('\n') 
                if line.strip() and not line.strip().startswith('*') and len(line.strip()) < 60
            ]
            if qual_lines:
                # Take the first non-empty qualification
                for qual in qual_lines:
                    # Remove leading numbers, asterisks, etc.
                    qual = re.sub(r'^[\d\.\*\-\•]+\s*', '', qual)
                    if qual:
                        return qual.strip()
        
            # Fallback: just return the cleaned content (limited to reasonable length)
            content = content.strip()
            if len(content) > 50:
                content = content[:47] + "..."
            return content
        
        elif item_type == ItemType.SUMMARY_PARAGRAPH:
            # Clean up whitespace but keep paragraph structure
            content = content.strip()
            # Ensure it doesn't have markdown code block markers
            content = re.sub(r'```text', '', content)
            content = re.sub(r'```', '', content)
            # Limit length for exec summary
            if len(content) > 300:
                content = content[:297] + "..."
            return content
        
        # Default: just clean up whitespace and limit length
        content = content.strip()
        if len(content) > 200:
            content = content[:197] + "..."
        return content

    def _generate_item_content(
        self,
        item: Item,
        section: Section,
        subsection: Optional[Subsection],
        job_description_data: Dict[str, Any],
        research_results: Dict[str, Any],
        structured_cv: StructuredCV,
        job_focus: Dict[str, Any]
    ) -> str:
        """
        Generate content for a specific item with awareness of its context.
        
        Args:
            item: The item to generate content for
            section: The section containing the item
            subsection: The subsection containing the item (if any)
            job_description_data: Parsed job description data
            research_results: Research findings
            structured_cv: The full CV structure
            job_focus: Key focus areas extracted from the job
            
        Returns:
            The generated content as a string
        """
        logger.info(f"Generating content for item type: {item.item_type}")
        
        # Skip generation for STATIC items
        if item.status == ItemStatus.STATIC:
            logger.info(f"Skipping generation for STATIC item")
            return item.content
            
        # Build appropriate prompt based on item type and context
        prompt = ""
        
        if item.item_type == ItemType.KEY_QUAL:
            prompt = self._build_key_qual_prompt_enhanced(
                job_description_data,
                research_results,
                section,
                item.user_feedback,
                job_focus,
                0  # Default index
            )
        elif item.item_type == ItemType.BULLET_POINT:
            prompt = self._build_bullet_point_prompt_enhanced(
                job_description_data,
                research_results,
                section,
                subsection,
                item.user_feedback,
                job_focus
            )
        elif item.item_type == ItemType.SUMMARY_PARAGRAPH:
            # Prepare context from other sections for executive summary
            key_quals = []
            key_experiences = []
            
            # Get key qualifications
            key_quals_section = structured_cv.get_section_by_name("Key Qualifications")
            if key_quals_section:
                for kq_item in key_quals_section.items:
                    if kq_item.content and kq_item.status == ItemStatus.GENERATED:
                        key_quals.append(kq_item.content)
            
            # Get key experiences
            exp_section = structured_cv.get_section_by_name("Professional Experience")
            if exp_section:
                for exp_subsection in exp_section.subsections:
                    for exp_item in exp_subsection.items:
                        if exp_item.content and exp_item.status == ItemStatus.GENERATED:
                            key_experiences.append(exp_item.content)
            
            # Now build the summary prompt with this context
            prompt = self._build_summary_prompt_enhanced(
                job_description_data,
                research_results,
                section,
                item.user_feedback,
                job_focus,
                key_quals,
                key_experiences
            )
        else:
            # Generic prompt for other item types
            raw_job_text = ""
            if hasattr(job_description_data, "get") and callable(job_description_data.get):
                raw_job_text = job_description_data.get("raw_text", "")
            elif hasattr(job_description_data, "raw_text"):
                raw_job_text = job_description_data.raw_text
                
            section_name = section.name if section else "Section"
            subsection_name = subsection.name if subsection else ""
            context = f"{section_name}" + (f" - {subsection_name}" if subsection_name else "")
            
            prompt = f"""
            Generate concise, professional CV content for the {context} section that demonstrates
            relevant skills and achievements for this job:
            
            {raw_job_text[:500]}...
            
            Respond with only the content text, no explanations or formatting.
            """
        
        try:
            # Log the prompt for debugging
            logger.info(f"Sending prompt for item generation: {prompt[:100]}...")
            
            # Generate content with LLM
            response = self.llm.generate_content(prompt)
            
            # Clean and return the generated content
            return self._clean_generated_content(response, item.item_type)
            
        except Exception as e:
            logger.error(f"Error generating item content: {str(e)}\nTraceback: {traceback.format_exc()}")
            return "Content generation failed. Please try regenerating with a simpler prompt."

    def _build_key_qual_prompt_enhanced(
        self,
        job_description_data: Dict[str, Any],
        research_results: Dict[str, Any],
        section: Section,
        user_feedback: Optional[str],
        job_focus: Dict[str, Any],
        index: int
    ) -> str:
        """
        Builds an enhanced prompt for generating key qualifications.
        
        Args:
            job_description_data: Parsed job description data
            research_results: Research findings
            section: The Key Qualifications section
            user_feedback: Any user feedback on this item
            job_focus: Key focus areas from the job
            index: Index of the qualification (to ensure variety)
            
        Returns:
            A prompt string for the LLM
        """
        # Extract job description text
        raw_job_text = ""
        if hasattr(job_description_data, "get") and callable(job_description_data.get):
            raw_job_text = job_description_data.get("raw_text", "")
        elif hasattr(job_description_data, "raw_text"):
            raw_job_text = job_description_data.raw_text
            
        # Gather existing qualifications to ensure variety
        existing_quals = []
        for item in section.items:
            if item.content and item.status == ItemStatus.GENERATED:
                existing_quals.append(item.content)
                
        # Convert index to skill type to ensure variety
        skill_types = ["technical", "soft", "domain-specific", "analytical", "leadership", "tool-specific"]
        skill_focus = skill_types[index % len(skill_types)]
        
        # Extract skills from job focus
        primary_skills = job_focus.get("primary_skills", [])
        if not primary_skills and "key_technologies" in job_focus:
            primary_skills = job_focus.get("key_technologies", [])
            
        skills_text = ", ".join(primary_skills[:5]) if primary_skills else ""
        
        # Build the prompt
        prompt = f"""
        Generate a single key qualification for a CV that is highly relevant for this job:
        
        Job Description: {raw_job_text[:300]}...
        
        Key Skills Required: {skills_text}
        
        Focus on a {skill_focus} skill or qualification.
        
        Existing qualifications (avoid duplicating these):
        {', '.join(existing_quals)}
        
        Respond with ONLY the qualification text - be concise (2-5 words).
        Example format: "Advanced SQL Database Design" or "Cross-functional Team Leadership"
        """
        
        # Add user feedback if available
        if user_feedback:
            prompt += f"\n\nConsider this feedback when creating the qualification: {user_feedback}"
            
        return prompt

    def _build_bullet_point_prompt_enhanced(
        self,
        job_description_data: Dict[str, Any],
        research_results: Dict[str, Any],
        section: Section,
        subsection: Optional[Subsection],
        user_feedback: Optional[str],
        job_focus: Dict[str, Any]
    ) -> str:
        """
        Builds an enhanced prompt for generating experience bullet points.
        
        Args:
            job_description_data: Parsed job description data
            research_results: Research findings
            section: The section containing the bullet point
            subsection: The subsection (e.g., job role) containing the bullet point
            user_feedback: Any user feedback on this item
            job_focus: Key focus areas from the job
            
        Returns:
            A prompt string for the LLM
        """
        # Extract job description text
        raw_job_text = ""
        if hasattr(job_description_data, "get") and callable(job_description_data.get):
            raw_job_text = job_description_data.get("raw_text", "")
        elif hasattr(job_description_data, "raw_text"):
            raw_job_text = job_description_data.raw_text
        
        # Get role information from subsection
        role_name = subsection.name if subsection else "Professional Role"
        
        # Gather existing bullet points to ensure variety
        existing_bullets = []
        if subsection:
            for item in subsection.items:
                if item.content and item.status == ItemStatus.GENERATED:
                    existing_bullets.append(item.content)
        
        # Extract skills from job focus
        primary_skills = job_focus.get("primary_skills", [])
        key_responsibilities = job_focus.get("primary_responsibilities", [])
        job_title = job_focus.get("job_title", "")
        
        # Build the prompt with more specific guidance
        prompt = f"""
        Create a single accomplishment-focused bullet point for the role of '{role_name}' in a CV.
        
        Target Job: {job_title if job_title else "The position"}
        
        Key skills needed: {', '.join(primary_skills[:5]) if primary_skills else ""}
        
        Key responsibilities: {', '.join(key_responsibilities[:3]) if key_responsibilities else ""}
        
        Role context: {role_name}
        
        The bullet point should:
        - Start with an action verb in past tense (e.g., "Developed", "Implemented", "Analyzed")
        - Include specific measurable achievements (numbers, percentages, timeframes)
        - Be relevant to the job requirements
        - Be concise (one sentence)
        - Show impact and results
        
        Existing bullet points (do not duplicate):
        {', '.join(existing_bullets)}
        
        Respond with ONLY the bullet point text - no explanations or formatting.
        """
        
        # Add user feedback if available
        if user_feedback:
            prompt += f"\n\nConsider this feedback when creating the bullet point: {user_feedback}"
            
        return prompt

    def _build_summary_prompt_enhanced(
        self,
        job_description_data: Dict[str, Any],
        research_results: Dict[str, Any],
        section: Section,
        user_feedback: Optional[str],
        job_focus: Dict[str, Any],
        key_quals: List[str],
        key_experiences: List[str]
    ) -> str:
        """
        Builds an enhanced prompt for generating executive summary.
        
        Args:
            job_description_data: Parsed job description data
            research_results: Research findings
            section: The summary section
            user_feedback: Any user feedback on this item
            job_focus: Key focus areas from the job
            key_quals: Key qualifications already generated
            key_experiences: Key experiences already generated
            
        Returns:
            A prompt string for the LLM
        """
        # Extract job description text
        raw_job_text = ""
        if hasattr(job_description_data, "get") and callable(job_description_data.get):
            raw_job_text = job_description_data.get("raw_text", "")
        elif hasattr(job_description_data, "raw_text"):
            raw_job_text = job_description_data.raw_text
            
        # Extract key information from job focus
        job_title = job_focus.get("job_title", "")
        primary_skills = job_focus.get("primary_skills", [])
        industry_context = job_focus.get("industry_context", [])
        
        # Build a context-rich prompt
        prompt = f"""
        Create a compelling executive summary for a professional CV targeting this job:
        
        Target Position: {job_title if job_title else "The position"}
        
        Job Context: {raw_job_text[:250]}...
        
        Key skills needed: {', '.join(primary_skills[:5]) if primary_skills else ""}
        
        Industry context: {', '.join(industry_context) if industry_context else ""}
        
        Key qualifications from CV: {', '.join(key_quals[:3]) if key_quals else ""}
        
        Key experiences from CV: {', '.join(key_experiences[:2]) if key_experiences else ""}
        
        The summary should:
        - Be 2-3 sentences and professional in tone
        - Highlight most relevant experience and skills for this specific job
        - Include years of experience if appropriate
        - Position the candidate as a strong match for the role
        - Avoid generic claims and clichés
        
        Respond with ONLY the summary paragraph - no explanations or formatting.
        """
        
        # Add user feedback if available
        if user_feedback:
            prompt += f"\n\nConsider this feedback when creating the summary: {user_feedback}"
            
        return prompt

# Add support for generating content with StructuredCV
def generate_structured_content(self, structured_cv: StructuredCV, job_description_data: Dict[str, Any]) -> StructuredCV:
    """
    Generate content for a StructuredCV object.

        Args:
        structured_cv: The StructuredCV to generate content for
        job_description_data: The parsed job description data

        Returns:
        The updated StructuredCV with generated content
    """
    # First, identify items marked for regeneration
    items_to_regenerate = structured_cv.get_items_by_status(ItemStatus.TO_REGENERATE)
    
    # Extract job details for context
    job_skills = []
    job_responsibilities = []
    if hasattr(job_description_data, "get") and callable(job_description_data.get):
        job_skills = job_description_data.get("skills", [])
        job_responsibilities = job_description_data.get("responsibilities", [])
    elif hasattr(job_description_data, "skills"):
        job_skills = job_description_data.skills
        job_responsibilities = job_description_data.responsibilities
    
    job_context = f"Skills: {', '.join(job_skills)}\nResponsibilities: {'. '.join(job_responsibilities)}"
    
    # Generate content for each item
    for item in items_to_regenerate:
        # Find which section/subsection this item belongs to
        item_obj, section, subsection = structured_cv.find_item_by_id(item.id)
        
        if not item_obj:
            continue  # Skip if item not found
            
        # Generate appropriate content based on item type
        if item_obj.item_type == ItemType.SUMMARY_PARAGRAPH:
            prompt = f"""
            You are an expert CV writer. Write a compelling professional summary paragraph tailored to this job:
            
            {job_context}
            
            Keep it concise (2-3 sentences) and focused on relevant skills and experience.
            """
        elif item_obj.item_type == ItemType.KEY_QUAL:
            prompt = f"""
            You are an expert CV writer. Create a single key qualification (skill or competency) that is highly relevant 
            to this job:
            
            {job_context}
            
            Respond with only the qualification phrase (2-5 words).
            """
        elif item_obj.item_type == ItemType.BULLET_POINT:
            role_context = f"Position: {subsection.name if subsection else 'Professional Role'}" if subsection else ""
            prompt = f"""
            You are an expert CV writer. Create a single accomplishment-focused bullet point for a resume:
            
            {role_context}
            Job Requirements:
            {job_context}
            
            The bullet point should:
            - Start with an action verb
            - Include specific measurable achievements when possible
            - Be relevant to the job requirements
            - Be concise (one sentence)
            
            Respond with ONLY the bullet point text.
            """
        else:
            prompt = f"""
            You are an expert CV writer. Create content for a CV that is relevant to this job:
            
            {job_context}
            
            Keep it concise and focused on relevance to the job requirements.
            """
        
        try:
            # Generate content with LLM
            generated_content = self.llm.generate_content(prompt)
            
            # Clean up the content
            generated_content = generated_content.strip()
            if generated_content.startswith('•') or generated_content.startswith('-'):
                generated_content = generated_content[1:].strip()
                
            # Update the item
            item_obj.content = generated_content
            item_obj.status = ItemStatus.GENERATED

        except Exception as e:
            print(f"Error generating content for item {item_obj.id}: {str(e)}")
            item_obj.content = f"Error: Failed to generate content"
            
    return structured_cv

# Add method to ContentWriterAgent class
ContentWriterAgent.generate_structured_content = generate_structured_content
