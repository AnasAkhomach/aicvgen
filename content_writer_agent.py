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

# Set up logging with enhanced format
logger = logging.getLogger(__name__)

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
        section_name = section.name.lower()
        
        if "key qualifications" in section_name or "core competencies" in section_name or "skills" in section_name:
            self._generate_key_qualifications(
                section=section,
                job_description_data=job_description_data,
                research_results=research_results,
                structured_cv=structured_cv,
                job_focus=job_focus
            )
            
        elif "experience" in section_name and "project" not in section_name:
            self._generate_experience_content(
                section=section,
                job_description_data=job_description_data,
                research_results=research_results,
                structured_cv=structured_cv,
                job_focus=job_focus
            )
            
        elif "project" in section_name:
            self._generate_projects_content(
                section=section,
                job_description_data=job_description_data,
                research_results=research_results,
                structured_cv=structured_cv,
                job_focus=job_focus
            )
            
        elif "summary" in section_name or "profile" in section_name:
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
        job_focus = self._extract_job_focus(job_description_data, research_results)
        
        for item_id in item_ids:
            # First check if it's a section ID
            section = structured_cv.find_section_by_id(item_id)
            if section:
                logger.info(f"Regenerating entire section: {section.name}")
                # Use the section-level generation method
                self._generate_section_content(
                    section=section,
                    structured_cv=structured_cv,
                    job_description_data=job_description_data,
                    research_results=research_results,
                    job_focus=job_focus,
                    is_regeneration=True
                )
                continue
                
            # If not a section, try to find individual item
            item, section, subsection = structured_cv.find_item_by_id(item_id)
            if not item:
                logger.warning("Item with ID %s not found in StructuredCV", item_id)
                continue
            
            logger.info(f"Regenerating item in section {section.name}")
            
            # Generate content based on item type and context
            generated_content = self._generate_item_content(
                item,
                section,
                subsection,
                job_description_data,
                research_results,
                structured_cv,
                job_focus
            )
            
            # Update item with generated content
            item.content = generated_content
            item.status = ItemStatus.GENERATED
        
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
        
        # Extract job focus once for efficiency
        job_focus = self._extract_job_focus(job_description_data, research_results)
        
        # Generate content section by section, in a specific order for best results
        logger.info("Generating content for all dynamic sections")
        
        # 1. First generate Key Qualifications
        key_quals_section = structured_cv.get_section_by_name("Key Qualifications")
        if key_quals_section and key_quals_section.content_type == "DYNAMIC":
            logger.info("Generating Key Qualifications section")
            self._generate_section_content(
                section=key_quals_section,
                structured_cv=structured_cv,
                job_description_data=job_description_data,
                research_results=research_results,
                job_focus=job_focus
            )
        
        # 2. Generate Professional Experience
        experience_section = structured_cv.get_section_by_name("Professional Experience")
        if experience_section and experience_section.content_type == "DYNAMIC":
            logger.info("Generating Professional Experience section")
            self._generate_section_content(
                section=experience_section,
                structured_cv=structured_cv,
                job_description_data=job_description_data,
                research_results=research_results,
                job_focus=job_focus
            )
        
        # 3. Generate Project Experience
        projects_section = structured_cv.get_section_by_name("Project Experience")
        if projects_section and projects_section.content_type == "DYNAMIC":
            logger.info("Generating Project Experience section")
            self._generate_section_content(
                section=projects_section,
                structured_cv=structured_cv,
                job_description_data=job_description_data,
                research_results=research_results,
                job_focus=job_focus
            )
        
        # 4. Generate Executive Summary (last, since it depends on other content)
        summary_section = structured_cv.get_section_by_name("Executive Summary")
        if summary_section and summary_section.content_type == "DYNAMIC":
            logger.info("Generating Executive Summary section")
            self._generate_section_content(
                section=summary_section,
                structured_cv=structured_cv,
                job_description_data=job_description_data,
                research_results=research_results,
                job_focus=job_focus
            )
        
        return structured_cv
    
    def _ensure_dynamic_essential_sections(self, structured_cv: StructuredCV):
        """
        Ensures essential sections exist and are marked as DYNAMIC.
        
        Args:
            structured_cv: The CV structure to check and modify
        """
        print("\n>>>>> ENSURING ESSENTIAL SECTIONS ARE DYNAMIC <<<<<")
        
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
            if item.status == ItemStatus.INITIAL or item.status == ItemStatus.TO_REGENERATE:
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
        Builds an enhanced prompt for generating a key qualification with better job focus.
        
        Args:
            job_description_data: Parsed job description data
            research_results: Research findings
            section: The Key Qualifications section
            user_feedback: Optional feedback from the user
            job_focus: Key focus areas from the job
            index: The index of this qualification (to differentiate)
            
        Returns:
            A prompt string for the LLM
        """
        # Get existing qualifications for context
        existing_quals = []
        for item in section.items:
            if item.content and item.status != ItemStatus.TO_REGENERATE:
                existing_quals.append(item.content)
        
        # Create a simple, efficient prompt 
        prompt = "Generate a concise, job-relevant key qualification (skill or expertise).\n\n"
        
        # Focus on specific job requirements
        prompt += "JOB FOCUS:\n"
        if job_focus["primary_skills"] and len(job_focus["primary_skills"]) > 0:
            prompt += f"Primary skills: {', '.join(job_focus['primary_skills'][:5])}\n"
        
        if existing_quals:
            prompt += f"\nExisting qualifications: {', '.join(existing_quals)}\n"
            prompt += "Generate a new qualification that complements these without duplication.\n"
        
        # Add specific instructions based on the index
        focus_prompts = [
            "Create a technical skill that is MOST relevant to this job.",
            "Create a soft skill relevant to the role.",
            "Create a qualification related to the primary job responsibilities.",
            "Create a skill related to the industry context.",
            "Create a skill that differentiates you from other candidates."
        ]
        
        prompt += f"\nINSTRUCTION: {focus_prompts[index % len(focus_prompts)]}\n"
        prompt += "Make it specific, concise (3-5 words), and directly relevant to the job.\n"
        
        # Add feedback if provided (keeping this minimal)
        if user_feedback:
            prompt += f"\nFeedback: \"{user_feedback}\"\n"
        
        return prompt

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
        
        # Process subsections (experience roles)
        for subsection in section.subsections:
            # Generate enhanced content for each bullet point
            for item in subsection.items:
                if item.status == ItemStatus.INITIAL or item.status == ItemStatus.TO_REGENERATE:
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

    def _build_bullet_point_prompt_enhanced(
        self,
        job_description_data: Dict[str, Any],
        research_results: Dict[str, Any],
        section: Section,
        subsection: Subsection,
        user_feedback: Optional[str],
        job_focus: Dict[str, Any]
    ) -> str:
        """
        Builds an enhanced prompt for generating experience bullet points with better job focus.
        
        Args:
            job_description_data: Parsed job description data
            research_results: Research findings
            section: The section containing the bullet point
            subsection: The subsection containing the bullet point
            user_feedback: Optional feedback from the user
            job_focus: Key focus areas from the job
            
        Returns:
            A prompt string for the LLM
        """
        # Get position/role name
        position_name = subsection.name if subsection else "position"
        
        # Get existing bullet points for context
        existing_bullets = []
        for item in subsection.items:
            if item.content and item.status != ItemStatus.TO_REGENERATE:
                existing_bullets.append(item.content)
        
        # Create a simplified, efficient prompt
        prompt = f"Create a single accomplishment-focused bullet point for position: {position_name}\n\n"
        
        # Add job focus (keep it minimal)
        skills = job_focus["primary_skills"][:3] if job_focus["primary_skills"] else []
        responsibilities = job_focus["primary_responsibilities"][:2] if job_focus["primary_responsibilities"] else []
        
        if skills:
            prompt += f"Key skills for this role: {', '.join(skills)}\n"
        if responsibilities:
            prompt += f"Key responsibilities: {', '.join(responsibilities)}\n"
        
        if existing_bullets:
            prompt += "\nExisting bullet points:\n"
            for bullet in existing_bullets[:2]:  # Limit to just 2 examples
                prompt += f"- {bullet}\n"
            prompt += "Create a new bullet point that complements these without duplication.\n"
        
        # Add concise instructions
        prompt += "\nBULLET POINT REQUIREMENTS:\n"
        prompt += "1. Start with an action verb\n"
        prompt += "2. Include specific achievement/impact\n"
        prompt += "3. Mention one skill relevant to the job\n"
        prompt += "4. Keep it concise (<120 characters)\n"
        
        # Add feedback if provided
        if user_feedback:
            prompt += f"\nFeedback: \"{user_feedback}\"\n"
        
        return prompt

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
            structured_cv: The full CV structure
            job_focus: Key focus areas from the job
        """
        logger.info("Generating Professional Side Projects section")
        
        # Process subsections (individual projects)
        for subsection in section.subsections:
            # Generate enhanced content for each bullet point
            for item in subsection.items:
                if item.status == ItemStatus.INITIAL or item.status == ItemStatus.TO_REGENERATE:
                    # Generate content for this item with enhanced job focus
                    prompt = self._build_project_bullet_point_prompt(
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
                        logger.error(f"Error generating project bullet for {subsection.name}: {str(e)}\nFull traceback:\n{traceback.format_exc()}")
                        item.content = f"Error generating content: {str(e)}"
                        item.status = ItemStatus.GENERATED  # Mark as generated anyway to avoid repeated errors

    def _build_project_bullet_point_prompt(
        self,
        job_description_data: Dict[str, Any],
        research_results: Dict[str, Any],
        section: Section,
        subsection: Subsection,
        user_feedback: Optional[str],
        job_focus: Dict[str, Any]
    ) -> str:
        """
        Builds a specialized prompt for generating side project bullet points with job focus.
        
        Args:
            job_description_data: Parsed job description data
            research_results: Research findings
            section: The section containing the bullet point
            subsection: The subsection containing the bullet point
            user_feedback: Optional feedback from the user
            job_focus: Key focus areas from the job
            
        Returns:
            A prompt string for the LLM
        """
        # Get project name and existing bullets
        project_name = subsection.name if subsection else "project"
        
        # Get existing bullet points for context
        existing_bullets = []
        for item in subsection.items:
            if item.content and item.status != ItemStatus.TO_REGENERATE:
                existing_bullets.append(item.content)
        
        # Create a simplified, efficient prompt
        prompt = f"Create a single accomplishment-focused bullet point for project: {project_name}\n\n"
        
        # Add minimal job focus
        skills = job_focus["primary_skills"][:3] if job_focus["primary_skills"] else []
        technologies = job_focus["key_technologies"][:3] if "key_technologies" in job_focus and job_focus["key_technologies"] else []
        
        if skills:
            prompt += f"Target skills: {', '.join(skills)}\n"
        if technologies: 
            prompt += f"Technical focus: {', '.join(technologies)}\n"
        
        if existing_bullets:
            prompt += "\nExisting bullet points:\n"
            for bullet in existing_bullets[:1]:  # Limit to just 1 example
                prompt += f"- {bullet}\n"
            prompt += "Create a complementary bullet point without duplication.\n"
        
        # Add concise instructions
        prompt += "\nPROJECT BULLET POINT REQUIREMENTS:\n"
        prompt += "1. Start with an action verb\n"
        prompt += "2. Highlight a specific technical challenge or achievement\n"
        prompt += "3. Mention tools/technologies relevant to the job\n"
        prompt += "4. Include measurable impact if possible\n"
        prompt += "5. Keep it concise (<120 characters)\n"
        
        # Add feedback if provided
        if user_feedback:
            prompt += f"\nFeedback: \"{user_feedback}\"\n"
        
        return prompt

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
        for item in section.items:
            if item.status == ItemStatus.INITIAL or item.status == ItemStatus.TO_REGENERATE:
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
        Builds an enhanced prompt for generating a professional summary with better job focus.
        
        Args:
            job_description_data: Parsed job description data
            research_results: Research findings
            section: The summary section
            user_feedback: Optional feedback from the user
            job_focus: Key focus areas from the job
            key_quals: Generated key qualifications to reference
            key_experiences: Generated experience bullet points to reference
            
        Returns:
            A prompt string for the LLM
        """
        # Create a simplified, efficient prompt
        prompt = "Create a concise professional summary paragraph for a CV (2-3 sentences).\n\n"
        
        # Add minimal but essential job focus
        job_title = job_focus.get("job_title", "")
        if job_title:
            prompt += f"Target position: {job_title}\n"
            
        # Add top skills (limited to 3)
        top_skills = job_focus["primary_skills"][:3] if job_focus["primary_skills"] else []
        if top_skills:
            prompt += f"Key skills to highlight: {', '.join(top_skills)}\n"
            
        # Add one or two key qualifications
        if key_quals and len(key_quals) > 0:
            qual_sample = key_quals[:2] if len(key_quals) >= 2 else key_quals
            prompt += f"\nKey qualifications: {', '.join(qual_sample)}\n"
            
        # Add instructions
        prompt += "\nSUMMARY REQUIREMENTS:\n"
        prompt += "1. Begin with professional identity and years of experience\n"
        prompt += "2. Highlight 2-3 most relevant skills that match the job\n"
        prompt += "3. Include one key achievement or value proposition\n"
        prompt += "4. Keep it concise but impactful (<250 characters)\n"
        
        # Add feedback if provided
        if user_feedback:
            prompt += f"\nFeedback: \"{user_feedback}\"\n"
        
        return prompt

    @log_execution_time
    def _generate_item_content(
        self,
        item: Item,
        section: Section,
        subsection: Optional[Subsection],
        job_description_data: Dict[str, Any],
        research_results: Dict[str, Any],
        structured_cv: StructuredCV,
        job_focus: Dict[str, Any] = None
    ) -> str:
        """
        Generate content for a specific item.
        
        Args:
            item: The item to generate content for
            section: The section containing the item
            subsection: The subsection containing the item (if applicable)
            job_description_data: Parsed job description data
            research_results: Research findings
            structured_cv: The full structured CV
            job_focus: Extracted job focus information
            
        Returns:
            Generated content as a string
        """
        if not job_focus:
            # Extract job focus if not provided
            job_focus = self._extract_job_focus(job_description_data, research_results)
        
        section_name = section.name.lower() if section else ""
        item_type = item.item_type
        user_feedback = item.user_feedback
        
        # Generate based on section and item type
        if "summary" in section_name or "profile" in section_name:
            # For summary/profile, use the specialized prompt
            job_context = job_description_data.get("raw_text", "") if isinstance(job_description_data, dict) else ""
            research_context = json.dumps(research_results.get("job_requirements_analysis", {}))
            prompt = self._build_summary_prompt(job_context, research_context, user_feedback)
            content = self.llm.generate_content(prompt)
            
            # Special cleanup for summary content
            content = self._clean_generated_content(content, ItemType.SUMMARY_PARAGRAPH)
            
        elif "key qualifications" in section_name or "core competencies" in section_name or "skills" in section_name:
            # For key qualifications, use the specialized prompt
            prompt = self._build_key_qual_prompt_enhanced(
                job_description_data=job_description_data,
                research_results=research_results,
                section=section,
                user_feedback=user_feedback,
                job_focus=job_focus,
                index=1  # Default to index 1 for individual item generation
            )
            content = self.llm.generate_content(prompt)
            
            # Special cleanup for key qualifications
            content = self._clean_generated_content(content, ItemType.KEY_QUAL)
            
        elif "experience" in section_name and "project" not in section_name:
            # For experience bullets, use the specialized prompt
            prompt = self._build_bullet_point_prompt_enhanced(
                job_description_data=job_description_data,
                research_results=research_results,
                section=section,
                subsection=subsection,
                user_feedback=user_feedback,
                job_focus=job_focus
            )
            content = self.llm.generate_content(prompt)
            
            # Special cleanup for experience bullets
            content = self._clean_generated_content(content, ItemType.BULLET_POINT)
            
        elif "project" in section_name:
            # For project bullets, use the specialized prompt
            prompt = self._build_project_bullet_point_prompt(
                job_description_data=job_description_data,
                research_results=research_results,
                section=section,
                subsection=subsection,
                user_feedback=user_feedback,
                job_focus=job_focus
            )
            content = self.llm.generate_content(prompt)
            
            # Special cleanup for project bullets
            content = self._clean_generated_content(content, ItemType.BULLET_POINT)
            
        else:
            # Fallback using the generic prompt
            job_context = job_description_data.get("raw_text", "") if isinstance(job_description_data, dict) else ""
            research_context = json.dumps(research_results)
            prompt = self._build_generic_prompt(
                job_context,
                research_context,
                section,
                subsection,
                item,
                user_feedback
            )
            content = self.llm.generate_content(prompt)
            
        # Update item status to GENERATED
        item.status = ItemStatus.GENERATED
        
        return content
    
    def _build_summary_prompt(
        self, 
        job_context: str, 
        research_context: str, 
        user_feedback: Optional[str]
    ) -> str:
        """
        Builds a prompt for generating an executive summary.

        Args:
            job_context: Context from the job description
            research_context: Context from research
            user_feedback: Optional feedback from the user

        Returns:
            A prompt string for the LLM
        """
        # Try to load the executive summary prompt template
        base_prompt = self.prompt_loader.load_prompt("executive_summary_prompt")
        
        # If no specific feedback is provided, use the template as is
        if not user_feedback:
            return base_prompt
        
        # Otherwise, add the feedback to the prompt
        return base_prompt + f"\n\nNote: The applicant provided the following feedback:\n\"{user_feedback}\"\nPlease take this feedback into account."
    
    def _build_key_qual_prompt(
        self, 
        job_context: str, 
        research_context: str, 
        section: Section, 
        user_feedback: Optional[str]
    ) -> str:
        """
        Builds a prompt for generating a key qualification.

        Args:
            job_context: Context from the job description
            research_context: Context from research
            section: The Key Qualifications section
            user_feedback: Optional feedback from the user

        Returns:
            A prompt string for the LLM
        """
        # Get existing qualifications for context
        existing_quals = []
        for item in section.items:
            if item.content and item.status != ItemStatus.TO_REGENERATE:
                existing_quals.append(item.content)
        
        # Try to load the key qualifications prompt template
        base_prompt = self.prompt_loader.load_prompt("key_qualifications_prompt")
        
        # Add context about existing qualifications
        prompt = base_prompt + f"\n\nExisting qualifications: {', '.join(existing_quals)}\n\nGenerate a new qualification that complements these but doesn't duplicate them."
        
        # Add feedback if provided
        if user_feedback:
            prompt += f"\n\nNote: The applicant provided the following feedback:\n\"{user_feedback}\"\nPlease take this feedback into account."
        
        return prompt
    
    def _build_bullet_point_prompt(
        self, 
        job_context: str, 
        research_context: str, 
        section: Section, 
        subsection: Optional[Subsection], 
        user_feedback: Optional[str]
    ) -> str:
        """
        Builds a prompt for generating a bullet point for experience or projects.

        Args:
            job_context: Context from the job description
            research_context: Context from research
            section: The section containing the bullet point
            subsection: The subsection containing the bullet point
            user_feedback: Optional feedback from the user

        Returns:
            A prompt string for the LLM
        """
        section_type = "experience" if section.name == "Professional Experience" else "project"
        position_name = subsection.name if subsection else "position"
        
        # Get existing bullet points for context
        existing_bullets = []
        if subsection:
            for item in subsection.items:
                if item.content and item.status != ItemStatus.TO_REGENERATE:
                    existing_bullets.append(item.content)
        
        # Try to load the appropriate prompt template
        prompt_name = "resume_role_prompt" if section_type == "experience" else "side_project_prompt"
        base_prompt = self.prompt_loader.load_prompt(prompt_name)
        
        # Add context about position and existing bullets
        prompt = f"{base_prompt}\n\nPosition/Project: {position_name}\n\nExisting bullet points:\n"
        for bullet in existing_bullets:
            prompt += f"- {bullet}\n"
        
        prompt += "\nGenerate a new bullet point that complements these but doesn't duplicate them."
        
        # Add feedback if provided
        if user_feedback:
            prompt += f"\n\nNote: The applicant provided the following feedback:\n\"{user_feedback}\"\nPlease take this feedback into account."
        
        return prompt
    
    def _build_generic_prompt(
        self, 
        job_context: str, 
        research_context: str, 
        section: Section, 
        subsection: Optional[Subsection], 
        item: Item, 
        user_feedback: Optional[str]
    ) -> str:
        """
        Builds a generic prompt for generating content for other item types.

        Args:
            job_context: Context from the job description
            research_context: Context from research
            section: The section containing the item
            subsection: The subsection containing the item (if any)
            item: The item to generate content for
            user_feedback: Optional feedback from the user

        Returns:
            A prompt string for the LLM
        """
        prompt = f"""
        You are an expert CV writer. Create content for the following section:

        Section: {section.name}
        Type: {item.item_type.value}

        Job Context:
        {job_context}

        {research_context}

        Create concise, relevant content appropriate for this CV section.
        """
        
        if user_feedback:
            prompt += f"""
            
            Note: The applicant provided the following feedback:
            "{user_feedback}"
            Please take this feedback into account.
            """
        
        return prompt
    
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
        
    # KEEPING THE FOLLOWING METHODS FOR COMPATIBILITY WITH EXISTING CODE
    
    def generate_batch(self, input_data: Dict[str, Any], batch_type: str) -> ContentData:
        """
        Legacy method for compatibility with existing tests and code.
        """
        structured_cv = self.run({
            "job_description_data": input_data.get("job_description_data", {}),
            "structured_cv": StructuredCV(), 
            "regenerate_item_ids": [],
            "research_results": input_data.get("research_results", {})
        })
        
        
        return structured_cv.to_content_data()

    def _ensure_essential_sections(self, structured_cv: StructuredCV):
        """
        Ensures essential sections exist in the CV.
        
        Args:
            structured_cv: The CV structure to check and modify
        """
        print("\n>>>>> ENSURING ESSENTIAL SECTIONS EXIST <<<<<")
        
        essential_sections = [
            {"name": "Professional profile:", "item_type": ItemType.SUMMARY_PARAGRAPH},
            {"name": "**Key Qualifications:**", "item_type": ItemType.KEY_QUAL},
            {"name": "Professional Side Projects", "item_type": ItemType.BULLET_POINT}
        ]
        
        section_names = [section.name for section in structured_cv.sections]
        
        for essential in essential_sections:
            # Check for alternative section names too
            alt_names = []
            if essential["name"] == "Professional profile:":
                alt_names = ["Executive Summary", "Summary", "About Me"]
            elif essential["name"] == "**Key Qualifications:**":
                alt_names = ["Key Qualifications", "Skills", "Core Competencies"]
            elif essential["name"] == "Professional Side Projects":
                alt_names = ["Side Projects", "Project Experience", "**Project experience:**"]
            
            # Check if this essential section or an alternative exists
            if essential["name"] not in section_names and not any(alt in section_names for alt in alt_names):
                logger.info(f"Adding missing essential section: {essential['name']}")
                
                # Create the missing section
                new_section = Section(
                    name=essential["name"],
                    content_type="DYNAMIC",
                    order=0  # Will be properly ordered later
                )
                
                # Add at least one empty item to the section
                new_section.items.append(Item(
                    content="",
                    status=ItemStatus.TO_REGENERATE,
                    item_type=essential["item_type"]
                ))
                
                # For Key Qualifications, add multiple items
                if essential["name"] == "**Key Qualifications:**":
                    # Add more empty items
                    for _ in range(5):  # Add 5 more for a total of 6 skills
                        new_section.items.append(Item(
                            content="",
                            status=ItemStatus.TO_REGENERATE,
                            item_type=essential["item_type"]
                        ))
                
                # Add the new section to the CV
                structured_cv.sections.append(new_section)

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
