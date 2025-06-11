from src.agents.agent_base import AgentBase
from src.services.llm import LLM
from src.core.state_manager import AgentIO
from src.models.data_models import (
    JobDescriptionData,
    StructuredCV,
    Section,
    Subsection,
    Item,
    ItemStatus,
    ItemType,
)
from src.config.logging_config import get_logger
from src.config.settings import get_config
from src.services.llm import LLMResponse
import json  # Import json for parsing LLM output
from typing import List, Optional, Dict, Any, Union
import re  # For regex parsing of Markdown
import logging  # For logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ParserAgent(AgentBase):
    """Agent responsible for parsing job descriptions and extracting key information, and parsing CVs into StructuredCV objects."""

    def __init__(self, name: str, description: str, llm: LLM):
        """
        Initialize the ParserAgent.

        Args:
            name: The name of the agent.
            description: A description of the agent's purpose.
            llm: The language model to use for parsing.
        """
        self.name = name
        self.description = description
        self.llm = llm
        
        # Initialize settings for prompt loading
        self.settings = get_config()

    async def run(self, input: dict) -> Dict[str, Any]:
        """
        Main entry point for the agent.

        Args:
            input: A dictionary containing the input data.

        Returns:
            A dictionary containing the parsed job description and/or parsed CV.
        """
        try:
            result = {}

            # Process job description if provided
            if "job_description" in input and input["job_description"]:
                job_data = await self.parse_job_description(input["job_description"])
                
                # Ensure job_data is properly structured as a dictionary for workflow compatibility
                if hasattr(job_data, '__dict__'):
                    # Convert JobDescriptionData object to dictionary
                    job_data_dict = {
                        "raw_text": job_data.raw_text,
                        "skills": job_data.skills,
                        "experience_level": job_data.experience_level,
                        "responsibilities": job_data.responsibilities,
                        "industry_terms": job_data.industry_terms,
                        "company_values": job_data.company_values
                    }
                    result["job_description_data"] = job_data_dict
                    logger.info(f"Job description parsed and converted to structured dictionary format")
                else:
                    # Fallback if job_data is already a dict
                    result["job_description_data"] = job_data
                    logger.info(f"Job description parsed successfully")

            # Process CV if provided
            if "cv_text" in input and input["cv_text"]:
                # Get job data from the result if we just parsed it, or from input
                job_data = result.get("job_description_data", input.get("job_description_data", None))
                structured_cv = self.parse_cv_text(input["cv_text"], job_data)
                result["structured_cv"] = structured_cv

            # Create empty CV structure if user chose "start from scratch"
            if "start_from_scratch" in input and input["start_from_scratch"]:
                # Get job data from the result if we just parsed it, or from input
                job_data = result.get("job_description_data", input.get("job_description_data", None))
                structured_cv = self.create_empty_cv_structure(job_data)
                result["structured_cv"] = structured_cv

            return result
        except Exception as e:
            logger.error(f"Error in ParserAgent.run: {str(e)}")
            # Re-raise the exception to ensure proper error propagation
            raise e
    
    async def run_async(self, input_data: Any, context: 'AgentExecutionContext') -> 'AgentResult':
        """Async run method for consistency with enhanced agent interface."""
        from .agent_base import AgentResult
        from src.models.validation_schemas import validate_agent_input, ValidationError
        
        try:
            # Validate input data using Pydantic schemas
            try:
                validated_input = validate_agent_input('parser', input_data)
                # Convert validated Pydantic model back to dict for processing
                input_data = validated_input.model_dump()
                logger.info("Input validation passed for ParserAgent")
            except ValidationError as ve:
                logger.error(f"Input validation failed for ParserAgent: {ve.message}")
                return AgentResult(
                    success=False,
                    output_data={"error": f"Input validation failed: {ve.message}"},
                    confidence_score=0.0,
                    error_message=f"Input validation failed: {ve.message}",
                    metadata={"agent_type": "parser", "validation_error": True}
                )
            except Exception as e:
                logger.error(f"Input validation error for ParserAgent: {str(e)}")
                return AgentResult(
                    success=False,
                    output_data={"error": f"Input validation error: {str(e)}"},
                    confidence_score=0.0,
                    error_message=f"Input validation error: {str(e)}",
                    metadata={"agent_type": "parser", "validation_error": True}
                )
            
            # Use the existing run method for the actual processing
            result = await self.run(input_data)
            
            return AgentResult(
                success=True,
                output_data=result,
                confidence_score=1.0,
                metadata={"agent_type": "parser"}
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                output_data={},
                confidence_score=0.0,
                error_message=str(e),
                metadata={"agent_type": "parser"}
            )

    async def parse_job_description(self, raw_text: str) -> JobDescriptionData:
        """
        Parses a raw job description using an LLM and extracts key information.

        Args:
            raw_text: The raw job description as a string.

        Returns:
            A JobDescriptionData object with the parsed content.
        """
        if not raw_text:
            logger.warning("Empty job description provided to ParserAgent.")
            # Return a default JobDescriptionData object for empty input
            job_data = JobDescriptionData(
                raw_text=raw_text,
                skills=[],
                experience_level="N/A",
                responsibilities=[],
                industry_terms=[],
                company_values=[],
            )
            self._job_data = job_data
            return job_data

        # Load prompt template from external file
        try:
            prompt_path = self.settings.get_prompt_path("job_description_parsing_prompt")
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
            logger.info("Successfully loaded job description parsing prompt template")
        except Exception as e:
            logger.error(f"Error loading job description parsing prompt template: {e}")
            # Fallback to basic prompt
            prompt_template = """
            Extract key information from the job description: {raw_text}
            Return as JSON with keys: skills, experience_level, responsibilities, industry_terms, company_values
            """
        
        # Format the prompt with actual data
        prompt = prompt_template.format(raw_text=raw_text)

        try:
            # Generate response using the LLM
            response = await self.llm.generate_content(prompt)

            # Try to parse the JSON response
            # Handle potential non-JSON prefix by looking for the first {
            if response and response.content and "{" in response.content:
                json_start = response.content.find("{")
                json_end = response.content.rfind("}") + 1
                json_str = response.content[json_start:json_end]
                parsed_data = json.loads(json_str)

                # Create JobDescriptionData from parsed JSON
                job_data = JobDescriptionData(
                    raw_text=raw_text,
                    skills=parsed_data.get("skills", []),
                    experience_level=parsed_data.get("experience_level", "N/A"),
                    responsibilities=parsed_data.get("responsibilities", []),
                    industry_terms=parsed_data.get("industry_terms", []),
                    company_values=parsed_data.get("company_values", []),
                )
                self._job_data = job_data
                return job_data
            else:
                logger.error("Failed to parse LLM response as JSON: " + (response.content[:200] if response and response.content else "No content") + "...")
                # Return a default object for failed parsing
                job_data = JobDescriptionData(
                    raw_text=raw_text,
                    skills=[],
                    experience_level="N/A",
                    responsibilities=[],
                    industry_terms=[],
                    company_values=[],
                    error="Failed to parse LLM response",
                )
                self._job_data = job_data
                return job_data

        except Exception as e:
            logger.error(f"Error in parse_job_description: {str(e)}")
            # Return a default JobDescriptionData object instead of raising exception
            # to prevent downstream AttributeError in EnhancedContentWriterAgent
            job_data = JobDescriptionData(
                raw_text=raw_text,
                skills=[],
                experience_level="N/A",
                responsibilities=[],
                industry_terms=[],
                company_values=[],
                error=f"Parsing failed: {str(e)}",
            )
            self._job_data = job_data
            return job_data

    def parse_cv_text(self, cv_text: str, job_data: JobDescriptionData) -> StructuredCV:
        """
        Parses CV text into a StructuredCV object.

        Args:
            cv_text: The raw CV text as a string.
            job_data: The parsed job description data.

        Returns:
            A StructuredCV object representing the parsed CV.
        """
        # Create a new StructuredCV
        structured_cv = StructuredCV()

        # Add metadata
        structured_cv.metadata["job_description"] = job_data.to_dict() if job_data else {}
        structured_cv.metadata["original_cv_text"] = cv_text

        # If no CV text, return empty structure
        if not cv_text:
            logger.warning("Empty CV text provided to ParserAgent.")
            return structured_cv

        # Process CV text line by line to extract sections
        lines = cv_text.split("\n")
        current_section = None
        current_subsection = None

        # Regular expressions for Markdown patterns - Made more flexible to capture variations
        section_pattern = re.compile(
            r"^#{1,3}\s+\**([^*#]+)\**$"
        )  # Matches ###, ##, or # headings with optional bold formatting
        subsection_pattern = re.compile(
            r"^#{3,4}\s+\**([^*#]+)\**$"
        )  # Matches #### or ### headings for subsections
        contact_pattern = re.compile(r"\*\*([^*]+)\*\*\s*\|\s*(.+)")  # **Name** | Contact Info
        bullet_pattern = re.compile(r"^\s*[\*\-]\s+(.+)$")  # * Bullet or - Bullet

        # Function to normalize section names by removing formatting and standardizing
        def normalize_section_name(name):
            # Remove any Markdown formatting (**, __, etc.)
            name = re.sub(r"[*_]", "", name).strip()

            # Remove trailing colon if present
            if name.endswith(":"):
                name = name[:-1].strip()

            # Map common variations to standard names
            section_map = {
                "summary": "Executive Summary",
                "profile": "Executive Summary",
                "overview": "Executive Summary",
                "about me": "Executive Summary",
                "professional profile": "Executive Summary",
                "executive summary": "Executive Summary",
                "skills": "Key Qualifications",
                "key skills": "Key Qualifications",
                "skill set": "Key Qualifications",
                "core competencies": "Key Qualifications",
                "technical skills": "Key Qualifications",
                "key qualifications": "Key Qualifications",
                "experience": "Professional Experience",
                "work experience": "Professional Experience",
                "employment history": "Professional Experience",
                "work history": "Professional Experience",
                "professional experience": "Professional Experience",
                "projects": "Project Experience",
                "personal projects": "Project Experience",
                "project experience": "Project Experience",
                "education": "Education",
                "academic background": "Education",
                "certifications": "Certifications",
                "certificates": "Certifications",
                "credentials": "Certifications",
                "languages": "Languages",
                "language proficiency": "Languages",
            }

            # Check for case-insensitive match in our map
            for key, value in section_map.items():
                if key.lower() == name.lower():
                    logger.info(f"Normalized section name: '{name}' -> '{value}'")
                    return value

            # If no match found, return the original name
            return name

        # Extract contact info from first few lines
        for i, line in enumerate(lines[:10]):  # Check only first 10 lines for contact info
            contact_match = contact_pattern.match(line)
            if contact_match:
                name = contact_match.group(1).strip()
                contact_info = contact_match.group(2).strip()
                structured_cv.metadata["name"] = name

                # Extract email, phone, LinkedIn, GitHub from contact info
                email_match = re.search(r"\[([^\]]+@[^\]]+)\]\(mailto:[^\)]+\)", contact_info)
                if email_match:
                    structured_cv.metadata["email"] = email_match.group(1)

                phone_match = re.search(r"📞\s*([^|]+)", contact_info)
                if phone_match:
                    structured_cv.metadata["phone"] = phone_match.group(1).strip()

                linkedin_match = re.search(r"\[LinkedIn\]\(([^\)]+)\)", contact_info)
                if linkedin_match:
                    structured_cv.metadata["linkedin"] = linkedin_match.group(1)

                github_match = re.search(r"\[GitHub\]\(([^\)]+)\)", contact_info)
                if github_match:
                    structured_cv.metadata["github"] = github_match.group(1)

                break

        # Process the rest of the document to extract sections, subsections, and items
        section_order = 0
        for line in lines:
            # Check for section headings
            section_match = section_pattern.match(line)
            if section_match:
                raw_section_name = section_match.group(1).strip()
                section_name = normalize_section_name(raw_section_name)

                # Determine if this is a dynamic section (to be tailored) or static section
                dynamic_sections = [
                    "Executive Summary",
                    "Key Qualifications",
                    "Professional Experience",
                    "Project Experience",
                ]
                content_type = (
                    "DYNAMIC"
                    if any(section_name.lower() == s.lower() for s in dynamic_sections)
                    else "STATIC"
                )

                logger.info(
                    f"Parsed section: '{raw_section_name}' -> '{section_name}' (Type: {content_type})"
                )

                current_section = Section(
                    name=section_name,
                    content_type=content_type,
                    order=section_order,
                    raw_text=line,
                )
                section_order += 1
                structured_cv.sections.append(current_section)
                current_subsection = None
                continue

            # Check for subsection headings
            subsection_match = subsection_pattern.match(line)
            if subsection_match and current_section:
                subsection_name = subsection_match.group(1).strip()
                current_subsection = Subsection(name=subsection_name, raw_text=line)
                current_section.subsections.append(current_subsection)
                continue

            # Check for bullet points
            bullet_match = bullet_pattern.match(line)
            if bullet_match and (current_section or current_subsection):
                content = bullet_match.group(1).strip()

                # Determine item type based on current section
                item_type = ItemType.BULLET_POINT
                if current_section and current_section.name == "Key Qualifications":
                    item_type = ItemType.KEY_QUAL
                elif current_section and current_section.name == "Executive Summary":
                    item_type = ItemType.SUMMARY_PARAGRAPH
                elif current_section and current_section.name == "Education":
                    item_type = ItemType.EDUCATION_ENTRY
                elif current_section and current_section.name == "Certifications":
                    item_type = ItemType.CERTIFICATION_ENTRY
                elif current_section and current_section.name == "Languages":
                    item_type = ItemType.LANGUAGE_ENTRY

                # Determine status (dynamic sections start as INITIAL, static as STATIC)
                status = (
                    ItemStatus.INITIAL
                    if (current_section and current_section.content_type == "DYNAMIC")
                    else ItemStatus.STATIC
                )

                item = Item(content=content, status=status, item_type=item_type)

                # Add to subsection if we're in one, otherwise directly to section
                if current_subsection:
                    current_subsection.items.append(item)
                elif current_section:
                    current_section.items.append(item)

            # Handle non-bullet content for sections like Executive Summary
            elif (
                line.strip()
                and current_section
                and not current_subsection
                and current_section.name == "Executive Summary"
            ):
                # For Executive Summary, we'll consider non-bullet content as summary paragraphs
                if not line.startswith("---") and not line.startswith("#"):
                    item = Item(
                        content=line.strip(),
                        status=(
                            ItemStatus.INITIAL
                            if current_section.content_type == "DYNAMIC"
                            else ItemStatus.STATIC
                        ),
                        item_type=ItemType.SUMMARY_PARAGRAPH,
                    )
                    current_section.items.append(item)

        # Add special handling for Key Qualifications section if it contains a list with | separators
        key_quals_section = next(
            (section for section in structured_cv.sections if section.name == "Key Qualifications"),
            None,
        )
        if key_quals_section and key_quals_section.items:
            # Check if the section contains a single line with | separators
            for i, item in enumerate(key_quals_section.items):
                if "|" in item.content:
                    # Split the content by the | character
                    qual_items = [qual.strip() for qual in item.content.split("|")]
                    # Remove the original item
                    key_quals_section.items.pop(i)
                    # Add each qualification as a separate item
                    for qual in qual_items:
                        if qual:  # Skip empty qualifications
                            key_quals_section.items.append(
                                Item(
                                    content=qual,
                                    status=ItemStatus.INITIAL,
                                    item_type=ItemType.KEY_QUAL,
                                )
                            )

        # Apply LLM-powered parsing for specific sections
        self._enhance_sections_with_llm(structured_cv, cv_text)
        
        return structured_cv

    def _enhance_sections_with_llm(self, structured_cv: StructuredCV, cv_text: str) -> None:
        """
        Enhance specific sections using LLM-powered parsing.
        
        Args:
            structured_cv: The structured CV to enhance
            cv_text: The original CV text for context
        """
        # Find and enhance Professional Experience section
        experience_section = next(
            (section for section in structured_cv.sections if section.name == "Professional Experience"),
            None,
        )
        if experience_section:
            enhanced_experience = self._parse_experience_section_with_llm(cv_text)
            if enhanced_experience:
                # Replace the basic parsed content with LLM-enhanced structure
                experience_section.subsections = enhanced_experience
                logger.info(f"Enhanced Professional Experience section with {len(enhanced_experience)} roles")
        
        # Find and enhance Project Experience section
        project_section = next(
            (section for section in structured_cv.sections if section.name == "Project Experience"),
            None,
        )
        if project_section:
            enhanced_projects = self._parse_projects_section_with_llm(cv_text)
            if enhanced_projects:
                # Replace the basic parsed content with LLM-enhanced structure
                project_section.subsections = enhanced_projects
                logger.info(f"Enhanced Project Experience section with {len(enhanced_projects)} projects")

    def _parse_experience_section_with_llm(self, cv_text: str) -> List[Subsection]:
        """
        Parse the Professional Experience section using LLM to extract structured role information.
        
        Args:
            cv_text: The complete CV text
            
        Returns:
            List of Subsection objects representing individual roles
        """
        try:
            # Extract the experience section from the CV text
            experience_text = self._extract_section_text(cv_text, "Professional Experience")
            if not experience_text:
                logger.warning("No Professional Experience section found in CV text")
                return []
            
            # Create prompt for LLM to parse experience
            prompt = f"""
            Parse the following Professional Experience section and extract structured information for each role.
            
            Experience Text:
            {experience_text}
            
            Please return a JSON array where each object represents a role with the following structure:
            {{
                "title": "Job Title",
                "company": "Company Name",
                "duration": "Start Date - End Date",
                "location": "City, State/Country (if available)",
                "responsibilities": ["responsibility 1", "responsibility 2", "responsibility 3"]
            }}
            
            Extract all roles found in the text. If some information is missing, use reasonable defaults or leave empty strings.
            Return only the JSON array, no additional text.
            """
            
            # Generate response using LLM
            response = self.llm.generate_content(prompt)
            
            if not response:
                logger.error("Empty response from LLM for experience parsing")
                return []
            
            # Parse JSON response
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start == -1 or json_end == 0:
                logger.error("No valid JSON array found in LLM response")
                return []
            
            json_str = response[json_start:json_end]
            parsed_roles = json.loads(json_str)
            
            # Convert to Subsection objects
            subsections = []
            for role in parsed_roles:
                # Create subsection name from title and company
                title = role.get("title", "Position")
                company = role.get("company", "Company")
                duration = role.get("duration", "")
                location = role.get("location", "")
                
                # Format subsection name
                subsection_name = f"{title} at {company}"
                if duration:
                    subsection_name += f" ({duration})"
                if location:
                    subsection_name += f" - {location}"
                
                subsection = Subsection(name=subsection_name)
                
                # Add responsibilities as items
                responsibilities = role.get("responsibilities", [])
                for responsibility in responsibilities:
                    if responsibility.strip():
                        item = Item(
                            content=responsibility.strip(),
                            status=ItemStatus.INITIAL,
                            item_type=ItemType.BULLET_POINT
                        )
                        subsection.items.append(item)
                
                subsections.append(subsection)
            
            logger.info(f"Successfully parsed {len(subsections)} roles from Professional Experience")
            return subsections
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for experience: {e}")
            return []
        except Exception as e:
            logger.error(f"Error in _parse_experience_section_with_llm: {e}")
            return []

    def _parse_projects_section_with_llm(self, cv_text: str) -> List[Subsection]:
        """
        Parse the Project Experience section using LLM to extract structured project information.
        
        Args:
            cv_text: The complete CV text
            
        Returns:
            List of Subsection objects representing individual projects
        """
        try:
            # Extract the projects section from the CV text
            projects_text = self._extract_section_text(cv_text, "Project Experience")
            if not projects_text:
                logger.warning("No Project Experience section found in CV text")
                return []
            
            # Create prompt for LLM to parse projects
            prompt = f"""
            Parse the following Project Experience section and extract structured information for each project.
            
            Projects Text:
            {projects_text}
            
            Please return a JSON array where each object represents a project with the following structure:
            {{
                "name": "Project Name",
                "technologies": "Technologies/Tools used (if available)",
                "duration": "Project duration or timeframe (if available)",
                "description": ["key point 1", "key point 2", "key point 3"]
            }}
            
            Extract all projects found in the text. Focus on technical achievements, technologies used, and impact.
            If some information is missing, use reasonable defaults or leave empty strings.
            Return only the JSON array, no additional text.
            """
            
            # Generate response using LLM
            response = self.llm.generate_content(prompt)
            
            if not response:
                logger.error("Empty response from LLM for projects parsing")
                return []
            
            # Parse JSON response
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start == -1 or json_end == 0:
                logger.error("No valid JSON array found in LLM response")
                return []
            
            json_str = response[json_start:json_end]
            parsed_projects = json.loads(json_str)
            
            # Convert to Subsection objects
            subsections = []
            for project in parsed_projects:
                # Create subsection name from project name and technologies
                name = project.get("name", "Project")
                technologies = project.get("technologies", "")
                duration = project.get("duration", "")
                
                # Format subsection name
                subsection_name = name
                if technologies:
                    subsection_name += f" ({technologies})"
                if duration:
                    subsection_name += f" - {duration}"
                
                subsection = Subsection(name=subsection_name)
                
                # Add description points as items
                description_points = project.get("description", [])
                for point in description_points:
                    if point.strip():
                        item = Item(
                            content=point.strip(),
                            status=ItemStatus.INITIAL,
                            item_type=ItemType.BULLET_POINT
                        )
                        subsection.items.append(item)
                
                subsections.append(subsection)
            
            logger.info(f"Successfully parsed {len(subsections)} projects from Project Experience")
            return subsections
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for projects: {e}")
            return []
        except Exception as e:
            logger.error(f"Error in _parse_projects_section_with_llm: {e}")
            return []

    def _extract_section_text(self, cv_text: str, section_name: str) -> str:
        """
        Extract the text content of a specific section from the CV using LLM.
        
        Args:
            cv_text: The complete CV text
            section_name: The name of the section to extract
            
        Returns:
            The text content of the section, or empty string if not found
        """
        try:
            # Create prompt for LLM to extract the specific section
            prompt = f"""
            Extract the "{section_name}" section from the following CV text.
            
            CV Text:
            {cv_text}
            
            Please return ONLY the content of the "{section_name}" section, without the section header.
            If the section is not found, return "SECTION_NOT_FOUND".
            Do not include any other text or explanations.
            
            The section content should include all subsections, bullet points, and details that belong to "{section_name}".
            """
            
            # Generate response using LLM
            response = self.llm.generate_content(prompt)
            
            if not response or response.strip() == "SECTION_NOT_FOUND":
                logger.warning(f"Section '{section_name}' not found in CV text")
                return ""
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error extracting section '{section_name}' with LLM: {e}")
            return ""

    def create_empty_cv_structure(self, job_data: JobDescriptionData) -> StructuredCV:
        """
        Creates an empty CV structure for the "Start from Scratch" option.

        Args:
            job_data: The parsed job description data.

        Returns:
            A StructuredCV object with empty sections.
        """
        # Create a new StructuredCV
        structured_cv = StructuredCV()

        # Add metadata
        structured_cv.metadata["job_description"] = job_data.to_dict() if job_data else {}
        structured_cv.metadata["start_from_scratch"] = True

        # Create standard CV sections with proper order
        sections = [
            {"name": "Executive Summary", "type": "DYNAMIC", "order": 0},
            {"name": "Key Qualifications", "type": "DYNAMIC", "order": 1},
            {"name": "Professional Experience", "type": "DYNAMIC", "order": 2},
            {"name": "Project Experience", "type": "DYNAMIC", "order": 3},
            {"name": "Education", "type": "STATIC", "order": 4},
            {"name": "Certifications", "type": "STATIC", "order": 5},
            {"name": "Languages", "type": "STATIC", "order": 6},
        ]

        # Create and add the sections
        for section_info in sections:
            section = Section(
                name=section_info["name"],
                content_type=section_info["type"],
                order=section_info["order"],
            )

            # For Executive Summary, add an empty item
            if section.name == "Executive Summary":
                section.items.append(
                    Item(
                        content="",
                        status=ItemStatus.TO_REGENERATE,
                        item_type=ItemType.SUMMARY_PARAGRAPH,
                    )
                )

            # For Key Qualifications, add empty items based on skills from job description
            if section.name == "Key Qualifications" and job_data and job_data.skills:
                # Limit to 8 skills maximum
                for skill in job_data.skills[:8]:
                    section.items.append(
                        Item(
                            content=skill,
                            status=ItemStatus.TO_REGENERATE,
                            item_type=ItemType.KEY_QUAL,
                        )
                    )

            # For Professional Experience, add an empty subsection
            if section.name == "Professional Experience":
                subsection = Subsection(name="Position Title at Company Name")
                # Add some default bullet points
                for _ in range(3):
                    subsection.items.append(
                        Item(
                            content="",
                            status=ItemStatus.TO_REGENERATE,
                            item_type=ItemType.BULLET_POINT,
                        )
                    )
                section.subsections.append(subsection)

            # For Project Experience, add an empty subsection
            if section.name == "Project Experience":
                subsection = Subsection(name="Project Name")
                # Add some default bullet points
                for _ in range(2):
                    subsection.items.append(
                        Item(
                            content="",
                            status=ItemStatus.TO_REGENERATE,
                            item_type=ItemType.BULLET_POINT,
                        )
                    )
                section.subsections.append(subsection)

            structured_cv.sections.append(section)

        return structured_cv

    def get_job_data(self) -> Dict[str, Any]:
        """
        Returns the previously parsed job description data.

        Returns:
            The parsed job description data as a dictionary, or an empty dict if no job has been parsed
        """
        if hasattr(self, "_job_data") and self._job_data:
            return self._job_data
        return {}
