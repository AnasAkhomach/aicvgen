from agent_base import AgentBase
from llm import LLM
from state_manager import (
    AgentIO,
    JobDescriptionData,
    StructuredCV,
    Section,
    Subsection,
    Item,
    ItemStatus,
    ItemType
)
import json  # Import json for parsing LLM output
from typing import List, Optional, Dict, Any, Union
import re  # For regex parsing of Markdown
import logging  # For logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("debug.log"),
                        logging.StreamHandler()
                    ])
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
    
    def run(self, input: dict) -> Dict[str, Any]:
        """
        Main entry point for the agent.
        
        Args:
            input: A dictionary containing the input data.
            
        Returns:
            A dictionary containing the parsed job description and/or parsed CV.
        """
        result = {}
        
        # Process job description if provided
        if "job_description" in input and input["job_description"]:
            job_data = self.parse_job_description(input["job_description"])
            result["job_description_data"] = job_data
            
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
    
    def parse_job_description(self, raw_text: str) -> JobDescriptionData:
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

        # Update prompt to explicitly request JSON output
        prompt = f"""
        You will extract key information from the job description below:

        -----BEGIN JOB DESCRIPTION-----
        {raw_text}
        -----END JOB DESCRIPTION-----

        Extract and return the following in a structured JSON format:
        1. A list of specific skills or technologies required (e.g. Python, React, AWS).
        2. The experience level required (e.g. Entry-level, Mid-level, Senior, etc.).
        3. A list of key responsibilities or tasks from the job.
        4. A list of industry-specific terms or keywords.
        5. A list of mentions of company values or culture.

        Format your response STRICTLY as a JSON object with these keys: skills, experience_level, responsibilities, industry_terms, company_values. 
        Ensure the output is properly formatted and valid JSON. Do not include anything except the JSON response.
        """

        try:
            # Generate response using the LLM
            response = self.llm.generate_content(prompt)
            
            # Try to parse the JSON response
            # Handle potential non-JSON prefix by looking for the first {
            if response and '{' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
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
                logger.error("Failed to parse LLM response as JSON: " + response[:200] + "...")
                # Return a default object for failed parsing
                job_data = JobDescriptionData(
                    raw_text=raw_text,
                    skills=[],
                    experience_level="N/A",
                    responsibilities=[],
                    industry_terms=[],
                    company_values=[],
                    error="Failed to parse LLM response"
                )
                self._job_data = job_data
                return job_data
                
        except Exception as e:
            logger.error(f"Error in parse_job_description: {str(e)}")
            # Return a default object for exceptions
            job_data = JobDescriptionData(
                raw_text=raw_text,
                skills=[],
                experience_level="N/A",
                responsibilities=[],
                industry_terms=[],
                company_values=[],
                error=f"Error: {str(e)}"
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
        lines = cv_text.split('\n')
        current_section = None
        current_subsection = None
        
        # Regular expressions for Markdown patterns - Made more flexible to capture variations
        section_pattern = re.compile(r'^#{1,3}\s+\**([^*#]+)\**$')  # Matches ###, ##, or # headings with optional bold formatting
        subsection_pattern = re.compile(r'^#{3,4}\s+\**([^*#]+)\**$')  # Matches #### or ### headings for subsections
        contact_pattern = re.compile(r'\*\*([^*]+)\*\*\s*\|\s*(.+)')  # **Name** | Contact Info
        bullet_pattern = re.compile(r'^\s*[\*\-]\s+(.+)$')  # * Bullet or - Bullet
        
        # Function to normalize section names by removing formatting and standardizing
        def normalize_section_name(name):
            # Remove any Markdown formatting (**, __, etc.)
            name = re.sub(r'[*_]', '', name).strip()
            
            # Remove trailing colon if present
            if name.endswith(':'):
                name = name[:-1].strip()
            
            # Map common variations to standard names
            section_map = {
                'summary': 'Executive Summary',
                'profile': 'Executive Summary',
                'overview': 'Executive Summary',
                'about me': 'Executive Summary',
                'professional profile': 'Executive Summary',
                'executive summary': 'Executive Summary',
                
                'skills': 'Key Qualifications',
                'key skills': 'Key Qualifications',
                'skill set': 'Key Qualifications',
                'core competencies': 'Key Qualifications',
                'technical skills': 'Key Qualifications',
                'key qualifications': 'Key Qualifications',
                
                'experience': 'Professional Experience',
                'work experience': 'Professional Experience',
                'employment history': 'Professional Experience',
                'work history': 'Professional Experience',
                'professional experience': 'Professional Experience',
                
                'projects': 'Project Experience',
                'personal projects': 'Project Experience',
                'project experience': 'Project Experience',
                
                'education': 'Education',
                'academic background': 'Education',
                
                'certifications': 'Certifications',
                'certificates': 'Certifications',
                'credentials': 'Certifications',
                
                'languages': 'Languages',
                'language proficiency': 'Languages'
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
                email_match = re.search(r'\[([^\]]+@[^\]]+)\]\(mailto:[^\)]+\)', contact_info)
                if email_match:
                    structured_cv.metadata["email"] = email_match.group(1)
                
                phone_match = re.search(r'📞\s*([^|]+)', contact_info)
                if phone_match:
                    structured_cv.metadata["phone"] = phone_match.group(1).strip()
                
                linkedin_match = re.search(r'\[LinkedIn\]\(([^\)]+)\)', contact_info)
                if linkedin_match:
                    structured_cv.metadata["linkedin"] = linkedin_match.group(1)
                
                github_match = re.search(r'\[GitHub\]\(([^\)]+)\)', contact_info)
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
                dynamic_sections = ["Executive Summary", "Key Qualifications", 
                               "Professional Experience", "Project Experience"]
                content_type = "DYNAMIC" if any(section_name.lower() == s.lower() for s in dynamic_sections) else "STATIC"
                
                logger.info(f"Parsed section: '{raw_section_name}' -> '{section_name}' (Type: {content_type})")
                
                current_section = Section(
                    name=section_name,
                    content_type=content_type,
                    order=section_order,
                    raw_text=line
                )
                section_order += 1
                structured_cv.sections.append(current_section)
                current_subsection = None
                continue
            
            # Check for subsection headings
            subsection_match = subsection_pattern.match(line)
            if subsection_match and current_section:
                subsection_name = subsection_match.group(1).strip()
                current_subsection = Subsection(
                    name=subsection_name,
                    raw_text=line
                )
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
                status = ItemStatus.INITIAL if (current_section and current_section.content_type == "DYNAMIC") else ItemStatus.STATIC
                
                item = Item(
                    content=content,
                    status=status,
                    item_type=item_type
                )
                
                # Add to subsection if we're in one, otherwise directly to section
                if current_subsection:
                    current_subsection.items.append(item)
                elif current_section:
                    current_section.items.append(item)
            
            # Handle non-bullet content for sections like Executive Summary
            elif line.strip() and current_section and not current_subsection and current_section.name == "Executive Summary":
                # For Executive Summary, we'll consider non-bullet content as summary paragraphs
                if not line.startswith('---') and not line.startswith('#'):
                    item = Item(
                        content=line.strip(),
                        status=ItemStatus.INITIAL if current_section.content_type == "DYNAMIC" else ItemStatus.STATIC,
                        item_type=ItemType.SUMMARY_PARAGRAPH
                    )
                    current_section.items.append(item)
        
        # Add special handling for Key Qualifications section if it contains a list with | separators
        key_quals_section = next((section for section in structured_cv.sections if section.name == "Key Qualifications"), None)
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
                            key_quals_section.items.append(Item(
                                content=qual,
                                status=ItemStatus.INITIAL,
                                item_type=ItemType.KEY_QUAL
                            ))
        
        return structured_cv
    
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
            {"name": "Languages", "type": "STATIC", "order": 6}
        ]
        
        # Create and add the sections
        for section_info in sections:
            section = Section(
                name=section_info["name"],
                content_type=section_info["type"],
                order=section_info["order"]
            )
            
            # For Executive Summary, add an empty item
            if section.name == "Executive Summary":
                section.items.append(Item(
                    content="",
                    status=ItemStatus.TO_REGENERATE,
                    item_type=ItemType.SUMMARY_PARAGRAPH
                ))
            
            # For Key Qualifications, add empty items based on skills from job description
            if section.name == "Key Qualifications" and job_data and job_data.skills:
                # Limit to 8 skills maximum
                for skill in job_data.skills[:8]:
                    section.items.append(Item(
                        content=skill,
                        status=ItemStatus.TO_REGENERATE,
                        item_type=ItemType.KEY_QUAL
                    ))
            
            # For Professional Experience, add an empty subsection
            if section.name == "Professional Experience":
                subsection = Subsection(
                    name="Position Title at Company Name"
                )
                # Add some default bullet points
                for _ in range(3):
                    subsection.items.append(Item(
                        content="",
                        status=ItemStatus.TO_REGENERATE,
                        item_type=ItemType.BULLET_POINT
                    ))
                section.subsections.append(subsection)
            
            # For Project Experience, add an empty subsection
            if section.name == "Project Experience":
                subsection = Subsection(
                    name="Project Name"
                )
                # Add some default bullet points
                for _ in range(2):
                    subsection.items.append(Item(
                        content="",
                        status=ItemStatus.TO_REGENERATE,
                        item_type=ItemType.BULLET_POINT
                    ))
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
