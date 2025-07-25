"""CV Template Loader Service for parsing Markdown templates into structured CV objects.

This service implements the CV-STRUCT-01 work item, providing functionality to parse
Markdown template files and convert them into StructuredCV objects with proper
Section and Subsection hierarchies.
"""

import re
import uuid
from pathlib import Path
from typing import List, Optional

from pydantic import ValidationError

from src.models.cv_models import StructuredCV, Section, Subsection, MetadataModel
from src.services.session_manager import SessionManager


class CVTemplateLoaderService:
    """Stateless service for loading CV templates from Markdown files.
    
    This service parses Markdown files with ## and ### headers to create
    structured CV objects with proper Section and Subsection hierarchies.
    """
    
    # Regex patterns for parsing headers
    SECTION_PATTERN = re.compile(r'^##(?!#)\s*(.*)$', re.MULTILINE)
    SUBSECTION_PATTERN = re.compile(r'^###(?!#)\s*(.*)$', re.MULTILINE)
    
    def load_from_markdown(self, file_path: str) -> StructuredCV:
        """Load and parse a Markdown template file into a StructuredCV object.
        
        Args:
            file_path: Path to the Markdown template file
            
        Returns:
            StructuredCV: Parsed CV structure with sections and subsections
            
        Raises:
            FileNotFoundError: If the template file doesn't exist
            ValueError: If the file is malformed or cannot be parsed
            ValidationError: If the resulting structure fails Pydantic validation
        """
        try:
            # Validate file exists
            template_path = Path(file_path)
            if not template_path.exists():
                raise FileNotFoundError(f"Template file not found: {file_path}")
            
            # Read file content
            try:
                content = template_path.read_text(encoding='utf-8')
            except UnicodeDecodeError as e:
                raise ValueError(f"Failed to read template file as UTF-8: {e}")
            
            if not content.strip():
                raise ValueError("Template file is empty")
            
            # Parse sections and subsections
            sections = self._parse_sections(content)
            
            if not sections:
                raise ValueError("No valid sections found in template file")
            
            # Create StructuredCV object
            cv_data = {
                'id': SessionManager.generate_session_id(),
                'metadata': MetadataModel(
                    created_by='cv_template_loader',
                    source_file=str(template_path),
                    template_version='1.0'
                ),
                'sections': sections
            }
            
            # Validate with Pydantic
            try:
                structured_cv = StructuredCV(**cv_data)
                return structured_cv
            except ValidationError as e:
                raise ValueError(f"Failed to create valid StructuredCV: {e}")
                
        except (FileNotFoundError, ValueError, ValidationError):
            # Re-raise expected exceptions
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            raise ValueError(f"Unexpected error loading template: {e}")
    
    def _parse_sections(self, content: str) -> List[Section]:
        """Parse sections from Markdown content.
        
        Args:
            content: Raw Markdown content
            
        Returns:
            List[Section]: Parsed sections with subsections
        """
        sections = []
        
        # Find all section headers
        section_matches = list(self.SECTION_PATTERN.finditer(content))
        
        for i, section_match in enumerate(section_matches):
            section_name = section_match.group(1).strip()
            section_start = section_match.end()
            
            # Determine section end (next section or end of content)
            if i + 1 < len(section_matches):
                section_end = section_matches[i + 1].start()
            else:
                section_end = len(content)
            
            section_content = content[section_start:section_end]
            
            # Parse subsections within this section
            subsections = self._parse_subsections(section_content)
            
            # Create section object
            section = Section(
                id=SessionManager.generate_session_id(),
                name=section_name,
                content_type='mixed',  # Default content type
                metadata=MetadataModel(
                    created_by='cv_template_loader',
                    section_type='template_section'
                ),
                subsections=subsections,
                items=[]  # Initialize empty items list as per requirements
            )
            
            sections.append(section)
        
        return sections
    
    def _parse_subsections(self, section_content: str) -> List[Subsection]:
        """Parse subsections from section content.
        
        Args:
            section_content: Content within a section
            
        Returns:
            List[Subsection]: Parsed subsections
        """
        subsections = []
        
        # Find all subsection headers within this section
        subsection_matches = list(self.SUBSECTION_PATTERN.finditer(section_content))
        
        for subsection_match in subsection_matches:
            subsection_name = subsection_match.group(1).strip()
            
            # Create subsection object
            subsection = Subsection(
                id=SessionManager.generate_session_id(),
                name=subsection_name,
                metadata=MetadataModel(
                    created_by='cv_template_loader',
                    subsection_type='template_subsection'
                ),
                items=[]  # Initialize empty items list as per requirements
            )
            
            subsections.append(subsection)
        
        return subsections