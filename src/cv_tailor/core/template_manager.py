import os
import re
from typing import Dict, List, Any, Tuple

class TemplateManager:
    """
    Manages CV templates, identifying static and dynamic sections for tailoring.
    """
    
    # Define which sections should remain static (not tailored)
    STATIC_SECTIONS = [
        "contact_info",
        "education",
        "certifications", 
        "languages"
    ]
    
    # Define which sections should be dynamic (tailored for each job)
    DYNAMIC_SECTIONS = [
        "executive_summary",
        "key_qualifications",
        "professional_experience",
        "project_experience"
    ]
    
    # Define the preferred section order
    PREFERRED_SECTION_ORDER = [
        "contact_info",
        "executive_summary",
        "key_qualifications",
        "professional_experience",
        "project_experience",
        "education",
        "certifications",
        "languages"
    ]
    
    def __init__(self, template_content: str = None, template_path: str = None):
        """
        Initialize the template manager with either direct content or a file path.
        
        Args:
            template_content: String content of the CV template
            template_path: Path to the CV template file
        """
        self.template_content = template_content
        self.sections = {}
        self.section_order = []
        
        if template_path and os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as file:
                self.template_content = file.read()
        
        if self.template_content:
            self._parse_template()
    
    def _parse_template(self):
        """Parse the template content into sections."""
        if not self.template_content:
            return
        
        lines = self.template_content.split('\n')
        
        # Extract the contact info from the first line
        if lines:
            self.sections["contact_info"] = lines[0]
            self.section_order.append("contact_info")
            
        current_section = None
        section_content = []
        
        # Process the rest of the template
        for i, line in enumerate(lines[1:], 1):
            # Check if this is a section header
            if line.startswith('###'):
                # Save the previous section if there is one
                if current_section:
                    self.sections[current_section] = '\n'.join(section_content)
                    section_content = []
                
                # Extract the new section name
                section_name = line.replace('###', '').strip().lower()
                
                # Normalize section names
                if "profile" in section_name or "summary" in section_name:
                    current_section = "executive_summary"
                elif "qualification" in section_name or "skill" in section_name:
                    current_section = "key_qualifications"
                elif "project" in section_name:
                    current_section = "project_experience"
                elif "experience" in section_name:
                    current_section = "professional_experience"
                elif "education" in section_name:
                    current_section = "education"
                elif "certification" in section_name:
                    current_section = "certifications"
                elif "language" in section_name:
                    current_section = "languages"
                else:
                    current_section = "other_" + section_name.replace(' ', '_')
                
                if current_section not in self.section_order:
                    self.section_order.append(current_section)
                
                # Add the header line
                section_content.append(line)
            elif current_section:
                section_content.append(line)
        
        # Save the last section
        if current_section and section_content:
            self.sections[current_section] = '\n'.join(section_content)
    
    def get_section(self, section_name: str) -> str:
        """Get the content of a specific section."""
        return self.sections.get(section_name, "")
    
    def get_all_static_sections(self) -> Dict[str, str]:
        """Get all static sections that should not be modified."""
        return {name: self.sections.get(name, "") for name in self.STATIC_SECTIONS if name in self.sections}
    
    def get_all_dynamic_sections(self) -> Dict[str, str]:
        """Get all dynamic sections that should be tailored."""
        return {name: self.sections.get(name, "") for name in self.DYNAMIC_SECTIONS if name in self.sections}
    
    def extract_experience_items(self) -> List[Dict[str, Any]]:
        """Extract structured experience items from the professional experience section."""
        experience_section = self.get_section("professional_experience")
        if not experience_section:
            return []
        
        experiences = []
        current_exp = {}
        
        lines = experience_section.split('\n')
        i = 0
        
        # Skip the section header
        while i < len(lines) and not lines[i].strip().startswith('####'):
            i += 1
        
        while i < len(lines):
            line = lines[i].strip()
            
            # New experience entry
            if line.startswith('####'):
                if current_exp and 'position' in current_exp:
                    experiences.append(current_exp)
                current_exp = {'position': line.replace('####', '').strip(), 'bullets': []}
            # Company and location info
            elif '*|' in line or '| *' in line:
                current_exp['company_info'] = line
            # Bullet points
            elif line.startswith('*') and current_exp:
                bullet = line.replace('*', '', 1).strip()
                current_exp['bullets'].append(bullet)
            
            i += 1
        
        # Add the last experience
        if current_exp and 'position' in current_exp:
            experiences.append(current_exp)
        
        return experiences
    
    def extract_project_items(self) -> List[Dict[str, Any]]:
        """Extract structured project items from the project experience section."""
        project_section = self.get_section("project_experience")
        if not project_section:
            return []
        
        projects = []
        current_proj = {}
        
        lines = project_section.split('\n')
        i = 0
        
        # Skip the section header
        while i < len(lines) and not lines[i].strip().startswith('####'):
            i += 1
        
        while i < len(lines):
            line = lines[i].strip()
            
            # New project entry
            if line.startswith('####'):
                if current_proj and 'name' in current_proj:
                    projects.append(current_proj)
                
                # Split the project name and technologies if they exist
                parts = line.replace('####', '').strip().split('|')
                current_proj = {
                    'name': parts[0].strip(),
                    'technologies': parts[1].strip() if len(parts) > 1 else "",
                    'bullets': []
                }
            # Bullet points
            elif line.startswith('*') and current_proj:
                bullet = line.replace('*', '', 1).strip()
                current_proj['bullets'].append(bullet)
            
            i += 1
        
        # Add the last project
        if current_proj and 'name' in current_proj:
            projects.append(current_proj)
        
        return projects
    
    def extract_key_qualifications(self) -> List[str]:
        """Extract key qualifications as a list of skills."""
        qualifications_section = self.get_section("key_qualifications")
        if not qualifications_section:
            return []
        
        # Extract the line with the qualifications (usually formatted with pipe separators)
        lines = qualifications_section.split('\n')
        for line in lines:
            if '|' in line:
                # Split by pipe and clean each skill
                skills = [skill.strip() for skill in line.split('|') if skill.strip()]
                return skills
        
        return []
    
    def extract_executive_summary(self) -> str:
        """Extract the executive summary text."""
        summary_section = self.get_section("executive_summary")
        if not summary_section:
            return ""
        
        # Remove the header and any extra formatting
        lines = summary_section.split('\n')
        summary_text = []
        
        for line in lines:
            # Skip the section header and divider lines
            if line.startswith('###') or line.startswith('---'):
                continue
            summary_text.append(line.strip())
        
        return ' '.join(summary_text).strip()
    
    def rebuild_from_tailored_content(self, tailored_content: Dict[str, Any]) -> str:
        """
        Rebuild the CV using the static sections from the template and 
        the dynamic sections from the tailored content.
        
        Args:
            tailored_content: Dictionary with keys matching section names and 
                              values containing the tailored content
        
        Returns:
            String containing the complete tailored CV
        """
        final_cv = []
        
        # Add sections in the preferred order
        for section in self.PREFERRED_SECTION_ORDER:
            if section in self.STATIC_SECTIONS and section in self.sections:
                # Use original content for static sections
                final_cv.append(self.sections.get(section, ""))
            elif section in tailored_content:
                # Use tailored content for dynamic sections
                final_cv.append(tailored_content.get(section, ""))
            # Special handling for executive_summary which might be under "summary" in tailored_content
            elif section == "executive_summary" and "summary" in tailored_content:
                # Create a formatted executive summary section
                summary_content = tailored_content.get("summary", "")
                if summary_content:
                    final_cv.append("### Executive Summary:\n\n" + summary_content + "\n---")
            elif section in self.sections:
                # Fallback to original if not provided in tailored content
                final_cv.append(self.sections.get(section, ""))
        
        # Add any remaining sections not explicitly ordered
        for section in self.section_order:
            if section not in self.PREFERRED_SECTION_ORDER:
                if section in self.STATIC_SECTIONS:
                    final_cv.append(self.sections.get(section, ""))
                elif section in tailored_content:
                    final_cv.append(tailored_content.get(section, ""))
                else:
                    final_cv.append(self.sections.get(section, ""))
        
        return '\n'.join(final_cv)
    
    def save_template_to_file(self, file_path: str) -> bool:
        """Save the template to a file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(self.template_content)
            return True
        except Exception as e:
            print(f"Error saving template to file: {e}")
            return False 