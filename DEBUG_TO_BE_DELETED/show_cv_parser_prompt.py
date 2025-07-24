#!/usr/bin/env python3
"""
Script to demonstrate the exact prompt passed to the CV parser.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.templates.content_templates import ContentTemplateManager
from src.models.workflow_models import ContentType

def show_cv_parser_prompt():
    """Show the exact prompt that gets passed to the CV parser."""
    
    # Initialize template manager
    template_manager = ContentTemplateManager()
    
    # Get the CV parsing template
    template = template_manager.get_template(
        name="cv_parsing_prompt", 
        content_type=ContentType.CV_PARSING
    )
    
    if not template:
        print("❌ CV parsing template not found!")
        return
    
    print("📋 CV Parser Template Found:")
    print(f"   Name: {template.name}")
    print(f"   Content Type: {template.content_type}")
    print(f"   Description: {template.description}")
    print(f"   Variables: {template.variables}")
    print("\n" + "="*80)
    
    # Example CV text
    example_cv_text = """John Doe
Email: john.doe@email.com
Phone: (555) 123-4567
LinkedIn: linkedin.com/in/johndoe

EXECUTIVE SUMMARY
Experienced software engineer with 5+ years in full-stack development.

PROFESSIONAL EXPERIENCE
Senior Software Engineer @ TechCorp Inc. | 2020 - Present
• Developed scalable web applications using React and Node.js
• Led a team of 3 junior developers
• Improved system performance by 40%

Software Engineer @ StartupXYZ | 2018 - 2020
• Built REST APIs using Python and Django
• Implemented CI/CD pipelines

EDUCATION
B.S. Computer Science, University of Technology | 2018

TECHNICAL SKILLS
Python, JavaScript, React, Node.js, Django, PostgreSQL"""
    
    # Format the template with example CV text
    formatted_prompt = template_manager.format_template(
        template, 
        {"raw_cv_text": example_cv_text}
    )
    
    print("🔍 FORMATTED PROMPT THAT GETS SENT TO LLM:")
    print("="*80)
    print(formatted_prompt)
    print("="*80)
    
    print("\n📝 Key Points:")
    print("• The template includes detailed JSON schema instructions")
    print("• The {raw_cv_text} placeholder gets replaced with actual CV content")
    print("• The LLM receives this formatted prompt to parse the CV into structured JSON")
    print("• System instructions (if provided) would be sent separately as a system message")

if __name__ == "__main__":
    show_cv_parser_prompt()