#!/usr/bin/env python3

import sys
sys.path.insert(0, 'src')

from src.agents.parser_agent import ParserAgent
from src.services.llm import LLM
from src.orchestration.workflow_definitions import WorkflowBuilder
from src.models.data_models import ContentType

def test_complete_workflow():
    """Test the complete workflow from CV parsing to content writer adaptation."""
    print("=== Testing Complete Workflow ===")
    
    # Initialize components
    parser = ParserAgent('test', 'test', LLM())
    
    # Sample CV with both experience and projects
    cv_text = """# John Doe

## Professional Experience

### Senior Software Engineer at TechCorp Inc.
**Duration:** January 2022 - Present
**Location:** San Francisco, CA
* Led development of microservices architecture serving 1M+ users
* Implemented CI/CD pipelines reducing deployment time by 60%
* Mentored junior developers and conducted code reviews

### Software Engineer at StartupXYZ
**Duration:** June 2020 - December 2021
**Location:** Remote
* Developed full-stack web applications using React and Node.js
* Optimized database queries improving application performance by 40%

## Project Experience

### E-commerce Platform
**Technologies:** React, Node.js, MongoDB
**Duration:** 3 months
* Built responsive web interface with modern UI/UX design
* Implemented secure payment processing system
* Optimized performance achieving 95+ PageSpeed score

### Data Analytics Dashboard
**Technologies:** Python, Flask, D3.js
**Duration:** 2 months
* Created interactive data visualization dashboard
* Integrated with multiple data sources and APIs
"""
    
    # Parse CV
    print("\n1. Parsing CV...")
    structured_cv = parser.parse_cv_text(cv_text, None)
    print(f"   ✅ CV parsed successfully")
    print(f"   - Sections found: {len(structured_cv.sections)}")
    
    # Check sections
    for section in structured_cv.sections:
        print(f"   - {section.name}: {len(section.subsections)} subsections")
        for subsection in section.subsections:
            print(f"     - {subsection.name}: {len(subsection.items)} items")
    
    # Test workflow adaptation
    print("\n2. Testing workflow adaptation...")
    builder = WorkflowBuilder()
    
    # Prepare input data
    input_data = {
        'structured_cv': structured_cv,
        'personal_info': {'name': 'John Doe', 'email': 'john@example.com'},
        'job_description': {'title': 'Senior Developer', 'company': 'TechCorp'}
    }
    
    # Test experience adaptation
    print("\n3. Testing experience adaptation...")
    adapted_experience = builder._adapt_for_content_writer(input_data, ContentType.EXPERIENCE)
    
    if 'content_item' in adapted_experience and 'data' in adapted_experience['content_item']:
        data = adapted_experience['content_item']['data']
        roles = data.get('roles', [])
        projects = data.get('projects', [])
        
        print(f"   ✅ Experience adaptation successful")
        print(f"   - Roles extracted: {len(roles)}")
        print(f"   - Projects extracted: {len(projects)}")
        
        # Show role details
        for i, role in enumerate(roles, 1):
            print(f"   - Role {i}: {role.get('title', 'N/A')} at {role.get('company', 'N/A')}")
            print(f"     Duration: {role.get('duration', 'N/A')}")
            print(f"     Responsibilities: {len(role.get('responsibilities', []))}")
        
        # Show project details
        for i, project in enumerate(projects, 1):
            print(f"   - Project {i}: {project.get('name', 'N/A')}")
            print(f"     Technologies: {project.get('technologies', 'N/A')}")
            print(f"     Duration: {project.get('duration', 'N/A')}")
    else:
        print("   ❌ Experience adaptation failed")
        print(f"   Adapted data structure: {adapted_experience}")
    
    print("\n🎯 Complete workflow test finished!")

if __name__ == "__main__":
    test_complete_workflow()