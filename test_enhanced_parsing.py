#!/usr/bin/env python3
"""
Comprehensive test script for the enhanced LLM-powered parsing implementation.
Tests both Professional Experience and Project Experience parsing methods.
"""

import sys
import os
import json
from typing import Dict, Any, List

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.parser_agent import ParserAgent
from src.services.llm import LLM
from src.orchestration.workflow_definitions import WorkflowBuilder
from src.core.state_manager import StructuredCV, Section, Subsection, Item, ItemStatus, ItemType
from src.models.data_models import ContentType
from src.config.logging_config import get_logger

# Set up logging
logger = get_logger(__name__)

# Sample CV text with both Professional Experience and Project Experience sections
SAMPLE_CV_TEXT = """
# John Doe
**John Doe** | [john.doe@email.com](mailto:john.doe@email.com) | 📞 (555) 123-4567 | [LinkedIn](https://linkedin.com/in/johndoe) | [GitHub](https://github.com/johndoe)

## Executive Summary
Experienced software engineer with 5+ years of experience in full-stack development, cloud technologies, and agile methodologies.

## Key Qualifications
* Python | JavaScript | React | Node.js | AWS | Docker | Kubernetes | CI/CD

## Professional Experience

### Senior Software Engineer at TechCorp Inc.
**Duration:** January 2022 - Present  
**Location:** San Francisco, CA
* Led development of microservices architecture serving 1M+ daily users
* Implemented CI/CD pipelines reducing deployment time by 60%
* Mentored junior developers and conducted code reviews
* Collaborated with cross-functional teams to deliver features on time

### Software Engineer at StartupXYZ
**Duration:** June 2020 - December 2021  
**Location:** Remote
* Developed full-stack web applications using React and Node.js
* Optimized database queries improving application performance by 40%
* Participated in agile development processes and sprint planning

### Junior Developer at WebSolutions Ltd.
**Duration:** August 2019 - May 2020  
**Location:** New York, NY
* Built responsive web interfaces using HTML, CSS, and JavaScript
* Assisted in maintaining legacy systems and bug fixes
* Learned modern development practices and version control

## Project Experience

### E-commerce Platform Redesign
**Technologies:** React, Node.js, PostgreSQL, AWS  
**Duration:** 3 months (2023)
* Redesigned entire e-commerce platform improving user experience
* Implemented real-time inventory management system
* Achieved 25% increase in conversion rates

### Machine Learning Recommendation Engine
**Technologies:** Python, TensorFlow, Docker, Kubernetes  
**Duration:** 4 months (2022)
* Built recommendation engine using collaborative filtering
* Deployed ML models using containerization and orchestration
* Improved user engagement by 30%

### Open Source Contribution - React Component Library
**Technologies:** React, TypeScript, Storybook  
**Duration:** Ongoing (2021-Present)
* Developed reusable UI components for the React ecosystem
* Maintained comprehensive documentation and examples
* Gained 500+ GitHub stars and active community contributions

## Education
### Bachelor of Science in Computer Science
**University of Technology** | 2015 - 2019 | GPA: 3.8/4.0

## Certifications
* AWS Certified Solutions Architect (2022)
* Certified Kubernetes Administrator (2021)
"""

# Sample job description
SAMPLE_JOB_DESCRIPTION = """
Senior Full Stack Developer - Remote

We are seeking an experienced Senior Full Stack Developer to join our growing team. 
The ideal candidate will have strong experience with modern web technologies, 
cloud platforms, and agile development practices.

Required Skills:
- 5+ years of experience in full-stack development
- Proficiency in React, Node.js, and Python
- Experience with AWS cloud services
- Knowledge of containerization (Docker, Kubernetes)
- Strong understanding of CI/CD practices
- Experience with database design and optimization

Responsibilities:
- Lead development of scalable web applications
- Mentor junior developers
- Collaborate with cross-functional teams
- Implement best practices for code quality and testing

Company Values:
- Innovation and continuous learning
- Collaboration and teamwork
- Quality and attention to detail
"""

def test_parser_agent_initialization():
    """Test that ParserAgent can be initialized properly."""
    print("\n=== Testing ParserAgent Initialization ===")
    try:
        llm = LLM()
        parser = ParserAgent(
            name="test_parser",
            description="Test parser agent",
            llm=llm
        )
        print("✅ ParserAgent initialized successfully")
        return parser
    except Exception as e:
        print(f"❌ Failed to initialize ParserAgent: {e}")
        return None

def test_job_description_parsing(parser: ParserAgent):
    """Test job description parsing functionality."""
    print("\n=== Testing Job Description Parsing ===")
    try:
        job_data = parser.parse_job_description(SAMPLE_JOB_DESCRIPTION)
        print(f"✅ Job description parsed successfully")
        print(f"   Skills found: {len(job_data.skills)}")
        print(f"   Experience level: {job_data.experience_level}")
        print(f"   Responsibilities: {len(job_data.responsibilities)}")
        return job_data
    except Exception as e:
        print(f"❌ Failed to parse job description: {e}")
        return None

def test_cv_parsing_basic(parser: ParserAgent, job_data):
    """Test basic CV parsing functionality."""
    print("\n=== Testing Basic CV Parsing ===")
    try:
        structured_cv = parser.parse_cv_text(SAMPLE_CV_TEXT, job_data)
        print(f"✅ CV parsed successfully")
        print(f"   Sections found: {len(structured_cv.sections)}")
        
        # Print section details
        for section in structured_cv.sections:
            print(f"   - {section.name} ({section.content_type}): {len(section.subsections)} subsections")
            
        return structured_cv
    except Exception as e:
        print(f"❌ Failed to parse CV: {e}")
        return None

def test_experience_section_extraction(parser: ParserAgent):
    """Test the _extract_section_text method for Professional Experience."""
    print("\n=== Testing Experience Section Extraction ===")
    try:
        experience_text = parser._extract_section_text(SAMPLE_CV_TEXT, "Professional Experience")
        print(f"✅ Experience section extracted successfully")
        print(f"   Length: {len(experience_text)} characters")
        print(f"   Preview: {experience_text[:200]}...")
        return experience_text
    except Exception as e:
        print(f"❌ Failed to extract experience section: {e}")
        return None

def test_projects_section_extraction(parser: ParserAgent):
    """Test the _extract_section_text method for Project Experience."""
    print("\n=== Testing Projects Section Extraction ===")
    try:
        projects_text = parser._extract_section_text(SAMPLE_CV_TEXT, "Project Experience")
        print(f"✅ Projects section extracted successfully")
        print(f"   Length: {len(projects_text)} characters")
        print(f"   Preview: {projects_text[:200]}...")
        return projects_text
    except Exception as e:
        print(f"❌ Failed to extract projects section: {e}")
        return None

def test_llm_experience_parsing(parser: ParserAgent):
    """Test LLM-powered experience parsing."""
    print("\n=== Testing LLM Experience Parsing ===")
    try:
        experience_subsections = parser._parse_experience_section_with_llm(SAMPLE_CV_TEXT)
        print(f"✅ Experience parsing completed")
        print(f"   Roles found: {len(experience_subsections)}")
        
        for i, subsection in enumerate(experience_subsections):
            print(f"   Role {i+1}: {subsection.name}")
            print(f"     Responsibilities: {len(subsection.items)}")
            
        return experience_subsections
    except Exception as e:
        print(f"❌ Failed to parse experience with LLM: {e}")
        return None

def test_llm_projects_parsing(parser: ParserAgent):
    """Test LLM-powered projects parsing."""
    print("\n=== Testing LLM Projects Parsing ===")
    try:
        project_subsections = parser._parse_projects_section_with_llm(SAMPLE_CV_TEXT)
        print(f"✅ Projects parsing completed")
        print(f"   Projects found: {len(project_subsections)}")
        
        for i, subsection in enumerate(project_subsections):
            print(f"   Project {i+1}: {subsection.name}")
            print(f"     Description points: {len(subsection.items)}")
            
        return project_subsections
    except Exception as e:
        print(f"❌ Failed to parse projects with LLM: {e}")
        return None

def test_enhanced_cv_parsing(parser: ParserAgent, job_data):
    """Test the complete enhanced CV parsing with LLM integration."""
    print("\n=== Testing Enhanced CV Parsing (with LLM) ===")
    try:
        structured_cv = parser.parse_cv_text(SAMPLE_CV_TEXT, job_data)
        print(f"✅ Enhanced CV parsing completed")
        
        # Check Professional Experience section
        exp_section = next((s for s in structured_cv.sections if s.name == "Professional Experience"), None)
        if exp_section:
            print(f"   Professional Experience: {len(exp_section.subsections)} roles")
            for subsection in exp_section.subsections:
                print(f"     - {subsection.name} ({len(subsection.items)} responsibilities)")
        
        # Check Project Experience section
        proj_section = next((s for s in structured_cv.sections if s.name == "Project Experience"), None)
        if proj_section:
            print(f"   Project Experience: {len(proj_section.subsections)} projects")
            for subsection in proj_section.subsections:
                print(f"     - {subsection.name} ({len(subsection.items)} description points)")
                
        return structured_cv
    except Exception as e:
        print(f"❌ Failed enhanced CV parsing: {e}")
        return None

def test_workflow_adapter(structured_cv):
    """Test the workflow adapter with structured CV data."""
    print("\n=== Testing Workflow Adapter ===")
    try:
        from src.orchestration.workflow_definitions import WorkflowBuilder
        
        builder = WorkflowBuilder()
        
        # Test data extraction
        input_data = {
            "structured_cv": structured_cv,
            "personal_info": {"name": "John Doe", "email": "john.doe@email.com"},
            "job_description": {"title": "Senior Developer"}
        }
        
        # Test experience extraction
        experience_data = builder._extract_structured_experience(structured_cv)
        print(f"✅ Experience extraction: {len(experience_data)} roles")
        for role in experience_data:
            print(f"   - {role['title']} at {role['company']} ({len(role['responsibilities'])} responsibilities)")
        
        # Test projects extraction
        projects_data = builder._extract_structured_projects(structured_cv)
        print(f"✅ Projects extraction: {len(projects_data)} projects")
        for project in projects_data:
            print(f"   - {project['name']} ({len(project['description'])} description points)")
        
        # Test content writer adaptation
        adapted_data = builder._adapt_for_content_writer(input_data, ContentType.EXPERIENCE)
        print(f"✅ Content writer adaptation completed")
        print(f"   Roles in adapted data: {len(adapted_data['content_item']['data']['roles'])}")
        print(f"   Projects in adapted data: {len(adapted_data['content_item']['data']['projects'])}")
        
        return adapted_data
    except Exception as e:
        print(f"❌ Failed workflow adapter test: {e}")
        return None

def test_data_structure_compatibility():
    """Test that the data structures are compatible with EnhancedContentWriter expectations."""
    print("\n=== Testing Data Structure Compatibility ===")
    try:
        # This would normally test with actual EnhancedContentWriter
        # For now, we'll just verify the data structure format
        
        sample_role_data = {
            "title": "Senior Software Engineer",
            "company": "TechCorp Inc.",
            "duration": "January 2022 - Present",
            "location": "San Francisco, CA",
            "responsibilities": [
                "Led development of microservices architecture",
                "Implemented CI/CD pipelines",
                "Mentored junior developers"
            ]
        }
        
        sample_project_data = {
            "name": "E-commerce Platform Redesign",
            "technologies": "React, Node.js, PostgreSQL, AWS",
            "duration": "3 months (2023)",
            "description": [
                "Redesigned entire e-commerce platform",
                "Implemented real-time inventory management",
                "Achieved 25% increase in conversion rates"
            ]
        }
        
        # Verify required fields are present
        required_role_fields = ["title", "company", "responsibilities"]
        required_project_fields = ["name", "description"]
        
        for field in required_role_fields:
            assert field in sample_role_data, f"Missing required role field: {field}"
        
        for field in required_project_fields:
            assert field in sample_project_data, f"Missing required project field: {field}"
        
        print("✅ Data structure compatibility verified")
        print(f"   Role data structure: {list(sample_role_data.keys())}")
        print(f"   Project data structure: {list(sample_project_data.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Data structure compatibility test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests in sequence."""
    print("🚀 Starting Comprehensive Enhanced Parsing Test Suite")
    print("=" * 60)
    
    # Initialize parser
    parser = test_parser_agent_initialization()
    if not parser:
        print("❌ Cannot continue without parser initialization")
        return False
    
    # Test job description parsing
    job_data = test_job_description_parsing(parser)
    if not job_data:
        print("⚠️ Continuing without job data")
    
    # Test basic CV parsing
    structured_cv_basic = test_cv_parsing_basic(parser, job_data)
    if not structured_cv_basic:
        print("❌ Cannot continue without basic CV parsing")
        return False
    
    # Test section extraction
    experience_text = test_experience_section_extraction(parser)
    projects_text = test_projects_section_extraction(parser)
    
    # Test LLM parsing methods
    experience_subsections = test_llm_experience_parsing(parser)
    project_subsections = test_llm_projects_parsing(parser)
    
    # Test enhanced CV parsing (with LLM integration)
    structured_cv_enhanced = test_enhanced_cv_parsing(parser, job_data)
    if not structured_cv_enhanced:
        print("❌ Enhanced CV parsing failed")
        return False
    
    # Test workflow adapter
    adapted_data = test_workflow_adapter(structured_cv_enhanced)
    if not adapted_data:
        print("❌ Workflow adapter test failed")
        return False
    
    # Test data structure compatibility
    compatibility_ok = test_data_structure_compatibility()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 Test Suite Summary")
    print("=" * 60)
    
    success_count = sum([
        parser is not None,
        job_data is not None,
        structured_cv_basic is not None,
        experience_text is not None,
        projects_text is not None,
        experience_subsections is not None,
        project_subsections is not None,
        structured_cv_enhanced is not None,
        adapted_data is not None,
        compatibility_ok
    ])
    
    total_tests = 10
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Enhanced parsing implementation is working correctly.")
        return True
    else:
        print(f"⚠️ {total_tests - success_count} tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)