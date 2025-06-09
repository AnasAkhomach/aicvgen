"""Unit tests for Enhanced Parser Agent.

Tests CV parsing from Markdown, job description extraction,
and content structure validation.
"""

import unittest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import tempfile
import os
import sys
from typing import Dict, Any, List

# Add the project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.parser_agent import ParserAgent
from src.models.data_models import (
    ContentType, ProcessingStatus, ExperienceItem, ProjectItem,
    QualificationItem, JobDescriptionData, CVData
)
from src.services.llm import LLMResponse


class TestEnhancedParserAgent(unittest.TestCase):
    """Test cases for Enhanced Parser Agent."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm_service = Mock()
        self.mock_config = Mock()
        self.mock_logger = Mock()
        
        # Sample CV markdown content for testing
        self.sample_cv_markdown = """
# John Doe
## Software Engineer

### Contact Information
- Email: john.doe@email.com
- Phone: (555) 123-4567
- LinkedIn: linkedin.com/in/johndoe

### Professional Summary
Experienced software engineer with 5+ years in full-stack development.

### Work Experience

#### Senior Software Engineer | TechCorp Inc. | 2021-2023
- Developed scalable web applications using React and Node.js
- Led a team of 4 developers on microservices architecture
- Improved system performance by 40% through optimization

#### Software Engineer | StartupXYZ | 2019-2021
- Built REST APIs using Python and Django
- Implemented CI/CD pipelines with Jenkins
- Collaborated with cross-functional teams

### Education

#### Bachelor of Science in Computer Science | University ABC | 2015-2019
- GPA: 3.8/4.0
- Relevant Coursework: Data Structures, Algorithms, Software Engineering

### Projects

#### E-commerce Platform | 2022
- Built full-stack e-commerce application
- Technologies: React, Node.js, MongoDB
- Features: Payment integration, inventory management

#### Task Management App | 2021
- Developed mobile app using React Native
- Implemented real-time synchronization
- Published on App Store and Google Play

### Skills
- Programming Languages: Python, JavaScript, Java, C++
- Frameworks: React, Node.js, Django, Express
- Databases: MongoDB, PostgreSQL, MySQL
- Tools: Git, Docker, Jenkins, AWS
"""
        
        # Sample job description for testing
        self.sample_job_description = """
Senior Full-Stack Developer

Company: InnovateTech Solutions
Location: San Francisco, CA
Type: Full-time

Job Description:
We are seeking a Senior Full-Stack Developer to join our dynamic team.
The ideal candidate will have extensive experience in modern web technologies
and a passion for building scalable applications.

Responsibilities:
- Design and develop web applications using React and Node.js
- Collaborate with product managers and designers
- Mentor junior developers
- Implement best practices for code quality and testing
- Work with cloud platforms (AWS, Azure)

Requirements:
- 5+ years of experience in full-stack development
- Proficiency in JavaScript, Python, or Java
- Experience with React, Angular, or Vue.js
- Knowledge of databases (SQL and NoSQL)
- Familiarity with DevOps practices
- Strong problem-solving skills
- Excellent communication abilities

Preferred Qualifications:
- Experience with microservices architecture
- Knowledge of containerization (Docker, Kubernetes)
- Previous leadership experience
- Bachelor's degree in Computer Science or related field

Benefits:
- Competitive salary ($120,000 - $150,000)
- Health insurance
- 401(k) matching
- Flexible work arrangements
- Professional development opportunities
"""

    @patch('src.agents.parser_agent.get_structured_logger')
    @patch('src.agents.parser_agent.get_config')
    @patch('src.agents.parser_agent.get_llm_service')
    def test_parser_agent_initialization(self, mock_llm_service, mock_config, mock_logger):
        """Test Parser Agent initialization."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create parser agent
        parser_agent = ParserAgent()
        
        # Verify initialization
        self.assertIsNotNone(parser_agent.llm_service)
        self.assertIsNotNone(parser_agent.settings)
        self.assertIsNotNone(parser_agent.logger)
        self.assertEqual(parser_agent.name, "ParserAgent")
        self.assertIn("parsing", parser_agent.description.lower())

    @patch('src.agents.parser_agent.get_structured_logger')
    @patch('src.agents.parser_agent.get_config')
    @patch('src.agents.parser_agent.get_llm_service')
    def test_cv_parsing_from_markdown(self, mock_llm_service, mock_config, mock_logger):
        """Test CV parsing from Markdown content."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Mock LLM response for CV parsing
        mock_llm_response = LLMResponse(
            content="""{
                "personal_info": {
                    "name": "John Doe",
                    "title": "Software Engineer",
                    "email": "john.doe@email.com",
                    "phone": "(555) 123-4567",
                    "linkedin": "linkedin.com/in/johndoe"
                },
                "summary": "Experienced software engineer with 5+ years in full-stack development.",
                "experiences": [
                    {
                        "title": "Senior Software Engineer",
                        "company": "TechCorp Inc.",
                        "duration": "2021-2023",
                        "responsibilities": [
                            "Developed scalable web applications using React and Node.js",
                            "Led a team of 4 developers on microservices architecture",
                            "Improved system performance by 40% through optimization"
                        ]
                    }
                ],
                "education": [
                    {
                        "degree": "Bachelor of Science in Computer Science",
                        "institution": "University ABC",
                        "duration": "2015-2019",
                        "gpa": "3.8/4.0"
                    }
                ],
                "projects": [
                    {
                        "name": "E-commerce Platform",
                        "year": "2022",
                        "description": "Built full-stack e-commerce application",
                        "technologies": ["React", "Node.js", "MongoDB"]
                    }
                ],
                "skills": {
                    "programming_languages": ["Python", "JavaScript", "Java", "C++"],
                    "frameworks": ["React", "Node.js", "Django", "Express"],
                    "databases": ["MongoDB", "PostgreSQL", "MySQL"],
                    "tools": ["Git", "Docker", "Jenkins", "AWS"]
                }
            }""",
            tokens_used=500,
            processing_time=2.5,
            model_used="groq",
            success=True
        )
        
        self.mock_llm_service.generate_content = AsyncMock(return_value=mock_llm_response)
        
        # Create parser agent
        parser_agent = ParserAgent()
        
        # Test CV parsing
        async def test_parsing():
            result = await parser_agent.parse_cv_from_markdown(self.sample_cv_markdown)
            
            # Verify parsing result
            self.assertIsInstance(result, dict)
            self.assertTrue(result.get('success', False))
            
            parsed_data = result.get('data')
            self.assertIsNotNone(parsed_data)
            
            # Verify personal info extraction
            personal_info = parsed_data.get('personal_info', {})
            self.assertEqual(personal_info.get('name'), "John Doe")
            self.assertEqual(personal_info.get('title'), "Software Engineer")
            self.assertEqual(personal_info.get('email'), "john.doe@email.com")
            
            # Verify experiences extraction
            experiences = parsed_data.get('experiences', [])
            self.assertGreater(len(experiences), 0)
            
            first_experience = experiences[0]
            self.assertEqual(first_experience.get('title'), "Senior Software Engineer")
            self.assertEqual(first_experience.get('company'), "TechCorp Inc.")
            
            # Verify skills extraction
            skills = parsed_data.get('skills', {})
            self.assertIn('programming_languages', skills)
            self.assertIn('Python', skills.get('programming_languages', []))
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_parsing())
        finally:
            loop.close()

    @patch('src.agents.parser_agent.get_structured_logger')
    @patch('src.agents.parser_agent.get_config')
    @patch('src.agents.parser_agent.get_llm_service')
    def test_job_description_extraction(self, mock_llm_service, mock_config, mock_logger):
        """Test job description extraction and parsing."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Mock LLM response for job description parsing
        mock_llm_response = LLMResponse(
            content="""{
                "job_title": "Senior Full-Stack Developer",
                "company": "InnovateTech Solutions",
                "location": "San Francisco, CA",
                "employment_type": "Full-time",
                "salary_range": "$120,000 - $150,000",
                "required_skills": [
                    "JavaScript", "Python", "Java", "React", "Angular", "Vue.js",
                    "SQL", "NoSQL", "DevOps", "Problem-solving", "Communication"
                ],
                "preferred_skills": [
                    "Microservices architecture", "Docker", "Kubernetes", "Leadership"
                ],
                "experience_required": "5+ years",
                "education_required": "Bachelor's degree in Computer Science or related field",
                "responsibilities": [
                    "Design and develop web applications using React and Node.js",
                    "Collaborate with product managers and designers",
                    "Mentor junior developers",
                    "Implement best practices for code quality and testing",
                    "Work with cloud platforms (AWS, Azure)"
                ],
                "benefits": [
                    "Competitive salary", "Health insurance", "401(k) matching",
                    "Flexible work arrangements", "Professional development opportunities"
                ]
            }""",
            tokens_used=400,
            processing_time=2.0,
            model_used="groq",
            success=True
        )
        
        self.mock_llm_service.generate_content = AsyncMock(return_value=mock_llm_response)
        
        # Create parser agent
        parser_agent = ParserAgent()
        
        # Test job description parsing
        async def test_job_parsing():
            result = await parser_agent.parse_job_description(self.sample_job_description)
            
            # Verify parsing result
            self.assertIsInstance(result, dict)
            self.assertTrue(result.get('success', False))
            
            job_data = result.get('data')
            self.assertIsNotNone(job_data)
            
            # Verify job details extraction
            self.assertEqual(job_data.get('job_title'), "Senior Full-Stack Developer")
            self.assertEqual(job_data.get('company'), "InnovateTech Solutions")
            self.assertEqual(job_data.get('location'), "San Francisco, CA")
            self.assertEqual(job_data.get('employment_type'), "Full-time")
            
            # Verify skills extraction
            required_skills = job_data.get('required_skills', [])
            self.assertIn('JavaScript', required_skills)
            self.assertIn('Python', required_skills)
            self.assertIn('React', required_skills)
            
            preferred_skills = job_data.get('preferred_skills', [])
            self.assertIn('Microservices architecture', preferred_skills)
            self.assertIn('Docker', preferred_skills)
            
            # Verify responsibilities extraction
            responsibilities = job_data.get('responsibilities', [])
            self.assertGreater(len(responsibilities), 0)
            self.assertTrue(any('React' in resp for resp in responsibilities))
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_job_parsing())
        finally:
            loop.close()

    @patch('src.agents.parser_agent.get_structured_logger')
    @patch('src.agents.parser_agent.get_config')
    @patch('src.agents.parser_agent.get_llm_service')
    def test_content_structure_validation(self, mock_llm_service, mock_config, mock_logger):
        """Test content structure validation."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create parser agent
        parser_agent = ParserAgent()
        
        # Test valid CV structure
        valid_cv_data = {
            'personal_info': {
                'name': 'John Doe',
                'email': 'john@email.com'
            },
            'experiences': [
                {
                    'title': 'Software Engineer',
                    'company': 'TechCorp',
                    'duration': '2021-2023'
                }
            ],
            'skills': {
                'programming_languages': ['Python', 'JavaScript']
            }
        }
        
        # Test validation
        is_valid = parser_agent.validate_cv_structure(valid_cv_data)
        self.assertTrue(is_valid)
        
        # Test invalid CV structure (missing required fields)
        invalid_cv_data = {
            'personal_info': {
                'name': 'John Doe'
                # Missing email
            },
            # Missing experiences and skills
        }
        
        is_invalid = parser_agent.validate_cv_structure(invalid_cv_data)
        self.assertFalse(is_invalid)

    @patch('src.agents.parser_agent.get_structured_logger')
    @patch('src.agents.parser_agent.get_config')
    @patch('src.agents.parser_agent.get_llm_service')
    def test_experience_item_extraction(self, mock_llm_service, mock_config, mock_logger):
        """Test extraction of individual experience items."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create parser agent
        parser_agent = ParserAgent()
        
        # Sample experience text
        experience_text = """
        Senior Software Engineer | TechCorp Inc. | 2021-2023
        - Developed scalable web applications using React and Node.js
        - Led a team of 4 developers on microservices architecture
        - Improved system performance by 40% through optimization
        """
        
        # Test experience extraction
        experience_item = parser_agent.extract_experience_item(experience_text)
        
        # Verify extraction
        self.assertIsInstance(experience_item, ExperienceItem)
        self.assertEqual(experience_item.title, "Senior Software Engineer")
        self.assertEqual(experience_item.company, "TechCorp Inc.")
        self.assertEqual(experience_item.duration, "2021-2023")
        self.assertGreater(len(experience_item.responsibilities), 0)
        self.assertIn("React", experience_item.responsibilities[0])

    @patch('src.agents.parser_agent.get_structured_logger')
    @patch('src.agents.parser_agent.get_config')
    @patch('src.agents.parser_agent.get_llm_service')
    def test_project_item_extraction(self, mock_llm_service, mock_config, mock_logger):
        """Test extraction of individual project items."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create parser agent
        parser_agent = ParserAgent()
        
        # Sample project text
        project_text = """
        E-commerce Platform | 2022
        - Built full-stack e-commerce application
        - Technologies: React, Node.js, MongoDB
        - Features: Payment integration, inventory management
        """
        
        # Test project extraction
        project_item = parser_agent.extract_project_item(project_text)
        
        # Verify extraction
        self.assertIsInstance(project_item, ProjectItem)
        self.assertEqual(project_item.name, "E-commerce Platform")
        self.assertEqual(project_item.year, "2022")
        self.assertIn("React", project_item.technologies)
        self.assertIn("Node.js", project_item.technologies)
        self.assertIn("MongoDB", project_item.technologies)

    @patch('src.agents.parser_agent.get_structured_logger')
    @patch('src.agents.parser_agent.get_config')
    @patch('src.agents.parser_agent.get_llm_service')
    def test_qualification_item_extraction(self, mock_llm_service, mock_config, mock_logger):
        """Test extraction of qualification/education items."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create parser agent
        parser_agent = ParserAgent()
        
        # Sample qualification text
        qualification_text = """
        Bachelor of Science in Computer Science | University ABC | 2015-2019
        - GPA: 3.8/4.0
        - Relevant Coursework: Data Structures, Algorithms, Software Engineering
        """
        
        # Test qualification extraction
        qualification_item = parser_agent.extract_qualification_item(qualification_text)
        
        # Verify extraction
        self.assertIsInstance(qualification_item, QualificationItem)
        self.assertEqual(qualification_item.degree, "Bachelor of Science in Computer Science")
        self.assertEqual(qualification_item.institution, "University ABC")
        self.assertEqual(qualification_item.duration, "2015-2019")
        self.assertIn("3.8/4.0", qualification_item.details)

    @patch('src.agents.parser_agent.get_structured_logger')
    @patch('src.agents.parser_agent.get_config')
    @patch('src.agents.parser_agent.get_llm_service')
    def test_error_handling_in_parsing(self, mock_llm_service, mock_config, mock_logger):
        """Test error handling in parsing operations."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Mock LLM service to raise an exception
        self.mock_llm_service.generate_content = AsyncMock(
            side_effect=Exception("LLM service error")
        )
        
        # Create parser agent
        parser_agent = ParserAgent()
        
        # Test error handling in CV parsing
        async def test_error_handling():
            result = await parser_agent.parse_cv_from_markdown(self.sample_cv_markdown)
            
            # Verify error handling
            self.assertIsInstance(result, dict)
            self.assertFalse(result.get('success', True))
            self.assertIsNotNone(result.get('error_message'))
            self.assertIn("error", result.get('error_message', '').lower())
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_error_handling())
        finally:
            loop.close()

    @patch('src.agents.parser_agent.get_structured_logger')
    @patch('src.agents.parser_agent.get_config')
    @patch('src.agents.parser_agent.get_llm_service')
    def test_malformed_content_handling(self, mock_llm_service, mock_config, mock_logger):
        """Test handling of malformed or incomplete content."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create parser agent
        parser_agent = ParserAgent()
        
        # Test with empty content
        empty_result = parser_agent.validate_cv_structure({})
        self.assertFalse(empty_result)
        
        # Test with malformed content
        malformed_content = "This is not a proper CV format"
        
        # Mock LLM response with invalid JSON
        mock_llm_response = LLMResponse(
            content="Invalid JSON content {malformed",
            tokens_used=50,
            processing_time=1.0,
            model_used="groq",
            success=True
        )
        
        self.mock_llm_service.generate_content = AsyncMock(return_value=mock_llm_response)
        
        async def test_malformed_handling():
            result = await parser_agent.parse_cv_from_markdown(malformed_content)
            
            # Should handle malformed content gracefully
            self.assertIsInstance(result, dict)
            # May succeed or fail depending on implementation, but should not crash
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_malformed_handling())
        finally:
            loop.close()


if __name__ == '__main__':
    unittest.main()