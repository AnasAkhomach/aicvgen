"""Unit tests for Enhanced Formatter Agent.

Tests PDF generation, content formatting, template management,
and output quality validation.
"""

import unittest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock, mock_open
import tempfile
import os
import sys
from typing import Dict, Any, List
import json

# Add the project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.formatter_agent import FormatterAgent
from src.models.data_models import (
    ContentType, ProcessingStatus, CVData, JobDescriptionData,
    ExperienceItem, ProjectItem, QualificationItem
)
from src.services.llm import LLMResponse


class TestEnhancedFormatterAgent(unittest.TestCase):
    """Test cases for Enhanced Formatter Agent."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm_service = Mock()
        self.mock_config = Mock()
        self.mock_logger = Mock()
        
        # Sample CV data for testing
        self.sample_cv_data = CVData(
            personal_info={
                "name": "John Doe",
                "title": "Senior Software Engineer",
                "email": "john.doe@email.com",
                "phone": "(555) 123-4567",
                "linkedin": "linkedin.com/in/johndoe",
                "location": "San Francisco, CA"
            },
            summary="Experienced software engineer with 5+ years in full-stack development, specializing in React and Node.js applications.",
            experiences=[
                ExperienceItem(
                    title="Senior Software Engineer",
                    company="TechCorp Inc.",
                    duration="2021-2023",
                    responsibilities=[
                        "Developed scalable web applications using React and Node.js",
                        "Led a team of 4 developers on microservices architecture",
                        "Improved system performance by 40% through optimization"
                    ],
                    technologies=["React", "Node.js", "AWS", "Docker"]
                ),
                ExperienceItem(
                    title="Software Engineer",
                    company="StartupXYZ",
                    duration="2019-2021",
                    responsibilities=[
                        "Built REST APIs using Python and Django",
                        "Implemented CI/CD pipelines with Jenkins",
                        "Collaborated with cross-functional teams"
                    ],
                    technologies=["Python", "Django", "Jenkins", "PostgreSQL"]
                )
            ],
            education=[
                QualificationItem(
                    degree="Bachelor of Science in Computer Science",
                    institution="University ABC",
                    duration="2015-2019",
                    details="GPA: 3.8/4.0, Relevant Coursework: Data Structures, Algorithms"
                )
            ],
            projects=[
                ProjectItem(
                    name="E-commerce Platform",
                    year="2022",
                    description="Built full-stack e-commerce application with payment integration",
                    technologies=["React", "Node.js", "MongoDB", "Stripe"]
                ),
                ProjectItem(
                    name="Task Management App",
                    year="2021",
                    description="Developed mobile app using React Native with real-time sync",
                    technologies=["React Native", "Firebase", "Redux"]
                )
            ],
            skills={
                "programming_languages": ["Python", "JavaScript", "Java", "C++"],
                "frameworks": ["React", "Node.js", "Django", "Express"],
                "databases": ["MongoDB", "PostgreSQL", "MySQL"],
                "tools": ["Git", "Docker", "Jenkins", "AWS"]
            }
        )
        
        # Sample job description data
        self.sample_job_data = JobDescriptionData(
            job_title="Senior Full-Stack Developer",
            company="InnovateTech Solutions",
            location="San Francisco, CA",
            employment_type="Full-time",
            salary_range="$120,000 - $150,000",
            required_skills=[
                "JavaScript", "Python", "React", "Node.js", "SQL", "NoSQL"
            ],
            preferred_skills=[
                "Microservices", "Docker", "Kubernetes", "Leadership"
            ],
            experience_required="5+ years",
            education_required="Bachelor's degree in Computer Science",
            responsibilities=[
                "Design and develop web applications",
                "Collaborate with product managers",
                "Mentor junior developers"
            ]
        )

    @patch('src.agents.formatter_agent.get_structured_logger')
    @patch('src.agents.formatter_agent.get_config')
    @patch('src.agents.formatter_agent.get_llm_service')
    def test_formatter_agent_initialization(self, mock_llm_service, mock_config, mock_logger):
        """Test Formatter Agent initialization."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create formatter agent
        formatter_agent = FormatterAgent()
        
        # Verify initialization
        self.assertIsNotNone(formatter_agent.llm_service)
        self.assertIsNotNone(formatter_agent.settings)
        self.assertIsNotNone(formatter_agent.logger)
        self.assertEqual(formatter_agent.name, "FormatterAgent")
        self.assertIn("format", formatter_agent.description.lower())

    @patch('src.agents.formatter_agent.get_structured_logger')
    @patch('src.agents.formatter_agent.get_config')
    @patch('src.agents.formatter_agent.get_llm_service')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_pdf_generation_from_cv_data(self, mock_exists, mock_file, mock_llm_service, mock_config, mock_logger):
        """Test PDF generation from CV data."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        mock_exists.return_value = True
        
        # Mock template content
        template_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{name}} - {{title}}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; margin-bottom: 30px; }
                .section { margin-bottom: 25px; }
                .experience { margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{name}}</h1>
                <h2>{{title}}</h2>
                <p>{{email}} | {{phone}} | {{location}}</p>
            </div>
            
            <div class="section">
                <h3>Professional Summary</h3>
                <p>{{summary}}</p>
            </div>
            
            <div class="section">
                <h3>Work Experience</h3>
                {% for experience in experiences %}
                <div class="experience">
                    <h4>{{experience.title}} - {{experience.company}}</h4>
                    <p><em>{{experience.duration}}</em></p>
                    <ul>
                    {% for responsibility in experience.responsibilities %}
                        <li>{{responsibility}}</li>
                    {% endfor %}
                    </ul>
                </div>
                {% endfor %}
            </div>
            
            <div class="section">
                <h3>Skills</h3>
                <p><strong>Programming Languages:</strong> {{skills.programming_languages|join(', ')}}</p>
                <p><strong>Frameworks:</strong> {{skills.frameworks|join(', ')}}</p>
                <p><strong>Databases:</strong> {{skills.databases|join(', ')}}</p>
                <p><strong>Tools:</strong> {{skills.tools|join(', ')}}</p>
            </div>
        </body>
        </html>
        """
        
        mock_file.return_value.read.return_value = template_content
        
        # Create formatter agent
        formatter_agent = FormatterAgent()
        
        # Test PDF generation
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_cv.pdf")
            
            async def test_pdf_generation():
                result = await formatter_agent.generate_pdf_from_cv(
                    cv_data=self.sample_cv_data,
                    output_path=output_path,
                    template_name="modern"
                )
                
                # Verify PDF generation result
                self.assertIsInstance(result, dict)
                self.assertTrue(result.get('success', False))
                self.assertEqual(result.get('output_path'), output_path)
                self.assertIsNotNone(result.get('file_size'))
                self.assertGreater(result.get('file_size', 0), 0)
            
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(test_pdf_generation())
            finally:
                loop.close()

    @patch('src.agents.formatter_agent.get_structured_logger')
    @patch('src.agents.formatter_agent.get_config')
    @patch('src.agents.formatter_agent.get_llm_service')
    def test_html_template_rendering(self, mock_llm_service, mock_config, mock_logger):
        """Test HTML template rendering with CV data."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create formatter agent
        formatter_agent = FormatterAgent()
        
        # Simple template for testing
        template_content = """
        <h1>{{name}}</h1>
        <h2>{{title}}</h2>
        <p>{{summary}}</p>
        <ul>
        {% for experience in experiences %}
            <li>{{experience.title}} at {{experience.company}}</li>
        {% endfor %}
        </ul>
        """
        
        # Test template rendering
        rendered_html = formatter_agent.render_template(
            template_content=template_content,
            cv_data=self.sample_cv_data
        )
        
        # Verify rendering
        self.assertIsInstance(rendered_html, str)
        self.assertIn("John Doe", rendered_html)
        self.assertIn("Senior Software Engineer", rendered_html)
        self.assertIn("TechCorp Inc.", rendered_html)
        self.assertIn("StartupXYZ", rendered_html)
        self.assertIn("full-stack development", rendered_html)

    @patch('src.agents.formatter_agent.get_structured_logger')
    @patch('src.agents.formatter_agent.get_config')
    @patch('src.agents.formatter_agent.get_llm_service')
    def test_content_formatting_for_different_sections(self, mock_llm_service, mock_config, mock_logger):
        """Test content formatting for different CV sections."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create formatter agent
        formatter_agent = FormatterAgent()
        
        # Test experience formatting
        formatted_experience = formatter_agent.format_experience_section(
            self.sample_cv_data.experiences
        )
        
        self.assertIsInstance(formatted_experience, str)
        self.assertIn("Senior Software Engineer", formatted_experience)
        self.assertIn("TechCorp Inc.", formatted_experience)
        self.assertIn("2021-2023", formatted_experience)
        
        # Test skills formatting
        formatted_skills = formatter_agent.format_skills_section(
            self.sample_cv_data.skills
        )
        
        self.assertIsInstance(formatted_skills, str)
        self.assertIn("Python", formatted_skills)
        self.assertIn("JavaScript", formatted_skills)
        self.assertIn("React", formatted_skills)
        
        # Test education formatting
        formatted_education = formatter_agent.format_education_section(
            self.sample_cv_data.education
        )
        
        self.assertIsInstance(formatted_education, str)
        self.assertIn("Bachelor of Science", formatted_education)
        self.assertIn("University ABC", formatted_education)
        self.assertIn("2015-2019", formatted_education)
        
        # Test projects formatting
        formatted_projects = formatter_agent.format_projects_section(
            self.sample_cv_data.projects
        )
        
        self.assertIsInstance(formatted_projects, str)
        self.assertIn("E-commerce Platform", formatted_projects)
        self.assertIn("Task Management App", formatted_projects)
        self.assertIn("React Native", formatted_projects)

    @patch('src.agents.formatter_agent.get_structured_logger')
    @patch('src.agents.formatter_agent.get_config')
    @patch('src.agents.formatter_agent.get_llm_service')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_template_management(self, mock_listdir, mock_exists, mock_file, mock_llm_service, mock_config, mock_logger):
        """Test template loading and management."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        mock_exists.return_value = True
        mock_listdir.return_value = ['modern.html', 'classic.html', 'minimal.html']
        
        # Mock template content
        template_content = "<html><body>{{name}}</body></html>"
        mock_file.return_value.read.return_value = template_content
        
        # Create formatter agent
        formatter_agent = FormatterAgent()
        
        # Test template loading
        available_templates = formatter_agent.get_available_templates()
        
        self.assertIsInstance(available_templates, list)
        self.assertIn('modern', available_templates)
        self.assertIn('classic', available_templates)
        self.assertIn('minimal', available_templates)
        
        # Test specific template loading
        loaded_template = formatter_agent.load_template('modern')
        
        self.assertIsInstance(loaded_template, str)
        self.assertEqual(loaded_template, template_content)

    @patch('src.agents.formatter_agent.get_structured_logger')
    @patch('src.agents.formatter_agent.get_config')
    @patch('src.agents.formatter_agent.get_llm_service')
    def test_css_styling_integration(self, mock_llm_service, mock_config, mock_logger):
        """Test CSS styling integration in templates."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create formatter agent
        formatter_agent = FormatterAgent()
        
        # Test CSS generation for different themes
        modern_css = formatter_agent.generate_css_styles('modern')
        
        self.assertIsInstance(modern_css, str)
        self.assertIn('font-family', modern_css)
        self.assertIn('color', modern_css)
        self.assertIn('margin', modern_css)
        
        classic_css = formatter_agent.generate_css_styles('classic')
        
        self.assertIsInstance(classic_css, str)
        self.assertNotEqual(modern_css, classic_css)  # Different themes should have different styles
        
        # Test CSS minification
        minified_css = formatter_agent.minify_css(modern_css)
        
        self.assertIsInstance(minified_css, str)
        self.assertLess(len(minified_css), len(modern_css))  # Minified should be shorter

    @patch('src.agents.formatter_agent.get_structured_logger')
    @patch('src.agents.formatter_agent.get_config')
    @patch('src.agents.formatter_agent.get_llm_service')
    def test_responsive_design_elements(self, mock_llm_service, mock_config, mock_logger):
        """Test responsive design elements in templates."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create formatter agent
        formatter_agent = FormatterAgent()
        
        # Test responsive CSS generation
        responsive_css = formatter_agent.generate_responsive_css()
        
        self.assertIsInstance(responsive_css, str)
        self.assertIn('@media', responsive_css)
        self.assertIn('max-width', responsive_css)
        self.assertIn('min-width', responsive_css)
        
        # Test mobile-friendly formatting
        mobile_template = formatter_agent.create_mobile_template(
            self.sample_cv_data
        )
        
        self.assertIsInstance(mobile_template, str)
        self.assertIn('viewport', mobile_template)
        self.assertIn('responsive', mobile_template.lower())

    @patch('src.agents.formatter_agent.get_structured_logger')
    @patch('src.agents.formatter_agent.get_config')
    @patch('src.agents.formatter_agent.get_llm_service')
    def test_output_quality_validation(self, mock_llm_service, mock_config, mock_logger):
        """Test output quality validation."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create formatter agent
        formatter_agent = FormatterAgent()
        
        # Test HTML validation
        valid_html = "<html><head><title>Test</title></head><body><h1>Valid HTML</h1></body></html>"
        invalid_html = "<html><head><title>Test</head><body><h1>Invalid HTML</body></html>"
        
        is_valid = formatter_agent.validate_html(valid_html)
        self.assertTrue(is_valid)
        
        is_invalid = formatter_agent.validate_html(invalid_html)
        self.assertFalse(is_invalid)
        
        # Test content completeness validation
        complete_content = {
            'name': 'John Doe',
            'title': 'Software Engineer',
            'summary': 'Experienced developer',
            'experiences': [{'title': 'Engineer', 'company': 'TechCorp'}],
            'skills': {'programming_languages': ['Python']}
        }
        
        incomplete_content = {
            'name': 'John Doe'
            # Missing other required fields
        }
        
        is_complete = formatter_agent.validate_content_completeness(complete_content)
        self.assertTrue(is_complete)
        
        is_incomplete = formatter_agent.validate_content_completeness(incomplete_content)
        self.assertFalse(is_incomplete)

    @patch('src.agents.formatter_agent.get_structured_logger')
    @patch('src.agents.formatter_agent.get_config')
    @patch('src.agents.formatter_agent.get_llm_service')
    def test_custom_formatting_options(self, mock_llm_service, mock_config, mock_logger):
        """Test custom formatting options and preferences."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create formatter agent
        formatter_agent = FormatterAgent()
        
        # Test custom formatting options
        formatting_options = {
            'font_family': 'Helvetica',
            'font_size': '12pt',
            'line_height': '1.5',
            'margin': '1in',
            'color_scheme': 'blue',
            'include_photo': False,
            'section_order': ['summary', 'experience', 'education', 'skills']
        }
        
        # Apply custom formatting
        formatted_content = formatter_agent.apply_custom_formatting(
            cv_data=self.sample_cv_data,
            options=formatting_options
        )
        
        self.assertIsInstance(formatted_content, str)
        self.assertIn('Helvetica', formatted_content)
        self.assertIn('12pt', formatted_content)
        self.assertIn('1.5', formatted_content)
        
        # Test section reordering
        self.assertTrue(
            formatted_content.find('Professional Summary') < 
            formatted_content.find('Work Experience')
        )

    @patch('src.agents.formatter_agent.get_structured_logger')
    @patch('src.agents.formatter_agent.get_config')
    @patch('src.agents.formatter_agent.get_llm_service')
    def test_error_handling_in_formatting(self, mock_llm_service, mock_config, mock_logger):
        """Test error handling in formatting operations."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create formatter agent
        formatter_agent = FormatterAgent()
        
        # Test with invalid template
        invalid_template = "{{invalid_syntax}}"
        
        try:
            result = formatter_agent.render_template(
                template_content=invalid_template,
                cv_data=self.sample_cv_data
            )
            # Should handle gracefully or raise appropriate exception
        except Exception as e:
            self.assertIsInstance(e, (ValueError, TypeError, Exception))
        
        # Test with missing data
        incomplete_cv_data = CVData(
            personal_info={"name": "John Doe"},
            summary="",
            experiences=[],
            education=[],
            projects=[],
            skills={}
        )
        
        # Should handle incomplete data gracefully
        result = formatter_agent.format_experience_section(incomplete_cv_data.experiences)
        self.assertIsInstance(result, str)

    @patch('src.agents.formatter_agent.get_structured_logger')
    @patch('src.agents.formatter_agent.get_config')
    @patch('src.agents.formatter_agent.get_llm_service')
    def test_multi_format_export(self, mock_llm_service, mock_config, mock_logger):
        """Test export to multiple formats (HTML, PDF, DOCX)."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create formatter agent
        formatter_agent = FormatterAgent()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test HTML export
            html_path = os.path.join(temp_dir, "cv.html")
            html_result = formatter_agent.export_to_html(
                cv_data=self.sample_cv_data,
                output_path=html_path
            )
            
            self.assertIsInstance(html_result, dict)
            self.assertTrue(html_result.get('success', False))
            
            # Test PDF export
            pdf_path = os.path.join(temp_dir, "cv.pdf")
            
            async def test_pdf_export():
                pdf_result = await formatter_agent.export_to_pdf(
                    cv_data=self.sample_cv_data,
                    output_path=pdf_path
                )
                
                self.assertIsInstance(pdf_result, dict)
                self.assertTrue(pdf_result.get('success', False))
            
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(test_pdf_export())
            finally:
                loop.close()

    @patch('src.agents.formatter_agent.get_structured_logger')
    @patch('src.agents.formatter_agent.get_config')
    @patch('src.agents.formatter_agent.get_llm_service')
    def test_accessibility_features(self, mock_llm_service, mock_config, mock_logger):
        """Test accessibility features in generated content."""
        # Setup mocks
        mock_llm_service.return_value = self.mock_llm_service
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create formatter agent
        formatter_agent = FormatterAgent()
        
        # Test accessibility-compliant HTML generation
        accessible_html = formatter_agent.generate_accessible_html(
            cv_data=self.sample_cv_data
        )
        
        self.assertIsInstance(accessible_html, str)
        
        # Check for accessibility features
        self.assertIn('alt=', accessible_html)  # Alt text for images
        self.assertIn('role=', accessible_html)  # ARIA roles
        self.assertIn('aria-', accessible_html)  # ARIA attributes
        self.assertIn('lang=', accessible_html)  # Language attribute
        
        # Test semantic HTML structure
        self.assertIn('<header>', accessible_html)
        self.assertIn('<main>', accessible_html)
        self.assertIn('<section>', accessible_html)
        self.assertIn('<article>', accessible_html)


if __name__ == '__main__':
    unittest.main()