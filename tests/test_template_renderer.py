import unittest
from unittest.mock import MagicMock
from template_renderer import TemplateRenderer
from state_manager import ContentData, AgentIO

class TestTemplateRenderer(unittest.TestCase):

    def setUp(self):
        """Set up mock objects and TemplateRenderer instance before each test."""
        self.mock_llm = MagicMock()
        self.renderer = TemplateRenderer(
            name="Template Renderer",
            description="Agent responsible for rendering CV templates.",
            model=self.mock_llm,
            input_schema=AgentIO(
                input={"content_data": ContentData},
                description="The renderer agent will receive a ContentData and return a rendered CV in markdown format"
            ),
            output_schema=AgentIO(
                output=str,
                description="The renderer agent will receive a ContentData and return a rendered CV in markdown format"
            )
        )

    def test_init(self):
        """Test that TemplateRenderer is initialized correctly."""
        self.assertEqual(self.renderer.name, "Template Renderer")
        self.assertEqual(self.renderer.description, "Agent responsible for rendering CV templates.")
        self.assertEqual(self.renderer.model, self.mock_llm)
        # Validate the structure of input_schema and output_schema instead of using isinstance
        self.assertIn("input", self.renderer.input_schema)
        self.assertIn("output", self.renderer.output_schema)
        self.assertEqual(self.renderer.input_schema['input'], {'content_data': ContentData})
        self.assertEqual(self.renderer.output_schema['output'], str)
        self.assertEqual(self.renderer.input_schema["description"], "The renderer agent will receive a ContentData and return a rendered CV in markdown format")
        self.assertEqual(self.renderer.output_schema["description"], "The renderer agent will receive a ContentData and return a rendered CV in markdown format")

    def test_run_with_all_fields(self):
        """Test run method with a ContentData object containing all fields."""
        content_data: ContentData = {
            "summary": "A results-oriented professional.",
            "experience_bullets": ["Achieved X by doing Y.", "Improved Z by implementing W."],
            "skills_section": "Python, JavaScript, Testing.",
            "projects": ["Project Alpha", "Project Beta"],
            "other_content": {"Awards": "Employee of the Year"}
        }

        expected_markdown = """# Tailored CV

## Summary
A results-oriented professional.

## Experience
- Achieved X by doing Y.
- Improved Z by implementing W.

## Skills
Python, JavaScript, Testing.

## Projects
- Project Alpha
- Project Beta

## Awards
Employee of the Year

"""

        rendered_cv = self.renderer.run(content_data)
        self.assertEqual(rendered_cv, expected_markdown)

    def test_run_with_missing_fields(self):
        """Test run method with a ContentData object missing some fields."""
        content_data: ContentData = {
            "summary": "", # Empty string
            "experience_bullets": [], # Empty list
            "skills_section": "Relevant Skills.",
            "projects": [], # Empty list
            "other_content": {} # Empty dictionary
        }

        expected_markdown = """# Tailored CV

## Skills
Relevant Skills.

"""

        rendered_cv = self.renderer.run(content_data)
        self.assertEqual(rendered_cv, expected_markdown)

    def test_run_with_only_required_fields(self):
        """Test run method with only fields that have content."""
        # Let's test with keys that might be missing or have falsy values

        content_data_falsy: ContentData = {
             "summary": "",
            "experience_bullets": [],
            "skills_section": "",
            "projects": [],
            "other_content": {}
        }
        expected_markdown_falsy = """# Tailored CV"""
        rendered_cv_falsy = self.renderer.run(content_data_falsy)
        self.assertEqual(rendered_cv_falsy, expected_markdown_falsy)

        # Test with some keys missing entirely
        content_data_missing: ContentData = {
            "summary": "Summary Present",
            # experience_bullets missing
            "skills_section": "Skills Present",
            # projects missing
            "other_content": {}
        }
        # Note: The original code uses .get(), so missing keys return None, which is falsy.
        # The output should only include sections for keys that exist and have truthy values.
        expected_markdown_missing = """# Tailored CV

## Summary
Summary Present

## Skills
Skills Present

"""
        rendered_cv_missing = self.renderer.run(content_data_missing)
        self.assertEqual(rendered_cv_missing, expected_markdown_missing)



if __name__ == '__main__':
    unittest.main()
