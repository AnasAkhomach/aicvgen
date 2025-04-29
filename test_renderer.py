import unittest
from template_renderer import TemplateRenderer
from state_manager import ContentData
from unittest.mock import MagicMock

class TestTemplateRenderer(unittest.TestCase):
    def setUp(self):
        mock_llm = MagicMock()
        self.renderer = TemplateRenderer(
            name="Test Renderer",
            description="Test description",
            model=mock_llm,
            input_schema={"input": dict, "output": str, "description": "Input schema"},
            output_schema={"input": dict, "output": str, "description": "Output schema"}
        )

    def test_run_with_all_fields(self):
        """Test run method with a ContentData object containing all fields."""
        content_data = ContentData(
            summary="A results-oriented professional.",
            experience_bullets=["Achieved X by doing Y.", "Improved Z by implementing W."],
            skills_section="Python, JavaScript, Testing.",
            projects=["Project Alpha", "Project Beta"],
            other_content={"Awards": "Employee of the Year"}
        )
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
        content_data = ContentData(
            summary="", # Empty string
            experience_bullets=[], # Empty list
            skills_section="Relevant Skills.",
            projects=[], # Empty list
            other_content={} # Empty dictionary
        )

        expected_markdown = """# Tailored CV

## Skills
Relevant Skills.

"""

        rendered_cv = self.renderer.run(content_data)
        self.assertEqual(rendered_cv, expected_markdown)

    def test_run_with_only_required_fields(self):
        """Test run method with only fields that have content."""
        # Let's test with keys that might be missing or have falsy values
        content_data_falsy = ContentData(
            summary="",
            experience_bullets=[],
            skills_section="",
            projects=[],
            other_content={}
        )
        expected_markdown_falsy = """# Tailored CV"""
        
        rendered_cv_falsy = self.renderer.run(content_data_falsy)
        self.assertEqual(rendered_cv_falsy, expected_markdown_falsy)

if __name__ == '__main__':
    unittest.main() 