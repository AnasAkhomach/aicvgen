import unittest
from formatter_agent import FormatterAgent
from state_manager import AgentIO, ContentData
from typing import Dict, Any, List

class TestFormatterAgent(unittest.TestCase):

    def setUp(self):
        """Set up FormatterAgent instance before each test."""
        self.agent = FormatterAgent(
            name="TestFormatterAgent",
            description="A test formatter agent."
        )

    def test_init(self):
        """Test that FormatterAgent is initialized correctly."""
        self.assertEqual(self.agent.name, "TestFormatterAgent")
        self.assertEqual(self.agent.description, "A test formatter agent.")
        # TypedDict does not support instance checks, check structure instead
        self.assertIsInstance(self.agent.input_schema, dict)
        self.assertIsInstance(self.agent.output_schema, dict)
        self.assertIn('input', self.agent.input_schema)
        self.assertIn('output', self.agent.input_schema)
        self.assertIn('description', self.agent.input_schema)
        self.assertIn('input', self.agent.output_schema)
        self.assertIn('output', self.agent.output_schema)
        self.assertIn('description', self.agent.output_schema)

        # Check a few aspects of the schema structure based on formatter_agent.py
        # We can't check the exact type hint (ContentData, Dict[str, Any]), but we can check if the keys exist
        self.assertIn('content_data', self.agent.input_schema['input'])
        self.assertIn('format_specifications', self.agent.input_schema['input'])
        # Check the output type annotation string representation (might vary slightly depending on Python version)
        # A more robust check might involve inspecting __args__ or __origin__ for complex types if needed.
        # For simple types like str, direct comparison might work, but for Dict[str, Any] it's harder.
        # Let's focus on the fact that the 'output' key exists and has some value indicating the type.
        # self.assertEqual(self.agent.output_schema['output'], str) # This check is okay for simple types

    def test_run_with_complete_content(self):
        """Test run method with ContentData containing data in all relevant fields."""
        content_data = ContentData(
            summary="This is a summary.",
            experience_bullets=["Bullet point 1.", "Bullet point 2."],
            skills_section="Skill1, Skill2.",
            projects=["Project A", "Project B"],
            other_content={'Awards': 'Award 1', 'Certifications': 'Cert 1'}
        )
        format_specifications = {"template_type": "markdown"}

        expected_output = (
            "## Summary\nThis is a summary.\n\n"
            "## Experience\n- Bullet point 1.\n- Bullet point 2.\n\n"
            "## Skills\nSkill1, Skill2.\n\n"
            "## Projects\n- Project A\n- Project B\n\n"
            "## Awards\nAward 1.\n\n"
            "## Certifications\nCert 1."
        ).strip()

        formatted_cv = self.agent.run({
            "content_data": content_data,
            "format_specifications": format_specifications
        })

        self.assertEqual(formatted_cv.strip(), expected_output)

    def test_run_with_empty_content(self):
        """Test run method with an empty ContentData object."""
        content_data = ContentData(
            summary="",
            experience_bullets=[],
            skills_section="",
            projects=[],
            other_content={}
        )
        format_specifications = {"template_type": "markdown"}

        expected_output = ""

        formatted_cv = self.agent.run({
            "content_data": content_data,
            "format_specifications": format_specifications
        })

        self.assertEqual(formatted_cv.strip(), expected_output)

    def test_run_with_partial_content(self):
        """Test run method with ContentData missing some fields."""
        content_data = ContentData(
            summary="Partial summary.",
            experience_bullets=[],  # Missing experiences
            skills_section="Partial skills.",
            projects=["Partial project."],
            other_content={}
        )
        format_specifications = {"template_type": "markdown"}

        expected_output = (
            "## Summary\nPartial summary.\n\n"
            "## Skills\nPartial skills.\n\n"
            "## Projects\n- Partial project."
        ).strip()

        formatted_cv = self.agent.run({
            "content_data": content_data,
            "format_specifications": format_specifications
        })

        self.assertEqual(formatted_cv.strip(), expected_output)

    def test_run_with_only_other_content(self):
        """Test run method with only data in other_content."""
        content_data = ContentData(
            summary="",
            experience_bullets=[],
            skills_section="",
            projects=[],
            other_content={'Awards': 'Award 1', 'Certifications': 'Cert 1'}
        )
        format_specifications = {"template_type": "markdown"}

        expected_output = (
            "## Awards\nAward 1\n\n"
            "## Certifications\nCert 1"
        ).strip()

        formatted_cv = self.agent.run({
            "content_data": content_data,
            "format_specifications": format_specifications
        })

        self.assertEqual(formatted_cv.strip(), expected_output)

if __name__ == '__main__':
    unittest.main()
