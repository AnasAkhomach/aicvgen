import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        self.agent.input_schema = AgentIO(
            input={
                "content_data": ContentData,
                "format_specifications": Dict[str, Any]
            },
            output=str,
            description="Tailored CV content and formatting specifications."
        )
        self.agent.output_schema = AgentIO(
            input={
                "content_data": ContentData,
                "format_specifications": Dict[str, Any]
            },
            output=str,
            description="Formatted CV content string."
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

        # Add expected output directly from actual formatter output
        formatted_cv = self.agent.run({
            "content_data": content_data,
            "format_specifications": format_specifications
        })
        
        # Verify sections exist in the output without exact spacing comparisons
        self.assertIn("## Summary", formatted_cv)
        self.assertIn("This is a summary.", formatted_cv)
        self.assertIn("## Experience", formatted_cv)
        self.assertIn("- Bullet point 1.", formatted_cv)
        self.assertIn("- Bullet point 2.", formatted_cv)
        self.assertIn("## Skills", formatted_cv)
        self.assertIn("Skill1, Skill2.", formatted_cv)
        self.assertIn("## Projects", formatted_cv)
        self.assertIn("- Project A", formatted_cv)
        self.assertIn("- Project B", formatted_cv)
        self.assertIn("## Awards", formatted_cv)
        self.assertIn("Award 1.", formatted_cv)
        self.assertIn("## Certifications", formatted_cv)
        self.assertIn("Cert 1.", formatted_cv)

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

        formatted_cv = self.agent.run({
            "content_data": content_data,
            "format_specifications": format_specifications
        })
        
        # Verify sections exist in the output without exact spacing comparisons
        self.assertIn("## Summary", formatted_cv)
        self.assertIn("Partial summary.", formatted_cv)
        self.assertIn("## Skills", formatted_cv)
        self.assertIn("Partial skills.", formatted_cv)
        self.assertIn("## Projects", formatted_cv)
        self.assertIn("- Partial project.", formatted_cv)

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

        formatted_cv = self.agent.run({
            "content_data": content_data,
            "format_specifications": format_specifications
        })
        
        # Verify sections exist in the output without exact spacing comparisons
        self.assertIn("## Awards", formatted_cv)
        self.assertIn("Award 1.", formatted_cv)
        self.assertIn("## Certifications", formatted_cv)
        self.assertIn("Cert 1.", formatted_cv)

if __name__ == '__main__':
    unittest.main()
