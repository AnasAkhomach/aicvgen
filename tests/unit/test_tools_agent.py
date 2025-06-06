import unittest
from src.agents.tools_agent import ToolsAgent
from src.core.state_manager import AgentIO
from typing import Dict, Any, List


class TestToolsAgent(unittest.TestCase):

    def setUp(self):
        """Set up ToolsAgent instance before each test."""
        self.agent = ToolsAgent(
            name="TestToolsAgent", description="A test tools agent."
        )

    def test_init(self):
        """Test that ToolsAgent is initialized correctly."""
        self.assertEqual(self.agent.name, "TestToolsAgent")
        self.assertEqual(self.agent.description, "A test tools agent.")
        # TypedDict does not support instance checks, check structure instead
        self.assertIsInstance(self.agent.input_schema, dict)
        self.assertIsInstance(self.agent.output_schema, dict)
        self.assertIn("input", self.agent.input_schema)
        self.assertIn("output", self.agent.input_schema)
        self.assertIn("description", self.agent.input_schema)
        self.assertIn("input", self.agent.output_schema)
        self.assertIn("output", self.agent.output_schema)
        self.assertIn("description", self.agent.output_schema)

        # Schemas are defined as Any in ToolsAgent, so just check that the keys exist
        # self.assertEqual(self.agent.input_schema['input'], Any) # This check is problematic with how Any is represented
        # self.assertEqual(self.agent.output_schema['output'], Any) # This check is problematic with how Any is represented

    def test_run_not_implemented(self):
        """Test that calling the run method raises NotImplementedError."""
        with self.assertRaises(NotImplementedError) as cm:
            self.agent.run(None)  # Pass None as input is Any
        self.assertIn("ToolsAgent methods should be called directly", str(cm.exception))

    def test_format_text_markdown(self):
        """Test format_text with markdown format."""
        text = "Some text to format."
        formatted_text = self.agent.format_text(text, format_type="markdown")
        self.assertEqual(formatted_text, f"Formatted (Markdown): {text}")

    def test_format_text_latex(self):
        """Test format_text with latex format."""
        text = "Another text to format."
        formatted_text = self.agent.format_text(text, format_type="latex")
        self.assertEqual(formatted_text, f"Formatted (LaTeX): {text}")

    def test_format_text_unknown(self):
        """Test format_text with an unknown format type."""
        text = "Yet another text."
        formatted_text = self.agent.format_text(text, format_type="unknown")
        self.assertEqual(formatted_text, f"Formatted ( desconocida): {text}")

    def test_validate_content_all_matched(self):
        """Test validate_content when all requirements are met."""
        content = "This content contains skill1 and skill2."
        requirements = ["skill1", "skill2"]

        validation_results = self.agent.validate_content(content, requirements)

        self.assertTrue(validation_results["is_valid"])
        self.assertIn("Content looks good", validation_results["feedback"])
        self.assertCountEqual(validation_results["matched_requirements"], requirements)
        self.assertEqual(validation_results["missing_requirements"], [])

    def test_validate_content_some_missing(self):
        """Test validate_content when some requirements are missing."""
        content = "This content only has skill1."
        requirements = ["skill1", "skill2", "skill3"]

        validation_results = self.agent.validate_content(content, requirements)

        self.assertFalse(validation_results["is_valid"])
        self.assertIn(
            "Content is missing some requirements", validation_results["feedback"]
        )
        self.assertCountEqual(validation_results["matched_requirements"], ["skill1"])
        self.assertCountEqual(
            validation_results["missing_requirements"], ["skill2", "skill3"]
        )

    def test_validate_content_empty_requirements(self):
        """Test validate_content with an empty list of requirements."""
        content = "Any content will do."
        requirements = []

        validation_results = self.agent.validate_content(content, requirements)

        self.assertTrue(validation_results["is_valid"])
        self.assertIn("Content looks good", validation_results["feedback"])
        self.assertEqual(validation_results["matched_requirements"], [])
        self.assertEqual(validation_results["missing_requirements"], [])


if __name__ == "__main__":
    unittest.main()
