import unittest
from quality_assurance_agent import QualityAssuranceAgent
from state_manager import AgentIO, JobDescriptionData
from typing import Dict, Any

class TestQualityAssuranceAgent(unittest.TestCase):

    def setUp(self):
        """Set up QualityAssuranceAgent instance before each test."""
        self.agent = QualityAssuranceAgent(
            name="TestQualityAssuranceAgent",
            description="A test quality assurance agent."
        )

    def test_init(self):
        """Test that QualityAssuranceAgent is initialized correctly."""
        self.assertEqual(self.agent.name, "TestQualityAssuranceAgent")
        self.assertEqual(self.agent.description, "A test quality assurance agent.")
        # TypedDict does not support instance checks, check structure instead
        self.assertIsInstance(self.agent.input_schema, dict)
        self.assertIsInstance(self.agent.output_schema, dict)
        self.assertIn('input', self.agent.input_schema)
        self.assertIn('output', self.agent.input_schema)
        self.assertIn('description', self.agent.input_schema)
        self.assertIn('input', self.agent.output_schema)
        self.assertIn('output', self.agent.output_schema)
        self.assertIn('description', self.agent.output_schema)

        # Check a few aspects of the schema structure based on quality_assurance_agent.py
        self.assertIn('formatted_cv_text', self.agent.input_schema['input'])
        self.assertIn('job_description', self.agent.input_schema['input'])
        # Checking the output type annotation string representation might be better
        # self.assertEqual(self.agent.output_schema['output'], Dict[str, Any]) # This check is problematic for Dict[str, Any]
        # A simpler check is just to ensure the 'output' key exists.
        self.assertIn('output', self.agent.output_schema)

    def test_run_quality_ok(self):
        """Test run with formatted CV text that should pass quality checks (simulated)."""
        formatted_cv_text = "This is a long enough CV text with Python and Java skills mentioned." * 10 # Make it long enough
        job_description_data = {"skills": ["Python", "Java"]}

        input_data = {
            "formatted_cv_text": formatted_cv_text,
            "job_description": job_description_data
        }

        quality_results = self.agent.run(input_data)

        self.assertTrue(quality_results["is_quality_ok"])
        self.assertEqual(quality_results["feedback"], "No major issues detected (simulated).")
        self.assertEqual(quality_results["suggestions"], [])

    def test_run_short_content(self):
        """Test run with short formatted CV text to trigger length feedback."""
        formatted_cv_text = "Short text."
        job_description_data = {"skills": []}

        input_data = {
            "formatted_cv_text": formatted_cv_text,
            "job_description": job_description_data
        }

        quality_results = self.agent.run(input_data)

        self.assertFalse(quality_results["is_quality_ok"])
        self.assertIn("Content seems very short", quality_results["feedback"])
        self.assertEqual(quality_results["suggestions"], []) # No keyword suggestions as no skills in JD

    def test_run_missing_keywords(self):
        """Test run with formatted CV text missing keywords from job description."""
        formatted_cv_text = "This CV mentions Python but not Java."
        job_description_data = {"skills": ["Python", "Java", "C++"]}

        input_data = {
            "formatted_cv_text": formatted_cv_text,
            "job_description": job_description_data
        }

        quality_results = self.agent.run(input_data)

        self.assertFalse(quality_results["is_quality_ok"])
        # The expected feedback now includes the short content message due to the simulated check
        self.assertIn("Content seems very short", quality_results["feedback"])
        self.assertIn("Missing potential ATS keywords: Java, C++.", quality_results["feedback"])
        self.assertIn("Consider incorporating keywords like: Java, C++.", " ".join(quality_results["suggestions"]))

    def test_run_empty_cv_text(self):
        """Test run with empty formatted CV text."""
        formatted_cv_text = ""
        job_description_data = {"skills": ["Python"]}

        input_data = {
            "formatted_cv_text": formatted_cv_text,
            "job_description": job_description_data
        }

        quality_results = self.agent.run(input_data)

        # Expecting both short content feedback and missing keywords feedback
        self.assertFalse(quality_results["is_quality_ok"])
        self.assertIn("Content seems very short", quality_results["feedback"])
        self.assertIn("Missing potential ATS keywords: Python.", quality_results["feedback"])
        self.assertIn("Consider incorporating keywords like: Python.", " ".join(quality_results["suggestions"]))

    def test_run_empty_job_description(self):
        """Test run with empty job description data."""
        formatted_cv_text = "This is a long enough CV text." * 10
        job_description_data = {}

        input_data = {
            "formatted_cv_text": formatted_cv_text,
            "job_description": job_description_data
        }

        quality_results = self.agent.run(input_data)

        # Should pass the keyword check as there are no required keywords
        self.assertTrue(quality_results["is_quality_ok"])
        self.assertEqual(quality_results["feedback"], "No major issues detected (simulated).")
        self.assertEqual(quality_results["suggestions"], [])

if __name__ == '__main__':
    unittest.main()
