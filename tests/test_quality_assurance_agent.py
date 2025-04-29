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
        # Treat input_schema and output_schema as dictionaries
        self.assertIn("input", self.agent.input_schema)
        self.assertIn("output", self.agent.output_schema)
        self.assertEqual(self.agent.input_schema["input"], {
            "formatted_cv_text": str,
            "job_description": Dict[str, Any]
        })
        self.assertEqual(self.agent.output_schema["output"], Dict[str, Any])
        self.assertEqual(self.agent.input_schema["description"], "Formatted CV content and job description for quality assurance.")
        self.assertEqual(self.agent.output_schema["description"], "Quality assurance results and feedback.")

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
        
        # The missing keywords should include Java and C++ but not Python
        self.assertIn("Missing potential ATS keywords:", quality_results["feedback"])
        # Since Python is mentioned in the CV text, it shouldn't be in the missing keywords
        self.assertIn("C++", quality_results["feedback"])
        
        # The suggestions should include incorporating the missing keywords
        self.assertTrue(any("Consider incorporating keywords" in suggestion for suggestion in quality_results["suggestions"]))

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
