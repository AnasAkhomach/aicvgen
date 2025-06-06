import unittest
from unittest.mock import MagicMock, patch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cv_analyzer_agent import CVAnalyzerAgent
from state_manager import AgentIO, CVData, JobDescriptionData
from typing import Dict, Any, List # Import List
import json

# Mock class for LLM dependency
class MockLLM:
    def generate_content(self, prompt: str) -> str:
        pass

class TestCVAnalyzerAgent(unittest.TestCase):

    def setUp(self):
        """Set up mock objects and CVAnalyzerAgent instance before each test."""
        self.mock_llm = MockLLM()
        self.agent = CVAnalyzerAgent(
            name="TestCVAnalyzerAgent",
            description="A test CV analyzer agent.",
            llm=self.mock_llm
        )

    def test_init(self):
        """Test that CVAnalyzerAgent is initialized correctly."""
        self.assertEqual(self.agent.name, "TestCVAnalyzerAgent")
        self.assertEqual(self.agent.description, "A test CV analyzer agent.")
        self.assertEqual(self.agent.llm, self.mock_llm)
        # Check that input_schema and output_schema are dictionaries with expected keys
        self.assertIsInstance(self.agent.input_schema, dict)
        self.assertIn('input', self.agent.input_schema)
        self.assertIn('output', self.agent.input_schema)
        self.assertIn('description', self.agent.input_schema)

        self.assertIsInstance(self.agent.output_schema, dict)
        self.assertIn('input', self.agent.output_schema)
        self.assertIn('output', self.agent.output_schema)
        self.assertIn('description', self.agent.output_schema)

    @patch.object(MockLLM, 'generate_content')
    def test_run_with_typical_cv(self, mock_generate_content):
        """Test run method with typical CV input and LLM response."""
        user_cv_data = CVData(
            raw_text="John Doe\nSummary: Experienced developer.\nExperience: Worked at Google.\nSkills: Python."
        )
        job_description_data = JobDescriptionData(
            raw_text="Software Engineer",
            skills=["Python"],
            experience_level="Mid",
            responsibilities=[],
            industry_terms=[],
            company_values=[]
        )

        input_data = {
            "user_cv": user_cv_data,
            "job_description": job_description_data
        }

        mock_llm_response_dict = {
            "summary": "Experienced developer.",
            "experiences": ["Worked at Google."],
            "skills": ["Python"],
            "education": [],
            "projects": []
        }
        mock_llm_response_json = json.dumps(mock_llm_response_dict)
        mock_generate_content.return_value = mock_llm_response_json

        extracted_data = self.agent.run(input_data)

        mock_generate_content.assert_called_once()
        called_prompt = mock_generate_content.call_args[0][0]
        # Access attributes using dictionary syntax instead of dot notation
        self.assertIn(user_cv_data["raw_text"], called_prompt)
        self.assertIn(job_description_data.raw_text, called_prompt)
        self.assertIn("Python", called_prompt)

        self.assertEqual(extracted_data, mock_llm_response_dict)
    @patch.object(MockLLM, 'generate_content')

    def test_run_with_empty_cv_text(self, mock_generate_content):
        """Test run method with empty CV text."""
        user_cv_data = CVData(raw_text="")
        job_description_data = JobDescriptionData(raw_text="Software Engineer", skills=["Python"], experience_level="Mid", responsibilities=[], industry_terms=[], company_values=[])

        input_data = {
            "user_cv": user_cv_data,
            "job_description": job_description_data
        }

        expected_empty_output = {
            "summary": "",
            "experiences": [],
            "skills": [],
            "education": [],
            "projects": []
        }

        extracted_data = self.agent.run(input_data)

        mock_generate_content.assert_not_called() # LLM should not be called for empty CV
        self.assertEqual(extracted_data, expected_empty_output)

    @patch.object(MockLLM, 'generate_content')
    def test_run_with_llm_json_decode_error(self, mock_generate_content):
        """Test run method when LLM returns invalid JSON."""
        user_cv_data = CVData(raw_text="Some CV text.")
        job_description_data = JobDescriptionData(raw_text="", skills=[], experience_level="", responsibilities=[], industry_terms=[], company_values=[])

        input_data = {
            "user_cv": user_cv_data,
            "job_description": job_description_data
        }

        mock_generate_content.return_value = "This is not valid JSON."

        extracted_data = self.agent.run(input_data)

        mock_generate_content.assert_called_once()
        self.assertIn("Error parsing CV", extracted_data["summary"])
        self.assertEqual(extracted_data["experiences"], [])

    @patch.object(MockLLM, 'generate_content')
    def test_run_with_llm_markdown_json(self, mock_generate_content):
        """Test run method with LLM response wrapped in markdown json."""
        user_cv_data = CVData(raw_text="Some CV text.")
        job_description_data = JobDescriptionData(raw_text="", skills=[], experience_level="", responsibilities=[], industry_terms=[], company_values=[])

        input_data = {
            "user_cv": user_cv_data,
            "job_description": job_description_data
        }

        mock_llm_response_dict = {
            "summary": "Markdown JSON test.",
            "experiences": [],
            "skills": [],
            "education": [],
            "projects": []
        }
        mock_llm_response_json_markdown = f"```json\n{json.dumps(mock_llm_response_dict)}\n```"
        mock_generate_content.return_value = mock_llm_response_json_markdown

        extracted_data = self.agent.run(input_data)

        mock_generate_content.assert_called_once()
        self.assertEqual(extracted_data, mock_llm_response_dict)

    @patch.object(MockLLM, 'generate_content')
    def test_run_with_llm_response_missing_keys(self, mock_generate_content):
        """Test run method with LLM response missing some keys."""
        user_cv_data = CVData(raw_text="Some CV text.")
        job_description_data = JobDescriptionData(raw_text="", skills=[], experience_level="", responsibilities=[], industry_terms=[], company_values=[])

        input_data = {
            "user_cv": user_cv_data,
            "job_description": job_description_data
        }

        # Simulate LLM response that only includes summary and skills
        mock_llm_response_dict = {
            "summary": "Summary only.",
            "skills": ["Skill1", "Skill2"]
        }
        mock_llm_response_json = json.dumps(mock_llm_response_dict)
        mock_generate_content.return_value = mock_llm_response_json

        extracted_data = self.agent.run(input_data)

        mock_generate_content.assert_called_once()
        # Assert that missing keys are included with default empty values
        self.assertEqual(extracted_data["summary"], "Summary only.")
        self.assertEqual(extracted_data["skills"], ["Skill1", "Skill2"])
        self.assertEqual(extracted_data["experiences"], [])
        self.assertEqual(extracted_data["education"], [])
        self.assertEqual(extracted_data["projects"], [])

    @patch.object(MockLLM, 'generate_content')
    def test_run_with_unexpected_exception(self, mock_generate_content):
        """Test run method with an unexpected exception during LLM call."""
        user_cv_data = CVData(raw_text="Some CV text.")
        job_description_data = JobDescriptionData(raw_text="", skills=[], experience_level="", responsibilities=[], industry_terms=[], company_values=[])

        input_data = {
            "user_cv": user_cv_data,
            "job_description": job_description_data
        }

        mock_generate_content.side_effect = Exception("LLM internal error")

        extracted_data = self.agent.run(input_data)

        mock_generate_content.assert_called_once()
        self.assertIn("Error analyzing CV", extracted_data["summary"])
        self.assertEqual(extracted_data["experiences"], [])

    @patch.object(MockLLM, 'generate_content')
    def test_run_with_null_response(self, mock_generate_content):
        """Test run method when LLM returns a null response."""
        user_cv_data = CVData(raw_text="Some CV text.")
        job_description_data = JobDescriptionData(raw_text="", skills=[], experience_level="", responsibilities=[], industry_terms=[], company_values=[])

        input_data = {
            "user_cv": user_cv_data,
            "job_description": job_description_data
        }

        # Simulate a null response from LLM
        mock_generate_content.return_value = None

        extracted_data = self.agent.run(input_data)

        mock_generate_content.assert_called_once()
        self.assertIn("Error analyzing CV", extracted_data["summary"])
        self.assertEqual(extracted_data["experiences"], [])

    @patch.object(MockLLM, 'generate_content')
    def test_run_with_partial_json(self, mock_generate_content):
        """Test run method when LLM returns partial JSON."""
        user_cv_data = CVData(raw_text="Some CV text.")
        job_description_data = JobDescriptionData(raw_text="", skills=[], experience_level="", responsibilities=[], industry_terms=[], company_values=[])

        input_data = {
            "user_cv": user_cv_data,
            "job_description": job_description_data
        }

        # Simulate partial JSON response from LLM
        mock_llm_response_dict = {
            "summary": "Partial summary."
        }
        mock_llm_response_json = json.dumps(mock_llm_response_dict)
        mock_generate_content.return_value = mock_llm_response_json

        extracted_data = self.agent.run(input_data)

        mock_generate_content.assert_called_once()
        self.assertEqual(extracted_data["summary"], "Partial summary.")
        self.assertEqual(extracted_data["experiences"], [])
        self.assertEqual(extracted_data["skills"], [])
        self.assertEqual(extracted_data["education"], [])
        self.assertEqual(extracted_data["projects"], [])


if __name__ == '__main__':
    unittest.main()
