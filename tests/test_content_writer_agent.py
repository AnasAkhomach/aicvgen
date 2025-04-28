import unittest
from unittest.mock import MagicMock, patch
from content_writer_agent import ContentWriterAgent
from state_manager import ContentData, AgentIO, JobDescriptionData, CVData
from typing import Dict, List, Any
import json

# Mock classes for dependencies
class MockLLM:
    def generate_content(self, prompt: str) -> str:
        pass

class MockToolsAgent:
    def format_text(self, text: str, format_type: str) -> str:
        pass

    def validate_content(self, content: str, requirements: List[str]) -> Dict[str, Any]:
        pass


# Removed the class-level patch @patch('google.generativeai')
class TestContentWriterAgent(unittest.TestCase):

    def setUp(self):
        """Set up mock objects and ContentWriterAgent instance before each test."""
        self.mock_llm = MockLLM()
        self.mock_tools_agent = MockToolsAgent()
        self.agent = ContentWriterAgent(
            name="TestContentWriterAgent",
            description="A test content writer agent.",
            llm=self.mock_llm,
            tools_agent=self.mock_tools_agent
        )

    @patch.object(MockLLM, 'generate_content')
    @patch.object(MockToolsAgent, 'format_text')
    @patch.object(MockToolsAgent, 'validate_content')
    def test_run_typical_input(self, mock_validate_content, mock_format_text, mock_generate_content): # Arguments now match the 3 method-level patches
        """Test run method with typical input data."""
        input_data = {
            "job_description_data": {
                "skills": ["Python", "Machine Learning"],
                "responsibilities": ["Develop models", "Deploy solutions"]
            },
            "relevant_experiences": [
                "Built ML model for fraud detection.",
                "Deployed scalable solutions on cloud platforms."
            ],
            "research_results": {"industry trends": "AI is growing"},
            "user_cv_data": {
                "summary": "Experienced ML Engineer.",
                "skills": ["Python", "SQL"],
                "education": ["M.S. in Computer Science"],
                "projects": ["Project X"]
            }
        }

        mock_llm_response = json.dumps({
            "summary": "Highly skilled ML Engineer with experience in model development and deployment.",
            "experience_bullets": [
                "Developed and deployed a fraud detection model using Python.",
                "Leveraged cloud platforms to deploy scalable ML solutions."
            ],
            "skills_section": "Python, Machine Learning, SQL, Cloud Computing.",
            "projects": ["Project X: Implemented a key feature."],
            "other_content": {}
        })

        mock_generate_content.return_value = mock_llm_response
        mock_format_text.return_value = "Formatted Summary"
        mock_validate_content.return_value = {"is_valid": True}

        generated_content = self.agent.run(input_data)

        mock_generate_content.assert_called_once() # Ensure LLM was called
        mock_format_text.assert_called_once() # Ensure format_text was called
        mock_validate_content.assert_called_once() # Ensure validate_content was called

        self.assertIsInstance(generated_content, ContentData)
        self.assertEqual(generated_content.summary, "Highly skilled ML Engineer with experience in model development and deployment.")
        self.assertEqual(len(generated_content.experience_bullets), 2)
        self.assertIn("Developed and deployed", generated_content.experience_bullets[0])
        self.assertEqual(generated_content.skills_section, "Python, Machine Learning, SQL, Cloud Computing.")
        self.assertEqual(len(generated_content.projects), 1)
        self.assertEqual(generated_content.projects[0], "Project X: Implemented a key feature.")
        self.assertEqual(generated_content.other_content, {})

    @patch.object(MockLLM, 'generate_content')
    @patch.object(MockToolsAgent, 'format_text')
    @patch.object(MockToolsAgent, 'validate_content')
    def test_run_empty_input(self, mock_validate_content, mock_format_text, mock_generate_content): # Arguments now match the 3 method-level patches
        """Test run method with empty input data."""
        input_data = {
            "job_description_data": {},
            "relevant_experiences": [],
            "research_results": {},
            "user_cv_data": {}
        }

        mock_llm_response = json.dumps({
            "summary": "",
            "experience_bullets": [],
            "skills_section": "",
            "projects": [],
            "other_content": {}
        })

        mock_generate_content.return_value = mock_llm_response
        mock_format_text.return_value = ""
        mock_validate_content.return_value = {"is_valid": False, "feedback": "No content generated."}

        generated_content = self.agent.run(input_data)

        mock_generate_content.assert_called_once() # Ensure LLM was called
        mock_format_text.assert_called_once() # Ensure format_text was called
        mock_validate_content.assert_called_once() # Ensure validate_content was called

        self.assertIsInstance(generated_content, ContentData)
        self.assertEqual(generated_content.summary, "")
        self.assertEqual(len(generated_content.experience_bullets), 0)
        self.assertEqual(generated_content.skills_section, "")
        self.assertEqual(len(generated_content.projects), 0)
        self.assertEqual(generated_content.other_content, {})

    @patch.object(MockLLM, 'generate_content')
    def test_generate_batch_summary(self, mock_generate_content): # Arguments now match the 1 method-level patch
        """Test generate_batch method for summary batch type."""
        input_data = {
            "job_description_data": {"skills": ["Communication"]},
            "relevant_experiences": [],
            "research_results": {},
            "user_cv_data": {"summary": ""}
        }
        batch_type = "summary"
        mock_llm_response = "A concise and effective communicator."
        mock_generate_content.return_value = mock_llm_response

        generated_content = self.agent.generate_batch(input_data, batch_type)

        mock_generate_content.assert_called_once() # Ensure LLM was called

        self.assertIsInstance(generated_content, ContentData)
        self.assertEqual(generated_content.summary, mock_llm_response)
        self.assertEqual(len(generated_content.experience_bullets), 0)

    @patch.object(MockLLM, 'generate_content')
    def test_generate_batch_experience_bullet(self, mock_generate_content): # Arguments now match the 1 method-level patch
        """Test generate_batch method for experience_bullet batch type."""
        input_data = {
            "job_description_data": {"responsibilities": ["Manage projects"]},
            "relevant_experiences": ["Led a team of 5."],
            "research_results": {},
            "user_cv_data": {}
        }
        batch_type = "experience_bullet"
        mock_llm_response = "- Successfully managed projects, leading a team of 5."
        mock_generate_content.return_value = mock_llm_response

        generated_content = self.agent.generate_batch(input_data, batch_type)

        mock_generate_content.assert_called_once() # Ensure LLM was called

        self.assertIsInstance(generated_content, ContentData)
        self.assertEqual(generated_content.summary, "")
        self.assertEqual(len(generated_content.experience_bullets), 1)
        self.assertEqual(generated_content.experience_bullets[0], mock_llm_response)

    def test_generate_batch_unsupported_type(self): # No patches here
        """Test generate_batch with an unsupported batch type."""
        input_data = {
            "job_description_data": {},
            "relevant_experiences": [],
            "research_results": {},
            "user_cv_data": {}
        }
        batch_type = "unsupported_type"

        with self.assertRaises(ValueError) as cm:
            self.agent.generate_batch(input_data, batch_type)
        
        self.assertEqual(str(cm.exception), "Unsupported batch type: unsupported_type")


if __name__ == '__main__':
    unittest.main()
