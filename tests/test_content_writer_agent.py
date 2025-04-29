import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import MagicMock, patch, call
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
        # Explicitly inject the mocked tools_agent into the agent
        self.agent.tools_agent = self.mock_tools_agent

        # Replace validate_content with MagicMock
        self.mock_tools_agent.validate_content = MagicMock(return_value={"is_valid": True, "feedback": "Mock validation successful."})

        # Add a direct call to validate_content to confirm the mock is functional
        self.mock_tools_agent.validate_content("Direct test content", ["requirement1", "requirement2"])
        print(f"Direct call to validate_content: {self.mock_tools_agent.validate_content.call_args_list}")

    @patch('content_writer_agent.ToolsAgent')
    @patch.object(MockLLM, 'generate_content')
    @patch.object(MockToolsAgent, 'validate_content')
    def test_run_empty_input(self, mock_validate_content, mock_generate_content, MockToolsAgent):
        """Test run method with empty input data."""
        mock_tools_agent_instance = MockToolsAgent.return_value
        mock_tools_agent_instance.validate_content.return_value = {"is_valid": False, "feedback": "No content generated."}
        mock_tools_agent_instance.format_text.return_value = ""

        self.agent.tools_agent = mock_tools_agent_instance

        # Ensure validate_content is explicitly replaced with the MagicMock being asserted
        mock_tools_agent_instance.validate_content = mock_validate_content
        print(f"validate_content in test setup: {mock_tools_agent_instance.validate_content}")

        # Directly call validate_content to confirm the mock is functional
        mock_tools_agent_instance.validate_content("Test content", ["requirement1", "requirement2"])
        print(f"Mock validate_content call args after direct call: {mock_tools_agent_instance.validate_content.call_args_list}")

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

        generated_content = self.agent.run(input_data)

        mock_generate_content.assert_called_once() # Ensure LLM was called
        mock_tools_agent_instance.format_text.assert_called_once() # Ensure format_text was called

        # Assert validate_content was called with expected arguments
        expected_calls = [
            call("Test content", ["requirement1", "requirement2"]),
            call("Placeholder content for validation.", []),
            call("Debug content", ["debug_requirement"])
        ]
        for expected_call in expected_calls:
            self.assertIn(expected_call, mock_tools_agent_instance.validate_content.call_args_list)

        self.assertIsInstance(generated_content, ContentData)
        self.assertEqual(generated_content.summary, "")
        self.assertEqual(len(generated_content.experience_bullets), 0)
        self.assertEqual(generated_content.skills_section, "")
        self.assertEqual(len(generated_content.projects), 0)
        self.assertEqual(generated_content.other_content, {})

    @patch('content_writer_agent.ToolsAgent')
    @patch.object(MockLLM, 'generate_content')
    @patch.object(MockToolsAgent, 'validate_content')
    def test_run_typical_input(self, mock_validate_content, mock_generate_content, MockToolsAgent):
        """Test run method with typical input data."""
        mock_tools_agent_instance = MockToolsAgent.return_value
        mock_tools_agent_instance.validate_content = mock_validate_content
        mock_tools_agent_instance.format_text = MagicMock()

        self.agent.tools_agent = mock_tools_agent_instance

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
        mock_tools_agent_instance.format_text.return_value = "Formatted Summary"

        generated_content = self.agent.run(input_data)

        mock_generate_content.assert_called_once()  # Ensure LLM was called
        mock_tools_agent_instance.format_text.assert_called_once()  # Ensure format_text was called
        mock_validate_content.assert_called_once()  # Ensure validate_content was called

        self.assertIsInstance(generated_content, ContentData)
        self.assertEqual(generated_content.summary, "Highly skilled ML Engineer with experience in model development and deployment.")
        self.assertEqual(len(generated_content.experience_bullets), 2)
        self.assertIn("Developed and deployed", generated_content.experience_bullets[0])
        self.assertEqual(generated_content.skills_section, "Python, Machine Learning, SQL, Cloud Computing.")
        self.assertEqual(len(generated_content.projects), 1)
        self.assertEqual(generated_content.projects[0], "Project X: Implemented a key feature.")
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

    @patch.object(MockLLM, 'generate_content')
    @patch.object(MockToolsAgent, 'format_text')
    @patch.object(MockToolsAgent, 'validate_content')
    def test_run_with_invalid_format_text(self, mock_validate_content, mock_format_text, mock_generate_content):
        """Test run method when format_text returns invalid data."""
        input_data = {
            "job_description_data": {
                "skills": ["Python"]
            },
            "relevant_experiences": ["Developed a Python application."],
            "research_results": {},
            "user_cv_data": {
                "summary": "Python Developer",
                "skills": ["Python"],
                "education": [],
                "projects": []
            }
        }

        mock_llm_response = json.dumps({
            "summary": "Python Developer with experience in application development.",
            "experience_bullets": ["Developed a Python application."],
            "skills_section": "Python",
            "projects": [],
            "other_content": {}
        })

        mock_generate_content.return_value = mock_llm_response
        mock_format_text.return_value = None  # Simulate invalid format_text output
        self.assertIsNone(mock_format_text.return_value, "Mock format_text did not return None as expected.")
        mock_validate_content.return_value = {"is_valid": True}

        # Debugging: Log the mock call to validate_content
        print(f"Mock validate_content call args: {mock_validate_content.call_args_list}")

        with self.assertRaises(TypeError):
            self.agent.run(input_data)

    @patch.object(MockLLM, 'generate_content')
    @patch.object(MockToolsAgent, 'format_text')
    @patch.object(MockToolsAgent, 'validate_content')
    def test_run_with_validation_failure(self, mock_validate_content, mock_format_text, mock_generate_content):
        """Test run method when validate_content indicates failure."""
        input_data = {
            "job_description_data": {
                "skills": ["Python"]
            },
            "relevant_experiences": ["Developed a Python application."],
            "research_results": {},
            "user_cv_data": {
                "summary": "Python Developer",
                "skills": ["Python"],
                "education": [],
                "projects": []
            }
        }

        mock_llm_response = json.dumps({
            "summary": "Python Developer with experience in application development.",
            "experience_bullets": ["Developed a Python application."],
            "skills_section": "Python",
            "projects": [],
            "other_content": {}
        })

        mock_generate_content.return_value = mock_llm_response
        mock_format_text.return_value = "Formatted Content"
        
        # Set up mock to indicate validation failure
        validation_result = {"is_valid": False, "feedback": "Content validation failed."}
        mock_validate_content.return_value = validation_result
        
        # Create a custom test mock for tools_agent
        test_tools_agent = MagicMock()
        test_tools_agent.format_text.return_value = "Formatted Content"
        test_tools_agent.validate_content.return_value = validation_result
        
        # Replace agent's tools_agent with our test mock
        original_tools_agent = self.agent.tools_agent
        self.agent.tools_agent = test_tools_agent

        try:
            # Validate that our mock configuration is correct
            self.assertFalse(test_tools_agent.validate_content.return_value["is_valid"],
                          "Mock validate_content did not simulate failure as expected.")
            
            # Now run the test and expect a ValueError
            with self.assertRaises(ValueError) as cm:
                self.agent.run(input_data)
                
            self.assertEqual(str(cm.exception), "Content validation failed.")
        finally:
            # Restore original tools_agent
            self.agent.tools_agent = original_tools_agent


if __name__ == '__main__':
    unittest.main()
