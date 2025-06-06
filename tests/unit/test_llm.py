import unittest
from unittest.mock import patch, MagicMock, call
import sys
import os

# Add the project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the LLM class
from src.services.llm import LLM


class TestLLM(unittest.TestCase):
    @patch("llm.genai")
    def test_llm_initialization_and_generate(self, mock_genai):
        """Test initialization and basic content generation."""
        # Set up the mock response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Mocked response text"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # Create an instance of LLM (which should call genai.configure)
        llm = LLM()

        # Verify configure was called
        mock_genai.configure.assert_called_once()

        # Verify GenerativeModel was instantiated correctly
        mock_genai.GenerativeModel.assert_called_with("gemini-2.0-flash")

        # Test the generate_content method
        result = llm.generate_content("Test prompt")

        # Verify the model was called with the prompt
        mock_model.generate_content.assert_called_with("Test prompt")

        # Verify we got the expected response
        self.assertEqual(result, "Mocked response text")

    @patch("llm.genai")
    def test_llm_generate_content_exception(self, mock_genai):
        """Test exception handling in generate_content."""
        # Set up the mock to raise an exception
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("Test exception")
        mock_genai.GenerativeModel.return_value = mock_model

        # Create an instance of LLM
        llm = LLM()

        # Call generate_content and verify it returns an error message instead of raising an exception
        result = llm.generate_content("Test prompt")
        self.assertTrue(result.startswith("The AI model encountered an issue:"))
        self.assertIn("Exception", result)

    @patch("llm.genai")
    def test_llm_generate_content_unexpected_data(self, mock_genai):
        """Test handling of unexpected response formats."""
        # Set up mock to return a response with None for text
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = None
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        # Create an instance of LLM
        llm = LLM()

        # Call generate_content
        result = llm.generate_content("Test prompt")

        # Verify we get an error message back
        self.assertEqual(result, "LLM returned an empty or invalid response")


if __name__ == "__main__":
    unittest.main()
