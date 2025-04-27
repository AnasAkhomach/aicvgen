import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import the class from llm.py
try:
    from llm import LLM
except ImportError as e:
    print(f"Error importing LLM: {e}")
    # Fallback if the direct import fails (e.g., due to path issues)
    LLM = None

# Mock the google.generativeai module
# This prevents actual API calls during tests
mock_genai = MagicMock()

# Create a mock GenerativeModel instance
mock_model_instance = MagicMock()
mock_response = MagicMock()
mock_response.text = "Mocked response text"
mock_model_instance.generate_content.return_value = mock_response

# Configure the mock GenerativeModel class to return the mock instance
mock_genai.GenerativeModel.return_value = mock_model_instance


class TestLLM(unittest.TestCase):

    @patch.dict(sys.modules, {'google.generativeai': mock_genai})
    def test_llm_initialization_and_generate(self):
        """Tests LLM initialization and content generation with mocks."""
        if LLM is None:
            self.skipTest("LLM class could not be imported.")

        # Reset mocks before test
        mock_genai.reset_mock()
        mock_genai.configure.reset_mock()
        mock_genai.GenerativeModel.reset_mock()
        mock_model_instance.generate_content.reset_mock()

        # Instantiate the LLM class
        llm_instance = LLM()

        # Assert that genai.configure was called (implicitly in __init__)
        mock_genai.configure.assert_called_once()

        # Assert that genai.GenerativeModel was called with 'gemini-2.0-flash'
        mock_genai.GenerativeModel.assert_called_once_with('gemini-2.0-flash')

        # Call the generate_content method
        prompt_text = "Test prompt"
        response_text = llm_instance.generate_content(prompt_text)

        # Assert that the model's generate_content was called with the prompt
        mock_model_instance.generate_content.assert_called_once_with(prompt_text)

        # Assert that the returned text is the mocked response text
        self.assertEqual(response_text, "Mocked response text")

if __name__ == '__main__':
    unittest.main()
