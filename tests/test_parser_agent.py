import unittest
from unittest.mock import MagicMock
from parser_agent import ParserAgent
from state_manager import JobDescriptionData, AgentIO

class TestParserAgent(unittest.TestCase):

    def setUp(self):
        """Set up mock objects and ParserAgent instance before each test."""
        self.mock_llm = MagicMock()
        # The original ParserAgent init uses `model` which seems to be intended as the llm object
        # We will pass the mock_llm as the `llm` argument.
        self.parser_agent = ParserAgent(name="TestParserAgent", description="Test Description", llm=self.mock_llm)

    def test_init(self):
        """Test that ParserAgent is initialized correctly."""
        self.assertEqual(self.parser_agent.name, "Parser Agent") # Note: name is hardcoded in __init__
        self.assertEqual(self.parser_agent.description, "Parses job descriptions and extracts key information.") # Note: description is hardcoded in __init__
        self.assertEqual(self.parser_agent.llm, self.mock_llm)
        self.assertIsInstance(self.parser_agent.input_schema, AgentIO)
        self.assertIsInstance(self.parser_agent.output_schema, AgentIO)
        self.assertEqual(self.parser_agent.input_schema["input"], {"job_description": str})
        self.assertEqual(self.parser_agent.output_schema["output"], JobDescriptionData)

    def test_run(self):
        """Test the run method of ParserAgent."""
        job_description_input = "This is a sample job description for a Python Developer."
        input_data = {"job_description": job_description_input}

        # Since the current run method doesn't use the LLM, we don't need to mock LLM's return value
        # We are testing the logic within the run method itself.

        parsed_data = self.parser_agent.run(input_data)

        # Assert that the returned object is a JobDescriptionData instance
        self.assertIsInstance(parsed_data, JobDescriptionData)

        # Assert that the raw_text is correctly captured
        self.assertEqual(parsed_data["raw_text"], job_description_input)

        # Assert that the hardcoded values are returned (based on the current implementation)
        self.assertEqual(parsed_data["skills"], ["Python"])
        self.assertEqual(parsed_data["experience_level"], "Mid-Level")
        self.assertEqual(parsed_data["responsibilities"], [])
        self.assertEqual(parsed_data["industry_terms"], [])
        self.assertEqual(parsed_data["company_values"], [])

if __name__ == '__main__':
    unittest.main()
