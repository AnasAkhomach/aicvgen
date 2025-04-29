import unittest
from unittest.mock import MagicMock
from parser_agent import ParserAgent
from state_manager import JobDescriptionData, AgentIO

class TestParserAgent(unittest.TestCase):

    def setUp(self):
        """Set up mock objects and ParserAgent instance before each test."""
        self.mock_llm = MagicMock()
        # Set up the mock LLM to return a properly formatted JSON response
        mock_response = '{"skills": ["Python"], "experience_level": "Mid-Level", "responsibilities": [], "industry_terms": [], "company_values": []}'
        self.mock_llm.generate_content.return_value = mock_response
        
        # Create the parser agent with the correct name and description
        self.parser_agent = ParserAgent(
            name="Parser Agent", 
            description="Parses job descriptions and extracts key information.",
            llm=self.mock_llm
        )

    def test_init(self):
        """Test that ParserAgent is initialized correctly."""
        self.assertEqual(self.parser_agent.name, "Parser Agent") # Note: name is hardcoded in __init__
        self.assertEqual(self.parser_agent.description, "Parses job descriptions and extracts key information.") # Note: description is hardcoded in __init__
        self.assertEqual(self.parser_agent.llm, self.mock_llm)
        
        # Check input_schema as a dictionary
        self.assertIn("input", self.parser_agent.input_schema)
        self.assertIn("output", self.parser_agent.input_schema) 
        self.assertIn("description", self.parser_agent.input_schema)
        
        # Check output_schema as a dictionary
        self.assertIn("input", self.parser_agent.output_schema)
        self.assertIn("output", self.parser_agent.output_schema)
        self.assertIn("description", self.parser_agent.output_schema)
        
        # Check specific content
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
        self.assertEqual(parsed_data.raw_text, job_description_input)

        # Assert that the hardcoded values are returned (based on the current implementation)
        self.assertEqual(parsed_data.skills, ["Python"])
        self.assertEqual(parsed_data.experience_level, "Mid-Level")
        self.assertEqual(parsed_data.responsibilities, [])
        self.assertEqual(parsed_data.industry_terms, [])
        self.assertEqual(parsed_data.company_values, [])

if __name__ == '__main__':
    unittest.main()
