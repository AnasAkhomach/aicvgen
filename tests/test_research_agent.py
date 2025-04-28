import unittest
from unittest.mock import MagicMock, patch
from research_agent import ResearchAgent
from state_manager import AgentIO, JobDescriptionData
from typing import Dict, Any
import time

# Mock class for LLM dependency
class MockLLM:
    def generate_content(self, prompt: str) -> str:
        pass

class TestResearchAgent(unittest.TestCase):

    def setUp(self):
        """Set up mock objects and ResearchAgent instance before each test."""
        self.mock_llm = MockLLM()
        self.agent = ResearchAgent(
            name="TestResearchAgent",
            description="A test research agent.",
            llm=self.mock_llm
        )

    def test_init(self):
        """Test that ResearchAgent is initialized correctly."""
        self.assertEqual(self.agent.name, "TestResearchAgent")
        self.assertEqual(self.agent.description, "A test research agent.")
        self.assertEqual(self.agent.llm, self.mock_llm)
        self.assertIsInstance(self.agent.input_schema, AgentIO)
        self.assertIsInstance(self.agent.output_schema, AgentIO)
        # Check a few aspects of the schema structure based on research_agent.py
        self.assertIn('job_description_data', self.agent.input_schema['input'])
        self.assertEqual(self.agent.output_schema['output'], Dict[str, Any]) # Output is a Dict

    @patch('research_agent.time.sleep', return_value=None)
    def test_run_with_job_description(self, mock_sleep):
        """Test run method with a typical job description dictionary."""
        job_description_data = {
            "skills": ["Python", "Data Analysis"],
            "responsibilities": ["Analyze data", "Report findings"],
            "industry_terms": ["Machine Learning", "Statistics"],
            "company_values": ["Innovation", "Teamwork"],
            "experience_level": "Senior"
        }

        input_data = {
            "job_description_data": job_description_data
        }

        research_results = self.agent.run(input_data)

        mock_sleep.assert_called() # Ensure sleep was called for simulated delay

        self.assertIsInstance(research_results, Dict)
        self.assertIn("company_info", research_results)
        self.assertIn("industry_trends", research_results)

        company_info = research_results["company_info"]
        self.assertIn("query", company_info)
        self.assertIn("results", company_info)
        self.assertIsInstance(company_info["results"], List)

        industry_trends = research_results["industry_trends"]
        self.assertIn("query", industry_trends)
        self.assertIn("results", industry_trends)
        self.assertIsInstance(industry_trends["results"], List)

        # Check if queries were formed using input data
        self.assertIn("company values", company_info["query"])
        self.assertIn("Current trends in", industry_trends["query"])
        self.assertIn("Python", industry_trends["query"])


    @patch('research_agent.time.sleep', return_value=None)
    def test_run_with_empty_job_description(self, mock_sleep):
        """Test run method with an empty job description dictionary."""
        job_description_data = {}

        input_data = {
            "job_description_data": job_description_data
        }

        research_results = self.agent.run(input_data)

        # No sleep should be called as no queries are formed
        mock_sleep.assert_not_called()

        self.assertIsInstance(research_results, Dict)
        self.assertEqual(len(research_results), 0) # Expect empty dictionary

if __name__ == '__main__':
    unittest.main()
