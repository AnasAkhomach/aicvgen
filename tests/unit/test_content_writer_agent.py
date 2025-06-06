import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest
from unittest.mock import MagicMock, patch, call
from src.agents.content_writer_agent import ContentWriterAgent
from src.core.state_manager import (
    ContentData,
    AgentIO,
    JobDescriptionData,
    CVData,
    StructuredCV,
    Section,
    Item,
    ItemStatus,
    ItemType,
)
from typing import Dict, List, Any
import json


# Mock classes for dependencies
class MockLLM:
    def generate_content(self, prompt: str) -> str:
        return "Mock generated content"


class MockToolsAgent:
    def format_text(self, text: str, format_type: str) -> str:
        return "Mock formatted text"

    def validate_content(self, content: str, requirements: List[str]) -> Dict[str, Any]:
        return {"is_valid": True, "feedback": "Mock validation successful."}


class TestContentWriterAgent(unittest.TestCase):

    def setUp(self):
        """Set up mock objects and ContentWriterAgent instance before each test."""
        self.mock_llm = MockLLM()
        self.mock_tools_agent = MockToolsAgent()
        self.agent = ContentWriterAgent(
            name="TestContentWriterAgent",
            description="A test content writer agent.",
            llm=self.mock_llm,
            tools_agent=self.mock_tools_agent,
        )
        # Explicitly inject the mocked tools_agent into the agent
        self.agent.tools_agent = self.mock_tools_agent

    def test_agent_initialization(self):
        """Test that the ContentWriterAgent initializes correctly."""
        self.assertEqual(self.agent.name, "TestContentWriterAgent")
        self.assertEqual(self.agent.description, "A test content writer agent.")
        self.assertEqual(self.agent.llm, self.mock_llm)
        self.assertEqual(self.agent.tools_agent, self.mock_tools_agent)

    def test_generate_cv_content(self):
        """Simple test to verify the agent can generate content for a StructuredCV."""
        # Create a job description data
        job_description_data = {
            "skills": ["Python", "Machine Learning"],
            "responsibilities": ["Develop models", "Deploy solutions"],
            "raw_text": "Looking for an ML Engineer with Python skills to develop models and deploy solutions.",
        }

        # Create a basic structured CV to pass to the agent
        structured_cv = StructuredCV()
        structured_cv.metadata["main_jd_text"] = job_description_data["raw_text"]

        # Add Executive Summary section
        summary_section = Section(
            name="Executive Summary", content_type="DYNAMIC", order=0
        )
        summary_section.items.append(
            Item(
                content="Data scientist with skills in ML and Python.",
                status=ItemStatus.TO_REGENERATE,
                item_type=ItemType.SUMMARY_PARAGRAPH,
            )
        )
        structured_cv.sections.append(summary_section)

        # Add Key Qualifications section
        key_quals_section = Section(
            name="Key Qualifications", content_type="DYNAMIC", order=1
        )
        key_quals_section.items.append(
            Item(
                content="Python",
                status=ItemStatus.TO_REGENERATE,
                item_type=ItemType.KEY_QUAL,
            )
        )
        structured_cv.sections.append(key_quals_section)

        # Create research results
        research_results = {
            "industry trends": "AI is growing",
            "job_requirements_analysis": {
                "core_technical_skills": ["Python", "ML"],
                "soft_skills": ["Communication"],
            },
        }

        # Run the agent with the valid input data structure
        input_data = {
            "structured_cv": structured_cv,
            "job_description_data": job_description_data,
            "research_results": research_results,
            "regenerate_item_ids": [],
        }

        # Mock the _generate_all_dynamic_content method to avoid the actual generation
        with patch.object(self.agent, "_generate_all_dynamic_content") as mock_generate:
            # Set the mock to return the input structured_cv
            mock_generate.return_value = structured_cv

            # Run the agent
            result = self.agent.run(input_data)

            # Verify the mock was called with the right arguments
            mock_generate.assert_called_once()

            # Verify we get back a StructuredCV object
            self.assertIsInstance(result, StructuredCV)

            # Verify it has the expected sections
            self.assertEqual(len(result.sections), 2)
            self.assertEqual(result.sections[0].name, "Executive Summary")
            self.assertEqual(result.sections[1].name, "Key Qualifications")


if __name__ == "__main__":
    unittest.main()
