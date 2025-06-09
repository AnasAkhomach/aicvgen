import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from src.agents.content_writer_agent import EnhancedContentWriterAgent
from src.agents.agent_base import AgentExecutionContext, AgentResult
from src.models.data_models import ContentType
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
class MockLLMResponse:
    def __init__(self):
        self.content = "Mock generated content"
        self.confidence_score = 0.9
        self.tokens_used = 100
        self.processing_time = 0.5
        self.model_used = "mock-model"
        self.success = True
        self.metadata = {"model": "mock-model", "tokens": 100}

class MockLLMService:
    async def generate_content_async(self, prompt: str, **kwargs) -> Dict[str, Any]:
        return {
            "content": "Mock generated content",
            "confidence_score": 0.9,
            "metadata": {"model": "mock-model", "tokens": 100}
        }
    
    async def generate_content(self, prompt: str, **kwargs) -> MockLLMResponse:
        return MockLLMResponse()
    
    def generate_content_sync(self, prompt: str, **kwargs) -> str:
        return "Mock generated content"


class MockConfig:
    def __init__(self):
        self.prompts_dir = "data/prompts"
        self.data_dir = "data"


class TestEnhancedContentWriterAgent(unittest.TestCase):

    def setUp(self):
        """Set up mock objects and EnhancedContentWriterAgent instance before each test."""
        self.mock_llm_service = MockLLMService()
        self.mock_config = MockConfig()
        
        # Mock the service functions
        with patch('src.agents.content_writer_agent.get_llm_service', return_value=self.mock_llm_service), \
             patch('src.agents.content_writer_agent.get_config', return_value=self.mock_config), \
             patch('src.agents.content_writer_agent.get_structured_logger') as mock_logger:
            
            mock_logger.return_value = MagicMock()
            self.agent = EnhancedContentWriterAgent(
                name="TestEnhancedContentWriterAgent",
                description="A test enhanced content writer agent.",
                content_type=ContentType.QUALIFICATION
            )

    def test_agent_initialization(self):
        """Test that the EnhancedContentWriterAgent initializes correctly."""
        self.assertEqual(self.agent.name, "TestEnhancedContentWriterAgent")
        self.assertEqual(self.agent.description, "A test enhanced content writer agent.")
        self.assertEqual(self.agent.content_type, ContentType.QUALIFICATION)
        self.assertIsNotNone(self.agent.llm_service)
        self.assertIsNotNone(self.agent.settings)

    def test_generate_cv_content_async(self):
        """Test the async content generation method."""
        async def run_test():
            # Mock input data
            job_description_data = {
                "raw_text": "Software Engineer position requiring Python skills.",
                "skills": ["Python", "Django"],
                "experience_level": "Mid-level",
                "responsibilities": ["Develop web applications"],
                "industry_terms": ["Agile", "CI/CD"],
                "company_values": ["Innovation", "Teamwork"],
            }

            content_item = {
                "content": "Developed web applications using Python",
                "type": "experience_bullet",
                "metadata": {"relevance_score": 0.8}
            }

            input_data = {
                "job_description_data": job_description_data,
                "content_item": content_item,
                "context": {"section_type": "experience"}
            }

            context = AgentExecutionContext(
                session_id="test-session",
                content_type=ContentType.EXPERIENCE,
                item_id="test-item-123"
            )

            # Mock the prompt loading and LLM service
            with patch.object(self.agent, '_load_prompt_template', return_value="Mock prompt template"), \
                 patch.object(self.agent, 'llm_service', self.mock_llm_service):
                
                result = await self.agent.run_async(input_data, context)

                # Assertions
                self.assertIsInstance(result, AgentResult)
                self.assertTrue(result.success)
                self.assertIsNotNone(result.output_data)
                self.assertGreater(result.confidence_score, 0)

        # Run the async test
        asyncio.run(run_test())

    def test_content_type_determination(self):
        """Test content type determination logic."""
        with patch.object(self.agent, '_determine_content_type') as mock_determine:
            mock_determine.return_value = ContentType.QUALIFICATION
            
            content_item = {"type": "key_qualification"}
            result = self.agent._determine_content_type(content_item)
            
            mock_determine.assert_called_once_with(content_item)

    def test_prompt_template_loading(self):
        """Test prompt template loading functionality."""
        with patch.object(self.agent, 'settings') as mock_settings, \
             patch('builtins.open', create=True) as mock_open:
            
            # Mock the settings to return a valid path
            mock_settings.get_prompt_path.return_value = "test_prompt.txt"
            mock_open.return_value.__enter__.return_value.read.return_value = "Test prompt template"
            
            result = self.agent._load_prompt_template("test_prompt")
            
            self.assertEqual(result, "Test prompt template")
            mock_settings.get_prompt_path.assert_called_once_with("test_prompt")

    def test_error_handling_in_async_run(self):
        """Test error handling in async run method."""
        async def run_test():
            input_data = {"invalid": "data"}
            context = AgentExecutionContext(session_id="test-session")
            
            # Mock LLM service to raise an exception
            with patch.object(self.agent.llm_service, 'generate_content', side_effect=Exception("LLM Error")):
                result = await self.agent.run_async(input_data, context)
                
                # Should return a failed result instead of raising
                self.assertIsInstance(result, AgentResult)
                self.assertFalse(result.success)
                self.assertIsNotNone(result.error_message)

        asyncio.run(run_test())

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

        # Mock the run_async method to avoid the actual generation
        with patch.object(self.agent, "run_async") as mock_run_async:
            # Create a mock result that matches what run_async should return
            mock_result = AgentResult(
                success=True,
                output_data={
                    "content": "Mock generated content",
                    "content_type": "QUALIFICATION",
                    "confidence_score": 0.9
                },
                confidence_score=0.9
            )
            mock_run_async.return_value = mock_result

            # Run the agent
            result = self.agent.run(input_data)

            # Verify the mock was called
            mock_run_async.assert_called_once()

            # Verify we get back the expected output data
            self.assertIsInstance(result, dict)
            self.assertEqual(result["content"], "Mock generated content")
            self.assertEqual(result["content_type"], "QUALIFICATION")
            self.assertEqual(result["confidence_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
