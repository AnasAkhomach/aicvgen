"""Unit tests for the CleaningAgent.

Tests the cleaning functionality for raw LLM outputs, particularly
for the "Big 10" skills generation feature.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from src.agents.cleaning_agent import CleaningAgent, get_cleaning_agent
from src.agents.agent_base import AgentExecutionContext, AgentResult
from src.models.data_models import ContentType


class TestCleaningAgent:
    """Test cases for CleaningAgent."""

    @pytest.fixture
    def cleaning_agent(self):
        """Create a CleaningAgent instance for testing."""
        return CleaningAgent()

    @pytest.fixture
    def execution_context(self):
        """Create a test execution context."""
        return AgentExecutionContext(
            session_id="test_session_123",
            item_id="test_item",
            metadata={"test": "data"}
        )

    def test_initialization(self, cleaning_agent):
        """Test that CleaningAgent initializes correctly."""
        assert cleaning_agent.name == "CleaningAgent"
        assert cleaning_agent.description == "Processes and cleans raw LLM outputs into structured data"
        assert cleaning_agent.content_type == ContentType.SKILLS
        assert cleaning_agent.input_schema.required_fields == ["raw_output", "output_type"]
        assert cleaning_agent.output_schema.required_fields == ["cleaned_data", "confidence_score"]

    @pytest.mark.asyncio
    async def test_clean_big_10_skills_numbered_list(self, cleaning_agent, execution_context):
        """Test cleaning Big 10 skills from numbered list format."""
        raw_output = """
        Here are the top 10 skills:
        1. Python Programming
        2. Machine Learning
        3. Data Analysis
        4. SQL Database Management
        5. Cloud Computing (AWS)
        6. API Development
        7. Version Control (Git)
        8. Problem Solving
        9. Team Collaboration
        10. Project Management
        """
        
        input_data = {
            "raw_output": raw_output,
            "output_type": "big_10_skills"
        }
        
        result = await cleaning_agent.run_async(input_data, execution_context)
        
        assert result.success is True
        assert "cleaned_data" in result.output_data
        
        cleaned_skills = result.output_data["cleaned_data"]
        assert len(cleaned_skills) == 10
        assert "Python Programming" in cleaned_skills
        assert "Machine Learning" in cleaned_skills
        assert "Project Management" in cleaned_skills
        assert result.confidence_score > 0.5

    @pytest.mark.asyncio
    async def test_clean_big_10_skills_bullet_points(self, cleaning_agent, execution_context):
        """Test cleaning Big 10 skills from bullet point format."""
        raw_output = """
        Top skills for this position:
        • JavaScript Development
        • React Framework
        • Node.js Backend
        • MongoDB Database
        • RESTful APIs
        • Agile Methodology
        • Unit Testing
        • DevOps Practices
        • UI/UX Design
        • Technical Documentation
        """
        
        input_data = {
            "raw_output": raw_output,
            "output_type": "big_10_skills"
        }
        
        result = await cleaning_agent.run_async(input_data, execution_context)
        
        assert result.success is True
        cleaned_skills = result.output_data["cleaned_data"]
        assert len(cleaned_skills) == 10
        assert "JavaScript Development" in cleaned_skills
        assert "React Framework" in cleaned_skills
        assert "Technical Documentation" in cleaned_skills

    @pytest.mark.asyncio
    async def test_clean_big_10_skills_json_format(self, cleaning_agent, execution_context):
        """Test cleaning Big 10 skills from JSON format."""
        raw_output = '''{
            "skills": [
                "Python Programming",
                "Data Science",
                "Machine Learning",
                "Statistical Analysis",
                "Deep Learning",
                "Natural Language Processing",
                "Computer Vision",
                "Big Data Processing",
                "Cloud Platforms",
                "Research & Development"
            ]
        }'''
        
        input_data = {
            "raw_output": raw_output,
            "output_type": "big_10_skills"
        }
        
        result = await cleaning_agent.run_async(input_data, execution_context)
        
        assert result.success is True
        cleaned_skills = result.output_data["cleaned_data"]
        assert len(cleaned_skills) == 10
        assert "Python Programming" in cleaned_skills
        assert "Natural Language Processing" in cleaned_skills

    @pytest.mark.asyncio
    async def test_clean_big_10_skills_comma_separated(self, cleaning_agent, execution_context):
        """Test cleaning Big 10 skills from comma-separated format."""
        raw_output = """
        The most relevant skills are: Java, Spring Framework, Microservices, Docker, Kubernetes, Jenkins, AWS, PostgreSQL, Redis, Elasticsearch
        """
        
        input_data = {
            "raw_output": raw_output,
            "output_type": "big_10_skills"
        }
        
        result = await cleaning_agent.run_async(input_data, execution_context)
        
        assert result.success is True
        cleaned_skills = result.output_data["cleaned_data"]
        assert len(cleaned_skills) == 10
        assert "Java" in cleaned_skills
        assert "Spring Framework" in cleaned_skills
        assert "Elasticsearch" in cleaned_skills

    @pytest.mark.asyncio
    async def test_clean_content_item(self, cleaning_agent, execution_context):
        """Test cleaning content items."""
        raw_output = """
        **This is bold text** and *this is italic*. Here's some `code` formatting.
        
        
        Multiple line breaks should be cleaned up.
        """
        
        input_data = {
            "raw_output": raw_output,
            "output_type": "content_item"
        }
        
        result = await cleaning_agent.run_async(input_data, execution_context)
        
        assert result.success is True
        cleaned_content = result.output_data["cleaned_data"]
        
        # Check that markdown formatting is removed
        assert "**" not in cleaned_content
        assert "*" not in cleaned_content
        assert "`" not in cleaned_content
        
        # Check that content is preserved
        assert "This is bold text" in cleaned_content
        assert "this is italic" in cleaned_content
        assert "code" in cleaned_content

    @pytest.mark.asyncio
    async def test_clean_generic_output(self, cleaning_agent, execution_context):
        """Test cleaning generic output."""
        raw_output = """
        This is some generic text with    multiple spaces.
        
        
        
        And excessive line breaks.
        """
        
        input_data = {
            "raw_output": raw_output,
            "output_type": "generic"
        }
        
        result = await cleaning_agent.run_async(input_data, execution_context)
        
        assert result.success is True
        cleaned_content = result.output_data["cleaned_data"]
        
        # Check that excessive whitespace is cleaned
        assert "    " not in cleaned_content
        assert "\n\n\n" not in cleaned_content

    @pytest.mark.asyncio
    async def test_confidence_score_calculation(self, cleaning_agent, execution_context):
        """Test confidence score calculation."""
        # Test with good quality input
        good_input = {
            "raw_output": "1. Python\n2. JavaScript\n3. SQL\n4. Docker\n5. AWS",
            "output_type": "big_10_skills"
        }
        
        result = await cleaning_agent.run_async(good_input, execution_context)
        good_confidence = result.confidence_score
        
        # Test with poor quality input
        poor_input = {
            "raw_output": "a\nb\nc",
            "output_type": "big_10_skills"
        }
        
        result = await cleaning_agent.run_async(poor_input, execution_context)
        poor_confidence = result.confidence_score
        
        # Good input should have higher confidence
        assert good_confidence > poor_confidence
        assert 0.0 <= good_confidence <= 1.0
        assert 0.0 <= poor_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_error_handling(self, cleaning_agent, execution_context):
        """Test error handling in cleaning agent."""
        # Test with invalid input
        input_data = {
            "raw_output": None,  # Invalid input
            "output_type": "big_10_skills"
        }
        
        result = await cleaning_agent.run_async(input_data, execution_context)
        
        # Should handle error gracefully
        assert result.success is False
        assert result.error_message is not None
        assert result.confidence_score == 0.0

    @pytest.mark.asyncio
    async def test_fallback_behavior(self, cleaning_agent, execution_context):
        """Test fallback behavior when parsing fails."""
        # Provide unstructured text that doesn't match any pattern
        raw_output = "This is just random text without any clear structure or skills listed anywhere in a recognizable format."
        
        input_data = {
            "raw_output": raw_output,
            "output_type": "big_10_skills"
        }
        
        result = await cleaning_agent.run_async(input_data, execution_context)
        
        # Should still succeed with fallback behavior
        assert result.success is True
        cleaned_skills = result.output_data["cleaned_data"]
        assert isinstance(cleaned_skills, list)
        # Fallback should return lines from the text
        assert len(cleaned_skills) >= 1

    def test_factory_function(self):
        """Test the factory function."""
        agent = get_cleaning_agent()
        assert isinstance(agent, CleaningAgent)
        assert agent.name == "CleaningAgent"

    @pytest.mark.asyncio
    async def test_raw_output_preservation(self, cleaning_agent, execution_context):
        """Test that raw output is preserved in the result."""
        raw_output = "1. Skill A\n2. Skill B\n3. Skill C"
        
        input_data = {
            "raw_output": raw_output,
            "output_type": "big_10_skills"
        }
        
        result = await cleaning_agent.run_async(input_data, execution_context)
        
        assert result.success is True
        assert result.output_data["raw_output"] == raw_output
        assert result.output_data["output_type"] == "big_10_skills"

    @pytest.mark.asyncio
    async def test_skills_limit_enforcement(self, cleaning_agent, execution_context):
        """Test that only top 10 skills are returned even if more are provided."""
        # Provide 15 skills
        raw_output = "\n".join([f"{i}. Skill {i}" for i in range(1, 16)])
        
        input_data = {
            "raw_output": raw_output,
            "output_type": "big_10_skills"
        }
        
        result = await cleaning_agent.run_async(input_data, execution_context)
        
        assert result.success is True
        cleaned_skills = result.output_data["cleaned_data"]
        
        # Should only return 10 skills
        assert len(cleaned_skills) == 10
        assert "Skill 1" in cleaned_skills
        assert "Skill 10" in cleaned_skills
        assert "Skill 15" not in cleaned_skills