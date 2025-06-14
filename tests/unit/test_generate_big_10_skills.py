"""Unit tests for the generate_big_10_skills functionality."""

import pytest
from unittest.mock import Mock, patch
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent


class TestGenerateBig10Skills:
    """Test cases for the Big 10 skills generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = EnhancedContentWriterAgent()
        self.sample_job_description = """
        We are looking for a Senior Software Engineer with expertise in Python, 
        React, AWS, Docker, and Kubernetes. The ideal candidate should have 
        experience with microservices architecture, CI/CD pipelines, and 
        database design. Strong communication skills and leadership experience 
        are essential.
        """
        self.sample_talents = "Python developer with 5 years experience in web development"
    
    @patch('src.agents.enhanced_content_writer.EnhancedContentWriterAgent._load_prompt_template')
    @patch('src.agents.enhanced_content_writer.get_llm_service')
    def test_generate_big_10_skills_success(self, mock_llm_service, mock_load_prompt):
        """Test successful generation of Big 10 skills."""
        # Mock the prompt template loading
        mock_load_prompt.return_value = "Generate skills based on: {job_description} and {my_talents}"
        
        # Mock the LLM service
        mock_llm = Mock()
        mock_llm_service.return_value = mock_llm
        mock_llm.generate_content.return_value = """
        1. Python Programming
        2. React Development
        3. AWS Cloud Services
        4. Docker Containerization
        5. Kubernetes Orchestration
        6. Microservices Architecture
        7. CI/CD Pipeline Management
        8. Database Design
        9. Leadership and Team Management
        10. Technical Communication
        """
        
        # Execute the method
        result = self.agent.generate_big_10_skills(
            job_description=self.sample_job_description,
            my_talents=self.sample_talents
        )
        
        # Assertions
        assert result["success"] is True
        assert len(result["skills"]) == 10
        assert "Python Programming" in result["skills"]
        assert "React Development" in result["skills"]
        assert result["raw_llm_output"] is not None
        assert "formatted_content" in result
        
        # Verify LLM was called
        mock_llm.generate_content.assert_called_once()
    
    @patch('src.agents.enhanced_content_writer.EnhancedContentWriterAgent._load_prompt_template')
    @patch('src.agents.enhanced_content_writer.get_llm_service')
    def test_generate_big_10_skills_llm_failure(self, mock_llm_service, mock_load_prompt):
        """Test handling of LLM failure during skills generation."""
        # Mock the prompt template loading
        mock_load_prompt.return_value = "Generate skills based on: {job_description} and {my_talents}"
        
        # Mock the LLM service to raise an exception
        mock_llm = Mock()
        mock_llm_service.return_value = mock_llm
        mock_llm.generate_content.side_effect = Exception("LLM service unavailable")
        
        # Execute the method
        result = self.agent.generate_big_10_skills(
            job_description=self.sample_job_description,
            my_talents=self.sample_talents
        )
        
        # Assertions
        assert result["success"] is False
        assert "error" in result
        assert "LLM service unavailable" in result["error"]
    
    def test_parse_big_10_skills_valid_input(self):
        """Test parsing of valid LLM output into skills list."""
        llm_output = """
        Here are the top 10 skills:
        1. Python Programming
        2. React Development
        3. AWS Cloud Services
        4. Docker Containerization
        5. Kubernetes Orchestration
        6. Microservices Architecture
        7. CI/CD Pipeline Management
        8. Database Design
        9. Leadership and Team Management
        10. Technical Communication
        """
        
        skills = self.agent._parse_big_10_skills(llm_output)
        
        assert len(skills) == 10
        assert "Python Programming" in skills
        assert "Technical Communication" in skills
    
    def test_parse_big_10_skills_insufficient_skills(self):
        """Test parsing when LLM provides fewer than 10 skills."""
        llm_output = """
        1. Python Programming
        2. React Development
        3. AWS Cloud Services
        """
        
        skills = self.agent._parse_big_10_skills(llm_output)
        
        # Should pad with generic skills to reach 10
        assert len(skills) == 10
        assert "Python Programming" in skills
        assert "Problem Solving" in skills  # Generic skill should be added
    
    def test_format_big_10_skills_display(self):
        """Test formatting of skills for display."""
        skills = ["Python Programming", "React Development", "AWS Cloud Services"]
        
        formatted = self.agent._format_big_10_skills_display(skills)
        
        assert "• Python Programming" in formatted
        assert "• React Development" in formatted
        assert "• AWS Cloud Services" in formatted