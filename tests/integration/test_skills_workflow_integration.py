"""Integration tests for the skills generation workflow."""

import pytest
from unittest.mock import Mock, patch
from src.orchestration.cv_workflow_graph import parser_node, generate_skills_node
from src.orchestration.state import AgentState
from src.models.data_models import StructuredCV, JobDescriptionData, Section, Item, ItemStatus, ItemType


class TestSkillsWorkflowIntegration:
    """Integration tests for parser -> generate_skills workflow sequence."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_job_description = """
        We are looking for a Senior Software Engineer with expertise in Python, 
        React, AWS, Docker, and Kubernetes. The ideal candidate should have 
        experience with microservices architecture, CI/CD pipelines, and 
        database design.
        """
        
        # Create a mock structured CV with Key Qualifications section
        self.mock_structured_cv = StructuredCV(
            sections=[
                Section(
                    name="Key Qualifications",
                    items=[],  # Will be populated by generate_skills_node
                    content_type="DYNAMIC"
                ),
                Section(
                    name="Professional Experience",
                    items=[Item(content="Sample experience")],
                    content_type="DYNAMIC"
                )
            ]
        )
        
        self.initial_state = {
            "job_description_data": JobDescriptionData(raw_text=self.sample_job_description),
            "structured_cv": self.mock_structured_cv,
            "error_messages": []
        }
    
    @patch('src.agents.parser_agent.ParserAgent.run_as_node')
    def test_parser_node_execution(self, mock_parser_run):
        """Test that parser_node executes correctly and doesn't set up queue."""
        # Mock parser agent response
        mock_parser_run.return_value = {
            "structured_cv": self.mock_structured_cv,
            "job_description_data": JobDescriptionData(raw_text=self.sample_job_description)
        }
        
        # Execute parser_node
        result = parser_node(self.initial_state)
        
        # Assertions
        assert "structured_cv" in result
        assert "job_description_data" in result
        # Parser should NOT set up queue anymore
        assert "items_to_process_queue" not in result
        assert "current_section_key" not in result
        
        # Verify parser agent was called
        mock_parser_run.assert_called_once()
    
    @patch('src.agents.enhanced_content_writer.EnhancedContentWriterAgent.generate_big_10_skills')
    def test_generate_skills_node_success(self, mock_generate_skills):
        """Test successful execution of generate_skills_node."""
        # Mock the skills generation
        mock_generate_skills.return_value = {
            "success": True,
            "skills": [
                "Python Programming",
                "React Development", 
                "AWS Cloud Services",
                "Docker Containerization",
                "Kubernetes Orchestration",
                "Microservices Architecture",
                "CI/CD Pipeline Management",
                "Database Design",
                "Leadership and Team Management",
                "Technical Communication"
            ],
            "raw_llm_output": "1. Python Programming\n2. React Development...",
            "formatted_content": "• Python Programming\n• React Development..."
        }
        
        # Execute generate_skills_node
        result = generate_skills_node(self.initial_state)
        
        # Assertions
        assert "structured_cv" in result
        assert "items_to_process_queue" in result
        assert "current_section_key" in result
        assert result["current_section_key"] == "key_qualifications"
        assert result["is_initial_generation"] is True
        
        # Check that the CV was updated with skills
        updated_cv = result["structured_cv"]
        assert len(updated_cv.big_10_skills) == 10
        assert "Python Programming" in updated_cv.big_10_skills
        assert updated_cv.big_10_skills_raw_output is not None
        
        # Check that Key Qualifications section was populated
        qual_section = None
        for section in updated_cv.sections:
            if section.name.lower().replace(":", "").strip() == "key qualifications":
                qual_section = section
                break
        
        assert qual_section is not None
        assert len(qual_section.items) == 10
        assert all(item.status == ItemStatus.GENERATED for item in qual_section.items)
        assert all(item.item_type == ItemType.KEY_QUALIFICATION for item in qual_section.items)
        
        # Check that queue was set up correctly
        assert len(result["items_to_process_queue"]) == 10
        
        # Verify generate_big_10_skills was called
        mock_generate_skills.assert_called_once_with(
            job_description=self.sample_job_description,
            my_talents=""
        )
    
    @patch('src.agents.enhanced_content_writer.EnhancedContentWriterAgent.generate_big_10_skills')
    def test_generate_skills_node_failure(self, mock_generate_skills):
        """Test handling of skills generation failure."""
        # Mock the skills generation failure
        mock_generate_skills.return_value = {
            "success": False,
            "error": "LLM service unavailable"
        }
        
        # Execute generate_skills_node
        result = generate_skills_node(self.initial_state)
        
        # Assertions
        assert "error_messages" in result
        assert len(result["error_messages"]) == 1
        assert "Skills generation failed" in result["error_messages"][0]
        assert "LLM service unavailable" in result["error_messages"][0]
    
    def test_generate_skills_node_missing_key_qualifications_section(self):
        """Test handling when Key Qualifications section is missing."""
        # Create CV without Key Qualifications section
        cv_without_qual = StructuredCV(
            sections=[
                Section(
                    name="Professional Experience",
                    items=[Item(content="Sample experience")],
                    content_type="DYNAMIC"
                )
            ]
        )
        
        state_without_qual = {
            "job_description_data": JobDescriptionData(raw_text=self.sample_job_description),
            "structured_cv": cv_without_qual,
            "error_messages": []
        }
        
        with patch('src.agents.enhanced_content_writer.EnhancedContentWriterAgent.generate_big_10_skills') as mock_generate:
            mock_generate.return_value = {
                "success": True,
                "skills": ["Python Programming"],
                "raw_llm_output": "1. Python Programming",
                "formatted_content": "• Python Programming"
            }
            
            # Execute generate_skills_node
            result = generate_skills_node(state_without_qual)
            
            # Should return error about missing section
            assert "error_messages" in result
            assert "Could not find 'Key Qualifications' section" in result["error_messages"][0]
    
    @patch('src.agents.parser_agent.ParserAgent.run_as_node')
    @patch('src.agents.enhanced_content_writer.EnhancedContentWriterAgent.generate_big_10_skills')
    def test_parser_to_generate_skills_sequence(self, mock_generate_skills, mock_parser_run):
        """Test the complete parser -> generate_skills sequence."""
        # Mock parser agent response
        mock_parser_run.return_value = {
            "structured_cv": self.mock_structured_cv,
            "job_description_data": JobDescriptionData(raw_text=self.sample_job_description)
        }
        
        # Mock skills generation
        mock_generate_skills.return_value = {
            "success": True,
            "skills": ["Python Programming", "React Development"],
            "raw_llm_output": "1. Python Programming\n2. React Development",
            "formatted_content": "• Python Programming\n• React Development"
        }
        
        # Execute parser_node first
        parser_result = parser_node(self.initial_state)
        
        # Use parser result as input to generate_skills_node
        skills_input = {**self.initial_state, **parser_result}
        skills_result = generate_skills_node(skills_input)
        
        # Assertions for the complete sequence
        assert "structured_cv" in skills_result
        assert "items_to_process_queue" in skills_result
        assert "current_section_key" in skills_result
        
        # Verify the CV has been updated with skills
        final_cv = skills_result["structured_cv"]
        assert len(final_cv.big_10_skills) == 2
        assert final_cv.big_10_skills_raw_output is not None
        
        # Verify both agents were called
        mock_parser_run.assert_called_once()
        mock_generate_skills.assert_called_once()