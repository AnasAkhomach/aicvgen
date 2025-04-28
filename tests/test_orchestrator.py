import unittest
from unittest.mock import MagicMock, call
from orchestrator import Orchestrator, WorkflowState, ContentData, ExperienceEntry, VectorStoreConfig
from state_manager import CVData # Import CVData

class TestOrchestrator(unittest.TestCase):

    def setUp(self):
        """Set up mock objects and Orchestrator instance before each test."""
        self.mock_parser_agent = MagicMock()
        self.mock_template_renderer = MagicMock()
        self.mock_vector_store_agent = MagicMock()
        self.mock_content_writer_agent = MagicMock() # Mock missing agent
        self.mock_research_agent = MagicMock() # Mock missing agent
        self.mock_cv_analyzer_agent = MagicMock() # Mock missing agent
        self.mock_tools_agent = MagicMock() # Mock missing agent
        self.mock_formatter_agent = MagicMock() # Mock missing agent
        self.mock_quality_assurance_agent = MagicMock() # Mock missing agent
        self.mock_llm = MagicMock() # Mock LLM

        # Mock the uuid4 to return a fixed value for predictable testing
        self.mock_uuid4 = MagicMock(return_value="mock-workflow-id")
        # Patch uuid4 in the orchestrator module for this test case
        unittest.mock.patch('orchestrator.uuid4', self.mock_uuid4).start()
        self.addCleanup(unittest.mock.patch.stopall) # Stop all patches after the test

        self.orchestrator = Orchestrator(
            parser_agent=self.mock_parser_agent,
            template_renderer=self.mock_template_renderer,
            vector_store_agent=self.mock_vector_store_agent,
            content_writer_agent=self.mock_content_writer_agent, # Add mock
            research_agent=self.mock_research_agent, # Add mock
            cv_analyzer_agent=self.mock_cv_analyzer_agent, # Add mock
            tools_agent=self.mock_tools_agent, # Add mock
            formatter_agent=self.mock_formatter_agent, # Add mock
            quality_assurance_agent=self.mock_quality_assurance_agent, # Add mock
            llm=self.mock_llm # Add mock
        )

    def test_init(self):
        """Test that Orchestrator is initialized correctly."""
        self.assertEqual(self.orchestrator.parser_agent, self.mock_parser_agent)
        self.assertEqual(self.orchestrator.template_renderer, self.mock_template_renderer)
        self.assertEqual(self.orchestrator.vector_store_agent, self.mock_vector_store_agent)
        # Assert that all other agents and LLM are also assigned
        self.assertEqual(self.orchestrator.content_writer_agent, self.mock_content_writer_agent)
        self.assertEqual(self.orchestrator.research_agent, self.mock_research_agent)
        self.assertEqual(self.orchestrator.cv_analyzer_agent, self.mock_cv_analyzer_agent)
        self.assertEqual(self.orchestrator.tools_agent, self.mock_tools_agent)
        self.assertEqual(self.orchestrator.formatter_agent, self.mock_formatter_agent)
        self.assertEqual(self.orchestrator.quality_assurance_agent, self.mock_quality_assurance_agent)
        self.assertEqual(self.orchestrator.llm, self.mock_llm)

        # The following assertions were moved from the original test_init
        # Note: These are properties of the vector_store_agent, not the orchestrator directly.
        # You might want to add tests for the VectorStoreAgent class separately if they don't exist.
        # self.assertIsInstance(self.orchestrator.vector_store_agent.config, VectorStoreConfig)
        # self.assertEqual(self.orchestrator.vector_store_agent.config.dimension, 768)
        # self.assertEqual(self.orchestrator.vector_store_agent.config.index_type, "IndexFlatL2")


    def test_run_workflow(self):
        """Test the main run_workflow method."""
        job_description = "Software Engineer with Python and experience in testing."
        # Use CVData object for user_cv
        user_cv_data = CVData(
            raw_text="My CV content.",
            experiences=["Experience 1", "Experience 2"],
            summary="",
            skills=[],
            education=[],
            projects=[]
        )
        
        # Define the expected return value of the parser agent
        mock_job_description_data = {
            "experience_level": "Mid-Level",
            "skills": ["Python", "testing", "AWS"],
             "raw_text": "Software Engineer with Python and experience in testing.", # Added raw_text for completeness
             "responsibilities": [],
             "industry_terms": [],
             "company_values": []
        }
        self.mock_parser_agent.run.return_value = mock_job_description_data

        # Define the expected return value of the CV Analyzer Agent
        mock_extracted_cv_data = {
            "summary": "Analyzed Summary",
            "experiences": ["Analyzed Experience 1", "Analyzed Experience 2"], # These should be added to the vector store
            "skills": ["Analyzed Skill 1"],
            "education": [],
            "projects": []
        }
        self.mock_cv_analyzer_agent.run.return_value = mock_extracted_cv_data

        # Define the expected return value of the Vector Store Agent Search
        mock_search_results = [MagicMock(text="Relevant Experience 1"), MagicMock(text="Relevant Experience 2")]
        self.mock_vector_store_agent.search.return_value = mock_search_results

        # Define the expected return value of the Research Agent
        mock_research_results = {"company_info": "..."}
        self.mock_research_agent.run.return_value = mock_research_results

        # Define the expected return value of the Content Writer Agent
        mock_generated_content = ContentData(
            summary="Generated Summary",
            experience_bullets=["Generated Bullet 1"],
            skills_section="Generated Skills",
            projects=[],
            other_content={}
        )
        self.mock_content_writer_agent.run.return_value = mock_generated_content
        
        # Define the expected return value of the Formatter Agent
        mock_formatted_cv = "Formatted CV Content"
        self.mock_formatter_agent.run.return_value = mock_formatted_cv

        # Define the expected return value of the Quality Assurance Agent
        mock_quality_results = {"is_quality_ok": True, "feedback": "OK", "suggestions": []}
        self.mock_quality_assurance_agent.run.return_value = mock_quality_results

        # Define the expected return value of the Template Renderer (This is the final step after human review simulation)
        mock_rendered_cv = "Final Rendered CV"
        self.mock_template_renderer.run.return_value = mock_rendered_cv


        # Run the workflow
        # The workflow should now run through all steps due to the simulated human review approval in the node.
        rendered_cv = self.orchestrator.run_workflow(job_description, user_cv_data)

        # Assertions (verify agent calls and final output)

        # 1. Check if parser_agent.run was called correctly
        self.mock_parser_agent.run.assert_called_once_with({"job_description": job_description})

        # 2. Check if cv_analyzer_agent.run was called correctly
        self.mock_cv_analyzer_agent.run.assert_called_once_with({"user_cv": user_cv_data, "job_description": mock_job_description_data})

        # 3. Check if vector_store_agent.run_add_item was called for each extracted experience
        expected_vector_store_add_calls = [
             call(ExperienceEntry(text="Analyzed Experience 1"), text="Analyzed Experience 1"),
             call(ExperienceEntry(text="Analyzed Experience 2"), text="Analyzed Experience 2"),
        ]
        self.mock_vector_store_agent.run_add_item.assert_has_calls(expected_vector_store_add_calls, any_order=False)
        self.assertEqual(self.mock_vector_store_agent.run_add_item.call_count, len(mock_extracted_cv_data['experiences']))

        # 4. Check if vector_store_agent.search was called correctly
        # The search query is constructed within the orchestrator based on parsed job description
        # We need to check if search was called with a string containing parts of the job description
        self.mock_vector_store_agent.search.assert_called_once()
        search_call_arg = self.mock_vector_store_agent.search.call_args[0][0] # Get the first argument of the first call
        self.assertIsInstance(search_call_arg, str)
        self.assertIn("Python", search_call_arg)
        self.assertIn("testing", search_call_arg)

        # 5. Check if research_agent.run was called correctly
        self.mock_research_agent.run.assert_called_once_with({"job_description_data": mock_job_description_data})

        # 6. Check if content_writer_agent.run was called correctly
        # This agent receives input constructed from previous steps' outputs
        expected_content_writer_input = {
            "job_description_data": mock_job_description_data,
            "relevant_experiences": [result.text for result in mock_search_results], # Should be based on search results
            "research_results": mock_research_results,
            "user_cv_data": mock_extracted_cv_data # Should be based on CV analyzer results
        }
        self.mock_content_writer_agent.run.assert_called_once_with(expected_content_writer_input)

        # 7. Check if formatter_agent.run was called correctly
        self.mock_formatter_agent.run.assert_called_once_with({
            "content_data": mock_generated_content,
            "format_specifications": {"template_type": "markdown", "style": "professional"}
        })

        # 8. Check if quality_assurance_agent.run was called correctly
        self.mock_quality_assurance_agent.run.assert_called_once_with({
             "formatted_cv_text": mock_formatted_cv,
             "job_description": mock_job_description_data
        })

        # 9. Check if template_renderer.run was called correctly (after simulated human review approval)
        # The human_review_node currently just passes the formatted_cv_text to render_cv if approved
        self.mock_template_renderer.run.assert_called_once_with(mock_formatted_cv) # Should receive the formatted CV text

        # 10. Check the final returned value from run_workflow
        # The orchestrator returns the value from the last step (render_cv)
        self.assertEqual(rendered_cv, mock_rendered_cv)

        # Note: Testing the exact state changes within the workflow_state dictionary during the run_workflow method
        # would require more complex mocking or a different testing approach (e.g., examining side effects).
        # The current tests verify the interactions with dependent agents and the final output.

if __name__ == '__main__':
    unittest.main()
