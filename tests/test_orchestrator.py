import unittest
from unittest.mock import MagicMock, call
from orchestrator import Orchestrator, WorkflowState, ContentData, ExperienceEntry, VectorStoreConfig

class TestOrchestrator(unittest.TestCase):

    def setUp(self):
        """Set up mock objects and Orchestrator instance before each test."""
        self.mock_parser_agent = MagicMock()
        self.mock_template_renderer = MagicMock()
        self.mock_vector_store_agent = MagicMock()
        
        # Mock the uuid4 to return a fixed value for predictable testing
        self.mock_uuid4 = MagicMock(return_value="mock-workflow-id")
        # Patch uuid4 in the orchestrator module for this test case
        unittest.mock.patch('orchestrator.uuid4', self.mock_uuid4).start()
        self.addCleanup(unittest.mock.patch.stopall) # Stop all patches after the test

        self.orchestrator = Orchestrator(
            parser_agent=self.mock_parser_agent,
            template_renderer=self.mock_template_renderer,
            vector_store_agent=self.mock_vector_store_agent
        )

    def test_init(self):
        """Test that Orchestrator is initialized correctly."""
        self.assertEqual(self.orchestrator.parser_agent, self.mock_parser_agent)
        self.assertEqual(self.orchestrator.template_renderer, self.mock_template_renderer)
        self.assertEqual(self.orchestrator.vector_store_agent, self.mock_vector_store_agent)
        self.assertEqual(self.orchestrator.workflow_id, "mock-workflow-id")
        self.assertIsInstance(self.orchestrator.vector_store_agent.config, VectorStoreConfig)
        self.assertEqual(self.orchestrator.vector_store_agent.config.dimension, 768)
        self.assertEqual(self.orchestrator.vector_store_agent.config.index_type, "IndexFlatL2")

    def test_run_workflow(self):
        """Test the main run_workflow method."""
        job_description = "Software Engineer with Python and experience in testing."
        user_cv = "My CV content."
        user_experiences = ["Experience 1", "Experience 2"]
        
        # Define the expected return value of the parser agent
        mock_job_description_data = {
            "experience_level": "Mid-Level",
            "skills": ["Python", "testing", "AWS"]
        }
        self.mock_parser_agent.run.return_value = mock_job_description_data

        # Define the expected return value of the template renderer
        mock_rendered_cv = "<html><body>Tailored CV</body></html>"
        self.mock_template_renderer.run.return_value = mock_rendered_cv

        # Run the workflow
        rendered_cv = self.orchestrator.run_workflow(job_description, user_cv, user_experiences)

        # Assertions

        # Check if parser_agent.run was called correctly
        self.mock_parser_agent.run.assert_called_once_with({"job_description": job_description})

        # Check if vector_store_agent.run_add_item was called for each experience
        expected_vector_store_calls = [
            call(ExperienceEntry(text="Experience 1"), text="Experience 1"),
            call(ExperienceEntry(text="Experience 2"), text="Experience 2")
        ]
        self.mock_vector_store_agent.run_add_item.assert_has_calls(expected_vector_store_calls, any_order=False)
        self.assertEqual(self.mock_vector_store_agent.run_add_item.call_count, len(user_experiences))

        # Check if template_renderer.run was called with the correct generated content
        # We need to construct the expected ContentData object based on the mock parser output
        expected_generated_content = ContentData(
            summary=f"Tailored summary for: {mock_job_description_data['experience_level']}",
            experience_bullets=[f"Tailored bullet for {skill}" for skill in mock_job_description_data["skills"]],
            skills_section=f"Skills section tailored to: {', '.join(mock_job_description_data['skills'])}",
            projects=["Project 1", "Project 2"],
            other_content={}
        )
        self.mock_template_renderer.run.assert_called_once_with(expected_generated_content)

        # Check the final returned value
        self.assertEqual(rendered_cv, mock_rendered_cv)

        # Note: Testing the exact state changes within the workflow_state dictionary during the run_workflow method
        # would require more complex mocking or a different testing approach (e.g., examining side effects).
        # The current tests verify the interactions with dependent agents and the final output.

if __name__ == '__main__':
    unittest.main()
