"""Integration tests for PDF generation workflow."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from src.orchestration.cv_workflow_graph import formatter_node, build_cv_workflow_graph
from src.orchestration.state import AgentState
from src.models.data_models import StructuredCV, Section, Item


class TestPDFWorkflowIntegration:
    """Integration tests for PDF generation in the workflow."""

    @pytest.fixture
    def sample_cv_state(self):
        """Create a complete CV state for testing."""
        cv_data = StructuredCV(
            id="integration-test-cv",
            metadata={
                "name": "Jane Smith",
                "email": "jane.smith@example.com",
                "phone": "+1-555-0456",
                "linkedin": "https://linkedin.com/in/janesmith",
            },
            sections=[
                Section(
                    name="Key Qualifications",
                    items=[
                        Item(content="Data Science"),
                        Item(content="Python Programming"),
                        Item(content="Statistical Analysis"),
                        Item(content="Machine Learning"),
                        Item(content="SQL Databases"),
                    ],
                ),
                Section(
                    name="Professional Experience",
                    items=[
                        Item(
                            content="Developed predictive models for customer behavior analysis"
                        ),
                        Item(
                            content="Implemented automated data pipelines processing 1M+ records daily"
                        ),
                        Item(
                            content="Led cross-functional team of 8 members in data migration project"
                        ),
                    ],
                ),
                Section(
                    name="Project Experience",
                    items=[
                        Item(
                            content="Built recommendation system increasing user engagement by 25%"
                        ),
                        Item(
                            content="Created real-time dashboard for executive reporting"
                        ),
                    ],
                ),
            ],
        )

        return AgentState(
            structured_cv=cv_data,
            error_messages=[],
            processing_queue=[],
            current_section="complete",
            current_item_index=0,
        )

    @patch("src.agents.formatter_agent.get_config")
    @patch("src.agents.formatter_agent.Environment")
    @patch("src.agents.formatter_agent.HTML")
    @patch("src.agents.formatter_agent.CSS")
    def test_formatter_node_integration(
        self, mock_css, mock_html, mock_env, mock_get_config, sample_cv_state
    ):
        """Test the formatter_node integration with FormatterAgent."""
        # Setup mocks
        mock_config = Mock()
        mock_config.project_root = Path("/test/project")
        mock_get_config.return_value = mock_config

        mock_template = Mock()
        mock_template.render.return_value = (
            "<html><body>Integration Test CV</body></html>"
        )
        mock_jinja_env = Mock()
        mock_jinja_env.get_template.return_value = mock_template
        mock_env.return_value = mock_jinja_env

        mock_css_instance = Mock()
        mock_css.return_value = mock_css_instance

        mock_html_instance = Mock()
        mock_html_instance.write_pdf.return_value = b"integration test pdf content"
        mock_html.return_value = mock_html_instance

        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "data" / "output"
            mock_config.project_root = Path(temp_dir)

            # Execute formatter_node
            result = formatter_node(sample_cv_state.model_dump())

            # Verify the result
            assert "final_output_path" in result
            assert "CV_integration-test-cv.pdf" in result["final_output_path"]

            # Verify state structure is maintained
            result_state = AgentState.model_validate(result)
            assert result_state.structured_cv.id == "integration-test-cv"
            assert result_state.structured_cv.metadata["name"] == "Jane Smith"
            assert len(result_state.structured_cv.sections) == 3

            # Verify template was called with correct CV data
            mock_template.render.assert_called_once()
            call_args = mock_template.render.call_args[1]
            assert "cv" in call_args
            assert call_args["cv"].id == "integration-test-cv"

    @patch("src.agents.formatter_agent.get_config")
    def test_formatter_node_error_handling(self, mock_get_config, sample_cv_state):
        """Test formatter_node error handling and state propagation."""
        # Setup mock to raise an exception
        mock_get_config.side_effect = Exception("Configuration error")

        # Execute formatter_node
        result = formatter_node(sample_cv_state.model_dump())

        # Verify error handling
        result_state = AgentState.model_validate(result)
        assert len(result_state.error_messages) > 0
        assert "PDF generation failed" in result_state.error_messages[0]
        assert "Configuration error" in result_state.error_messages[0]

        # Verify original state is preserved
        assert result_state.structured_cv.id == "integration-test-cv"

    def test_workflow_graph_includes_formatter(self):
        """Test that the workflow graph properly includes the formatter node."""
        workflow = build_cv_workflow_graph()

        # Verify formatter node is in the graph
        assert "formatter" in workflow.nodes

        # Verify the workflow structure
        # The formatter should be reachable from other nodes
        compiled_graph = workflow.compile()
        assert compiled_graph is not None

    @patch("src.agents.formatter_agent.get_config")
    @patch("src.agents.formatter_agent.Environment")
    @patch("src.agents.formatter_agent.HTML")
    @patch("src.agents.formatter_agent.CSS")
    def test_end_to_end_pdf_generation_flow(
        self, mock_css, mock_html, mock_env, mock_get_config, sample_cv_state
    ):
        """Test the complete flow from state to PDF generation."""
        # Setup comprehensive mocks
        mock_config = Mock()
        mock_config.project_root = Path("/test/project")
        mock_get_config.return_value = mock_config

        # Mock template rendering with realistic HTML
        expected_html = """
        <!DOCTYPE html>
        <html>
        <head><title>Jane Smith</title></head>
        <body>
            <h1>Jane Smith</h1>
            <section>
                <h2>Key Qualifications</h2>
                <p>Data Science | Python Programming | Statistical Analysis</p>
            </section>
            <section>
                <h2>Professional Experience</h2>
                <ul>
                    <li>Developed predictive models for customer behavior analysis</li>
                    <li>Implemented automated data pipelines processing 1M+ records daily</li>
                </ul>
            </section>
        </body>
        </html>
        """

        mock_template = Mock()
        mock_template.render.return_value = expected_html
        mock_jinja_env = Mock()
        mock_jinja_env.get_template.return_value = mock_template
        mock_env.return_value = mock_jinja_env

        mock_css_instance = Mock()
        mock_css.return_value = mock_css_instance

        mock_html_instance = Mock()
        mock_html_instance.write_pdf.return_value = b"comprehensive test pdf content"
        mock_html.return_value = mock_html_instance

        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_config.project_root = Path(temp_dir)

            # Execute the complete flow
            result = formatter_node(sample_cv_state.model_dump())

            # Verify comprehensive results
            result_state = AgentState.model_validate(result)

            # Check final output path
            assert result_state.final_output_path is not None
            assert "CV_integration-test-cv.pdf" in result_state.final_output_path

            # Verify no errors occurred
            assert len(result_state.error_messages) == 0

            # Verify all CV data is preserved
            assert result_state.structured_cv.metadata["name"] == "Jane Smith"
            assert (
                result_state.structured_cv.metadata["email"] == "jane.smith@example.com"
            )
            assert len(result_state.structured_cv.sections) == 3

            # Verify Key Qualifications section
            key_quals = next(
                s
                for s in result_state.structured_cv.sections
                if s.name == "Key Qualifications"
            )
            assert len(key_quals.items) == 5
            assert any("Data Science" in item.content for item in key_quals.items)

            # Verify Professional Experience section
            prof_exp = next(
                s
                for s in result_state.structured_cv.sections
                if s.name == "Professional Experience"
            )
            assert len(prof_exp.items) == 3
            assert any("predictive models" in item.content for item in prof_exp.items)

            # Verify template rendering was called with complete data
            mock_template.render.assert_called_once()
            call_args = mock_template.render.call_args[1]
            rendered_cv = call_args["cv"]
            assert rendered_cv.metadata["name"] == "Jane Smith"
            assert len(rendered_cv.sections) == 3

            # Verify PDF generation was called with proper parameters
            mock_html.assert_called_once_with(
                string=expected_html, base_url=str(Path(temp_dir) / "src" / "templates")
            )
            mock_html_instance.write_pdf.assert_called_once()

    def test_formatter_node_state_validation(self):
        """Test that formatter_node properly validates and handles state."""
        # Test with invalid state (missing structured_cv)
        invalid_state = {"structured_cv": None, "error_messages": []}

        result = formatter_node(invalid_state)
        result_state = AgentState.model_validate(result)

        # Should handle gracefully and add error message
        assert len(result_state.error_messages) > 0
        assert "No CV data found" in result_state.error_messages[0]
