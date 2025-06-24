import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
import asyncio
from uuid import uuid4
from unittest.mock import patch, AsyncMock
from src.orchestration.cv_workflow_graph import CVWorkflowGraph
from src.orchestration.state import AgentState
from src.models.data_models import (
    StructuredCV,
    Section,
    Item,
    MetadataModel,
    JobDescriptionData,
)
from src.core.application_startup import get_startup_manager


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    """Ensure the application is initialized before running tests in this module."""
    startup_manager = get_startup_manager()
    if not startup_manager.is_initialized:
        startup_manager.initialize_application()


@pytest.mark.asyncio
async def test_workflow_routes_to_error_handler_on_node_failure():
    """
    Simulate a node failure and verify the workflow routes to error_handler and terminates.
    """
    # Each test gets a fresh workflow with a new session ID
    workflow_graph = CVWorkflowGraph()
    workflow = workflow_graph.app

    # Minimal valid StructuredCV with one section and one item
    item = Item(content="Test content")
    section = Section(name="key_qualifications", items=[item])
    structured_cv = StructuredCV(sections=[section], metadata=MetadataModel())
    job_description_data = JobDescriptionData(raw_text="Test job description")

    # Create an AgentState with an error message to trigger error routing
    state = AgentState(
        structured_cv=structured_cv,
        cv_text="Test CV",
        error_messages=["Simulated node failure"],
        content_generation_queue=[],
        current_section_key="key_qualifications",
        job_description_data=job_description_data,
    )

    # Patch the agent that would run first
    with patch(
        "src.agents.parser_agent.ParserAgent.run_as_node",
        new_callable=AsyncMock,
    ) as mock_parser_run:
        # Make the patched agent return a dictionary to update the state
        mock_parser_run.return_value = {"parser_status": "mocked_complete"}

        # Invoke the workflow
        result_state = await workflow.ainvoke(state.model_dump())

    # The workflow should have gone through the error handler.
    # The default error handler just logs and removes the error, so we expect an empty list.
    assert not result_state.get("error_messages")


@pytest.mark.asyncio
async def test_error_handler_clears_error_and_applies_recovery():
    """
    Test that the error handler clears errors (simplified version).
    """
    workflow_graph = CVWorkflowGraph()
    workflow = workflow_graph.app

    item = Item(content="Test content")
    section = Section(name="key_qualifications", items=[item])
    structured_cv = StructuredCV(sections=[section], metadata=MetadataModel())
    job_description_data = JobDescriptionData(raw_text="Test job description")

    # Create state with error messages that should trigger error handler
    state = AgentState(
        structured_cv=structured_cv,
        cv_text="Test CV",
        error_messages=["Test error"],
        current_item_id=str(uuid4()),
        content_generation_queue=[],
        current_section_key="key_qualifications",
        job_description_data=job_description_data,
    )

    # Patch the parser to avoid LLM calls that cause event loop issues
    with patch(
        "src.agents.parser_agent.ParserAgent.run_as_node",
        new_callable=AsyncMock,
    ) as mock_parser_run:
        # Make the patched agent return a dictionary to update the state
        mock_parser_run.return_value = {"parser_status": "mocked_complete"}

        # Invoke the workflow - errors should route to error handler
        result_state = await workflow.ainvoke(state.model_dump())

    # Verify the error was handled and cleared
    assert not result_state.get("error_messages")
