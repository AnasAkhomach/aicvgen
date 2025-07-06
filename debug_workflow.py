#!/usr/bin/env python3
"""
Debug script to trace workflow execution and understand routing issues.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestration.state import AgentState
from src.models.cv_models import StructuredCV, Section, Item, ItemStatus, JobDescriptionData
from src.models.agent_output_models import ResearchFindings, CVAnalysisResult, ResearchStatus
from src.models.workflow_models import UserFeedback, ContentType, UserAction
from src.orchestration.cv_workflow_graph import CVWorkflowGraph, WorkflowNodes
from src.core.container import ContainerSingleton
from datetime import datetime


async def debug_workflow_routing():
    """Debug the workflow routing to understand why agents aren't being called."""
    
    # Create sample data
    sample_cv = StructuredCV(
        sections=[
            Section(
                name="key_qualifications",
                items=[
                    Item(
                        content="Python programming expertise",
                        status=ItemStatus.PENDING
                    ),
                    Item(
                        content="AWS cloud architecture",
                        status=ItemStatus.PENDING
                    )
                ]
            )
        ]
    )
    
    sample_job_data = JobDescriptionData(
        raw_text="Software Engineer at Tech Corp. Requirements: Python, AWS. Responsibilities: Develop software.",
        job_title="Software Engineer",
        company_name="Tech Corp",
        responsibilities=["Develop software"],
        skills=["Python", "AWS"]
    )
    
    # Create initial state
    initial_state = AgentState(
        session_id="test_session_123",
        trace_id="test_trace_456",
        cv_text="Sample CV text content",
        structured_cv=sample_cv,
        job_description_data=sample_job_data,
        research_findings=ResearchFindings(
            status=ResearchStatus.SUCCESS,
            research_timestamp=datetime.now(),
            key_terms=["Python", "Software Engineering"],
            enhancement_suggestions=["Add more technical details"]
        ),
        cv_analysis_results=CVAnalysisResult(
            summary="Sample analysis",
            key_skills=["Python", "AWS"],
            match_score=0.8
        ),
        current_section_key="key_qualifications",
        current_section_index=0,
        items_to_process_queue=[],
        current_item_id=None,
        current_content_type=ContentType.QUALIFICATION,
        is_initial_generation=True,
        error_messages=[],
        node_execution_metadata={}
    )
    
    print(f"Initial state:")
    print(f"  current_section_index: {initial_state.current_section_index}")
    print(f"  current_section_key: {initial_state.current_section_key}")
    print(f"  has research_findings: {initial_state.research_findings is not None}")
    print(f"  has cv_analysis_results: {initial_state.cv_analysis_results is not None}")
    print(f"  node_execution_metadata: {initial_state.node_execution_metadata}")
    
    # Create mock agents
    mock_agents = {
        'jd_parser_agent': Mock(),
        'cv_parser_agent': Mock(),
        'research_agent': Mock(),
        'cv_analyzer_agent': Mock(),
        'key_qualifications_writer_agent': Mock(),
        'professional_experience_writer_agent': Mock(),
        'projects_writer_agent': Mock(),
        'executive_summary_writer_agent': Mock(),
        'qa_agent': Mock(),
        'formatter_agent': Mock()
    }
    
    # Configure mock agents
    for agent_name, agent in mock_agents.items():
        agent.run_as_node = AsyncMock()
        if agent_name == 'key_qualifications_writer_agent':
            # Configure for regeneration scenario
            call_count = 0
            def mock_writer_side_effect(state):
                nonlocal call_count
                call_count += 1
                print(f"Key qualifications writer called! Call #{call_count}")
                
                if call_count == 1:
                    # First call - request regeneration
                    qual_section = next((s for s in state.structured_cv.sections if s.name == "key_qualifications"), None)
                    target_item_id = qual_section.items[0].id if qual_section and qual_section.items else "qual_1"
                    
                    return state.model_copy(update={
                        "user_feedback": UserFeedback(
                            action=UserAction.REGENERATE,
                            content="Please regenerate this content",
                            target_item_id=target_item_id
                        )
                    })
                else:
                    # Subsequent calls - successful regeneration
                    updated_cv = state.structured_cv.model_copy(deep=True)
                    for section in updated_cv.sections:
                        if section.name == "key_qualifications":
                            if section.items:
                                section.items[0].content = "Regenerated Python programming expertise"
                                section.items[0].status = ItemStatus.COMPLETED
                            break
                    
                    return state.model_copy(update={
                        "structured_cv": updated_cv,
                        "user_feedback": None
                    })
            
            agent.run_as_node.side_effect = mock_writer_side_effect
        else:
            # Other agents return success
            agent.run_as_node.return_value = initial_state.model_copy(
                update={"node_execution_metadata": {agent_name: "success"}}
            )
    
    # Mock the container and create workflow
    container = ContainerSingleton.get_instance()
    with patch.object(container, 'job_description_parser_agent') as mock_jd_provider, \
         patch.object(container, 'user_cv_parser_agent') as mock_cv_provider, \
         patch.object(container, 'research_agent') as mock_research_provider, \
         patch.object(container, 'cv_analyzer_agent') as mock_analyzer_provider, \
         patch.object(container, 'key_qualifications_writer_agent') as mock_qual_provider, \
         patch.object(container, 'professional_experience_writer_agent') as mock_exp_provider, \
         patch.object(container, 'projects_writer_agent') as mock_proj_provider, \
         patch.object(container, 'executive_summary_writer_agent') as mock_exec_provider, \
         patch.object(container, 'quality_assurance_agent') as mock_qa_provider, \
         patch.object(container, 'formatter_agent') as mock_formatter_provider:
        
        # Set up mock return values
        mock_jd_provider.return_value = mock_agents['jd_parser_agent']
        mock_cv_provider.return_value = mock_agents['cv_parser_agent']
        mock_research_provider.return_value = mock_agents['research_agent']
        mock_analyzer_provider.return_value = mock_agents['cv_analyzer_agent']
        mock_qual_provider.return_value = mock_agents['key_qualifications_writer_agent']
        mock_exp_provider.return_value = mock_agents['professional_experience_writer_agent']
        mock_proj_provider.return_value = mock_agents['projects_writer_agent']
        mock_exec_provider.return_value = mock_agents['executive_summary_writer_agent']
        mock_qa_provider.return_value = mock_agents['qa_agent']
        mock_formatter_provider.return_value = mock_agents['formatter_agent']
        
        # Create workflow
        cv_workflow_graph = container.cv_workflow_graph()
        
        print(f"\n=== Testing entry router ===")
        # Test entry router first
        entry_result = await cv_workflow_graph._entry_router_node(initial_state)
        print(f"Entry router result:")
        print(f"  entry_route: {entry_result.node_execution_metadata.get('entry_route')}")
        
        print(f"\n=== Testing supervisor routing ===")
        # Test supervisor routing
        supervisor_result = await cv_workflow_graph.supervisor_node(entry_result)
        print(f"Supervisor result:")
        print(f"  next_node: {supervisor_result.node_execution_metadata.get('next_node')}")
        print(f"  current_section_index: {supervisor_result.current_section_index}")
        
        print(f"\n=== Running full workflow ===")
        # Run the full workflow
        final_state_dict = await cv_workflow_graph.app.ainvoke(initial_state)
        final_state = AgentState.model_validate(final_state_dict)
        
        print(f"\n=== Final results ===")
        print(f"Key qualifications writer call count: {mock_agents['key_qualifications_writer_agent'].run_as_node.call_count}")
        print(f"Final section index: {final_state.current_section_index}")
        print(f"Final metadata: {final_state.node_execution_metadata}")
        print(f"Final errors: {final_state.error_messages}")
        print(f"User feedback: {final_state.user_feedback}")


if __name__ == "__main__":
    asyncio.run(debug_workflow_routing())