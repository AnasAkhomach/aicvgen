#!/usr/bin/env python3
"""
Debug script to test the workflow and capture debug output.
"""

import asyncio
import sys
import os
from unittest.mock import Mock, patch

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.orchestration.state import AgentState
from src.orchestration.cv_workflow_graph import CVWorkflowGraph
from src.core.container import ContainerSingleton
from src.models.cv_models import StructuredCV, Section, Item, ItemStatus
from src.models.workflow_models import UserFeedback, UserAction
from src.models.agent_output_models import ResearchFindings, ResearchStatus, QualityAssuranceAgentOutput

async def debug_workflow():
    """Debug the workflow execution."""
    
    # Create initial state similar to the test
    from uuid import uuid4
    
    # Create items with proper UUIDs
    qual_item1 = Item(content="Python programming", status=ItemStatus.PENDING)
    qual_item1.id = uuid4()
    qual_item2 = Item(content="Machine learning", status=ItemStatus.PENDING)
    qual_item2.id = uuid4()
    
    exp_item = Item(content="Senior Developer at TechCorp", status=ItemStatus.PENDING)
    exp_item.id = uuid4()
    
    proj_item = Item(content="AI chatbot project", status=ItemStatus.PENDING)
    proj_item.id = uuid4()
    
    exec_item = Item(content="Experienced software engineer", status=ItemStatus.PENDING)
    exec_item.id = uuid4()
    
    initial_state = AgentState(
        cv_text="Experienced Python developer",
        structured_cv=StructuredCV(
            sections=[
                Section(
                    name="key_qualifications",
                    items=[qual_item1, qual_item2]
                ),
                Section(
                    name="professional_experience",
                    items=[exp_item]
                ),
                Section(
                    name="project_experience",
                    items=[proj_item]
                ),
                Section(
                    name="executive_summary",
                    items=[exec_item]
                )
            ]
        ),
        current_section_index=0,
        current_item_id=str(qual_item1.id),  # Set to first item's ID as string
        automated_mode=True,  # Enable automated mode
        node_execution_metadata={}
    )
    
    print(f"Initial state:")
    print(f"  Section index: {initial_state.current_section_index}")
    print(f"  Current item: {initial_state.current_item_id}")
    print(f"  Automated mode: {initial_state.automated_mode}")
    print(f"  Metadata: {initial_state.node_execution_metadata}")
    
    # Create mock agents
    from unittest.mock import AsyncMock
    mock_agents = {}
    agent_names = [
        'jd_parser_agent', 'cv_parser_agent', 'research_agent', 'cv_analyzer_agent',
        'key_qualifications_writer_agent', 'professional_experience_writer_agent',
        'projects_writer_agent', 'executive_summary_writer_agent', 'qa_agent', 'formatter_agent'
    ]
    
    for name in agent_names:
        mock_agents[name] = AsyncMock()
        mock_agents[name].run_as_node = AsyncMock()
    
    # Configure mock responses
    from src.models.cv_models import JobDescriptionData
    mock_agents['jd_parser_agent'].run_as_node.return_value = {
        "job_description_data": JobDescriptionData(
            raw_text="Software Engineer position at TechCorp",
            job_title="Software Engineer",
            company_name="TechCorp",
            skills=["Python", "Django", "AWS"]
        )
    }
    
    mock_agents['cv_parser_agent'].run_as_node.return_value = {
        "structured_cv": initial_state.structured_cv
    }
    
    mock_agents['research_agent'].run_as_node.return_value = {
        "research_findings": ResearchFindings(
            status=ResearchStatus.SUCCESS,
            key_terms=["Python", "Django", "AWS"],
            enhancement_suggestions=["Add more technical details"],
            confidence_score=0.8
        )
    }
    
    mock_agents['cv_analyzer_agent'].run_as_node.return_value = {
        "cv_analysis_results": {
            "analysis_summary": "CV analysis completed successfully",
            "strengths": ["Strong technical skills"],
            "areas_for_improvement": ["Add more quantifiable achievements"]
        }
    }
    
    # Configure writer agents to mark items as completed
    updated_cv = initial_state.structured_cv.model_copy(deep=True)
    for section in updated_cv.sections:
        for item in section.items:
            item.status = ItemStatus.COMPLETED
            item.content = f"Enhanced {item.content}"
    
    for writer_agent in ['key_qualifications_writer_agent', 'professional_experience_writer_agent', 
                       'projects_writer_agent', 'executive_summary_writer_agent']:
        mock_agents[writer_agent].run_as_node.return_value = {
            "structured_cv": updated_cv
        }
    
    mock_agents['qa_agent'].run_as_node.return_value = {
        "quality_check_results": QualityAssuranceAgentOutput(
            overall_passed=True,
            recommendations=["CV looks good overall"]
        )
    }
    
    mock_agents['formatter_agent'].run_as_node.return_value = {
        "final_output_path": "/path/to/generated_cv.pdf"
    }
    
    # Mock the container's agent providers
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
        
        # Initialize CVWorkflowGraph directly with mocked agents
        cv_workflow_graph = CVWorkflowGraph(
            session_id="debug-session",
            job_description_parser_agent=mock_agents['jd_parser_agent'],
            user_cv_parser_agent=mock_agents['cv_parser_agent'],
            research_agent=mock_agents['research_agent'],
            cv_analyzer_agent=mock_agents['cv_analyzer_agent'],
            key_qualifications_writer_agent=mock_agents['key_qualifications_writer_agent'],
            professional_experience_writer_agent=mock_agents['professional_experience_writer_agent'],
            projects_writer_agent=mock_agents['projects_writer_agent'],
            executive_summary_writer_agent=mock_agents['executive_summary_writer_agent'],
            qa_agent=mock_agents['qa_agent'],
            formatter_agent=mock_agents['formatter_agent']
        )
        
        print("\n=== Starting workflow execution ===\n")
        
        # Execute the workflow
        final_state_dict = await cv_workflow_graph.app.ainvoke(initial_state)
        final_state = AgentState.model_validate(final_state_dict)
        
        print(f"\n=== Final state ===\n")
        print(f"Final section index: {final_state.current_section_index}")
        print(f"Final current_item_id: {final_state.current_item_id}")
        print(f"Final metadata: {final_state.node_execution_metadata}")
        print(f"Final workflow_status: {getattr(final_state, 'workflow_status', 'None')}")
        print(f"Final output path: {final_state.final_output_path}")
        print(f"Error messages: {final_state.error_messages}")
        
        # Check agent call counts
        for name, agent in mock_agents.items():
            print(f"{name} call count: {agent.run_as_node.call_count}")

if __name__ == "__main__":
    asyncio.run(debug_workflow())