#!/usr/bin/env python3
"""Debug script to test agent creation."""

from unittest.mock import Mock
from src.agents.professional_experience_writer_agent import ProfessionalExperienceWriterAgent
from src.services.llm_service import EnhancedLLMService
from src.templates.content_templates import ContentTemplateManager

# Create mock services
mock_llm_service = Mock(spec=EnhancedLLMService)
mock_llm_service.api_key = "test-api-key"

mock_template_manager = Mock(spec=ContentTemplateManager)
mock_template = Mock()
mock_template.render.return_value = "Test template content"
mock_template_manager.get_template.return_value = mock_template

try:
    # Create agent
    agent = ProfessionalExperienceWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        session_id="test_session",
        settings={}
    )
    
    print(f"Agent created successfully: {agent.name}")
    print(f"Output parser type: {type(agent.output_parser)}")
    print(f"Has get_format_instructions: {hasattr(agent.output_parser, 'get_format_instructions')}")
    
    if hasattr(agent.output_parser, 'get_format_instructions'):
        try:
            instructions = agent.output_parser.get_format_instructions()
            print(f"Format instructions work: {instructions[:100]}...")
        except Exception as e:
            print(f"Error calling get_format_instructions: {e}")
    
except Exception as e:
    print(f"Error creating agent: {e}")
    import traceback
    traceback.print_exc()