#!/usr/bin/env python3

import sys
sys.path.append('.')

from unittest.mock import patch
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent

try:
    with patch('src.agents.enhanced_content_writer.get_llm_service'):
        with patch('src.agents.enhanced_content_writer.get_config'):
            agent = EnhancedContentWriterAgent()
            print('Agent created successfully')
            print(f'Parser agent type: {type(agent.parser_agent)}')
            print(f'Parser agent name: {agent.parser_agent.name}')
except Exception as e:
    print(f'Error creating agent: {e}')
    import traceback
    traceback.print_exc()