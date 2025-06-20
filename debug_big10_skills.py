#!/usr/bin/env python3

import sys
sys.path.append('.')
import asyncio
from unittest.mock import patch, Mock, AsyncMock
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.services.llm_service import LLMResponse

async def test_big10_skills():
    try:
        with patch('src.agents.enhanced_content_writer.get_llm_service'):
            with patch('src.agents.enhanced_content_writer.get_config'):
                agent = EnhancedContentWriterAgent()
                print('Agent created successfully')
                
                # Mock LLM response
                mock_response = LLMResponse(
                    content="Some response",
                    processing_time=1.0,
                    tokens_used=150
                )
                
                agent.llm_service.generate_content = AsyncMock(return_value=mock_response)
                
                # Mock parser agent to raise exception
                agent.parser_agent._parse_big_10_skills = Mock(side_effect=Exception("Parser error"))
                
                # Mock template loading
                with patch.object(agent, '_load_prompt_template', return_value="Test template: {main_job_description_raw} {my_talents}"):
                    result = await agent.generate_big_10_skills(
                        job_description="Software Engineer position",
                        my_talents="Python developer"
                    )
                
                print(f"Result: {result}")
                print(f"Result type: {type(result)}")
                print(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
                
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_big10_skills())