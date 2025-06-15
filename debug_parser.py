import asyncio
import sys
sys.path.append('c:\\Users\\Nitro\\Desktop\\aicvgen')

from src.agents.parser_agent import ParserAgent
from src.services.llm import LLM

async def test_parser():
    try:
        llm = LLM()
        agent = ParserAgent('test_parser', 'test parser agent', llm)
        
        # Test with a simple job description
        test_input = {
            'job_description': 'Software Engineer position requiring Python skills and experience with web development'
        }
        
        print(f"Testing parser agent with input: {test_input}")
        result = await agent.run(test_input)
        
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        if isinstance(result, dict):
            print(f"Result keys: {list(result.keys())}")
            if 'job_description_data' in result:
                print(f"Job description data: {result['job_description_data']}")
                print(f"Job description data type: {type(result['job_description_data'])}")
            else:
                print("WARNING: 'job_description_data' key not found in result")
        else:
            print("ERROR: Result is not a dictionary")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_parser())