#!/usr/bin/env python3
"""
Simple debug script to test only the parser agent.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents.parser_agent import ParserAgent
from src.services.llm import LLM

async def test_parser_only():
    """Test only the parser agent."""
    print("Testing Parser Agent Only...")
    
    try:
        print("1. Creating LLM instance...")
        llm = LLM()
        print("   LLM created successfully")
        
        print("2. Creating ParserAgent instance...")
        parser_agent = ParserAgent(
            name="debug_parser",
            description="Debug parser agent for testing",
            llm=llm
        )
        print("   ParserAgent created successfully")
        
        print("3. Preparing test input...")
        parser_input = {
            "job_description": "Software Engineer position requiring Python and AI experience.",
            "cv_text": "Test User\ntest@example.com\n\nSoftware Developer at Tech Corp (2020-2023)\nDeveloped Python applications"
        }
        print(f"   Input prepared: {parser_input}")
        
        print("4. Running parser agent...")
        result = await parser_agent.run(parser_input)
        
        print("5. Parser agent completed successfully!")
        print(f"   Result type: {type(result)}")
        print(f"   Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        print(f"   Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR OCCURRED:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_parser_only())
    print(f"\nTest result: {'SUCCESS' if success else 'FAILED'}")