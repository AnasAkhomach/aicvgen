#!/usr/bin/env python3
"""
Test script to verify error propagation from agents to orchestrator.
This script introduces a temporary error and checks if it's properly handled.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents.agent_base import AgentExecutionContext, AgentResult
from src.agents.parser_agent import ParserAgent
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.agents.cv_analyzer_agent import CVAnalyzerAgent
from src.models.data_models import ContentType
from src.config.logging_config import get_structured_logger
from src.services.llm import get_llm_service

logger = get_structured_logger("error_propagation_test")


class TestAgent(ParserAgent):
    """Test agent that deliberately fails to test error propagation."""
    
    def __init__(self, should_fail: bool = True):
        llm_service = get_llm_service()
        super().__init__(
            name="TestAgent",
            description="Test agent for error propagation testing",
            llm=llm_service
        )
        self.should_fail = should_fail
    
    async def run_async(self, input_data, context):
        """Override run_async to introduce a controlled failure."""
        if self.should_fail:
            # Deliberately raise an exception to test error propagation
            raise ValueError("Deliberate test error: Agent failure simulation")
        
        # If not failing, call the parent method
        return await super().run_async(input_data, context)


async def test_error_propagation():
    """Test that agent errors are properly propagated to the orchestrator."""
    
    logger.info("Starting error propagation test...")
    
    # Create test context
    context = AgentExecutionContext(
        session_id="test_session_001",
        item_id="test_item_001",
        content_type=ContentType.EXPERIENCE
    )
    
    # Test 1: Agent that fails
    logger.info("Test 1: Testing agent that deliberately fails...")
    failing_agent = TestAgent(should_fail=True)
    
    test_input = {
        "cv_text": "Test CV content",
        "job_description": "Test job description"
    }
    
    try:
        result = await failing_agent.run_async(test_input, context)
        
        # Check if the result indicates failure
        if result.success:
            logger.error("ERROR: Agent reported success when it should have failed!")
            return False
        else:
            logger.info(f"SUCCESS: Agent correctly reported failure with error: {result.error_message}")
            
            # Verify error message is descriptive
            if "Deliberate test error" in result.error_message:
                logger.info("SUCCESS: Error message is descriptive and contains expected content")
            else:
                logger.warning(f"WARNING: Error message may not be descriptive enough: {result.error_message}")
    
    except Exception as e:
        logger.error(f"ERROR: Unexpected exception not caught by agent: {str(e)}")
        return False
    
    # Test 2: Agent that succeeds
    logger.info("Test 2: Testing agent that should succeed...")
    succeeding_agent = TestAgent(should_fail=False)
    
    try:
        result = await succeeding_agent.run_async(test_input, context)
        
        if result.success:
            logger.info("SUCCESS: Agent correctly reported success")
        else:
            logger.error(f"ERROR: Agent reported failure when it should have succeeded: {result.error_message}")
            return False
    
    except Exception as e:
        logger.error(f"ERROR: Unexpected exception in succeeding agent: {str(e)}")
        return False
    
    # Test 3: Test other agents for proper error handling structure
    logger.info("Test 3: Testing other agents for proper error handling...")
    
    agents_to_test = [
        ("CVAnalyzerAgent", CVAnalyzerAgent()),
        ("EnhancedContentWriterAgent", EnhancedContentWriterAgent(ContentType.EXPERIENCE))
    ]
    
    for agent_name, agent in agents_to_test:
        logger.info(f"Testing {agent_name}...")
        
        # Test with invalid input to trigger error handling
        invalid_input = None  # This should trigger error handling
        
        try:
            result = await agent.run_async(invalid_input, context)
            
            if hasattr(result, 'success') and hasattr(result, 'error_message'):
                logger.info(f"SUCCESS: {agent_name} has proper AgentResult structure")
                
                if not result.success:
                    logger.info(f"SUCCESS: {agent_name} correctly reported failure for invalid input")
                    logger.info(f"Error message: {result.error_message}")
                else:
                    logger.warning(f"WARNING: {agent_name} reported success with invalid input")
            else:
                logger.error(f"ERROR: {agent_name} does not return proper AgentResult structure")
                return False
        
        except Exception as e:
            logger.error(f"ERROR: {agent_name} threw unhandled exception: {str(e)}")
            return False
    
    logger.info("All error propagation tests completed successfully!")
    return True


async def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("ERROR PROPAGATION TEST")
    logger.info("=" * 60)
    
    success = await test_error_propagation()
    
    if success:
        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED: Error propagation is working correctly!")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("\n" + "=" * 60)
        logger.error("TESTS FAILED: Error propagation needs attention!")
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)