#!/usr/bin/env python3
"""Debug script to test AgentResult with new method names."""

import sys
sys.path.append('.')

from src.models.agent_models import AgentResult
from src.models.agent_output_models import ResearchAgentOutput, ResearchFindings, ResearchStatus

def test_agent_result():
    """Test AgentResult.create_success method."""
    print("Testing AgentResult with new method names...")
    
    try:
        # Create test data
        research_findings = ResearchFindings(status=ResearchStatus.SUCCESS)
        output = ResearchAgentOutput(research_findings=research_findings)
        
        print(f"Created output: {output}")
        
        # Test create_success method
        result = AgentResult.create_success(
            agent_name="TestAgent",
            output_data=output,
            message="Test message"
        )
        
        print(f"Success! Created result: {result}")
        print(f"Result success: {result.success}")
        print(f"Result output_data: {result.output_data}")
        print(f"Result metadata: {result.metadata}")
        
        # Test create_failure method
        failure_result = AgentResult.create_failure(
            agent_name="TestAgent",
            error_message="Test error message"
        )
        
        print(f"\nFailure result: {failure_result}")
        print(f"Failure success: {failure_result.success}")
        print(f"Failure error_message: {failure_result.error_message}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_agent_result()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")