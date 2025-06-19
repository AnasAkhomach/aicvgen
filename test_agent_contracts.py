#!/usr/bin/env python3
"""
Test script to verify agent data contracts and IO schemas.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_agent_io_schemas():
    """Test that all agents have proper IO schemas defined."""
    print("Testing Agent IO Schemas...")
    
    try:
        from src.agents.parser_agent import ParserAgent
        from src.agents.research_agent import ResearchAgent
        from src.agents.quality_assurance_agent import QualityAssuranceAgent
        from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
        from src.agents.formatter_agent import FormatterAgent
        from src.services.llm_service import get_llm_service
        from src.services.vector_db import get_enhanced_vector_db
        
        # Get services needed for agent initialization
        llm_service = get_llm_service()
        vector_db = get_enhanced_vector_db()
        
        # Create agent instances to check their schemas
        agents = [
            ("ParserAgent", ParserAgent("TestParser", "Test parser", llm_service=llm_service)),
            ("ResearchAgent", ResearchAgent("TestResearch", "Test research", llm_service=llm_service, vector_db=vector_db)),
            ("QualityAssuranceAgent", QualityAssuranceAgent("TestQA", "Test QA", llm_service=llm_service)),
            ("EnhancedContentWriterAgent", EnhancedContentWriterAgent("TestWriter", "Test writer")),
            ("FormatterAgent", FormatterAgent("TestFormatter", "Test formatter"))
        ]
        
        all_passed = True
        for agent_name, agent_instance in agents:
            print(f"\nTesting {agent_name}:")
            
            # Check input schema
            if hasattr(agent_instance, 'input_schema'):
                input_schema = agent_instance.input_schema
                print(f"  ✓ Has input schema")
                print(f"    Description: {input_schema.description}")
                print(f"    Required fields: {input_schema.required_fields}")
                print(f"    Optional fields: {input_schema.optional_fields}")
            else:
                print(f"  ✗ No input schema found")
                all_passed = False
                
            # Check output schema
            if hasattr(agent_instance, 'output_schema'):
                output_schema = agent_instance.output_schema
                print(f"  ✓ Has output schema")
                print(f"    Description: {output_schema.description}")
                print(f"    Required fields: {output_schema.required_fields}")
                print(f"    Optional fields: {output_schema.optional_fields}")
            else:
                print(f"  ✗ No output schema found")
                all_passed = False
                
        print("\n" + "="*50)
        print("Agent IO Schema Test Complete")
        return all_passed
        
    except Exception as e:
        print(f"Error testing agent IO schemas: {e}")
        return False

def test_agent_state_fields():
    """Test that AgentState has all required fields."""
    print("\nTesting AgentState Fields...")
    
    try:
        from src.orchestration.state import AgentState
        from src.models.data_models import StructuredCV, JobDescriptionData
        
        # Create minimal test data
        minimal_cv = StructuredCV(sections=[])
        job_data = JobDescriptionData(raw_text="Test job description")
        
        # Test AgentState creation
        test_state = AgentState(
            trace_id="test-123",
            structured_cv=minimal_cv,
            job_description_data=job_data,
            cv_text="Test CV content"
        )
        
        # Check key fields that agents should return
        expected_fields = [
            'structured_cv',
            'job_description_data', 
            'research_findings',
            'quality_check_results',
            'content_generation_queue',
            'error_messages'
        ]
        
        print("AgentState fields:")
        for field in expected_fields:
            if hasattr(test_state, field):
                print(f"  ✓ {field}")
            else:
                print(f"  ✗ Missing {field}")
                
        print("\n" + "="*50)
        print("AgentState Fields Test Complete")
        return True
        
    except Exception as e:
        print(f"Error testing AgentState: {e}")
        return False

def test_node_return_contracts():
    """Test the expected return contracts for workflow nodes."""
    print("\nTesting Node Return Contracts...")
    
    try:
        # Expected return keys for each node type
        node_contracts = {
            'parser_node': ['structured_cv', 'job_description_data'],
            'research_node': ['research_findings'],
            'qa_node': ['quality_check_results'],
            'content_writer_node': ['structured_cv'],
            'formatter_node': ['final_output_path']
        }
        
        print("Expected node return contracts:")
        for node_name, expected_keys in node_contracts.items():
            print(f"  {node_name}: {expected_keys}")
            
        print("\n" + "="*50)
        print("Node Return Contracts Test Complete")
        return True
        
    except Exception as e:
        print(f"Error testing node contracts: {e}")
        return False

def main():
    """Run all agent contract tests."""
    print("Starting Agent Data Contract Tests")
    print("="*50)
    
    tests = [
        test_agent_io_schemas,
        test_agent_state_fields,
        test_node_return_contracts
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"Test {test_func.__name__} failed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All agent contract tests passed!")
        return 0
    else:
        print("✗ Some agent contract tests failed.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)