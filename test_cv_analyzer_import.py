#!/usr/bin/env python3
"""
Test script to check CV analyzer agent imports.
"""

try:
    import sys
    sys.path.insert(0, '.')
    
    print("Testing CV analyzer agent import...")
    from tests.unit.test_cv_analyzer_agent import TestCVAnalyzerAgent
    print("✓ CV analyzer agent test import successful")
    
    # Test the agent import directly
    from src.agents.cv_analyzer_agent import CVAnalyzerAgent
    print("✓ CV analyzer agent import successful")
    
    # Test AgentResult import
    from src.agents.agent_base import AgentResult
    print("✓ AgentResult import successful")
    
    print("All imports successful!")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()