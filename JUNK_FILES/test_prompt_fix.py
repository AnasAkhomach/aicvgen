#!/usr/bin/env python3
"""
Simple test script to verify the prompt template fix for CV generation.
"""

import asyncio
import json
from pathlib import Path
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.agents.agent_base import AgentExecutionContext
from src.models.data_models import ContentType

def load_test_data():
    """Load test data from a recent session."""
    sessions_dir = Path("data/sessions")
    
    # Find the most recent session with state.json
    latest_session = None
    for session_dir in sessions_dir.iterdir():
        if session_dir.is_dir() and (session_dir / "state.json").exists():
            latest_session = session_dir
    
    if not latest_session:
        raise FileNotFoundError("No session with state.json found")
    
    # Load the state data
    with open(latest_session / "state.json", 'r', encoding='utf-8') as f:
        state_data = json.load(f)
    
    return state_data

def extract_experience_section(state_data):
    """Extract the Professional Experience section from state data."""
    sections = state_data.get("sections", [])
    
    for section in sections:
        if section.get("name") == "Professional Experience":
            return section
    
    raise ValueError("Professional Experience section not found")

def create_test_context(state_data, experience_section):
    """Create test context for the agent."""
    
    # Extract job data
    job_data = state_data.get("metadata", {}).get("job_description", {})
    
    # Create generation context
    generation_context = {
        "original_cv_text": state_data.get("metadata", {}).get("original_cv_text", ""),
        "job_data": job_data
    }
    
    # Create input data structure
    input_data = {
        "job_description_data": job_data,
        "content_item": experience_section,
        "context": generation_context
    }
    
    # Create execution context
    context = AgentExecutionContext(
        session_id="test-session",
        item_id="test-experience",
        content_type=ContentType.EXPERIENCE,
        input_data=input_data
    )
    
    return context

async def test_prompt_template_fix():
    """Test the prompt template fix."""
    
    print("=== Testing Prompt Template Fix ===")
    
    try:
        # Load test data
        print("Loading test data...")
        state_data = load_test_data()
        print(f"✓ Loaded state data with {len(state_data.get('sections', []))} sections")
        
        # Extract experience section
        print("Extracting Professional Experience section...")
        experience_section = extract_experience_section(state_data)
        print(f"✓ Found experience section with {len(experience_section.get('subsections', []))} subsections")
        
        # Create agent
        print("Creating Enhanced Content Writer Agent...")
        agent = EnhancedContentWriterAgent(
            name="TestExperienceWriter",
            description="Test agent for experience content",
            content_type=ContentType.EXPERIENCE
        )
        print("✓ Agent created successfully")
        
        # Create test context
        print("Creating test context...")
        context = create_test_context(state_data, experience_section)
        print("✓ Test context created")
        
        # Test the agent
        print("\nTesting agent execution...")
        result = await agent.run_async(context.input_data, context)
        
        print(f"\n=== AGENT RESULT ===")
        print(f"Success: {result.success}")
        print(f"Confidence Score: {result.confidence_score}")
        print(f"Processing Time: {result.processing_time}s")
        
        if result.error_message:
            print(f"Error: {result.error_message}")
        
        if result.output_data:
            print(f"\n=== GENERATED CONTENT ===")
            if isinstance(result.output_data, dict):
                for key, value in result.output_data.items():
                    print(f"{key}: {value}")
            else:
                print(result.output_data)
        
        print(f"\n=== METADATA ===")
        for key, value in result.metadata.items():
            print(f"{key}: {value}")
            
        return result.success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_prompt_template_fix())
    
    if success:
        print("\n✅ Prompt template fix test PASSED!")
    else:
        print("\n❌ Prompt template fix test FAILED!")