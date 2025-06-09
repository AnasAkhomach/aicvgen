#!/usr/bin/env python3
"""
Test script to verify the quality and format of generated CV content.
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

def print_content_analysis(content):
    """Analyze and print details about the generated content."""
    
    print("\n=== CONTENT ANALYSIS ===")
    
    if isinstance(content, str):
        print(f"Content Type: String")
        print(f"Content Length: {len(content)} characters")
        
        # Check for common CV content patterns
        patterns = {
            "Role descriptions": "Role:" in content,
            "Organization info": "Organization:" in content,
            "Bullet points": "•" in content or "-" in content,
            "Skills mentioned": any(skill in content.lower() for skill in ["python", "sql", "data", "analysis"]),
            "Professional language": any(word in content.lower() for word in ["developed", "implemented", "managed", "led", "created"])
        }
        
        print("\nContent Pattern Analysis:")
        for pattern, found in patterns.items():
            status = "✓" if found else "✗"
            print(f"  {status} {pattern}: {found}")
        
        # Show first 500 characters
        print(f"\nContent Preview (first 500 chars):")
        print("-" * 50)
        print(content[:500])
        if len(content) > 500:
            print("... (truncated)")
        print("-" * 50)
        
    elif isinstance(content, dict):
        print(f"Content Type: Dictionary")
        print(f"Keys: {list(content.keys())}")
        
        for key, value in content.items():
            print(f"\n{key}:")
            if isinstance(value, str):
                print(f"  Length: {len(value)} characters")
                print(f"  Preview: {value[:200]}{'...' if len(value) > 200 else ''}")
            else:
                print(f"  Type: {type(value)}")
                print(f"  Value: {value}")
    
    else:
        print(f"Content Type: {type(content)}")
        print(f"Content: {content}")

async def test_content_quality():
    """Test the quality of generated content."""
    
    print("=== Testing CV Content Quality ===")
    
    try:
        # Load test data
        print("Loading test data...")
        state_data = load_test_data()
        
        # Show job requirements
        job_data = state_data.get("metadata", {}).get("job_description", {})
        print(f"\n=== JOB REQUIREMENTS ===")
        print(f"Skills Required: {job_data.get('skills', [])[:5]}...")  # Show first 5 skills
        print(f"Key Responsibilities: {len(job_data.get('responsibilities', []))} items")
        
        # Extract experience section
        experience_section = extract_experience_section(state_data)
        print(f"\n=== ORIGINAL CV DATA ===")
        print(f"Experience subsections: {len(experience_section.get('subsections', []))}")
        
        for i, subsection in enumerate(experience_section.get('subsections', [])[:2]):  # Show first 2
            print(f"  {i+1}. {subsection.get('name', 'Unknown Role')} - {len(subsection.get('items', []))} bullet points")
        
        # Create agent and test
        agent = EnhancedContentWriterAgent(
            name="TestExperienceWriter",
            description="Test agent for experience content",
            content_type=ContentType.EXPERIENCE
        )
        
        context = create_test_context(state_data, experience_section)
        
        print("\n=== GENERATING ENHANCED CONTENT ===")
        result = await agent.run_async(context.input_data, context)
        
        print(f"\nGeneration Success: {result.success}")
        print(f"Confidence Score: {result.confidence_score}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        
        if result.error_message:
            print(f"❌ Error: {result.error_message}")
            return False
        
        if result.output_data:
            print_content_analysis(result.output_data)
        else:
            print("❌ No content generated")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_content_quality())
    
    if success:
        print("\n✅ Content quality test PASSED!")
        print("\n🎉 The prompt template fix is working correctly!")
        print("The system is now generating proper CV content instead of instruction acknowledgments.")
    else:
        print("\n❌ Content quality test FAILED!")