#!/usr/bin/env python3

import sys
import traceback

print("Testing CB-02 Pydantic model refactoring...")
print("=" * 50)

try:
    from src.models.research_models import ResearchFindings, ResearchStatus
    print("✓ Research models imported successfully")
    
    # Test creating an empty research finding
    empty_research = ResearchFindings.create_empty()
    print(f"✓ Created empty research findings: {empty_research.status}")
    
except Exception as e:
    print(f"✗ Failed to import/test research models: {e}")
    traceback.print_exc()

try:
    from src.models.quality_models import QualityCheckResults, QualityStatus
    print("✓ Quality models imported successfully")
    
    # Test creating empty quality results
    empty_quality = QualityCheckResults.create_empty()
    print(f"✓ Created empty quality results: {empty_quality.overall_status}")
    
except Exception as e:
    print(f"✗ Failed to import/test quality models: {e}")
    traceback.print_exc()

try:
    from src.orchestration.state import AgentState
    from src.models.data_models import StructuredCV
    print("✓ AgentState imported successfully")
    
    # Test that AgentState has the new typed fields
    # Create a minimal StructuredCV for testing
    test_cv = StructuredCV(sections=[])
    state = AgentState(structured_cv=test_cv)
    print(f"✓ AgentState created with research_findings type: {type(state.research_findings)}")
    print(f"✓ AgentState created with quality_check_results type: {type(state.quality_check_results)}")
    
except Exception as e:
    print(f"✗ Failed to import/test AgentState: {e}")
    traceback.print_exc()

print("\n" + "=" * 50)
print("CB-02 refactoring test completed.")
