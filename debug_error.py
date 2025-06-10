#!/usr/bin/env python3

import sys
import os
import traceback

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.models.data_models import ContentType
from src.agents.agent_base import AgentExecutionContext

async def test_content_writer_error():
    """Test to capture the exact error location."""
    try:
        # Initialize the content writer
        writer = EnhancedContentWriterAgent()
        
        # Create test data that matches the log
        input_data = {
            'job_description_data': "Job Description\nJob Description\n\nJob Title\nSupply Chain Apprentice M/F\nPurpose of the position\nAs a Supply Chain Apprentice, and under the supervision of the Supply Chain Analyst, you contribute to the design, improvement and support of tools, systems and processes.\nMissions\nBuild, implement and monitor indicators;\nAnalyze performance (service rate, adherence rate, late deliveries, etc.);\nPropose and participate in the implementation of organizational changes to the Supply Chain;\nBe a stakeholder in process improvement projects in collaboration with field teams;\nParticipate in the construction of the S&OP process;\nBuild simulation and optimization models.\nDesired profile\nYou are preparing a Master's degree in operations management and wish to discover the industrial environment;\nYou are organized and rigorous;\nYou have analytical skills and an ability to interpret data;\nYou are comfortable with oral and written communication;\nEssential mastery of office tools.\nCONTRACT\nApprentice\nContract duration\n24 months\nTravel\nOne-off (potentially monthly)\nCandidate criteria\n\nMinimum education level required\n3- License\nMinimum experience level required\nBeginner\nLocation of the position\n\nRegion, Department\nNormandy, Manche (50)\nCommune\nSAINT PAIR SUR MER\n\nGeneral information\n\nAttached entity\nFor 30 years, we have been designing and manufacturing luxury clothing in France for the finest fashion houses.\n\nWe strive to share our expertise, promote job creation, and develop career opportunities in our region.\n\nOur clients are experiencing strong growth and place their trust in us. To support them and expand our business, we are continuing our structuring approach and recruiting across all our professions.\n\n1,100 employees, spread across our 15 workshops, contribute every day to promoting excellent French craftsmanship. What if it were you?\nReference\n2025-278\n\nAttached entity\nFor 30 years, we have been designing and manufacturing luxury clothing in France for the finest fashion houses.\n\nWe strive to share our expertise, promote job creation, and develop career opportunities in our region.\n\nOur clients are experiencing strong growth and place their trust in us. To support them and expand our business, we are continuing our structuring approach and recruiting across all our professions.\n\n1,100 employees, spread across our 15 workshops, contribute every day to promoting excellent French craftsmanship. What if it were you?",
            'content_item': {
                'type': 'experience',
                'data': {
                    'roles': [],
                    'projects': [],
                    'personal_info': {}
                }
            },
            'context': {
                'workflow_type': 'job_tailored_cv',
                'content_type': 'experience',
                'personal_info': {}
            }
        }
        
        # Create a minimal context
        context = AgentExecutionContext(
            session_id="test_session",
            item_id="test_item",
            content_type=ContentType.EXPERIENCE,
            metadata={}
        )
        
        print("Testing content writer with empty roles...")
        result = await writer.run_async(input_data, context)
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"\n=== ERROR CAPTURED ===")
        print(f"Error: {e}")
        print(f"Error type: {type(e)}")
        print("\n=== FULL TRACEBACK ===")
        traceback.print_exc()
        print("\n=== END TRACEBACK ===")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_content_writer_error())