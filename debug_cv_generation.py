#!/usr/bin/env python3
"""
Debug script to test CV generation workflow and identify issues.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

# Set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = str(src_path)

from src.integration.enhanced_cv_system import EnhancedCVIntegration
from src.config.logging_config import setup_logging

# Setup logging
setup_logging()

async def test_cv_generation():
    """Test the CV generation workflow with debug output."""
    print("=== CV Generation Debug Test ===")
    
    # Initialize the enhanced CV system
    print("\n1. Initializing Enhanced CV System...")
    enhanced_cv = EnhancedCVIntegration()
    
    # Test data - simulating what should be passed to the workflow
    print("\n2. Setting up test data...")
    
    # Sample personal info
    personal_info = {
        "name": "Nouha KARIM",
        "email": "nouhakarim305@gmail.com",
        "phone": "+33 765732271",
        "location": "Reims",
        "title": "ETUDIANTE EN M2 OPTION GÉNIE DES SYSTÈMES INDUSTRIELS"
    }
    
    # Sample experience data
    experience = [
        {
            "title": "Student in Industrial Systems Engineering",
            "company": "University",
            "duration": "Current",
            "description": "Studying industrial systems engineering with focus on production management, supply chain, and quality management.",
            "skills": ["EXCEL", "POWER BI", "VBA", "PYTHON", "SYLOB", "SAP BUSINESS ONE"]
        }
    ]
    
    # Sample job description
    job_description = """
    We are looking for a Junior Industrial Engineer to join our team.
    
    Requirements:
    - Bachelor's or Master's degree in Industrial Engineering
    - Experience with ERP systems (SAP preferred)
    - Knowledge of data analysis tools (Excel, Power BI)
    - Understanding of supply chain management
    - Strong analytical and problem-solving skills
    - Fluency in French and English
    
    Responsibilities:
    - Analyze and optimize production processes
    - Manage inventory and supply chain operations
    - Create reports and dashboards using Power BI
    - Collaborate with cross-functional teams
    - Implement quality management systems
    """
    
    print(f"Personal Info: {personal_info}")
    print(f"Experience: {len(experience)} items")
    print(f"Job Description: {len(job_description)} characters")
    
    # Test the workflow
    print("\n3. Running CV generation workflow...")
    try:
        result = await enhanced_cv.generate_job_tailored_cv(
            personal_info=personal_info,
            experience=experience,
            job_description=job_description
        )
        
        print("\n4. Workflow completed successfully!")
        print(f"Result type: {type(result)}")
        
        if isinstance(result, dict):
            print(f"Result keys: {list(result.keys())}")
            
            # Check each section
            for key, value in result.items():
                print(f"\n--- {key.upper()} ---")
                if isinstance(value, str):
                    print(f"Length: {len(value)} characters")
                    if len(value) > 0:
                        print(f"Preview: {value[:200]}..." if len(value) > 200 else value)
                    else:
                        print("⚠️  EMPTY CONTENT!")
                elif isinstance(value, (list, dict)):
                    print(f"Type: {type(value)}, Length: {len(value)}")
                    if len(value) == 0:
                        print("⚠️  EMPTY COLLECTION!")
                    else:
                        print(f"Content: {value}")
                else:
                    print(f"Type: {type(value)}, Value: {value}")
        else:
            print(f"Unexpected result type: {type(result)}")
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"\n❌ Error during CV generation: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n=== Debug Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_cv_generation())