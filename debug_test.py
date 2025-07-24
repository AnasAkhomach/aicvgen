#!/usr/bin/env python3

import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from tests.unit.test_projects_writer_agent import (
    test_projects_writer_agent_success,
    mock_llm_service,
    mock_template_manager,
    mock_settings,
    sample_structured_cv,
    sample_job_description_data
)
import pytest

async def run_test():
    try:
        # Create the fixtures
        llm_service = mock_llm_service()
        template_manager = mock_template_manager()
        settings = mock_settings()
        cv = sample_structured_cv()
        jd = sample_job_description_data()
        
        # Run the test
        await test_projects_writer_agent_success(
            llm_service, template_manager, settings, cv, jd
        )
        print("Test passed!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_test())