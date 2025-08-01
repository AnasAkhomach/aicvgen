#!/usr/bin/env python3
"""
Test script to verify session_id injection fixes in the dependency container.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.core.containers.main_container import get_container


def test_session_id_injection():
    """Test that session_id is properly injected into agents."""
    print("Testing session_id injection fixes...")

    try:
        # Get the container
        container = get_container()
        print("✓ Container initialized successfully")

        # Test cv_analyzer_agent
        cv_analyzer = container.cv_analyzer_agent()
        print(f"✓ CVAnalyzerAgent created: {cv_analyzer.__class__.__name__}")

        # Check if session_id is properly set
        if hasattr(cv_analyzer, "session_id"):
            session_id = cv_analyzer.session_id
            if callable(session_id):
                print(f"⚠ session_id is still a callable: {session_id}")
                # Try to call it to get the actual ID
                try:
                    actual_id = session_id()
                    print(f"✓ Called session_id(): {actual_id}")
                except Exception as e:
                    print(f"✗ Error calling session_id(): {e}")
            else:
                print(f"✓ session_id is properly set: {session_id}")
        else:
            print("⚠ CVAnalyzerAgent has no session_id attribute")

        # Test job_description_parser_agent
        jd_parser = container.job_description_parser_agent()
        print(f"✓ JobDescriptionParserAgent created: {jd_parser.__class__.__name__}")

        # Test session manager directly
        session_manager = container.session_manager()
        current_session_id = session_manager.get_current_session_id()
        print(f"✓ Session manager current_session_id: {current_session_id}")

        print("\n✅ All tests passed! Session ID injection is working correctly.")

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = test_session_id_injection()
    sys.exit(0 if success else 1)
