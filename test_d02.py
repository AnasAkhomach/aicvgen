"""Test script for D-02 implementation"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_import():
    """Test that we can import the consolidated function correctly"""
    try:
        # Test importing from frontend (should work)
        from src.frontend.state_helpers import initialize_session_state

        print(
            "✅ Successfully imported initialize_session_state from frontend.state_helpers"
        )

        # Test that core.state_helpers still works for other functions
        from src.core.state_helpers import create_initial_agent_state

        print(
            "✅ Successfully imported create_initial_agent_state from core.state_helpers"
        )

        # Test that trying to import initialize_session_state from core fails
        try:
            from src.core.state_helpers import initialize_session_state as core_init

            print(
                "❌ ERROR: Should not be able to import initialize_session_state from core.state_helpers"
            )
            return False
        except ImportError:
            print(
                "✅ Correctly cannot import initialize_session_state from core.state_helpers"
            )

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


if __name__ == "__main__":
    success = test_import()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
