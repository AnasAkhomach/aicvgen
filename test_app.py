#!/usr/bin/env python3
"""
Minimal Streamlit app to test if the application framework can start.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set page config first
st.set_page_config(page_title="Debug Test", page_icon="🔍", layout="wide")


def main():
    """Minimal main function for testing."""
    st.title("🔍 Debug Test Application")
    st.write("If you can see this page, the basic Streamlit setup is working.")

    # Test logging import
    try:
        from src.config.logging_config import get_logger, setup_logging

        st.success("✅ Logging imports successful")

        # Setup minimal logging
        logger = setup_logging(log_to_console=True, log_to_file=False)
        logger.info("Streamlit app started successfully")
        st.success("✅ Logging setup successful")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
