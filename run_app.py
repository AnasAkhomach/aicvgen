#!/usr/bin/env python3
"""
Entry point for the CV Generator Streamlit application.
This script properly sets up the Python path and runs the application.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Now import and run the Streamlit app
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    import sys
    
    # Set up the command line arguments for Streamlit
    sys.argv = [
        "streamlit",
        "run",
        str(src_path / "api" / "main.py"),
        "--server.port=8501"
    ]
    
    # Run Streamlit
    stcli.main()