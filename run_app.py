#!/usr/bin/env python3
"""
Application launcher that properly sets up the Python path for the reorganized project structure.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now we can import and run the main application
if __name__ == '__main__':
    try:
        from src.core.main import main
        main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)