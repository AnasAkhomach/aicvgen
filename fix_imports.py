#!/usr/bin/env python3
"""
Utility script to fix Python path for running individual modules during development.
This script adds the project root to sys.path so that 'src' imports work correctly.

Usage:
    python fix_imports.py <module_path>
    
Example:
    python fix_imports.py src/agents/agent_base.py
    python fix_imports.py src/core/performance_monitor.py
"""

import sys
import os
from pathlib import Path

def setup_python_path():
    """Add project root to Python path for proper imports."""
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    print(f"Added {project_root} to Python path")

def run_module(module_path):
    """Run a Python module with proper path setup."""
    setup_python_path()
    
    # Convert relative path to absolute
    if not os.path.isabs(module_path):
        module_path = os.path.join(os.getcwd(), module_path)
    
    if not os.path.exists(module_path):
        print(f"Error: Module {module_path} not found")
        sys.exit(1)
    
    print(f"Running {module_path}...")
    
    # For modules with relative imports, we need to set the package correctly
    # Convert file path to module name
    project_root = Path(__file__).parent
    rel_path = Path(module_path).relative_to(project_root)
    
    # Remove .py extension and convert path separators to dots
    module_name = str(rel_path.with_suffix('')).replace(os.sep, '.')
    
    # Determine package name (parent module)
    parts = module_name.split('.')
    if len(parts) > 1:
        package_name = '.'.join(parts[:-1])
    else:
        package_name = None
    
    # Execute the module
    with open(module_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Set __file__, __name__, and __package__ for the executed module
    globals_dict = {
        '__file__': module_path,
        '__name__': '__main__',
        '__package__': package_name
    }
    
    exec(code, globals_dict)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_imports.py <module_path>")
        print("Example: python fix_imports.py src/agents/agent_base.py")
        sys.exit(1)
    
    module_path = sys.argv[1]
    run_module(module_path)