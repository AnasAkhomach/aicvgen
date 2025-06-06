#!/usr/bin/env python3
"""
Script to fix sys.path statements in all test files after reorganization.
"""

import os
import re
from pathlib import Path

def fix_test_file_paths(test_dir):
    """Fix sys.path statements in all test files."""
    test_files = []
    
    # Find all Python test files
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                test_files.append(os.path.join(root, file))
    
    print(f"Found {len(test_files)} test files")
    
    updated_count = 0
    
    for test_file in test_files:
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix sys.path statements
            # Pattern 1: sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            pattern1 = r'sys\.path\.insert\(0, os\.path\.abspath\(os\.path\.join\(os\.path\.dirname\(__file__\), "\.\."\)\)\)'
            replacement1 = 'sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))'
            content = re.sub(pattern1, replacement1, content)
            
            # Pattern 2: sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
            pattern2 = r'sys\.path\.insert\(0, os\.path\.abspath\(os\.path\.join\(os\.path\.dirname\(__file__\), \'\.\.\'\'\)\)\)'
            replacement2 = 'sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))'
            content = re.sub(pattern2, replacement2, content)
            
            # Also fix any remaining old import patterns that might have been missed
            old_imports = {
                'from agent_base import': 'from src.agents.agent_base import',
                'import agent_base': 'from src.agents import agent_base',
                'from content_writer_agent import': 'from src.agents.content_writer_agent import',
                'from cv_analyzer_agent import': 'from src.agents.cv_analyzer_agent import',
                'from formatter_agent import': 'from src.agents.formatter_agent import',
                'from parser_agent import': 'from src.agents.parser_agent import',
                'from quality_assurance_agent import': 'from src.agents.quality_assurance_agent import',
                'from research_agent import': 'from src.agents.research_agent import',
                'from tools_agent import': 'from src.agents.tools_agent import',
                'from vector_store_agent import': 'from src.agents.vector_store_agent import',
                'from llm import': 'from src.services.llm import',
                'from vector_db import': 'from src.services.vector_db import',
                'from template_manager import': 'from src.utils.template_manager import',
                'from template_renderer import': 'from src.utils.template_renderer import',
                'from orchestrator import': 'from src.core.orchestrator import',
                'from state_manager import': 'from src.core.state_manager import',
                'from main import': 'from src.core.main import',
            }
            
            for old_import, new_import in old_imports.items():
                content = content.replace(old_import, new_import)
            
            # Write back if content changed
            if content != original_content:
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Updated: {test_file}")
                updated_count += 1
            else:
                print(f"No changes: {test_file}")
                
        except Exception as e:
            print(f"Error processing {test_file}: {e}")
    
    print(f"\nUpdated {updated_count} test files")

def main():
    """Main function."""
    project_root = Path(__file__).parent.parent
    test_dir = project_root / 'tests'
    
    print(f"Fixing test file paths in: {test_dir}")
    fix_test_file_paths(test_dir)
    
    print("\nDone! You can now run tests with:")
    print("python -m pytest tests/unit/ -v")

if __name__ == '__main__':
    main()