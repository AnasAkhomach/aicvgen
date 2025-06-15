#!/usr/bin/env python3
"""
Automated Import Statement Updater
This script updates import statements after project reorganization.
"""

import os
import re
from pathlib import Path

def update_imports_in_file(file_path):
    """Update import statements in a single Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Define import mappings (old -> new)
        import_mappings = {
            # Core imports
            'from src.core.main import': 'from src.core.main import',
            'import src.core.main as main': 'import src.core.main as main',
            'from src.core.orchestrator import': 'from src.core.enhanced_orchestrator import',
            'import src.core.orchestrator as orchestrator': 'import src.core.orchestrator as orchestrator',
            'from src.core.state_manager import': 'from src.core.state_manager import',
            'import src.core.state_manager as state_manager': 'import src.core.state_manager as state_manager',
            
            # Agent imports
            'from src.agents.agent_base import': 'from src.agents.agent_base import',
            'import src.agents.agent_base as agent_base': 'import src.agents.agent_base as agent_base',
            'from src.agents.content_writer_agent import': 'from src.agents.content_writer_agent import',
            'import src.agents.content_writer_agent as content_writer_agent': 'import src.agents.content_writer_agent as content_writer_agent',
            'from src.agents.cv_analyzer_agent import': 'from src.agents.cv_analyzer_agent import',
            'import src.agents.cv_analyzer_agent as cv_analyzer_agent': 'import src.agents.cv_analyzer_agent as cv_analyzer_agent',
            'from src.agents.formatter_agent import': 'from src.agents.formatter_agent import',
            'import src.agents.formatter_agent as formatter_agent': 'import src.agents.formatter_agent as formatter_agent',
            'from src.agents.parser_agent import': 'from src.agents.parser_agent import',
            'import src.agents.parser_agent as parser_agent': 'import src.agents.parser_agent as parser_agent',
            'from src.agents.quality_assurance_agent import': 'from src.agents.quality_assurance_agent import',
            'import src.agents.quality_assurance_agent as quality_assurance_agent': 'import src.agents.quality_assurance_agent as quality_assurance_agent',
            'from src.agents.research_agent import': 'from src.agents.research_agent import',
            'import src.agents.research_agent as research_agent': 'import src.agents.research_agent as research_agent',
            'from src.agents.tools_agent import': 'from src.agents.tools_agent import',
            'import src.agents.tools_agent as tools_agent': 'import src.agents.tools_agent as tools_agent',
            'from src.agents.vector_store_agent import': 'from src.agents.vector_store_agent import',
            'import src.agents.vector_store_agent as vector_store_agent': 'import src.agents.vector_store_agent as vector_store_agent',
            
            # Service imports
            'from src.services.llm import': 'from src.services.llm import',
            'import src.services.llm as llm': 'import src.services.llm as llm',
            'from src.services.vector_db import': 'from src.services.vector_db import',
            'import src.services.vector_db as vector_db': 'import src.services.vector_db as vector_db',
            
            # Utility imports
            'from src.utils.template_manager import': 'from src.utils.template_manager import',
            'import src.utils.template_manager as template_manager': 'import src.utils.template_manager as template_manager',
            'from src.utils.template_renderer import': 'from src.utils.template_renderer import',
            'import src.utils.template_renderer as template_renderer': 'import src.utils.template_renderer as template_renderer',
        }
        
        # Apply import mappings
        for old_import, new_import in import_mappings.items():
            content = content.replace(old_import, new_import)
        
        # Update relative imports within the same package
        # For files in src/agents/, update relative imports
        if 'src/agents/' in str(file_path):
            content = re.sub(r'from \.([a-zA-Z_][a-zA-Z0-9_]*) import', r'from src.agents.\1 import', content)
        
        # Update path references
        path_mappings = {
            # Template paths
            'src/templates/cv_template.md': 'src/templates/cv_template.md',
            'src/templates/tailored_cv.md': 'src/templates/tailored_cv.md',
            'src/templates/Anas_Akhomach-main-template-en.md': 'src/templates/Anas_Akhomach-main-template-en.md',
            
            # Prompt paths
            'data/prompts/': 'data/prompts/',
            
            # Job description paths
            'data/job_descriptions/': 'data/job_descriptions/',
            
            # Session data paths
            'data/sessions/': 'data/sessions/',  # Already correct
        }
        
        for old_path, new_path in path_mappings.items():
            content = content.replace(f"'{old_path}", f"'{new_path}")
            content = content.replace(f'"{old_path}', f'"{new_path}')
        
        # Write back if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {file_path}")
            return True
        else:
            print(f"No changes: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def find_python_files(directory):
    """Find all Python files in the directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def main():
    """Main function to update all import statements."""
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")
    
    # Find all Python files
    python_files = find_python_files(project_root)
    print(f"Found {len(python_files)} Python files")
    
    updated_count = 0
    
    # Update each file
    for file_path in python_files:
        if update_imports_in_file(file_path):
            updated_count += 1
    
    print(f"\nSummary: Updated {updated_count} out of {len(python_files)} files")
    
    # Additional manual updates needed
    print("\nManual updates still needed:")
    print("1. Check Dockerfile for path updates")
    print("2. Update any hardcoded paths in configuration files")
    print("3. Update README.md with new project structure")
    print("4. Run tests to verify everything works")

if __name__ == '__main__':
    main()