import os
import re

def fix_imports_in_file(file_path):
    """Fix imports in a single Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern 1: Fix absolute imports starting with 'from src.'
        content = re.sub(
            r'from src\.([^\s]+) import',
            lambda m: f'from ..{m.group(1)} import',
            content
        )
        
        # Pattern 2: Fix imports starting with 'from ...'
        content = re.sub(
            r'from \.\.\.([^\s]+) import',
            lambda m: f'from ..{m.group(1)} import',
            content
        )
        
        # Pattern 3: Fix imports within agents directory that reference other agents
        if 'src\\agents\\' in file_path or 'src/agents/' in file_path:
            # Fix imports like 'from ..agent_base import' to 'from .agent_base import'
            content = re.sub(
                r'from \.\.([^.][^\s]*) import',
                lambda m: f'from .{m.group(1)} import' if not any(x in m.group(1) for x in ['config', 'core', 'models', 'services', 'orchestration', 'utils', 'frontend', 'integration', 'templates']) else f'from ..{m.group(1)} import',
                content
            )
        
        # Pattern 4: Fix imports in __init__.py files
        if file_path.endswith('__init__.py'):
            # Fix imports like 'from ..module import' to 'from .module import' for same-level modules
            content = re.sub(
                r'from \.\.([^.][^\s]*) import',
                lambda m: f'from .{m.group(1)} import',
                content
            )
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed imports in {file_path}")
            return True
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Fix imports in all Python files in the src directory."""
    src_dir = 'src'
    
    if not os.path.exists(src_dir):
        print(f"Directory {src_dir} not found")
        return
    
    fixed_count = 0
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_imports_in_file(file_path):
                    fixed_count += 1
    
    print(f"\nFixed imports in {fixed_count} files")

if __name__ == '__main__':
    main()