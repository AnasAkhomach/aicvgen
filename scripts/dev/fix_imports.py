import os
import re

def fix_relative_imports_in_file(file_path):
    """Fix relative imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace relative imports with absolute imports
        # Pattern 1: from ..module import something -> from src.module import something
        content = re.sub(r'^from \.\.(\w+(?:\.\w+)*)', r'from src.\1', content, flags=re.MULTILINE)
        
        # Pattern 2: from .module import something -> from src.current_package.module import something
        # We need to determine the current package from the file path
        rel_path = os.path.relpath(file_path, 'src').replace(os.sep, '.')
        current_package = '.'.join(rel_path.split('.')[:-1])  # Remove filename
        
        if current_package:
            content = re.sub(r'^from \.(\w+(?:\.\w+)*)', rf'from src.{current_package}.\1', content, flags=re.MULTILINE)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'Fixed: {file_path}')
            return True
        return False
    except Exception as e:
        print(f'Error processing {file_path}: {e}')
        return False

def fix_all_relative_imports(root_dir='src'):
    """Fix relative imports in all Python files in the src directory."""
    fixed_count = 0
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_relative_imports_in_file(file_path):
                    fixed_count += 1
    
    print(f'\nTotal files fixed: {fixed_count}')

if __name__ == '__main__':
    fix_all_relative_imports()