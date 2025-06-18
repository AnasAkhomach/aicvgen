import os
import re

def fix_agents_imports_in_file(file_path):
    """Fix incorrect .agents.agent_base imports in agent files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix imports like 'from .agents.agent_base import' to 'from .agent_base import'
        content = re.sub(
            r'from \.agents\.agent_base import',
            'from .agent_base import',
            content
        )
        
        # Fix other similar patterns
        content = re.sub(
            r'from \.agents\.([^\s]+) import',
            r'from .\1 import',
            content
        )
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed agents imports in {file_path}")
            return True
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Fix agents imports in all Python files in the agents directory."""
    agents_dir = 'src/agents'
    
    if not os.path.exists(agents_dir):
        print(f"Directory {agents_dir} not found")
        return
    
    fixed_count = 0
    
    for file in os.listdir(agents_dir):
        if file.endswith('.py'):
            file_path = os.path.join(agents_dir, file)
            if fix_agents_imports_in_file(file_path):
                fixed_count += 1
    
    print(f"\nFixed agents imports in {fixed_count} files")

if __name__ == '__main__':
    main()