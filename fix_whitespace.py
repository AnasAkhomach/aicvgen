#!/usr/bin/env python3
"""Script to remove trailing whitespace from a file."""

import sys

def fix_trailing_whitespace(filepath):
    """Remove trailing whitespace from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove trailing whitespace from each line
    lines = content.splitlines()
    cleaned_lines = [line.rstrip() for line in lines]
    
    # Write back with proper line endings
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_lines))
        if cleaned_lines:  # Add final newline if file is not empty
            f.write('\n')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python fix_whitespace.py <filepath>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    fix_trailing_whitespace(filepath)
    print(f"Fixed trailing whitespace in {filepath}")