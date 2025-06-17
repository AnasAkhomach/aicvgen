# Development Utilities

This document describes utility scripts for development and debugging.

## fix_imports.py

A utility script to run individual Python modules during development with proper import path setup.

### Problem

When running Python files directly (e.g., `python src/agents/agent_base.py`), you may encounter import errors like:
- `ModuleNotFoundError: No module named 'src'`
- `ImportError: attempted relative import with no known parent package`

This happens because the project uses absolute imports starting with `src.` and relative imports, which require proper Python path configuration.

### Solution

Use the `fix_imports.py` script to run individual modules with correct import resolution:

```bash
# Instead of:
python src/agents/agent_base.py

# Use:
python fix_imports.py src/agents/agent_base.py
```

### Usage Examples

```bash
# Run agent modules
python fix_imports.py src/agents/agent_base.py
python fix_imports.py src/agents/cleaning_agent.py

# Run core modules
python fix_imports.py src/core/performance_monitor.py
python fix_imports.py src/core/state_manager.py

# Run any module in the src directory
python fix_imports.py src/services/llm_service.py
```

### How It Works

1. **Path Setup**: Adds the project root directory to `sys.path` so `src` imports work
2. **Package Resolution**: Automatically determines the correct `__package__` value for relative imports
3. **Module Execution**: Runs the target module with proper `__name__`, `__file__`, and `__package__` settings

### Proper Application Launch

For normal application usage, always use the official entry points:

```bash
# Recommended: Use the official launcher
python run_app.py

# Alternative: Direct Streamlit launch
streamlit run app.py
```

These entry points properly configure the Python path and environment.

## Testing

For running tests, use pytest from the project root:

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/unit/test_agent_base.py
pytest tests/unit/test_config_consolidation.py

# Run with verbose output
pytest -v
```

Tests are configured to work with the proper import paths automatically.