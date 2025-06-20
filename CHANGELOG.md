# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed
- `tests/unit/test_task_1_1_fixes.py` - Task-specific test file no longer needed
- `tests/unit/test_task_1_2_fixes.py` - Task-specific test file no longer needed
- `tests/unit/test_task_2_1_agent_state_alignment.py` - Task-specific test file no longer needed
- `tests/unit/test_task_2_2_input_validation.py` - Task-specific test file no longer needed
- `tests/unit/test_execute_workflow_simplification.py` - Workflow-specific test file no longer needed
- `tests/unit/test_latex_escaping.py` - LaTeX-specific test file no longer needed
- `tests/unit/test_ui_backend_state_transition.py` - UI transition test file no longer needed
- `tests/unit/test_parser_agent.py` - Outdated test for regex-based ParserAgent (replaced by LLM-first approach)

These deprecated test files were removed as their functionality has been integrated into the main test suite, improving maintainability and reducing confusion.

### Added
- Centralized error recovery service with intelligent error classification and recovery strategies
- Enhanced error handling with circuit breaker pattern integration
- Comprehensive error logging and monitoring capabilities
- Centralized retry logic with intelligent delay calculation and error-type-specific strategies
- Comprehensive structured logging implementation across all modules with sensitive data filtering

### Fixed
- Critical startup errors in formatter_agent.py (SyntaxError)
- Agent output contract validation - Added `validate_node_output` decorator to ensure consistent data structures across all workflow nodes (Task CB-06)
- Import resolution issues in models/__init__.py and environment.py
- Missing tenacity imports in rate_limiter.py preventing application startup
- Module loading and dependency injection problems
- Duplicate retry logic across services consolidated into EnhancedLLMService
- Removed dead code references to ContentOptimizationAgent in content_aggregator.py
- Removed unused agent initialization in enhanced_cv_system.py and startup_optimizer.py
- Eliminated duplicated JSON parsing logic across all agents

### Changed
- Increased max_retries from 3 to 5 in LLM service for better reliability
- Converted rate_limiter methods from async to sync for consistency
- Removed tenacity-based retry decorators in favor of centralized approach
- Refactored ContentOptimizationAgent anti-pattern into explicit workflow graph nodes
- Cleaned up references to non-existent ContentOptimizationAgent across codebase
- Re-architected ParserAgent to use LLM-first approach instead of regex-heavy parsing
- Updated all AgentIO schemas to accurately reflect actual agent inputs and outputs
- Consolidated JSON parsing to use centralized utility methods from EnhancedAgentBase

### Removed
- Obsolete import-fixing scripts (fix_agents_imports.py, fix_all_imports.py, fix_imports.py)
- Deprecated traditional web frontend files (index.html, script.js, styles.css)
- Dead code and unused file references throughout the codebase