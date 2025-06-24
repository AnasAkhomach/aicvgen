# Changelog

## Revision 2

### **Task C-1: Unify State and Configuration (AD-01, CONF-01)**

- **Status**: DONE
- **Implementation**:
  - Created `src/utils/state_utils.py` to centralize agent state creation.
  - Migrated `create_initial_agent_state` logic to the new module.
  - Updated `src/frontend/callbacks.py` to use the new centralized function.
  - Deleted redundant state helper files: `src/core/state_helpers.py` and `src/frontend/state_helpers.py`.
  - Refactored `src/config/logging_config.py` to be environment-aware (`development` vs. `production`).
  - Removed obsolete logging configurations: `logging_config_simple.py` and `logging_config.py.backup`.
- **Tests**:
  - Added `tests/unit/test_state_utils.py` to verify the correctness of `create_initial_agent_state`.
  - Added `tests/unit/test_logging_config.py` to ensure logging configuration adapts correctly to the `APP_ENV`.
- **Notes**: The refactoring of `logging_config.py` drastically simplified the codebase by replacing a complex, custom implementation with a standard, library-based approach. This enhances maintainability and aligns with industry best practices.

### **Task C-2: Enforce Strict Agent Return Contracts (CB-01)**

- **Status**: DONE
- **Implementation**:
  - Refactored `CVAnalyzerAgent.run_async` to propagate exceptions on error, never returning a failed `AgentResult` with default output.
  - Updated all related unit tests to expect exceptions, not fallback outputs, on error.
- **Tests**:
  - All tests in `tests/unit/test_cv_analyzer_agent.py` pass, verifying contract enforcement and error handling.
- **Notes**:
  - This ensures orchestration and downstream consumers always receive valid, expected models or explicit errors.

### **Task C-3: Align Linter and Runtime Environments (CONF-03)**

- **Status**: DONE
- **Implementation**:
  - Refactored `src/agents/cv_analyzer_agent.py` for strict return contracts, robust error handling, and architectural consistency.
  - Constructor now takes a config dict; import order, docstrings, and formatting fixed.
  - Logger usage corrected (no `exc_info` in `extra`, correct argument order).
  - `log_decision` checks for `log_agent_decision` before calling.
  - `_extract_json_from_llm_response` always returns a value.
  - Added missing fields (`summary`, `key_skills`) to `CVAnalysisResult` and `extracted_data` to `CVAnalyzerAgentOutput`.
  - Error handling updated so `LLMResponseParsingError` is handled and reported as expected by tests.
  - **EnhancedContentWriterAgent**: Refactored all logger calls for structured logging (no positional args), fixed all linter/runtime errors, and added missing `RateLimitLog` dataclass to `src/models/data_models.py`.
  - Updated import in `rate_limiter.py` to use new `RateLimitLog` location.
- **Tests**:
  - Updated/fixed `tests/unit/test_cv_analyzer_agent.py` to match new model and error handling.
  - Added `tests/unit/test_enhanced_content_writer.py` for linter/runtime contract and Big 10 skills logic.
  - All unit tests for `CVAnalyzerAgent` and `EnhancedContentWriterAgent` now pass.
- **Notes**:
  - Dockerfile and `.pylintrc` confirmed for Python 3.11+.
  - All linter errors for these modules resolved.
  - Ready to proceed to next agent/module for C-3.

### **Task A-2: Encapsulate Parsing Logic within ParserAgent (AD-02)**

- **Status**: DONE
- **Implementation**:
  - Moved all logic from `cv_conversion_utils.py` and `cv_structure_utils.py` into `ParserAgent` as private methods.
  - Refactored `ParserAgent` to use these new private methods and removed all imports of the utility modules.
  - Deleted `src/agents/cv_conversion_utils.py` and `src/agents/cv_structure_utils.py`.
  - Ensured all public methods return validated `StructuredCV` or `JobDescriptionData` objects.
- **Tests**:
  - Added `tests/unit/test_parser_agent.py` to verify full conversion from raw text to `StructuredCV`.
  - All unit tests for `ParserAgent` now pass.
- **Notes**:
  - Parsing logic is now fully encapsulated within `ParserAgent`, improving maintainability and architectural consistency.

### **Task: Agent, Logging, and Linter/Test Hygiene (June 2025)**

- **Status**: DONE
- **Implementation**:
  - Refactored `src/agents/parser_agent.py` for strict logger usage, linter compliance, and robust handling of Pydantic list fields (no FieldInfo errors).
  - Patched all logger calls to use f-strings and keyword arguments only.
  - Ensured all list fields (`items`, `subsections`, `sections`) are always lists before using `append`.
  - Refactored and fixed all logging configuration and environment detection logic in `src/config/logging_config.py`.
  - Updated and fixed all related unit tests in `tests/unit/test_logging_config.py` to avoid patching core logging classes and to check handler/formatter effects directly.
  - Moved `APP_ENV` lookup to runtime in `setup_logging` for testability.
- **Tests**:
  - All unit tests for parser agent and logging config pass (`pytest` green).
  - Regression test suite run: all core, agent, and logging tests pass.
- **Notes**:
  - Codebase is now linter-clean and production-ready regarding agent, logger, and test hygiene.
  - Ready to proceed to next architectural task from `TASK_BLUEPRINT.md` (e.g., dependency injection consolidation).

## [Unreleased]
### Added
- Task A-3: Consolidated all dependency injection registration logic into a single `configure_container()` function in `src/core/dependency_injection.py`.
- Added unit test `test_configure_container_registers_all_dependencies` to verify all core services and agents are registered and resolvable.

### Changed
- Removed redundant `register_agents`, `register_agents_and_services`, `register_services`, and `register_core_services` methods from the DI container.
- Updated `application_startup.py` to use `configure_container()`.
- Fixed all logger calls in `FormatterAgent` to use f-strings and keyword arguments only.
- Fixed DI system to always wrap instances in `DependencyInstance` for correct lifecycle management.

### Tests
- All unit tests pass, including the new DI container test.

### Notes
- This completes Task A-3 (CONF-02) per the architecture blueprint. All DI is now consolidated and testable. No breaking changes to agent/service construction signatures.
