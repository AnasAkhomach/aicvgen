# Audit log for unit test failures and obsolete tests in aicvgen (MVP stabilization)

This file will be used to track which unit tests are outdated, failing due to codebase changes, or require refactoring/removal. Each test file will be reviewed and marked for action.

Legend:

- [REMOVE] Test is obsolete, references removed/renamed APIs, or is fundamentally incompatible with current codebase.
- [FIX/REWRITE] Test logic is valuable but needs to be updated for new APIs/models or import paths.
- [KEEP] Test is valid and passes or only needs minor updates.

---

## [FIX/REWRITE] Import path issues (src not found)
- tests/unit/test_agent_error_handling.py
- tests/unit/test_agent_state_contracts.py
- tests/unit/test_api_key_management.py
- tests/unit/test_application_startup.py
- tests/unit/test_centralized_json_parsing.py
- tests/unit/test_cleaning_agent.py
- tests/unit/test_consolidated_caching.py
- tests/unit/test_cv_analyzer_agent.py

Action: Fix import path so 'src' is recognized as a package root for all unit tests. If not feasible, refactor imports to be relative or update PYTHONPATH in test runner.

(Review in progress)
