# AI CV Generator MVP - Task Execution Contract

## Current Task: P0 - Refactor Bloated Entry Point (main.py)

### Issue Description
The `src/core/main.py` file violates the Separation of Concerns principle by containing both UI rendering logic and application state management directly in the main function. This monolithic structure severely hinders maintainability, testability, and scalability.

### Objectives
1. Extract UI rendering logic into a dedicated `UIManager` class
2. Extract state management logic into a dedicated `StateManager` class
3. Refactor `main.py` to be a thin entry point that orchestrates between managers
4. Create comprehensive unit tests for the new manager classes
5. Maintain backward compatibility with existing functionality

### Acceptance Criteria
- [ ] `StateManager` class created in `src/core/state_manager.py`
- [ ] `UIManager` class created in `src/ui/ui_manager.py`
- [ ] `main.py` refactored to thin orchestrator pattern
- [ ] Unit tests for `StateManager` in `tests/unit/test_state_manager.py`
- [ ] Unit tests for `UIManager` in `tests/unit/test_ui_manager.py`
- [x] Integration tests for new `main.py` orchestration logic
- [ ] All existing functionality preserved (manual E2E test)
- [ ] No regression in existing automated tests

### Priority: P0 (Critical)
This refactoring is foundational for future development and addresses a critical architectural violation that makes the system difficult to understand, test, and extend.
