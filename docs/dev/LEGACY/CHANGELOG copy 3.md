# CHANGELOG

## AD-01 - Enforce Constructor-Based Dependency Injection
- **Priority:** P1
- **Status:** DONE
- **Description:** Enforce constructor-based DI for all agents and services. Remove all `get_...()` service locator calls. Update `DependencyContainer` for instantiation and injection. Update tests to use constructor injection.
- **Implementation:**
    - All agent and service classes now require dependencies via their `__init__` constructor.
    - The DI container uses explicit factory functions for registration and instantiation.
    - All `get_...()` calls for business/service dependencies have been removed from agent/service logic.
- **Tests:**
    - Unit tests for agent constructors verify correct assignment of dependencies.
    - Integration tests inject mock services via constructors.
- **Notes:**
    - Logging and config `get_...()` calls remain as they are not business/service dependencies.
    - Local imports are used in DI registration to avoid circular import issues.

## WF-01 - Fix Race Condition in `route_after_qa` Router
- **Priority:** P2
- **Status:** DONE
- **Description:** Re-prioritize user feedback in `route_after_qa` before error checks. Add logging for path taken. Update tests for state with both error and user feedback.
- **Implementation:**
  - `route_after_qa` now checks for user feedback (regenerate) before error messages.
  - Logging added for both user feedback and error paths.
- **Tests:**
  - `test_route_after_qa_regenerate_takes_precedence_over_error` verifies user intent is honored over errors.
- **Notes:**
  - No further changes required; logic and tests already correct.

## CS-01 - Decompose Monolithic `EnhancedLLMService`
- **Priority:** P1
- **Status:** DONE
- **Description:** Decompose `EnhancedLLMService` into smaller components (`LLMClient`, `RateLimiter`, `AdvancedCache`). Refactor orchestration logic. Update DI container. Add unit tests for new components.
- **Implementation:**
  - `LLMClient` handles direct API calls.
  - `LLMRetryHandler` wraps retry logic using tenacity.
  - `EnhancedLLMService` composes these and orchestrates cache, rate limit, retry, and error handling.
  - DI container injects all dependencies.
- **Tests:**
  - Unit tests for `LLMClient` and `LLMRetryHandler` (`test_llmclient_and_retryhandler`).
  - Orchestration logic in `EnhancedLLMService` is covered by unit tests.
- **Notes:**
  - No further decomposition required; code and tests match blueprint.

## D-03 & D-01 - Consolidate Duplicated Logic
- **Priority:** P2
- **Status:** DONE
- **Description:** Centralize error classification and fallback logic. Create `error_classification.py`. Refactor agents/services to use new utilities. Remove local fallback logic. Update error handling to use `ErrorRecoveryService`.
- **Implementation:**
  - All error classification logic is centralized in `src/utils/error_classification.py` and used by services/agents.
  - Local fallback logic removed from agents; all fallback handled by `ErrorRecoveryService`.
  - `RateLimiter` and `EnhancedLLMService` refactored to use centralized utilities.
- **Tests:**
  - Unit tests for all error classification utilities (`test_error_classification.py`).
- **Notes:**
  - No duplicated error logic remains; all error handling is now consistent and maintainable.

## PB-01 - Make StateManager I/O Asynchronous
- **Priority:** P2
- **Status:** DONE
- **Description:** Convert `StateManager` I/O to async using thread executor. Update all usages and tests.
- **Implementation:**
  - `save_state` and `load_state` are async and use `asyncio.to_thread` for file I/O.
  - All usages and tests updated to use async interface.
- **Tests:**
  - Unit test `test_state_manager_async_save_and_load` verifies async save/load.
- **Notes:**
  - No further changes required; implementation matches blueprint and is production-ready.

