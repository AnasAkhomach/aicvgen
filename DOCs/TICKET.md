### **Work Item ID:** `REM-P1-01`
**Task Title:** `Execute Foundational Directory & Component Restructuring`

**Acceptance Criteria (AC):**
1.  The following new directories are created and exist in the `src/` directory:
    *   `src/core/containers/`
    *   `src/core/facades/`
    *   `src/core/managers/`
    *   `src/core/utils/`
2.  The following files are moved and renamed to their new locations:
    *   `src/core/container.py` is now at `src/core/containers/main_container.py`
    *   `src/core/workflow_manager.py` is now at `src/core/managers/workflow_manager.py`
    *   `src/services/session_manager.py` is now at `src/core/managers/session_manager.py`
3.  All files previously in `src/utils/` are now located in `src/core/utils/`.
4.  The old `src/utils/` directory is deleted.
5.  The application is in a non-runnable state due to broken imports, which is expected before the next ticket.

**Technical Implementation Notes:**
*   Use standard file system commands (`mkdir`, `mv`) to perform these operations.
*   This task is purely structural. Do not attempt to fix any broken Python imports.
*   The goal is to set the stage for the next ticket, which will handle all import-related fixes project-wide.
