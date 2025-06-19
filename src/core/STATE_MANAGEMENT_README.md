# State Management Architecture

This document explains the three-layer state management architecture in the AI CV Generator application.

## Overview

The application uses a clear separation of concerns across three distinct state layers:

1. **Streamlit `session_state`** - UI state storage
2. **LangGraph `AgentState`** - Workflow execution state
3. **Filesystem `StateManager`** - Persistence layer

## Layer Responsibilities

### 1. Streamlit `session_state`

**Purpose:** Stores raw user input and UI control flags only.

**What it stores:**
- Raw form inputs (`job_description_input`, `cv_text_input`, `start_from_scratch_input`)
- UI control flags (`processing`, `run_workflow`, `stop_processing`)
- API configuration (`user_gemini_api_key`, `api_key_validated`)
- Progress tracking for UI display (`current_step`, `progress`, `status_message`)
- Session management (`session_id`)

**What it MUST NOT store:**
- Complex Pydantic models (`StructuredCV`, `JobDescriptionData`, `AgentState`)
- Processed workflow data
- Agent outputs

**Key Files:**
- `src/frontend/state_helpers.py` - Initializes session state
- `src/frontend/ui_components.py` - UI components that read/write session state

### 2. LangGraph `AgentState`

**Purpose:** Single source of truth for workflow execution. Contains all structured data and workflow state.

**What it manages:**
- Structured data models (`structured_cv`, `job_description_data`)
- Workflow control (`current_section_key`, `current_item_id`, `items_to_process_queue`)
- Agent outputs (`research_findings`, `quality_check_results`)
- Error tracking (`error_messages`)
- User feedback and final outputs

**Key Principles:**
- AgentState is passed between all workflow nodes
- Agents only read from and write to AgentState
- No direct access to session_state from agents
- Immutable updates through LangGraph state transitions

**Key Files:**
- `src/orchestration/state.py` - AgentState definition
- `src/core/state_helpers.py` - Contains `create_initial_agent_state()` function

### 3. Filesystem `StateManager`

**Purpose:** Persistence layer for saving/loading completed workflows.

**When to use:**
- Save final successful `AgentState` or `StructuredCV` at workflow completion
- Load previous state when starting a new session
- **NEVER** called from within agent `run_as_node` methods
- **ONLY** called from main application logic (`app.py` or orchestrator)

**Key Files:**
- `src/core/state_manager.py` - StateManager implementation

## Data Flow

```
User Input → session_state → create_initial_agent_state() → AgentState → Workflow → StateManager (save)
                ↑                                                                            ↓
            UI Updates ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←← StateManager (load)
```

## Key Functions

### `create_initial_agent_state()`

**Location:** `src/core/state_helpers.py`

**Purpose:** Single point of conversion from UI state to AgentState. This function:
- Reads raw input from `session_state`
- Creates properly structured Pydantic models
- Returns a fully initialized `AgentState`
- Establishes AgentState as the workflow's source of truth

## Best Practices

1. **UI Components:** Only read/write to `session_state` for raw inputs and UI flags
2. **Agents:** Only interact with `AgentState`, never access `session_state` directly
3. **State Conversion:** Always use `create_initial_agent_state()` to convert UI state to workflow state
4. **Persistence:** Only use `StateManager` from main application logic, not from agents
5. **Error Handling:** Store errors in `AgentState.error_messages`, display them via UI

## Validation

To verify proper state management:

1. **Code Review:** Ensure no complex Pydantic models in `session_state`
2. **Agent Review:** Confirm agents only use `AgentState`, not `StateManager`
3. **Integration Tests:** Verify workflows complete successfully with proper state transitions
4. **Persistence Tests:** Confirm `StateManager` saves/loads state correctly at workflow boundaries