# In src/frontend/callbacks.py
import streamlit as st
import asyncio
from typing import Optional
from src.orchestration.state import AgentState, UserFeedback

def handle_user_action(action: str, item_id: str):
    """
    Handle user actions like 'accept' or 'regenerate' for CV items.
    Updates the agent_state in session state and sets a flag to run the backend workflow.
    """
    # Get current state
    agent_state: Optional[AgentState] = st.session_state.get('agent_state')
    
    if not agent_state:
        st.error("No agent state found")
        return
    
    # Create user feedback based on action
    if action == 'accept':
        feedback_type = "accept"
        feedback_data = {"item_id": item_id}
    elif action == 'regenerate':
        feedback_type = "regenerate"
        feedback_data = {"item_id": item_id}
    else:
        st.error(f"Unknown action: {action}")
        return
    
    # Create UserFeedback object
    user_feedback = UserFeedback(
        feedback_type=feedback_type,
        data=feedback_data
    )
    
    # Update the agent state with user feedback
    agent_state.user_feedback = user_feedback
    st.session_state.agent_state = agent_state
    
    # Set flag to trigger backend workflow
    st.session_state.run_workflow = True
    
    # Show feedback to user
    if action == 'accept':
        st.success(f"✅ Item accepted")
    elif action == 'regenerate':
        st.info(f"🔄 Regenerating item...")
    
    # Trigger rerun to process the feedback
    st.rerun()


def handle_workflow_execution(trace_id: str = None):
    """
    Handle the execution of the LangGraph workflow.
    This function manages the async workflow execution and state updates.
    
    Args:
        trace_id: Optional trace ID for observability tracking
    """
    from src.orchestration.cv_workflow_graph import cv_graph_app
    from src.config.logging_config import setup_logging
    import uuid
    
    logger = setup_logging()
    
    # Generate trace_id if not provided
    if trace_id is None:
        trace_id = str(uuid.uuid4())
    
    try:
        # If it's the first run, create a new state from UI inputs
        if st.session_state.agent_state is None:
            from src.core.state_helpers import create_agent_state_from_ui
            st.session_state.agent_state = create_agent_state_from_ui()
        
        # Ensure trace_id is set in the agent state
        st.session_state.agent_state.trace_id = trace_id
        
        logger.info(
            "Invoking LangGraph workflow",
            extra={
                'trace_id': trace_id,
                'session_id': st.session_state.get('session_id'),
                'agent_state_keys': list(st.session_state.agent_state.model_dump().keys())
            }
        )
        
        # Invoke the LangGraph backend
        # The 'ainvoke' method takes the current state and returns the new state
        new_state_dict = asyncio.run(
            cv_graph_app.ainvoke(st.session_state.agent_state.model_dump())
        )
        
        # Overwrite the old state with the new state
        from src.orchestration.state import AgentState
        st.session_state.agent_state = AgentState.model_validate(new_state_dict)
        
        # Ensure trace_id is preserved
        st.session_state.agent_state.trace_id = trace_id
        
        logger.info(
            "LangGraph workflow completed",
            extra={
                'trace_id': trace_id,
                'session_id': st.session_state.get('session_id'),
                'has_errors': bool(st.session_state.agent_state.error_messages)
            }
        )
        
        # Clear feedback so the same action doesn't run again on the next rerun
        if st.session_state.agent_state.user_feedback:
            st.session_state.agent_state.user_feedback = None
            
    except Exception as e:
        logger.error(
            f"LangGraph workflow execution failed: {str(e)}",
            extra={
                'trace_id': trace_id,
                'session_id': st.session_state.get('session_id'),
                'error_type': type(e).__name__
            }
        )
        st.error(f"An error occurred during processing: {e}")
        # Optionally reset state on critical failure
        # st.session_state.agent_state = None
    finally:
        st.session_state.processing = False
        st.rerun()  # Force a re-render with the new state