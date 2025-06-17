"""Callback Functions for UI Interactions in the AI CV Generator.

This module contains callback functions that handle user interactions,
workflow triggers, and backend integration.
"""

import streamlit as st
import asyncio
import time
from typing import Dict, Any, Optional
from src.config.logging_config import setup_logging
from src.core.state_helpers import (
    update_processing_state,
    update_token_usage,
    check_budget_limits,
    add_error_message,
    set_processing_state,
    create_agent_state_from_ui
)
from src.core.enhanced_orchestrator import EnhancedOrchestrator
from src.core.state_manager import StateManager
from src.core.enhanced_cv_system import EnhancedCVConfig, IntegrationMode, get_enhanced_cv_integration

# Initialize logging
logger = setup_logging()


def handle_cv_generation() -> None:
    """Handle the CV generation workflow trigger.
    
    This function is called by the old UI components and simply sets the
    run_workflow flag for the new main.py controller to handle.
    """
    try:
        # Check if user has provided API key
        if not st.session_state.user_gemini_api_key:
            st.error("⚠️ Please enter your Gemini API key in the sidebar before starting.")
            return

        # Validate inputs
        job_description = st.session_state.get('job_description', '')
        cv_content = st.session_state.get('cv_text', '')
        start_from_scratch = st.session_state.get('start_from_scratch', False)
        
        if not job_description.strip():
            st.error("⚠️ Please provide a job description.")
            return
            
        if not cv_content.strip() and not start_from_scratch:
            st.error("⚠️ Please provide CV content or check 'Start from scratch'.")
            return

        # Store inputs for the new workflow
        st.session_state.job_description_input = job_description
        st.session_state.cv_text_input = cv_content
        st.session_state.start_from_scratch_input = start_from_scratch
        
        # Set flag to trigger workflow in main.py
        st.session_state.run_workflow = True
        
        # Mark API key as validated
        st.session_state.api_key_validated = True
        
        logger.info("CV generation workflow triggered")

    except Exception as e:
        logger.error(f"Error in CV generation trigger: {str(e)}")
        st.error(f"❌ Error starting CV generation: {str(e)}")


def _run_cv_generation_workflow(orchestrator: EnhancedOrchestrator, agent_state) -> None:
    """Run the CV generation workflow asynchronously.
    
    Args:
        orchestrator: The enhanced orchestrator instance
        agent_state: The initial agent state
    """
    async def run_workflow():
        try:
            # Update progress
            update_processing_state(progress=10, message="Starting workflow...")
            
            # Check budget limits periodically
            def check_budget():
                budget_status = check_budget_limits()
                if budget_status["session_exceeded"] or budget_status["daily_exceeded"]:
                    st.session_state.stop_processing = True
                    return False
                return True

            # Run the main workflow
            update_processing_state(progress=20, message="Processing job description...")
            
            if not check_budget() or st.session_state.stop_processing:
                raise Exception("Processing stopped due to budget limits or user request")

            # Execute the workflow through the orchestrator
            result = await orchestrator.run_full_workflow(agent_state)
            
            update_processing_state(progress=80, message="Finalizing results...")
            
            if result and result.get("success"):
                # Update session state with results
                if "structured_cv" in result:
                    st.session_state.state_manager.current_cv = result["structured_cv"]
                
                update_processing_state(
                    processing=False,
                    progress=100,
                    message="CV generation completed successfully!"
                )
                
                # Update token usage (placeholder - should be tracked by orchestrator)
                update_token_usage(session_tokens=1000, daily_tokens=1000)
                
                st.success("✅ CV generation completed successfully!")
                
                # Switch to review tab
                st.session_state.active_tab = "review"
                
            else:
                error_msg = result.get("error", "Unknown error occurred") if result else "No result returned"
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}")
            add_error_message(f"Workflow execution failed: {str(e)}")
            update_processing_state(
                processing=False,
                progress=0,
                message=f"Error: {str(e)}"
            )
            st.error(f"❌ CV generation failed: {str(e)}")

    # Run the async workflow
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_workflow())
    except Exception as e:
        logger.error(f"Error in async workflow execution: {str(e)}")
        add_error_message(f"Async workflow failed: {str(e)}")
        update_processing_state(processing=False, message="Async execution failed")
        st.error(f"❌ Async execution failed: {str(e)}")
    finally:
        loop.close()
        # Force UI update
        st.rerun()


def handle_item_regeneration(item_id: str, parent_section: Dict[str, Any], parent_subsection: Optional[Dict[str, Any]], item: Dict[str, Any], state_manager: StateManager) -> None:
    """Handle regeneration of a single item.
    
    Args:
        item_id: Unique identifier for the item
        parent_section: The parent section containing the item
        parent_subsection: The parent subsection (if any)
        item: The item to regenerate
        state_manager: The state manager instance
    """
    try:
        # Set item status to processing
        item["status"] = "processing"
        set_processing_state(item_id, True)

        # Show processing indicator
        with st.spinner(f"Regenerating {item.get('title', 'item')}..."):
            # Get enhanced CV integration
            if "enhanced_cv_integration_config" not in st.session_state:
                config = EnhancedCVConfig(
                    mode=IntegrationMode.PRODUCTION,
                    enable_caching=True,
                    enable_monitoring=True,
                )
                st.session_state.enhanced_cv_integration_config = config.to_dict()

            config = EnhancedCVConfig.from_dict(st.session_state.enhanced_cv_integration_config)
            enhanced_cv_integration = get_enhanced_cv_integration(config)

            # Get the orchestrator
            orchestrator = enhanced_cv_integration.orchestrator

            # Process the single item using the new interface
            async def process_item():
                try:
                    # The new orchestrator method takes only item_id and uses state manager internally
                    updated_cv = await orchestrator.process_single_item(str(item_id))
                    return {"success": True, "structured_cv": updated_cv}
                except Exception as e:
                    logger.error(f"Error processing single item {item_id}: {str(e)}")
                    return {"success": False, "error": str(e)}

            # Run the async processing
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(process_item())
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")
                result = {"success": False, "error": str(e)}
            finally:
                loop.close()

            # Handle the result
            if result and result.get("success"):
                item["status"] = "completed"
                st.success(f"✅ Successfully regenerated {item.get('title', 'item')}!")

                # Check for raw LLM output in the updated CV structure
                updated_cv = result.get("structured_cv")
                if updated_cv and hasattr(updated_cv, 'model_dump'):
                    cv_data = updated_cv.model_dump()
                    # Look for raw output in the specific item that was processed
                    raw_output = find_raw_output_for_item(cv_data, str(item_id))
                    if raw_output:
                        with st.expander("🔍 View Raw LLM Output", expanded=False):
                            st.code(raw_output, language="text")

            else:
                item["status"] = "error"
                error_msg = result.get("error", "Unknown error occurred") if result else "No result returned"
                st.error(f"❌ Failed to regenerate item: {error_msg}")

    except Exception as e:
        logger.error(f"Error in handle_item_regeneration: {str(e)}")
        item["status"] = "error"
        st.error(f"❌ Error processing item: {str(e)}")
    finally:
        set_processing_state(item_id, False)
        # Force a rerun to update the UI
        time.sleep(1)
        st.rerun()


def handle_section_regeneration(section_id: str, section: Dict[str, Any], state_manager: StateManager) -> None:
    """Handle regeneration of an entire section.
    
    Args:
        section_id: Unique identifier for the section
        section: The section to regenerate
        state_manager: The state manager instance
    """
    try:
        # Set section status to processing
        section["status"] = "processing"
        
        with st.spinner(f"Regenerating section: {section.get('title', 'Unknown')}..."):
            # Get all items in the section
            items_to_process = []
            
            if "items" in section:
                items_to_process.extend(section["items"])
            elif "subsections" in section:
                for subsection in section["subsections"]:
                    if "items" in subsection:
                        items_to_process.extend(subsection["items"])
            
            # Process each item in the section
            for item in items_to_process:
                item_id = item.get("id")
                if item_id:
                    handle_item_regeneration(item_id, section, None, item, state_manager)
            
            section["status"] = "completed"
            st.success(f"✅ Successfully regenerated section: {section.get('title', 'Unknown')}!")
            
    except Exception as e:
        logger.error(f"Error in handle_section_regeneration: {str(e)}")
        section["status"] = "error"
        st.error(f"❌ Error regenerating section: {str(e)}")
    finally:
        # Force a rerun to update the UI
        st.rerun()


def handle_save_session() -> None:
    """Handle saving the current session."""
    try:
        from src.core.session_utils import save_session
        
        if save_session(st.session_state.state_manager):
            st.success("Session saved successfully!")
        else:
            st.error("Failed to save session")
            
    except Exception as e:
        logger.error(f"Error saving session: {str(e)}")
        st.error(f"❌ Error saving session: {str(e)}")


def handle_load_session(session_id: str) -> None:
    """Handle loading an existing session.
    
    Args:
        session_id: The session ID to load
    """
    try:
        from src.core.session_utils import load_session
        
        if load_session(session_id):
            st.success("Session loaded successfully!")
            st.rerun()
        else:
            st.error("Failed to load session")
            
    except Exception as e:
        logger.error(f"Error loading session: {str(e)}")
        st.error(f"❌ Error loading session: {str(e)}")


def handle_new_session() -> None:
    """Handle creating a new session."""
    try:
        from src.core.state_helpers import reset_session_state
        
        # Reset session state while preserving budget limits
        reset_session_state(preserve_keys=["session_token_limit", "daily_token_limit"])
        
        st.success("New session created!")
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error creating new session: {str(e)}")
        st.error(f"❌ Error creating new session: {str(e)}")


def handle_stop_processing() -> None:
    """Handle stopping the current processing."""
    try:
        st.session_state.stop_processing = True
        update_processing_state(processing=False, message="Processing stopped by user")
        
        st.warning("⏹️ Processing stopped by user")
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error stopping processing: {str(e)}")
        st.error(f"❌ Error stopping processing: {str(e)}")


def handle_update_budget_limits(session_limit: int, daily_limit: int) -> None:
    """Handle updating budget limits.
    
    Args:
        session_limit: New session token limit
        daily_limit: New daily token limit
    """
    try:
        st.session_state.session_token_limit = session_limit
        st.session_state.daily_token_limit = daily_limit
        
        st.success("Budget limits updated successfully!")
        logger.info(f"Budget limits updated: session={session_limit}, daily={daily_limit}")
        
    except Exception as e:
        logger.error(f"Error updating budget limits: {str(e)}")
        st.error(f"❌ Error updating budget limits: {str(e)}")


def handle_api_key_update(api_key: str) -> None:
    """Handle API key update.
    
    Args:
        api_key: The new API key
    """
    try:
        if api_key and api_key != st.session_state.user_gemini_api_key:
            st.session_state.user_gemini_api_key = api_key
            st.session_state.api_key_validated = False  # Reset validation when key changes
            
            # Clear orchestrator config to force reinitialization
            st.session_state.orchestrator_config = None
            st.session_state.enhanced_cv_integration_config = None
            
            logger.info("API key updated")
            
    except Exception as e:
        logger.error(f"Error updating API key: {str(e)}")
        st.error(f"❌ Error updating API key: {str(e)}")


def find_raw_output_for_item(cv_data: Dict[str, Any], item_id: str) -> Optional[str]:
    """Helper function to find raw LLM output for a specific item in the CV structure.
    
    Args:
        cv_data: The CV data structure
        item_id: The item ID to search for
        
    Returns:
        The raw LLM output for the item, if found
    """
    try:
        # Recursively search through the CV structure for the item with matching ID
        def search_for_item(data, target_id):
            if isinstance(data, dict):
                # Check if this is an item with the target ID
                if data.get('id') == target_id:
                    return data.get('raw_llm_output')
                # Recursively search in nested structures
                for key, value in data.items():
                    result = search_for_item(value, target_id)
                    if result:
                        return result
            elif isinstance(data, list):
                for item in data:
                    result = search_for_item(item, target_id)
                    if result:
                        return result
            return None

        return search_for_item(cv_data, item_id)
    except Exception as e:
        logger.error(f"Error finding raw output for item {item_id}: {str(e)}")
        return None


def handle_export_cv(format_type: str) -> None:
    """Handle CV export in different formats.
    
    Args:
        format_type: The export format ('markdown', 'text', 'pdf')
    """
    try:
        if not st.session_state.state_manager or not st.session_state.state_manager.current_cv:
            st.error("No CV data available for export")
            return
        
        cv_data = st.session_state.state_manager.current_cv
        
        if format_type == "markdown":
            # Generate markdown content
            from src.core.ui_components import _generate_cv_markdown
            
            if hasattr(cv_data, 'model_dump'):
                content_data = cv_data.model_dump().get("content", {})
            else:
                content_data = getattr(cv_data, 'content', {})
            
            markdown_content = _generate_cv_markdown(content_data, show_raw_output=False)
            
            st.download_button(
                label="Download Markdown",
                data=markdown_content,
                file_name=f"tailored_cv_{time.strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )
            
        elif format_type == "text":
            # Generate plain text content
            from src.core.ui_components import _generate_cv_markdown
            
            if hasattr(cv_data, 'model_dump'):
                content_data = cv_data.model_dump().get("content", {})
            else:
                content_data = getattr(cv_data, 'content', {})
            
            markdown_content = _generate_cv_markdown(content_data, show_raw_output=False)
            plain_text = (
                markdown_content.replace("#", "")
                .replace("*", "")
                .replace("---", "")
            )
            
            st.download_button(
                label="Download Text",
                data=plain_text,
                file_name=f"tailored_cv_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )
            
        elif format_type == "pdf":
            st.info("PDF export coming soon!")
            
        else:
            st.error(f"Unsupported export format: {format_type}")
            
    except Exception as e:
        logger.error(f"Error exporting CV as {format_type}: {str(e)}")
        st.error(f"❌ Error exporting CV: {str(e)}")