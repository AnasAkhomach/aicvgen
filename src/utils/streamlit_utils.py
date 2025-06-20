"""Streamlit utility functions."""

import streamlit as st


def configure_page(
    page_title: str = "AI CV Generator",
    page_icon: str = "📄",
    layout: str = "wide",
    initial_sidebar_state: str = "expanded"
) -> bool:
    """Configure Streamlit page settings if not already configured.
    
    Args:
        page_title: Title for the page
        page_icon: Icon for the page
        layout: Layout mode ('wide' or 'centered')
        initial_sidebar_state: Initial sidebar state
        
    Returns:
        True if configuration was set, False if already configured
    """
    try:
        # Check if page config is already set using session state
        if "_page_config_set" not in st.session_state:
            st.set_page_config(
                page_title=page_title,
                page_icon=page_icon,
                layout=layout,
                initial_sidebar_state=initial_sidebar_state
            )
            st.session_state["_page_config_set"] = True
            return True
        return False
    except (RuntimeError, ValueError):
        # If config is already set or other error, continue
        return False