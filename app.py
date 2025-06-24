#!/usr/bin/env python3
"""
Main launcher for the AI CV Generator Streamlit application.
This file ensures proper initialization order for Streamlit.
"""

import streamlit as st
from src.core.main import main

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="AI CV Generator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

if __name__ == "__main__":
    main()
