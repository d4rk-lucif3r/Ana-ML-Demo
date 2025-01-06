import streamlit as st

def init_session_state():
    """Initialize session state variables"""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = None
