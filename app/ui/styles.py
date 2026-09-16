"""
Custom CSS injected into the Streamlit app.
"""

import streamlit as st

_CSS = """
<style>
.main-header {
    text-align: center;
    color: #1f77b4;
    margin-bottom: 2rem;
    font-size: 2.5rem;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}
.alert-critical {
    background: linear-gradient(90deg, #ffebee 0%, #ffcdd2 100%);
    border-left: 5px solid #f44336;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(244, 67, 54, 0.2);
}
.alert-high {
    background: linear-gradient(90deg, #fff3e0 0%, #ffe0b2 100%);
    border-left: 5px solid #ff9800;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(255, 152, 0, 0.2);
}
.alert-medium {
    background: linear-gradient(90deg, #f3e5f5 0%, #e1bee7 100%);
    border-left: 5px solid #9c27b0;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(156, 39, 176, 0.2);
}
.debug-panel {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 5px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-family: monospace;
}
.status-running {
    color: #4caf50;
    font-weight: bold;
    font-size: 1.1rem;
}
.status-stopped {
    color: #f44336;
    font-weight: bold;
    font-size: 1.1rem;
}
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)
