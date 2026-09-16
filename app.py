"""
AI Surveillance Dashboard
Entry point — run with: streamlit run app.py
"""

from app.ui.dashboard import main_dashboard
from app.core.session import init_session_state

import streamlit as st

st.set_page_config(
    page_title="🔒 AI Surveillance Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

import atexit
from app.core.camera import release_camera
atexit.register(release_camera)


def main():
    init_session_state()
    main_dashboard()


if __name__ == "__main__":
    main()
