"""
Session state initialization.
All st.session_state keys are declared here so they're easy to find and modify.
"""

import time
from collections import deque
import streamlit as st


def init_session_state():
    """Initialize all session state keys with their defaults (idempotent)."""
    defaults = {
        # Events & stats
        "events": [],
        "stats": {
            "total_events": 0,
            "weapons_detected": 0,
            "fights_detected": 0,
            "theft_attempts": 0,
            "unattended_objects": 0,
        },

        # System flags
        "is_running": False,
        "clear_events_flag": False,
        "simulation_mode": False,
        "debug_mode": True,

        # Camera state
        "camera_available": False,
        "camera_cap": None,
        "camera_backend": None,

        # Detection settings
        "detection_sensitivity": 1.2,
        "confidence_threshold": 0.75,
        "event_cooldown_setting": 5,

        # Timing / counters
        "last_event_time": time.time(),
        "last_webcam_event": 0,
        "last_status_message": 0,
        "webcam_frame_count": 0,
        "frames_processed": 0,
        "events_generated": 0,
        "consecutive_frames": 0,
        "ui_refresh_counter": 0,

        # History buffers (deque instances can't be serialised by Streamlit —
        # initialise once and leave alone on subsequent calls)
        "motion_history": deque(maxlen=10),
        "frame_hash_history": deque(maxlen=15),
        "frame_analysis_log": deque(maxlen=50),
        "last_motion_values": deque(maxlen=5),

        # Cooldown tracker
        "event_cooldown": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
