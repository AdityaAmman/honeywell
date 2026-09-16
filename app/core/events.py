"""
Event storage and statistics helpers.
"""

import streamlit as st


def record_event(event: dict):
    """Append event to the log and update stats counters."""
    if event.get("is_status_message"):
        return
    st.session_state.events.append(event)
    st.session_state.stats["total_events"] += 1

    mapping = {
        "weapon_detected":   "weapons_detected",
        "fight_detected":    "fights_detected",
        "theft_detected":    "theft_attempts",
        "unattended_object": "unattended_objects",
    }
    key = mapping.get(event["event"])
    if key:
        st.session_state.stats[key] += 1


def clear_all_events():
    """Reset the event log and all counters."""
    st.session_state.events = []
    st.session_state.stats = {
        "total_events": 0,
        "weapons_detected": 0,
        "fights_detected": 0,
        "theft_attempts": 0,
        "unattended_objects": 0,
    }
    st.session_state.events_generated = 0
    st.session_state.frames_processed = 0
    st.session_state.event_cooldown.clear()
    st.session_state.frame_hash_history.clear()
    st.session_state.motion_history.clear()
    st.session_state.last_motion_values.clear()
    st.session_state.frame_analysis_log.clear()
