"""
Main dashboard — ties all UI and core modules together.
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from app.core.camera import read_frame, release_camera
from app.core.detection import (
    analyze_frame,
    generate_simulation_event,
    generate_webcam_event,
)
from app.core.events import clear_all_events, record_event
from app.core.video import process_uploaded_video
from app.ui.alerts import display_live_alerts, display_metrics
from app.ui.analytics import display_analytics, display_debug_panel
from app.ui.sidebar import create_sidebar
from app.ui.styles import inject_css


def main_dashboard():
    inject_css()

    st.markdown('<h1 class="main-header">🔒 AI Surveillance Dashboard</h1>', unsafe_allow_html=True)
    st.caption("Real-time threat detection with webcam, video upload, and simulation support.")

    source_type = create_sidebar()

    # ── Clear-events deferred flag ────────────────────────────────────────
    if st.session_state.get("clear_events_flag"):
        clear_all_events()
        st.session_state.clear_events_flag = False
        st.success("✅ All events cleared!")
        time.sleep(0.4)
        st.rerun()

    # ── Control bar ───────────────────────────────────────────────────────
    _render_control_bar(source_type)

    # ── System status line ────────────────────────────────────────────────
    status_col1, status_col2 = st.columns([1, 3])
    with status_col1:
        if st.session_state.is_running:
            st.markdown('<p class="status-running">🟢 SYSTEM ACTIVE</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-stopped">🔴 SYSTEM INACTIVE</p>', unsafe_allow_html=True)
    with status_col2:
        if st.session_state.is_running:
            sens = st.session_state.detection_sensitivity
            conf = st.session_state.confidence_threshold
            if st.session_state.camera_available:
                st.success(
                    f"📹 Camera: ACTIVE | Sensitivity: {sens:.1f}× | "
                    f"Confidence: {conf:.0%} | Events: {st.session_state.events_generated}"
                )
            elif st.session_state.get("simulation_mode"):
                st.info(
                    f"🎮 Simulation: ACTIVE | Sensitivity: {sens:.1f}× | "
                    f"Events: {st.session_state.events_generated}"
                )

    st.divider()

    # ── Live event generation ─────────────────────────────────────────────
    if st.session_state.is_running:
        if st.session_state.get("simulation_mode"):
            event = generate_simulation_event()
            if event:
                record_event(event)
                st.session_state.events_generated += 1

        elif (
            source_type == "Local Webcam"
            and st.session_state.camera_available
        ):
            _process_webcam_tick()

    # ── Metrics ───────────────────────────────────────────────────────────
    display_metrics(st.session_state.stats)
    st.divider()

    # ── Main content ──────────────────────────────────────────────────────
    col_alerts, col_info = st.columns([2, 1])

    with col_alerts:
        display_live_alerts(st.session_state.events)

    with col_info:
        _render_info_panel()

    # ── Debug / analytics ─────────────────────────────────────────────────
    if st.session_state.debug_mode:
        st.divider()
        display_debug_panel()

    display_analytics(st.session_state.events)

    # ── Auto-refresh while running ────────────────────────────────────────
    if st.session_state.is_running:
        time.sleep(0.1)
        st.rerun()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _render_control_bar(source_type: str):
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

    with col1:
        st.subheader("🎮 Control Centre")

    with col2:
        if st.button("▶️ Start Monitoring", type="primary", use_container_width=True):
            _start_monitoring(source_type)

    with col3:
        if st.button("⏹️ Stop Monitoring", use_container_width=True):
            st.session_state.is_running = False
            st.warning("🔴 System STOPPED")
            st.rerun()

    with col4:
        if st.button("🎲 Test Event", use_container_width=True):
            test_event = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "event":     "weapon_detected",
                "track_id":  999,
                "class":     "knife",
                "severity":  "critical",
                "confidence": 0.89,
                "details":   "Manual test — Knife at (X:320, Y:240)",
                "source":    "manual_test",
                "coordinates": "(320, 240)",
            }
            record_event(test_event)
            st.success("🚨 Test event generated!")
            st.rerun()

    with col5:
        if st.button("🗑️ Clear Events", use_container_width=True):
            st.session_state.clear_events_flag = True
            st.rerun()


def _start_monitoring(source_type: str):
    can_start = False
    reason = ""

    if source_type == "Local Webcam" and st.session_state.camera_available:
        can_start = True
        reason = "Local Webcam"
    elif source_type == "Upload Video" and "uploaded_video_path" in st.session_state:
        can_start = True
        reason = "Video Analysis"
    elif source_type == "Simulation Mode":
        can_start = True
        reason = "Simulation Mode"

    if not can_start:
        st.error("❌ Cannot start — check video source configuration")
        return

    # Reset counters
    st.session_state.is_running = True
    st.session_state.event_cooldown.clear()
    st.session_state.frame_hash_history.clear()
    st.session_state.consecutive_frames = 0
    st.session_state.motion_history.clear()
    st.session_state.frames_processed = 0
    st.session_state.events_generated = 0

    if reason == "Video Analysis":
        process_uploaded_video(st.session_state.uploaded_video_path)

    st.success(f"🟢 AI Surveillance ACTIVE! (Source: {reason})")
    st.rerun()


def _process_webcam_tick():
    """Read one frame and maybe generate an event."""
    ret, frame = read_frame()
    if not ret or frame is None:
        st.error("📹 Camera error — stopping.")
        st.session_state.is_running = False
        return

    st.session_state.webcam_frame_count += 1
    st.session_state.frames_processed += 1

    analysis = analyze_frame(frame)
    st.session_state.last_motion_values.append(analysis["motion_intensity"])

    event = generate_webcam_event(frame)
    if event:
        record_event(event)

    # Periodic status
    if st.session_state.webcam_frame_count % 45 == 0:
        avg_motion = np.mean(list(st.session_state.last_motion_values))
        st.success(
            f"📹 Surveillance Active — Motion: {avg_motion:.1f} | "
            f"Frames: {st.session_state.frames_processed} | "
            f"Events: {st.session_state.events_generated}"
        )


def _render_info_panel():
    st.subheader("📊 System Analytics")

    if st.session_state.is_running:
        sens = st.session_state.detection_sensitivity
        conf = st.session_state.confidence_threshold
        cool = st.session_state.event_cooldown_setting

        if st.session_state.camera_available:
            st.info(
                f"🛡️ **Live Camera Active**\n\n"
                f"• Sensitivity: {sens:.1f}×\n"
                f"• Confidence: {conf:.0%}\n"
                f"• Cooldown: {cool}s\n"
                f"• Frames: {st.session_state.frames_processed}\n"
                f"• Events: {st.session_state.events_generated}"
            )
            if st.session_state.frames_processed > 0:
                rate = st.session_state.events_generated / st.session_state.frames_processed * 100
                if rate > 0:
                    st.success(f"📈 Detection Rate: {rate:.3f}%")
                else:
                    st.warning("📉 No events yet")
                    st.info(
                        "💡 **Tips:**\n"
                        "• Move objects in camera view\n"
                        "• Increase Detection Sensitivity\n"
                        "• Lower Confidence Threshold"
                    )
        elif st.session_state.get("simulation_mode"):
            st.info(
                f"🎮 **Simulation Mode Active**\n\n"
                f"• Generating realistic events\n"
                f"• Sensitivity: {sens:.1f}×\n"
                f"• Events: {st.session_state.events_generated}"
            )

    # Export
    events = st.session_state.events
    if events:
        df = pd.DataFrame(events)
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Export Event Log",
            data=csv,
            file_name=f"surveillance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.metric("📈 Events Logged", len(events))
        last = events[-1]
        st.write(f"**🕐 Last Activity:** {last['timestamp']}")
        st.write(f"**📋 Event Type:** {last['event'].replace('_', ' ').title()}")
        st.write(f"**⚡ Severity:** {last['severity'].title()}")
        st.write(f"**🎯 Confidence:** {last.get('confidence', 0):.0%}")
        st.write(f"**📡 Source:** {last.get('source', '').replace('_', ' ').title()}")
    else:
        st.info("📊 No events to export yet")

    st.divider()
    st.subheader("🔧 System Features")
    for feature in [
        "Real-time AI Detection", "Multi-camera Support", "Video Analysis",
        "Configurable Sensitivity", "Event Export", "Debug Mode",
    ]:
        st.success(f"✅ {feature}")
