"""
Configuration sidebar.
"""

import os
import tempfile

import cv2
import streamlit as st

from app.core.camera import initialize_camera, release_camera, read_frame
from app.core.detection import analyze_frame, YOLO_AVAILABLE


def create_sidebar():
    """Render the sidebar and return the current source_type string."""
    with st.sidebar:
        st.header("⚙️ Detection Control Panel")

        st.session_state.debug_mode = st.toggle(
            "🔧 Debug Mode", value=st.session_state.debug_mode
        )
        st.divider()

        # ── Video source ──────────────────────────────────────────────────
        st.subheader("📹 Video Input")
        source_type = st.selectbox(
            "Source Type:", ["Local Webcam", "Upload Video", "Simulation Mode"]
        )

        # Reset when source type changes
        prev = st.session_state.get("current_source_type")
        if prev != source_type:
            _handle_source_change(prev, source_type)

        if source_type == "Local Webcam":
            _webcam_controls()
        elif source_type == "Upload Video":
            _upload_controls()
        else:  # Simulation Mode
            st.info(
                "🎮 **Simulation Mode**\n\n"
                "Generates realistic security events for demonstration purposes."
            )
            st.session_state.simulation_mode = True

        st.divider()

        # ── Detection settings ────────────────────────────────────────────
        st.subheader("🎯 AI Detection Settings")
        st.multiselect(
            "Active Detections:",
            ["🔪 Weapon Detection", "🥊 Violence Detection",
             "💰 Theft Detection", "📦 Unattended Objects"],
            default=["🔪 Weapon Detection", "🥊 Violence Detection",
                     "💰 Theft Detection", "📦 Unattended Objects"],
        )

        st.session_state.confidence_threshold = st.slider(
            "AI Confidence Threshold:", 0.1, 1.0, 0.75, 0.05,
            help="Lower → more sensitive",
        )
        st.session_state.detection_sensitivity = st.slider(
            "Detection Sensitivity:", 0.5, 2.0, 1.2, 0.1,
            help="Higher → more events generated",
        )

        st.subheader("⏱️ Event Control")
        st.session_state.event_cooldown_setting = st.slider(
            "Event Cooldown (seconds):", 1, 15, 5, 1,
            help="Minimum time between similar events",
        )

        st.divider()

        # ── System status ─────────────────────────────────────────────────
        st.subheader("📊 System Status")
        if YOLO_AVAILABLE:
            st.success("✅ YOLO AI Model: Ready")
        else:
            st.warning("⚠️ YOLO AI: Simulated")

        if st.session_state.camera_available:
            st.success("✅ Camera: Connected")
            if st.session_state.is_running:
                st.success("✅ Processing: Active")
        else:
            st.info("📷 Camera: Not Connected")

        live = [e for e in st.session_state.events if e.get("source") == "webcam_live"]
        if live:
            st.info(f"🔴 Live Events: {len(live)}")

        from datetime import datetime
        st.info(f"🕒 Time: {datetime.now().strftime('%H:%M:%S')}")

    return source_type


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _handle_source_change(prev: str | None, new: str):
    """Clean up after switching video source."""
    if "uploaded_video_path" in st.session_state:
        try:
            os.unlink(st.session_state.uploaded_video_path)
        except Exception:
            pass
        del st.session_state.uploaded_video_path

    if prev == "Local Webcam":
        st.session_state.is_running = False
        release_camera()

    st.session_state.current_source_type = new
    st.session_state.simulation_mode = (new == "Simulation Mode")
    st.session_state.event_cooldown.clear()
    st.session_state.frame_hash_history.clear()
    st.rerun()


def _webcam_controls():
    camera_id = st.selectbox(
        "Camera ID:", [0, 1, 2],
        help="Try different IDs if camera 0 doesn't work",
    )

    st.info(
        "💡 **Camera Tips:**\n"
        "• Close other camera apps\n"
        "• Try different Camera IDs\n"
        "• Restart if camera gets stuck"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔌 Initialize Camera", use_container_width=True):
            with st.spinner("Initializing camera…"):
                if initialize_camera(camera_id):
                    st.success("✅ Camera initialized!")
                    st.rerun()
    with col2:
        if st.button("📷 Release Camera", use_container_width=True):
            st.session_state.is_running = False
            release_camera()
            st.success("Camera released")
            st.rerun()

    if st.session_state.camera_available:
        if st.button("📸 Test Camera", use_container_width=True):
            ret, frame = read_frame()
            if ret and frame is not None:
                analysis = analyze_frame(frame)
                st.success("✅ Camera test successful!")
                st.info(
                    f"📏 Frame: {frame.shape[1]}×{frame.shape[0]} "
                    f"| Motion: {analysis['motion_intensity']:.1f}"
                )
            else:
                st.error("❌ Camera test failed")

        st.success("✅ Camera ready for monitoring")
    else:
        st.warning("📷 Camera not initialized")


def _upload_controls():
    uploaded_file = st.file_uploader(
        "📁 Upload video file:", type=["mp4", "avi", "mov", "mkv"]
    )
    if uploaded_file is None:
        st.info("👆 Please upload a video file")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.session_state.uploaded_video_path = temp_path

    cap = cv2.VideoCapture(temp_path)
    if cap.isOpened():
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = n_frames / fps if fps > 0 else 0
        cap.release()
        st.success(f"✅ Uploaded: **{uploaded_file.name}**")
        st.info(f"Duration: {duration:.1f}s | {fps} FPS | {n_frames} frames")
    else:
        cap.release()
        st.error("❌ Could not read video file")
