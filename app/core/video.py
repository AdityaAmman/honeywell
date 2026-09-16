"""
Uploaded-video processing pipeline.
"""

import os
import time

import cv2
import streamlit as st

from app.core.detection import generate_video_event
from app.core.events import record_event


def process_uploaded_video(video_path: str):
    """
    Process every N-th frame of the uploaded video, generating security events.
    Shows a progress bar and status text while running.
    """
    if not os.path.exists(video_path):
        st.error(f"Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video file.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_num = 0
    events_generated = 0
    last_event_frame = -1

    st.session_state.frame_hash_history.clear()
    st.session_state.event_cooldown.clear()

    while st.session_state.is_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        progress_bar.progress(min(frame_num / frame_count, 1.0))
        status_text.text(f"🎬 Analysing frame {frame_num}/{frame_count}…")

        # Attempt event generation every 30 frames with minimum gap
        if frame_num % 30 == 0 and (frame_num - last_event_frame) >= 90:
            event = generate_video_event(frame, frame_num, fps)
            if event:
                record_event(event)
                events_generated += 1
                last_event_frame = frame_num

        # Throttle to avoid hogging the CPU
        if frame_num % 15 != 0:
            continue
        time.sleep(0.01)

    cap.release()
    progress_bar.progress(1.0)
    status_text.text(
        f"✅ Analysis complete! Generated {events_generated} security events."
    )
