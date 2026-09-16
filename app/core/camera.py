"""
Camera initialisation, capture, and teardown.
"""

import time
import cv2
import streamlit as st


def initialize_camera(camera_id: int = 0) -> bool:
    """
    Try each OpenCV backend in turn until one returns a readable frame.
    Returns True on success, False on failure.
    """
    try:
        if st.session_state.camera_cap is not None:
            st.session_state.camera_cap.release()
            time.sleep(0.5)

        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        cap = None
        working_backend = None

        for backend in backends:
            try:
                cap = cv2.VideoCapture(camera_id, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        working_backend = backend
                        break
                cap.release()
                cap = None
            except Exception:
                if cap:
                    cap.release()
                cap = None

        if cap is None or not cap.isOpened():
            st.error(
                f"❌ Cannot access camera {camera_id}. "
                "Check whether another application is using it."
            )
            return False

        # Best-effort property configuration
        for prop, val in [
            (cv2.CAP_PROP_FRAME_WIDTH, 640),
            (cv2.CAP_PROP_FRAME_HEIGHT, 480),
            (cv2.CAP_PROP_FPS, 15),
            (cv2.CAP_PROP_BUFFERSIZE, 1),
        ]:
            try:
                cap.set(prop, val)
            except Exception:
                pass

        ret, _ = cap.read()
        if not ret:
            cap.release()
            st.error(f"❌ Camera {camera_id} opened but cannot read frames.")
            return False

        st.session_state.camera_cap = cap
        st.session_state.camera_available = True
        st.session_state.camera_backend = working_backend
        return True

    except Exception as exc:
        st.error(f"❌ Camera initialisation error: {exc}")
        if "cap" in locals() and cap:
            cap.release()
        return False


def release_camera():
    """Safely release the camera and reset related state."""
    try:
        if st.session_state.get("camera_cap") is not None:
            st.session_state.camera_cap.release()
            st.session_state.camera_cap = None
        st.session_state.camera_available = False
    except Exception:
        pass


def read_frame():
    """
    Read a single frame from the active camera.
    Returns (True, frame) or (False, None).
    """
    cap = st.session_state.get("camera_cap")
    if cap is None:
        return False, None
    try:
        return cap.read()
    except Exception:
        return False, None
