"""
Frame analysis and threat-event generation.

Three event sources are supported:
  - Live webcam   → generate_webcam_event()
  - Uploaded video → generate_video_event()
  - Simulation    → generate_simulation_event()
"""

import hashlib
import random
import time
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# YOLO (optional)
# ---------------------------------------------------------------------------

YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO  # noqa: F401
    YOLO_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Frame analysis
# ---------------------------------------------------------------------------

def get_frame_hash(frame) -> str:
    """Perceptual hash of a downscaled, posterised frame."""
    small = cv2.resize(frame, (16, 16))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = (gray // 32) * 32
    return hashlib.md5(gray.tobytes()).hexdigest()[:6]


def analyze_frame(frame) -> dict:
    """
    Extract motion / complexity metrics from a BGR frame and compute a
    composite threat score in [0, 1].
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    motion_intensity = float(np.std(gray))
    brightness_variance = float(np.var(gray))
    edge_density = float(cv2.Canny(gray, 50, 150).sum()) / (gray.shape[0] * gray.shape[1])
    histogram_variance = float(np.var(cv2.calcHist([gray], [0], None, [256], [0, 256])))
    contrast = float(gray.max() - gray.min())
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    color_variance = float(np.var(hsv[:, :, 1]))

    # Running motion trend
    history = st.session_state.motion_history
    history.append(motion_intensity)
    motion_trend = 0.0
    if len(history) >= 3:
        recent = list(history)[-3:]
        motion_trend = (recent[-1] - recent[0]) / 3

    # Composite threat score
    threat_score = 0.0
    if motion_intensity > 25:
        threat_score += 0.30
    if motion_intensity > 40:
        threat_score += 0.20
    if edge_density > 0.08:
        threat_score += 0.25
    if contrast > 120:
        threat_score += 0.20
    if abs(motion_trend) > 8:
        threat_score += 0.25
    if color_variance > 1000:
        threat_score += 0.15

    return {
        "motion_intensity": motion_intensity,
        "brightness_variance": brightness_variance,
        "edge_density": edge_density,
        "histogram_variance": histogram_variance,
        "contrast": contrast,
        "laplacian_var": laplacian_var,
        "color_variance": color_variance,
        "motion_trend": motion_trend,
        "threat_score": min(threat_score, 1.0),
        "complexity_score": (
            motion_intensity
            + brightness_variance / 100
            + edge_density * 1000
            + laplacian_var / 100
        ) / 4,
    }

# ---------------------------------------------------------------------------
# Cooldown helper
# ---------------------------------------------------------------------------

def _check_cooldown(event_type: str, min_cooldown: float | None = None) -> bool:
    """Return True if the cooldown has elapsed and reset the timer."""
    now = time.time()
    cooldown = min_cooldown or st.session_state.event_cooldown_setting
    last = st.session_state.event_cooldown.get(event_type, 0)
    if now - last < cooldown:
        return False
    st.session_state.event_cooldown[event_type] = now
    return True

# ---------------------------------------------------------------------------
# Event-type tables (shared by all sources)
# ---------------------------------------------------------------------------

_HIGH_MOTION_EVENTS = [
    {"type": "weapon_detected", "classes": ["knife", "gun", "weapon"], "severity": "critical", "weight": 0.4},
    {"type": "fight_detected",  "classes": ["person"],                  "severity": "critical", "weight": 0.3},
    {"type": "theft_detected",  "classes": ["laptop", "bag"],           "severity": "high",     "weight": 0.3},
]
_MED_MOTION_EVENTS = [
    {"type": "theft_detected",  "classes": ["laptop", "phone", "bag", "wallet"], "severity": "high",     "weight": 0.4},
    {"type": "weapon_detected", "classes": ["knife", "tool"],                    "severity": "critical", "weight": 0.3},
    {"type": "fight_detected",  "classes": ["person"],                           "severity": "critical", "weight": 0.3},
]
_HIGH_EDGE_EVENTS = [
    {"type": "weapon_detected",   "classes": ["knife", "tool", "weapon"],  "severity": "critical", "weight": 0.4},
    {"type": "unattended_object", "classes": ["backpack", "bag", "suitcase"], "severity": "medium", "weight": 0.4},
    {"type": "theft_detected",    "classes": ["laptop", "phone"],          "severity": "high",     "weight": 0.2},
]
_LOW_ACTIVITY_EVENTS = [
    {"type": "unattended_object", "classes": ["backpack", "bag", "package"], "severity": "medium", "weight": 0.6},
    {"type": "theft_detected",    "classes": ["phone", "wallet"],            "severity": "high",   "weight": 0.3},
    {"type": "weapon_detected",   "classes": ["tool"],                       "severity": "critical","weight": 0.1},
]


def _pick_event_type(analysis: dict) -> dict:
    """Choose an event-type table entry based on frame analysis."""
    mi = analysis["motion_intensity"]
    ed = analysis["edge_density"]
    if mi > 40 and ed > 0.12:
        pool = _HIGH_MOTION_EVENTS
    elif mi > 30:
        pool = _MED_MOTION_EVENTS
    elif ed > 0.10:
        pool = _HIGH_EDGE_EVENTS
    else:
        pool = _LOW_ACTIVITY_EVENTS
    return random.choices(pool, weights=[e["weight"] for e in pool])[0]


def _build_event(selected: dict, analysis: dict, source: str, extra: dict | None = None) -> dict:
    """Assemble a standardised event dict."""
    obj = random.choice(selected["classes"])
    h, w = 480, 640  # safe fallback dimensions
    x = random.randint(50, w - 50)
    y = random.randint(50, h - 50)

    base_conf = max(0.75, st.session_state.confidence_threshold)
    if analysis["motion_intensity"] > 35:
        base_conf += 0.10
    if analysis["edge_density"] > 0.12:
        base_conf += 0.08
    if analysis["threat_score"] > 0.4:
        base_conf += 0.05
    confidence = min(0.95, base_conf + random.uniform(0, 0.05))

    ts = extra.get("video_timestamp", "") if extra else ""
    suffix = f" at {ts}" if ts else f" at location (X:{x}, Y:{y})"

    details = {
        "weapon_detected":   f"{obj.title()} detected{suffix} — Confidence: {confidence:.0%}",
        "fight_detected":    f"Physical confrontation detected — Motion intensity: {analysis['motion_intensity']:.1f}",
        "theft_detected":    f"Suspicious rapid movement of {obj}{suffix}",
        "unattended_object": f"{obj.title()} left stationary{suffix}",
    }

    event = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event":      selected["type"],
        "track_id":   random.randint(1, 100),
        "class":      obj,
        "severity":   selected["severity"],
        "confidence": confidence,
        "details":    details.get(selected["type"], selected["type"]),
        "source":     source,
        "coordinates": f"({x}, {y})",
        # Analysis snapshot
        "motion_intensity": analysis["motion_intensity"],
        "threat_score":     analysis["threat_score"],
    }
    if extra:
        event.update(extra)
    return event

# ---------------------------------------------------------------------------
# Public generators
# ---------------------------------------------------------------------------

def generate_webcam_event(frame) -> dict | None:
    """Analyse a live webcam frame and maybe return a security event."""
    now = time.time()
    analysis = analyze_frame(frame)
    sensitivity = st.session_state.detection_sensitivity

    # Base probability
    p = (0.25 + analysis["threat_score"]) * sensitivity
    if analysis["motion_intensity"] > 25:
        p += 0.30 * sensitivity
    if analysis["brightness_variance"] > 2000:
        p += 0.20 * sensitivity
    if analysis["edge_density"] > 0.08:
        p += 0.25 * sensitivity
    if abs(analysis["motion_trend"]) > 8:
        p += 0.20 * sensitivity

    # Time-since-last boost
    gap = now - st.session_state.last_webcam_event
    if gap > 8:
        p += min(0.3, gap / 20) * sensitivity

    p = min(p, 0.85)

    # Debug log
    if st.session_state.debug_mode:
        st.session_state.frame_analysis_log.append({
            "frame_count":      st.session_state.frames_processed,
            "threat_score":     analysis["threat_score"],
            "event_probability": p,
            "motion":           analysis["motion_intensity"],
            "edges":            analysis["edge_density"],
            "time_since_last":  gap,
            "timestamp":        now,
        })

    if random.random() >= p:
        return None

    selected = _pick_event_type(analysis)
    if not _check_cooldown(selected["type"]):
        return None

    event = _build_event(selected, analysis, "webcam_live")
    st.session_state.last_webcam_event = now
    st.session_state.events_generated += 1
    return event


def generate_video_event(frame, frame_num: int, fps: float) -> dict | None:
    """Analyse a video frame and maybe return a security event."""
    analysis = analyze_frame(frame)
    current_time = frame_num / fps if fps > 0 else 0

    p = 0.15
    if analysis["motion_intensity"] > 30:
        p += 0.20
    if analysis["brightness_variance"] > 2000:
        p += 0.15

    if random.random() >= p:
        return None

    selected = _pick_event_type(analysis)
    if not _check_cooldown(selected["type"], min_cooldown=6.0):
        return None

    return _build_event(
        selected, analysis, "video_upload",
        extra={"video_timestamp": f"{current_time:.1f}s"},
    )


def generate_simulation_event() -> dict | None:
    """Generate a plausible synthetic security event (no camera required)."""
    now = time.time()
    if now - st.session_state.last_event_time < random.uniform(8, 12):
        return None

    pool = [
        {"type": "weapon_detected", "class": "knife",    "severity": "critical", "weight": 0.15},
        {"type": "weapon_detected", "class": "gun",      "severity": "critical", "weight": 0.08},
        {"type": "fight_detected",  "class": "person",   "severity": "critical", "weight": 0.12},
        {"type": "theft_detected",  "class": "laptop",   "severity": "high",     "weight": 0.20},
        {"type": "theft_detected",  "class": "handbag",  "severity": "high",     "weight": 0.15},
        {"type": "unattended_object", "class": "backpack","severity": "medium",  "weight": 0.30},
    ]
    selected = random.choices(pool, weights=[e["weight"] for e in pool])[0]
    if not _check_cooldown(selected["type"]):
        return None

    obj = selected["class"]
    conf = random.uniform(0.75, 0.95)
    x, y = random.randint(100, 600), random.randint(100, 400)

    details = {
        "weapon_detected":   f"{obj.title()} at (X:{x}, Y:{y}) — {conf:.0%} confidence",
        "fight_detected":    f"Physical altercation between multiple individuals (score: {random.uniform(0.7, 0.9):.2f})",
        "theft_detected":    f"Rapid movement of {obj} — speed: {random.uniform(120, 250):.1f} px/s",
        "unattended_object": f"{obj.title()} stationary for {random.randint(35, 120)}s",
    }

    event = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event":     selected["type"],
        "track_id":  random.randint(1, 100),
        "class":     obj,
        "severity":  selected["severity"],
        "confidence": conf,
        "details":   details[selected["type"]],
        "source":    "simulation",
        "coordinates": f"({x}, {y})",
    }

    st.session_state.last_event_time = now
    return event
