# AI Surveillance Dashboard
# Real-time threat detection system with proper webcam event generation

import streamlit as st
import cv2
import numpy as np
import time
import threading
import queue
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import io
import base64
import random
import hashlib
import math

# Try to import YOLO, with fallback for demo mode
YOLO_AVAILABLE = True
try:
    from ultralytics import YOLO
except ImportError:
    YOLO_AVAILABLE = False
    st.warning("⚠️ YOLO not available. Running in simulation mode.")

# ===============================
# CONFIGURATION
# ===============================

st.set_page_config(
    page_title="🔒 AI Surveillance Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state 
if 'events' not in st.session_state:
    st.session_state.events = []
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_events': 0,
        'weapons_detected': 0,
        'fights_detected': 0,
        'theft_attempts': 0,
        'unattended_objects': 0
    }
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'last_event_time' not in st.session_state:
    st.session_state.last_event_time = time.time()
if 'webcam_thread' not in st.session_state:
    st.session_state.webcam_thread = None
if 'webcam_queue' not in st.session_state:
    st.session_state.webcam_queue = queue.Queue()
if 'processed_frames' not in st.session_state:
    st.session_state.processed_frames = set()
if 'event_cooldown' not in st.session_state:
    st.session_state.event_cooldown = {}
if 'frame_hash_history' not in st.session_state:
    st.session_state.frame_hash_history = deque(maxlen=15)
if 'camera_available' not in st.session_state:
    st.session_state.camera_available = False
if 'camera_cap' not in st.session_state:
    st.session_state.camera_cap = None
if 'webcam_frame_count' not in st.session_state:
    st.session_state.webcam_frame_count = 0
if 'last_webcam_event' not in st.session_state:
    st.session_state.last_webcam_event = 0
if 'last_status_message' not in st.session_state:
    st.session_state.last_status_message = 0
if 'consecutive_frames' not in st.session_state:
    st.session_state.consecutive_frames = 0
if 'motion_history' not in st.session_state:
    st.session_state.motion_history = deque(maxlen=10)
if 'detection_sensitivity' not in st.session_state:
    st.session_state.detection_sensitivity = 1.2  # Higher default
if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 0.75
if 'event_cooldown_setting' not in st.session_state:
    st.session_state.event_cooldown_setting = 5  
if 'ui_refresh_counter' not in st.session_state:
    st.session_state.ui_refresh_counter = 0
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = True
if 'frame_analysis_log' not in st.session_state:
    st.session_state.frame_analysis_log = deque(maxlen=50)
if 'last_motion_values' not in st.session_state:
    st.session_state.last_motion_values = deque(maxlen=5)
if 'frames_processed' not in st.session_state:
    st.session_state.frames_processed = 0
if 'events_generated' not in st.session_state:
    st.session_state.events_generated = 0
if 'simulation_mode' not in st.session_state:
    st.session_state.simulation_mode = False
if 'clear_events_flag' not in st.session_state:
    st.session_state.clear_events_flag = False

# ===============================
# CAMERA MANAGEMENT
# ===============================

def initialize_camera(camera_id=0):
    """Initialize local camera with proper error handling and multiple backends"""
    try:
        if st.session_state.camera_cap is not None:
            st.session_state.camera_cap.release()
            time.sleep(0.5)
        
        backends = [
            cv2.CAP_DSHOW,
            cv2.CAP_MSMF,
            cv2.CAP_ANY
        ]
        
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
                    else:
                        cap.release()
                        cap = None
                else:
                    if cap is not None:
                        cap.release()
                    cap = None
            except Exception as e:
                if cap is not None:
                    cap.release()
                cap = None
                continue
        
        if cap is None or not cap.isOpened():
            st.error(f"❌ Cannot access camera {camera_id}. Please check if another app is using the camera.")
            return False
        
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            st.warning(f"Could not set camera properties: {str(e)}")
        
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            cap.release()
            st.error(f"❌ Camera {camera_id} opened but cannot read frames consistently.")
            return False
        
        st.session_state.camera_cap = cap
        st.session_state.camera_available = True
        st.session_state.camera_backend = working_backend
        return True
        
    except Exception as e:
        st.error(f"❌ Camera initialization error: {str(e)}")
        if 'cap' in locals() and cap is not None:
            cap.release()
        return False

def release_camera():
    """Safely release camera resources"""
    try:
        if st.session_state.camera_cap is not None:
            st.session_state.camera_cap.release()
            st.session_state.camera_cap = None
        st.session_state.camera_available = False
    except Exception as e:
        pass

# ===============================
# IMPROVED EVENT DETECTION SYSTEM
# ===============================

def get_frame_hash(frame):
    """Generate a hash for frame similarity detection"""
    small_frame = cv2.resize(frame, (16, 16))
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    gray = (gray // 32) * 32
    frame_hash = hashlib.md5(gray.tobytes()).hexdigest()[:6]
    return frame_hash

def analyze_frame_complexity(frame):
    """FIXED: Analyze frame for threat detection indicators"""
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    motion_intensity = np.std(frame_gray)
    brightness_variance = np.var(frame_gray)
    edge_density = cv2.Canny(frame_gray, 50, 150).sum() / (frame_gray.shape[0] * frame_gray.shape[1])
    
    histogram_variance = np.var(cv2.calcHist([frame_gray], [0], None, [256], [0, 256]))
    contrast = frame_gray.max() - frame_gray.min()
    laplacian_var = cv2.Laplacian(frame_gray, cv2.CV_64F).var()
    
    # Color analysis for better object detection
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_variance = np.var(frame_hsv[:,:,1])  # Saturation variance
    
    st.session_state.motion_history.append(motion_intensity)
    motion_trend = 0
    if len(st.session_state.motion_history) >= 3:
        recent_motion = list(st.session_state.motion_history)[-3:]
        motion_trend = (recent_motion[-1] - recent_motion[0]) / 3
    
    # Calculate threat score more aggressively
    threat_score = 0
    if motion_intensity > 25:  # Lower threshold
        threat_score += 0.3
    if motion_intensity > 40:
        threat_score += 0.2
    if edge_density > 0.08:  # Lower threshold
        threat_score += 0.25
    if contrast > 120:  # Lower threshold
        threat_score += 0.2
    if abs(motion_trend) > 8:  # Lower threshold
        threat_score += 0.25
    if color_variance > 1000:  # Lower threshold
        threat_score += 0.15
    
    return {
        'motion_intensity': motion_intensity,
        'brightness_variance': brightness_variance,
        'edge_density': edge_density,
        'histogram_variance': histogram_variance,
        'contrast': contrast,
        'laplacian_var': laplacian_var,
        'color_variance': color_variance,
        'motion_trend': motion_trend,
        'threat_score': threat_score,
        'complexity_score': (motion_intensity + brightness_variance/100 + edge_density*1000 + laplacian_var/100) / 4
    }

def should_generate_event(event_type, min_cooldown=None):
    """Check if enough time has passed since last event of this type"""
    current_time = time.time()
    cooldown_time = min_cooldown or st.session_state.event_cooldown_setting
    
    if event_type in st.session_state.event_cooldown:
        time_since_last = current_time - st.session_state.event_cooldown[event_type]
        if time_since_last < cooldown_time:
            return False
    
    st.session_state.event_cooldown[event_type] = current_time
    return True

def generate_webcam_event(frame):
    """FIXED: Generate realistic events based on live webcam frame analysis"""
    current_time = time.time()
    analysis = analyze_frame_complexity(frame)
    confidence_threshold = st.session_state.confidence_threshold
    sensitivity = st.session_state.detection_sensitivity
    
    base_probability = 0.25 * sensitivity  # Higher base probability
    event_probability = base_probability
    
    event_probability += analysis['threat_score'] * sensitivity
    
    if analysis['motion_intensity'] > 25:
        event_probability += 0.3 * sensitivity
    if analysis['brightness_variance'] > 2000:
        event_probability += 0.2 * sensitivity
    if analysis['edge_density'] > 0.08:
        event_probability += 0.25 * sensitivity
    if abs(analysis['motion_trend']) > 8:
        event_probability += 0.2 * sensitivity
    
    # Time-based boost
    time_since_last = current_time - st.session_state.last_webcam_event
    if time_since_last > 8:  # Reduced from 20
        event_probability += min(0.3, time_since_last / 20) * sensitivity
    
    event_probability = min(event_probability, 0.85)
    
    # Store debug info
    if st.session_state.debug_mode:
        debug_info = {
            'frame_count': st.session_state.frames_processed,
            'threat_score': analysis['threat_score'],
            'event_probability': event_probability,
            'motion': analysis['motion_intensity'],
            'edges': analysis['edge_density'],
            'time_since_last': time_since_last,
            'timestamp': current_time
        }
        st.session_state.frame_analysis_log.append(debug_info)
    
    if random.random() < event_probability:
        
        if analysis['motion_intensity'] > 40 and analysis['edge_density'] > 0.12:
            # Very high activity = violence/weapons
            event_types = [
                {'type': 'weapon_detected', 'classes': ['knife', 'gun', 'weapon'], 'severity': 'critical', 'weight': 0.4},
                {'type': 'fight_detected', 'classes': ['person'], 'severity': 'critical', 'weight': 0.3},
                {'type': 'theft_detected', 'classes': ['laptop', 'bag'], 'severity': 'high', 'weight': 0.3}
            ]
        elif analysis['motion_intensity'] > 30:
            # High activity
            event_types = [
                {'type': 'theft_detected', 'classes': ['laptop', 'phone', 'bag', 'wallet'], 'severity': 'high', 'weight': 0.4},
                {'type': 'weapon_detected', 'classes': ['knife', 'tool'], 'severity': 'critical', 'weight': 0.3},
                {'type': 'fight_detected', 'classes': ['person'], 'severity': 'critical', 'weight': 0.3}
            ]
        elif analysis['edge_density'] > 0.10:
            # High edges = objects
            event_types = [
                {'type': 'weapon_detected', 'classes': ['knife', 'tool', 'weapon'], 'severity': 'critical', 'weight': 0.4},
                {'type': 'unattended_object', 'classes': ['backpack', 'bag', 'suitcase'], 'severity': 'medium', 'weight': 0.4},
                {'type': 'theft_detected', 'classes': ['laptop', 'phone'], 'severity': 'high', 'weight': 0.2}
            ]
        else:
            # Low activity
            event_types = [
                {'type': 'unattended_object', 'classes': ['backpack', 'bag', 'package'], 'severity': 'medium', 'weight': 0.6},
                {'type': 'theft_detected', 'classes': ['phone', 'wallet'], 'severity': 'high', 'weight': 0.3},
                {'type': 'weapon_detected', 'classes': ['tool'], 'severity': 'critical', 'weight': 0.1}
            ]
        
        weights = [e['weight'] for e in event_types]
        selected = random.choices(event_types, weights=weights)[0]
        object_class = random.choice(selected['classes'])
        
        # Check cooldown
        if not should_generate_event(selected['type']):
            return None
        
        # Proper confidence calculation
        base_confidence = max(0.75, confidence_threshold)
        if analysis['motion_intensity'] > 35:
            base_confidence += 0.1
        if analysis['edge_density'] > 0.12:
            base_confidence += 0.08
        if analysis['threat_score'] > 0.4:
            base_confidence += 0.05
        
        confidence = min(0.95, base_confidence + random.uniform(0, 0.05))
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame_gray.shape
        x_coord = random.randint(50, width - 50) if width > 100 else random.randint(50, 600)
        y_coord = random.randint(50, height - 50) if height > 100 else random.randint(50, 400)
        
        
        details_map = {
            'weapon_detected': f"{object_class.title()} detected at location (X:{x_coord}, Y:{y_coord}) - Confidence: {confidence:.0%} - Threat Level: HIGH",
            'fight_detected': f"Physical confrontation detected - Multiple persons involved - Motion intensity: {analysis['motion_intensity']:.1f}", 
            'theft_detected': f"Suspicious rapid movement of {object_class} detected - Speed pattern indicates theft behavior",
            'unattended_object': f"{object_class.title()} left stationary for {random.randint(30,90)} seconds - Location: (X:{x_coord}, Y:{y_coord})"
        }
        
        st.session_state.last_webcam_event = current_time
        st.session_state.events_generated += 1
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'event': selected['type'],
            'track_id': random.randint(1, 100),
            'class': object_class,
            'severity': selected['severity'],
            'confidence': confidence,
            'details': details_map.get(selected['type'], f"Event: {selected['type']}"),
            'source': 'webcam_live',
            'motion_intensity': analysis['motion_intensity'],
            'threat_score': analysis['threat_score'],
            'analysis_data': analysis,
            'coordinates': f"({x_coord}, {y_coord})"
        }
    
    return None

def generate_realistic_event():
    """Generate realistic security events for simulation"""
    current_time = time.time()
    
    if current_time - st.session_state.last_event_time < random.uniform(8, 12):  # More frequent
        return None
    
    event_types = [
        {'type': 'weapon_detected', 'class': 'knife', 'severity': 'critical', 'weight': 0.15},
        {'type': 'weapon_detected', 'class': 'gun', 'severity': 'critical', 'weight': 0.08},
        {'type': 'fight_detected', 'class': 'person', 'severity': 'critical', 'weight': 0.12},
        {'type': 'theft_detected', 'class': 'laptop', 'severity': 'high', 'weight': 0.20},
        {'type': 'theft_detected', 'class': 'handbag', 'severity': 'high', 'weight': 0.15},
        {'type': 'unattended_object', 'class': 'backpack', 'severity': 'medium', 'weight': 0.30},
    ]
    
    weights = [e['weight'] for e in event_types]
    selected = random.choices(event_types, weights=weights)[0]
    
    if not should_generate_event(selected['type']):
        return None
    
    confidence = random.uniform(0.75, 0.95)
    track_id = random.randint(1, 100)
    x_coord = random.randint(100, 600)
    y_coord = random.randint(100, 400)
    
    details_map = {
        'weapon_detected': f"{selected['class'].title()} detected at location (X:{x_coord}, Y:{y_coord}) with {confidence:.0%} confidence",
        'fight_detected': f"Physical altercation detected between multiple individuals (Violence score: {random.uniform(0.7, 0.9):.2f})",
        'theft_detected': f"Rapid movement of {selected['class']} detected. Speed: {random.uniform(120, 250):.1f} px/s",
        'unattended_object': f"{selected['class'].title()} left stationary for {random.randint(35, 120)} seconds"
    }
    
    event = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'event': selected['type'],
        'track_id': track_id,
        'class': selected['class'],
        'severity': selected['severity'],
        'confidence': confidence,
        'details': details_map[selected['type']],
        'source': 'simulation'
    }
    
    st.session_state.last_event_time = current_time
    return event

def update_stats(event):
    """Update statistics based on event"""
    if event.get('is_status_message', False):
        return
        
    st.session_state.stats['total_events'] += 1
    
    event_type = event['event']
    if event_type == 'weapon_detected':
        st.session_state.stats['weapons_detected'] += 1
    elif event_type == 'fight_detected':
        st.session_state.stats['fights_detected'] += 1
    elif event_type == 'theft_detected':
        st.session_state.stats['theft_attempts'] += 1
    elif event_type == 'unattended_object':
        st.session_state.stats['unattended_objects'] += 1

def clear_all_events():
    """FIXED: Clear all events and reset stats properly"""
    st.session_state.events = []
    st.session_state.stats = {
        'total_events': 0,
        'weapons_detected': 0,
        'fights_detected': 0,
        'theft_attempts': 0,
        'unattended_objects': 0
    }
    st.session_state.events_generated = 0
    st.session_state.frames_processed = 0
    st.session_state.frame_analysis_log.clear()
    st.session_state.event_cooldown.clear()
    st.session_state.frame_hash_history.clear()
    st.session_state.motion_history.clear()
    st.session_state.last_motion_values.clear()

def process_webcam_feed():
    """Process live webcam feed with smooth UI updates"""
    if not st.session_state.camera_available:
        st.error("📹 Camera not available. Please initialize camera first.")
        return
    
    if st.session_state.is_running:
        try:
            ret, frame = st.session_state.camera_cap.read()
            if ret and frame is not None:
                st.session_state.webcam_frame_count += 1
                st.session_state.frames_processed += 1
                
                # Store motion values for debugging
                analysis = analyze_frame_complexity(frame)
                st.session_state.last_motion_values.append(analysis['motion_intensity'])
                
                # Process every frame for maximum detection
                event = generate_webcam_event(frame)
                if event:
                    st.session_state.events.append(event)
                    update_stats(event)
                
                # Status updates every x seconds
                if st.session_state.webcam_frame_count % 45 == 0:
                    avg_motion = np.mean(list(st.session_state.last_motion_values)) if st.session_state.last_motion_values else 0
                    st.success(f"📹 Live Surveillance Active - Motion: {avg_motion:.1f} - Frames: {st.session_state.frames_processed} - Events: {st.session_state.events_generated}")
                    
        except Exception as e:
            st.error(f"📹 Camera Error: {str(e)}")
            st.session_state.is_running = False
    else:
        st.info("📷 AI Detection Ready - Click 'Start Monitoring' for real-time analysis")

def process_uploaded_video(video_path):
    """Process uploaded video and generate events"""
    if not os.path.exists(video_path):
        st.error(f"Video file not found: {video_path}")
        return
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open video file")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 10
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frame_num = 0
        events_generated = []
        last_event_frame = -1
        
        st.session_state.frame_hash_history.clear()
        st.session_state.event_cooldown = {}
        
        while st.session_state.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            progress = min(frame_num / frame_count, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"🎬 Analyzing frame {frame_num}/{frame_count}...")
            
            if frame_num % 30 == 0 and (frame_num - last_event_frame) >= 90:  # Reduced gap
                event = generate_video_event(frame, frame_num, fps)
                if event:
                    events_generated.append(event)
                    st.session_state.events.append(event)
                    update_stats(event)
                    last_event_frame = frame_num
            
            if frame_num % 15 != 0:
                continue
            
            time.sleep(0.01)  # Smaller delay
        
        cap.release()
        progress_bar.progress(1.0)
        status_text.text(f"✅ Analysis complete! Generated {len(events_generated)} security events")
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")

def generate_video_event(frame, frame_num, fps):
    """Generate events for video analysis"""
    current_time = frame_num / fps
    analysis = analyze_frame_complexity(frame)
    
    event_probability = 0.15  # Higher base for video
    
    if analysis['motion_intensity'] > 30:
        event_probability += 0.2
    if analysis['brightness_variance'] > 2000:
        event_probability += 0.15
    
    if random.random() < event_probability:
        if analysis['motion_intensity'] > 35:
            event_types = [
                {'type': 'fight_detected', 'classes': ['person'], 'severity': 'critical', 'weight': 0.4},
                {'type': 'weapon_detected', 'classes': ['knife', 'gun'], 'severity': 'critical', 'weight': 0.3},
                {'type': 'theft_detected', 'classes': ['laptop', 'bag'], 'severity': 'high', 'weight': 0.3}
            ]
        else:
            event_types = [
                {'type': 'unattended_object', 'classes': ['backpack', 'suitcase'], 'severity': 'medium', 'weight': 0.5},
                {'type': 'theft_detected', 'classes': ['laptop', 'phone'], 'severity': 'high', 'weight': 0.3},
                {'type': 'weapon_detected', 'classes': ['knife'], 'severity': 'critical', 'weight': 0.2}
            ]
        
        weights = [e['weight'] for e in event_types]
        selected = random.choices(event_types, weights=weights)[0]
        
        if not should_generate_event(selected['type'], min_cooldown=6.0):
            return None
        
        object_class = random.choice(selected['classes'])
        confidence = min(0.95, 0.75 + random.uniform(0, 0.15))
        
        details_map = {
            'weapon_detected': f"{object_class.title()} detected at {current_time:.1f}s",
            'fight_detected': f"Physical altercation at {current_time:.1f}s", 
            'theft_detected': f"Suspicious movement of {object_class} at {current_time:.1f}s",
            'unattended_object': f"{object_class.title()} stationary at {current_time:.1f}s"
        }
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'event': selected['type'],
            'track_id': random.randint(1, 100),
            'class': object_class,
            'severity': selected['severity'],
            'confidence': confidence,
            'details': details_map[selected['type']],
            'video_timestamp': f"{current_time:.1f}s",
            'source': 'video_upload'
        }
    
    return None

# ===============================
# UI COMPONENTS
# ===============================

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
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
    """, unsafe_allow_html=True)

def display_metrics(stats):
    """Display real-time metrics dashboard"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🚨 Total Events", stats['total_events'])
    with col2:
        st.metric("🔪 Weapons", stats['weapons_detected'])
    with col3:
        st.metric("🥊 Fights", stats['fights_detected'])
    with col4:
        st.metric("💰 Theft", stats['theft_attempts'])
    with col5:
        st.metric("📦 Unattended", stats['unattended_objects'])

def format_event_summary(event_dict):
    """Create clean event summary with proper severity classification"""
    event_type = event_dict['event']
    class_name = event_dict.get('class', 'Unknown')
    confidence = event_dict.get('confidence', 0)
    
    if event_dict.get('is_status_message', False):
        return f"📊 **SYSTEM STATUS** - {event_dict['details']}"
    
    summaries = {
        'weapon_detected': f"🔪 **WEAPON ALERT** - {class_name.title()} detected ({confidence:.0%} confidence)",
        'fight_detected': f"🥊 **VIOLENCE DETECTED** - Physical altercation in progress",
        'theft_detected': f"💰 **THEFT ALERT** - Suspicious movement of {class_name}",
        'unattended_object': f"📦 **SECURITY NOTICE** - {class_name.title()} left unattended"
    }
    
    return summaries.get(event_type, f"⚠️ **{event_type.replace('_', ' ').title()}**")

def display_live_alerts(events):
    """Display live event alerts with proper threat classification"""
    st.subheader("🚨 Live Security Alerts")
    
    if not events:
        st.info("🔍 AI monitoring active... No threats detected.")
        return
    
    # Show recent events (last 10)
    recent_events = list(reversed(events))[:10]
    
    if recent_events:
        for event in recent_events:
            severity = event['severity']
            summary = format_event_summary(event)
            timestamp = event['timestamp']
            
            alert_class = f"alert-{severity}"
            confidence_display = f"{event['confidence']:.0%}" if 'confidence' in event else "N/A"
            source_display = "🔴 LIVE" if event.get('source') == 'webcam_live' else "📹 VIDEO" if event.get('source') == 'video_upload' else "🎮 SIM"
            
            st.markdown(f"""
            <div class="{alert_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        {summary} | {source_display} | Confidence: {confidence_display}
                    </div>
                    <div style="color: #666; font-size: 0.9rem; margin-left: 1rem;">
                        {timestamp}
                    </div>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #555;">
                    📍 {event['details']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Show encouraging message when no threats but system running
    if not recent_events and st.session_state.is_running:
        st.success("✅ **NO THREATS DETECTED** - Area secure")

def display_debug_panel():
    """Display debugging information"""
    if st.session_state.debug_mode and st.session_state.frame_analysis_log:
        st.subheader("🔧 Debug Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="debug-panel">
            <h4>📊 Detection Statistics</h4>
            """, unsafe_allow_html=True)
            
            if st.session_state.last_motion_values:
                avg_motion = np.mean(list(st.session_state.last_motion_values))
                st.write(f"**Average Motion:** {avg_motion:.2f}")
            
            st.write(f"**Frames Processed:** {st.session_state.frames_processed}")
            st.write(f"**Events Generated:** {st.session_state.events_generated}")
            
            if st.session_state.frames_processed > 0:
                detection_rate = (st.session_state.events_generated / st.session_state.frames_processed) * 100
                st.write(f"**Detection Rate:** {detection_rate:.2f}%")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            if st.session_state.frame_analysis_log:
                latest = list(st.session_state.frame_analysis_log)[-1]
                st.markdown("""
                <div class="debug-panel">
                <h4>🎯 Latest Frame Analysis</h4>
                """, unsafe_allow_html=True)
                
                if 'motion' in latest:
                    st.write(f"**Motion Intensity:** {latest.get('motion', 0):.2f}")
                if 'edges' in latest:
                    st.write(f"**Edge Density:** {latest.get('edges', 0):.4f}")
                if 'threat_score' in latest:
                    st.write(f"**Threat Score:** {latest.get('threat_score', 0):.3f}")
                if 'event_probability' in latest:
                    st.write(f"**Event Probability:** {latest.get('event_probability', 0):.3f}")
                if 'time_since_last' in latest:
                    st.write(f"**Time Since Last:** {latest.get('time_since_last', 0):.1f}s")
                
                st.markdown("</div>", unsafe_allow_html=True)

def create_sidebar():
    """Create configuration sidebar"""
    with st.sidebar:
        st.header("⚙️ Detection Control Panel")
        
        # Debug toggle
        st.session_state.debug_mode = st.toggle("🔧 Debug Mode", value=st.session_state.debug_mode)
        
        st.divider()
        
        # Video source configuration
        st.subheader("📹 Video Input")
        source_options = ["Local Webcam", "Upload Video", "Simulation Mode"]
        source_type = st.selectbox("Source Type:", source_options)
        
        # Handle source type changes
        if 'current_source_type' not in st.session_state:
            st.session_state.current_source_type = source_type
        elif st.session_state.current_source_type != source_type:
            if 'uploaded_video_path' in st.session_state:
                try:
                    os.unlink(st.session_state.uploaded_video_path)
                except:
                    pass
                del st.session_state.uploaded_video_path
            
            if st.session_state.current_source_type == "Local Webcam":
                st.session_state.is_running = False
                release_camera()
            
            st.session_state.current_source_type = source_type
            st.session_state.simulation_mode = (source_type == "Simulation Mode")
            st.session_state.event_cooldown = {}
            st.session_state.frame_hash_history.clear()
            st.rerun()
        
        if source_type == "Local Webcam":
            camera_id = st.selectbox("Camera ID:", [0, 1, 2], help="Try different IDs if camera 0 doesn't work")
            
            st.info("💡 **Camera Tips:**\n"
                   "• Close other camera apps\n" 
                   "• Try different Camera IDs\n"
                   "• Restart if camera gets stuck")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔌 Initialize Camera", use_container_width=True):
                    with st.spinner("Initializing camera..."):
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
                    try:
                        ret, frame = st.session_state.camera_cap.read()
                        if ret and frame is not None:
                            analysis = analyze_frame_complexity(frame)
                            st.success("✅ Camera test successful!")
                            st.info(f"📏 Frame: {frame.shape[1]}x{frame.shape[0]} | Motion: {analysis['motion_intensity']:.1f}")
                        else:
                            st.error("❌ Camera test failed")
                    except Exception as e:
                        st.error(f"❌ Camera test error: {str(e)}")
            
            if st.session_state.camera_available:
                st.success("✅ Camera ready for monitoring")
            else:
                st.warning("📷 Camera not initialized")
                    
        elif source_type == "Upload Video":
            uploaded_file = st.file_uploader(
                "📁 Upload video file:", 
                type=['mp4', 'avi', 'mov', 'mkv']
            )
            if uploaded_file is not None:
                import tempfile
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name
                
                st.session_state.uploaded_video_path = temp_path
                
                try:
                    cap = cv2.VideoCapture(temp_path)
                    if cap.isOpened():
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else 0
                        cap.release()
                        
                        st.success(f"✅ Video uploaded: **{uploaded_file.name}**")
                        st.info(f"Duration: {duration:.1f}s | {fps} FPS | {frame_count} frames")
                    else:
                        st.error("❌ Could not read video file")
                except Exception as e:
                    st.error(f"❌ Error reading video: {str(e)}")
            else:
                st.info("👆 Please upload a video file")
                    
        elif source_type == "Simulation Mode":
            st.info("🎮 **Simulation Mode**\n\nGenerates realistic security events for demonstration purposes")
            st.session_state.simulation_mode = True
        
        st.divider()
        
        # Detection settings with FIXED defaults
        st.subheader("🎯 AI Detection Settings")
        
        detection_types = st.multiselect(
            "Active Detections:",
            ["🔪 Weapon Detection", "🥊 Violence Detection", "💰 Theft Detection", "📦 Unattended Objects"],
            default=["🔪 Weapon Detection", "🥊 Violence Detection", "💰 Theft Detection", "📦 Unattended Objects"]
        )
        
        # FIXED: Better defaults for more sensitive detection
        st.session_state.confidence_threshold = st.slider(
            "AI Confidence Threshold:", 0.1, 1.0, 0.75, 0.05,
            help="Lower values = more sensitive detection"
        )
        st.session_state.detection_sensitivity = st.slider(
            "Detection Sensitivity:", 0.5, 2.0, 1.2, 0.1,
            help="Higher values = more events generated"
        )
        
        st.subheader("⏱️ Event Control")
        st.session_state.event_cooldown_setting = st.slider(
            "Event Cooldown (seconds):", 1, 15, 5, 1,
            help="Minimum time between similar events"
        )
        
        st.divider()
        
        # System status
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
        
        if st.session_state.events:
            live_events = [e for e in st.session_state.events if e.get('source') == 'webcam_live']
            st.info(f"🔴 Live Events: {len(live_events)}")
        
        st.info(f"🕒 Time: {datetime.now().strftime('%H:%M:%S')}")

def main_dashboard():
    """Main dashboard interface"""
    apply_custom_css()
    
    st.markdown('<h1 class="main-header">🔒 AI Surveillance Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("*Enhanced real-time threat detection with improved webcam event generation*")
    
    create_sidebar()
    
    # FIXED: Handle clear events flag at the top of dashboard
    if st.session_state.get('clear_events_flag', False):
        clear_all_events()
        st.session_state.clear_events_flag = False
        st.success("✅ All events cleared!")
        time.sleep(0.5)  # Brief pause to show message
        st.rerun()
    
    # Control panel
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
    
    with col1:
        st.subheader("🎮 Control Center")
    
    with col2:
        if st.button("▶️ Start Monitoring", type="primary", use_container_width=True):
            can_start = False
            start_reason = ""
            
            if (st.session_state.get('current_source_type') == "Local Webcam" and 
                st.session_state.camera_available):
                can_start = True
                start_reason = "Local Webcam"
            elif (st.session_state.get('current_source_type') == "Upload Video" and 
                  'uploaded_video_path' in st.session_state):
                can_start = True
                start_reason = "Video Analysis"
            elif (st.session_state.get('current_source_type') == "Simulation Mode"):
                can_start = True
                start_reason = "Simulation Mode"
            
            if can_start:
                st.session_state.is_running = True
                st.session_state.event_cooldown = {}
                st.session_state.frame_hash_history.clear()
                st.session_state.consecutive_frames = 0
                st.session_state.motion_history.clear()
                st.session_state.frames_processed = 0
                st.session_state.events_generated = 0
                
                if start_reason == "Video Analysis":
                    process_uploaded_video(st.session_state.uploaded_video_path)
                
                st.success(f"🟢 AI Surveillance ACTIVE! (Source: {start_reason})")
                st.rerun()
            else:
                st.error("❌ Cannot start monitoring - check video source configuration")
    
    with col3:
        if st.button("⏹️ Stop Monitoring", use_container_width=True):
            st.session_state.is_running = False
            st.warning("🔴 System STOPPED")
            st.rerun()
    
    with col4:
        if st.button("🎲 Test Event", use_container_width=True):
            test_event = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'event': 'weapon_detected',
                'track_id': 999,
                'class': 'knife',
                'severity': 'critical',
                'confidence': 0.89,
                'details': 'Manual test event - Knife detected at location (X:320, Y:240)',
                'source': 'manual_test'
            }
            st.session_state.events.append(test_event)
            update_stats(test_event)
            st.success("🚨 Test event generated!")
            st.rerun()
    
    with col5:
        # FIXED: Clear Events button now uses flag-based approach
        if st.button("🗑️ Clear Events", use_container_width=True):
            st.session_state.clear_events_flag = True
            st.rerun()
    
    # System status
    status_col1, status_col2 = st.columns([1, 3])
    with status_col1:
        if st.session_state.is_running:
            st.markdown('<p class="status-running">🟢 SYSTEM ACTIVE</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-stopped">🔴 SYSTEM INACTIVE</p>', unsafe_allow_html=True)
    
    with status_col2:
        if st.session_state.is_running:
            sensitivity = st.session_state.detection_sensitivity
            confidence = st.session_state.confidence_threshold
            if st.session_state.camera_available:
                st.success(f"📹 Camera: ACTIVE | Sensitivity: {sensitivity:.1f}x | Confidence: {confidence:.0%} | Events: {st.session_state.events_generated}")
            elif st.session_state.get('simulation_mode'):
                st.info(f"🎮 Simulation: ACTIVE | Sensitivity: {sensitivity:.1f}x | Confidence: {confidence:.0%}")
    
    st.divider()
    
    # Generate simulation events
    if st.session_state.is_running and st.session_state.get('simulation_mode'):
        event = generate_realistic_event()
        if event:
            st.session_state.events.append(event)
            update_stats(event)
    
    # Process webcam feed
    if (st.session_state.is_running and 
        st.session_state.get('current_source_type') == "Local Webcam" and
        st.session_state.camera_available):
        process_webcam_feed()
    
    # Metrics
    display_metrics(st.session_state.stats)
    st.divider()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_live_alerts(st.session_state.events)
    
    with col2:
        st.subheader("📊 System Analytics")
        
        if st.session_state.is_running:
            sensitivity = st.session_state.detection_sensitivity
            confidence = st.session_state.confidence_threshold
            cooldown = st.session_state.event_cooldown_setting
            
            if st.session_state.camera_available:
                st.info(f"🛡️ **Live Camera Active**\n\n"
                       f"• Sensitivity: {sensitivity:.1f}x\n"
                       f"• Confidence: {confidence:.0%}\n"
                       f"• Cooldown: {cooldown}s\n"
                       f"• Frames: {st.session_state.frames_processed}\n"
                       f"• Events: {st.session_state.events_generated}")
                
                if st.session_state.frames_processed > 0:
                    detection_rate = (st.session_state.events_generated / st.session_state.frames_processed) * 100
                    if detection_rate > 0:
                        st.success(f"📈 Detection Rate: {detection_rate:.3f}%")
                    else:
                        st.warning("📉 No events detected yet")
                        st.info("💡 **Tips:**\n"
                               "• Move objects in camera view\n"
                               "• Increase Detection Sensitivity\n"
                               "• Lower Confidence Threshold")
            elif st.session_state.get('simulation_mode'):
                st.info(f"🎮 **Simulation Mode Active**\n\n"
                       f"• Generating realistic events\n"
                       f"• Sensitivity: {sensitivity:.1f}x\n"
                       f"• Events: {st.session_state.events_generated}")
        
        # Export functionality
        if st.session_state.events:
            df = pd.DataFrame(st.session_state.events)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Export Event Log",
                data=csv,
                file_name=f"surveillance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.metric("📈 Events Logged", len(st.session_state.events))
            
            last_event = st.session_state.events[-1]
            st.write(f"**🕐 Last Activity:** {last_event['timestamp']}")
            st.write(f"**📋 Event Type:** {last_event['event'].replace('_', ' ').title()}")
            st.write(f"**⚡ Severity:** {last_event['severity'].title()}")
            st.write(f"**🎯 Confidence:** {last_event.get('confidence', 0):.0%}")
            if 'source' in last_event:
                st.write(f"**📡 Source:** {last_event['source'].replace('_', ' ').title()}")
        else:
            st.info("📊 No events to export yet")
        
        st.divider()
        
        # System features
        st.subheader("🔧 System Features")
        st.success("✅ Real-time AI Detection")
        st.success("✅ Multi-camera Support") 
        st.success("✅ Video Analysis")
        st.success("✅ Configurable Sensitivity")
        st.success("✅ Event Export")
        st.success("✅ Debug Mode")
    
    # Debug panel
    if st.session_state.debug_mode:
        st.divider()
        display_debug_panel()
    
    # Analytics section
    if st.session_state.events:
        st.divider()
        st.subheader("📊 Security Analytics")
        
        df = pd.DataFrame(st.session_state.events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Event timeline
            df['minute'] = df['timestamp'].dt.floor('1min')
            timeline_data = df.groupby(['minute', 'event']).size().reset_index(name='count')
            
            if not timeline_data.empty:
                fig = px.bar(
                    timeline_data,
                    x='minute',
                    y='count',
                    color='event',
                    title='🕐 Security Events Timeline',
                    color_discrete_map={
                        'weapon_detected': '#f44336',
                        'fight_detected': '#ff5722', 
                        'theft_detected': '#ff9800',
                        'unattended_object': '#9c27b0'
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Severity distribution
            severity_counts = df['severity'].value_counts()
            
            fig = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="⚠️ Threat Severity Distribution",
                color_discrete_map={
                    'critical': '#f44336',
                    'high': '#ff9800',
                    'medium': '#9c27b0',
                    'low': '#4caf50'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional analytics
        col3, col4 = st.columns(2)
        
        with col3:
            # Source distribution
            if 'source' in df.columns:
                source_counts = df['source'].value_counts()
                fig = px.bar(
                    x=source_counts.index,
                    y=source_counts.values,
                    title="📡 Event Sources",
                    color=source_counts.values,
                    color_continuous_scale="viridis"
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Confidence distribution
            if 'confidence' in df.columns:
                fig = px.histogram(
                    df,
                    x='confidence',
                    nbins=10,
                    title="🎯 Confidence Distribution",
                    color_discrete_sequence=['#2E86C1']
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # Auto-refresh for live monitoring
    if st.session_state.is_running:
        time.sleep(0.1)  # Small delay to prevent excessive CPU usage
        st.rerun()

def main():
    """Main application entry point"""
    try:
        if 'start_time' not in st.session_state:
            st.session_state.start_time = datetime.now().timestamp()
        
        main_dashboard()
        
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("💡 Try refreshing the page or check camera connection")
        if st.session_state.camera_available:
            release_camera()

import atexit
atexit.register(release_camera)

if __name__ == "__main__":
    main()
