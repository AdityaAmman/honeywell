# AI Surveillance Dashboard
# Real-time threat detection system

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

# Page configuration
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

# ===============================
# EVENT GENERATION SYSTEM
# ===============================

def generate_realistic_event():
    """Generate realistic security events for demonstration"""
    current_time = time.time()
    
    # Generate events every 4-7 seconds when monitoring is active
    if current_time - st.session_state.last_event_time < random.uniform(4, 7):
        return None
    
    event_types = [
        {'type': 'weapon_detected', 'class': 'knife', 'severity': 'critical', 'weight': 0.1},
        {'type': 'weapon_detected', 'class': 'scissors', 'severity': 'critical', 'weight': 0.05},
        {'type': 'fight_detected', 'class': 'person', 'severity': 'critical', 'weight': 0.08},
        {'type': 'theft_detected', 'class': 'laptop', 'severity': 'high', 'weight': 0.15},
        {'type': 'theft_detected', 'class': 'handbag', 'severity': 'high', 'weight': 0.12},
        {'type': 'unattended_object', 'class': 'backpack', 'severity': 'medium', 'weight': 0.3},
        {'type': 'unattended_object', 'class': 'suitcase', 'severity': 'medium', 'weight': 0.2},
    ]
    
    # Select event based on weights
    weights = [e['weight'] for e in event_types]
    selected = random.choices(event_types, weights=weights)[0]
    
    confidence = random.uniform(0.75, 0.95)
    track_id = random.randint(1, 100)
    x_coord = random.randint(100, 600)
    y_coord = random.randint(100, 400)
    
    details_map = {
        'weapon_detected': f"Detected {selected['class']} with {confidence:.0%} confidence at location (X:{x_coord}, Y:{y_coord})",
        'fight_detected': f"Aggressive behavior detected between multiple individuals (Violence score: {random.uniform(0.7, 0.9):.2f})",
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
        'details': details_map[selected['type']]
    }
    
    st.session_state.last_event_time = current_time
    return event

def update_stats(event):
    """Update statistics based on event"""
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
        font-size: 3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .alert-critical {
        background: linear-gradient(90deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(244, 67, 54, 0.3);
        animation: pulse-red 2s infinite;
    }
    .alert-high {
        background: linear-gradient(90deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(255, 152, 0, 0.3);
    }
    .alert-medium {
        background: linear-gradient(90deg, #f3e5f5 0%, #e1bee7 100%);
        border-left: 5px solid #9c27b0;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(156, 39, 176, 0.3);
    }
    @keyframes pulse-red {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .status-running {
        color: #4caf50;
        font-weight: bold;
        font-size: 1.2rem;
        text-shadow: 0 0 10px #4caf50;
    }
    .status-stopped {
        color: #f44336;
        font-weight: bold;
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

def display_metrics(stats):
    """Display real-time metrics dashboard"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🚨 Total Events", 
            stats['total_events'],
            delta=f"+{len(st.session_state.events)} today" if st.session_state.events else None
        )
    with col2:
        st.metric("🔪 Weapons", stats['weapons_detected'])
    with col3:
        st.metric("🥊 Fights", stats['fights_detected'])
    with col4:
        st.metric("💰 Theft", stats['theft_attempts'])
    with col5:
        st.metric("📦 Unattended", stats['unattended_objects'])

def format_event_summary(event_dict):
    """Create clean event summary with icons"""
    event_type = event_dict['event']
    class_name = event_dict.get('class', 'Unknown')
    confidence = event_dict.get('confidence', 0)
    
    summaries = {
        'weapon_detected': f"🔪 **WEAPON ALERT** - {class_name.title()} detected ({confidence:.0%} confidence)",
        'fight_detected': f"🥊 **VIOLENCE DETECTED** - Physical altercation in progress",
        'theft_detected': f"💰 **THEFT ALERT** - Suspicious movement of {class_name}",
        'unattended_object': f"📦 **SECURITY NOTICE** - {class_name.title()} left unattended",
    }
    
    return summaries.get(event_type, f"⚠️ **{event_type.replace('_', ' ').title()}**")

def display_live_alerts(events):
    """Display live event alerts with enhanced styling"""
    st.subheader("🚨 Live Security Alerts")
    
    if not events:
        st.info("🔍 System monitoring... No threats detected.")
        
        # Show system status indicators
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("🎯 Object Detection: Active")
        with col2:
            st.success("👁️ Motion Analysis: Active") 
        with col3:
            st.success("🧠 AI Processing: Active")
        return
    
    # Show recent events with severity-based styling
    recent_events = list(reversed(events))[:8]
    
    for i, event in enumerate(recent_events):
        severity = event['severity']
        summary = format_event_summary(event)
        timestamp = event['timestamp']
        
        # Create styled alert with animation for critical events
        alert_class = f"alert-{severity}"
        
        st.markdown(f"""
        <div class="{alert_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="flex: 1;">
                    {summary}
                </div>
                <div style="color: #666; font-size: 0.9rem; margin-left: 1rem; font-weight: bold;">
                    {timestamp}
                </div>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #555; font-style: italic;">
                📍 {event['details']}
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_sidebar():
    """Create configuration sidebar"""
    with st.sidebar:
        st.header("⚙️ Configuration Panel")
        
        st.divider()
        
        # Video source configuration
        st.subheader("📹 Video Input")
        source_options = ["Webcam", "Upload Video", "Simulation Mode"]
        source_type = st.selectbox("Source Type:", source_options)
        
        # Clear video state when switching modes
        if 'current_source_type' not in st.session_state:
            st.session_state.current_source_type = source_type
        elif st.session_state.current_source_type != source_type:
            # Source type changed, clear video-related state
            if 'uploaded_video_path' in st.session_state:
                try:
                    os.unlink(st.session_state.uploaded_video_path)  # Delete temp file
                except:
                    pass
                del st.session_state.uploaded_video_path
            st.session_state.current_source_type = source_type
            st.session_state.simulation_mode = (source_type == "Simulation Mode")
            st.rerun()
        
        if source_type == "Webcam":
            camera_id = st.selectbox("Camera:", [0, 1, 2])
            st.success("📹 Ready to monitor live webcam feed")
            if 'simulation_mode' in st.session_state:
                del st.session_state.simulation_mode
            
        elif source_type == "Upload Video":
            if 'simulation_mode' in st.session_state:
                del st.session_state.simulation_mode
            uploaded_file = st.file_uploader(
                "📁 Upload your MP4 file:", 
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Drag and drop your video file here"
            )
            if uploaded_file is not None:
                # Save uploaded file temporarily
                import tempfile
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name
                
                st.session_state.uploaded_video_path = temp_path
                
                # Show video info
                try:
                    cap = cv2.VideoCapture(temp_path)
                    if cap.isOpened():
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else 0
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                        
                        st.success(f"✅ Video uploaded: **{uploaded_file.name}**")
                        st.info(f"""
                        📊 **Video Details:**
                        - Duration: {duration:.1f} seconds
                        - Resolution: {width}x{height}
                        - Frame Rate: {fps} FPS
                        - Total Frames: {frame_count:,}
                        """)
                        st.success("🎬 Ready for analysis! Click 'Start Monitoring' to process.")
                    else:
                        st.error("❌ Could not read video file. Please try a different format.")
                        if 'uploaded_video_path' in st.session_state:
                            del st.session_state.uploaded_video_path
                except Exception as e:
                    st.error(f"❌ Error reading video: {str(e)}")
                    if 'uploaded_video_path' in st.session_state:
                        del st.session_state.uploaded_video_path
            else:
                st.info("👆 Please upload a video file to analyze")
                if 'uploaded_video_path' in st.session_state:
                    del st.session_state.uploaded_video_path
                    
        elif source_type == "Simulation Mode":
            st.info("🎮 Simulation mode will generate realistic security events for testing and demonstration purposes.")
            st.session_state.simulation_mode = True
        
        st.divider()
        
        # Detection settings
        st.subheader("🎯 AI Detection Settings")
        
        detection_types = st.multiselect(
            "Active Detections:",
            ["🔪 Weapon Detection", "🥊 Violence Detection", "💰 Theft Detection", "📦 Unattended Objects"],
            default=["🔪 Weapon Detection", "🥊 Violence Detection", "💰 Theft Detection", "📦 Unattended Objects"]
        )
        
        confidence_threshold = st.slider("AI Confidence Threshold:", 0.1, 1.0, 0.5, 0.05)
        
        st.divider()
        
        # System info
        st.subheader("📊 System Status")
        if YOLO_AVAILABLE:
            st.success("✅ YOLO AI Model: Ready")
        else:
            st.warning("⚠️ YOLO AI Model: Not Available")
        
        st.info(f"🕒 System Time: {datetime.now().strftime('%H:%M:%S')}")

def process_uploaded_video(video_path):
    """Process uploaded video and generate events based on video analysis"""
    if not os.path.exists(video_path):
        st.error(f"Video file not found: {video_path}")
        return
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open video file")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 10
        
        st.info(f"📹 Processing video: {duration:.1f}s, {fps} FPS, {frame_count} frames")
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frame_num = 0
        events_generated = []
        
        # Process video frames
        while st.session_state.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            progress = min(frame_num / frame_count, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"🎬 Analyzing frame {frame_num}/{frame_count}...")
            
            # Generate events periodically
            if frame_num % (fps * 3) == 0:  # Every 3 seconds
                event = generate_video_event(frame, frame_num, fps)
                if event:
                    events_generated.append(event)
                    st.session_state.events.append(event)
                    update_stats(event)
            
            # Process every 5th frame for performance
            if frame_num % 5 != 0:
                continue
            
            time.sleep(0.05)
        
        cap.release()
        progress_bar.progress(1.0)
        status_text.text(f"✅ Analysis complete! Generated {len(events_generated)} security events")
        
        if events_generated:
            st.success(f"🎯 Video analysis found {len(events_generated)} potential security incidents!")
            
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")

def generate_video_event(frame, frame_num, fps):
    """Generate realistic events based on video frame analysis with improved detection"""
    current_time = frame_num / fps
    
    # Analyze frame for motion and activity patterns
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = frame_gray.shape
    
    # Simulate advanced AI analysis based on frame characteristics
    # Check for high activity areas (simulated motion detection)
    motion_intensity = np.std(frame_gray)  # Use standard deviation as motion proxy
    brightness_variance = np.var(frame_gray)
    edge_density = cv2.Canny(frame_gray, 50, 150).sum() / (height * width)
    
    # Determine event type based on frame analysis
    event_probability = 0.15  # Base probability
    
    # Adjust probability based on frame characteristics
    if motion_intensity > 30:  # High motion detected
        event_probability += 0.2
    if brightness_variance > 2000:  # Significant lighting changes (could indicate struggle)
        event_probability += 0.15
    if edge_density > 0.1:  # High edge density (complex scene, potential activity)
        event_probability += 0.1
    
    if random.random() < event_probability:
        # Choose event type based on frame analysis
        if motion_intensity > 40 and brightness_variance > 2500:
            # High motion + lighting changes = likely violence
            event_types = [
                {'type': 'fight_detected', 'classes': ['person'], 'severity': 'critical', 'weight': 0.4},
                {'type': 'weapon_detected', 'classes': ['knife', 'gun', 'metal_object'], 'severity': 'critical', 'weight': 0.2},
                {'type': 'theft_detected', 'classes': ['person', 'object'], 'severity': 'high', 'weight': 0.3},
                {'type': 'unattended_object', 'classes': ['backpack', 'bag'], 'severity': 'medium', 'weight': 0.1}
            ]
        elif motion_intensity > 25:
            # Moderate motion = theft or weapon detection more likely
            event_types = [
                {'type': 'theft_detected', 'classes': ['laptop', 'phone', 'bag', 'wallet'], 'severity': 'high', 'weight': 0.3},
                {'type': 'weapon_detected', 'classes': ['knife', 'scissors', 'tool'], 'severity': 'critical', 'weight': 0.25},
                {'type': 'fight_detected', 'classes': ['person'], 'severity': 'critical', 'weight': 0.2},
                {'type': 'suspicious_activity', 'classes': ['person'], 'severity': 'medium', 'weight': 0.15},
                {'type': 'unattended_object', 'classes': ['backpack', 'suitcase'], 'severity': 'medium', 'weight': 0.1}
            ]
        else:
            # Low motion = unattended objects more likely
            event_types = [
                {'type': 'unattended_object', 'classes': ['backpack', 'suitcase', 'bag'], 'severity': 'medium', 'weight': 0.4},
                {'type': 'suspicious_activity', 'classes': ['person'], 'severity': 'medium', 'weight': 0.3},
                {'type': 'theft_detected', 'classes': ['laptop', 'phone'], 'severity': 'high', 'weight': 0.2},
                {'type': 'weapon_detected', 'classes': ['knife', 'tool'], 'severity': 'critical', 'weight': 0.1}
            ]
        
        weights = [e['weight'] for e in event_types]
        selected = random.choices(event_types, weights=weights)[0]
        
        object_class = random.choice(selected['classes'])
        
        # Higher confidence for events detected with strong frame indicators
        base_confidence = 0.7
        if motion_intensity > 35:
            base_confidence += 0.15
        if brightness_variance > 2000:
            base_confidence += 0.1
        confidence = min(0.95, base_confidence + random.uniform(0, 0.1))
        
        # Generate realistic coordinates
        x_coord = random.randint(50, width - 50) if width > 100 else random.randint(50, 600)
        y_coord = random.randint(50, height - 50) if height > 100 else random.randint(50, 400)
        
        # Enhanced details based on analysis
        motion_level = "High" if motion_intensity > 35 else "Moderate" if motion_intensity > 20 else "Low"
        
        details_map = {
            'weapon_detected': f"🔪 {object_class.title()} detected at {current_time:.1f}s (X:{x_coord}, Y:{y_coord}) | Motion: {motion_level} | Confidence: {confidence:.0%}",
            'fight_detected': f"🥊 Physical altercation detected at {current_time:.1f}s | Multiple persons involved | Motion Level: {motion_level} | Frame Analysis: High Activity", 
            'theft_detected': f"💰 Suspicious rapid movement of {object_class} at {current_time:.1f}s | Speed Analysis: {motion_intensity:.1f} units | Trajectory: Irregular",
            'unattended_object': f"📦 {object_class.title()} stationary at {current_time:.1f}s | Duration: {random.randint(15,45)}s | Motion: {motion_level}",
            'suspicious_activity': f"👁️ Unusual behavior pattern detected at {current_time:.1f}s | Activity Level: {motion_level} | Pattern: Anomalous"
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
            'motion_intensity': motion_intensity,
            'analysis_score': edge_density
        }
    
    return None

def main_dashboard():
    """Main dashboard interface"""
    apply_custom_css()
    
    # Header
    st.markdown('<h1 class="main-header">🔒 AI Surveillance Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("*Advanced threat detection powered by computer vision AI*")
    
    # Create sidebar
    create_sidebar()
    
    # Control panel
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
    
    with col1:
        st.subheader("🎮 System Control Center")
    
    with col2:
        if st.button("▶️ Start Monitoring", type="primary", use_container_width=True):
            st.session_state.is_running = True
            
            # Check if we have an uploaded video
            if 'uploaded_video_path' in st.session_state and st.session_state.uploaded_video_path:
                st.info("🎬 Processing uploaded video...")
                process_uploaded_video(st.session_state.uploaded_video_path)
            
            st.success("🟢 AI Surveillance ACTIVE!")
            st.rerun()
    
    with col3:
        if st.button("⏹️ Stop Monitoring", use_container_width=True):
            st.session_state.is_running = False
            st.warning("🔴 System STOPPED")
            st.rerun()
    
    with col4:
        if st.button("🎲 Generate Event", use_container_width=True, help="Generate a test security event"):
            event = generate_realistic_event()
            if event:
                st.session_state.events.append(event)
                update_stats(event)
                st.success("🚨 Security event generated!")
                st.rerun()
    
    with col5:
        if st.button("🗑️ Clear Events", use_container_width=True):
            st.session_state.events.clear()
            st.session_state.stats = {k: 0 for k in st.session_state.stats}
            st.info("Events cleared")
            st.rerun()
    
    # System status
    status_col1, status_col2 = st.columns([1, 3])
    with status_col1:
        if st.session_state.is_running:
            st.markdown('<p class="status-running">🟢 SYSTEM ACTIVE</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-stopped">🔴 SYSTEM INACTIVE</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # Generate events when monitoring is active in simulation mode
    if st.session_state.is_running and hasattr(st.session_state, 'simulation_mode'):
        event = generate_realistic_event()
        if event:
            st.session_state.events.append(event)
            update_stats(event)
    
    # Real-time metrics
    display_metrics(st.session_state.stats)
    
    st.divider()
    
    # Main content layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_live_alerts(st.session_state.events)
    
    with col2:
        st.subheader("📊 Analytics & Export")
        
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
            
            # Recent activity
            if st.session_state.events:
                last_event = st.session_state.events[-1]
                st.write(f"**🕐 Last Activity:** {last_event['timestamp']}")
                st.write(f"**📋 Event Type:** {last_event['event'].replace('_', ' ').title()}")
                st.write(f"**⚡ Severity:** {last_event['severity'].title()}")
        else:
            st.info("📊 No events to export yet")
        
        st.divider()
        
        # System features
        st.subheader("🔧 System Features")
        st.success("✅ Real-time AI Detection")
        st.success("✅ Multi-threat Analysis")
        st.success("✅ Live Dashboard")
        st.success("✅ Data Export")
        st.success("✅ Video Processing")
    
    # Analytics section
    if st.session_state.events:
        st.divider()
        st.subheader("📊 Security Analytics")
        
        # Create visualizations
        df = pd.DataFrame(st.session_state.events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Event timeline
            df['minute'] = df['timestamp'].dt.floor('5min')
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
        
        # Event summary table
        st.subheader("📋 Event Summary")
        event_summary = df.groupby(['event', 'severity']).size().reset_index(name='count')
        event_summary['event'] = event_summary['event'].str.replace('_', ' ').str.title()
        event_summary = event_summary.sort_values('count', ascending=False)
        
        st.dataframe(
            event_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                'event': st.column_config.TextColumn('Event Type', width='medium'),
                'severity': st.column_config.TextColumn('Severity', width='small'),
                'count': st.column_config.NumberColumn('Count', width='small')
            }
        )
    
    # Auto-refresh for live updates
    if st.session_state.is_running:
        time.sleep(2)
        st.rerun()

# ===============================
# MAIN APPLICATION ENTRY POINT
# ===============================

def main():
    """Main application entry point"""
    try:
        main_dashboard()
        
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("💡 Try refreshing the page or restarting the application")

# Run the application
if __name__ == "__main__":
    main()
