# AI Surveillance Dashboard

A comprehensive real-time security monitoring system built with Streamlit that uses computer vision AI to detect various security threats and incidents.

## Overview

This surveillance dashboard provides an advanced threat detection system capable of monitoring video feeds, analyzing security events, and providing real-time alerts. The system supports multiple input sources including webcams, uploaded videos, and simulation mode for testing purposes.

## GitHub Repository and Streamlit Deployment

### Repository Structure
This project is hosted on GitHub and designed for easy deployment with Streamlit. The main application file is `surveillance_dashboard.py` which contains the complete dashboard implementation.

### Streamlit Cloud Deployment
The application can be deployed directly to Streamlit Cloud:

1. Fork or clone the repository to your GitHub account
2. Connect your GitHub account to Streamlit Cloud
3. Select the repository and specify `surveillance_dashboard.py` as the main file
4. The app will automatically deploy and be accessible via a public URL

### Local Development with Streamlit
For local development and testing:

```bash
# Clone the repository
git clone <repository-url>
cd ai-surveillance-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run surveillance_dashboard.py
# Alternatively use the streamlit dashboard and run the app by selecting the repository
```

### Streamlit Features Utilized
- **Session State Management**: Persistent data across user interactions
- **File Uploader**: Video file upload with validation
- **Real-time Updates**: Auto-refresh functionality for live monitoring
- **Interactive Widgets**: Sliders, buttons, and multi-select components
- **Custom CSS**: Enhanced styling for security alerts
- **Data Visualization**: Integration with Plotly for charts and graphs
- **Download Functionality**: CSV export for event logs

## Features

### Core Functionality

**Real-time Monitoring**
- Live video feed analysis from webcams
- Uploaded video file processing
- Simulation mode for testing and demonstration
- Continuous threat detection and event logging

**AI-Powered Detection**
- Weapon detection (knives, scissors, metal objects)
- Violence and fight detection
- Theft and suspicious activity monitoring
- Unattended object identification
- Configurable confidence thresholds

**Event Management**
- Real-time event generation and logging
- Severity classification (critical, high, medium, low)
- Detailed event descriptions with timestamps
- Automatic statistics tracking

**Dashboard Interface**
- Live security alerts with visual styling
- Real-time metrics display
- System status monitoring
- Interactive control panel

**Analytics and Reporting**
- Event timeline visualization
- Threat severity distribution charts
- Event summary tables
- CSV export functionality for event logs

## Installation

### Prerequisites

```bash
pip install streamlit
pip install opencv-python
pip install numpy
pip install pandas
pip install plotly
pip install ultralytics  # Optional for YOLO integration
```

### Basic Setup

1. Clone or download the surveillance dashboard code
2. Install the required dependencies
3. Run the application:

```bash
streamlit run surveillance_dashboard.py
```

## System Architecture

### Configuration Management

The system uses Streamlit's session state to manage:
- Event history and statistics
- System running status
- Video source configuration
- Temporary file handling

### Video Processing Modes

**Webcam Mode** provides direct camera feed access with configurable camera ID selection for real-time frame processing. This mode enables live monitoring of connected cameras with immediate threat detection and analysis.
**Upload Mode** supports multiple video formats including MP4, AVI, MOV, and MKV files. The system handles temporary file management, extracts and displays video metadata such as duration and resolution, and performs comprehensive frame-by-frame analysis of the uploaded content.
**Simulation Mode** generates realistic security events without requiring actual video processing. This mode is particularly useful for testing system functionality, demonstrating capabilities, and training purposes without needing live video feeds or uploaded files.

### Event Generation System

The system employs sophisticated algorithms to generate realistic security events:

**Event Types**
- Weapon Detection: Identifies dangerous objects with confidence scoring
- Violence Detection: Analyzes motion patterns for aggressive behavior
- Theft Detection: Monitors rapid object movements
- Unattended Objects: Tracks stationary items over time

**Event Attributes**
- Timestamp and location coordinates
- Confidence scores and tracking IDs
- Severity classification
- Detailed description with context

### AI Analysis Components

**Frame Analysis**
- Motion intensity calculation using standard deviation
- Brightness variance detection for lighting changes
- Edge density analysis for scene complexity
- Activity pattern recognition

**Threat Assessment**
- Dynamic probability calculation based on frame characteristics
- Multi-factor event type determination
- Confidence scoring with environmental factors
- Real-time risk evaluation

## User Interface Components

### Main Dashboard

**Control Panel**
- Start/Stop monitoring buttons
- Manual event generation for testing
- Event log clearing functionality
- System status indicators

**Metrics Display**
- Total events counter with daily delta
- Category-specific counters (weapons, fights, theft, unattended)
- Real-time statistics updates

**Live Alerts Section**
- Severity-based visual styling with CSS animations
- Chronological event display with timestamps
- Detailed event descriptions
- System status indicators when no threats detected

### Configuration Sidebar

**Video Source Settings**
- Source type selection (webcam/upload/simulation)
- Camera ID configuration
- File upload interface with validation
- Video metadata display

**Detection Settings**
- Multi-select detection type configuration
- Confidence threshold slider
- Real-time settings application

### Analytics Dashboard

**Visualization Components**
- Event timeline bar charts with 5-minute intervals
- Severity distribution pie charts
- Color-coded threat categorization

**Data Export**
- CSV export with timestamp-based filenames
- Event summary statistics
- Recent activity indicators

## Technical Implementation

### Session State Management

The application maintains persistent state across user interactions:

```python
st.session_state.events = []  # Event history
st.session_state.stats = {}   # Statistics counters
st.session_state.is_running = False  # System status
```

### Video Processing Pipeline

**Upload Handling**
- Temporary file creation and management
- Video validation and metadata extraction
- Frame-by-frame processing with progress indicators
- Automatic cleanup of temporary files

**Analysis Engine**
- Motion detection using OpenCV
- Statistical analysis of frame characteristics
- Event probability calculation
- Real-time event generation



## Event Types and Detection Logic

### Weapon Detection
- Identifies knives, scissors, and metal objects
- Uses confidence thresholds for validation
- Provides coordinate-based location data
- Includes motion analysis for context

### Violence Detection
- Analyzes motion intensity patterns
- Detects brightness variations indicating struggle
- Monitors multiple person interactions
- Scores violence likelihood

### Theft Detection
- Tracks rapid object movements
- Monitors irregular movement patterns
- Analyzes speed and trajectory data
- Identifies suspicious behavior patterns

### Unattended Object Monitoring
- Detects stationary objects over time
- Tracks duration of abandonment
- Identifies common unattended items
- Provides location-specific alerts

## Data Export and Reporting

### CSV Export Format
- Timestamp, event type, and severity
- Confidence scores and tracking IDs
- Object classes and detailed descriptions
- Location coordinates and analysis metrics

### Analytics Visualization
- Time-series event plotting
- Severity distribution analysis
- Event frequency tracking
- Pattern recognition reporting

## Configuration Options

### Detection Parameters
- Confidence threshold adjustment (0.1 to 1.0)
- Event type selection and filtering
- Analysis sensitivity settings

### Display Preferences
- Alert styling and animation controls
- Refresh rate configuration
- Dashboard layout options

## Error Handling and Reliability

### File Management
- Automatic temporary file cleanup
- Video format validation
- Error recovery mechanisms

### System Monitoring
- Component availability checking
- Graceful degradation for missing dependencies
- User-friendly error messaging

## Requirements

### System Requirements
- Python 3.7+ with Streamlit framework
- OpenCV for video processing
- Sufficient system memory for video analysis


This surveillance dashboard provides a comprehensive solution for security monitoring with advanced AI analysis capabilities, user-friendly interface and extensive customization options for all types of scenarios.
