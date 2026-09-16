"""
Plotly-based analytics section.
"""

import pandas as pd
import plotly.express as px
import streamlit as st


_EVENT_COLOURS = {
    "weapon_detected":   "#f44336",
    "fight_detected":    "#ff5722",
    "theft_detected":    "#ff9800",
    "unattended_object": "#9c27b0",
}

_SEVERITY_COLOURS = {
    "critical": "#f44336",
    "high":     "#ff9800",
    "medium":   "#9c27b0",
    "low":      "#4caf50",
}


def display_analytics(events: list):
    """Render timeline, severity pie, source bar, and confidence histogram."""
    if not events:
        return

    st.divider()
    st.subheader("📊 Security Analytics")

    df = pd.DataFrame(events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    col1, col2 = st.columns(2)

    with col1:
        df["minute"] = df["timestamp"].dt.floor("1min")
        timeline = df.groupby(["minute", "event"]).size().reset_index(name="count")
        if not timeline.empty:
            fig = px.bar(
                timeline, x="minute", y="count", color="event",
                title="🕐 Security Events Timeline",
                color_discrete_map=_EVENT_COLOURS,
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        sev_counts = df["severity"].value_counts()
        fig = px.pie(
            values=sev_counts.values, names=sev_counts.index,
            title="⚠️ Threat Severity Distribution",
            color_discrete_map=_SEVERITY_COLOURS,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        if "source" in df.columns:
            src_counts = df["source"].value_counts()
            fig = px.bar(
                x=src_counts.index, y=src_counts.values,
                title="📡 Event Sources",
                color=src_counts.values,
                color_continuous_scale="viridis",
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        if "confidence" in df.columns:
            fig = px.histogram(
                df, x="confidence", nbins=10,
                title="🎯 Confidence Distribution",
                color_discrete_sequence=["#2E86C1"],
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)


def display_debug_panel():
    """Show per-frame debug metrics (visible only when debug_mode is on)."""
    if not st.session_state.debug_mode or not st.session_state.frame_analysis_log:
        return

    import numpy as np

    st.subheader("🔧 Debug Information")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="debug-panel"><h4>📊 Detection Statistics</h4>', unsafe_allow_html=True)
        vals = list(st.session_state.last_motion_values)
        if vals:
            st.write(f"**Average Motion:** {np.mean(vals):.2f}")
        st.write(f"**Frames Processed:** {st.session_state.frames_processed}")
        st.write(f"**Events Generated:** {st.session_state.events_generated}")
        if st.session_state.frames_processed > 0:
            rate = st.session_state.events_generated / st.session_state.frames_processed * 100
            st.write(f"**Detection Rate:** {rate:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        latest = list(st.session_state.frame_analysis_log)[-1]
        st.markdown('<div class="debug-panel"><h4>🎯 Latest Frame Analysis</h4>', unsafe_allow_html=True)
        for label, key in [
            ("Motion Intensity", "motion"),
            ("Edge Density",     "edges"),
            ("Threat Score",     "threat_score"),
            ("Event Probability","event_probability"),
            ("Time Since Last",  "time_since_last"),
        ]:
            val = latest.get(key)
            if val is not None:
                fmt = f"{val:.4f}" if isinstance(val, float) else val
                st.write(f"**{label}:** {fmt}")
        st.markdown("</div>", unsafe_allow_html=True)
