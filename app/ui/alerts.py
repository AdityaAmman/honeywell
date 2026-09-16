"""
Live alert display and metrics panel.
"""

import streamlit as st


_SOURCE_LABEL = {
    "webcam_live":   "🔴 LIVE",
    "video_upload":  "📹 VIDEO",
    "simulation":    "🎮 SIM",
    "manual_test":   "🧪 TEST",
}

_EVENT_SUMMARY = {
    "weapon_detected":   lambda obj, conf: f"🔪 **WEAPON ALERT** — {obj.title()} detected ({conf:.0%} confidence)",
    "fight_detected":    lambda obj, conf: "🥊 **VIOLENCE DETECTED** — Physical altercation in progress",
    "theft_detected":    lambda obj, conf: f"💰 **THEFT ALERT** — Suspicious movement of {obj}",
    "unattended_object": lambda obj, conf: f"📦 **SECURITY NOTICE** — {obj.title()} left unattended",
}


def display_metrics(stats: dict):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🚨 Total Events", stats["total_events"])
    col2.metric("🔪 Weapons",      stats["weapons_detected"])
    col3.metric("🥊 Fights",       stats["fights_detected"])
    col4.metric("💰 Theft",        stats["theft_attempts"])
    col5.metric("📦 Unattended",   stats["unattended_objects"])


def display_live_alerts(events: list):
    st.subheader("🚨 Live Security Alerts")

    if not events:
        st.info("🔍 AI monitoring active… No threats detected.")
        return

    recent = list(reversed(events))[:10]
    for event in recent:
        severity = event.get("severity", "medium")
        obj = event.get("class", "unknown")
        conf = event.get("confidence", 0.0)

        fmt = _EVENT_SUMMARY.get(event["event"])
        summary = fmt(obj, conf) if fmt else f"⚠️ **{event['event'].replace('_', ' ').title()}**"

        source_label = _SOURCE_LABEL.get(event.get("source", ""), "⚪")
        conf_display = f"{conf:.0%}"

        st.markdown(
            f"""
            <div class="alert-{severity}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="flex:1;">
                        {summary} | {source_label} | Confidence: {conf_display}
                    </div>
                    <div style="color:#666; font-size:0.9rem; margin-left:1rem;">
                        {event['timestamp']}
                    </div>
                </div>
                <div style="margin-top:0.5rem; font-size:0.85rem; color:#555;">
                    📍 {event['details']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not recent and st.session_state.is_running:
        st.success("✅ **NO THREATS DETECTED** — Area secure")
