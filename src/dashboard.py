"""
Streamlit Dashboard for Real-time Harmful Content Detection Monitoring
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import time
from utils import MongoDBHandler

# Page configuration
st.set_page_config(
    page_title="Harmful Content Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px;
        margin: 5px 0;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 10px;
        margin: 5px 0;
    }
    .alert-low {
        background-color: #fff9c4;
        border-left: 5px solid #ffeb3b;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_db_handler():
    """Get MongoDB handler (cached)"""
    return MongoDBHandler()


def format_timestamp(ts):
    """Format timestamp for display"""
    if isinstance(ts, str):
        return ts
    return ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "N/A"


def display_alert(alert):
    """Display single alert"""
    level = alert.get("level", "LOW")
    detection_type = alert.get("detection_type", "Unknown")
    timestamp = format_timestamp(alert.get("timestamp"))
    details = alert.get("details", "")
    confidence = alert.get("confidence", 0)

    alert_class = f"alert-{level.lower()}"

    st.markdown(
        f"""
    <div class="{alert_class}">
        <strong>🚨 {level} ALERT</strong> - {detection_type}<br>
        <small>Time: {timestamp} | Confidence: {confidence:.1%}</small><br>
        {details}
    </div>
    """,
        unsafe_allow_html=True,
    )


def main():
    """Main dashboard"""

    # Title
    st.title("🚨 Harmful Content Detection Dashboard")
    st.markdown("Real-time monitoring of livestream content analysis")

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    # Auto-refresh settings
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 30, 5)

    # Time range filter
    st.sidebar.selectbox(
        "Time Range", ["Last 1 hour", "Last 6 hours", "Last 24 hours", "Last 7 days"]
    )

    # Alert level filter
    alert_levels = st.sidebar.multiselect(
        "Alert Levels", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"]
    )

    # Get database handler
    try:
        db = get_db_handler()
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        st.stop()

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Overview", "🚨 Alerts", "📹 Video Detection", "🎤 Audio Detection"]
    )

    with tab1:
        st.header("System Overview")

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        # Get recent data
        recent_detections = db.get_recent_detections(limit=1000)
        recent_alerts = db.get_recent_alerts(limit=100)

        # Calculate metrics
        total_detections = len(recent_detections)
        harmful_detections = sum(1 for d in recent_detections if d.get("is_harmful"))
        total_alerts = len(recent_alerts)
        high_alerts = sum(1 for a in recent_alerts if a.get("level") == "HIGH")

        with col1:
            st.metric("Total Detections", total_detections)

        with col2:
            st.metric(
                "Harmful Content",
                harmful_detections,
                delta=f"{(harmful_detections / total_detections * 100):.1f}%"
                if total_detections > 0
                else "0%",
            )

        with col3:
            st.metric("Total Alerts", total_alerts)

        with col4:
            st.metric(
                "High Priority",
                high_alerts,
                delta=f"{(high_alerts / total_alerts * 100):.1f}%"
                if total_alerts > 0
                else "0%",
            )

        st.markdown("---")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Alert Distribution by Level")
            if recent_alerts:
                alert_levels_count = {}
                for alert in recent_alerts:
                    level = alert.get("level", "UNKNOWN")
                    alert_levels_count[level] = alert_levels_count.get(level, 0) + 1

                fig = px.pie(
                    values=list(alert_levels_count.values()),
                    names=list(alert_levels_count.keys()),
                    color=list(alert_levels_count.keys()),
                    color_discrete_map={
                        "HIGH": "#f44336",
                        "MEDIUM": "#ff9800",
                        "LOW": "#ffeb3b",
                    },
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No alerts yet")

        with col2:
            st.subheader("Detection Types")
            if recent_alerts:
                detection_types = {}
                for alert in recent_alerts:
                    det_type = alert.get("detection_type", "Unknown")
                    detection_types[det_type] = detection_types.get(det_type, 0) + 1

                fig = px.bar(
                    x=list(detection_types.keys()),
                    y=list(detection_types.values()),
                    labels={"x": "Detection Type", "y": "Count"},
                    color=list(detection_types.values()),
                    color_continuous_scale="reds",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No detections yet")

        # Timeline
        st.subheader("Alert Timeline")
        if recent_alerts:
            # Create timeline data
            timeline_data = []
            for alert in recent_alerts:
                timeline_data.append(
                    {
                        "timestamp": alert.get("timestamp"),
                        "level": alert.get("level", "UNKNOWN"),
                        "type": alert.get("detection_type", "Unknown"),
                        "confidence": alert.get("confidence", 0),
                    }
                )

            df = pd.DataFrame(timeline_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            fig = px.scatter(
                df,
                x="timestamp",
                y="confidence",
                color="level",
                symbol="type",
                color_discrete_map={
                    "HIGH": "#f44336",
                    "MEDIUM": "#ff9800",
                    "LOW": "#ffeb3b",
                },
                hover_data=["type", "level"],
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No timeline data yet")

    with tab2:
        st.header("Recent Alerts")

        # Filter alerts
        filtered_alerts = [
            a for a in recent_alerts if a.get("level", "LOW") in alert_levels
        ]

        if filtered_alerts:
            st.write(f"Showing {len(filtered_alerts)} alerts")

            # Display alerts
            for alert in filtered_alerts[:20]:  # Show last 20
                display_alert(alert)
        else:
            st.info("No alerts matching the selected filters")

    with tab3:
        st.header("Video Detection Results")

        # Get video detections
        video_detections = [d for d in recent_detections if "frame_id" in d]

        if video_detections:
            st.write(f"Total frames processed: {len(video_detections)}")

            # Stats
            col1, col2, col3 = st.columns(3)

            with col1:
                harmful_frames = sum(1 for d in video_detections if d.get("is_harmful"))
                st.metric(
                    "Harmful Frames",
                    harmful_frames,
                    delta=f"{(harmful_frames / len(video_detections) * 100):.1f}%",
                )

            with col2:
                total_objects = sum(
                    d.get("total_detections", 0) for d in video_detections
                )
                st.metric("Total Objects Detected", total_objects)

            with col3:
                harmful_objects = sum(
                    d.get("harmful_count", 0) for d in video_detections
                )
                st.metric("Harmful Objects", harmful_objects)

            # Recent detections table
            st.subheader("Recent Detections")

            detection_data = []
            for det in video_detections[:50]:
                detection_data.append(
                    {
                        "Frame ID": det.get("frame_id"),
                        "Timestamp": format_timestamp(det.get("timestamp")),
                        "Total Detections": det.get("total_detections", 0),
                        "Harmful": det.get("is_harmful", False),
                        "Harmful Count": det.get("harmful_count", 0),
                    }
                )

            df = pd.DataFrame(detection_data)
            st.dataframe(df, width="stretch")
        else:
            st.info("No video detections yet")

    with tab4:
        st.header("Audio Detection Results")

        # Get audio detections
        audio_detections = [d for d in recent_detections if "chunk_id" in d]

        if audio_detections:
            st.write(f"Total audio chunks processed: {len(audio_detections)}")

            # Stats
            col1, col2 = st.columns(2)

            with col1:
                toxic_chunks = sum(1 for d in audio_detections if d.get("is_toxic"))
                st.metric(
                    "Toxic Speech Detected",
                    toxic_chunks,
                    delta=f"{(toxic_chunks / len(audio_detections) * 100):.1f}%",
                )

            with col2:
                total_toxic_score = sum(
                    d.get("toxic_score", 0) for d in audio_detections
                )
                st.metric("Total Toxic Keywords", total_toxic_score)

            # Recent transcriptions
            st.subheader("Recent Transcriptions")

            for det in audio_detections[:10]:
                if det.get("is_toxic"):
                    st.warning(f"""
                    **Chunk {det.get("chunk_id")}** - Toxic Score: {det.get("toxic_score", 0)}  
                    Keywords: {", ".join(det.get("matched_keywords", []))}  
                    Text: {det.get("transcribed_text", "N/A")}
                    """)
                else:
                    st.info(f"""
                    **Chunk {det.get("chunk_id")}** - Clean  
                    Text: {det.get("transcribed_text", "N/A")}
                    """)
        else:
            st.info("No audio detections yet")

    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
