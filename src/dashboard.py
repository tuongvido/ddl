"""
Streamlit Dashboard for Real-time Harmful Content Detection Monitoring
Upgraded Version: Includes Image Evidence, Working Time Filters, and Grid Layout.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import time
from utils import MongoDBHandler, decode_base64_to_image
from config import MONGO_COLLECTION_VIDEO_SUMMARY

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Harmful Content Monitor",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .alert-card {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 5px solid #ccc;
        color: #1a1a1a;
        font-weight: 500;
    }
    .alert-HIGH { 
        background-color: #ffebee; 
        border-color: #f44336;
        color: #c62828;
    }
    .alert-MEDIUM { 
        background-color: #fff3e0; 
        border-color: #ff9800;
        color: #e65100;
    }
    .alert-LOW { 
        background-color: #fffde7; 
        border-color: #fbc02d;
        color: #f57f17;
    }
    
    /* Make the alert level text more visible */
    .alert-card strong {
        font-size: 1.1em;
    }
    
    /* Image Grid Styling */
    .stImage { border-radius: 5px; }
</style>
""",
    unsafe_allow_html=True,
)


# --- 2. HELPER FUNCTIONS ---


@st.cache_resource
def get_db_handler():
    """Get MongoDB handler (cached to avoid reconnecting)"""
    return MongoDBHandler()


def format_timestamp(ts):
    """Format timestamp for display"""
    if not ts:
        return "N/A"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def get_start_time(time_range_str):
    """Convert dropdown selection to actual timestamp"""
    now = datetime.now()
    if time_range_str == "Last 1 hour":
        return (now - timedelta(hours=1)).timestamp()
    elif time_range_str == "Last 6 hours":
        return (now - timedelta(hours=6)).timestamp()
    elif time_range_str == "Last 24 hours":
        return (now - timedelta(hours=24)).timestamp()
    elif time_range_str == "Last 7 days":
        return (now - timedelta(days=7)).timestamp()
    elif time_range_str == "All Time":
        return 0  # Show all data from beginning
    return (now - timedelta(hours=1)).timestamp()  # Default


def display_alert_row(alert):
    """Render a single alert using HTML/CSS"""
    level = alert.get("type", "LOW")
    ts = format_timestamp(alert.get("timestamp"))
    details = alert.get("details", "")
    type_ = alert.get("detection_type", "Unknown")
    conf = alert.get("confidence", 0)

    st.markdown(
        f"""
        <div class="alert-card alert-{level}">
            <strong>🚨 [{level}] {type_}</strong> <span style="float:right">{ts}</span><br>
            <small>Confidence: {conf:.1%}</small><br>
            {details}
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- 3. MAIN DASHBOARD LOGIC ---


def convert_to_timestamp(ts):
    """Convert timestamp to float if it's a datetime object"""
    if isinstance(ts, datetime):
        return ts.timestamp()
    return float(ts) if ts else 0


def main():
    st.title("🚨 Tiktok ")
    st.markdown("Real-time AI analysis for harmful content detection")

    # --- SIDEBAR: SETTINGS ---
    st.sidebar.title("⚙️ Configuration")

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 3, 60, 5)

    # Filters
    st.sidebar.subheader("Filters")
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last 1 hour", "Last 6 hours", "Last 24 hours", "Last 7 days", "All Time"],
        index=4,  # Default to "Last 24 hours"
    )
    selected_levels = st.sidebar.multiselect(
        "Alert Levels", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"]
    )

    # --- DATA FETCHING ---
    try:
        db = get_db_handler()
    except Exception as e:
        st.error(f"❌ Database Error: {e}")
        st.stop()

    # Calculate timestamps
    start_ts = get_start_time(time_range)

    # Fetch data (Fetching a bit more to filter in Python for simplicity)
    raw_detections = db.get_recent_detections(limit=1000)
    raw_alerts = db.get_recent_alerts(limit=500)

    detections = [
        d
        for d in raw_detections
        if convert_to_timestamp(d.get("timestamp", 0)) >= start_ts
    ]

    alerts = [
        a
        for a in raw_alerts
        if convert_to_timestamp(a.get("timestamp", 0)) >= start_ts
        and a.get("type", "LOW") in selected_levels
    ]

    # --- TABS LAYOUT ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Overview", "🚨 Alerts Log", "📹 Video Evidence", "🎤 Audio Analysis", "📋 Video Reports"]
    )

    # === TAB 1: OVERVIEW ===
    with tab1:
        # Top Metrics
        col1, col2, col3, col4 = st.columns(4)

        harmful_count = len(detections)
        high_priority = len([a for a in alerts if a.get("type") == "HIGH"])

        col1.metric("Total Frames Analyzed", len(detections))
        col2.metric("Harmful Detections", harmful_count)
        col3.metric("Total Alerts", len(alerts))
        col4.metric("High Priority", high_priority)

        st.divider()

        # Charts
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Alerts Timeline")
            if alerts:
                df_alerts = pd.DataFrame(alerts)
                df_alerts["datetime"] = pd.to_datetime(df_alerts["timestamp"], unit="s")
                fig = px.scatter(
                    df_alerts,
                    x="datetime",
                    y="confidence",
                    color="type",
                    symbol="detection_type",
                    color_discrete_map={
                        "HIGH": "red",
                        "MEDIUM": "orange",
                        "LOW": "gold",
                    },
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No alerts in this period.")

        with c2:
            st.subheader("Detection Distribution")
            if alerts:
                counts = {}
                for a in alerts:
                    t = a.get("detection_type", "Unknown")
                    counts[t] = counts.get(t, 0) + 1

                fig = px.pie(
                    names=list(counts.keys()), values=list(counts.values()), hole=0.4
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No data available.")

    # === TAB 2: ALERTS LOG ===
    with tab2:
        st.subheader("Live Alert Feed")
        if alerts:
            for alert in alerts[:20]:  # Show latest 20
                display_alert_row(alert)

            if len(alerts) > 20:
                st.caption(f"... and {len(alerts) - 20} more alerts.")
        else:
            st.success("No alerts matching filters.")

    # === TAB 3: VIDEO EVIDENCE (Major Upgrade) ===
    with tab3:
        st.subheader("📸 Harmful Content Gallery")

        harmful_frames = [d for d in detections if d.get("is_harmful") and "data" in d]

        if harmful_frames:
            st.warning(f"Found {len(harmful_frames)} frames with harmful content.")

            # Grid Layout (3 Columns)
            cols = st.columns(3)

            # Show latest 30 frames to avoid memory issues
            for idx, item in enumerate(harmful_frames[:30]):
                col = cols[idx % 3]  # Distribute items across columns

                with col:
                    with st.container(border=True):
                        # Decode & Display Image
                        try:
                            img = decode_base64_to_image(item["data"])
                            if img is not None:
                                # OpenCV is BGR, Streamlit needs RGB/BGR specified
                                # Use channels="BGR" to let Streamlit know format
                                st.image(img, channels="BGR", width="stretch")
                            else:
                                st.error("Image decode failed")
                        except Exception:
                            st.error("Image error")

                        # Meta info
                        ts = format_timestamp(item.get("timestamp"))
                        st.markdown(f"**Time:** {ts}")

                        # List Detections
                        for det in item.get("harmful_detections", []):
                            st.markdown(
                                f"🔴 **{det.get('class')}**: {det.get('confidence', 0):.1%}"
                            )
        else:
            st.success("No harmful video frames detected in this period.")

    # === TAB 4: AUDIO ANALYSIS ===
    with tab4:
        st.subheader("🎙️ Audio & Speech Analysis")

        # Lấy dữ liệu audio (có chunk_id) và lọc theo thời gian
        audio_events = [
            d
            for d in raw_detections
            if "chunk_id" in d
            and convert_to_timestamp(d.get("timestamp", 0)) >= start_ts
        ]

        if audio_events:
            # Sắp xếp mới nhất lên đầu
            audio_events.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

            for item in audio_events[:30]:  # Hiển thị 30 mẫu mới nhất
                # 1. Lấy thông tin từ DB
                timestamp = format_timestamp(item.get("timestamp"))

                # Thông tin Text (Toxic)
                text = item.get("transcribed_text", "")
                is_toxic = item.get("is_toxic", False)

                # Thông tin Âm thanh (Screaming, Explosion...)
                sound_label = item.get("sound_label", "Speech")
                sound_conf = item.get("sound_confidence", 0.0)
                is_screaming = item.get("is_screaming", False)  # Flag từ consumer

                # 2. Xử lý hiển thị

                # --- CASE A: ÂM THANH NGUY HIỂM (TIẾNG NỔ, SÚNG, HÉT) ---
                # Kiểm tra flag is_screaming hoặc check thủ công label
                harmful_sounds = [
                    "Screaming",
                    "Yelling",
                    "Explosion",
                    "Gunshot, gunfire",
                    "Bang",
                ]

                if is_screaming or (sound_label in harmful_sounds and sound_conf > 0.3):
                    st.markdown(
                        f"""
                    <div style="background-color: #ffebee; padding: 15px; border-radius: 8px; border-left: 6px solid #f44336; margin-bottom: 15px;">
                        <h4 style="color: #c62828; margin:0;">🔊 DANGER SOUND: {sound_label}</h4>
                        <span style="font-size: 0.9em; color: #555;">Detected at: {timestamp}</span><br>
                        <strong style="color: #c62828;">Confidence:</strong> <span style="color: #c62828;"> {sound_conf:.1%} </span>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # --- CASE B: LỜI NÓI ĐỘC HẠI ---
                if is_toxic:
                    st.error(
                        f'🤬 **Toxic Speech Detected** ({timestamp})\n\n> "{text}"'
                    )

                # --- CASE C: BÌNH THƯỜNG (Ẩn bớt để đỡ rối) ---
                # Chỉ hiện nếu không phải nguy hiểm và có text
                elif not is_screaming and not is_toxic:
                    with st.expander(f"ℹ️ Clean Audio Log - {timestamp}"):
                        st.markdown(f"**Sound:** {sound_label} ({sound_conf:.1%})")
                        st.markdown(f"**Transcript:** *{text}*")

        else:
            st.info("No audio analysis data found in the selected period.")

    # === TAB 5: VIDEO REPORTS ===
    with tab5:
        st.subheader("📋 Video Analysis Reports")
        
        try:
            # Fetch all video sessions
            video_sessions = list(db.db[MONGO_COLLECTION_VIDEO_SUMMARY].find().sort("start_time", -1).limit(50))
            
            if not video_sessions:
                st.info("No video analysis reports found. Reports will appear here after processing videos.")
            else:
                st.success(f"Found {len(video_sessions)} video analysis sessions")
                
                # Display summary statistics
                col1, col2, col3 = st.columns(3)
                
                total_videos = len(video_sessions)
                toxic_videos = len([s for s in video_sessions if s.get("is_toxic", False)])
                clean_videos = total_videos - toxic_videos
                
                col1.metric("Total Videos Analyzed", total_videos)
                col2.metric("Toxic Videos", toxic_videos, delta_color="inverse")
                col3.metric("Clean Videos", clean_videos, delta_color="normal")
                
                st.divider()
                
                # Filters for reports
                st.markdown("### Filter Reports")
                filter_col1, filter_col2 = st.columns(2)
                
                with filter_col1:
                    toxicity_filter = st.selectbox(
                        "Toxicity Status",
                        ["All", "Toxic Only", "Clean Only"],
                        index=0
                    )
                
                with filter_col2:
                    sort_by = st.selectbox(
                        "Sort By",
                        ["Most Recent", "Oldest First", "Most Toxic", "Longest Duration"],
                        index=0
                    )
                
                # Apply filters
                filtered_sessions = video_sessions.copy()
                
                if toxicity_filter == "Toxic Only":
                    filtered_sessions = [s for s in filtered_sessions if s.get("is_toxic", False)]
                elif toxicity_filter == "Clean Only":
                    filtered_sessions = [s for s in filtered_sessions if not s.get("is_toxic", False)]
                
                # Apply sorting
                if sort_by == "Oldest First":
                    filtered_sessions.reverse()
                elif sort_by == "Most Toxic":
                    filtered_sessions.sort(
                        key=lambda x: sum(x.get("detection_counts", {}).values()),
                        reverse=True
                    )
                elif sort_by == "Longest Duration":
                    filtered_sessions.sort(
                        key=lambda x: x.get("video_info", {}).get("duration_seconds", 0),
                        reverse=True
                    )
                
                st.divider()
                
                # Display each video report
                for session in filtered_sessions:
                    summary = session.get("summary")
                    
                    # Handle None summary - use session data directly
                    if summary is None:
                        summary = {}
                    
                    video_info = summary.get("video_info", session.get("video_info", {}))
                    is_toxic = session.get("is_toxic", False)
                    
                    # Card container
                    with st.container(border=True):
                        # Header
                        status_emoji = "⚠️" if is_toxic else "✅"
                        status_text = "TOXIC CONTENT DETECTED" if is_toxic else "CLEAN"
                        status_color = "#ffebee" if is_toxic else "#e8f5e9"
                        
                        st.markdown(
                            f"""
                            <div style="background-color: {status_color}; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                                <h3 style="margin:0;">{status_emoji} {video_info.get('video_name', 'Unknown Video')}</h3>
                                <p style="margin:5px 0 0 0; color: #666;">Session ID: {session.get('session_id', 'N/A')}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Video Info
                        info_col1, info_col2, info_col3 = st.columns(3)
                        with info_col1:
                            st.metric("Duration", f"{video_info.get('duration_seconds', 0):.1f}s")
                        with info_col2:
                            st.metric("Total Frames", video_info.get('total_frames', 0))
                        with info_col3:
                            st.metric("FPS", f"{video_info.get('fps', 0):.1f}")
                        
                        # Detection Statistics
                        if is_toxic:
                            st.markdown("#### Detection Summary")
                            det_stats = summary.get("detection_statistics", {})
                            toxic_cats = session.get("toxic_categories", [])
                            
                            stats_col1, stats_col2, stats_col3 = st.columns(3)
                            
                            with stats_col1:
                                if "violent_video" in toxic_cats:
                                    st.error(f"🎬 Violent Video: {det_stats.get('violent_video_count', 0)} times")
                            
                            with stats_col2:
                                if "violent_audio" in toxic_cats:
                                    st.error(f"🔊 Violent Audio: {det_stats.get('violent_audio_count', 0)} times")
                            
                            with stats_col3:
                                if "toxic_speech" in toxic_cats:
                                    st.error(f"💬 Toxic Speech: {det_stats.get('toxic_speech_count', 0)} times")
                            
                            # Detected Labels
                            detected_labels = summary.get("detected_labels", {})
                            
                            label_col1, label_col2 = st.columns(2)
                            
                            with label_col1:
                                video_actions = detected_labels.get("video_actions", [])
                                if video_actions:
                                    st.markdown("**🎬 Violent Video Actions:**")
                                    for item in video_actions[:5]:  # Top 5
                                        st.markdown(f"- {item['label']}: **{item['count']}** times")
                            
                            with label_col2:
                                audio_events = detected_labels.get("audio_events", [])
                                if audio_events:
                                    st.markdown("**🔊 Violent Audio Events:**")
                                    for item in audio_events[:5]:  # Top 5
                                        st.markdown(f"- {item['label']}: **{item['count']}** times")
                            
                            # Toxic Speech Samples
                            toxic_samples = summary.get("toxic_speech_samples", [])
                            if toxic_samples:
                                with st.expander(f"💬 Toxic Speech Samples ({len(toxic_samples)} found)"):
                                    for i, sample in enumerate(toxic_samples[:5], 1):
                                        ts = datetime.fromtimestamp(sample['timestamp']).strftime("%H:%M:%S")
                                        st.markdown(
                                            f"""
                                            **{i}. [{ts}]** (Confidence: {sample['confidence']:.1%})  
                                            > "{sample['text']}"
                                            """
                                        )
                        else:
                            st.success("✅ No toxic content detected in this video")
                        
                        # Analysis Time
                        analysis_time = summary.get("analysis_time", {})
                        if analysis_time:
                            st.caption(
                                f"Analyzed: {analysis_time.get('start_time', 'N/A')} → {analysis_time.get('end_time', 'N/A')} "
                                f"(Processing: {analysis_time.get('processing_duration_seconds', 0):.1f}s)"
                            )
                        
                        # Full Report Button
                        if summary.get("report_text"):
                            with st.expander("📄 View Full Text Report"):
                                st.code(summary["report_text"], language=None)
        
        except Exception as e:
            st.error(f"Error loading video reports: {e}")
            import traceback
            st.code(traceback.format_exc())

    # --- FOOTER & REFRESH ---
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
