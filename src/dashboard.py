"""
Streamlit Dashboard – Per-Video Session Monitoring
Each video session has its own 5-tab panel.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, date, time as dtime
import time

from utils import MongoDBHandler, decode_base64_to_image
from config import MONGO_COLLECTION_VIDEO_SUMMARY


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Harmful Content Monitor",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
header.stAppHeader {
    visibility: hidden;
}
.stMainBlockContainer {
    padding: 10px 80px
}
.stSidebar {
    padding: 40px 0
}
#detecting-harmful-video-content-on-tik-tok {
    padding: 20px 0 60px 0;
    text-align: center;
}
.alert-card {
    padding: 10px;
    border-radius: 6px;
    margin-bottom: 10px;
    border-left: 5px solid #ccc;
    font-weight: 500;
}
.alert-HIGH { background:#ffebee; border-color:#f44336; color:#c62828; }
.alert-MEDIUM { background:#fff3e0; border-color:#ff9800; color:#e65100; }
.alert-LOW { background:#fffde7; border-color:#fbc02d; color:#f57f17; }
.toxic-panel {
    background-color: #ffebee;
    border-left: 6px solid #f44336;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 12px;
}

.normal-panel {
    background-color: #ffffff;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================
@st.cache_resource
def get_db():
    return MongoDBHandler()


def ts(ts):
    if isinstance(ts, datetime):
        return ts.timestamp()
    return float(ts or 0)


def fmt(ts_):
    if not ts_:
        return "N/A"
    return datetime.fromtimestamp(ts_).strftime("%Y-%m-%d %H:%M:%S")


def render_alert(alert):
    st.markdown(
        f"""
        <div class="alert-card alert-{alert.get('type','LOW')}">
            <strong>[{alert.get('type')}] {alert.get('detection_type','Unknown')}</strong>
            <span style="float:right">{fmt(alert.get('timestamp'))}</span><br>
            Confidence: {alert.get('confidence',0):.1%}<br>
            {alert.get('details','')}
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# MAIN
# =========================================================
def main():
    st.title("Detecting harmful video content on TikTok")

    st.sidebar.header("Filter")

    start_date = st.sidebar.date_input(
        "Start date",
        value=date.today(),
        key="start_date",
    )

    start_time = st.sidebar.time_input(
        "Start time",
        value=dtime(0, 0),
        key="start_time",
    )

    end_date = st.sidebar.date_input(
        "End date",
        value=date.today(),
        key="end_date",
    )

    end_time = st.sidebar.time_input(
        "End time",
        value=dtime(23, 59),
        key="end_time",
    )

    auto_refresh = st.sidebar.checkbox("Auto Refresh", True)
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 3, 60, 30)

    db = get_db()

    video_summary = list(
        db.db[MONGO_COLLECTION_VIDEO_SUMMARY]
        .find()
        .sort("start_time", -1)
        .limit(20)
    )

    if not video_summary:
        st.info("No video found")
        return

    for video in video_summary:
        session_id = video.get("session_id")
        video_info = video.get("video_info", {})
        video_name = os.path.basename(video_info.get("video_path", ""))
        is_toxic = video.get("is_toxic", False)

        status_badge = (
            "🔴 TOXIC CONTENT DETECTED"
            if is_toxic
            else "🟢 NO TOXIC CONTENT"
        )

        expander_title = (
            f"🎬 {video_name} \u2003 | \u2003 {status_badge}"
        )

        detections = [
            d for d in db.get_detections_by_session(session_id)
        ]

        alerts = [
            a for a in db.get_alerts_by_session(session_id)
        ]

        with st.expander(expander_title, expanded=False):

            if is_toxic:
                st.error("⚠️ TOXIC CONTENT DETECTED")

            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["📊 Overview", "🚨 Alerts", "📹 Video Evidence", "🎤 Audio", "📋 Report"]
            )
            
            summary = video.get("summary")

            # ---------------- TAB 1 ----------------
            with tab1:
                col1, col2, col3, col4 = st.columns(4)

                harmful_frames = [d for d in detections if d.get("is_harmful")]

                col1.metric("Frames", len(detections))
                col2.metric("Harmful Frames", len(harmful_frames))
                col3.metric("Alerts", len(alerts))
                col4.metric(
                    "HIGH Alerts", len([a for a in alerts if a.get("type") == "HIGH"])
                )

                if alerts:
                    df = pd.DataFrame(alerts)
                    fig = px.scatter(
                        df,
                        x="frame_id",
                        y="confidence",
                        color="type",
                        symbol="detection_type",
                        labels={
                            "frame_id": "Frame ID",
                            "confidence": "Confidence"
                        },
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Full Report Button
                if summary.get("report_text"):
                    with st.expander("📄 View Full Text Report"):
                        st.code(summary["report_text"], language=None)

            # ---------------- TAB 2 ----------------
            with tab2:
                if alerts:
                    for a in alerts[:30]:
                        render_alert(a)
                else:
                    st.success("No alerts")

            # ---------------- TAB 3 ----------------
            with tab3:
                frames = [
                    d for d in detections
                    if d.get("is_harmful") and "data" in d
                ]

                if not frames:
                    st.info("No harmful frames")
                else:
                    cols = st.columns(3)
                    for i, f in enumerate(frames[:30]):
                        with cols[i % 3]:
                            img = decode_base64_to_image(f["data"])
                            if img is not None:
                                st.image(img, channels="BGR", width="stretch")
                            st.caption(fmt(f.get("timestamp")))

            # ---------------- TAB 4 ----------------
            with tab4:
                audio = [d for d in detections if "chunk_id" in d]

                if not audio:
                    st.info("No audio data")

                for a in audio[:30]:
                    timestamp = fmt(a.get("timestamp"))

                    if a.get("is_screaming"):
                        st.error(f"🔊 Dangerous sound ({timestamp})")

                    if a.get("is_toxic"):
                        st.error(
                            f"💬 Toxic speech ({timestamp})\n\n{a.get('transcribed_text','')}"
                        )

            # ---------------- TAB 5 ----------------
            with tab5:
                if not summary:
                    st.info("No report available")
                else:
                    st.metric(
                        "Status",
                        "TOXIC" if video.get("is_toxic") else "CLEAN",
                    )

                    st.json(summary)


    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
