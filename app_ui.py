import os
import cv2
import time
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st

from config import PROJECT_DIR, YOLO_GUNS_WEIGHTS, OUTPUT_DIR
from pipeline_manager import RobberyDetectionPipeline

st.set_page_config(
    page_title="Robbery & Threat Detection (YOLO + CLIP + Florence-2)",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        color: #ef4444;
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        text-align: center;
        margin-bottom: 20px;
    }
    .alert-card {
        background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 16px;
        color: #fecaca;
        margin-bottom: 15px;
    }
    .suppressed-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #38bdf8;
        border-radius: 12px;
        padding: 16px;
        color: #e2e8f0;
        margin-bottom: 15px;
    }
    .normal-card {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 16px;
        color: #a7f3d0;
        margin-bottom: 15px;
    }
    .stage-header {
        color: #38bdf8;
        font-weight: 700;
        font-size: 1.2rem;
        border-bottom: 1px solid #334155;
        padding-bottom: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚨 Robbery & Threat Detection Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-Stage Real-Time Pipeline: <b>YOLO Edge Detection</b> ➔ <b>Target Crop & Super-Resolution</b> ➔ <b>Florence-2 VLM Auditor & CLIP Verification</b></div>', unsafe_allow_html=True)

# Initialize pipeline lazily in session state
@st.cache_resource
def load_pipeline():
    return RobberyDetectionPipeline()

pipeline = load_pipeline()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    st.subheader("1. YOLO Edge Settings")
    conf_threshold = st.slider("YOLO Confidence Threshold", min_value=0.20, max_value=0.85, value=0.45, step=0.05)
    pipeline.stage1_scanner.model_path = YOLO_GUNS_WEIGHTS
    
    st.subheader("2. Crop & Super-Resolution")
    upscale_factor = st.select_slider("Super-Resolution Upscale", options=[1.5, 2.0, 3.0], value=2.0)
    pipeline.stage2_enhancer.upscale_factor = upscale_factor

    st.subheader("3. VLM Verification Status")
    st.success("🟢 VLM Auditor Active: Suppresses False Positives")
    st.caption(f"📁 **Project Folder:** `{PROJECT_DIR}`")
    st.caption(f"🧠 **OpenAI CLIP:** `openAI clip`")
    st.caption(f"👁️ **Florence-2 VLM:** `models`")

# Main Dashboard Interface
input_type = st.radio("Select Input Source", ["📁 Upload Image / Video File", "🖼️ Sample Security Test Scene"], horizontal=True)

# Incident History Log Session State
if "incident_log" not in st.session_state:
    st.session_state["incident_log"] = []

if input_type == "📁 Upload Image / Video File":
    uploaded_file = st.file_uploader(
        "Choose an image or video file (JPG, PNG, WEBP, MP4, AVI, MOV, MKV)...",
        type=["jpg", "jpeg", "png", "webp", "bmp", "mp4", "avi", "mov", "mkv", "webm"]
    )

    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        is_video = file_ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]

        if is_video:
            tfile = os.path.join(OUTPUT_DIR, "temp_input_video.mp4")
            with open(tfile, "wb") as f:
                f.write(uploaded_file.read())

            st.subheader("🎥 Video Stream Analysis Mode")
            st.video(tfile)

            if st.button("🚀 Analyze Video Stream with 3-Stage Pipeline", type="primary", use_container_width=True):
                cap = cv2.VideoCapture(tfile)
                
                col1, col2 = st.columns([1.2, 1.0])
                with col1:
                    st.markdown('<div class="stage-header">1. Real-Time Detection Stream</div>', unsafe_allow_html=True)
                    st_frame = st.empty()
                with col2:
                    st.markdown('<div class="stage-header">2 & 3. Stage Outputs & VLM Verification</div>', unsafe_allow_html=True)
                    st_alert = st.empty()
                    st_crop = st.empty()

                frame_count = 0
                alerts_fired = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if frame_count % 3 != 0:  # Skip frames for real-time speed
                        continue

                    res = pipeline.process_frame(frame)
                    ann_rgb = cv2.cvtColor(res["annotated_frame"], cv2.COLOR_BGR2RGB)
                    st_frame.image(ann_rgb, caption=f"Processing Video Frame #{frame_count}", use_column_width=True)

                    if res["triggered"]:
                        if res["verified_threat"]:
                            alerts_fired += 1
                            st_alert.markdown(f"""
                            <div class="alert-card">
                                <h3>🚨 REAL THREAT CONFIRMED (Frame #{frame_count})</h3>
                                <p><b>Category:</b> {res['top_category']} ({res['top_score']}%)</p>
                                <p><b>VLM Explanation:</b> "{res['reasoning_sentence']}"</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st_alert.markdown(f"""
                            <div class="suppressed-card">
                                <h3>🟢 FALSE ALARM SUPPRESSED (Frame #{frame_count})</h3>
                                <p><b>YOLO Candidate:</b> Detected potential object, but Florence-2 VLM verified no weapon present.</p>
                                <p><b>VLM Explanation:</b> "{res['reasoning_sentence']}"</p>
                            </div>
                            """, unsafe_allow_html=True)

                        if res["crop_pil"] is not None:
                            st_crop.image(res["crop_pil"], caption="Stage 2 Context Crop (Person + Object)", width=320)

                        # Log incident
                        st.session_state["incident_log"].append({
                            "Timestamp": time.strftime("%H:%M:%S"),
                            "Source": uploaded_file.name,
                            "Frame #": frame_count,
                            "Status": "🚨 THREAT CONFIRMED" if res["verified_threat"] else "🟢 FALSE ALARM SUPPRESSED",
                            "Trigger": ", ".join([d["label"] for d in res["detections"]]),
                            "CLIP Category": res["top_category"],
                            "Florence-2 Explanation": res["reasoning_sentence"]
                        })

                cap.release()
                if alerts_fired == 0:
                    st.success("✅ Video analysis complete. No verified security threats detected.")
                else:
                    st.warning(f"⚠️ Video analysis complete. Fired {alerts_fired} verified threat alert(s).")
        else:
            # Process as single image
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame is not None:
                col1, col2, col3 = st.columns([1.2, 1.0, 1.2])

                with st.spinner("Processing image through 3-Stage Pipeline..."):
                    res = pipeline.process_frame(frame)

                with col1:
                    st.markdown('<div class="stage-header">1. YOLO Edge Detection</div>', unsafe_allow_html=True)
                    ann_rgb = cv2.cvtColor(res["annotated_frame"], cv2.COLOR_BGR2RGB)
                    st.image(ann_rgb, caption="Stage 1: Full Frame Annotations", use_column_width=True)

                with col2:
                    st.markdown('<div class="stage-header">2. Target Crop & Super-Res</div>', unsafe_allow_html=True)
                    if res["triggered"] and res["crop_pil"] is not None:
                        st.image(res["crop_pil"], caption="Stage 2: Enhanced Crop ROI (Context Buffer)", use_column_width=True)
                        st.caption(f"📏 Enhanced Size: `{res['crop_pil'].width} x {res['crop_pil'].height} px`")
                    else:
                        st.info("No candidate detection for cropping.")

                with col3:
                    st.markdown('<div class="stage-header">3. Reasoning Engine</div>', unsafe_allow_html=True)
                    if res["triggered"]:
                        if res["verified_threat"]:
                            st.markdown(f"""
                            <div class="alert-card">
                                <h3>🚨 REAL THREAT CONFIRMED</h3>
                                <p><b>Category:</b> {res['top_category']} ({res['top_score']}%)</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="suppressed-card">
                                <h3>🟢 FALSE ALARM SUPPRESSED</h3>
                                <p>YOLO detected candidate, but Florence-2 VLM verified no weapon or threat present.</p>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("#### 🧠 Florence-2 VLM Reasoning Sentence:")
                        st.success(f'"{res["reasoning_sentence"]}"')

                        st.markdown("#### 📊 CLIP Zero-Shot Scores:")
                        if res["clip_scores"]:
                            df_clip = pd.DataFrame(list(res["clip_scores"].items()), columns=["Category", "Probability %"])
                            st.dataframe(df_clip, use_container_width=True)

                        # Log incident
                        st.session_state["incident_log"].append({
                            "Timestamp": time.strftime("%H:%M:%S"),
                            "Source": uploaded_file.name,
                            "Frame #": 1,
                            "Status": "🚨 THREAT CONFIRMED" if res["verified_threat"] else "🟢 FALSE ALARM SUPPRESSED",
                            "Trigger": ", ".join([d["label"] for d in res["detections"]]),
                            "CLIP Category": res["top_category"],
                            "Florence-2 Explanation": res["reasoning_sentence"]
                        })
                    else:
                        st.markdown("""
                        <div class="normal-card">
                            <h3>🟢 Normal Scene</h3>
                            <p>No weapons or threat objects detected in edge scan.</p>
                        </div>
                        """, unsafe_allow_html=True)

elif input_type == "🖼️ Sample Security Test Scene":
    sample_path = r"E:\Aitronix Projects\Robber Detection\test_security_scene.png"
    if os.path.exists(sample_path):
        frame = cv2.imread(sample_path)
        col1, col2, col3 = st.columns([1.2, 1.0, 1.2])

        with st.spinner("Processing Sample Security Scene..."):
            res = pipeline.process_frame(frame)

        with col1:
            st.markdown('<div class="stage-header">1. YOLO Edge Detection</div>', unsafe_allow_html=True)
            ann_rgb = cv2.cvtColor(res["annotated_frame"], cv2.COLOR_BGR2RGB)
            st.image(ann_rgb, caption="Stage 1: Full Frame Annotations", use_column_width=True)

        with col2:
            st.markdown('<div class="stage-header">2. Target Crop & Super-Res</div>', unsafe_allow_html=True)
            if res["triggered"] and res["crop_pil"] is not None:
                st.image(res["crop_pil"], caption="Stage 2: Enhanced Crop ROI", use_column_width=True)

        with col3:
            st.markdown('<div class="stage-header">3. Reasoning Engine</div>', unsafe_allow_html=True)
            if res["triggered"]:
                if res["verified_threat"]:
                    st.markdown(f"""
                    <div class="alert-card">
                        <h3>🚨 REAL THREAT CONFIRMED</h3>
                        <p><b>Category:</b> {res['top_category']} ({res['top_score']}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="suppressed-card">
                        <h3>🟢 FALSE ALARM SUPPRESSED</h3>
                        <p>YOLO detected candidate, but Florence-2 VLM verified no weapon present.</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("#### 🧠 Florence-2 VLM Reasoning Sentence:")
                st.success(f'"{res["reasoning_sentence"]}"')

# Display Incident History Logs
st.divider()
st.subheader("📋 Incident Alert History Logs")
if st.session_state["incident_log"]:
    df_log = pd.DataFrame(st.session_state["incident_log"]).drop_duplicates()
    st.dataframe(df_log, use_container_width=True)
else:
    st.info("No security trigger incidents logged yet.")
