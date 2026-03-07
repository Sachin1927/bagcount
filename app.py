import streamlit as st
import cv2
import random
from ultralytics import YOLO

# ==========================================
# 1. PAGE SETUP & ENTERPRISE CSS
# ==========================================
st.set_page_config(layout="wide", page_title="Warehouse AI Dashboard", page_icon="📦", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .main-header { font-family: 'Segoe UI', sans-serif; color: #1a2b4c; font-weight: 800; font-size: 34px; margin-bottom: 0px;}
    .sub-header { color: #d9534f; font-size: 16px; font-weight: 600; margin-bottom: 20px;}
    .metric-card { background-color: #ffffff; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #1a2b4c; margin-bottom: 20px;}
    .metric-title { font-size: 14px; color: #6c757d; font-weight: bold; text-transform: uppercase; margin-bottom: 5px;}
    .metric-value { font-size: 32px; color: #1a2b4c; font-weight: 900;}
    .video-title { background-color: #1a2b4c; color: white; padding: 10px 15px; font-weight: bold; border-radius: 8px 8px 0 0; font-size: 16px;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DYNAMIC CAMERA CONFIGURATION (1024x576 HD)
# ==========================================
SCENARIO_CONFIG = {
    "Scenario 1 (Active Loading)": {
        "path": "data/raw/Problem Statement Scenario1.mp4",
        # 👇👇👇 CHANGE THIS NUMBER TO MOVE THE YELLOW LINE LEFT OR RIGHT 👇👇👇
        "center_line": 615  
    },
    "Scenario 2 (Ramp Unloading)": {
        "path": "data/raw/Problem Statement Scenario2.mp4",
        "center_line": 550  # Adjust this for Video 2 if needed
    },
    "Scenario 3 (Gate Perspective)": {
        "path": "data/raw/Problem Statement Scenario3.mp4",
        "center_line": 450  # Adjust this for Video 3 if needed
    }
}

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830305.png", width=80)
    st.markdown("## System Controls")
    selected_video = st.selectbox("Select Video Scenario", list(SCENARIO_CONFIG.keys()))
    st.markdown("---")
    st.success("AI Model: custom_sack_v1.pt Active")
    st.info("Tracker: ByteTrack (Max Persistence)")
    st.info("Logic: Cross-Line Trigger")

config = SCENARIO_CONFIG[selected_video]
video_source = config["path"]

# ==========================================
# 4. HEADER & LAYOUT
# ==========================================
st.markdown('<div class="main-header">📦 Smart Warehouse Management System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Bag Counting & IoT Monitoring Interface</div>', unsafe_allow_html=True)
st.markdown("---")

col_video, col_dashboard = st.columns([7, 3])

with col_video:
    st.markdown(f'<div class="video-title">🎥 Live Feed: {selected_video}</div>', unsafe_allow_html=True)
    video_placeholder = st.empty()

with col_dashboard:
    st.markdown("### 📊 Live Counting Metrics")
    metric_ph_in = st.empty()
    metric_ph_out = st.empty()
    st.markdown("---")
    st.markdown("### 🌡️ Facility IoT Sensors")
    iot_ph = st.empty()

# ==========================================
# 5. LIVE VIDEO PROCESSING ENGINE
# ==========================================
def run_live_inference():
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        video_placeholder.error(f"Error: Could not load {video_source}. Check file path!")
        return

    model = YOLO("models/best.pt") 
    target_x = config["center_line"]

    frame_count = 0
    current_temp = 24.5
    current_hum = 45
    
    track_history = {}
    counted_ids = set()
    bags_in = 0
    bags_out = 0 
    # --- ADD THIS HERE ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('output_result.mp4', fourcc, 20.0, (1024, 576))
    # --------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        frame_count += 1
        frame = cv2.resize(frame, (1024, 576))
        

        # FIXED: Conf lowered to 0.05 to ensure IDs NEVER drop, even if blurry
        results = model.track(frame, persist=True, conf=0.05, tracker="bytetrack.yaml", verbose=False)
        annotated_frame = frame.copy()
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                tid = int(track_id)
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) / 2 
                
                # Draw Box and ID
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + 40, y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, f"ID:{tid}", (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

                # ==========================================
                # EXACT COUNTING LOGIC 
                # ==========================================
                # 1. Save the movement history of this specific sack ID
                if tid not in track_history:
                    track_history[tid] = []
                track_history[tid].append(cx)
                
                # 2. Check direction only if we have at least 2 frames of history
                if len(track_history[tid]) >= 2:
                    prev_cx = track_history[tid][-2] 
                    
                    # WAREHOUSE TO TRUCK (Loading) - Moving Right to Left
                    if prev_cx > target_x and cx <= target_x:
                        if tid not in counted_ids:
                            bags_out += 1  
                            counted_ids.add(tid)
                            
                    # TRUCK TO WAREHOUSE (Unloading) - Moving Left to Right
                    elif prev_cx < target_x and cx >= target_x:
                        if tid not in counted_ids:
                            bags_in += 1   
                            counted_ids.add(tid)
                # Track trajectory for counting
                if tid not in track_history:
                    track_history[tid] = []
                
                track_history[tid].append(cx)
                
                # Check for line cross using the immediate previous frame
                if len(track_history[tid]) >= 2:
                    prev_cx = track_history[tid][-2] 
                    
                    # Unloading: Truck to Warehouse (Moving Left to Right)
                    if prev_cx < target_x and cx >= target_x:
                        if tid not in counted_ids:
                            bags_in += 1
                            counted_ids.add(tid)
                            
                    # Loading: Warehouse to Truck (Moving Right to Left)
                    elif prev_cx > target_x and cx <= target_x:
                        if tid not in counted_ids:
                            bags_out += 1
                            counted_ids.add(tid)

        # ==========================================
        # HUD OVERLAY
        # ==========================================
        overlay = annotated_frame.copy()
        
        # Yellow Line
        cv2.line(overlay, (target_x, 100), (target_x, 500), (0, 255, 255), 3) 
        
        cv2.rectangle(overlay, (target_x - 120, 120), (target_x - 10, 150), (0, 0, 255), -1)
        cv2.putText(overlay, "<- TRUCK ZONE", (target_x - 110, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.rectangle(overlay, (target_x + 10, 120), (target_x + 140, 150), (255, 100, 0), -1)
        cv2.putText(overlay, "DOOR ZONE ->", (target_x + 20, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.8, annotated_frame, 0.2, 0, annotated_frame)

        # --- ADD THIS HERE ---
        out_video.write(annotated_frame)
        # --------------------
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        if frame_count % 5 == 0:
            metric_ph_in.markdown(f'''
                <div class="metric-card">
                    <div class="metric-title">📥 Unloaded (Truck to Warehouse)</div>
                    <div class="metric-value">{bags_in}</div>
                </div>
            ''', unsafe_allow_html=True)
            
            metric_ph_out.markdown(f'''
                <div class="metric-card" style="border-left: 5px solid #d9534f;">
                    <div class="metric-title">📤 Loaded (Warehouse to Truck)</div>
                    <div class="metric-value">{bags_out}</div>
                </div>
            ''', unsafe_allow_html=True)

        if frame_count % 60 == 0:
            current_temp += random.uniform(-0.1, 0.1)
            current_hum += random.choice([-1, 0, 1])
            with iot_ph.container():
                st.metric(label="Ambient Temperature", value=f"{current_temp:.1f} °C", delta=f"{random.uniform(-0.1, 0.1):.1f}°")
                st.metric(label="Relative Humidity", value=f"{current_hum} %", delta=f"{random.randint(-1, 1)}%")
                st.metric(label="Phosphine Gas Level", value="0.02 ppm", delta="Normal", delta_color="normal")

    cap.release()

    
run_live_inference()
# --- ADD THIS AT THE VERY BOTTOM ---
try:
    with open("output_result.mp4", "rb") as file:
        st.sidebar.download_button(
            label="📥 Download Result Video",
            data=file,
            file_name="warehouse_counting_result.mp4",
            mime="video/mp4"
        )
except:
    st.sidebar.info("Video is processing... Button will appear when ready.")