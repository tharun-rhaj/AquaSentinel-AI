import os
import random
import time
from datetime import datetime

import cv2
import streamlit as st
from ultralytics import YOLO

# ---------- CONFIG ----------
st.set_page_config(page_title="AquaSentinel AI", layout="wide")

# Replace the text below with your IP Webcam URL, for example:
#   http://<phone-ip>:8080/video
# Do NOT upload your real camera URL to GitHub.
IP_CAMERA_URL = "enter your camera url"

EVIDENCE_DIR = "evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

ZONE = (40, 40, 260, 260)  # x1, y1, x2, y2 on the 960x540 frame
SAVE_COOLDOWN = 5          # seconds between saved evidence images
GALLERY_SIZE = 9           # newest images shown in the gallery

# Simulated GPS (no real GPS receiver is used)
BASE_LAT = 11.3410
BASE_LON = 77.7172

last_save_time = 0
last_alert = None


def simulated_gps():
    lat = BASE_LAT + random.uniform(-0.0005, 0.0005)
    lon = BASE_LON + random.uniform(-0.0005, 0.0005)
    return round(lat, 6), round(lon, 6)


model = YOLO("yolov8n.pt")

# ---------- UI ----------
st.markdown("<h1>🌊 AquaSentinel AI - Maritime Surveillance Dashboard</h1>",
            unsafe_allow_html=True)

feed_box = st.empty()
alert_box = st.empty()
gallery_box = st.empty()

# ---------- CAMERA ----------
if IP_CAMERA_URL == "enter your camera url":
    st.error("Set IP_CAMERA_URL at the top of dashboard.py to your camera URL first.")
    st.stop()

cap = cv2.VideoCapture(IP_CAMERA_URL)
if not cap.isOpened():
    st.error("Camera not reachable. Check IP_CAMERA_URL.")
    st.stop()


def get_latest_frame(cap):
    # Skip buffered frames so the feed stays close to real time
    for _ in range(5):
        cap.grab()
    return cap.retrieve()


# ---------- LOOP ----------
while True:

    ret, frame = get_latest_frame(cap)
    if not ret:
        st.warning("Feed lost...")
        time.sleep(1)
        continue

    frame = cv2.resize(frame, (960, 540))

    # Draw restricted zone
    cv2.rectangle(frame, (ZONE[0], ZONE[1]), (ZONE[2], ZONE[3]), (0, 0, 255), 2)
    cv2.putText(frame, "RESTRICTED ZONE", (ZONE[0], ZONE[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    results = model(frame, conf=0.4, verbose=False)

    illegal = False
    person = False

    for r in results:
        for box, cls in zip(r.boxes.xyxy.cpu().numpy(),
                            r.boxes.cls.cpu().numpy()):

            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if (ZONE[0] < cx < ZONE[2] and ZONE[1] < cy < ZONE[3]
                    and label == "boat"):
                illegal = True

            if label == "person":
                person = True

    # ---------- ALERT ----------
    if illegal:
        last_alert = "🚨 Boat inside restricted zone"
    elif person:
        last_alert = "⚠️ Person detected"
    else:
        last_alert = None

    if last_alert:
        alert_box.error(last_alert)
    else:
        alert_box.success("System monitoring...")

    # ---------- SAVE EVIDENCE ----------
    now = time.time()
    if last_alert and now - last_save_time > SAVE_COOLDOWN:

        lat, lon = simulated_gps()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cv2.putText(frame, f"Time: {timestamp}", (10, 520),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, f"GPS (simulated): {lat},{lon}", (10, 540),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        filename = os.path.join(
            EVIDENCE_DIR,
            f"illegal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(filename, frame)

        last_save_time = now

    # ---------- SHOW FEED ----------
    feed_box.image(frame, channels="BGR", use_container_width=True)

    # ---------- GALLERY ----------
    with gallery_box.container():
        images = sorted(
            [f for f in os.listdir(EVIDENCE_DIR)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))],
            reverse=True)[:GALLERY_SIZE]
        cols = st.columns(3)
        for i, img in enumerate(images):
            try:
                cols[i % 3].image(os.path.join(EVIDENCE_DIR, img),
                                  caption=img,
                                  use_container_width=True)
            except Exception:
                pass

    time.sleep(0.03)
