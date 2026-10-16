import streamlit as st
import cv2
import os
import time
import random
from ultralytics import YOLO
from datetime import datetime

# ---------- CONFIG ----------
st.set_page_config(page_title="AI Marintime Surveillance", layout="wide")

IP_CAMERA_URL = "http://10.17.158.57:8080/video"
EVIDENCE_DIR = "evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

ZONE = (40,40,260,260)
SAVE_COOLDOWN = 5

last_save_time = 0
last_alert = None

def random_gps():
    lat = round(random.uniform(8.0,13.0),6)
    lon = round(random.uniform(75.0,80.0),6)
    return lat,lon

model = YOLO("yolov8n.pt")

# ---------- UI ----------
st.markdown("<h1>🌊 AI Marintime Surveillance Dashboard</h1>", unsafe_allow_html=True)

feed_box = st.empty()
alert_box = st.empty()
gallery_box = st.empty()

# ---------- CAMERA ----------
cap = cv2.VideoCapture(IP_CAMERA_URL)
if not cap.isOpened():
    st.error("Camera not reachable")
    st.stop()

# ---------- FUNCTION TO DROP OLD FRAMES ----------
def get_latest_frame(cap):
    # drop buffered frames
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

    frame = cv2.resize(frame,(960,540))

    # draw restricted zone
    cv2.rectangle(frame,(ZONE[0],ZONE[1]),(ZONE[2],ZONE[3]),(0,0,255),2)
    cv2.putText(frame,"RESTRICTED ZONE",(ZONE[0],ZONE[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    results = model(frame, conf=0.4)

    illegal=False
    person=False

    for r in results:
        for box,cls in zip(r.boxes.xyxy.cpu().numpy(),
                           r.boxes.cls.cpu().numpy()):

            x1,y1,x2,y2 = map(int,box)
            label=model.names[int(cls)]
            cx,cy=(x1+x2)//2,(y1+y2)//2

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,label,(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

            if ZONE[0]<cx<ZONE[2] and ZONE[1]<cy<ZONE[3] and label=="boat":
                illegal=True

            if label=="person":
                person=True

    # ---------- ALERT ----------
    if illegal:
        last_alert="🚨 Illegal Boat Detected"
    elif person:
        last_alert="⚠️ Person in water detected"
    else:
        last_alert=None

    if last_alert:
        alert_box.error(last_alert)
    else:
        alert_box.success("System monitoring...")

    # ---------- SAVE ----------
    now=time.time()
    if last_alert and now-last_save_time > SAVE_COOLDOWN:

        lat,lon=random_gps()
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cv2.putText(frame,f"Time: {timestamp}",(10,520),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
        cv2.putText(frame,f"GPS: {lat},{lon}",(10,540),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)

        filename=f"{EVIDENCE_DIR}/illegal_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(filename,frame)

        last_save_time=now

    # ---------- SHOW FEED ----------
    feed_box.image(frame, channels="BGR", use_container_width=True)

    # ---------- GALLERY ----------
    with gallery_box.container():
        images = sorted(os.listdir(EVIDENCE_DIR), reverse=True)
        cols = st.columns(3)
        for i,img in enumerate(images):
            try:
                cols[i%3].image(os.path.join(EVIDENCE_DIR,img),
                                caption=img,
                                use_container_width=True)
            except:
                pass

    time.sleep(0.03)
