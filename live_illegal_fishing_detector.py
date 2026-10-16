import os
import random
from datetime import datetime

import cv2
from ultralytics import YOLO

# ----------------------------
# CAMERA STREAM
# Replace the text below with your IP Webcam URL, for example:
#   http://<phone-ip>:8080/video
# Do NOT upload your real camera URL to GitHub.
# ----------------------------
STREAM_URL = "enter your camera url"

if STREAM_URL == "enter your camera url":
    print("Set STREAM_URL at the top of this file to your camera URL first.")
    raise SystemExit(1)

EVIDENCE_DIR = "evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = YOLO("yolov8n.pt")  # nano model = fast enough for live demo

cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print("Cannot connect to camera. Check STREAM_URL.")
    raise SystemExit(1)

print("Camera connected. Press 'q' to quit.")

# ----------------------------
# SIMULATED GPS (no real GPS receiver is used)
# ----------------------------
BASE_LAT = 11.3410
BASE_LON = 77.7172


def get_simulated_gps():
    lat = BASE_LAT + random.uniform(-0.0005, 0.0005)
    lon = BASE_LON + random.uniform(-0.0005, 0.0005)
    return round(lat, 6), round(lon, 6)


# ----------------------------
# RESTRICTED ZONE (rectangle, pixel coordinates on the 1280x720 frame)
# ----------------------------
ZONE_TOP_LEFT = (100, 80)
ZONE_BOTTOM_RIGHT = (900, 600)

# ----------------------------
# ALERT COOLDOWN (prevents alert spam)
# ----------------------------
last_alert_time = 0
ALERT_COOLDOWN = 10  # seconds

# ============================
# MAIN LOOP
# ============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        break

    frame = cv2.resize(frame, (1280, 720))

    # Draw restricted zone
    cv2.rectangle(frame, ZONE_TOP_LEFT, ZONE_BOTTOM_RIGHT, (0, 0, 255), 3)
    cv2.putText(frame, "RESTRICTED ZONE",
                (ZONE_TOP_LEFT[0], ZONE_TOP_LEFT[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    results = model(frame, conf=0.45, verbose=False)
    current_time = datetime.now().timestamp()

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            if label not in ("boat", "ship"):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Boat Detected", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            inside_zone = (ZONE_TOP_LEFT[0] < center_x < ZONE_BOTTOM_RIGHT[0] and
                           ZONE_TOP_LEFT[1] < center_y < ZONE_BOTTOM_RIGHT[1])

            if inside_zone:
                cv2.putText(frame, "!!! ILLEGAL FISHING DETECTED !!!",
                            (250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 0, 255), 4)

                if current_time - last_alert_time > ALERT_COOLDOWN:
                    lat, lon = get_simulated_gps()
                    print("\nALERT: Boat inside restricted zone")
                    print(f"Location (simulated): {lat}, {lon}")

                    filename = os.path.join(
                        EVIDENCE_DIR,
                        f"illegal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"Evidence saved: {filename}")

                    last_alert_time = current_time

    cv2.imshow("AquaSentinel AI - Maritime Surveillance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
