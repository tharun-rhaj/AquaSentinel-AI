import cv2
import random
from ultralytics import YOLO
from datetime import datetime

# ----------------------------
# LOAD MODEL
# ----------------------------
model = YOLO("yolov8n.pt")  
# (nano model = faster for live demo)

# ----------------------------
# MOBILE CAMERA STREAM
# Replace with your IP Webcam URL
# ----------------------------
STREAM_URL = "http://10.17.158.57:8080/video"

cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("❌ Cannot connect to camera")
    exit()

print("✅ Drone camera connected!")

# ----------------------------
# SIMULATED GPS
# (Stable for hackathon demo)
# ----------------------------
BASE_LAT = 11.3410
BASE_LON = 77.7172

def get_simulated_gps():
    lat = BASE_LAT + random.uniform(-0.0005, 0.0005)
    lon = BASE_LON + random.uniform(-0.0005, 0.0005)
    return round(lat, 6), round(lon, 6)

# ----------------------------
# ILLEGAL ZONE (Rectangle)
# You can adjust these numbers
# ----------------------------
ZONE_TOP_LEFT = (100, 80)
ZONE_BOTTOM_RIGHT = (900, 600)

# ----------------------------
# ALERT COOLDOWN
# Prevents alert spam
# ----------------------------
last_alert_time = 0
ALERT_COOLDOWN = 10  # seconds


# ============================
# MAIN LOOP
# ============================
while True:
    ret, frame = cap.read()

    if not ret:
        print("⚠️ Frame not received")
        break

    # Resize for performance
    frame = cv2.resize(frame, (1280, 720))

    # Draw illegal zone
    cv2.rectangle(frame, ZONE_TOP_LEFT, ZONE_BOTTOM_RIGHT, (0,0,255), 3)
    cv2.putText(frame, "RESTRICTED ZONE",
                (ZONE_TOP_LEFT[0], ZONE_TOP_LEFT[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,0,255), 3)

    # Run detection
    results = model(frame, conf=0.45)

    current_time = datetime.now().timestamp()

    for r in results:
        boxes = r.boxes

        for box in boxes:

            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # YOLO sometimes detects boats as:
            # boat / ship
            if label not in ["boat", "ship"]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Draw detection
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, "Boat Detected",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2)

            # Check illegal zone
            if (ZONE_TOP_LEFT[0] < center_x < ZONE_BOTTOM_RIGHT[0] and
                ZONE_TOP_LEFT[1] < center_y < ZONE_BOTTOM_RIGHT[1]):

                cv2.putText(frame,
                            "!!! ILLEGAL FISHING DETECTED !!!",
                            (250,40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0,0,255),
                            4)

                # Alert cooldown
                if current_time - last_alert_time > ALERT_COOLDOWN:

                    lat, lon = get_simulated_gps()

                    print("\n🚨 ALERT: Illegal Boat Detected!")
                    print(f"📍 Location: {lat}, {lon}")

                    # Save evidence screenshot
                    filename = f"illegal_boat_{datetime.now().strftime('%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame)

                    print(f"📸 Evidence saved: {filename}")

                    last_alert_time = current_time

    cv2.imshow("AI Maritime Surveillance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
