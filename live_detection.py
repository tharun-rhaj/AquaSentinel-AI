from ultralytics import YOLO
import cv2

# Load fastest model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("enter your camera url")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # VERY IMPORTANT -> resize for speed
    frame = cv2.resize(frame, (640, 480))

    # Detect boats + persons
    results = model(frame, classes=[0, 8], conf=0.4)

    annotated_frame = results[0].plot()

    cv2.imshow("AI DRONE SURVEILLANCE", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
