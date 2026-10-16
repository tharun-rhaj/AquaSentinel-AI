import cv2

cap = cv2.VideoCapture("enter your camera url")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not detected")
        break

    cv2.imshow("LIVE DRONE FEED", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
