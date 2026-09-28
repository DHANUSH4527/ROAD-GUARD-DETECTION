from ultralytics import YOLO
import cv2
import time
import winsound

# Load trained model
model = YOLO("runs/detect/train5/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Alert cooldown
last_alert_time = 0
alert_delay = 2   # seconds between alerts

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, device=0, imgsz=480, conf=0.6)

    annotated = results[0].plot()

    hazard_detected = False
    hazard_type = ""

    # Check detections
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        label = model.names[cls]

        if conf > 0.6:
            hazard_detected = True
            hazard_type = label

    # Alert logic
    current_time = time.time()

    if hazard_detected and (current_time - last_alert_time) > alert_delay:

        if hazard_type == "pothole":
            print("⚠ Pothole detected!")
            winsound.Beep(1000, 120)

        elif hazard_type == "speed_breaker":
            print("⚠ Speed breaker detected!")
            winsound.Beep(800, 120)

        last_alert_time = current_time

    # Visual warning
    if hazard_detected:
        cv2.putText(
            annotated,
            f"WARNING: {hazard_type.upper()}",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("Road Hazard Detection System", annotated)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()