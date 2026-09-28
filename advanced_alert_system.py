from ultralytics import YOLO
import cv2
import time
import winsound
import pyttsx3
import geocoder

# Load trained model
model = YOLO("runs/detect/train5/weights/best.pt")

# Initialize voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 170)

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Alert cooldown
last_alert_time = 0
alert_delay = 2

# FPS tracking
prev_time = 0

def get_location():
    try:
        g = geocoder.ip('me')
        return g.latlng
    except:
        return None


while True:

    ret, frame = cap.read()
    if not ret:
        print("Camera frame not received")
        break

    # Run detection using GPU
    results = model(frame, device=0, imgsz=480, conf=0.6)

    annotated = results[0].plot()

    hazard_detected = False
    hazard_type = ""
    distance_level = ""

    for box in results[0].boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]

        x1, y1, x2, y2 = box.xyxy[0]

        box_area = (x2-x1) * (y2-y1)

        if conf > 0.6:

            hazard_detected = True
            hazard_type = label

            # Distance estimation based on bounding box size
            if box_area > 90000:
                distance_level = "VERY CLOSE"
            elif box_area > 40000:
                distance_level = "CLOSE"
            else:
                distance_level = "FAR"

    current_time = time.time()

    # Dynamic alert system
    if hazard_detected and (current_time - last_alert_time) > alert_delay:

        location = get_location()

        message = f"{hazard_type} ahead {distance_level}"

        print("⚠ Hazard detected:", message)

        if location:
            print("Location:", location)

        # Dynamic beep alert
        if distance_level == "FAR":
            winsound.Beep(700, 120)

        elif distance_level == "CLOSE":
            winsound.Beep(900, 200)

        elif distance_level == "VERY CLOSE":
            winsound.Beep(1200, 400)

        # Voice alert
        engine.say(message)
        engine.runAndWait()

        last_alert_time = current_time

    # Visual color warning
    if hazard_detected:

        color = (0,255,0)

        if distance_level == "FAR":
            color = (0,255,255)

        elif distance_level == "CLOSE":
            color = (0,165,255)

        elif distance_level == "VERY CLOSE":
            color = (0,0,255)

        cv2.putText(
            annotated,
            f"{hazard_type.upper()} - {distance_level}",
            (40,60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3
        )

    # FPS display
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(
        annotated,
        f"FPS: {int(fps)}",
        (20,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.imshow("AI Road Hazard Detection & Alert System", annotated)

    # Exit key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()