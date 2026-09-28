from ultralytics import YOLO
import cv2
import winsound
import time
from twilio.rest import Client

# ------------------ TWILIO SETUP ------------------
ACCOUNT_SID = ""
AUTH_TOKEN = ""
TWILIO_NUMBER = ""
USER_NUMBER = ""

client = Client(ACCOUNT_SID, AUTH_TOKEN)

# ------------------ MODEL ------------------
model = YOLO("runs/detect/train5/weights/best.pt")

# ------------------ CAMERA ------------------
url = "http://10.251.37.18:8080/video"
cap = cv2.VideoCapture(url)

frame_skip = 2
frame_count = 0

# ------------------ DETECTION CONTROL ------------------
CONFIRM_FRAMES = 3
detect_counter = 0

# ------------------ ALERT CONTROL ------------------
last_msg_time = 0
MSG_COOLDOWN = 30  # seconds

# ------------------ LABELS ------------------
labels = {
    0: "POTHOLE",
    1: "SPEED BREAKER"
}

print("🚀 Detection Started... Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera not receiving frames")
        break

    frame_count += 1

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)

    # Skip frames
    if frame_count % frame_skip != 0:
        cv2.imshow("Phone Camera Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # ------------------ AI INFERENCE ------------------
    results = model(frame, imgsz=480, conf=0.30, iou=0.5, verbose=False)

    detected_class = None

    if results[0].boxes is not None and len(results[0].boxes) > 0:

        boxes = results[0].boxes

        # Extract class IDs and confidence scores
        class_ids = boxes.cls.tolist()
        confidences = boxes.conf.tolist()

        # Pick highest confidence detection
        max_index = confidences.index(max(confidences))
        detected_class = int(class_ids[max_index])

        print("Detected classes:", class_ids)
        print("Selected class:", detected_class)

        detect_counter += 1
    else:
        detect_counter = 0

    # ------------------ CONFIRM DETECTION ------------------
    if detect_counter >= CONFIRM_FRAMES:

        winsound.Beep(1200, 200)

        current_time = time.time()

        if current_time - last_msg_time > MSG_COOLDOWN:

            if detected_class == 0:
                msg = "⚠️ POTHOLE WARNING: Pothole detected ahead. Please slow down."
            elif detected_class == 1:
                msg = "⚠️ SPEED BREAKER WARNING: Speed breaker detected ahead. Reduce speed."
            else:
                msg = "⚠️ ROAD WARNING: Obstacle detected ahead."

            try:
                message = client.messages.create(
                    body=msg,
                    from_=TWILIO_NUMBER,
                    to=USER_NUMBER
                )
                print("📩 SMS sent:", message.sid)

            except Exception as e:
                print("❌ SMS Error:", e)

            last_msg_time = current_time

        detect_counter = 0

    # ------------------ DISPLAY ------------------
    annotated_frame = results[0].plot(labels=True, conf=True)
    cv2.imshow("Phone Camera – AI Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ------------------ CLEANUP ------------------
cap.release()
cv2.destroyAllWindows()
