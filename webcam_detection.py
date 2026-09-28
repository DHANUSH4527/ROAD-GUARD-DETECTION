from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/train5/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Run detection
    results = model(frame, device=0, imgsz=480, conf=0.6)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Show result
    cv2.imshow("Pothole & Speedbreaker Detection", annotated_frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()