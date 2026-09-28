# Road Hazard Detection System

This project was developed to detect road hazards such as potholes and speed breakers using computer vision and deep learning. The goal is to help improve road safety by identifying hazards in real time through a camera feed and alerting the user when a hazard is detected.

The system uses a YOLO-based object detection model trained on a custom dataset of potholes and speed breakers. It supports webcam input and mobile camera input and can generate alerts when hazards are detected.

## Features

* Detects potholes and speed breakers in real time
* Works with webcam and mobile camera feeds
* Audio alert system for detected hazards
* Custom-trained YOLO model
* Snapshot capture of detected hazards
* Flutter mobile application for user interaction

## Technologies Used

* Python
* OpenCV
* YOLOv8
* Flutter
* Dart

## Project Structure

* `webcam_detection.py` – Detection using webcam
* `mobile_camera_detection.py` – Detection using mobile camera
* `advanced_alert_system.py` – Hazard alert system
* `alert_detection.py` – Detection with alerts
* `train.py` – Model training script
* `robo_dataset/` – Training dataset
* `road_hazard_ai_app/` – Flutter application

## How to Run

1. Clone the repository.
2. Install the required Python packages.
3. Run any of the detection scripts depending on the input source.

Example:

```bash
python webcam_detection.py
```

or

```bash
python mobile_camera_detection.py
```

## Purpose

Road hazards such as potholes and speed breakers are a common cause of accidents and vehicle damage. This project was created as an attempt to use computer vision techniques to automatically identify such hazards and provide timely alerts to road users.

## Future Work

* Improve detection accuracy
* Add GPS location tracking
* Store detected hazards in a database
* Generate road condition reports
* Deploy the system as a complete mobile application

## Author

SINGARAJU DHANUSH KESAVA VARMA
