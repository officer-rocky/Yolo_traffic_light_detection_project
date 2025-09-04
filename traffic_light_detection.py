import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------
# Load YOLOv8 model (pretrained)
# ----------------------------
model = YOLO("yolov8n.pt")  # small model, fast for testing

# ----------------------------
# Define HSV ranges for Red, Yellow, Green
# ----------------------------
red_lower1 = np.array([0, 120, 70])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 120, 70])
red_upper2 = np.array([180, 255, 255])

yellow_lower = np.array([15, 100, 100])
yellow_upper = np.array([40, 255, 255])


green_lower = np.array([40, 70, 70])
green_upper = np.array([90, 255, 255])

# ----------------------------
# Video capture
# ----------------------------
cap = cv2.VideoCapture("traffic2.mp4")  # replace with your video file path

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# ----------------------------
# Main loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break  # Stop loop when video ends

    # Run YOLO on frame
    results = model(frame)

    # Loop through detected boxes
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # bounding boxes
        class_ids = result.boxes.cls.cpu().numpy()  # class IDs

        for i, box in enumerate(boxes):
            class_id = int(class_ids[i])
            if class_id != 9:  # Keep only traffic lights
                continue

            x1, y1, x2, y2 = map(int, box)

            # Filter small boxes / weird shapes
            w = x2 - x1
            h = y2 - y1
            aspect_ratio = h / float(w)
            if w < 10 or h < 10 or not (1.0 < aspect_ratio < 4.0):
                continue

            # Crop ROI and convert to HSV
            roi = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # HSV masks
            red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            green_mask = cv2.inRange(hsv, green_lower, green_upper)

            # Decide color
            red_count = cv2.countNonZero(red_mask)
            yellow_count = cv2.countNonZero(yellow_mask)
            green_count = cv2.countNonZero(green_mask)
            max_count = max(red_count, yellow_count, green_count)

            if max_count < 50:  # ignore false positives with no color
                continue

            if red_count == max_count:
                color_label = "RED"
            elif yellow_count == max_count:
                color_label = "YELLOW"
            else:
                color_label = "GREEN"

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, color_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Traffic Light Detection", frame)

    # Exit if ESC or 'q' pressed
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        print("Stopped by user.")
        break

# ----------------------------
# Release resources
# ----------------------------
cap.release()
cv2.destroyAllWindows()
