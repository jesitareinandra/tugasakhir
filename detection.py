import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('D:\\jesi\\runs\\detect\\train\\weights\\best.pt')  # Ganti dengan model yang telah dilatih

# Define class names for traffic signs (example names, replace with actual ones as needed)
class_names = [
    'bus_stop', 'do_not_enter', 'do_not_turn_l', 'do_not_turn_r', 'do_not_u_turn', 'don-t_stop', 'enter_left_lane', 'left_right_lane', 'no_parking', 'parking', 'ped_crossing', 'railway', 'stop', 't_intersection_l', 'traffic_light', 'u_turn', 'warning'
]

# Define a function to process the frame
def process_frame(frame):
    # Perform inference
    results = model(frame)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()

    # Extract predictions
    predictions = results[0].boxes
    for box in predictions:
        cls_id = int(box.cls.cpu().numpy())  # Extract class id as a Python int
        label = class_names[cls_id] if cls_id < len(class_names) else f'Class {cls_id}'
        conf = float(box.conf.cpu().numpy())  # Extract confidence score as a Python float
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Extract bounding box coordinates as a list of floats
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert coordinates to Python ints
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return annotated_frame

# Open a video capture object (0 for webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Process the frame
        annotated_frame = process_frame(frame)

        # Display the resulting frame
        cv2.imshow('Real-time Traffic Sign Detection', annotated_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
