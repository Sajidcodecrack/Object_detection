import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO('D:/MicroModel/Goggles_dataset/best.pt')  # Path to the trained YOLOv8 model weights

# Start video capture (0 for the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Perform detection
    results = model.predict(frame, save=False, show=False)

    # Draw bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Display the frame with annotations
    cv2.imshow('Goggles Detection', annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
