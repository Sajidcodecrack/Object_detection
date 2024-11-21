import cv2
from ultralytics import YOLO

# Load the trained model (use the correct local path to the model weights)
model = YOLO('D:/MicroModel/maskdetection/runs/detect/train/weights')  # Replace with your local path to the weights

# Capture video from the webcam
cap = cv2.VideoCapture(0)  # Use `0` for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Draw bounding boxes on the frame
    annotated_frame = results[0].plot()

    # If you don't want to display the image, just use this to keep the loop running
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
