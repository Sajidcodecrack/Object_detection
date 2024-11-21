import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO('D:/Machine learning/mask_cup/best.pt')  # Path to the trained model weights

# Start video capture
cap = cv2.VideoCapture(0)  # 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Draw bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('objects detection: ', annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
