import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO('"D:/MicroModel/maskdetection/best.pt"')

# Open the webcam
cap = cv2.VideoCapture(0)  # '0' is the default camera; change if using an external camera.

while True:
    ret, frame = cap.read()  # Read the frame from the webcam

    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Visualize the results
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow('Face Mask Detection', annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
