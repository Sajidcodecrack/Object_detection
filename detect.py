import cv2
from ultralytics import YOLO
import pyttsx3
import threading

# Initialize text-to-speech engine
engine = pyttsx3.init()


# Function to speak detected object names (run in a separate thread)
def speak(text):
    engine.say(text)
    engine.runAndWait()


# Load the trained model
model = YOLO("D:/MicroModel/Nueromorphic/best.pt")  # Path to the trained model weights

# Start video capture
cap = cv2.VideoCapture(0)  # 0 for default camera

# To store previous detection
last_detected_objects = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Set to store detected objects in the current frame
    current_detected_objects = set()

    # Get names of detected objects
    for result in results:
        for obj in result.boxes:
            class_name = model.names[int(obj.cls)]  # Get object class name
            current_detected_objects.add(class_name)  # Add to the current set

    # Check for new detections by comparing with previous frame detections
    new_detections = current_detected_objects - last_detected_objects
    if new_detections:
        # Create a thread for speaking the new detections
        detection_text = ", ".join(new_detections)
        t = threading.Thread(target=speak, args=(f"Detected  {detection_text}",))
        t.start()

    # Update the last detected objects for comparison in the next frame
    last_detected_objects = current_detected_objects

    # Draw bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("Detect ", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


# import cv2
# from ultralytics import YOLO

# # Load the trained model
# model = YOLO("D:/MicroModel/Nueromorphic/best.pt")  # Path to the trained model weights

# # Start video capture
# cap = cv2.VideoCapture(0)  # 0 for default camera

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Perform detection
#     results = model(frame)

#     # Draw bounding boxes on the frame
#     annotated_frame = results[0].plot()

#     # Display the frame
#     cv2.imshow('Detect ', annotated_frame)

#     # Exit on 'q' key
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAl0lWindows()
