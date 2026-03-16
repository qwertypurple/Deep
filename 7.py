pip install ultralytics opencv-python

# YOLO Object Detection

from ultralytics import YOLO
import cv2

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

# Detect objects in image
results = model("test.jpg")

# Display detection results
for r in results:

    # Draw bounding boxes and labels
    img = r.plot()

    # Show image with detected objects
    cv2.imshow("Detection Result", img)

    # Wait until a key is pressed
    cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
