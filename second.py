import cv2
import numpy as np
import time

# Load YOLOv4 model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load COCO classes
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the tracker
tracker = cv2.TrackerCSRT_create()

# Read the video
video = cv2.VideoCapture("test.mp4")

# Function to get the output from YOLOv4
def get_objects(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids

# Read the first frame
ret, frame = video.read()

# Get objects in the first frame
boxes, confidences, class_ids = get_objects(frame)

# Select the object to track
bbox = cv2.selectROI("Tracking", frame, False)
tracker.init(frame, bbox)

while True:
    # Read a new frame
    ret, frame = video.read()
    if not ret:
        break

    start_time = time.time()
    # Update the tracker
    success, bbox = tracker.update(frame)
    end_time = time.time()
    processing_time = end_time - start_time
    print("Processing time: {:.2f} ms".format(processing_time * 1000))

    if success:
        # Draw the bounding box around the tracked object
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        # If tracking failed, display a message
        cv2.putText(frame, "Lost", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
