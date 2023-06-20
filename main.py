import cv2
import time

# Initialize the tracker
tracker = cv2.TrackerKCF_create()

# Read the video
video = cv2.VideoCapture("test.mp4")

# Read the first frame
ret, frame = video.read()

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
