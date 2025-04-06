import cv2

# Load pre-trained DNN model (Caffe)
net = cv2.dnn.readNetFromCaffe(
    "D:\\projects\\postureproject\\webcam_model\\deploy.prototxt",
    "D:\\projects\\postureproject\\webcam_model\\res10_300x300_ssd_iter_140000.caffemodel"
)

# Initialize video capture
cap = cv2.VideoCapture(2)

# Get the resolution of the DroidCam feed
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set a maximum height for the window
max_window_height = 720  # Change this to your preferred height

# Calculate the scaling factor to maintain aspect ratio
scaling_factor = max_window_height / original_height
window_width = int(original_width * scaling_factor)
window_height = int(original_height * scaling_factor)

# Create a named window with a fixed size
cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Detection", window_width, window_height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare input for the DNN model
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Loop through detections and draw boxes
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Adjust threshold as needed
            box = detections[0, 0, i, 3:7] * \
                  [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the feed
    cv2.imshow("Face Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(10) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
