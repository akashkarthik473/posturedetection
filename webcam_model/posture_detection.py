import cv2
from ultralytics import YOLO
from win10toast import ToastNotifier
import time

# 1) Load your trained custom YOLO model
model = YOLO("C:\\Users\\myste\\Documents\\projects\\posturedetection\\runs\\detect\\train10\\weights\\best.pt")

# 2) Optionally, define your custom class names if not stored in model
#    If your model has class names baked in, you can skip this
# classes = ["good_posture", "bad_posture"]  # Example

# 3) Start webcam
cap = cv2.VideoCapture(2)  # or whatever camera index (e.g., 2) you use
print(model.names)


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
cv2.namedWindow("Posture Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Posture Detection", window_width, window_height)



while True:
    ret, frame = cap.read()
    if not ret:
        break

    toast = ToastNotifier()
    #    You can pass 'conf=0.5' or other args to tweak detection
    results = model(frame, conf=0.5)

    # 5) Loop through the detected boxes
    #    YOLOv8 returns results[0] as the first (and only) batch
    detections = results[0].boxes  # Boxes object
    for box in detections:
        # box.xyxy:  [x1, y1, x2, y2]
        # box.conf:  confidence
        # box.cls:   class index
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls_id = int(box.cls[0])

        # If your model already stores class names, you can get them:
        label = f"{model.names[int(box.cls[0])]} ({conf:.2f})"

        if "Bad" in label:
            toast.show_toast(
            "BAD POSTURE",
            "Fix your posture",
            duration = 5,
            threaded=True
        )

        # Otherwise, define your own array "classes" and do:
        # label = classes[cls_id]  # if you manually set your classes

        # label = f"Class {cls_id} ({conf:.2f})"

        # 6) Draw bounding box and label on the frame
        color = (0, 255, 0)  # BGR
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            frame,
            label,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            color,
            5
        )


    # 7) Show the video frame with detections
    cv2.imshow("Posture Detection", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()
