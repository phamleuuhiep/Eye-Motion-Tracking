import cv2
import numpy as np

# -------------------------------
# Load Haar Cascade
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    "D:/Documents/Study/DADN/haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise IOError("Face cascade XML file not found!")

# -------------------------------
# Video Source
# -------------------------------
# cap = cv2.VideoCapture(0)  # webcam
cap = cv2.VideoCapture("D:/Documents/Study/DADN/eye_face_test.mp4")  # video

if not cap.isOpened():
    raise IOError("Cannot open video file")

# -------------------------------
# Smooth Bounding Box Helper
# -------------------------------
def smooth_bbox(old, new, alpha=0.25):
    """Smooths bounding box transitions to avoid jitter."""
    if old is None:
        return new
    return tuple(int(old[i] * (1 - alpha) + new[i] * alpha) for i in range(4))


prev_face = None  # previous smoothed face bbox

# -------------------------------
# Main Loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Stop when video ends

    # Resize for consistent processing
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve contrast (important for video)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # -------------------------------
    # Detect Faces
    # -------------------------------
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)  # reduce false positives
    )

    if len(faces) > 0:
        face_box = faces[0]
        face_box = smooth_bbox(prev_face, face_box)
        prev_face = face_box

        x, y, w, h = face_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    else:
        prev_face = None  # reset if detection lost

    # -------------------------------
    # Display Frame
    # -------------------------------
    cv2.imshow("Face Detection", frame)

    # -------------------------------
    # FPS Correction for Video Playback
    # -------------------------------
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30  # fallback

    delay = int((1000 / fps) * 1.2)  # slightly slower for stability

    if cv2.waitKey(delay) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()










