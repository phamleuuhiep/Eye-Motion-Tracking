# import cv2

# # Load Haar cascade XML for face detection
# face_cascade = cv2.CascadeClassifier('D:/Documents/Study/DADN/haarcascade_frontalface_default.xml')

# # Load Haar cascade XML for eye detection
# eyes_cascade = cv2.CascadeClassifier('D:/Documents/Study/DADN/haarcascade_eye_tree_eyeglasses.xml')

# # Check if the face cascade XML loaded successfully
# if face_cascade.empty():
#     raise IOError('Face cascade XML file not found')

# # Check if the eye cascade XML loaded successfully
# if eyes_cascade.empty():
#     raise IOError('Eye cascade XML file not found')

# # Initialize video parameters
# # cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("D:/Documents/Study/DADN/face_eye_test.mp4")  # video file
# scale_factor = 1

# # Loop to read video frames
# while True:
#     _, frame = cap.read()
#     frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

#     # Convert frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces from grayscale frame
#     face_rects = face_cascade.detectMultiScale(gray, 1.1, 3)

#     # Loop to draw rectangles around detected faces
#     for (x, y, w, h) in face_rects:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255.0, 0), 2)

#         # Extract grayscale face region
#         face_gray = gray[y:y + h, x:x + w]

#         # Extract color face region
#         face_color = frame[y:y + h, x:x + w]

#         # Detect eyes inside the detected face region
#         eyes = eyes_cascade.detectMultiScale(face_gray)

#         # Draw circles around detected eyes
#         for (x_eye, y_eye, w_eye, h_eye) in eyes:
#             center = (int(x_eye + w_eye / 2), int(y_eye + h_eye / 2))
#             radius = int(0.3 * (w_eye + h_eye))
#             color = (255, 0, 0)
#             cv2.circle(face_color, center=center, radius=radius, color=color, thickness=2)

#     # Display result
#     cv2.imshow("Face detect", frame)

#     k = cv2.waitKey(10)
#     if k == 27:  # ESC key to exit
#         break

# cap.release()
# cv2.destroyAllWindows()






































import cv2
import numpy as np

# -------------------------------
# Haar Cascades
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    "D:/Documents/Study/DADN/haarcascade_frontalface_default.xml"
)
eyes_cascade = cv2.CascadeClassifier(
    "D:/Documents/Study/DADN/haarcascade_eye_tree_eyeglasses.xml"
)

if face_cascade.empty():
    raise IOError("Face cascade XML file not found")
if eyes_cascade.empty():
    raise IOError("Eye cascade XML file not found")

# -------------------------------
# Video Source (Webcam or File)
# -------------------------------
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("D:/Documents/Study/DADN/eye_face_test.mp4")

if not cap.isOpened():
    raise IOError("Cannot open video file")

# -------------------------------
# Helper for smoothing bounding boxes
# -------------------------------
def smooth_bbox(old, new, alpha=0.3):
    if old is None:
        return new
    return tuple(int(old[i] * (1 - alpha) + new[i] * alpha) for i in range(4))


prev_face = None  # for smoothing tracking

# -------------------------------
# Main Loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Normalize size for consistent tracking (video-friendly)
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve video contrast (CLAHE helps Haar detection)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # -------------------------------
    # Detect Face
    # -------------------------------
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) > 0:
        face_box = faces[0]
        # smoothing face bounding box
        face_box = smooth_bbox(prev_face, face_box)
        prev_face = face_box

        x, y, w, h = face_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        # -------------------------------
        # Detect Eyes inside face region
        # -------------------------------
        eyes = eyes_cascade.detectMultiScale(
            face_gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(20, 20)
        )

        for (ex, ey, ew, eh) in eyes:
            cx = int(ex + ew / 2)
            cy = int(ey + eh / 2)
            radius = int(0.3 * (ew + eh))

            cv2.circle(face_color, (cx, cy), radius, (255, 0, 0), 2)

    else:
        prev_face = None  # reset smoothing when detection lost

    # -------------------------------
    # Display
    # -------------------------------
    cv2.imshow("Face & Eye Detection", frame)

    # -------------------------------
    # Slow down playback based on video FPS
    # -------------------------------
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30
    delay = int(1000 / fps * 1.2)  

    if cv2.waitKey(delay) == 27:  
        break

cap.release()
cv2.destroyAllWindows()
