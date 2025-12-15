# import cv2
# import numpy as np

# # -------------------------------
# # Haar Cascades
# # -------------------------------
# face_cascade = cv2.CascadeClassifier(
#     "D:/Documents/Study/DADN/haarcascade_frontalface_default.xml"
# )
# eyes_cascade = cv2.CascadeClassifier(
#     "D:/Documents/Study/DADN/haarcascade_eye_tree_eyeglasses.xml"
# )

# if face_cascade.empty():
#     raise IOError("Face cascade XML file not found")
# if eyes_cascade.empty():
#     raise IOError("Eye cascade XML file not found")

# # -------------------------------
# # Video Source (Webcam or File)
# # -------------------------------
# # cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("D:/Documents/Study/DADN/eye_face_test.mp4")

# if not cap.isOpened():
#     raise IOError("Cannot open video file")

# # -------------------------------
# # Helper for smoothing bounding boxes
# # -------------------------------
# def smooth_bbox(old, new, alpha=0.3):
#     if old is None:
#         return new
#     return tuple(int(old[i] * (1 - alpha) + new[i] * alpha) for i in range(4))


# prev_face = None  # for smoothing tracking

# # -------------------------------
# # Main Loop
# # -------------------------------
# eye_centers = []
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Normalize size for consistent tracking (video-friendly)
#     frame = cv2.resize(frame, (640, 480))
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Improve video contrast (CLAHE helps Haar detection)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     gray = clahe.apply(gray)

#     # -------------------------------
#     # Detect Face
#     # -------------------------------
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(80, 80),
#     )

#     if len(faces) > 0:
#         face_box = faces[0]
#         # smoothing face bounding box
#         face_box = smooth_bbox(prev_face, face_box)
#         prev_face = face_box

#         x, y, w, h = face_box
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         face_gray = gray[y:y + h, x:x + w]
#         face_color = frame[y:y + h, x:x + w]

#         # -------------------------------
#         # Detect Eyes inside face region
#         # -------------------------------
#         eyes = eyes_cascade.detectMultiScale(
#             face_gray, 
#             scaleFactor=1.1, 
#             minNeighbors=5, 
#             minSize=(20, 20)
#         )

#         for (ex, ey, ew, eh) in eyes:
#             cx = int(ex + ew / 2)
#             cy = int(ey + eh / 2)
#             radius = int(0.3 * (ew + eh))

#             cv2.circle(face_color, (cx, cy), radius, (255, 0, 0), 2)
#             eye_centers.append((x + cx, y + cy))

#     else:
#         prev_face = None  # reset smoothing when detection lost

#     # -------------------------------
#     # Display
#     # -------------------------------
#     cv2.imshow("Face & Eye Detection", frame)

#     # -------------------------------
#     # Slow down playback based on video FPS
#     # -------------------------------
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if fps <= 0 or fps > 120:
#         fps = 30
#     delay = int(1000 / fps * 1.2)  

#     if cv2.waitKey(delay) == 27:  
#         break

# cap.release()


# import math

# jitters = []
# for i in range(1, len(eye_centers)):
#     x1, y1 = eye_centers[i-1]
#     x2, y2 = eye_centers[i]
#     d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#     jitters.append(d)

# if len(jitters) > 0:
#     print("Haar-only Jitter:", sum(jitters)/len(jitters))



# cv2.destroyAllWindows()












import cv2
import numpy as np
import math

# -------------------------------
# Haar Cascades
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eyes_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)

if face_cascade.empty():
    raise IOError("Face cascade XML file not found")
if eyes_cascade.empty():
    raise IOError("Eye cascade XML file not found")

# -------------------------------
# Video Source
# -------------------------------
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("D:/Documents/Study/DADN/eye_face_test.mp4")

if not cap.isOpened():
    raise IOError("Cannot open video file")

# -------------------------------
# Helper: Smooth face bounding box
# -------------------------------
def smooth_bbox(old, new, alpha=0.3):
    if old is None:
        return new
    return tuple(int(old[i] * (1 - alpha) + new[i] * alpha) for i in range(4))

prev_face = None

# -------------------------------
# Storage for evaluation
# -------------------------------
left_eye = []     # (frame_id, (x, y))
right_eye = []    # (frame_id, (x, y))

total_frames = 0
valid_eye_frames = 0

# -------------------------------
# Main Loop
# -------------------------------
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    total_frames += 1

    # Normalize frame size
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # -------------------------------
    # Face Detection
    # -------------------------------
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) > 0:
        face = smooth_bbox(prev_face, faces[0])
        prev_face = face

        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_gray = gray[y:y + h, x:x + w]

        # -------------------------------
        # Eye Detection (Two Eyes)
        # -------------------------------
        eyes = eyes_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )

        if len(eyes) >= 2:
            # Sort by x position (left → right)
            eyes = sorted(eyes, key=lambda e: e[0])
            (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes[:2]

            # Left eye (global coordinates)
            lx = int(x + ex1 + ew1 / 2)
            ly = int(y + ey1 + eh1 / 2)

            # Right eye (global coordinates)
            rx = int(x + ex2 + ew2 / 2)
            ry = int(y + ey2 + eh2 / 2)

            left_eye.append((frame_id, (lx, ly)))
            right_eye.append((frame_id, (rx, ry)))

            valid_eye_frames += 1

            # Visualization
            cv2.circle(frame, (lx, ly), 6, (255, 0, 0), 2)
            cv2.circle(frame, (rx, ry), 6, (0, 0, 255), 2)

    else:
        prev_face = None

    # -------------------------------
    # Display
    # -------------------------------
    cv2.imshow("Haar-only Two-Eye Detection", frame)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30
    delay = int(1000 / fps * 1.2)

    if cv2.waitKey(delay) == 27:
        break

cap.release()
cv2.destroyAllWindows()

# -------------------------------
# JITTER COMPUTATION
# -------------------------------
def compute_jitter(track):
    values = []
    for i in range(1, len(track)):
        f1, p1 = track[i - 1]
        f2, p2 = track[i]
        if f2 - f1 == 1:  # only consecutive frames
            d = math.dist(p1, p2)
            values.append(d)
    return np.mean(values) if values else None


left_jitter = compute_jitter(left_eye)
right_jitter = compute_jitter(right_eye)

if left_jitter is not None:
    print("Left Eye Jitter (px):", left_jitter)
if right_jitter is not None:
    print("Right Eye Jitter (px):", right_jitter)

if left_jitter is not None and right_jitter is not None:
    avg_jitter = (left_jitter + right_jitter) / 2
    print("Average Eye Jitter (px):", avg_jitter)

# -------------------------------
# STABILITY COMPUTATION (Haar-only)
# -------------------------------
if total_frames > 0:
    haar_stability = valid_eye_frames / total_frames
    print("Haar-only Tracking Stability:", haar_stability)
# -------------------------------


