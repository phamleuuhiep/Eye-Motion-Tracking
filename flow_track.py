# import cv2
# import numpy as np

# # -------------------------------
# # Load Haar cascades
# # -------------------------------
# face_cascade = cv2.CascadeClassifier('D:/Documents/Study/DADN/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('D:/Documents/Study/DADN/haarcascade_eye_tree_eyeglasses.xml')

# if face_cascade.empty():
#     raise IOError("Face cascade not found.")
# if eye_cascade.empty():
#     raise IOError("Eye cascade not found.")

# # -------------------------------
# # Optical Flow Parameters
# # -------------------------------
# lk_params = dict(
#     winSize=(15, 15),
#     maxLevel=2,
#     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
# )

# feature_params = dict(
#     maxCorners=80,
#     qualityLevel=0.3,
#     minDistance=5,
#     blockSize=3
# )



# def estimate_direction(flow_vectors):
#     if len(flow_vectors) == 0:
#         return "Unknown"

#     flow_vectors = np.array(flow_vectors)

#     # Remove noise (very small movements)
#     mag = np.linalg.norm(flow_vectors, axis=1)
#     flow_vectors = flow_vectors[mag > 0.03]  # threshold noise

#     if len(flow_vectors) == 0:
#         return "Center"

#     # Use median instead of mean
#     dx = np.median(flow_vectors[:, 0])
#     dy = np.median(flow_vectors[:, 1])

#     # Thresholds adapted for eye movement
#     threshold_x = 0.20
#     threshold_y = 0.20

#     if dx > threshold_x:
#         return "Right"
#     elif dx < -threshold_x:
#         return "Left"

#     if dy > threshold_y:
#         return "Down"
#     elif dy < -threshold_y:
#         return "Up"

#     return "Center"

# # -------------------------------
# # Main Program
# # -------------------------------
# def main():
#     # cap = cv2.VideoCapture(0) ## Use 0 for default camera
#     cap = cv2.VideoCapture("D:/Documents/Study/DADN/gaze_test.mp4")

#     prev_gray = None
#     prev_points = None
#     flow_vectors = []
#     last_direction = None

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Step 1: Detect face
#         faces = face_cascade.detectMultiScale(gray, 1.2, 4)

#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

#             roi_gray = gray[y:y+h, x:x+w]
#             roi_color = frame[y:y+h, x:x+w]

#             # Step 2: Detect eyes
#             eyes = eye_cascade.detectMultiScale(roi_gray)

#             for (ex, ey, ew, eh) in eyes:
#                 cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255,0,0), 2)

#                 eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
#                 eye_global_pos = (x+ex, y+ey)

#                 # Step 3: Initialize good features to track (one time or re-init)
#                 if prev_points is None:
#                     points = cv2.goodFeaturesToTrack(
#                         eye_gray,
#                         maxCorners=50,
#                         qualityLevel=0.3,
#                         minDistance=3,
#                         blockSize=3
#                     )

#                     if points is not None:
#                         # Convert to global coordinates
#                         prev_points = points + np.array([[eye_global_pos[0], eye_global_pos[1]]])

#                         prev_gray = gray.copy()
#                         continue

#                 # Step 4: Optical Flow Tracking
#                 if prev_points is not None and prev_gray is not None:

#                     # Force float32 format
#                     prev_points = prev_points.astype(np.float32)

#                     if len(prev_points) == 0:
#                         prev_points = None
#                         prev_gray = gray.copy()
#                         continue

#                     # Calculate optical flow
#                     new_points, status, _ = cv2.calcOpticalFlowPyrLK(
#                         prev_gray, gray, prev_points, None, **lk_params
#                     )

#                     if new_points is None or status is None:
#                         prev_points = None
#                         prev_gray = gray.copy()
#                         continue

#                     good_new = new_points[status == 1]
#                     good_old = prev_points[status == 1]

#                     if len(good_new) == 0:
#                         prev_points = None
#                         prev_gray = gray.copy()
#                         continue

#                     flow_vectors = []

#                     # Draw flow vectors
#                     for (new, old) in zip(good_new, good_old):
#                         a, b = new.ravel()
#                         c, d = old.ravel()

#                         flow_vectors.append([a - c, b - d])

#                         cv2.arrowedLine(frame,
#                                         (int(c), int(d)),
#                                         (int(a), int(b)),
#                                         (0,255,255), 2)

#                     prev_points = good_new.reshape(-1, 1, 2)

#                 prev_gray = gray.copy()

#         # Step 5: Estimate eye direction
#         direction = estimate_direction(flow_vectors)
#         # print("Detected direction:", direction)
#         if direction != last_direction:
#             print(f"[Direction changed] → {direction}")
#             last_direction = direction


#         cv2.putText(frame, f"Direction: {direction}", (20,40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

#         cv2.imshow("Eye Tracking with Optical Flow", frame)

#         if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()











import cv2
import numpy as np

# -------------------------------
# FACE & EYE DETECTORS
# -------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

# -------------------------------
# PARAMETERS
# -------------------------------
MIN_POINTS = 8
REINIT_INTERVAL = 5  # force re-detection every N frames
last_direction = None


# -------------------------------
# DETECTION FUNCTIONS
# -------------------------------
def detect_face(gray):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    return faces[0]  # take first face


def detect_eye(gray_face):
    eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 5)
    if len(eyes) == 0:
        return None
    return eyes[0]


# -------------------------------
# GAZE ESTIMATION
# -------------------------------
def estimate_direction(vectors, threshold=0.8):
    if len(vectors) == 0:
        return "Center"

    vectors = np.array(vectors)
    dx = np.median(vectors[:, 0])
    dy = np.median(vectors[:, 1])

    if dx > threshold:
        return "Right"
    elif dx < -threshold:
        return "Left"
    elif dy > threshold:
        return "Down"
    elif dy < -threshold:
        return "Up"
    else:
        return "Center"


# -------------------------------
# MAIN PROGRAM
# -------------------------------
def main():
    global last_direction

    # cap = cv2.VideoCapture(0)  # webcam
    cap = cv2.VideoCapture("D:/Documents/Study/DADN/gaze_test.mp4")  # video file

    prev_gray = None
    prev_points = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Normalize video size
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # CLAHE improves video stability
        clahe = cv2.createCLAHE(2.0, (8, 8))
        gray = clahe.apply(gray)

        frame_count += 1
        need_reinit = False

        # Conditions requiring reinitialization
        if prev_points is None or len(prev_points) < MIN_POINTS:
            need_reinit = True
        if frame_count % REINIT_INTERVAL == 0:
            need_reinit = True

        # -------------------------------
        # REINITIALIZE DETECTION
        # -------------------------------
        if need_reinit:
            face = detect_face(gray)
            if face is not None:
                (x, y, w, h) = face
                face_roi = gray[y:y+h, x:x+w]

                eye = detect_eye(face_roi)
                if eye is not None:
                    (ex, ey, ew, eh) = eye

                    eye_roi = face_roi[ey:ey+eh, ex:ex+ew]

                    points = cv2.goodFeaturesToTrack(
                        eye_roi,
                        maxCorners=50,
                        qualityLevel=0.01,
                        minDistance=5,
                        blockSize=7
                    )

                    if points is not None:
                        points[:, :, 0] += x + ex
                        points[:, :, 1] += y + ey

                        prev_points = points
                        prev_gray = gray.copy()
                        continue

        # -------------------------------
        # OPTICAL FLOW
        # -------------------------------
        if prev_points is not None and prev_gray is not None:

            prev_points = prev_points.astype(np.float32)

            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_points, None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )

            if new_points is not None and status is not None:
                good_new = new_points[status == 1]
                good_old = prev_points[status == 1]

                # If too few points, force re-detect next frame
                if len(good_new) < MIN_POINTS:
                    prev_points = None
                    prev_gray = gray.copy()
                    continue

                flow_vectors = []

                # Draw optical flow arrows
                for (new, old) in zip(good_new, good_old):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    flow_vectors.append([a - c, b - d])
                    cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)

                prev_points = good_new.reshape(-1, 1, 2)

                # -------------------------------
                # DIRECTION ESTIMATION
                # -------------------------------
                direction = estimate_direction(flow_vectors)

                # Log only if direction changed
                if direction != last_direction:
                    print(f"[Direction changed] → {direction}")
                    last_direction = direction

                cv2.putText(frame, f"Direction: {direction}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            prev_gray = gray.copy()

        # -------------------------------
        # DISPLAY
        # -------------------------------
        cv2.imshow("Eye Tracking", frame)
        if cv2.waitKey(50) == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
