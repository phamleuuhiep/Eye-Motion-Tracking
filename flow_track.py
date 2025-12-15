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

    frame_jitters = []
    lost_frames = 0

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
                    lost_frames += 1
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


                if len(flow_vectors) > 0: ### CALCULATE JITTER
                    mags = [np.linalg.norm(v) for v in flow_vectors]
                    frame_jitters.append(np.mean(mags))


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

    if len(frame_jitters) > 0:
        print("Optical Flow Jitter (mean px):", np.mean(frame_jitters))
    stability = 1 - lost_frames / frame_count if frame_count > 0 else 0
    print("Tracking stability:", stability)



    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
