import cv2
import numpy as np

# Function to start tracking using optical flow
def start_tracking():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("D:/Documents/Study/DADN/gaze_test.mp4")  # video file
    scaling_factor = 0.5

    # Number of frames to track and number of frames to skip
    num_frames_to_track = 5
    num_frames_jump = 2

    # Tracking paths and frame index
    tracking_paths = []
    frame_index = 0

    # Parameters for calcOpticalFlowPyrLK in OpenCV
    # * winSize: window size for tracking
    # * maxLevel: pyramid levels
    # * criteria: termination criteria
    tracking_params = dict(
        winSize=(6, 6),
        maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Main loop
    while True:
        # Read frame
        _, frame = cap.read()

        # Resize frame
        frame = cv2.resize(frame, None,
                           fx=scaling_factor,
                           fy=scaling_factor,
                           interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Copy frame for drawing
        output_img = frame.copy()

        # If there are tracking paths, compute optical flow
        if len(tracking_paths) > 0:
            prev_img, current_img = prev_gray, frame_gray

            # Create feature points from tracking paths
            feature_points_0 = np.float32(
                [tp[-1] for tp in tracking_paths]
            ).reshape(-1, 1, 2)

            # Compute forward optical flow
            feature_points_1, _, _ = cv2.calcOpticalFlowPyrLK(
                prev_img,
                current_img,
                feature_points_0,
                None,
                **tracking_params
            )

            # Compute backward optical flow
            feature_points_0_rev, _, _ = cv2.calcOpticalFlowPyrLK(
                current_img,
                prev_img,
                feature_points_1,
                None,
                **tracking_params
            )

            # Compute difference between forward and backward flow
            diff_feature_points = abs(feature_points_1 - feature_points_0_rev) \
                .reshape(-1, 2).max(-1)

            # Select good feature points
            good_points = diff_feature_points < 1

            # Create new tracking paths
            new_tracking_paths = []

            # Loop through good points
            for tp, (x, y), good_flag in zip(
                tracking_paths,
                feature_points_1.reshape(-1, 2),
                good_points
            ):
                if not good_flag:
                    continue

                # Append new point
                tp.append((x, y))

                # Keep only recent frames
                if len(tp) > num_frames_to_track:
                    del tp[0]

                new_tracking_paths.append(tp)

                # Draw point
                cv2.circle(output_img, (int(x), int(y)),
                           1, (0, 255, 0), -1)

            # Update tracking paths
            tracking_paths = new_tracking_paths

            # Draw polylines for tracking paths
            cv2.polylines(
                output_img,
                [np.int32(tp) for tp in tracking_paths],
                False,
                (0, 150, 0)
            )

        # Add new feature points every few frames
        if not frame_index % num_frames_jump:

            # Create mask and draw circles
            mask = np.zeros_like(frame_gray)
            mask[:] = 255

            for x, y in [np.int32(tp[-1]) for tp in tracking_paths]:
                cv2.circle(mask, (x, y), 6, 0, -1)

            # Detect new good features
            feature_points = cv2.goodFeaturesToTrack(
                frame_gray,
                mask=mask,
                maxCorners=100,
                qualityLevel=0.3,
                minDistance=5,
                blockSize=3
            )

            # Append good features to tracking paths
            if feature_points is not None:
                for m, n in np.float32(feature_points).reshape(-1, 2):
                    tracking_paths.append([(m, n)])

        frame_index += 1
        prev_gray = frame_gray

        # Show output
        cv2.imshow("Optical Flow", output_img)

        # Keyboard handling
        k = cv2.waitKey(1)
        if k == 27:
            break


if __name__ == '__main__':
    start_tracking()
    cv2.destroyAllWindows()
