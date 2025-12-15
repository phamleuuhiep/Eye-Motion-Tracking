# Eye Motion Tracking using Haar Cascades and Optical Flow

This project implements a real-time **eye motion and gaze direction tracking system** based on classical computer vision techniques. The system combines **Haar Cascade detection** with **Lucas–Kanade Optical Flow** to improve temporal stability and reduce jitter in eye tracking.

---

## 📌 Project Overview

Eye tracking is a fundamental task in human–computer interaction (HCI), assistive technologies, and behavioral analysis. However, frame-by-frame detection methods often suffer from instability and jitter.

This project addresses that issue by:
- Detecting face and eyes using Haar cascades
- Tracking eye motion using Optical Flow
- Comparing **Haar-only** vs **Haar + Optical Flow** approaches quantitatively

---

## ✨ Key Features

- Face and eye detection using Haar Cascades
- Optical Flow–based feature point tracking
- Gaze direction estimation (Left / Right / Up / Down / Center)
- Quantitative evaluation using:
  - **Jitter** (temporal smoothness)
  - **Tracking Stability**
- Real-time visualization
- Video-based and webcam-based input support

---

## 🧠 Methodology

### 1. Haar Cascade Detection
- Detects face and eye regions independently in each frame
- Simple and fast, but sensitive to noise and illumination changes

### 2. Optical Flow Enhancement
- Uses Lucas–Kanade Optical Flow to track feature points inside eye regions
- Reduces frame-to-frame re-detection
- Improves temporal consistency and robustness

---

## 📊 Evaluation Metrics

### 🔹 Jitter
Measures the **average inter-frame displacement** of eye positions (in pixels).

- Lower jitter → smoother and more stable tracking

### 🔹 Tracking Stability
Measures the proportion of frames where tracking is successfully maintained:

\[
\text{Stability} = 1 - \frac{\text{Lost Frames}}{\text{Total Frames}}
\]

- Higher stability → fewer tracking failures

---

## 📈 Experimental Results

| Method                 | Jitter (px) | Stability |
|------------------------|-------------|-----------|
| Haar-only              | 4.53        | 0.83      |
| Haar + Optical Flow    | 1.88        | 0.997     |

**Observation:**
- Optical Flow reduces jitter by more than **2×**
- Tracking stability improves from **~83% to ~99.7%**

---

## 📂 Project Structure

```text
Eye-Motion-Tracking/
│
├── face.py            # Face detection using Haar cascades
├── eye.py             # Haar-only eye detection & jitter/stability evaluation
├── flow_track.py      # Haar + Optical Flow eye tracking
├── optical.py         # Optical Flow utilities
├── main.py            # Main execution script
│
├── haarcascade_frontalface_default.xml
├── haarcascade_eye_tree_eyeglasses.xml
│
├── test videos (*.mp4)
├── README.md
```
---
## ▶️ How to Run
1️⃣ Install Dependencies

Make sure Python ≥ 3.8 is installed. Then install required libraries:

pip install opencv-python numpy


If you use a virtual environment, activate it before installing dependencies.

2️⃣ Prepare Input Source

There are 2 options to use the project:

Webcam

Pre-recorded video (.mp4)

Edit the following line in the source files if needed:

cv2.VideoCapture(0)             # Webcam
cv2.VideoCapture("video.mp4")   # Video file

3️⃣ Run Haar-only Eye Tracking
python eye.py


This mode performs:

Face and eye detection using Haar Cascades

Jitter and tracking stability evaluation (without Optical Flow)

4️⃣ Run Haar + Optical Flow Tracking
python flow_track.py


This mode performs:

Initial Haar-based eye detection

Feature point tracking using Lucas–Kanade Optical Flow

Gaze direction estimation

Quantitative evaluation (jitter & stability)

## 🎥 Demo Videos

🎬 Demo videos illustrating system performance are available at the following link:

👉 Google Drive – Demo Videos:
https://drive.google.com/drive/folders/1hjI2Q5UmiDAs2vcX3wgf38wDBj49h3H5

The demos include:

Haar-only eye tracking

Haar + Optical Flow tracking

## ⚠️ Limitations

Despite the improved stability achieved using Optical Flow, the system still has several limitations:

Haar cascades are sensitive to lighting variations and noise

Only works reliably for near-frontal face orientations

No explicit pupil or iris localization

Optical Flow may drift over time and requires periodic re-initialization

Performance degrades under large head rotations or occlusions

## 🚀 Future Work

Potential improvements and extensions include:

Integrating facial landmark models (MediaPipe, Dlib)

Adding pupil and iris segmentation for precise gaze estimation

Applying Kalman filtering or temporal smoothing

Incorporating head pose estimation

Replacing Haar cascades with deep learning–based detectors

Extending the system to multi-face tracking

## 📜 License

This project is released for academic and research purposes only.

## 👤 Author

Pham Le Huu Hiep
Faculty of Computer Science
Ho Chi Minh City University of Technology (HCMUT)
Vietnam National University – Ho Chi Minh City

## ⭐ Acknowledgements

OpenCV Library

Haar Cascade Classifiers

Lucas–Kanade Optical Flow algorithm

Computer Vision research community

