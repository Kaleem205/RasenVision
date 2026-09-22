# RasenVision 🌀⚡

An interactive, real-time computer vision application that tracks hand gestures via MediaPipe and renders a procedural, crystal-blue **Rasenshuriken** effect directly onto the user's palm. Built with Python, OpenCV, and NumPy.

---

## ✨ Features

* **Real-Time Hand Tracking & Gesture Recognition:** Uses MediaPipe to track hand landmarks and classify poses (Closed Fist, Open Palm, and transitional states).
* **Robust Finite State Machine (FSM):** Manages gesture states (`IDLE`, `CHARGING`, `ACTIVATED`) with built-in tolerance for transitional hand shapes to prevent accidental resets.
* **Procedural Crystal-Blue Rasenshuriken Renderer:**
  * High-density animated white threads and crystal-cyan energy lines spinning dynamically.
  * 3D particle wind simulation with depth-based brightness gradients and trailing curves.
  * Massive, fast-spinning outer wind-shuriken blades with additive blending for real light interaction.
* **Smooth Animation Curves:** Exponential pop-in growth scaling matching authentic visual effects.

---

## 🛠️ Project Structure

```text
RasenVision/
├── assets/                 # Reference media and screen recordings
├── src/
│   ├── fsm/
│   │   └── gesture_fsm.py      # Finite state machine for gesture progression
│   ├── recognition/
│   │   ├── motion_detector.py  # Tracks dynamic motion (e.g., throwing)
│   │   └── pose_classifier.py  # MediaPipe wrapper & finger extension logic
│   ├── rendering/
│   │   └── procedural_renderer.py # Mathematical particle and shuriken shader logic
│   └── main.py                 # Application entry point and video loop
├── tests/                  # Unit and integration test suites
├── pyproject.toml          # Project metadata and build configurations
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* Webcam

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Kaleem205/RasenVision.git
cd RasenVision
```

2. Create and activate a virtual environment:

```bash
python -m venv rasenvision
source rasenvision/bin/activate  # On Windows use: rasenvision\Scripts\activate
```

3. Install dependencies:

```bash
pip install .
# Or manually:
pip install opencv-python mediapipe numpy
```

---

## 🎮 Usage

Run the main application script:

```bash
python src/main.py
```

* **Step 1 (`IDLE`):** Make a CLOSED FIST and hold it steady on screen to begin charging.
* **Step 2 (`CHARGING`):** Slowly open your hand into an OPEN PALM. The application tolerates transitional finger states without dropping your progress.
* **Step 3 (`ACTIVATED`):** The crystal-blue Rasenshuriken fully materializes and sits on your palm with active wind trails and particles.

---

## 🧰 Tech Stack

* **Language:** Python
* **Computer Vision:** OpenCV (`cv2`), MediaPipe
* **Math & Graphics:** NumPy, Trigonometric procedural geometry algorithms
