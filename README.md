# ✋🤖 AI-Powered Gesture Control Drone

This project enables **hands-free control of drones** using real-time **hand gesture recognition**, **palm tracking**, and **gesture-based commands**. By integrating computer vision, machine learning, and MAVLink-based drone communication, the system lets users intuitively control drones using simple hand motions in front of a webcam.

> 🚀 Designed for next-gen HCI, drone innovation, and gesture-based robotics research.

---

## 📁 Directory Structure

. ├── app.py ├── gesture_to_command.py ├── hand_coordinates.txt ├── keypoint_classification.ipynb ├── point_history_classification.ipynb │ ├── model/ │ ├── keypoint_classifier/ │ │ ├── keypoint.csv │ │ ├── keypoint_classifier.hdf5 │ │ ├── keypoint_classifier.tflite │ │ ├── keypoint_classifier.py │ │ └── keypoint_classifier_label.csv │ │ │ └── point_history_classifier/ │ ├── point_history.csv │ ├── point_history_classifier.hdf5 │ ├── point_history_classifier.tflite │ ├── point_history_classifier.py │ └── point_history_classifier_label.csv │ └── utils/ └── cvfpscalc.py


---

## 🧠 How It Works

### 1. **Hand Detection & Palm Tracking**
- Uses **MediaPipe** to detect and track 21 hand landmarks in real-time.
- Extracts the palm center (landmark `0`) for directional input.

### 2. **Gesture Recognition**
- Static gestures (e.g., ✋✊☝️👌) classified using `keypoint_classifier.tflite`.
- Dynamic gestures (e.g., waving or pointing motions) processed with `point_history_classifier.tflite`.

### 3. **Drone Command Execution**
- The palm coordinates are written to `hand_coordinates.txt` by `app.py`.
- `gesture_to_command.py` reads these coordinates and sends drone commands using MAVLink:
  - Arm/disarm
  - Move left/right
  - Maintain hover altitude

---

## 🚦 Key Components

### `app.py`
- Captures live webcam feed
- Detects hand, extracts gestures
- Logs coordinates to file for drone interaction

### `gesture_to_command.py`
- Connects to APM flight controller via COM port
- Arms drone, sets flight mode
- Converts gesture directions to MAVLink movement commands

---

## 📚 Model Training

| Notebook | Purpose |
|----------|---------|
| `keypoint_classification.ipynb` | Train hand sign recognition using 21 landmarks |
| `point_history_classification.ipynb` | Train dynamic gesture recognition using finger trajectory history |

---

## 🔧 Setup

### 🔗 Dependencies
- Python 3.8+
- MediaPipe
- OpenCV
- TensorFlow Lite
- pymavlink

Install dependencies:

pip install -r requirements.txt

🛠️ Running the System
Start gesture detection:

python app.py
Start drone controller interface:

python gesture_to_command.py

💡 Make sure your drone is connected (via COM3 or appropriate port) and set to GUIDED mode.

💥 Features
🔴 Real-time gesture and palm tracking

🟢 Gesture-to-drone command mapping

🔵 Modular, extendable ML architecture

🟡 Compatible with APM 2.8 / Pixhawk via MAVLink

🟣 Training and logging tools included

🎯 Use Cases
Autonomous aerial filming for vloggers & content creators

Touch-free drone control for accessibility

Drone racing without physical controllers

Interactive robotics education

🎬 Demo

![Gesture Control Demo](./demo.gif")

![Gesture Control Demo](./demo_1.gif)

👨‍💻 Contributors
You – Core Developer, System Architect

📄 License
This project is licensed under the MIT License.

🤝 Support
If you find this project helpful, feel free to give it a ⭐ on GitHub, open issues for bugs, or fork it for your own use!

yaml
Copy
Edit

---

Let me know if you want:
- A demo badge or GitHub Actions CI badge added
- A quick `requirements.txt` to go with it
- Help embedding demo images/GIFs

I can also generate a `LICENSE` file and `.gitignore` if you're publishing it as a complete repo.

