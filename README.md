# DeepLearning_screen_detection

# Screen Watcher Detection

A real-time screen-watching detection system using pose estimation with YOLOv11 and optional gaze estimation via MediaPipe (if enabled). Built for macOS and Windows.


## Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/DeepLearning_screen_detection.git
cd DeepLearning_screen_detection
```
2. Create and activate a virtual environment
```bash
python3 -m venv screenwatch-venv
source screenwatch-venv/bin/activate
```
4. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
▶️ Running the Application
Make sure both yolo11x.pt and yolo11x-pose.pt are in the root directory.
python3 main.py
Press q to exit the webcam window.
