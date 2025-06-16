# DeepLearning Screen Watching Detection

A real-time multi-camera system to detect whether a user is watching a screen, and raise an alert if someone else is also watching.  
It combines **pose estimation (YOLOv11)** and **gaze detection (MediaPipe)**, with a full **desktop + HTTPS mobile interface**.

---

## Features

- Real-time head pose estimation (YOLOv11 or YOLOv11n for mobile)
- Gaze detection using iris/eye landmarks (MediaPipe)
- Fusion of both methods to infer screen attention
- **HTTPS Flask server** for secure access via **mobile camera**
- Automatic **QR code generation**
- Platform-compatible launch scripts for Windows & Unix
- Modular and extendable code (separation between core logic, web app, and archives)

---

## How It Works

| Component              | Description                                     |
|-----------------------|-------------------------------------------------|
| **YOLOv11**           | Detects people and estimates body keypoints     |
| **MediaPipe FaceMesh**| Tracks gaze direction using iris landmarks      |
| **Fusion logic**      | Combines head pose + gaze to determine attention|
| **Flask Web App**     | Receives mobile camera frames via HTTPS         |
| **QR Code**           | Easy access to mobile camera view               |

---

## 📁 Project Structure

```
DeepLearning_screen_detection/
├── 📱 web/                     # Mobile camera application
│   ├── web_app_https.py       # HTTPS mobile server
│   ├── templates/
│   │   └── index.html         # html code for frontend
│   ├── qr_code_https.png      # QR code
├── 🧪 main/                    # WIP: Initial logic and prototyping
│   └── main.py
├── 📚 Archive/                # Legacy code
├── requirements.txt          # Python dependencies
├── run_mobile_https.sh       # HTTPS mobile launcher (recommended)
├── run_mobile_https.ps1      # Windows launcher script
├── .gitignore
└── README.md

```

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/DeepLearning_screen_detection.git
cd DeepLearning_screen_detection

```

### 2. Setup Environment
```bash
# Create and activate virtual environment
python3 -m venv screenwatch-venv
source screenwatch-venv/bin/activate  # On Windows: screenwatch-venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

```

### 2. Run the Code (Launch Mobile Detection via HTTPS with camera support)
```bash
source screenwatch-venv/bin/activate

# On Mac/Linux
chmod +x run_mobile_https.sh
./run_mobile_https.sh

# On Windows (PowerShell)
.\run_mobile_https.ps1

```

### 📱 How to Use on Mobile
1. **Start the Server**: Run the HTTPS Flask server using the script (`run_mobile_https.sh` or `.ps1`)
2. **QR Code**: A QR code will appear in the terminal and be saved as `qr_code_https.png`
3. **Scan**: Scan the QR code with your phone to open the web app
4. **Camera Permissions**: Allow access to your mobile camera when prompted
5. **Detection**: Look at your screen — if someone else appears in the frame, an alert will trigger
6. **Status Page**: Open `/status` in your mobile browser to monitor gaze without sending frames
7. **Reset Stats**: Send a POST request to `/reset` to clear total looking time


---

## 📋 Requirements

```txt
numpy==1.26.4
opencv-python==4.11.0.86
torch==2.2.2
torchvision==0.17.2
mediapipe==0.10.21
ultralytics==8.3.136
flask==3.0.0
qrcode==7.4.2
pillow==10.1.0
pyOpenSSL==25.1.0
```
---

## Gaze & Pose Fusion
- `main.py` handles real-time webcam input, fusing head pose detection and gaze tracking.
- `web_app_https.py` processes base64-encoded mobile camera frames via HTTPS.
- The mobile version uses **YOLOv11n (nano)** for faster inference on CPU.

---

## Notes

- This project uses your **local IP address** for mobile access, not a public domain.
- HTTPS certificates are generated **adhoc**, so browsers will show a security warning (click *Advanced* → *Proceed anyway*).
- **YOLOv11n** is used on mobile for **low-latency inference**.

---

## Demo & Video

👉 *Insert demo video link here *

---

## Authors & Contributors

- Lucas Brunner  
- Isaac Chaljub Restrepo
- Louis-Esmel Kodo
- Robert Koegel
- Diego López Pizarro
- Alejandro Felipe Pérez Vargas
- Spencer Sveda Wood
