# 🎯 DeepLearning Screen Detection

A real-time screen watching detection system with desktop, web, and mobile camera interfaces using pose estimation (YOLOv11) and gaze detection (MediaPipe)

> **📱 Mobile Camera**: Uses HTTPS for reliable camera access. The system processes your phone's camera feed locally using AI models for real-time gaze and pose detection.

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

### 1. Setup Environment
```bash
git clone https://github.com/YOUR_USERNAME/DeepLearning_screen_detection.git
cd DeepLearning_screen_detection

# Create and activate virtual environment
python3 -m venv screenwatch-venv
source screenwatch-venv/bin/activate  # On Windows: screenwatch-venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```
### 2. Run the Code
```bash
source screenwatch-venv/bin/activate

# On Mac/Linux
chmod +x run_mobile_https.sh
./run_mobile_https.sh

# On Windows
.\run_mobile_https.ps1

```
### Mobile Usage
1. **QR Code**: Scan the generated `qr_code_https.png` with your phone
2. **Camera Permissions**: Allow camera access when prompted
3. **Detection**: Point camera at your face while looking at phone screen
4. **Alert**: If a second person watches your screen an alert should pop up





