# 🎯 DeepLearning Screen Detection

A real-time screen watching detection system with desktop, web, and mobile camera interfaces using pose estimation (YOLOv11) and gaze detection (MediaPipe). Perfect for productivity tracking and focus monitoring.

> **📱 Mobile Camera**: Uses HTTPS for reliable camera access. The system processes your phone's camera feed locally using AI models for real-time gaze and pose detection.

## 📁 Project Structure

```
DeepLearning_screen_detection/
├── 📱 web/                     # Mobile camera application
│   ├── web_app_https.py       # HTTPS mobile server (recommended)
│   ├── web_app.py             # HTTP mobile server (backup)
│   ├── templates/
│   │   └── index.html         # Mobile camera interface
│   ├── qr_code_https.png      # HTTPS QR code
│   └── qr_code.png           # HTTP QR code
├── 🖥️  main/                  # Desktop application
│   └── main.py               # Desktop GUI version
├── 🤖 models/                 # YOLO model files
│   ├── yolo11x.pt            # Full detection model (109MB)
│   ├── yolo11x-pose.pt       # Full pose model (113MB)
│   ├── yolo11n.pt            # Nano detection model (5MB)
│   └── yolo11n-pose.pt       # Nano pose model (6MB)
├── 📚 Archive/                # Legacy code
├── 📦 screenwatch-venv/       # Virtual environment
├── requirements.txt          # Python dependencies
├── run_desktop.sh            # Desktop launcher
├── run_mobile_https.sh       # HTTPS mobile launcher (recommended)
├── run_web.sh               # HTTP mobile launcher (backup)
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/DeepLearning_screen_detection.git
cd DeepLearning_screen_detection

# Create and activate virtual environment
python3 -m venv screenwatch-venv
source screenwatch-venv/bin/activate  
# On Windows: screenwatch-venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
#If that doesn't work, run: pythom -m pip install pip
pip install -r requirements.txt
```

### 2. Download Models
The models will be downloaded automatically on first run, or you can download them manually:
```bash
# For desktop (full models)
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt'); YOLO('yolo11x-pose.pt')"

# For mobile (nano models - faster)
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11n-pose.pt')"
```

### 3. Install HTTPS Dependencies (for mobile camera)
```bash
pip install pyOpenSSL
```

## 🖥️ Desktop Application

### Features
- High-resolution detection (480x360)
- Real-time pose and gaze tracking
- Visual feedback with skeletal overlay
- Time tracking with live statistics
- Hardware acceleration (MPS/CUDA when available)

### Running Desktop Version
```bash
source screenwatch-venv/bin/activate
cd main
python main.py
# OR use launcher script:
./run_desktop.sh #On Mac/Linux
.\run_desktop.ps1 #On Windows
```

### Desktop Controls
- **Q**: Quit application
- **Space**: Pause/Resume (if implemented)
- Live statistics displayed on screen

## 📱 Mobile Camera Application

### Features
- **Phone Camera Detection**: Uses your phone's native camera via getUserMedia API
- **Real-time Analysis**: Combined pose estimation and gaze tracking on device
- **Mobile Optimized**: Touch-friendly interface designed for phones
- **Camera Controls**: Start/stop camera, switch between front/back cameras
- **Live Detection Feedback**: Shows Gaze ✅/❌, Pose ✅/❌, Face ✅/❌ status
- **Performance Stats**: Focus time, efficiency tracking, session management
- **HTTPS Support**: Secure connection for reliable camera access
- **QR Code Access**: Instant mobile deployment
- **Low Latency**: Uses nano YOLO models (5-6MB each) for fast processing

### Running Mobile Camera Version (HTTPS - Recommended)
```bash
source screenwatch-venv/bin/activate
# HTTPS version (recommended for camera permissions)
./run_mobile_https.sh #On Mac/Linux
.\run_mobile_https.ps1 #On Windows

If you run into permission denied errors, run chmod +x run_mobile_https.sh and then ./run_mobile_https.sh again
```

### Alternative HTTP Version (Limited Camera Support)
```bash
# HTTP version (camera may not work on all browsers)
./run_web.sh #On Mac/Linux
.\run_web.ps1 #On Windows

If you run into permission denied errors, run chmod +x run_web.sh and then ./run_web.sh again
```

### Mobile Usage
1. **QR Code**: Scan the generated `qr_code_https.png` with your phone
2. **Security Warning**: Click "Advanced" → "Proceed" to accept self-signed certificate
3. **Camera Permissions**: Allow camera access when prompted
4. **Detection**: Point camera at your face while looking at phone screen

**📋 Troubleshooting Camera Access:**
- **Use HTTPS version** for best camera compatibility
- **iOS Safari**: Settings → Safari → Camera → Allow
- **Android Chrome**: Tap camera icon in address bar → Allow
- **Alternative**: Try Firefox or Chrome if Safari doesn't work

### Mobile Interface Features
- **Native camera access** through phone's camera (not laptop streaming)
- **Touch-optimized controls** with haptic feedback
- **Real-time detection badges** showing which methods are active
- **Camera switching** between front and back cameras
- **Live status indicators** with animated pulsing dots
- **Session statistics** and efficiency percentage tracking
- **1-second analysis intervals** for optimal mobile performance

## 🛠️ Technical Details

### Models Used
- **Desktop**: 
  - `yolo11x.pt` - Full detection model (109MB, higher accuracy)
  - `yolo11x-pose.pt` - Full pose model (113MB, higher accuracy)
- **Web/Mobile**: 
  - `yolo11n.pt` - Nano detection model (5MB, optimized speed)
  - `yolo11n-pose.pt` - Nano pose model (6MB, optimized speed)
- **Gaze Detection**: MediaPipe Face Mesh for iris tracking

### Detection Methods
1. **Pose Estimation**: Analyzes head orientation using facial keypoints
2. **Gaze Tracking**: Uses iris position relative to eye corners
3. **Combined Logic**: OR operation - triggers if either method detects attention

### Performance Specifications
| Feature | Desktop | Web/Mobile |
|---------|---------|------------|
| Resolution | 480x360 | 320x240 |
| Frame Rate | 30 FPS | 15 FPS |
| Model Size | ~220MB | ~11MB |
| Detection Rate | ~20 FPS | ~10 FPS |
| Hardware Acceleration | MPS/CUDA/CPU | CPU Only |

## 🎨 Customization

### Sensitivity Tuning
Edit detection thresholds in the respective files:

**Gaze Sensitivity:**
```python
# More strict (requires more precise looking)
if abs(gaze_vector[0]) < 0.10 and abs(gaze_vector[1]) < 0.10:

# More lenient (easier to trigger)
if abs(gaze_vector[0]) < 0.20 and abs(gaze_vector[1]) < 0.20:
```

**Pose Sensitivity:**
```python
# In is_looking_at_camera() function
threshold = 0.2  # More strict
threshold = 0.4  # More lenient
```

### Camera Settings
```python
# Higher resolution (desktop)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Lower resolution (mobile optimization)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
```

## 🌐 Network & Deployment

### Web App Network Requirements
- Computer and phone on same Wi-Fi network
- Firewall allowing port 8080
- For external access: router port forwarding

### Production Deployment
```bash
# Install production WSGI server
pip install gunicorn

# Run with gunicorn
cd web
gunicorn -w 4 -b 0.0.0.0:8080 web_app:app
```

## 🐛 Troubleshooting

### Common Issues

**"Camera Permission Denied" on Mobile**
```bash
# Use HTTPS version (recommended)
./run_mobile_https.sh

# Accept security warning on phone:
# Click "Advanced" → "Proceed to [your-ip] (unsafe)"
```

**"ModuleNotFoundError: No module named 'cv2'"**
```bash
# Ensure virtual environment is activated
source screenwatch-venv/bin/activate
pip install opencv-python
```

**"Address already in use" (Port 8080)**
```bash
# Port conflict - try different port or kill existing process
sudo lsof -ti:8080 | xargs kill -9
```

**"Camera not found" (Desktop)**
- Check camera permissions
- Try different camera indices (0, 1, 2...)
- Disable other camera applications

**"HTTPS failed" Error**
```bash
# Install SSL dependencies
pip install pyOpenSSL
# Or use HTTP fallback (limited camera support)
./run_web.sh
```

**Poor Detection Accuracy**
- Ensure good lighting
- Position camera at eye level
- Adjust detection thresholds
- Check camera focus and resolution

### Performance Optimization

**For slower devices:**
```python
# Reduce frame rate
time.sleep(0.2)  # 5 FPS instead of 10 FPS

# Lower resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
```

## 📊 Use Cases

- **Productivity Tracking**: Monitor focus time during work
- **Study Sessions**: Track attention during learning
- **Screen Time Awareness**: Understand digital habits
- **Parental Controls**: Monitor children's screen engagement
- **Accessibility**: Hands-free interaction systems
- **Research**: Attention and engagement studies

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is open source. Feel free to use and modify for your needs.

## 🎯 Future Enhancements

- [ ] Multiple person tracking
- [ ] Eye strain detection
- [ ] Break reminders
- [ ] Data export (CSV, JSON)
- [ ] Real-time notifications
- [ ] Cloud dashboard
- [ ] Calibration wizard
- [ ] Voice commands

---

**Enjoy tracking your screen time with style!** 🎯📱💻

For questions or support, please open an issue on GitHub.
