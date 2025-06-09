# 🎯 DeepLearning Screen Detection

A real-time screen watching detection system with desktop, web, and mobile camera interfaces using pose estimation (YOLOv11) and gaze detection (MediaPipe). Perfect for productivity tracking and focus monitoring.

## 📁 Project Structure

```
DeepLearning_screen_detection/
├── 📱 web/                     # Mobile web application
│   ├── web_app.py             # Flask server
│   ├── templates/             # HTML templates
│   └── qr_code.png           # Generated QR code
├── 🖥️  desktop/               # Desktop application
│   └── main.py               # Desktop GUI version
├── 🤖 models/                 # YOLO model files
│   ├── yolo11x.pt            # Full detection model (109MB)
│   ├── yolo11x-pose.pt       # Full pose model (113MB)
│   ├── yolo11n.pt            # Nano detection model (5MB)
│   └── yolo11n-pose.pt       # Nano pose model (6MB)
├── 📚 Archive/                # Legacy code
├── 📦 screenwatch-venv/       # Virtual environment
├── requirements.txt          # Python dependencies
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
source screenwatch-venv/bin/activate  # On Windows: screenwatch-venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download Models
The models will be downloaded automatically on first run, or you can download them manually:
```bash
# For desktop (full models)
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt'); YOLO('yolo11x-pose.pt')"

# For web (nano models - faster)
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11n-pose.pt')"
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
cd desktop
python main.py
```

### Desktop Controls
- **Q**: Quit application
- **Space**: Pause/Resume (if implemented)
- Live statistics displayed on screen

## 📱 Mobile Web Application

### Features
- **Real-time Detection**: Combined pose estimation and gaze tracking
- **Mobile Optimized**: Responsive design for phones/tablets
- **QR Code Access**: Instant mobile deployment
- **Performance Stats**: Focus time and efficiency tracking
- **Low Latency**: Uses nano YOLO models (5-6MB each)
- **Cross-Platform**: Works on any device with a web browser

### Running Web Version
```bash
source screenwatch-venv/bin/activate
cd web
python web_app.py
```

### Mobile Access
1. **QR Code**: Scan the generated `qr_code.png` with your phone
2. **Manual**: Open browser and go to `http://YOUR_LOCAL_IP:8080`
3. **Same Network**: Ensure your phone and computer are on the same Wi-Fi

### Web Interface Features
- Live video feed from your webcam
- Real-time status indicator (green/red pulsing dot)
- Total attention time counter
- Session statistics and efficiency percentage
- Touch-optimized reset button

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

**"ModuleNotFoundError: No module named 'cv2'"**
```bash
# Ensure virtual environment is activated
source screenwatch-venv/bin/activate
pip install opencv-python
```

**"Address already in use" (Port 5000)**
```bash
# macOS AirPlay conflict - web app now uses port 8080
# Or disable AirPlay Receiver in System Preferences
```

**"Camera not found"**
- Check camera permissions
- Try different camera indices (0, 1, 2...)
- Disable other camera applications

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
