import cv2
import numpy as np
from ultralytics import YOLO
import time
import mediapipe as mp
import torch
from flask import Flask, render_template, Response, jsonify
import threading
import queue
import qrcode
import socket
import base64
from io import BytesIO

class GazeDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
        
        # Indices for left and right eye landmarks
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Indices for iris landmarks
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # Reference points for gaze direction
        self.LEFT_EYE_REF = [362, 263]  # Left eye corners
        self.RIGHT_EYE_REF = [33, 133]  # Right eye corners

    def get_gaze_ratio(self, landmarks, eye_indices, iris_indices, eye_ref_indices):
        # Get the eye region
        eye_points = np.array([landmarks[i] for i in eye_indices])
        iris_points = np.array([landmarks[i] for i in iris_indices])
        ref_points = np.array([landmarks[i] for i in eye_ref_indices])
        
        # Calculate the center of the eye and iris
        eye_center = np.mean(eye_points, axis=0)
        iris_center = np.mean(iris_points, axis=0)
        
        # Calculate the eye width for normalization
        eye_width = np.linalg.norm(ref_points[0] - ref_points[1])
        
        # Calculate the gaze vector (from eye center to iris center)
        gaze_vector = iris_center - eye_center
        
        # Normalize the gaze vector by eye width
        if eye_width > 0:
            gaze_vector = gaze_vector / eye_width
        
        return gaze_vector, eye_center

    def detect_gaze(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]
        
        results = self.face_mesh.process(frame_rgb)
        
        looking_at_screen = False
        gaze_vectors = []
        face_centers = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([[lm.x * frame_w, lm.y * frame_h] for lm in face_landmarks.landmark])
                
                # Get gaze vectors for both eyes
                left_gaze, left_center = self.get_gaze_ratio(landmarks, self.LEFT_EYE, self.LEFT_IRIS, self.LEFT_EYE_REF)
                right_gaze, right_center = self.get_gaze_ratio(landmarks, self.RIGHT_EYE, self.RIGHT_IRIS, self.RIGHT_EYE_REF)
                
                # Average the gaze vectors
                gaze_vector = (left_gaze + right_gaze) / 2
                gaze_vectors.append(gaze_vector)
                
                # Calculate face center
                face_center = (left_center + right_center) / 2
                face_centers.append(face_center)
                
                # Check if looking at screen (gaze vector should be pointing roughly forward)
                if abs(gaze_vector[0]) < 0.15 and abs(gaze_vector[1]) < 0.15:  # More strict thresholds
                    looking_at_screen = True
        
        return looking_at_screen, gaze_vectors, face_centers

def calculate_head_pose(keypoints):
    nose = keypoints[0]
    left_eye = keypoints[1]
    right_eye = keypoints[2]
    left_ear = keypoints[3]
    right_ear = keypoints[4]
    
    eye_center = (left_eye + right_eye) / 2
    head_direction = eye_center - nose
    ear_vector = right_ear - left_ear
    head_tilt = np.arctan2(ear_vector[1], ear_vector[0])
    
    return head_direction, head_tilt

def is_looking_at_camera(head_direction, head_tilt, threshold=0.3):
    norm = np.linalg.norm(head_direction)
    if norm == 0:
        return False
    
    normalized_direction = head_direction / norm
    is_facing_forward = abs(normalized_direction[0]) < threshold and normalized_direction[1] > -0.1
    is_not_tilted = abs(head_tilt) < np.pi/4
    
    return is_facing_forward and is_not_tilted

class ScreenWatchDetector:
    def __init__(self):
        # Use CPU for compatibility
        self.device = 'cpu'
        print(f"Using device: {self.device}")

        # Load the YOLOv11 detection and pose models (use nano versions for mobile)
        self.det_model = YOLO('../models/yolo11n.pt')  # Nano version for speed
        self.pose_model = YOLO('../models/yolo11n-pose.pt')  # Nano version for speed
        
        # Move models to appropriate device
        self.det_model.to(self.device)
        self.pose_model.to(self.device)
        
        # Initialize the gaze detector
        self.gaze_detector = GazeDetector()

        # Find and initialize webcam
        self.cap = None
        for camera_index in range(5):
            test_cap = cv2.VideoCapture(camera_index)
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret:
                    print(f"Found camera at index {camera_index}")
                    self.cap = test_cap
                    break
                else:
                    test_cap.release()
            else:
                test_cap.release()
        
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        
        # Set camera properties for mobile optimization
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        
        # Initialize variables for screen watching detection
        self.looking_at_screen = False
        self.looking_start_time = None
        self.total_looking_time = 0
        self.last_status = "Not looking at screen"
        
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
            
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform detection
        det_results = self.det_model(frame_rgb, verbose=False)
        bboxes = []
        
        for result in det_results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # Class 0 is person
                    bboxes.append(box.xyxy[0].cpu().numpy())
        
        pose_looking = False
        
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            person_crop = frame_rgb[y1:y2, x1:x2]
            pose_results = self.pose_model(person_crop, verbose=False)
            
            for pose in pose_results:
                if pose.keypoints is not None and pose.keypoints.conf is not None:
                    kpts = pose.keypoints.xy[0].cpu().numpy()
                    confs = pose.keypoints.conf[0].cpu().numpy()
                    kpts[:, 0] += x1
                    kpts[:, 1] += y1
                    
                    # Check if person is looking at camera using pose
                    if len(confs) >= 5 and all(confs[i] > 0.3 for i in range(5)):
                        head_direction, head_tilt = calculate_head_pose(kpts)
                        if is_looking_at_camera(head_direction, head_tilt):
                            pose_looking = True
        
        # Detect gaze using MediaPipe
        gaze_looking, gaze_vectors, face_centers = self.gaze_detector.detect_gaze(frame)
        
        # Combine both detection methods
        current_looking = gaze_looking or pose_looking
        
        # Update looking time
        current_time = time.time()
        if current_looking:
            if not self.looking_at_screen:
                self.looking_start_time = current_time
            self.looking_at_screen = True
        else:
            if self.looking_at_screen:
                if self.looking_start_time is not None:
                    self.total_looking_time += current_time - self.looking_start_time
                self.looking_start_time = None
            self.looking_at_screen = False
        
        # Update status
        status = "Looking at screen"
        if self.looking_at_screen:
            if gaze_looking and pose_looking:
                status += " (Both methods)"
            elif gaze_looking:
                status += " (Gaze)"
            else:
                status += " (Pose)"
        else:
            status = "Not looking at screen"
        
        self.last_status = status
        
        # Add status text to frame
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 255, 0) if self.looking_at_screen else (0, 0, 255), 2)
        
        # Add total looking time
        minutes = int(self.total_looking_time // 60)
        seconds = int(self.total_looking_time % 60)
        time_text = f"Total: {minutes:02d}:{seconds:02d}"
        cv2.putText(frame, time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame, {
            'looking_at_screen': self.looking_at_screen,
            'status': status,
            'total_looking_time': self.total_looking_time,
            'formatted_time': f"{minutes:02d}:{seconds:02d}"
        }

# Global detector instance
detector = ScreenWatchDetector()

# Flask app
app = Flask(__name__)

def get_local_ip():
    """Get the local IP address"""
    try:
        # Connect to a remote server to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def generate_qr_code(url):
    """Generate QR code for the given URL"""
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url)
    qr.make(fit=True)
    
    qr_img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64 for web display
    buffered = BytesIO()
    qr_img.save(buffered)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame, data = detector.process_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)  # Reduce frame rate for mobile
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    frame, data = detector.process_frame()
    return jsonify(data if data else {
        'looking_at_screen': False,
        'status': 'No data',
        'total_looking_time': 0,
        'formatted_time': '00:00'
    })

if __name__ == '__main__':
    # Get local IP and port
    local_ip = get_local_ip()
    port = 8080
    url = f"http://{local_ip}:{port}"
    
    print(f"\n{'='*50}")
    print(f"Screen Detection Web App")
    print(f"{'='*50}")
    print(f"Local URL: {url}")
    print(f"Scan this QR code with your phone:")
    print(f"{'='*50}")
    
    # Generate and display QR code
    qr_b64 = generate_qr_code(url)
    
    # Save QR code as file for easy access
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_img.save("web/qr_code.png")
    print(f"QR code saved as 'qr_code.png'")
    print(f"{'='*50}\n")
    
    app.run(host='0.0.0.0', port=port, debug=False) 