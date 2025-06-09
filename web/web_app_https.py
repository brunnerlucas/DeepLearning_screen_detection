import cv2
import numpy as np
from ultralytics import YOLO
import time
import mediapipe as mp
import torch
from flask import Flask, render_template, request, jsonify
import qrcode
import socket
import base64
from io import BytesIO
import os
import ssl

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

class MobileDetector:
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
        
        # Initialize variables for screen watching detection
        self.looking_at_screen = False
        self.looking_start_time = None
        self.total_looking_time = 0
        self.last_status = "Not looking at screen"
        self.session_start_time = time.time()
        
    def process_image(self, image_data):
        """Process a single image from mobile camera"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return None
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform person detection
            det_results = self.det_model(frame_rgb, verbose=False)
            bboxes = []
            
            for result in det_results:
                if result.boxes is not None:
                    for box in result.boxes:
                        if int(box.cls[0]) == 0:  # Class 0 is person
                            bboxes.append(box.xyxy[0].cpu().numpy())
            
            pose_looking = False
            
            # Pose detection on detected persons
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
            
            # Calculate times
            # minutes = int(self.total_looking_time // 60)
            # seconds = int(self.total_looking_time % 60)
            live_total = self.total_looking_time
            if self.looking_at_screen and self.looking_start_time is not None:
                live_total += current_time - self.looking_start_time

            minutes = int(live_total // 60)
            seconds = int(live_total % 60)
            
            return {
                'looking_at_screen': self.looking_at_screen,
                'status': status,
                'total_looking_time': live_total,
                'formatted_time': f"{minutes:02d}:{seconds:02d}",
                'gaze_detected': gaze_looking,
                'pose_detected': pose_looking,
                'faces_found': len(face_centers),
                'persons_found': len(bboxes)
            }
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return {
                'looking_at_screen': False,
                'status': f'Error: {str(e)}',
                'total_looking_time': self.total_looking_time,
                'formatted_time': '00:00',
                'gaze_detected': False,
                'pose_detected': False,
                'faces_found': 0,
                'persons_found': 0
            }

# Global detector instance
detector = MobileDetector()

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_frame():
    """Receive and analyze image from mobile camera"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data received'}), 400
        
        result = detector.process_image(image_data)
        
        if result is None:
            return jsonify({'error': 'Failed to process image'}), 500
            
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in analyze_frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_session():
    """Reset session statistics"""
    global detector
    detector.total_looking_time = 0
    detector.session_start_time = time.time()
    detector.looking_at_screen = False
    detector.looking_start_time = None
    detector.last_status = "Session reset"
    
    return jsonify({'message': 'Session reset successfully'})

@app.route('/status')
def get_status():
    """Get current status without processing new image"""
    # minutes = int(detector.total_looking_time // 60)
    # seconds = int(detector.total_looking_time % 60)
    current_time = time.time()
    live_total = detector.total_looking_time
    if detector.looking_at_screen and detector.looking_start_time is not None:
        live_total += current_time - detector.looking_start_time

    minutes = int(live_total // 60)
    seconds = int(live_total % 60)

    return jsonify({
        'looking_at_screen': live_total,
        'status': detector.last_status,
        'total_looking_time': detector.total_looking_time,
        'formatted_time': f"{minutes:02d}:{seconds:02d}"
    })

if __name__ == '__main__':
    # Install pyOpenSSL if not available
    try:
        import OpenSSL
    except ImportError:
        print("Installing pyOpenSSL for HTTPS support...")
        os.system("pip install pyOpenSSL")
    
    # Get local IP and port
    local_ip = get_local_ip()
    port = 8080
    url = f"https://{local_ip}:{port}"  # HTTPS URL
    
    print(f"\n{'='*50}")
    print(f"📱 Mobile Camera Screen Detection (HTTPS)")
    print(f"{'='*50}")
    print(f"Secure URL: {url}")
    print(f"Scan this QR code with your phone:")
    print(f"{'='*50}")
    
    # Generate and save QR code with HTTPS URL
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_img.save("qr_code_https.png")
    print(f"QR code saved as 'qr_code_https.png'")
    print(f"This version uses your PHONE'S camera!")
    print(f"WARNING: You'll see a security warning - click 'Advanced' -> 'Proceed'")
    print(f"{'='*50}\n")
    
    try:
        # Try to run with HTTPS
        app.run(host='0.0.0.0', port=port, debug=False, ssl_context='adhoc')
    except Exception as e:
        print(f"HTTPS failed: {e}")
        print("Falling back to HTTP...")
        print("Note: Camera may not work on HTTP - try using a different browser or localhost")
        app.run(host='0.0.0.0', port=port, debug=False) 