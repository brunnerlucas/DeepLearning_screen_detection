import cv2
import numpy as np
from ultralytics import YOLO
import time
import mediapipe as mp
import math

class GazeDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
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
                # The y-component should be close to 0 (not looking up or down)
                # The x-component should be close to 0 (not looking left or right)
                if abs(gaze_vector[0]) < 0.15 and abs(gaze_vector[1]) < 0.15:  # More strict thresholds
                    looking_at_screen = True
                
                # Draw the gaze direction
                # Scale the arrow length based on the frame size
                arrow_length = min(frame_w, frame_h) * 0.2
                gaze_end = face_center + gaze_vector * arrow_length
                
                # Draw arrows for both eyes
                cv2.arrowedLine(frame, 
                              (int(left_center[0]), int(left_center[1])),
                              (int(left_center[0] + left_gaze[0] * arrow_length), 
                               int(left_center[1] + left_gaze[1] * arrow_length)),
                              (0, 255, 0), 2)
                
                cv2.arrowedLine(frame, 
                              (int(right_center[0]), int(right_center[1])),
                              (int(right_center[0] + right_gaze[0] * arrow_length), 
                               int(right_center[1] + right_gaze[1] * arrow_length)),
                              (0, 255, 0), 2)
                
                # Draw the average gaze direction
                cv2.arrowedLine(frame, 
                              (int(face_center[0]), int(face_center[1])),
                              (int(gaze_end[0]), int(gaze_end[1])),
                              (255, 0, 0), 2)
        
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

def is_looking_at_camera(head_direction, head_tilt, threshold=0.3):  # Reduced threshold
    norm = np.linalg.norm(head_direction)
    if norm == 0:
        return False
    
    normalized_direction = head_direction / norm
    is_facing_forward = abs(normalized_direction[0]) < threshold and normalized_direction[1] > -0.1
    is_not_tilted = abs(head_tilt) < np.pi/4  # Back to 45 degrees
    
    return is_facing_forward and is_not_tilted

def draw_pose(image, keypoints_xy, keypoints_conf, thickness=2):
    if keypoints_xy is None or len(keypoints_xy) == 0 or keypoints_conf is None:
        return image

    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8),
        (8, 10), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    for person_idx, (kpts, confs) in enumerate(zip(keypoints_xy, keypoints_conf)):
        for i, (x, y) in enumerate(kpts):
            if confs[i] > 0.3:
                cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)

        for i, j in skeleton:
            if confs[i] > 0.3 and confs[j] > 0.3:
                pt1 = (int(kpts[i][0]), int(kpts[i][1]))
                pt2 = (int(kpts[j][0]), int(kpts[j][1]))
                cv2.line(image, pt1, pt2, (255, 0, 0), thickness)

    return image

def main():
    # Load the YOLOv11 detection and pose models
    det_model = YOLO('yolo11x.pt')
    pose_model = YOLO('yolo11x-pose.pt')
    
    # Initialize the gaze detector
    gaze_detector = GazeDetector()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    print("Press 'q' to quit the application")
    
    # Initialize variables for screen watching detection
    looking_at_screen = False
    looking_start_time = None
    total_looking_time = 0
    last_update_time = time.time()
    update_interval = 1.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally to handle inverted camera
        frame = cv2.flip(frame, 1)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform detection
        det_results = det_model(frame_rgb)
        bboxes = []
        
        for result in det_results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # Class 0 is person
                    bboxes.append(box.xyxy[0].cpu().numpy())
        
        keypoints_xy = []
        keypoints_conf = []
        pose_looking = False
        
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            person_crop = frame_rgb[y1:y2, x1:x2]
            pose_results = pose_model(person_crop)
            
            for pose in pose_results:
                if pose.keypoints is not None:
                    kpts = pose.keypoints.xy[0].cpu().numpy()
                    confs = pose.keypoints.conf[0].cpu().numpy()
                    kpts[:, 0] += x1
                    kpts[:, 1] += y1
                    keypoints_xy.append(kpts)
                    keypoints_conf.append(confs)
                    
                    # Check if person is looking at camera using pose
                    if all(confs[i] > 0.3 for i in range(5)):
                        head_direction, head_tilt = calculate_head_pose(kpts)
                        if is_looking_at_camera(head_direction, head_tilt):
                            pose_looking = True
        
        # Detect gaze using MediaPipe
        gaze_looking, gaze_vectors, face_centers = gaze_detector.detect_gaze(frame)
        
        # Combine both detection methods
        current_looking = gaze_looking or pose_looking
        
        # Update looking time
        current_time = time.time()
        if current_looking:
            if not looking_at_screen:
                looking_start_time = current_time
            looking_at_screen = True
        else:
            if looking_at_screen:
                if looking_start_time is not None:
                    total_looking_time += current_time - looking_start_time
                looking_start_time = None
            looking_at_screen = False
        
        # Update display every second
        if current_time - last_update_time >= update_interval:
            last_update_time = current_time
        
        # Draw pose on frame
        annotated_frame = draw_pose(frame.copy(), keypoints_xy, keypoints_conf)
        
        # Add status text with detection method
        status = "Looking at screen"
        if looking_at_screen:
            if gaze_looking and pose_looking:
                status += " (Both methods)"
            elif gaze_looking:
                status += " (Gaze)"
            else:
                status += " (Pose)"
        else:
            status = "Not looking at screen"
            
        cv2.putText(annotated_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 255, 0) if looking_at_screen else (0, 0, 255), 2)
        
        # Add total looking time
        minutes = int(total_looking_time // 60)
        seconds = int(total_looking_time % 60)
        time_text = f"Total looking time: {minutes:02d}:{seconds:02d}"
        cv2.putText(annotated_frame, time_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('Screen Watching Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 