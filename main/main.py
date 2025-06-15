import cv2
import numpy as np
from ultralytics import YOLO
import time
import mediapipe as mp
import math
import torch

class GazeDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
        self.LEFT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
        self.RIGHT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
        self.LEFT_IRIS = [474,475,476,477]
        self.RIGHT_IRIS = [469,470,471,472]
        self.LEFT_REF = [362,263]
        self.RIGHT_REF = [33,133]

    def get_gaze_ratio(self, landmarks, eye_indices, iris_indices, ref_indices):
        eye_pts = np.array([landmarks[i] for i in eye_indices])
        iris_pts = np.array([landmarks[i] for i in iris_indices])
        ref_pts = np.array([landmarks[i] for i in ref_indices])
        center_eye = eye_pts.mean(axis=0)
        center_iris = iris_pts.mean(axis=0)
        width = np.linalg.norm(ref_pts[0] - ref_pts[1])
        vec = center_iris - center_eye
        if width > 0:
            vec = vec / width
        return vec

    def detect_gaze(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        results = self.face_mesh.process(frame_rgb)
        gaze_vectors = []
        if results.multi_face_landmarks:
            for lm in results.multi_face_landmarks:
                pts = np.array([[p.x * w, p.y * h] for p in lm.landmark])
                left = self.get_gaze_ratio(pts, self.LEFT_EYE, self.LEFT_IRIS, self.LEFT_REF)
                right = self.get_gaze_ratio(pts, self.RIGHT_EYE, self.RIGHT_IRIS, self.RIGHT_REF)
                gaze_vectors.append((left + right) / 2)
        return gaze_vectors

def calculate_head_pose(keypoints):
    nose, le, re, le_r, re_r = keypoints[:5]
    eye_center = (le + re) / 2
    head_dir = eye_center - nose
    ear_vec = re_r - le_r
    head_tilt = math.atan2(ear_vec[1], ear_vec[0])
    return head_dir, head_tilt

def is_looking_at_camera(direction, tilt, threshold=0.3):
    norm = np.linalg.norm(direction)
    if norm == 0:
        return False
    d = direction / norm
    return abs(d[0]) < threshold and d[1] > -0.1 and abs(tilt) < math.pi/4

def draw_pose(image, keypoints_xy, keypoints_conf, thickness=2):
    # existing draw_pose code unchanged
    if keypoints_xy is None or len(keypoints_xy) == 0 or keypoints_conf is None:
        return image
    skeleton = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    for kpts, confs in zip(keypoints_xy, keypoints_conf):
        for i,(x,y) in enumerate(kpts):
            if confs[i]>0.3:
                cv2.circle(image,(int(x),int(y)),3,(0,255,0),-1)
        for i,j in skeleton:
            if confs[i]>0.3 and confs[j]>0.3:
                pt1=(int(kpts[i][0]),int(kpts[i][1]))
                pt2=(int(kpts[j][0]),int(kpts[j][1]))
                cv2.line(image,pt1,pt2,(255,0,0),thickness)
    return image

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    det_model = YOLO('../models/yolo11n.pt').to(device)
    pose_model = YOLO('yolo11n-pose.pt').to(device)
    torch.set_grad_enabled(False)
    if device == 'cuda': torch.backends.cudnn.benchmark = True
    gaze_detector = GazeDetector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)
    cap.set(cv2.CAP_PROP_FPS,30)
    if not cap.isOpened(): raise IOError("Cannot open webcam")
    print("Press 'q' to quit the application")

    # initialize timers
    looking_at_screen = False
    looking_start_time = None
    total_looking_time = 0
    last_update_time = time.time()
    update_interval = 1.0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame,1)
        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        # person detection
        det_results = det_model(frame_rgb)
        keypoints_xy = []
        keypoints_conf = []
        pose_looking_count = 0
        for result in det_results:
            for box in result.boxes:
                if int(box.cls[0])==0:
                    x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
                    crop = frame_rgb[y1:y2, x1:x2]
                    for p in pose_model(crop):
                        if p.keypoints is not None:
                            kpts = p.keypoints.xy[0].cpu().numpy()
                            confs = p.keypoints.conf[0].cpu().numpy()
                            kpts[:,0]+=x1; kpts[:,1]+=y1
                            keypoints_xy.append(kpts)
                            keypoints_conf.append(confs)
                            if len(confs)>=5 and all(confs[i]>0.3 for i in range(5)):
                                dir_vec, tilt = calculate_head_pose(kpts)
                                if is_looking_at_camera(dir_vec, tilt):
                                    pose_looking_count += 1

        # gaze detection count
        gaze_vectors = gaze_detector.detect_gaze(frame)
        gaze_looking_count = sum(1 for gv in gaze_vectors if abs(gv[0])<0.15 and abs(gv[1])<0.15)

        # determine total people looking
        people_looking = max(pose_looking_count, gaze_looking_count)

        # current_looking unchanged for timer
        current_looking = (people_looking > 0)

        # update timer logic unchanged
        current_time = time.time()
        if current_looking:
            if not looking_at_screen:
                looking_start_time = current_time
            looking_at_screen = True
        else:
            if looking_at_screen and looking_start_time is not None:
                total_looking_time += current_time - looking_start_time
            looking_start_time = None
            looking_at_screen = False

        if current_time - last_update_time >= update_interval:
            last_update_time = current_time

        # draw pose skeleton
        annotated = draw_pose(frame.copy(), keypoints_xy, keypoints_conf)

        # intruder alert UI above timer
        alert_text = "safe" if people_looking<=1 else "intruder detected"
        cv2.putText(annotated, alert_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if people_looking<=1 else (0,0,255), 2)

        # existing status text
        status = "Looking at screen"
        if looking_at_screen:
            status += " (Pose+Gaze)" if pose_looking_count>0 and gaze_looking_count>0 else (" (Gaze)" if gaze_looking_count>0 else " (Pose)")
        else:
            status = "Not looking at screen"
        cv2.putText(annotated, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if looking_at_screen else (0,0,255), 2)

        # timer display unchanged
        live = total_looking_time + (current_time-looking_start_time if looking_at_screen and looking_start_time else 0)
        m = int(live//60); s = int(live%60)
        time_text = f"Total looking time: {m:02d}:{s:02d}"
        cv2.putText(annotated, time_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow('Screen Watching Detection', annotated)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
