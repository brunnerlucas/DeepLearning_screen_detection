import cv2
import numpy as np
import torch
from ultralytics import YOLO

def draw_pose(image, keypoints_xy, keypoints_conf, thickness=2):
    if keypoints_xy is None or len(keypoints_xy) == 0 or keypoints_conf is None:
        return image

    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8),
        (8, 10), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    for kpts, confs in zip(keypoints_xy, keypoints_conf):
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
    device_det = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device_pose = 'cpu'  # <-- FORCE POSE MODEL TO CPU
    # Load YOLO detection and pose models
    det_model = YOLO('yolo11x.pt')
    pose_model = YOLO('yolo11x-pose.pt')

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    print("Press 'q' to quit the application")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect people
        det_results = det_model.predict(source=frame_rgb, device=device_det, verbose=False)
        bboxes = []

        for result in det_results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # Person class
                    bboxes.append(box.xyxy[0].cpu().numpy())

        keypoints_xy = []
        keypoints_conf = []
        annotated_frame = frame.copy()

        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            person_crop = frame_rgb[y1:y2, x1:x2]

            pose_results = pose_model.predict(source=person_crop, device=device_pose, verbose=False)


            for pose in pose_results:
                if pose.keypoints is not None:
                    kpts = pose.keypoints.xy[0].cpu().numpy()
                    confs = pose.keypoints.conf[0].cpu().numpy()
                    kpts[:, 0] += x1
                    kpts[:, 1] += y1
                    keypoints_xy.append(kpts)
                    keypoints_conf.append(confs)

                    # Looking-at-screen heuristic
                    if confs[0] > 0.3 and confs[1] > 0.3 and confs[2] > 0.3 and confs[5] > 0.3 and confs[6] > 0.3:
                        nose = kpts[0]
                        left_eye = kpts[1]
                        right_eye = kpts[2]
                        left_shoulder = kpts[5]
                        right_shoulder = kpts[6]

                        eye_level_diff = abs(left_eye[1] - right_eye[1])
                        shoulder_center = (left_shoulder[0] + right_shoulder[0]) / 2
                        nose_offset = abs(nose[0] - shoulder_center)

                        if eye_level_diff < 20 and nose_offset < 50:
                            cv2.putText(annotated_frame, "Looking at screen ✅", (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        else:
                            cv2.putText(annotated_frame, "Not looking ❌", (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        annotated_frame = draw_pose(annotated_frame, keypoints_xy, keypoints_conf)
        cv2.imshow('Pose Estimation & Gaze Check', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
