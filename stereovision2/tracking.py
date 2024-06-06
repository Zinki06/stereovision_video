import cv2
import time
import numpy as np

def initialize_tracker(frame, point):
    tracker = cv2.TrackerCSRT_create()
    bbox = (int(point[0]) - 10, int(point[1]) - 10, 20, 20)
    tracker.init(frame, bbox)
    return tracker

def update_tracker(frame, point):
    tracker = cv2.TrackerCSRT_create()
    bbox = (int(point[0]) - 10, int(point[1]) - 10, 20, 20)
    tracker.init(frame, bbox)
    return tracker

def track_object(tracker, frame, frame_left_gray, frame_right_gray, point_left, point_right):
    success, bbox = tracker.update(frame)
    if success:
        x, y, w, h = [int(v) for v in bbox]
        center_left = (x + w // 2, y + h // 2)
        cv2.circle(frame, center_left, 5, (0, 255, 0), -1)

        center_right = center_left
        stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(frame_left_gray, frame_right_gray).astype(np.float32) / 16.0
        disparity_value = np.abs(center_left[0] - center_right[0])
        
        B = 0.1  # 기준선 (미터)
        f = 0.02  # 초점 거리 (미터)
        depth = (B * f) / disparity_value if disparity_value != 0 else 0

        timestamp = time.time()
        return success, timestamp, center_left[0], center_left[1], depth

    return False, None, None, None, None
