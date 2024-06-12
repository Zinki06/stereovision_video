import cv2
import time

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

def track_object(tracker, frame, frame_gray, disparity, point):
    success, bbox = tracker.update(frame)
    if success:
        x, y, w, h = [int(v) for v in bbox]
        center_x = x + w // 2
        center_y = y + h // 2
        depth = compute_depth(disparity, center_x, center_y)
        timestamp = time.time()
        return success, timestamp, center_x, center_y, depth

    return False, None, None, None, None

def compute_depth(disparity, x, y):
    disparity_value = disparity[y, x]
    B = 0.1  # Baseline (meter)
    f = 0.02  # Focal length (meter)
    depth= (B * f) / disparity_value if disparity_value != 0 else 0
    return depth