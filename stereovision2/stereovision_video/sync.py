import cv2
import numpy as np
from scipy.signal import correlate

def synchronize_videos(cap_left, cap_right):
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        return None, None, None

    frame_left_gray = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    frame_right_gray = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    correlation = correlate(frame_left_gray.flatten(), frame_right_gray.flatten())
    lag = np.argmax(correlation) - (len(frame_left_gray.flatten()) - 1)

    return cap_left, cap_right, lag
