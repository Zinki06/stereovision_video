import cv2

def get_initial_points(left_img, right_img):
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp_left, des_left = orb.detectAndCompute(left_gray, None)
    kp_right, des_right = orb.detectAndCompute(right_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_left, des_right)
    matches = sorted(matches, key=lambda x: x.distance)

    point_left = kp_left[matches[0].queryIdx].pt
    point_right = kp_right[matches[0].trainIdx].pt

    return point_left, point_right
