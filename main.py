import cv2
import time
import pandas as pd
import os
import numpy as np
from sync import synchronize_videos
from feature_matching import get_initial_points
from tracking import initialize_tracker, track_object, update_tracker
from utils import equalize_histogram, resize_to_match, ensure_output_folder, save_to_csv, save_to_text_file

OUTPUT_FOLDER = 'stereovision2/output'
selected_point = None
data = []

def main():
    global selected_point
    # 비디오 파일 로드 및 확인
    cap_left, cap_right = load_videos('stereovision2/target/3_LEFT.mp4', 'stereovision2/target/3_RIGHT.mp4')

    if not cap_left or not cap_right:
        return

    # 비디오 동기화
    cap_left, cap_right, lag = synchronize_videos(cap_left, cap_right)
    print(f"Videos synchronized with lag: {lag}")

    # 첫 번째 프레임 가져오기
    ret_left, left_img = cap_left.read()
    ret_right, right_img = cap_right.read()
    if not ret_left or not ret_right:
        print("Error: Could not read initial frames.")
        return

    # 이미지 표시 및 포인트 선택
    cv2.imshow('Select Object', left_img)
    cv2.setMouseCallback('Select Object', select_point)
    print("Click on the object to track and press 'Enter' or close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if selected_point is None:
        print("No point selected.")
        return

    point_left = selected_point

    # 추적기 초기화
    tracker = initialize_tracker(left_img, point_left)
    print("Tracker initialized successfully.")

    # 데이터 기록
    track_objects(cap_left, cap_right, tracker, point_left)

    # 출력 폴더가 없으면 생성
    ensure_output_folder(OUTPUT_FOLDER)

    # CSV 및 텍스트 파일로 저장
    csv_path = os.path.join(OUTPUT_FOLDER, 'tracked_coordinates.csv')
    txt_path = os.path.join(OUTPUT_FOLDER, 'tracked_coordinates.txt')
    save_to_csv(data, csv_path)
    save_to_text_file(data, txt_path)

    # 결과 이미지 저장 폴더 생성
    result_folder = os.path.splitext(txt_path)[0]
    ensure_output_folder(result_folder)

    # 결과 이미지 저장
    for idx, frame_data in enumerate(data):
        save_stereo_images(frame_data[4], frame_data[5], frame_data[6], result_folder, idx)

def load_videos(left_video_path, right_video_path):
    cap_left = cv2.VideoCapture(left_video_path)
    cap_right = cv2.VideoCapture(right_video_path)

    if not cap_left.isOpened():
        print(f"Error: Could not open left video file {left_video_path}.")
        return None, None
    if not cap_right.isOpened():
        print(f"Error: Could not open right video file {right_video_path}.")
        return None, None

    print("Videos loaded successfully.")
    return cap_left, cap_right

def select_point(event, x, y, flags, param):
    global selected_point
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = (x, y)
        print(f"Point selected: {selected_point}")

def track_objects(cap_left, cap_right, tracker, point_left):
    global selected_point, data
    frame_idx = 0
    while cap_left.isOpened() and cap_right.isOpened():
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or frame_left is None:
            print("Warning: Left frame is empty.")
            break
        if not ret_right or frame_right is None:
            print("Warning: Right frame is empty.")
            break

        frame_left, frame_right = resize_to_match(frame_left, frame_right)
        frame_left_gray = equalize_histogram(frame_left)
        frame_right_gray = equalize_histogram(frame_right)

        # 깊이 맵 생성
        disparity = compute_disparity(frame_left_gray, frame_right_gray)
        depth_map = compute_depth_map(disparity)

        cv2.imshow('Disparity', (disparity - disparity.min()) / (disparity.max() - disparity.min()))
        cv2.imshow('Depth Map', (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()))

        success, timestamp, x, y, depth = track_object(tracker, frame_left, frame_left_gray, disparity, point_left)
        if success:
            data.append([timestamp, x, y, depth, frame_left, frame_right, disparity])
            print(f"Tracked point at time {timestamp}: ({x}, {y}, Depth: {depth})")
        else:
            print(f"Object lost at time {timestamp}. Click on the object to reselect or press 'Enter' to skip.")

            cv2.imshow('Left Video', frame_left)
            selected_point = None  # Reset the selected point
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    cap_left.release()
                    cap_right.release()
                    cv2.destroyAllWindows()
                    return
                elif key == 13:  # Enter key
                    break
                elif selected_point is not None:  # Point has been selected
                    tracker = update_tracker(frame_left, selected_point)
                    point_left = selected_point
                    selected_point = None  # Reset the selected point after updating tracker
                    break

        cv2.imshow('Left Video', frame_left)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

def compute_disparity(left_gray, right_gray):
    num_disparities = 64  # 16의 배수로 설정
    block_size = 15       # 블록 크기 설정
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    stereo.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
    stereo.setPreFilterSize(9)
    stereo.setPreFilterCap(31)
    stereo.setMinDisparity(0)
    stereo.setTextureThreshold(10)
    stereo.setUniquenessRatio(15)
    stereo.setSpeckleRange(32)
    stereo.setSpeckleWindowSize(100)
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disparity

def compute_depth_map(disparity):
    B = 0.1  # meter
    f = 0.02  # meter
    depth_map = (B * f) / (disparity + 1e-6)  # Add small value to avoid division by zero
    depth_map[disparity <= 0] = 0  # Set depth to 0 where disparity is invalid
    return depth_map

def save_stereo_images(left_img, right_img, disparity, output_folder, frame_idx):
    ensure_output_folder(output_folder)
    cv2.imwrite(os.path.join(output_folder, f"left_{frame_idx}.png"), left_img)
    cv2.imwrite(os.path.join(output_folder, f"right_{frame_idx}.png"), right_img)
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(output_folder, f"disparity_{frame_idx}.png"), disparity_normalized)

if __name__ == "__main__":
    main()