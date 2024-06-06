import cv2
import time
import pandas as pd
import os
from sync import synchronize_videos
from feature_matching import get_initial_points
from tracking import initialize_tracker, track_object
from utils import equalize_histogram, resize_to_match, ensure_output_folder, save_to_csv, save_to_text_file

OUTPUT_FOLDER = 'stereovision2/output'
selected_point = None

def main():
    global selected_point
    # 비디오 파일 로드 및 확인
    cap_left, cap_right = load_videos('stereovision2/target/IMG_0257.MOV', 'stereovision2/target/IMG_1054.MOV')
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
    data = track_objects(cap_left, cap_right, tracker, point_left)
    if not data:
        print("No data tracked.")
        return

    # 출력 폴더가 없으면 생성
    ensure_output_folder(OUTPUT_FOLDER)

    # CSV 및 텍스트 파일로 저장
    save_to_csv(data, os.path.join(OUTPUT_FOLDER, 'tracked_coordinates.csv'))
    save_to_text_file(data, os.path.join(OUTPUT_FOLDER, 'tracked_coordinates.txt'))

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
    data = []
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

        success, timestamp, x, y, depth = track_object(tracker, frame_left, frame_left_gray, frame_right_gray, point_left, point_left)
        if success:
            data.append([timestamp, x, y, depth])
            print(f"Tracked point at time {timestamp}: ({x}, {y}, Depth: {depth})")
        else:
            print(f"Object lost at time {timestamp}")

        cv2.imshow('Left Video', frame_left)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
    return data

if __name__ == "__main__":
    main()
