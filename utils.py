import os
import pandas as pd
import cv2
import numpy as np

def equalize_histogram(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    return frame_gray

def resize_to_match(frame1, frame2):
    height, width = frame1.shape[:2]
    frame2_resized = cv2.resize(frame2, (width, height), interpolation=cv2.INTER_AREA)
    return frame1, frame2_resized

def ensure_output_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def save_to_csv(data, filename):
    """
    Save tracking data to a CSV file. If the file already exists, create a new file with an incremented counter.
    """
    df = pd.DataFrame(data, columns=['Timestamp', 'X', 'Y', 'Depth', 'Frame Left', 'Frame Right', 'Disparity'])
    base, extension = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}{extension}"
        counter += 1
    df.to_csv(new_filename, index=False)
    print(f"Data saved to {new_filename}")

def save_to_text_file(data, filename):
    """
    Save tracking data to a text file. If the file already exists, create a new file with an incremented counter.
    """
    base, extension = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}{extension}"
        counter += 1

    with open(new_filename, 'w') as file:
        file.write(f"{'Timestamp':<20} {'X':<10} {'Y':<10} {'Depth':<10}\n")
        file.write("="*60 + "\n")
        for entry in data:
            timestamp, x, y, depth, _, _, _ = entry
            file.write(f"{timestamp:<20} {x:<10} {y:<10} {depth:<10}\n")
    print(f"Data saved to {new_filename}")

def save_images(left_img, right_img, disparity, output_folder, frame_idx):
    ensure_output_folder(output_folder)
    left_filename = os.path.join(output_folder, f"left_{frame_idx}.png")
    right_filename = os.path.join(output_folder, f"right_{frame_idx}.png")
    disparity_filename = os.path.join(output_folder, f"disparity_{frame_idx}.png")

    cv2.imwrite(left_filename, left_img)
    cv2.imwrite(right_filename, right_img)
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(disparity_filename, disparity_normalized)
    print(f"Images saved to {output_folder} with index {frame_idx}")