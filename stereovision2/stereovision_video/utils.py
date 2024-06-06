import os
import pandas as pd
import cv2

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
    df = pd.DataFrame(data, columns=['Timestamp', 'X', 'Y', 'Depth'])
    base, extension = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}{extension}"
        counter += 1
    df.to_csv(new_filename, index=False)
    print(f"Data saved to {new_filename}")

def save_to_text_file(data, filename):
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
            file.write(f"{entry[0]:<20} {entry[1]:<10} {entry[2]:<10} {entry[3]:<10}\n")
    print(f"Data saved to {new_filename}")
