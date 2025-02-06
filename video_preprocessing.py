import cv2
import os
import numpy as np

def extract_frames(video_path, save_folder, frame_size=(224, 224)):
    """
    Extract frames from a video and save them as image files (NumPy arrays).
    Args:
    - video_path (str): Path to the video file.
    - save_folder (str): Directory where the frames will be saved.
    - frame_size (tuple): Desired size to resize each frame.
    """
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    
    # Create directory to save frames
    os.makedirs(save_folder, exist_ok=True)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame (if needed) and normalize pixel values to [0, 1]
        frame_resized = cv2.resize(frame, frame_size) / 255.0  # Normalize to [0, 1]
        
        # Save frame as a NumPy array
        frame_filename = os.path.join(save_folder, f"frame_{frame_count:04d}.npy")
        np.save(frame_filename, frame_resized)  # Save as NumPy array
        frame_count += 1

    # Release video capture
    cap.release()
    print(f"Extracted {frame_count} frames and saved to {save_folder}")

# Example usage
video_path = "data/raw/subway_video_1.mp4"
save_folder = "data/processed/subway_video_1"
extract_frames(video_path, save_folder)
