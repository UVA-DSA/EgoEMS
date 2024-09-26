import cv2
import os
from moviepy.editor import VideoFileClip

#THIS FILE DISPLAYS THE GOPRO VIDEO (MP4) WITH FRAME NUMBER FOR VISUAL SYNCHRONIZATION
#FILE DIRECTORY ASSUMES FILES HAVE BEEN DOWNLOADED TO LOCAL SPACE WITH SAME ORGANIZATION AS SCRATCH

# Open the video file
data_folder = r"C:\Users\vht2gm\Desktop\ng_sep_raw\05-09-2024"
person = r"bryan"
trial = r"0"
general_dir = os.path.join(data_folder,person, trial)

for file in os.listdir(general_dir):
    if file.endswith(".MP4"):
        video_path = os.path.join(general_dir, file)
cap = cv2.VideoCapture(video_path)

skip_frames = 1

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0
start_frame = 0
end_frame = frame_count - 1

# Get the base name of the video file without extension
base_name = os.path.splitext(os.path.basename(video_path))[0]
print(f"Processing Video: {base_name} ({frame_count} frames)")

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4

def display_frame(frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {frame_number}.")
        return
    cv2.putText(frame, f'Frame: {frame_number}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('Video', frame)

def save_clip_with_audio(start_frame, end_frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"Saving clip .", end='', flush=True)

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {start_frame}.")
        return
    
    output_file = f"{base_name}_clipped_with_audio_start_{start_frame}_end_{end_frame}.mp4"
    
    # Use moviepy to handle video and audio
    with VideoFileClip(video_path) as video:
        fps = video.fps
        start_time = (start_frame / fps)
        end_time = (end_frame / fps)
        
        clip = video.subclip(start_time, end_time)
        clip.write_videofile(output_file, codec='libx264', audio_codec='aac')
    
    print(f"\nClip saved as '{output_file}'")

def save_frame(current_frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {current_frame}.")
        return
    
    cv2.imwrite(f"{base_name}_frame_{current_frame}.jpg", frame)

# Initial display
display_frame(current_frame)

while True:
    key = cv2.waitKey(0)
    if key == 27:  # ESC key to exit
        break
    elif key == ord('a'):  # a key
        current_frame = max(0, current_frame - skip_frames)
        display_frame(current_frame)
    elif key == ord('d'):  # d key
        current_frame = min(frame_count - 1, current_frame + skip_frames)
        display_frame(current_frame)
    elif key == ord('s'):  # 's' key to set start frame
        start_frame = current_frame
        print(f"Start frame set to {start_frame}")
    elif key == ord('e'):  # 'e' key to set end frame
        end_frame = current_frame
        print(f"End frame set to {end_frame}")
    elif key == ord('c'):  # 'c' key to save the clip
        save_clip_with_audio(start_frame, end_frame)
    elif key == ord('f'):  # 'f' key to save the frame
        save_frame(current_frame)

cap.release()
cv2.destroyAllWindows()
