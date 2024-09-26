import cv2
import numpy as np
import pandas as pd

# Path to sync data output file
sync_data_path = 'sync_data.csv'
# Path to the two video files
kinect_path = '/Users/kesharaw/Desktop/repos/EgoExoEMS/TestData/Kinect/2024-09-05-19-07-43.mkv'
gopro_path = '/Users/kesharaw/Desktop/repos/EgoExoEMS/TestData/GoPro/GX010321-encoded.mp4'

# Path to timestamp files
kinect_timestamp_path = '/Users/kesharaw/Desktop/repos/EgoExoEMS/TestData/Kinect/2024-09-05-19-07-43_timestamps.txt'
go_pro_timestamp_path = '/Users/kesharaw/Desktop/repos/EgoExoEMS/TestData/GoPro/GX010321_timestamps.csv'

kinect_file_id = kinect_path.split('/')[-1].split('.')[0]
gopro_file_id = gopro_path.split('/')[-1].split('.')[0]

# Read the timestamp files, kinect is a txt file and gopro is a csv file
kinect_timestamps = pd.read_csv(kinect_timestamp_path, header=None, names=['timestamp'], dtype=str)
gopro_timestamps = pd.read_csv(go_pro_timestamp_path)

# Convert Kinect timestamps to a list of strings (one per frame)
kinect_timestamps_list = kinect_timestamps['timestamp'].tolist()

# Open the video files
cap1 = cv2.VideoCapture(gopro_path)
cap2 = cv2.VideoCapture(kinect_path)

# Create a pandas dataframe to store the data
df = pd.DataFrame(columns=['gopro_file_id', 'kinect_file_id', 'sync_gopro_frame', 'sync_kinect_frame', 'sync_gopro_time', 'sync_kinect_time','sync_offset_gp-kinect'])

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Cannot open one or both video files.")
    exit()

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 3  # Font scale
font_color = (0, 255, 255)  # Yellow color
font_thickness = 7  # Font thickness

# Variables for playback control
playing_1 = False  # Initially paused for video 1
playing_2 = False  # Initially paused for video 2
frame_delay = 30  # Delay between frames in milliseconds

# Create the display window
cv2.namedWindow('Video Frame', cv2.WINDOW_NORMAL)

def show_frame(cap, font, font_scale, font_color, font_thickness):
    """ Function to read and display the current frame with the frame number """
    ret, frame = cap.read()
    
    if not ret:
        return None

    # Get the current frame number
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Display the frame number on the video
    text = f"Frame: {frame_number}"
    cv2.putText(frame, text, (50, 100), font, font_scale, font_color, font_thickness)

    # Resize the frame for display
    frame = cv2.resize(frame, (640, 360))  # Half-size to fit side by side
    
    return frame, frame_number

# Show the first frames immediately
frame1, gopro_frame_number = show_frame(cap1, font, font_scale, font_color, font_thickness)
frame2, kinect_frame_number = show_frame(cap2, font, font_scale, font_color, font_thickness)

if frame1 is None or frame2 is None:
    print("Error: Could not retrieve frames from one or both videos.")
    exit()

while True:
    if playing_1:
        frame1, gopro_frame_number = show_frame(cap1, font, font_scale, font_color, font_thickness)
        if frame1 is None:
            print("Reached the end of video 1.")
            playing_1 = False

    if playing_2:
        frame2, kinect_frame_number = show_frame(cap2, font, font_scale, font_color, font_thickness)
        if frame2 is None:
            print("Reached the end of video 2.")
            playing_2 = False

    # Combine the two frames side by side if both are available
    if frame1 is not None and frame2 is not None:
        combined_frame = np.hstack((frame1, frame2))
        cv2.imshow('Video Frame', combined_frame)

    # Wait for a key press
    key = cv2.waitKey(frame_delay if (playing_1 or playing_2) else 0) & 0xFF

    # Controls for video 1
    if key == ord(' '):  # Spacebar to toggle play/pause for video 1
        playing_1 = not playing_1
    if key == ord('d') and not playing_1:  # Step forward for video 1
        frame1, gopro_frame_number = show_frame(cap1, font, font_scale, font_color, font_thickness)
    if key == ord('a') and not playing_1:  # Rewind for video 1
        current_pos_1 = cap1.get(cv2.CAP_PROP_POS_FRAMES)
        new_pos_1 = max(0, current_pos_1 - 2)
        cap1.set(cv2.CAP_PROP_POS_FRAMES, new_pos_1)
        frame1, gopro_frame_number = show_frame(cap1, font, font_scale, font_color, font_thickness)

    # Controls for video 2
    if key == 13:  # Enter key to toggle play/pause for video 2
        playing_2 = not playing_2
    if key == ord('l') and not playing_2:  # Step forward for video 2
        frame2, kinect_frame_number = show_frame(cap2, font, font_scale, font_color, font_thickness)
    if key == ord('j') and not playing_2:  # Rewind for video 2
        current_pos_2 = cap2.get(cv2.CAP_PROP_POS_FRAMES)
        new_pos_2 = max(0, current_pos_2 - 2)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, new_pos_2)
        frame2, kinect_frame_number = show_frame(cap2, font, font_scale, font_color, font_thickness)

    # Save the current information when 't' is pressed
    if key == ord('t'):
        # Retrieve the corresponding timestamps from the timestamp files
        gopro_time = gopro_timestamps.iloc[gopro_frame_number]['epoch']
        kinect_time = kinect_timestamps_list[kinect_frame_number]  # Get the timestamp from the list
        
        # Create a new DataFrame row
        new_row = pd.DataFrame([{
            'gopro_file_id': gopro_file_id,
            'kinect_file_id': kinect_file_id,
            'sync_gopro_frame': gopro_frame_number,
            'sync_kinect_frame': kinect_frame_number,
            'sync_gopro_time': gopro_time,
            'sync_kinect_time': kinect_time,
            'sync_offset_gp-kinect': int(gopro_time) - int(kinect_time)
        }])

        # Append the new row to the DataFrame using pd.concat()
        df = pd.concat([df, new_row], ignore_index=True)

        print(f"Saved sync point: GoPro Frame: {gopro_frame_number}, Kinect Frame: {kinect_frame_number}, GoPro Time: {gopro_time}, Kinect Time: {kinect_time}")

    # Press 'q' to quit
    if key == ord('q'):
        break

# Save the collected data to a CSV file
df.to_csv(sync_data_path, index=False)

# Release the video captures and close the window
cap1.release()
cap2.release()
cv2.destroyAllWindows()
