import cv2
import numpy as np
import pandas as pd
import os

base_path = '/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/Sep_2024/Raw'  # Update with your actual path
# Load the CSV file
csv_file_path = f'{base_path}/file_paths_for_sync.csv'
csv_data = pd.read_csv(csv_file_path)

# Index to track the current video
current_video_index = 0

# Function to load video paths from the CSV based on the current index
def load_video_paths(index):
    if index < len(csv_data):
        row = csv_data.iloc[index]
        gopro_path = row['gopro_file_path']
        kinect_path = row['kinect_file_path']
        gopro_file_id = os.path.basename(gopro_path).split('.')[0]  # Extract file name without extension
        kinect_file_id = os.path.basename(kinect_path).split('.')[0]  # Extract file name without extension
        gopro_timestamp_path = f'{os.path.dirname(gopro_path)}/{gopro_file_id.split("_")[0]}.csv'
        kinect_timestamp_path = f'{os.path.dirname(kinect_path)}/ts.txt'
        return gopro_path, kinect_path, gopro_file_id, kinect_file_id, gopro_timestamp_path, kinect_timestamp_path
    else:
        print("No more videos to process.")
        return None, None, None, None, None, None

# Load the first set of video paths and IDs
gopro_path, kinect_path, gopro_file_id, kinect_file_id, gopro_timestamp_path,kinect_timestamp_path  = load_video_paths(current_video_index)

# Read the timestamp files, kinect is a txt file and gopro is a csv file
kinect_timestamps = pd.read_csv(kinect_timestamp_path, header=None, names=['timestamp'], dtype=str)
gopro_timestamps = pd.read_csv(gopro_timestamp_path)

# Convert Kinect timestamps to a list of strings (one per frame)
kinect_timestamps_list = kinect_timestamps['timestamp'].tolist()

if gopro_path is None or kinect_path is None:
    exit()  # Exit if no valid video paths

# Load the video files
cap1 = cv2.VideoCapture(gopro_path)
cap2 = cv2.VideoCapture(kinect_path)

# Path to the existing CSV file for sync data
csv_path = f'{base_path}/sync_offset_data.csv'  # Update with your actual path

# Load existing CSV file or create a new DataFrame if it doesn't exist
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    df = pd.DataFrame(columns=['gopro_file_id', 'kinect_file_id', 'sync_gopro_frame', 'sync_kinect_frame', 'sync_gopro_time', 'sync_kinect_time'])

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.5  # Font scale for titles
font_color = (0, 255, 255)  # Yellow color
font_thickness = 3  # Font thickness for titles

# Variables for playback control
playing_1 = False  # Initially paused for video 1
playing_2 = False  # Initially paused for video 2
frame_delay = 30  # Delay between frames in milliseconds

# Create the display window
cv2.namedWindow('Video Frame', cv2.WINDOW_NORMAL)

def show_frame(cap, font, font_scale, font_color, font_thickness, file_id):
    """ Function to read and display the current frame with the frame number and file ID """
    ret, frame = cap.read()
    
    if not ret:
        return None, None

    # Get the current frame number
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Display the frame number on the video
    text = f"Frame: {frame_number}"
    cv2.putText(frame, text, (50, 50), font, font_scale, font_color, font_thickness)

    # Calculate text size for the file ID
    text_size = cv2.getTextSize(file_id, font, font_scale, font_thickness)[0]

    # Calculate position to center the file ID at the bottom middle
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = frame.shape[0] - 20  # 20 pixels above the bottom

    # Display the file ID at the bottom middle of the frame
    cv2.putText(frame, file_id, (text_x, text_y), font, font_scale, font_color, font_thickness)

    # Resize the frame for display
    frame = cv2.resize(frame, (640, 360))  # Half-size to fit side by side
    
    return frame, frame_number

# Show the first frames immediately
frame1, gopro_frame_number = show_frame(cap1, font, font_scale, font_color, font_thickness, gopro_file_id)
frame2, kinect_frame_number = show_frame(cap2, font, font_scale, font_color, font_thickness, kinect_file_id)

if frame1 is None or frame2 is None:
    print("Error: Could not retrieve frames from one or both videos.")
    exit()

while True:
    if playing_1:
        frame1, gopro_frame_number = show_frame(cap1, font, font_scale, font_color, font_thickness, gopro_file_id)
        if frame1 is None:
            print("Reached the end of video 1.")
            playing_1 = False

    if playing_2:
        frame2, kinect_frame_number = show_frame(cap2, font, font_scale, font_color, font_thickness, kinect_file_id)
        if frame2 is None:
            print("Reached the end of video 2.")
            playing_2 is False

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
        frame1, gopro_frame_number = show_frame(cap1, font, font_scale, font_color, font_thickness, gopro_file_id)
    if key == ord('a') and not playing_1:  # Rewind for video 1
        current_pos_1 = cap1.get(cv2.CAP_PROP_POS_FRAMES)
        new_pos_1 = max(0, current_pos_1 - 2)
        cap1.set(cv2.CAP_PROP_POS_FRAMES, new_pos_1)
        frame1, gopro_frame_number = show_frame(cap1, font, font_scale, font_color, font_thickness, gopro_file_id)

    # Controls for video 2
    if key == 13:  # Enter key to toggle play/pause for video 2
        playing_2 = not playing_2
    if key == ord('l') and not playing_2:  # Step forward for video 2
        frame2, kinect_frame_number = show_frame(cap2, font, font_scale, font_color, font_thickness, kinect_file_id)
    if key == ord('j') and not playing_2:  # Rewind for video 2
        current_pos_2 = cap2.get(cv2.CAP_PROP_POS_FRAMES)
        new_pos_2 = max(0, current_pos_2 - 2)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, new_pos_2)
        frame2, kinect_frame_number = show_frame(cap2, font, font_scale, font_color, font_thickness, kinect_file_id)

    # Save the current information when 't' is pressed and load the next video
    if key == ord('t'):

                # Retrieve the corresponding timestamps from the timestamp files
        gopro_time = gopro_timestamps.iloc[gopro_frame_number]['epoch']
        kinect_time = kinect_timestamps_list[kinect_frame_number]  # Get the timestamp from the list
        
                # Check if a row with the same gopro_file_id already exists in the DataFrame
        existing_row_index = df[df['gopro_file_id'] == gopro_file_id].index

        if not existing_row_index.empty:
            # Update the existing row
            df.loc[existing_row_index, 'sync_gopro_frame'] = gopro_frame_number
            df.loc[existing_row_index, 'sync_kinect_frame'] = kinect_frame_number
            df.loc[existing_row_index, 'sync_gopro_time'] = gopro_time
            df.loc[existing_row_index, 'sync_kinect_time'] = kinect_time
            df.loc[existing_row_index, 'sync_offset_gp-kinect'] = int(gopro_time) - int(kinect_time)
            print(f"Updated sync point for {gopro_file_id}: GoPro Frame: {gopro_frame_number}, Kinect Frame: {kinect_frame_number}, GoPro Time: {gopro_time}, Kinect Time: {kinect_time}")
        else:
            # Append a new row to the DataFrame
            new_row = pd.DataFrame([{
                'gopro_file_id': gopro_file_id,
                'kinect_file_id': kinect_file_id,
                'sync_gopro_frame': gopro_frame_number,
                'sync_kinect_frame': kinect_frame_number,
                'sync_gopro_time': gopro_time,
                'sync_kinect_time': kinect_time,
                'sync_offset_gp-kinect': int(gopro_time) - int(kinect_time)

            }])
            df = pd.concat([df, new_row], ignore_index=True)
            print(f"Saved new sync point: GoPro Frame: {gopro_frame_number}, Kinect Frame: {kinect_frame_number}, GoPro Time: {gopro_time}, Kinect Time: {kinect_time}")

        df['sync_offset_gp-kinect'] = df['sync_offset_gp-kinect'].astype(int)
        df.to_csv(csv_path, index=False)
        print(f"Sync data saved to {csv_path}")
        # Save data here (add your sync logic)
        print("Moving to next video...")

        # Increment the video index and load the next set of videos
        current_video_index += 1
        gopro_path, kinect_path, gopro_file_id, kinect_file_id, gopro_timestamp_path,kinect_timestamp_path = load_video_paths(current_video_index)

        if gopro_path is None or kinect_path is None:
            print("No more videos.")
            break  # Exit if no more videos

        # Release the current video captures
        cap1.release()
        cap2.release()

        # Load the next videos
        cap1 = cv2.VideoCapture(gopro_path)
        cap2 = cv2.VideoCapture(kinect_path)

        # Show the first frames of the new videos
        frame1, gopro_frame_number = show_frame(cap1, font, font_scale, font_color, font_thickness, gopro_file_id)
        frame2, kinect_frame_number = show_frame(cap2, font, font_scale, font_color, font_thickness, kinect_file_id)

    # Press 'q' to quit
    if key == ord('q'):
        break

# Release the video captures and close the window
cap1.release()
cap2.release()
cv2.destroyAllWindows()
