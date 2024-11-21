import cv2

# Path to the video file
video_path = '/Users/anonymous/Desktop/repos/EgoExoEMS/TestData/Kinect/2024-09-05-19-07-43.mkv'

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1  # Font scale
font_color = (0, 255, 0)  # Green color
font_thickness = 2  # Font thickness

# Variables for playback control
playing = False  # Initially paused
frame_delay = 30  # Delay between frames in milliseconds

# Create the display window
cv2.namedWindow('Video Frame', cv2.WINDOW_NORMAL)

def show_frame(cap, font, font_scale, font_color, font_thickness):
    """ Function to read and display the current frame with the frame number """
    ret, frame = cap.read()
    
    if not ret:
        print("Reached the end of the video.")
        return False

    # Get the current frame number
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Display the frame number on the video
    text = f"Frame: {frame_number}"
    cv2.putText(frame, text, (50, 50), font, font_scale, font_color, font_thickness)

    # Resize the frame for display
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('Video Frame', frame)
    
    return True

# Show the first frame immediately
if not show_frame(cap, font, font_scale, font_color, font_thickness):
    exit()  # Exit if there is an error showing the first frame

while True:
    if playing:
        if not show_frame(cap, font, font_scale, font_color, font_thickness):
            break
        # Wait for the specified delay for automatic playback
        key = cv2.waitKey(frame_delay) & 0xFF
    else:
        # If paused, wait indefinitely for a key press
        key = cv2.waitKey(0) & 0xFF

    # If spacebar is pressed, toggle play/pause
    if key == ord(' '):
        playing = not playing

    # Press 'q' to quit
    if key == ord('q'):
        break

    # Right arrow key (step forward one frame if paused)
    if key == ord('d') and not playing:
        if not show_frame(cap, font, font_scale, font_color, font_thickness):
            break

    # Left arrow key (rewind one frame if paused)
    if key == ord('a') and not playing:
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_pos = max(0, current_pos - 2)  # Go back by one frame (skip one due to zero-based index)
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)  # Set the new frame position

        # Display the updated frame after rewinding
        if not show_frame(cap, font, font_scale, font_color, font_thickness):
            break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
