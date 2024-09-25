# %%
import cv2

# Path to the video file
video_path = '/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/Sep_2024/Raw/05-09-2024/debrah/cardiac_arrest/3/GoPro/GX010335.MP4'

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 3  # Adjust the size of the font
font_color = (0, 255, 0)  # Green color
font_thickness = 5  # Thickness of the text

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Reached the end of the video or error occurred.")
        break

    # Get the current frame number
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Display the frame number on the video (top-left corner)
    text = f"Frame: {frame_number}"
    cv2.putText(frame, text, (50, 100), font, font_scale, font_color, font_thickness)

    # Show the frame with the frame number
    cv2.imshow('Video Frame', frame)

    # Wait for a key press, 0 means wait indefinitely
    key = cv2.waitKey(0) & 0xFF

    # Right arrow key (to go to the next frame)
    if key == ord('d'):
        continue

    # Left arrow key (rewind a frame)
    if key == ord('a'):
        # Move one frame back
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 2)
        continue

    # Press 'q' to quit the viewer
    if key == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()



