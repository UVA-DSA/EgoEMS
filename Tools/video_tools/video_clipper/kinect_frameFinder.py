import cv2
import os

#THIS SCRIPT DISPLAYS THE KINECT CAMERA FRAMES, ASSUMING VIDEO AND ASSOCIATED TXT FILE
#HAVE BEEN MOVED INTO A LOCAL DIRECTORY
#PRESS 'S' TO GET CURRENT FRAME TIMESTAMP PRINTED 

data_folder = r"C:\Users\anonymous\Desktop\ng_sep_raw\05-09-2024"
person = r"anonymous"
trial = r"0"
general_dir = os.path.join(data_folder,person, trial)

for file in os.listdir(general_dir):
    if file.endswith(".mkv"):
        video = os.path.join(general_dir, file)
    if file.endswith(".txt"):
        txt = os.path.join(general_dir, file)



# Load the text file into a list
with open(txt, 'r') as file:
    text_lines = file.readlines()

# Initialize video capture
video_capture = cv2.VideoCapture(video)

# Get the total number of frames in the video
video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {video_length}")

# count = 0
# while True:
#     ret, frame = video_capture.read()
#     if not ret:
#         break
#     count += 1

video_capture.release()
cv2.destroyAllWindows()

# Reinitialize video capture to process frames
video_capture = cv2.VideoCapture(video)

count = 0
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    count += 1

    # Display frame with current frame count overlay
    if count % 1 == 0 or count == video_length:
        cv2.putText(frame, f"Frame: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Frame', frame)

        # Wait for a key press
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Display the value from the text file corresponding to the current frame count
            if count - 1 < len(text_lines):
                print(f"Row {count}: {text_lines[count - 1].strip()}")
            else:
                print(f"Row {count} is out of range in the text file.")

video_capture.release()
cv2.destroyAllWindows()
