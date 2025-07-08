from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
import numpy as np
import cv2
import argparse
import json
import os

def crop_img_bb(img, hand_bb, pad, show=False):
    if len(img.shape) == 2:
        h, w = img.shape
    elif len(img.shape) == 3:
        h, w, _ = img.shape
    else:
        raise Exception("Invalid image shape")
    hand_bb = [int(bb) for bb in hand_bb]
    img_crop = img[max(0, hand_bb[1] - pad):min(h, hand_bb[3] + pad), max(0, hand_bb[0] - pad):min(w, hand_bb[2] + pad)]
    if show:
        cv2.imshow("Image with Bounding Box", img_crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img_crop


def get_bb(results):
    bbx_list = results.xyxy
    conf_list = results.confidence
    if len(conf_list) == 0:
        return []
    max_conf_arg = np.argmax(conf_list)
    bb = bbx_list[max_conf_arg]
    return bb


class WristDet_mediapipe:
    def __init__(self):
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=4,
            min_detection_confidence=0.6
        )

    ### This function is used to get the keypoints of the closest hand wrist
    # def get_kypts(self, image):
    #     height, width, _ = image.shape
    #     results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #     xy_vals = []
    #     z_vals = []
    #     if results.multi_hand_landmarks:
    #         for hand_landmarks in results.multi_hand_landmarks:
    #             for i in range(21):
    #                 x = int(hand_landmarks.landmark[i].x * width)
    #                 y = int(hand_landmarks.landmark[i].y * height)
    #                 z = hand_landmarks.landmark[i].z
    #                 xy_vals.append((x, y))
    #                 z_vals.append(z)
    #         if len(xy_vals) == 42:
    #             closest_hand = np.argmin([z_vals[0], z_vals[21]])
    #             start_coord = 0 if closest_hand == 0 else 21
    #             xy_vals = xy_vals[start_coord:start_coord + 21]
    #     return image, xy_vals

    ### This function is used to get the keypoints of all hands
    def get_kypts(self, image):
        height, width, _ = image.shape
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        all_hands = []  # To store landmarks for all hands

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                xy_vals = []
                for i in range(21):
                    x = int(hand_landmarks.landmark[i].x * width)
                    y = int(hand_landmarks.landmark[i].y * height)
                    xy_vals.append((x, y))
                all_hands.append(xy_vals)  # Add current hand's landmarks to the list

        return image, all_hands


### This function is used to get the keypoints of the closest hand wrist
# def get_kpts(img, wrst, base_model):
#     results = base_model.predict(img)
#     bb = get_bb(results)

#     if bb is None or len(bb) == 0:  # Check if bounding box is empty
#         print("No bounding box detected.")
#         return {"x": [], "y": []}  # Return empty keypoints
    
#     pad = 80
#     img_crop = crop_img_bb(img, bb, pad, show=False)
#     image, xy_vals = wrst.get_kypts(img_crop)

#     if not xy_vals:  # Check if keypoints are detected
#         print("No keypoints detected.")
#         return {"x": [], "y": []}  # Return empty keypoints

#     x_vals = [int(val[0] + bb[0] - pad) for val in xy_vals]
#     y_vals = [int(val[1] + bb[1] - pad) for val in xy_vals]
#     kpt_dict = {"x": x_vals, "y": y_vals}
#     return kpt_dict

### This function is used to get the keypoints of all hands
def get_kpts(img, wrst, base_model, frame_num, video_id):
    results = base_model.predict(img)
    bb = get_bb(results)

    if bb is None or len(bb) == 0:  # Check if bounding box is empty
        print("No bounding box detected.")
        return {"hands": []}  # Return empty list for hands
    
    pad = 80
    img_crop = crop_img_bb(img, bb, pad, show=False)


    image, all_hands = wrst.get_kypts(img_crop)


    hands_data = []
    for hand in all_hands:
        x_vals = [int(val[0] + bb[0] - pad) for val in hand]
        y_vals = [int(val[1] + bb[1] - pad) for val in hand]
        hands_data.append({"x": x_vals, "y": y_vals})

    # visualize the cropped image with keypoints
    if len(hands_data) > 0:
        for hand in hands_data:
            img = draw_keypoints(img, hand["x"], hand["y"])

        # save the cropped image as debug with a unique name for each frame
        cv2.imwrite(f"./debug/{video_id}_{frame_num}.jpg", img)

    return {"hands": hands_data}



def process_video(video_path, output_json_path, wrst, base_model, video_id):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    keypoints_data = {}
    frame_num = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # resize the frame to 640x480
        frame = cv2.resize(frame, (640, 480))

        # Get keypoints for the current frame
        kpt_dict = get_kpts(frame, wrst, base_model, frame_num, video_id)

        # Store keypoints with the frame number
        keypoints_data[frame_num] = kpt_dict

        # Print or process keypoints (optional)
        print(f"Frame {frame_num}: {kpt_dict}")
        frame_num += 1

    # Save keypoints to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(keypoints_data, json_file, indent=4)

    cap.release()


def draw_keypoints(frame, x_vals, y_vals):
    """Draw keypoints on the frame."""
    for x, y in zip(x_vals, y_vals):
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green circles for keypoints
    return frame


def process_and_visualize_video(video_path, output_json_path, output_video_path, wrst, base_model):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get the video frame width, height, and FPS to create VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter object to save the video with keypoints
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    keypoints_data = {}
    frame_num = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Get keypoints for the current frame
        kpt_dict = get_kpts(frame, wrst, base_model)
        
        # Draw keypoints on the frame
        if kpt_dict["x"] and kpt_dict["y"]:
            frame_with_kpts = draw_keypoints(frame, kpt_dict["x"], kpt_dict["y"])
        else:
            frame_with_kpts = frame  # No keypoints, use the original frame

        # Store keypoints with the frame number
        keypoints_data[frame_num] = kpt_dict

        # Save the frame with keypoints to the output video
        out.write(frame_with_kpts)

        # Print or process keypoints (optional)
        # print(f"Frame {frame_num}: {kpt_dict}")
        frame_num += 1

    # Save keypoints to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(keypoints_data, json_file, indent=4)

    # Release video objects
    cap.release()
    out.release()




def process_videos_in_directory(root_path, wrst, base_model, view):
    # Recursively search for videos inside 'GoPro' subdirectories
    flag = False

    videos_to_process = []


    for dirpath, _, filenames in os.walk(root_path):
        if view == "exo": 
            if 'chest_compressions' in dirpath:
                for filename in filenames:
                    if filename.endswith(('.mkv')):  # You can add more video formats if needed
                        print("=" * 60)
                        print("*" * 60)
                        print(f"Processing directory: {dirpath}")

                        print(f"Processing video: {filename}")
                        video_path = os.path.join(dirpath, filename)
                        video_id = filename.replace('.mp4', '')  # Remove the file extension to get the video ID
                        output_json_path = os.path.join(dirpath, f'{video_id}_keypoints.json')
                        output_video_path = os.path.join(dirpath, f'{video_id}_keypoints.mp4')

                        print(f"Output JSON path: {output_json_path}")

                        # Process the video and save keypoints to JSON and a new video
                        # process_and_visualize_video(video_path, output_json_path, output_video_path, wrst, base_model)
            
                        # Process the video and save keypoints to JSON

                        # check if the output JSON file already exists
                        if os.path.exists(output_json_path):
                            print(f"Output JSON file {output_json_path} already exists. Skipping processing for {video_path}.")
                            continue


                        process_video(video_path, output_json_path, wrst, base_model, video_id)
                        print("*" * 60)
                        print("=" * 60)
        else:
            for filename in filenames:
                if filename.endswith(('.mp4')):  # You can add more video formats if needed
                    print("=" * 60)
                    print("*" * 60)
                    print(f"Processing directory: {dirpath}")

                    print(f"Processing video: {filename}")
                    video_path = os.path.join(dirpath, filename)
                    videos_to_process.append(video_path)

                    # video_id = filename.replace('.mp4', '')  # Remove the file extension to get the video ID
                    # output_json_path = os.path.join(dirpath, f'{video_id}_resized_640x480_keypoints.json')
                    # output_video_path = os.path.join(dirpath, f'{video_id}_keypoints.mp4')

                    # if os.path.exists(output_json_path):
                    #     print(f"Output JSON file {output_json_path} already exists. Skipping processing for {video_path}.")
                    #     continue

                    # Process the video and save keypoints to JSON and a new video
                    # process_and_visualize_video(video_path, output_json_path, output_video_path, wrst, base_model)
          
                    # Process the video and save keypoints to JSON
                    # process_video(video_path, output_json_path, wrst, base_model, video_id)
                    print("*" * 60)
                    print("=" * 60)

    # save the list of videos to process in a text file
    if videos_to_process:
        with open(os.path.join(root_path, 'videos_to_process.txt'), 'w') as f:
            for video in videos_to_process:
                f.write(video + '\n')
        print(f"List of videos to process saved to {os.path.join(root_path, 'videos_to_process.txt')}")




def process_videos_in_list(
    root_path,
    wrst,
    base_model,
    view,
    video_list_txt: str = None
):
    """
    Either:
      - Read `video_list_txt`, expecting one full path per line,
        and process only those videos
    Or:
      - Walk `root_path` and auto-discover videos exactly as before.
    """
    videos_to_process = []




    # 1) If user supplied a txt file, load that list
    if video_list_txt and os.path.isfile(video_list_txt):
        print(f"Reading video list from: {video_list_txt}")
        with open(video_list_txt, 'r') as f:
            for line in f:
                path = line.strip()
                if path:
                    videos_to_process.append(path)

    else:
        print("No video list provided or file not found. Auto-discovering videos...")
        # 2) Otherwise fallback to the old recursive walk
        for dirpath, _, filenames in os.walk(root_path):
            if view == "exo":
                if 'chest_compressions' not in dirpath:
                    continue
                exts = ('.mkv',)
            else:
                exts = ('.mp4',)

            for filename in filenames:
                if filename.endswith(exts):
                    full = os.path.join(dirpath, filename)
                    videos_to_process.append(full)

    # shuffle the list of videos to process for randomness
    np.random.shuffle(videos_to_process)
    # 3) Process each video exactly once
    for video_path in videos_to_process:
        dirpath, filename = os.path.split(video_path)
        video_id, ext = os.path.splitext(filename)

        if view == "exo":
            output_json = os.path.join(dirpath, f"{video_id}_resized_640x480_keypoints.json")
        else:
            output_json = os.path.join(dirpath, f"{video_id}_resized_640x480_keypoints.json")

        print(f"Video ID: {video_id}"
              f"\nOutput JSON: {output_json}")
        # skip if already done
        if os.path.exists(output_json):
            print(f"SKIP (exists): {output_json}")
            continue

        print("=" * 60)
        print(f"Processing video: {video_path}")
        process_video(video_path, output_json, wrst, base_model, video_id)
        print("=" * 60)




if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='path to root directory of videos')
    parser.add_argument('--view', type=str, required=True, help='ego or exo')
    parser.add_argument('--video_list_txt', type=str, required=False, help='path to video list text file', default=None)

    args = parser.parse_args()

    # Prepare GroundingDINO model
    base_model = GroundingDINO(ontology=CaptionOntology({"hand": "hand"}))

    wrst = WristDet_mediapipe()


    # Process all videos inside GoPro subdirectories
    # process_videos_in_directory(args.root_dir, wrst, base_model, args.view)


    process_videos_in_list(args.root_dir, wrst, base_model, args.view, args.video_list_txt)
