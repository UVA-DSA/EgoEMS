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
            max_num_hands=2,
            min_detection_confidence=0.5
        )

    def get_kypts(self, image):
        height, width, _ = image.shape
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        xy_vals = []
        z_vals = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(21):
                    x = int(hand_landmarks.landmark[i].x * width)
                    y = int(hand_landmarks.landmark[i].y * height)
                    z = hand_landmarks.landmark[i].z
                    xy_vals.append((x, y))
                    z_vals.append(z)
            if len(xy_vals) == 42:
                closest_hand = np.argmin([z_vals[0], z_vals[21]])
                start_coord = 0 if closest_hand == 0 else 21
                xy_vals = xy_vals[start_coord:start_coord + 21]
        return image, xy_vals


def get_kpts(img, wrst, base_model):
    results = base_model.predict(img)
    bb = get_bb(results)

    if not bb:  # Check if bounding box is empty
        print("No bounding box detected.")
        return {"x": [], "y": []}  # Return empty keypoints

    pad = 80
    img_crop = crop_img_bb(img, bb, pad, show=False)
    image, xy_vals = wrst.get_kypts(img_crop)

    if not xy_vals:  # Check if keypoints are detected
        print("No keypoints detected.")
        return {"x": [], "y": []}  # Return empty keypoints

    x_vals = [int(val[0] + bb[0] - pad) for val in xy_vals]
    y_vals = [int(val[1] + bb[1] - pad) for val in xy_vals]
    kpt_dict = {"x": x_vals, "y": y_vals}
    return kpt_dict


def process_video(video_path, output_json_path, wrst, base_model):
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

        # Get keypoints for the current frame
        kpt_dict = get_kpts(frame, wrst, base_model)

        # Store keypoints with the frame number
        keypoints_data[frame_num] = kpt_dict

        # Print or process keypoints (optional)
        print(f"Frame {frame_num}: {kpt_dict}")
        frame_num += 1

    # Save keypoints to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(keypoints_data, json_file, indent=4)

    cap.release()


def process_videos_in_directory(root_path, wrst, base_model):
    # Recursively search for videos inside 'GoPro' subdirectories
    flag = False
    for dirpath, _, filenames in os.walk(root_path):
        if 'Kinect' in dirpath:
            for filename in filenames:
                if filename.endswith(('final.mkv')):  # You can add more video formats if needed
                    print("=" * 60)
                    print("*" * 60)
                    print(f"Processing directory: {dirpath}")

                    print(f"Processing video: {filename}")
                    video_path = os.path.join(dirpath, filename)
                    video_id = filename.split('.')[0]
                    output_json_path = os.path.join(dirpath, f'{video_id}_keypoints.json')

                    print(f"Processing video: {video_path}")
                    # Process the video and save keypoints to JSON
                    process_video(video_path, output_json_path, wrst, base_model)
                    print("*" * 60)
                    print("=" * 60)
                    flag = True
                    break
        if flag:
            break


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='path to root directory of videos')

    args = parser.parse_args()

    # Prepare GroundingDINO model
    base_model = GroundingDINO(ontology=CaptionOntology({"hand": "hand"}))

    wrst = WristDet_mediapipe()

    # Process all videos inside GoPro subdirectories
    process_videos_in_directory(args.root_dir, wrst, base_model)
