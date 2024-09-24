import os
import json
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor

# Set environment variable for Apple MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Select device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print("\nSupport for MPS devices is preliminary. Performance may be degraded.\n")

# Helper functions for visualization
def show_mask(mask, ax, obj_id=None, random_color=False):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else np.array([*plt.get_cmap("tab10")(0 if obj_id is None else obj_id)[:3], 0.6])
    mask_image = mask.reshape(*mask.shape[-2:], 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# Load model predictor
sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# Function to create array of frame names
def frame_array(video_dir):
    video_dir = os.path.join(video_dir, 'original')
    frame_names = sorted([p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]], key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names


# Helper function to calculate the IoU between two bounding boxes
def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
    - box1, box2: Bounding boxes in the format [xmin, ymin, xmax, ymax]
    
    Returns:
    IoU value (float)
    """
    x1, y1, x2, y2 = box1
    x1_b, y1_b, x2_b, y2_b = box2

    # Determine the coordinates of the intersection rectangle
    inter_x1 = max(x1, x1_b)
    inter_y1 = max(y1, y1_b)
    inter_x2 = min(x2, x2_b)
    inter_y2 = min(y2, y2_b)

    # Compute the area of the intersection rectangle
    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)

    # Compute the area of both bounding boxes
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_b - x1_b + 1) * (y2_b - y1_b + 1)

    # Compute the IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# Function to get bounding boxes with highest confidence for each class, ensuring non-overlapping boxes
def get_bbox_for_video(bbox_json_path):
    """
    Get the highest confidence bounding boxes for each object class within each frame in the video, 
    ensuring that multiple boxes for the same class do not overlap by more than 50%.

    Parameters:
    - bbox_json_path: Path to the JSON file containing bounding box data.

    Returns:
    A dictionary containing the highest confidence bounding boxes per frame for each object class.
    """
    with open(bbox_json_path, 'r') as f:
        data = json.load(f)

    class_to_obj_id = {"bvm": 1, "hands": 2, "defib pads": 3}
    video_data = {}

    for video, frames in data.items():
        video_dir = f'/scratch/cjh9fw/Rivanna/2024/repos/EgoExoEMS/Tools/ai_object_bbox_generation/detr/outputs/bboxes/{video}/'
        frame_names = frame_array(video_dir)

        # Store highest confidence boxes per class per frame
        video_frames_bboxes = []

        # Iterate over each frame in the video
        for frame_info in frames:
            highest_conf_bboxes = {}

            # Iterate over each bounding box in the frame
            for bbox in frame_info['bboxes']:
                class_name = bbox['class']
                confidence = bbox['confidence']

                # Ignore low confidence boxes
                if confidence < 0.9:
                    continue

                current_bbox = [int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])]

                # Check if there is an existing bbox for the class
                if class_name in highest_conf_bboxes:
                    # Calculate IoU with all existing bboxes for this class
                    overlaps = [calculate_iou(current_bbox, existing['bbox']) for existing in highest_conf_bboxes[class_name]]
                    
                    # Add new bbox if no overlaps are greater than 50%
                    if all(iou < 0.5 for iou in overlaps):
                        highest_conf_bboxes[class_name].append({
                            'bbox': current_bbox,
                            'confidence': confidence,
                            'ann_obj_id': class_to_obj_id.get(class_name, 4)
                        })
                else:
                    # Initialize a new list for this class and add the bbox
                    highest_conf_bboxes[class_name] = [{
                        'bbox': current_bbox,
                        'confidence': confidence,
                        'ann_obj_id': class_to_obj_id.get(class_name, 4)
                    }]

            # Append the highest confidence bboxes for this frame
            if highest_conf_bboxes:
                video_frames_bboxes.append({
                    'frame_counter': frame_info['frame_counter'],
                    'bboxes': highest_conf_bboxes,
                    'ann_frame_idx': frame_info['frame_counter'],
                    'actual_frame': frame_info['frame']
                })

        if video_frames_bboxes:
            video_data[video] = {
                'video_dir': video_dir,
                'frame_names': frame_names,
                'class_conf_bboxes': video_frames_bboxes
            }

    return video_data if video_data else None

# Function to get bounding boxes with highest confidence for each class for multiple frames
def get_bboxes_for_video(bbox_json_path):
    """
    Get the highest confidence bounding boxes for each object class across all frames in the video.
    
    Parameters:
    - bbox_json_path: Path to the JSON file containing bounding box data.
    
    Returns:
    A dictionary containing the video directory, a list of bounding boxes, confidence, 
    annotation object ID, annotation frame index, object class, and frame names for each class.
    """
    # Load JSON data
    with open(bbox_json_path, 'r') as f:
        data = json.load(f)

    # Mapping object classes to unique IDs
    class_to_obj_id = {
        "bvm": 1,
        "hands": 2,
        "defib pads": 3
    }

    video_data = {}

    # Iterate over each video and its frames
    for video, frames in data.items():
        video_dir = f'/scratch/cjh9fw/Rivanna/2024/repos/EgoExoEMS/Tools/ai_object_bbox_generation/detr/outputs/bboxes/{video}/'
        print(f'Processing video: {video}')

        # Generate original frame array
        frame_names = frame_array(video_dir)

        # Dictionary to store a list of highest confidence bboxes for each class
        class_conf_bboxes = {}

        # Iterate over frames to find the highest confidence bbox for each object class
        for frame_info in frames:
            frame_number = frame_info['frame_counter']
            bboxes = frame_info['bboxes']

            # if frame_number % 10 != 0:
            #     continue

            # Check each bounding box for its class and confidence
            for bbox in bboxes:
                class_name = bbox['class']
                confidence = bbox['confidence']

                if(confidence < 0.9):
                    continue

                # # Initialize the list for the class if not already present
                # if class_name not in class_conf_bboxes:
                #     class_conf_bboxes[class_name] = []

                # Append the bbox if it's the highest for this class at this frame
                if class_name not in class_conf_bboxes or confidence > class_conf_bboxes[class_name]['confidence']:

                    class_conf_bboxes[class_name].append({
                        'bbox': [int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])],
                        'confidence': confidence,
                        'ann_obj_id': class_to_obj_id.get(class_name, 4),
                        'ann_frame_idx': frame_number,
                        'actual_frame': frame_info['frame']
                    })

        # If we found bounding boxes for any classes, store the video data
        if class_conf_bboxes:
            video_data[video] = {
                'video_dir': video_dir,
                'class_conf_bboxes': class_conf_bboxes,
                'frame_names': frame_names
            }

    return video_data if video_data else None





# Processing the frames for segmentation and tracking
def add_objects_to_predictor(predictor, inference_state, bbox, ann_obj_id, ann_frame_idx):
    return predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, box=bbox)

def track_and_generate_mask(predictor, inference_state):
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}
    return video_segments

def render_video_segments(ann_frame_idx, frame_names, video_dir, video_segments, vis_frame_stride=10):
    for out_frame_idx in range(ann_frame_idx, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.show()

# Save rendered frames as segmented images
def render_video_segments_to_video(ann_frame_idx, frame_names, video_dir, video_segments, output_video_path):
    os.makedirs(output_video_path, exist_ok=True)
    for out_frame_idx in range(ann_frame_idx, len(frame_names)):
        plt.figure(figsize=(6, 4))
        plt.title(f"Frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))

        # check if out_frame_idx exists in video_segments first
        if out_frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_video_path}/segmented_{out_frame_idx}.png", bbox_inches='tight', pad_inches=0)
        plt.close()


# # exit the program
# exit()

# Iterate over video data and process
def generate_segmentations(result):
    for video, data in result.items():
        video_dir, frame_names = data['video_dir'], data['frame_names']

        # if(video != 'GX010309_clipped_with_audio'):
        #     continue

        print(f"Processing video: {video}")

        original_video_dir = os.path.join(video_dir, 'original')
        inference_state = predictor.init_state(video_path=original_video_dir)
        predictor.reset_state(inference_state)

       # Iterate over each frame's bounding boxes
        for frame_data in data['class_conf_bboxes']:
            frame_counter = frame_data['frame_counter']
            highest_conf_bboxes = frame_data['bboxes']  # Bounding boxes for this frame
            ann_frame_idx = frame_data['ann_frame_idx']

            for class_name, class_data in highest_conf_bboxes.items():
                if class_name == 'hands':
                    # print('Skipping hands')
                    continue

                # print(f"Processing class: {class_name}, frame: {frame_counter}, class_data: {class_data}")
                
                for box_data in class_data:
                    bbox, ann_obj_id = box_data['bbox'], box_data['ann_obj_id']

                    # Add objects to the predictor
                    add_objects_to_predictor(predictor, inference_state, bbox, ann_obj_id, ann_frame_idx)

        video_segments = track_and_generate_mask(predictor, inference_state)

        print(f"Video segments: {len(video_segments)}")
        # print(f"Video segment keys: {(video_segments.keys())}")
        output_video_path = f"{video_dir}/segmentations/"
        render_video_segments_to_video(0, frame_names, original_video_dir, video_segments, output_video_path)

        # break

def generate_segments_multiple_frames(result):
    if result:
        for video, data in result.items():
            video_dir, frame_names = data['video_dir'], data['frame_names']

            original_video_dir = os.path.join(video_dir, 'original')
            inference_state = predictor.init_state(video_path=original_video_dir)
            predictor.reset_state(inference_state)

            print(f"Processing video: {video}")


            for class_name, class_data in data['class_conf_bboxes'].items():
                if class_name == 'hands':
                    print('Skipping hands')
                    continue
                # Iterate over bounding boxes for this class
                print(f"Processing class: {class_name}, class_data: {class_data}")
                for bbox_data in class_data:
                    bbox, ann_obj_id, ann_frame_idx = bbox_data['bbox'], bbox_data['ann_obj_id'], bbox_data['ann_frame_idx']
                    add_objects_to_predictor(predictor, inference_state, bbox, ann_obj_id, ann_frame_idx)
                
            video_segments = track_and_generate_mask(predictor, inference_state)
            output_video_path = f"{video_dir}/segmentations/"
            render_video_segments_to_video(0, frame_names, video_dir, video_segments, output_video_path)

            # break


# Bounding box extraction from JSON
bbox_json_path = '../../ai_object_bbox_generation/detr/outputs/bboxes/bboxes.json'


result = get_bbox_for_video(bbox_json_path)
# Print result in pretty format
# if result:
#     print(json.dumps(result["GX010309_clipped_with_audio"], indent=4))
generate_segmentations(result)

# does not work rn
# result = get_bboxes_for_video(bbox_json_path)
# generate_segments_multiple_frames(result)
