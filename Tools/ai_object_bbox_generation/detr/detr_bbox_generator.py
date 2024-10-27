import torch, torchvision
torch.set_grad_enabled(False);
from utils.utils import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Enable or disable saving annotated images
save_images = True
# Set the desired target FPS for downsampling
target_fps = 1  # You can adjust this value as needed
# Define the video file path and output path
# video_root_path = '/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/Sep_2024/24-09-2024'
video_root_path = '/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final'

# load model
model = torch.hub.load('facebookresearch/detr',
                       'detr_resnet50',
                       pretrained=False,
                       num_classes=num_classes)

# Load the saved weights into the model
checkpoint = torch.load('./weights/24_aug_6class.pth', map_location='cpu')

model.load_state_dict(checkpoint['model'], strict=True)
# move model to the device
model = model.to(device)

model.eval()


# iterate recursively over the video_root_path to get all video files
video_files = []
for root, dirs, files in os.walk(video_root_path):
    for file in files:
        if file.endswith('encoded_trimmed.mp4'):
            video_files.append(os.path.join(root, file))

print(f'Found {len(video_files)} video files in the directory', video_files)

# generate boxes for each video file

all_bbox_data = generate_bboxes(video_files, model, target_fps, save_images=save_images)

# Save the bounding box data to a JSON file
json_output_path = f'{video_root_path}/bbox_data.json'
with open(json_output_path, 'w') as f:
    json.dump(all_bbox_data, f)