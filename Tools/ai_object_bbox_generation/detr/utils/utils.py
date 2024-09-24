import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
import torchvision.transforms as T
torch.set_grad_enabled(False);
import pycocotools.coco as coco
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import time
import os
import cv2
from PIL import Image
import os
import time
import json

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# assert(first_class_index in [0, 1])
first_class_index = 0

# There is one class, balloon, with ID n°0.
num_classes = 7

finetuned_classes = [
      "nothing", "bvm", "defib pads", "hands", "kelly", "syringe", "tourniquet"
]

# # There is one class, balloon, with ID n°0.
# num_classes = 4

# finetuned_classes = [
#       "nothing", "bvm", "defib pads", "hands"
# ]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    return b

def plot_finetuned_results(pil_img, prob=None, boxes=None):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if prob is not None and boxes is not None:
      for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
          ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
          cl = p.argmax()
          text = f'{finetuned_classes[cl]}: {p[cl]:0.2f}'
          ax.text(xmin, ymin, text, fontsize=15,
                  bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


def filter_bboxes_from_outputs(outputs,
                               threshold=0.7, image=None):
  
  # keep only predictions with confidence above threshold
  probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > threshold

  probas_to_keep = probas[keep]

  print(probas_to_keep)
  # convert boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], image.size)
  
  return probas_to_keep, bboxes_scaled

def run_worflow(my_image, my_model):
  # mean-std normalize the input image (batch-size: 1)
  img = transform(my_image).unsqueeze(0)

  # propagate through the model
  start_t = time.time()
  outputs = my_model(img)
  end_t = time.time()

  print("Inference time: ",end_t-start_t)
  for threshold in [0.7]:
    
    probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs,
                                                              threshold=threshold, image=my_image)

    plot_finetuned_results(my_image,
                           probas_to_keep, 
                           bboxes_scaled)


def generate_bboxes(video_files, model, json_output_path, target_fps, save_images=False):

    bbox_all = {}
    for video_path in video_files:
        video_id = video_path.split('/')[-1].split('.')[0]
        img_output_path = f'./outputs/bboxes/{video_id}/bbox_annotated/'
        original_img_output_path = f'./outputs/bboxes/{video_id}/original/'

        print("Processing video file:", video_path)
        
        if not os.path.exists(json_output_path):
            os.makedirs(json_output_path)

        if save_images and not os.path.exists(img_output_path):
            os.makedirs(img_output_path)
            os.makedirs(original_img_output_path)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


        # Calculate the frame skipping rate
        skip_rate = original_fps // target_fps

        # Dictionary to store bounding boxes for each processed frame
        bbox_data = []

        # Convert RGB colors to BGR for OpenCV
        COLORS_BGR = [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in COLORS]
        frame_counter = 0

        # Process the video with frame skipping
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Only process frames at the specified interval (skip frames)
            if i % skip_rate != 0:
                continue
            
            backup_frame = frame.copy()

            # Convert the frame to a PIL image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            start_t = time.time()

            # Run inference on the current frame
            img = transform(pil_image).unsqueeze(0)
            img = img.to(device)
            outputs = model(img)
            end_t = time.time()

            probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs, threshold=0.7, image=pil_image)

            # Save bounding boxes and class labels to the JSON data structure
            frame_bboxes = []
            for p, (xmin, ymin, xmax, ymax), c in zip(probas_to_keep, bboxes_scaled.tolist(), COLORS_BGR):
                cl = p.argmax()  # Class index
                class_label = finetuned_classes[cl]  # Get the corresponding class label
                confidence = p[cl].item()

                # Add bounding box info to the JSON dictionary
                frame_bboxes.append({
                    'class': class_label,
                    'confidence': confidence,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax
                })

                # Annotate the frame
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=c, thickness=2)
                text = f'{class_label}: {confidence:.2f}'
                cv2.putText(frame, text, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)

            # Save the bounding box data for the current frame
            bbox_data.append({
                'frame': i,
                'frame_counter': frame_counter,
                'bboxes': frame_bboxes
            })


            # Write the annotated frame to the output directory
            if save_images:
                cv2.imwrite(f'{img_output_path}/frame_{i}.jpg', frame)
                # Save the original frame to the original directory
                cv2.imwrite(f'{original_img_output_path}/{frame_counter:04d}.jpg', backup_frame)

            frame_counter += 1

            # Print progress for every processed frame
            print(f'Processed frame {i}/{total_frames} at downsampled rate (every {skip_rate}th frame)')
            print(f'Inference time for frame {i}: {end_t - start_t} seconds')

            

 
        # Release resources when done
        cap.release()
        cv2.destroyAllWindows()

        print(f'Processed frames and bounding box data.')
        
        # Update the main dictionary with the data for the current video
        bbox_all[video_id] = bbox_data
        
        # Save bounding box data to a JSON file
    with open(f'{json_output_path}/bboxes.json', 'w') as f:
        json.dump(bbox_all, f, indent=4)
