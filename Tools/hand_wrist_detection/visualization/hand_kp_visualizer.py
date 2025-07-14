import os
import json
import cv2

# ——— EDIT THESE PATHS AS NEEDED ———

video_root = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/GoPro_CPR_Clips/ego_gopro_cpr_clips/train_root/chest_compressions"


json_path  = f"{video_root}/P0_ts4_ks4_0.000_64.367_ego_resized_640x480_keypoints.json"
video_path = os.path.splitext(json_path)[0].replace("_resized_640x480_keypoints", "") + ".mp4"


# constants
JSON_W, JSON_H   = 640, 480
MODEL_W, MODEL_H = 224, 224

# load keypoints
with open(json_path, "r") as f:
    kp_data = json.load(f)

# prepare output folder
base = os.path.splitext(os.path.basename(json_path))[0]
out_dir = f"./images/{base}"
os.makedirs(out_dir, exist_ok=True)

# open the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Couldn't open video: {video_path}")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) shrink to 640×480 so it matches your JSON coordinate system
    frame_640 = cv2.resize(frame, (JSON_W, JSON_H))

    # 2) draw keypoints (if any) at their raw (x,y)
    key = str(frame_idx)
    if key in kp_data:
        for hand in kp_data[key].get("hands", []):
            xs, ys = hand.get("x", []), hand.get("y", [])
            for x, y in zip(xs, ys):
                xi, yi = int(round(x)), int(round(y))
                # skip anything out of bounds
                if not (0 <= xi < JSON_W and 0 <= yi < JSON_H):
                    continue

                xi = max(0, min(int(round(x)), JSON_W - 1))
                yi = max(0, min(int(round(y)), JSON_H - 1))
                cv2.circle(frame_640, (xi, yi), radius=3, color=(0,255,0), thickness=-1)

    # 3) now down-sample the *annotated* 640×480 to 224×224
    frame_224 = cv2.resize(frame_640, (MODEL_W, MODEL_H))

    # 4) save
    out_path = os.path.join(out_dir, f"frame_{frame_idx:04d}.png")
    cv2.imwrite(out_path, frame_224)

    frame_idx += 1

cap.release()
print(f"Done! Saved {frame_idx} frames to {out_dir}/")
