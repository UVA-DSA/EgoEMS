import glob
from video2numpy import video2numpy
import numpy as np

VIDS = glob.glob("/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/ms1/cardiac_arrest/0/GoPro/GX010391_encoded_trimmed.mp4")
FRAME_DIR = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/ms1/cardiac_arrest/0/GoPro/"

video2numpy(VIDS, FRAME_DIR)

# # load the numpy file
# frames = np.load(f"{FRAME_DIR}/GX010391_encoded_trimmed.npy")
# print(frames.shape)