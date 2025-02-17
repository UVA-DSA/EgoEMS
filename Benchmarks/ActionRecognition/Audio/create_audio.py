from moviepy.editor import VideoFileClip
import os
import json
from tqdm import tqdm


if __name__ == "__main__":
    # Load the video file

    # path = '/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/Sep_2024/Raw/05-09-2024/'


    # path = '/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/Sep_2024/Raw/20-09-2024/chas/cardiac_scenario'
    # path = '/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/Final'
    path = '/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final'
    for root, dirs, files in tqdm(os.walk(path)):
        if 'GoPro' in dirs:
            for file in os.listdir(os.path.join(root, 'GoPro')):

                if "_encoded_trimmed.mp4" in file.lower():
                    video = VideoFileClip(os.path.join(root, 'GoPro', file))

                    audio = video.audio

                    if not os.path.exists(os.path.join(root, 'audio')):
                        os.makedirs(os.path.join(root, 'audio'))

                    if f"{file.split('.mp4')[0]}.mp3" in os.listdir(os.path.join(root, 'audio')):
                        print(f"skip {file.split('.mp4')[0]}.mp3")
                        continue

                    # print(os.path.join(root, 'audio', f"{file.split('.mp4')[0]}.mp3"))
                    audio.write_audiofile(os.path.join(root, 'audio', f"{file.split('.mp4')[0]}.mp3"))