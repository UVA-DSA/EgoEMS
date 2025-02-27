import os
import glob

# Base directory where files are located
base_directory = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final"
video_feature_directory = "/home/cjh9fw/Desktop/2024/repos/video_features/extracted_features/clip/ViT-B_32"
ego_files_dirs = "./ego_file_paths.txt"
exo_files_dirs = "./exo_file_paths.txt"

# Read the ego_files_dirs and exo_files_dirs files
with open(ego_files_dirs, "r") as f:
    ego_files_dirs = f.read().splitlines()
with open(exo_files_dirs, "r") as f:
    exo_files_dirs = f.read().splitlines()

# Create a dictionary to map video names to their file paths
ego_files_dirs = {os.path.basename(path).split(".mp4")[0]: path for path in ego_files_dirs}
exo_files_dirs = {os.path.basename(path).split(".mp4")[0]: path for path in exo_files_dirs}

# print("ego_files_dirs", ego_files_dirs)

extracted_video_features = glob.glob(f"{video_feature_directory}/*_clip.npy", recursive=True)

for file_path in extracted_video_features:

    # get file name
    file_name = os.path.basename(file_path)
    # get video name
    video_name = file_name.split("_clip.npy")[0]

    # find the corresponding video file path from ego_files_dirs and exo_files_dirs
    if video_name in ego_files_dirs:

        # get the corresponding video file path from ego_files_dirs
        video_file_path = ego_files_dirs[video_name]

        # get the directory of the video file path
        video_file_directory = os.path.dirname(video_file_path)
        # go back one directory from the video file directory
        video_file_directory = os.path.dirname(video_file_directory)

        # create a directory for clip_ego if it doesn't exist
        video_file_directory = os.path.join(video_file_directory, "clip_ego")

        if not os.path.exists(video_file_directory):
            os.makedirs(video_file_directory)

        # get subject and trial from video file directory
        subject = video_file_directory.split("/")[-4]
        trial = video_file_directory.split("/")[-2]
        # print("--" * 20)
        # print("subject", subject)
        # print("trial", trial)
        # print(f"Copying {file_name} to {video_file_directory}")
        # print("--" * 20)

        # copy the file to the video file directory
        clip_feature_name = f"{subject}_{trial}_ego_clip.npy"
        destination_file_path = os.path.join(video_file_directory, clip_feature_name)
        copy_command = f'cp {file_path} "{destination_file_path}"'

        print("--" * 20, "EGO", "--" * 20)
        print(f"EGO Copying {file_name} to ")
        print("destination_file_path", destination_file_path)
        print("copy_command", copy_command)

        # copy the file to the video file directory
        os.system(copy_command)
        print("--" * 20)



    # if video_name in exo_files_dirs:

    #     # get the corresponding video file path from exo_files_dirs
    #     video_file_path = exo_files_dirs[video_name]

    #     # get the directory of the video file path
    #     video_file_directory = os.path.dirname(video_file_path)
    #     # go back one directory from the video file directory
    #     video_file_directory = os.path.dirname(video_file_directory)

    #     # create a directory for clip_ego if it doesn't exist
    #     video_file_directory = os.path.join(video_file_directory, "clip_exo")
    #     print("exo video_file_directory", video_file_directory)

    #     if not os.path.exists(video_file_directory):
    #         os.makedirs(video_file_directory)   

    #     # get subject and trial from video file directory
    #     subject = video_file_directory.split("/")[-4]
    #     trial = video_file_directory.split("/")[-2]
    #     # print("--" * 20)
    #     # print("subject", subject)
    #     # print("trial", trial)
    #     # print(f"Copying {file_name} to {video_file_directory}")
    #     # print("--" * 20)

    #     # copy the file to the video file directory
    #     clip_feature_name = f"{subject}_{trial}_exo_clip.npy"
    #     destination_file_path = os.path.join(video_file_directory, clip_feature_name)
    #     copy_command = f'cp {file_path} "{destination_file_path}"'

    #     print("--" * 20, "EXO", "--" * 20)
    #     print(f"EXO Copying {file_name} to ")
    #     print("destination_file_path", destination_file_path)
    #     print("copy_command", copy_command)

    #     # copy the file to the video file directory
    #     os.system(copy_command)
    #     print("--" * 20)
