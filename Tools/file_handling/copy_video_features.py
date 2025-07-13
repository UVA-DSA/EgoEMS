import os
import glob

# ─── CONFIG: set to None to disable that branch ──────────────────────────────
ego_file_list_path = "/home/cjh9fw/Desktop/2024/repos/video_features/videos_to_extract/cvpr_ego_file_paths_new.txt"   # ← set to None if you don’t want ego
# ego_file_list_path = None   # ← set to None if you don’t want ego

# exo_file_list_path = "/home/cjh9fw/Desktop/2024/repos/video_features/videos_to_extract/cvpr_exo_file_paths.txt"   # ← set to None if you don’t want exo
exo_file_list_path = None   # ← set to None if you don’t want exo


# video_feature_directory = "/home/cjh9fw/Desktop/2024/repos/video_features/extracted_features/cars/cars_ego_file_paths_special/i3d"
# video_feature_directory = "/home/cjh9fw/Desktop/2024/repos/video_features/extracted_features/cars/cars_ego_file_paths_special/resnet/resnet50"
# video_feature_directory = "/home/cjh9fw/Desktop/2024/repos/video_features/extracted_features/cars/cars_ego_file_paths_special/clip/ViT-B_32"


# video_feature_directory = "/home/cjh9fw/Desktop/2024/repos/video_features/extracted_features/opvrs/opvrs_ego_file_paths_p3/i3d" # todo yet

# video_feature_directory = "/scratch/cjh9fw/Rivanna/2025/egoems_video_features_backup/cars_backup/cars_ego_file_paths/clip/ViT-B_32/" # todo yet


# video_feature_directory = "/standard/UVA-DSA/NIST EMS Project Data/video_features/cvpr/cvpr_ego_file_paths/resnet/resnet50" # resnet cvpr old
# video_feature_directory = "/standard/UVA-DSA/NIST EMS Project Data/video_features/cvpr/cvpr_ego_file_paths/i3d" # i3d cvpr old

video_feature_directory = "/standard/UVA-DSA/NIST EMS Project Data/video_features/cvpr/missing_cvpr_ego_file_paths/i3d" # resnet cvpr exo


success_count = 0
failure_count = 0

# ─── LOAD FILE‑PATH MAPS ─────────────────────────────────────────────────────
def load_mapping(list_path):
    """
    Reads a file of mp4 paths (one per line) and returns
    a dict mapping basename (sans “.mp4”) → full path.
    """
    mapping = {}
    if list_path is None:
        return mapping
    with open(list_path, "r") as f:
        for line in f:
            path = line.strip()
            key = os.path.basename(path).rsplit(".mp4", 1)[0]
            mapping[key] = path
    return mapping


# helper to do the actual copying
def handle_branch(mapping, tag):

    if video_name not in mapping:
        return False
    vid_path = mapping[video_name]
    # go up two levels from the GoPro file to get subject/trial folder
    base_dir = os.path.dirname(os.path.dirname(vid_path))

    # extract subject & trial for naming
    parts  = base_dir.split(os.sep)
    subject, scenario, trial = parts[-3], parts[-2], parts[-1]
    
    # # special case 
    # base_dir = f"/standard/UVA-DSA/NIST EMS Project Data/EgoEMS_AAAI2026/{subject}/{scenario}/{trial}" # todo yet

    if "resnet" in video_feature_directory:
        target_dir = os.path.join(base_dir, f"resnet_{tag}")
    elif "clip" in video_feature_directory:
        target_dir = os.path.join(base_dir, f"clip_{tag}")
    elif "i3d" in video_feature_directory:
        target_dir = os.path.join(base_dir, f"i3d_{tag}")
    else: 
        raise ValueError("Unknown video feature directory structure.")
    
    print(f"Creating target directory: {target_dir}")
    os.makedirs(target_dir, exist_ok=True)

    # extract subject & trial for naming
    subject = subject.replace("_","")
    scenario = scenario.replace("_","")


    if "resnet" in video_feature_directory:
        dst_name = f"{subject}_{scenario}_{trial}_{tag}_resnet.npy"
    elif "clip" in video_feature_directory:
        dst_name = f"{subject}_{scenario}_{trial}_{tag}_clip.npy"
    elif "i3d" in video_feature_directory:
        # i3d files are named differently
        dst_name = f"{subject}_{scenario}_{trial}_{tag}_i3d.npy"

    dst_path = os.path.join(target_dir, dst_name)


    print(f"[{tag.upper()}] copying {feat_path} → {dst_path}")

    # check if the file already exists
    if os.path.exists(dst_path):
        print(f"File {dst_path} already exists, skipping copy.")
        return False
    
    os.system(f'cp "{feat_path}" "{dst_path}"')

    print(f"Copied {feat_path} to {dst_path}")

    return True

    


if __name__ == "__main__":

    print(f"Copying video features from {video_feature_directory}...")

    # load file-path maps
    ego_map = load_mapping(ego_file_list_path)
    exo_map = load_mapping(exo_file_list_path)

    print(f"Loaded {len(ego_map)} ego file paths.")
    print(f"Loaded {len(exo_map)} exo file paths.")

    # process each feature file
    # glob for all resnet feature files
    print("\n\n")
    print("*-" * 20)
    print("Processing resnet features...")
    success_count = 0
    failure_count = 0
    for feat_path in glob.glob(f"{video_feature_directory}/*_resnet.npy"):
        basename    = os.path.basename(feat_path)
        video_name  = basename.split("_resnet.npy")[0]

        print("-" * 20)
        print(f"Processing {basename}...")
        # run ego branch only if ego_map was loaded
        if len(ego_map) > 0:
            if handle_branch(ego_map, "ego"):
                success_count += 1
            else:
                failure_count += 1

        # run exo branch only if exo_map was loaded
        if len(exo_map) > 0:
            if handle_branch(exo_map, "exo"):
                success_count += 1
            else:
                failure_count += 1

        print("-" * 20)

    print(f"Successfully copied {success_count} resnet features, failed to copy {failure_count}.")

    # glob for all clip feature files
    print("\n\n")
    print("*-" * 20)
    print("Processing clip features...")
    for feat_path in glob.glob(f"{video_feature_directory}/*_clip.npy"):
        basename    = os.path.basename(feat_path)
        video_name  = basename.split("_clip.npy")[0]

        print("-" * 20)
        print(f"Processing {basename}...")
        # run ego branch only if ego_map was loaded
        if len(ego_map) > 0:
            if handle_branch(ego_map, "ego"):
                success_count += 1
            else:
                failure_count += 1

        # run exo branch only if exo_map was loaded
        if len(exo_map) > 0:
            if handle_branch(exo_map, "exo"):
                success_count += 1
            else:
                failure_count += 1

        print("-" * 20)


    # glob for all i3d flow feature files
    print("\n\n")
    print("*-" * 20)
    print("Processing i3d flow features...")
    for feat_path in glob.glob(f"{video_feature_directory}/*_flow.npy"):
        basename    = os.path.basename(feat_path)
        video_name  = basename.split("_flow.npy")[0]

        print("-" * 20)
        print(f"Processing {basename}...")
        # run ego branch only if ego_map was loaded
        if len(ego_map) > 0:
            if handle_branch(ego_map, "flow") : 
                success_count += 1
            else:
                failure_count += 1

        # run exo branch only if exo_map was loaded
        if len(exo_map) > 0:
            if handle_branch(exo_map, "exo"):
                success_count += 1
            else:
                failure_count += 1

        print("-" * 20)
    print(f"Successfully copied {success_count} flow features, failed to copy {failure_count}.")

    # glob for all i3d rgb feature files
    print("\n\n")
    print("*-" * 20)
    print("Processing i3d rgb features...")
    success_count = 0
    failure_count = 0

    for feat_path in glob.glob(f"{video_feature_directory}/*_rgb.npy"):
        basename    = os.path.basename(feat_path)
        video_name  = basename.split("_rgb.npy")[0]

        print("-" * 20)
        print(f"Processing {basename}...")
        # run ego branch only if ego_map was loaded
        if len(ego_map) > 0:
            if handle_branch(ego_map, "rgb"):
                success_count += 1
            else:
                failure_count += 1

        # run exo branch only if exo_map was loaded
        if len(exo_map) > 0:
            handle_branch(exo_map, "exo")


        print("-" * 20)
    print(f"Successfully copied {success_count} rgb features, failed to copy {failure_count}.")